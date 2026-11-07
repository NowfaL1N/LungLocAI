"""LungLocAI web server.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import hashlib
import hmac
import io

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from skimage import measure

from .pipeline import classifier, lungrads
from .pipeline.ct_io import load_ct
from .pipeline.features import FEATURE_COLUMNS
from .pipeline.scan_to_table import scan_to_table
from . import report
from . import retrain as retrainer
from .storage import anon_id, get_store

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data" / "jobs"
SAMPLES = ROOT / "data" / "samples"
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LungLocAI")
STATE: dict[str, dict] = {}
CACHE: dict[str, dict] = {}          # in-memory arrays for the viewer (last few jobs)
RUN_LOCK = threading.Semaphore(1)    # one CT at a time (memory)
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "2"))      # uploaded scans + results are deleted after this
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "1500"))
STORE = get_store()                  # private research store (None = storage disabled)
LIVE_MODELS = ROOT / "data" / "models_live"   # retrained models pulled from the store
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
MODELS = classifier.load_models(LIVE_MODELS)
STORE_STATUS = {"configured": STORE is not None, "backend": getattr(STORE, "name", None), "error": None,
                "saved": 0, "failed": 0}
RETRAIN = {"state": "idle", "log": [], "report": None}

STEPS = [("load", "Reading CT"), ("lungs", "3-D volume + lung extraction"), ("candidates", "Finding nodules"),
         ("features", "Building feature table"), ("xgboost", "XGBoost classification"), ("report", "Writing report")]


def _set(job, **kw):
    STATE[job].update(kw)
    (JOBS / job / "status.json").write_text(json.dumps({k: v for k, v in STATE[job].items() if k != "result"}))


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _impression(nods, L, n_cand, has_mal):
    out = []
    if L["full_body"]:
        out.append(f"Extended-coverage CT ({L['scan_length_mm']:.0f} mm). Lungs were automatically located and "
                   f"extracted ({L['lung_length_mm']:.0f} mm cranio-caudal) for analysis.")
    if not nods:
        out.append(f"No lung nodules reported after examining {n_cand} candidate regions.")
        return out
    right = sum(n["side"] == "Right" for n in nods)
    out.append(f"{len(nods)} lung nodule{'s' if len(nods) > 1 else ''} reported "
               f"({right} right, {len(nods) - right} left).")
    top = nods[0]
    if has_mal:
        sus = [n for n in nods if n["ai_label"].startswith("Suspicious")]
        out.append(f"{len(sus)} nodule{'s are' if len(sus) != 1 else ' is'} classified by XGBoost as suspicious "
                   "for lung cancer." if sus else "No nodule exceeds the model's malignancy threshold.")
        out.append(f"Most suspicious: #{top['display_id']}, {top['diameter_mm']:.1f} mm {top['nodule_type']} nodule, "
                   f"{top['side'].lower()} lung {top['zone'].lower()} zone; AI cancer probability "
                   f"{top['p_malignant'] * 100:.0f}% ({top['ai_risk']} risk); Lung-RADS-style category {top['lungrads']}.")
    else:
        out.append(f"Largest: #{top['display_id']}, {top['diameter_mm']:.1f} mm {top['nodule_type']} nodule, "
                   f"{top['side'].lower()} lung {top['zone'].lower()} zone; Lung-RADS-style category {top['lungrads']}.")
    urgent = [n for n in nods if n["priority"] == "Urgent review"]
    if urgent:
        out.append("Recommend prompt radiologist review of nodule(s) " + ", ".join(f"#{n['display_id']}" for n in urgent)
                   + " (" + lungrads.MANAGEMENT[urgent[0]["lungrads"]].split(".")[0].lower() + ").")
    return out


def _mesh(mask, step=2, offset=(0, 0, 0), image=None):
    if mask.sum() < 10:
        return None
    m = np.pad(mask, 1).astype(np.float32)
    try:
        v, f, _, _ = measure.marching_cubes(m, 0.5, step_size=step)
    except Exception:
        return None
    v = v - 1 + np.array(offset)
    # voxel (z,y,x) -> world (x,y,z)
    o = np.array(image.GetOrigin()); sp = np.array(image.GetSpacing())
    D = np.array(image.GetDirection()).reshape(3, 3)
    xyz = (D @ (v[:, ::-1] * sp).T).T + o
    return {"x": np.round(xyz[:, 0], 1).tolist(), "y": np.round(xyz[:, 1], 1).tolist(),
            "z": np.round(xyz[:, 2], 1).tolist(), "i": f[:, 0].tolist(), "j": f[:, 1].tolist(), "k": f[:, 2].tolist()}


def run_job(job: str):
    d = JOBS / job
    t0 = time.time()
    with RUN_LOCK:
        try:
            _set(job, state="running", step="load", message="Reading CT")
            img, info = load_ct(d / "upload")

            def prog(step, msg=""):
                _set(job, step=step, message=msg)

            L, cands, df, bb = scan_to_table(img, prog)
            _set(job, step="xgboost", message=f"Running XGBoost on {len(df)} rows")
            df, notes = classifier.classify(df, MODELS)
            has_mal = "malignancy_model" in MODELS

            _set(job, step="report", message="Writing report")
            if len(df):
                df["lungrads"] = [lungrads.categorize(r.diameter_mm, r.core_solid_fraction, r.nodule_type) for r in df.itertuples()]
                df["priority"] = [lungrads.overall_priority(c, p) for c, p in zip(df.lungrads, df.p_malignant)]
                df["management"] = [lungrads.MANAGEMENT[c] for c in df.lungrads]
            rep = df[df.is_nodule == 1].copy() if len(df) else df
            if len(rep):
                sort_key = "p_malignant" if has_mal else "diameter_mm"
                rep = rep.sort_values(sort_key, ascending=False)
                rep["display_id"] = range(1, len(rep) + 1)
                df = df.merge(rep[["candidate_id", "display_id"]], on="candidate_id", how="left")
            df.insert(0, "seriesuid", info.get("series_uid", ""))
            df.to_csv(d / "table.csv", index=False)

            # label map for the viewer
            labels = np.zeros(L["array"].shape, np.uint8)
            for i, c in enumerate(cands, start=1):
                sub = labels[c["box"]]
                sub[c["mask"]] = i
            np.savez_compressed(d / "volume.npz", arr=L["array"], lung=L["lung"], labels=labels)
            CACHE[job] = {"arr": L["array"], "lung": L["lung"], "labels": labels}
            while len(CACHE) > 3:
                CACHE.pop(next(iter(CACHE)))

            nodules = [{k: _jsonable(v) for k, v in r.items()} for r in rep.to_dict("records")]
            closeups = {}
            for n in nodules:
                png = report.nodule_closeup(L["array"], labels, n)
                (d / f"nodule_{n['candidate_id']}.png").write_bytes(png)
                closeups[n["candidate_id"]] = png
            ov = report.overview_png(L["overview"])
            (d / "overview.png").write_bytes(ov)

            # 3-D scene
            scene ={"lung": _mesh_scaled(L["lung"], L["image"]), "nodules": []}
            for n in nodules:
                c = cands[int(n["candidate_id"]) - 1]
                off = [s.start for s in c["box"]]
                mm = _mesh(c["mask"], step=1, offset=off, image=L["image"])
                if mm:
                    mm.update(id=n["display_id"], risk=n["ai_risk"], label=n["ai_label"], d=n["diameter_mm"],
                              p=n.get("p_malignant"))
                    scene["nodules"].append(mm)
            (d / "scene.json").write_text(json.dumps(scene))

            model_info = {"loaded": list(MODELS), "demo": any(m.meta.get("demo") for m in MODELS.values()),
                          "details": {k: m.info() for k, m in MODELS.items()}}
            result = {
                "job_id": job,
                "scan": info,
                "lungs": {k: L[k] for k in ("full_body", "scan_length_mm", "lung_length_mm", "lung_volume_ml")},
                "shape": list(L["array"].shape),
                "n_candidates": int(len(df)),
                "n_rejected": int((df.is_nodule == 0).sum()) if len(df) else 0,
                "nodules": nodules,
                "impression": _impression(nodules, L, len(df), has_mal),
                "models": model_info,
                "warnings": notes["warnings"],
                "feature_columns": FEATURE_COLUMNS,
                "seconds": round(time.time() - t0, 1),
            }
            (d / "result.json").write_text(json.dumps(result))
            report.build_pdf(d / "report.pdf", result, closeups, ov)
            consent = (d / "consent").exists()
            result["consent"] = consent and STORE is not None
            (d / "result.json").write_text(json.dumps(result))
            _set(job, state="done", step="done", message="Finished", result=result)
            shutil.rmtree(d / "upload", ignore_errors=True)
            if consent and STORE is not None:
                threading.Thread(target=_store_case, args=(job, result, closeups, ov), daemon=True).start()
        except Exception as e:
            traceback.print_exc()
            _set(job, state="error", message=str(e) or e.__class__.__name__)


def _store_case(job, result, closeups, overview):
    """Save an anonymised copy of the findings in the private store (only with the uploader's consent)."""
    d = JOBS / job
    case_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    try:
        if hasattr(STORE, "check") and STORE.check().get("private") is False:
            raise RuntimeError(f"refusing to store: {STORE.repo} is PUBLIC - make it private first")
        r = json.loads(json.dumps(result))
        uid = r["scan"].get("series_uid", "")
        r["scan"] = {k: v for k, v in r["scan"].items() if k not in ("source_file", "series_uid", "origin")}
        r["scan"].update(source_file="anonymised upload", series_uid=anon_id(uid))
        r["job_id"] = case_id
        r.pop("consent", None)
        pdf = d / "report_anon.pdf"
        report.build_pdf(pdf, r, closeups, overview)
        table = pd.read_csv(d / "table.csv")
        table["seriesuid"] = case_id
        files = {"result.json": json.dumps(r, indent=1).encode(), "report.pdf": pdf.read_bytes(),
                 "table.csv": table.to_csv(index=False).encode()}
        for cid, png in closeups.items():
            files[f"nodule_{cid}.png"] = png
        nods = r["nodules"]
        ps = [n["p_malignant"] for n in nods if n.get("p_malignant") is not None]
        STORE.save_case(case_id, files, {
            "case_id": case_id, "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_type": "full-body" if r["lungs"]["full_body"] else "chest",
            "n_candidates": r["n_candidates"], "n_nodules": len(nods),
            "n_suspicious": sum(1 for n in nods if str(n.get("ai_label", "")).startswith("Suspicious")),
            "max_p_malignant": round(max(ps), 3) if ps else "", "slice_mm": r["scan"].get("spacing_mm", ["", "", ""])[2],
            "model": "demo" if r["models"].get("demo") else "trained"})
        STORE_STATUS["saved"] += 1
        STATE.get(job, {})["stored"] = case_id
    except Exception as e:
        traceback.print_exc()
        STORE_STATUS["failed"] += 1
        STORE_STATUS["error"] = f"{time.strftime('%H:%M')} {e}"[:300]


def _mesh_scaled(lung, image, f=3):
    """Lung surface from a downsampled mask, mapped back to full-resolution world coordinates."""
    small = lung[::f, ::f, ::f]
    if small.sum() < 10:
        return None
    v, fc, _, _ = measure.marching_cubes(np.pad(small, 1).astype(np.float32), 0.5, step_size=1)
    v = (v - 1) * f
    o = np.array(image.GetOrigin()); sp = np.array(image.GetSpacing())
    D = np.array(image.GetDirection()).reshape(3, 3)
    xyz = (D @ (v[:, ::-1] * sp).T).T + o
    return {"x": np.round(xyz[:, 0], 1).tolist(), "y": np.round(xyz[:, 1], 1).tolist(),
            "z": np.round(xyz[:, 2], 1).tolist(), "i": fc[:, 0].tolist(), "j": fc[:, 1].tolist(), "k": fc[:, 2].tolist()}


def _start(job):
    STATE[job] = {"job_id": job, "state": "queued", "step": "load", "message": "Queued",
                  "steps": STEPS, "created": time.time()}
    threading.Thread(target=run_job, args=(job,), daemon=True).start()


def _arrays(job):
    if job not in CACHE:
        p = JOBS / job / "volume.npz"
        if not p.exists():
            raise HTTPException(404, "Volume not available")
        z = np.load(p)
        CACHE[job] = {"arr": z["arr"], "lung": z["lung"], "labels": z["labels"]}
    return CACHE[job]


def _result(job):
    p = JOBS / job / "result.json"
    if not p.exists():
        raise HTTPException(404, "Result not ready")
    return json.loads(p.read_text())


# ------------------------------------------------------------------ API
@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), consent: str = Form("0")):
    job = uuid.uuid4().hex[:16]
    up = JOBS / job / "upload"
    up.mkdir(parents=True)
    if consent in ("1", "true", "on", "yes"):
        (JOBS / job / "consent").touch()
    total = 0
    for f in files:
        name = Path(f.filename or "upload.bin").name
        with open(up / name, "wb") as out:
            while chunk := await f.read(4 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1e6:
                    shutil.rmtree(JOBS / job, ignore_errors=True)
                    raise HTTPException(413, f"Upload too large (limit {MAX_UPLOAD_MB:.0f} MB).")
                out.write(chunk)
    _start(job)
    return {"job_id": job}


def _cleanup_loop():
    """Delete every job (uploaded scan, volume, report) older than JOB_TTL_HOURS."""
    while True:
        cutoff = time.time() - JOB_TTL_HOURS * 3600
        for d in JOBS.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff and STATE.get(d.name, {}).get("state") != "running":
                    shutil.rmtree(d, ignore_errors=True)
                    STATE.pop(d.name, None)
                    CACHE.pop(d.name, None)
            except Exception:
                pass
        time.sleep(600)


@app.on_event("startup")
def _start_cleanup():
    threading.Thread(target=_cleanup_loop, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"job_ttl_hours": JOB_TTL_HOURS, "max_upload_mb": MAX_UPLOAD_MB, "research_storage": STORE is not None}


@app.on_event("startup")
def _make_samples():
    """Create the two synthetic sample scans on first start (keeps the download small)."""
    if not list(SAMPLES.glob("*.mhd")):
        import subprocess, sys
        for extra, seed in (([], "101"), (["--full-body"], "202")):
            subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_ct.py"), "--out", str(SAMPLES),
                            "--n", "1", "--seed", seed, *extra], check=False)


@app.post("/api/demo")
def demo(kind: str = "chest"):
    src = sorted(SAMPLES.glob(f"*{'fullbody' if kind == 'fullbody' else 'chest'}*.mhd"))
    if not src:
        raise HTTPException(404, "No sample scans. Run: python tools/make_synthetic_ct.py --out data/samples --n 1")
    job = uuid.uuid4().hex[:16]
    up = JOBS / job / "upload"
    up.mkdir(parents=True)
    shutil.copy(src[0], up)
    shutil.copy(src[0].with_suffix(".raw"), up)
    _start(job)
    return {"job_id": job}


@app.get("/api/jobs/{job}")
def status(job: str):
    if job in STATE:
        s = dict(STATE[job])
        return JSONResponse({k: v for k, v in s.items() if k != "result"} | ({"result": s["result"]} if "result" in s else {}))
    p = JOBS / job / "result.json"
    if p.exists():
        return {"job_id": job, "state": "done", "steps": STEPS, "result": json.loads(p.read_text())}
    raise HTTPException(404, "Unknown job")


@app.get("/api/jobs/{job}/slice")
def slice_png(job: str, axis: str = "axial", index: int = 0, lung: int = 0):
    if axis not in ("axial", "coronal", "sagittal"):
        raise HTTPException(400, "axis")
    a = _arrays(job)
    nods = _result(job)["nodules"]
    return Response(report.render_slice(a["arr"], a["labels"], a["lung"], axis, index, nods, bool(lung)),
                    media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/jobs/{job}/scene")
def scene(job: str):
    return FileResponse(JOBS / job / "scene.json", media_type="application/json")


@app.get("/api/jobs/{job}/nodule/{cid}.png")
def nodule_png(job: str, cid: int):
    p = JOBS / job / f"nodule_{cid}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/jobs/{job}/overview.png")
def overview(job: str):
    return FileResponse(JOBS / job / "overview.png", media_type="image/png")


@app.get("/api/jobs/{job}/report.pdf")
def pdf(job: str):
    return FileResponse(JOBS / job / "report.pdf", media_type="application/pdf", filename=f"LungLocAI_report_{job}.pdf")


@app.get("/api/jobs/{job}/table.csv")
def table(job: str):
    return FileResponse(JOBS / job / "table.csv", media_type="text/csv", filename=f"LungLocAI_features_{job}.csv")


@app.get("/api/model")
def model_info():
    return {"loaded": list(MODELS), "details": {k: m.info() for k, m in MODELS.items()},
            "feature_columns": FEATURE_COLUMNS}


# ------------------------------------------------------------------ owner-only admin
def _admin_token() -> str:
    return hmac.new(ADMIN_PASSWORD.encode(), b"lunglocai-admin-session", hashlib.sha256).hexdigest()


def _require_admin(request: Request, need_store: bool = True):
    if not ADMIN_PASSWORD:
        raise HTTPException(503, "Admin is disabled: set the ADMIN_PASSWORD environment variable.")
    if not hmac.compare_digest(request.cookies.get("llai_admin", ""), _admin_token()):
        raise HTTPException(401, "Not logged in")
    if need_store and STORE is None:
        raise HTTPException(503, "Research storage is not configured (DATA_REPO + DATA_REPO_TOKEN).")


@app.post("/api/admin/login")
async def admin_login(request: Request):
    body = await request.json()
    if not ADMIN_PASSWORD or not hmac.compare_digest(str(body.get("password", "")).encode(), ADMIN_PASSWORD.encode()):
        time.sleep(1.5)                                      # slow down guessing
        raise HTTPException(401, "Wrong password" if ADMIN_PASSWORD else "Admin is disabled (no ADMIN_PASSWORD set)")
    resp = JSONResponse({"ok": True})
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie("llai_admin", _admin_token(), httponly=True, secure=secure, samesite="strict", max_age=12 * 3600)
    return resp


@app.post("/api/admin/logout")
def admin_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("llai_admin")
    return resp


@app.get("/api/admin/status")
def admin_status(request: Request):
    _require_admin(request, need_store=False)
    info = {"store": dict(STORE_STATUS), "models": {k: m.info() for k, m in MODELS.items()},
            "retrain": {k: RETRAIN[k] for k in ("state", "report")}}
    if STORE is None:
        return info
    try:
        if hasattr(STORE, "check"):
            info["store"].update(STORE.check())
        info["base_table"] = bool(STORE.list("training/base_table.csv"))
        info["n_cases"] = len(STORE.cases())
        labels = STORE.labels()
        info["n_labels"] = len(labels)
        info["label_counts"] = {k: sum(1 for l in labels if l["label"] == k)
                                for k in ("malignant", "benign", "not_nodule", "unsure")}
    except Exception as e:
        info["store"]["error"] = str(e)[:300]
    return info


@app.get("/api/admin/cases")
def admin_cases(request: Request):
    _require_admin(request)
    done = {}
    for l in STORE.labels():
        done[l["case_id"]] = done.get(l["case_id"], 0) + 1
    rows = STORE.cases()
    for r in rows:
        r["labelled"] = done.get(r["case_id"], 0)
    return sorted(rows, key=lambda r: r["stored_at"], reverse=True)


def _safe_case(case_id: str) -> str:
    if not case_id.replace("-", "").isalnum() or len(case_id) > 40:
        raise HTTPException(400, "bad case id")
    return case_id


@app.get("/api/admin/cases/{case_id}")
def admin_case(case_id: str, request: Request):
    _require_admin(request)
    raw = STORE.read(f"cases/{_safe_case(case_id)}/result.json")
    if not raw:
        raise HTTPException(404)
    r = json.loads(raw)
    lab = {str(l["candidate_id"]): l["label"] for l in STORE.labels() if l["case_id"] == case_id}
    for n in r.setdefault("nodules", []):
        n["owner_label"] = lab.get(str(n["candidate_id"]), "")
    return r


@app.get("/api/admin/cases/{case_id}/files/{name}")
def admin_case_file(case_id: str, name: str, request: Request):
    _require_admin(request)
    if "/" in name or ".." in name or "\\" in name:
        raise HTTPException(400)
    data = STORE.read(f"cases/{_safe_case(case_id)}/{name}")
    if data is None:
        raise HTTPException(404)
    mt = {"png": "image/png", "pdf": "application/pdf", "csv": "text/csv",
          "json": "application/json"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
    return Response(data, media_type=mt, headers={"Cache-Control": "private, max-age=600"})


@app.post("/api/admin/label")
async def admin_label(request: Request):
    _require_admin(request)
    b = await request.json()
    try:
        STORE.set_label(_safe_case(str(b["case_id"])), int(b["candidate_id"]), str(b.get("label", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/admin/base-table")
async def admin_base_table(request: Request, file: UploadFile = File(...)):
    _require_admin(request)
    raw = await file.read()
    try:
        t = pd.read_csv(io.BytesIO(raw), nrows=200)
    except Exception:
        raise HTTPException(400, "Not a readable CSV file")
    from .pipeline.lla_features import LLA_FEATURE_COLUMNS
    feats = [c for c in FEATURE_COLUMNS + LLA_FEATURE_COLUMNS if c in t.columns]
    labels = [c for c in ("is_nodule", "malignant", "label") if c in t.columns]
    if len(feats) < 5 or not labels:
        raise HTTPException(400, f"The CSV needs the website's feature columns ({len(feats)} of {len(FEATURE_COLUMNS)} found) "
                                 "and an 'is_nodule' and/or 'malignant' label column. Build it with training/build_luna16_table.py.")
    STORE.write("training/base_table.csv", raw, f"Upload base training table ({len(raw) // 1024} KB)")
    return {"ok": True, "features": len(feats), "labels": labels}


def _run_retrain():
    global MODELS
    RETRAIN.update(state="running", log=[], report=None)
    try:
        rep = retrainer.retrain(STORE, MODELS, LIVE_MODELS, log=lambda m: RETRAIN["log"].append(str(m)))
        RETRAIN["report"] = rep
        MODELS = classifier.load_models(LIVE_MODELS)
        RETRAIN["state"] = "done"
    except Exception as e:
        traceback.print_exc()
        RETRAIN["log"].append(f"ERROR: {e}")
        RETRAIN["state"] = "error"


@app.post("/api/admin/retrain")
def admin_retrain(request: Request):
    _require_admin(request)
    if RETRAIN["state"] == "running":
        raise HTTPException(409, "Retraining is already running")
    RETRAIN["state"] = "running"
    threading.Thread(target=_run_retrain, daemon=True).start()
    return {"started": True}


@app.post("/api/admin/selftest")
def admin_selftest(request: Request):
    """Write, read back and verify a small file in the private store."""
    _require_admin(request)
    t0 = time.time()
    token = uuid.uuid4().hex
    info = STORE.check() if hasattr(STORE, "check") else {"backend": STORE.name}
    if info.get("private") is False:
        raise HTTPException(400, "The storage repository is PUBLIC. Make it private before storing anything.")
    STORE.write("_selftest/last.txt", token.encode(), "Storage self-test")
    ok = (STORE.read("_selftest/last.txt") or b"").decode() == token
    return {"ok": ok, "seconds": round(time.time() - t0, 2), **info}


@app.get("/api/admin/retrain")
def admin_retrain_status(request: Request):
    _require_admin(request, need_store=False)
    return RETRAIN


@app.on_event("startup")
def _pull_models():
    """Load the latest retrained models from the private store (containers start empty)."""
    global MODELS
    if STORE is None:
        return
    try:
        got = False
        for name in ("nodule_model", "malignancy_model"):
            a, b = STORE.read(f"models/{name}.json"), STORE.read(f"models/{name}.meta.json")
            if a and b:
                LIVE_MODELS.mkdir(parents=True, exist_ok=True)
                (LIVE_MODELS / f"{name}.json").write_bytes(a)
                (LIVE_MODELS / f"{name}.meta.json").write_bytes(b)
                got = True
        if got:
            MODELS = classifier.load_models(LIVE_MODELS)
    except Exception as e:
        STORE_STATUS["error"] = f"model pull: {e}"[:300]


@app.get("/admin", include_in_schema=False)
def admin_page():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/admin.html")


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
