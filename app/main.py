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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from skimage import measure

from .pipeline import classifier, lungrads
from .pipeline.ct_io import load_ct
from .pipeline.features import FEATURE_COLUMNS
from .pipeline.scan_to_table import scan_to_table
from . import report

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
MODELS = classifier.load_models()

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
            _set(job, state="done", step="done", message="Finished", result=result)
            shutil.rmtree(d / "upload", ignore_errors=True)
        except Exception as e:
            traceback.print_exc()
            _set(job, state="error", message=str(e) or e.__class__.__name__)


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
async def analyze(files: list[UploadFile] = File(...)):
    job = uuid.uuid4().hex[:16]
    up = JOBS / job / "upload"
    up.mkdir(parents=True)
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


@app.get("/api/config")
def config():
    return {"job_ttl_hours": JOB_TTL_HOURS, "max_upload_mb": MAX_UPLOAD_MB}


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


@app.post("/api/model/reload")
def reload_models():
    global MODELS
    MODELS = classifier.load_models()
    return model_info()


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
