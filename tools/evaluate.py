"""Measure the accuracy of the whole pipeline (detection + XGBoost) against an answer key.

Works for the synthetic test set AND for real LUNA16 scans:

    python tools/evaluate.py --scans test_scans --annotations test_scans/annotations.csv
    python tools/evaluate.py --scans D:/LUNA16/subset9 --annotations D:/LUNA16/annotations.csv --max-scans 20

--scans       folder with CT files (.mhd, .nii, .nii.gz, .nrrd, or DICOM .zip); the file name
              (without extension) must equal the 'seriesuid' in the annotations, like LUNA16
--annotations CSV with seriesuid, coordX, coordY, coordZ, diameter_mm  [+ optional 'malignant' 0/1]

A true nodule counts as DETECTED when a reported nodule's centre lies within max(radius, 4 mm)
of it. Reports: detection sensitivity, false positives per scan, size error, left/right accuracy and,
when 'malignant' labels exist, cancer-classification accuracy, sensitivity, specificity and AUC.
Writes <scans>/evaluation_results.csv (one row per true nodule) and evaluation_summary.txt.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline import classifier  # noqa: E402
from app.pipeline.ct_io import load_ct  # noqa: E402
from app.pipeline.scan_to_table import scan_to_table  # noqa: E402

EXTS = (".mhd", ".nii.gz", ".nii", ".nrrd", ".zip")


def stem(p: Path) -> str:
    n = p.name
    for e in EXTS:
        if n.lower().endswith(e):
            return n[: -len(e)]
    return p.stem


def find_scans(folder: Path):
    out = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name.lower().endswith(EXTS):
            out.setdefault(stem(p), p)
    for p in sorted(folder.glob("subset*/*.mhd")):          # LUNA16 layout
        out.setdefault(stem(p), p)
    return out


def load(path: Path):
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(path, tmp)
    if path.suffix.lower() == ".mhd":
        for ext in (".raw", ".zraw"):
            if path.with_suffix(ext).exists():
                shutil.copy(path.with_suffix(ext), tmp)
    try:
        return load_ct(tmp)[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--min-sensitivity", type=float, default=0.0,
                    help="exit with an error if detection sensitivity is below this (used by CI)")
    a = ap.parse_args()

    folder = Path(a.scans)
    ann = pd.read_csv(a.annotations)
    scans = find_scans(folder)
    uids = [u for u in ann.seriesuid.unique() if u in scans]
    if a.max_scans:
        uids = uids[: a.max_scans]
    if not uids:
        sys.exit("No scan files in --scans match the seriesuid values in --annotations.")
    models = classifier.load_models()
    has_mal = "malignant" in ann.columns and "malignancy_model" in models
    print(f"Evaluating {len(uids)} scans, {len(ann[ann.seriesuid.isin(uids)])} true nodules; "
          f"models: {list(models)}{' (DEMO)' if any(m.meta.get('demo') for m in models.values()) else ''}\n")

    rows, fps, t0 = [], [], time.time()
    for k, uid in enumerate(uids, 1):
        img = load(scans[uid])
        L, cands, df, _ = scan_to_table(img)
        df, _ = classifier.classify(df, models)
        rep = df[df.is_nodule == 1].reset_index(drop=True) if len(df) else df
        used = set()
        g = ann[ann.seriesuid == uid]
        for _, t in g.iterrows():
            hit = None
            if len(rep):
                d = np.sqrt((rep.world_x - t.coordX) ** 2 + (rep.world_y - t.coordY) ** 2 + (rep.world_z - t.coordZ) ** 2)
                i = int(d.idxmin())
                if d[i] <= max(t.diameter_mm / 2, 4.0):
                    hit = i
                    used.add(i)
            r = {"seriesuid": uid, "true_x": t.coordX, "true_y": t.coordY, "true_z": t.coordZ,
                 "true_diameter_mm": t.diameter_mm, "true_side": "Right" if t.coordX < 0 else "Left",
                 "true_malignant": t.get("malignant", np.nan), "true_kind": t.get("kind", ""),
                 "detected": hit is not None}
            if hit is not None:
                n = rep.loc[hit]
                r.update(found_diameter_mm=round(n.diameter_mm, 1), found_side=n.side, found_type=n.nodule_type,
                         p_malignant=n.get("p_malignant", np.nan), ai_label=n.ai_label,
                         distance_mm=round(float(np.sqrt((n.world_x - t.coordX) ** 2 + (n.world_y - t.coordY) ** 2
                                                         + (n.world_z - t.coordZ) ** 2)), 1))
            rows.append(r)
        fp = len(rep) - len(used)
        fps.append(fp)
        det = sum(1 for r in rows if r["seriesuid"] == uid and r["detected"])
        print(f"[{k}/{len(uids)}] {uid}: {det}/{len(g)} true nodules found, {fp} false positive(s) "
              f"({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(folder / "evaluation_results.csv", index=False)
    det = res[res.detected]
    lines = ["", "=" * 60, "LUNGLOCAI EVALUATION SUMMARY", "=" * 60,
             f"Scans: {len(uids)}    True nodules: {len(res)}",
             "", "Detection (finding the nodules)",
             f"  Sensitivity           : {len(det)}/{len(res)} = {len(det) / len(res) * 100:.1f}%",
             f"  False positives / scan: {np.mean(fps):.2f}  (total {sum(fps)})",
             f"  Mean size error       : {np.mean(np.abs(det.found_diameter_mm - det.true_diameter_mm)):.1f} mm"
             if len(det) else "  Mean size error       : n/a",
             f"  Mean position error   : {det.distance_mm.mean():.1f} mm" if len(det) else "",
             f"  Left/right correct    : {(det.found_side == det.true_side).mean() * 100:.0f}%" if len(det) else ""]
    if has_mal and res.true_malignant.notna().any():
        m = res[res.true_malignant.notna()].copy()
        m["pred"] = (m.ai_label.fillna("") == "Suspicious for lung cancer").astype(int)  # missed nodule -> not flagged
        y = m.true_malignant.astype(int)
        tp = int(((m.pred == 1) & (y == 1)).sum()); tn = int(((m.pred == 0) & (y == 0)).sum())
        fp_ = int(((m.pred == 1) & (y == 0)).sum()); fn = int(((m.pred == 0) & (y == 1)).sum())
        dm = m[m.detected]
        lines += ["", "Lung-cancer classification (XGBoost, all true nodules; missed = not flagged)",
                  f"  Accuracy   : {(tp + tn) / len(m) * 100:.1f}%",
                  f"  Sensitivity: {tp}/{tp + fn} malignant nodules flagged = {tp / max(tp + fn, 1) * 100:.1f}%",
                  f"  Specificity: {tn}/{tn + fp_} benign nodules not flagged = {tn / max(tn + fp_, 1) * 100:.1f}%",
                  f"  AUC (detected nodules): {auc(dm.true_malignant.astype(int), dm.p_malignant):.3f}",
                  f"  Confusion  : TP={tp}  FN={fn}  TN={tn}  FP={fp_}"]
    if any(mm.meta.get("demo") for mm in models.values()):
        lines += ["", "NOTE: DEMO model (trained on synthetic phantoms). Results on synthetic test scans show the",
                  "pipeline works; they are NOT clinical accuracy. Evaluate on real LUNA16 scans for that."]
    lines += [f"", f"Per-nodule results: {folder / 'evaluation_results.csv'}"]
    text = "\n".join(lines)
    print(text)
    (folder / "evaluation_summary.txt").write_text(text, encoding="utf-8")
    if len(det) / len(res) < a.min_sensitivity:
        sys.exit(f"FAILED: detection sensitivity {len(det) / len(res):.2f} < {a.min_sensitivity}")


if __name__ == "__main__":
    main()
