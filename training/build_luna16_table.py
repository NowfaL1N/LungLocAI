"""Build the LungLocAI training table from LUNA16 CT scans.

Every CT goes through the SAME pipeline as the website (3-D volume -> lung extraction -> nodule
candidates -> feature row), and each row is labelled from the LUNA16 annotations:

    is_nodule = 1  if the candidate lies within max(radius, 3 mm) of an annotated nodule
    malignant = 1/0 if a malignancy label file is supplied (see --malignancy)

Annotated nodules that the detector missed are added with region growing (source="annotation"),
so every radiologist-confirmed nodule is in the table.

Usage (Colab / local):
    python training/build_luna16_table.py \
        --luna-dir /content/LUNA16            # folder containing subset0..subset9 (*.mhd + *.raw)
        --annotations /content/LUNA16/annotations.csv \
        --out data/lunglocai_table.csv \
        [--malignancy lidc_malignancy.csv]    # columns: seriesuid,coordX,coordY,coordZ,malignancy (1-5) or malignant (0/1)
        [--workers 2] [--max-scans 50]

The script is resumable: scans already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline.features import FEATURE_COLUMNS, INFO_COLUMNS, compute_features, lung_bbox_info  # noqa: E402
from app.pipeline.nodules import region_grow_at  # noqa: E402
from app.pipeline.scan_to_table import scan_to_table  # noqa: E402

ANN = None
MAL = None


def _malignancy_label(row, mal, radius):
    if mal is None or mal.empty:
        return np.nan
    d = np.sqrt((mal.coordX - row.world_x) ** 2 + (mal.coordY - row.world_y) ** 2 + (mal.coordZ - row.world_z) ** 2)
    i = d.idxmin()
    if d[i] > max(radius, 5.0):
        return np.nan
    if "malignant" in mal.columns and not pd.isna(mal.loc[i, "malignant"]):
        return int(mal.loc[i, "malignant"])
    s = float(mal.loc[i, "malignancy"])          # LIDC mean rating 1..5
    if s >= 3.5:
        return 1
    if s <= 2.5:
        return 0
    return np.nan                                  # indeterminate (3) -> excluded from malignancy training


def process(path: str):
    uid = Path(path).stem
    try:
        img = sitk.ReadImage(path)
        L, cands, df, bb = scan_to_table(img)
        ann = ANN[ANN.seriesuid == uid]
        mal = MAL[MAL.seriesuid == uid] if MAL is not None else None
        df["source"] = "detector"
        df["is_nodule"] = 0
        df["matched_diameter_mm"] = np.nan
        matched = set()
        for j, a in ann.iterrows():
            if df.empty:
                break
            d = np.sqrt((df.world_x - a.coordX) ** 2 + (df.world_y - a.coordY) ** 2 + (df.world_z - a.coordZ) ** 2)
            hit = d <= max(a.diameter_mm / 2.0, 3.0)
            if hit.any():
                df.loc[hit, "is_nodule"] = 1
                df.loc[hit, "matched_diameter_mm"] = a.diameter_mm
                matched.add(j)
        # add the annotated nodules the detector missed
        extra = []
        for j, a in ann.iterrows():
            if j in matched:
                continue
            idx = L["image"].TransformPhysicalPointToContinuousIndex((a.coordX, a.coordY, a.coordZ))
            zyx = np.array(idx[::-1])
            if np.any(zyx < 0) or np.any(zyx >= np.array(L["array"].shape)):
                continue
            c = region_grow_at(L["array"], L["lung"], zyx, a.diameter_mm / 2.0)
            r = compute_features(c, L["array"], L["lung"], L["side_map"], L["image"], bb)
            r.update(candidate_id=len(df) + len(extra) + 1, source="annotation", is_nodule=1,
                     matched_diameter_mm=a.diameter_mm)
            extra.append(r)
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        if mal is not None or "malignant" in ann.columns:
            src = mal if (mal is not None and not mal.empty) else ann
            df["malignant"] = [(_malignancy_label(r, src, r.diameter_mm / 2) if r.is_nodule else 0)
                               for r in df.itertuples()]
        df.insert(0, "seriesuid", uid)
        df["full_body_scan"] = int(L["full_body"])
        return uid, df, None
    except Exception:
        return uid, None, traceback.format_exc()


def _init(ann, mal):
    global ANN, MAL
    ANN, MAL = ann, mal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--luna-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--malignancy", default=None)
    ap.add_argument("--out", default="data/lunglocai_table.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-scans", type=int, default=0)
    a = ap.parse_args()

    ann = pd.read_csv(a.annotations)
    mal = pd.read_csv(a.malignancy) if a.malignancy else None
    scans = sorted(str(p) for p in Path(a.luna_dir).rglob("*.mhd"))
    out = Path(a.out)
    done = set(pd.read_csv(out, usecols=["seriesuid"]).seriesuid) if out.exists() else set()
    todo = [s for s in scans if Path(s).stem not in done]
    if a.max_scans:
        todo = todo[: a.max_scans]
    print(f"{len(scans)} scans found, {len(done)} already in table, processing {len(todo)}")

    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(ann, mal)) as pool:
        for k, (uid, df, err) in enumerate(pool.imap_unordered(process, todo), start=1):
            if err:
                print(f"[{k}/{len(todo)}] {uid} FAILED\n{err}")
                continue
            df.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"[{k}/{len(todo)}] {uid}: {len(df)} rows, {int(df.is_nodule.sum())} nodules "
                  f"({time.time() - t0:.0f}s)")
    t = pd.read_csv(out)
    print(f"\nTable: {out}  rows={len(t)}  scans={t.seriesuid.nunique()}  nodule rows={int(t.is_nodule.sum())}")
    if "malignant" in t.columns:
        print("malignant label counts:", t.malignant.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
