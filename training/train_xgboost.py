"""Train the XGBoost model(s) used by the website from the LungLocAI table.

    # 1) nodule vs. false-positive (labels straight from LUNA16 annotations)
    python training/train_xgboost.py --table data/lunglocai_table.csv --task nodule

    # 2) lung cancer (malignant) vs. benign, on nodule rows only (needs a 'malignant' column)
    python training/train_xgboost.py --table data/lunglocai_table.csv --task malignancy

Your own table works too, as long as its feature columns are named like the extractor's
(see app/pipeline/features.py::FEATURE_COLUMNS). Use --rename to map different names, e.g.
    --rename "Diameter:diameter_mm,MeanHU:mean_hu"   and   --label-col YourLabelColumn

Validation is grouped by scan (GroupKFold on seriesuid) so candidates from the same CT never sit
in both train and test folds. Outputs models/<task>_model.json + .meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline.features import FEATURE_COLUMNS  # noqa: E402
from app.pipeline.trainer import cv_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--task", choices=["nodule", "malignancy"], required=True)
    ap.add_argument("--label-col", default=None, help="default: is_nodule (nodule) / malignant (malignancy)")
    ap.add_argument("--group-col", default="seriesuid")
    ap.add_argument("--rename", default="", help='"old:new,old2:new2"')
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--demo", action="store_true", help="mark model as DEMO (synthetic data)")
    a = ap.parse_args()

    df = pd.read_csv(a.table)
    if a.rename:
        df = df.rename(columns=dict(p.split(":") for p in a.rename.split(",")))
    label = a.label_col or ("is_nodule" if a.task == "nodule" else "malignant")
    if label not in df.columns:
        sys.exit(f"Label column '{label}' not in table. Columns: {list(df.columns)}")
    if a.task == "malignancy" and "is_nodule" in df.columns and label != "is_nodule":
        df = df[df.is_nodule == 1]
    df = df[df[label].notna()].copy()
    y = df[label].astype(int).values

    feats = [f for f in FEATURE_COLUMNS if f in df.columns]
    unknown = [c for c in df.select_dtypes("number").columns
               if c not in FEATURE_COLUMNS and c not in (label, "is_nodule", "malignant", "candidate_id",
                                                         "voxel_z", "voxel_y", "voxel_x", "world_x", "world_y",
                                                         "world_z", "matched_diameter_mm", "full_body_scan",
                                                         "core_solid_fraction")]
    if len(feats) < 5:
        sys.exit("Fewer than 5 feature columns match the website's extractor. Use --rename to map your "
                 f"column names to: {FEATURE_COLUMNS}")
    if unknown:
        print(f"NOTE: ignoring numeric columns the website cannot compute at inference time: {unknown}")
    print(f"task={a.task} label={label} rows={len(df)} positives={int(df[label].astype(int).sum())} features={len(feats)}")
    groups = df[a.group_col].values if a.group_col in df.columns else None
    final, info = cv_train(df, label, groups=groups, folds=a.folds, rounds=a.rounds)
    print(json.dumps(info["metrics"], indent=2))
    threshold, n_final = info["threshold"], info["rounds"]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{a.task}_model"
    final.save_model(str(out / f"{name}.json"))
    meta = dict(info, task=a.task, label=label, trained_on=Path(a.table).name, demo=bool(a.demo),
                created=datetime.now().isoformat(timespec="seconds"))
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / name}.json (+ .meta.json), threshold={threshold:.3f}, rounds={n_final}")


if __name__ == "__main__":
    main()
