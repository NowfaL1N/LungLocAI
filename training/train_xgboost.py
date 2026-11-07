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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
                             average_precision_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline.features import FEATURE_COLUMNS  # noqa: E402

PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=4, eta=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, tree_method="hist")


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
    X = df[feats].astype(float).values
    print(f"task={a.task} label={label} rows={len(df)} positives={int(y.sum())} features={len(feats)}")

    groups = df[a.group_col].values if a.group_col in df.columns else None
    n_groups = len(set(groups)) if groups is not None else 0
    if groups is not None and n_groups >= a.folds:
        splits = GroupKFold(n_splits=a.folds).split(X, y, groups)
        cv_kind = f"GroupKFold({a.folds}) by {a.group_col}"
    else:
        splits = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0).split(X, y)
        cv_kind = f"StratifiedKFold({a.folds})"

    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    params = dict(PARAMS, scale_pos_weight=spw)
    oof = np.zeros(len(y))
    best_iters = []
    for k, (tr, te) in enumerate(splits):
        dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=feats)
        dte = xgb.DMatrix(X[te], label=y[te], feature_names=feats)
        bst = xgb.train(params, dtr, a.rounds, evals=[(dte, "val")], early_stopping_rounds=50, verbose_eval=False)
        best_iters.append(bst.best_iteration + 1)
        oof[te] = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        print(f"  fold {k + 1}: AUC={roc_auc_score(y[te], oof[te]) if len(set(y[te])) > 1 else float('nan'):.3f}")

    fpr, tpr, thr = roc_curve(y, oof)
    j = np.argmax(tpr - fpr)
    threshold = float(np.clip(thr[j], 0.05, 0.95))
    pred = (oof >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "cv": cv_kind,
        "auc": round(float(roc_auc_score(y, oof)), 4),
        "average_precision": round(float(average_precision_score(y, oof)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "precision": round(float(tp / max(tp + fp, 1)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }
    print(json.dumps(metrics, indent=2))

    n_final = max(int(np.median(best_iters)), 40)
    final = xgb.train(params, xgb.DMatrix(X, label=y, feature_names=feats), n_final)
    gain = final.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:10]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{a.task}_model"
    final.save_model(str(out / f"{name}.json"))
    meta = {
        "task": a.task, "label": label, "features": feats, "threshold": round(threshold, 4),
        "rounds": n_final, "params": params, "metrics": metrics,
        "trained_on": Path(a.table).name, "n_rows": int(len(df)), "n_positive": int(y.sum()),
        "n_scans": int(n_groups), "demo": bool(a.demo),
        "created": datetime.now().isoformat(timespec="seconds"),
        "top_features": [[k, round(v, 2)] for k, v in top],
    }
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / name}.json (+ .meta.json), threshold={threshold:.3f}, rounds={n_final}")


if __name__ == "__main__":
    main()
