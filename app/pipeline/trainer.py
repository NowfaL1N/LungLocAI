"""XGBoost training core shared by training/train_xgboost.py (command line) and the admin Retrain button."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .features import FEATURE_COLUMNS

PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=4, eta=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, tree_method="hist")


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


def metrics_at(y, p, threshold) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": round(safe_auc(y, p), 4),
        "average_precision": round(float(average_precision_score(y, p)), 4) if y.sum() else float("nan"),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "precision": round(float(tp / max(tp + fp, 1)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def feature_list(df: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_COLUMNS if f in df.columns]


def cv_train(df: pd.DataFrame, label: str, groups=None, folds: int = 5, rounds: int = 600, log=print):
    """Cross-validated XGBoost. Returns (final_booster, info) where info has features, threshold, metrics..."""
    feats = feature_list(df)
    X = df[feats].astype(float).values
    y = df[label].astype(int).values
    n_groups = len(set(groups)) if groups is not None else 0
    k = max(2, min(folds, n_groups if n_groups else folds, int(min(y.sum(), (y == 0).sum()))))
    if groups is not None and n_groups >= k:
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
        cv_kind = f"GroupKFold({k}) by scan"
    else:
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(X, y))
        cv_kind = f"StratifiedKFold({k})"
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    params = dict(PARAMS, scale_pos_weight=spw)
    oof = np.full(len(y), np.nan)
    best = []
    for i, (tr, te) in enumerate(splits):
        dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=feats)
        dte = xgb.DMatrix(X[te], label=y[te], feature_names=feats)
        bst = xgb.train(params, dtr, rounds, evals=[(dte, "val")], early_stopping_rounds=50, verbose_eval=False)
        best.append(bst.best_iteration + 1)
        oof[te] = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        log(f"  fold {i + 1}: AUC={safe_auc(y[te], oof[te]):.3f}")
    ok = ~np.isnan(oof)
    fpr, tpr, thr = roc_curve(y[ok], oof[ok])
    threshold = float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.05, 0.95))
    metrics = metrics_at(y[ok], oof[ok], threshold)
    metrics["cv"] = cv_kind
    n_final = max(int(np.median(best)), 40)
    final = xgb.train(params, xgb.DMatrix(X, label=y, feature_names=feats), n_final)
    gain = final.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:10]
    info = {"features": feats, "threshold": round(threshold, 4), "rounds": n_final, "params": params,
            "metrics": metrics, "n_rows": int(len(df)), "n_positive": int(y.sum()), "n_scans": int(n_groups),
            "top_features": [[k_, round(v, 2)] for k_, v in top]}
    return final, info


def predict(booster: xgb.Booster, feats: list[str], df: pd.DataFrame) -> np.ndarray:
    X = df.reindex(columns=feats).astype(float).values
    return booster.predict(xgb.DMatrix(X, feature_names=feats, missing=np.nan))
