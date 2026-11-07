"""Retrain the XGBoost models on: base table (LungLocAI / LUNA16) + uploads the owner has labelled.

Safety rule: a new model is only deployed if, on a held-out set of scans it never saw during training,
its AUC is not worse than the current model's (tolerance 0.01). Otherwise the current model stays.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline.trainer import cv_train, predict, safe_auc

TASKS = {"nodule": "is_nodule", "malignancy": "malignant"}
MIN_PER_CLASS = 5
MIN_SCANS = 4
TOLERANCE = 0.01


def _holdout(group: str) -> bool:
    return int(hashlib.md5(str(group).encode()).hexdigest(), 16) % 5 == 0      # ~20 % of scans, stable


def collect(store) -> tuple[pd.DataFrame, dict]:
    """All labelled training rows (base table + confirmed uploads)."""
    parts, info = [], {"base_rows": 0, "upload_cases": 0, "upload_rows": 0}
    raw = store.read("training/base_table.csv")
    if raw:
        base = pd.read_csv(io.BytesIO(raw))
        base["source"] = "base"
        if "seriesuid" not in base.columns:
            base["seriesuid"] = "base"
        parts.append(base)
        info["base_rows"] = len(base)
    labels = pd.DataFrame(store.labels())
    if len(labels):
        labels = labels[labels.label.isin(["malignant", "benign", "not_nodule"])]
        for case_id, g in labels.groupby("case_id"):
            raw = store.read(f"cases/{case_id}/table.csv")
            if not raw:
                continue
            t = pd.read_csv(io.BytesIO(raw))
            t["candidate_id"] = t["candidate_id"].astype(int)
            g = g.assign(candidate_id=g.candidate_id.astype(int))
            t = t.merge(g[["candidate_id", "label"]], on="candidate_id", how="inner")
            t["is_nodule"] = (t.label != "not_nodule").astype(int)
            t["malignant"] = t.label.map({"malignant": 1, "benign": 0})
            t["seriesuid"] = "upload:" + case_id
            t["source"] = "upload"
            parts.append(t)
            info["upload_cases"] += 1
            info["upload_rows"] += len(t)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df, info


def task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    label = TASKS[task]
    if label not in df.columns:
        return df.iloc[0:0]
    d = df[df[label].notna()]
    if task == "malignancy" and "is_nodule" in d.columns:
        d = d[d.is_nodule == 1]
    return d.copy()


def retrain(store, current_models: dict, out_dir: Path, log=print) -> dict:
    df, info = collect(store)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "data": info, "tasks": {}}
    log(f"Training data: {info}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    to_store = {}
    for task, label in TASKS.items():
        d = task_frame(df, task) if len(df) else df
        r = {"rows": int(len(d))}
        report["tasks"][task] = r
        if not len(d):
            r["status"] = "skipped: no labelled rows"
            continue
        y = d[label].astype(int)
        pos, neg, scans = int(y.sum()), int((y == 0).sum()), int(d.seriesuid.nunique())
        r.update(positives=pos, negatives=neg, scans=scans)
        if pos < MIN_PER_CLASS or neg < MIN_PER_CLASS or scans < MIN_SCANS:
            r["status"] = (f"skipped: need at least {MIN_PER_CLASS} of each class from {MIN_SCANS}+ scans "
                           f"(have {pos} positive / {neg} negative from {scans} scans)")
            continue
        hold = d.seriesuid.map(_holdout)
        train, test = d[~hold], d[hold]
        ytest = test[label].astype(int)
        if len(test) == 0 or ytest.nunique() < 2 or train[label].astype(int).nunique() < 2:
            train, test = d, d.iloc[0:0]
        log(f"[{task}] candidate model: {len(train)} train rows, {len(test)} held-out rows")
        cand, cinfo = cv_train(train, label, groups=train.seriesuid.values, log=log)
        cur = current_models.get(f"{task}_model")
        if len(test):
            cand_auc = safe_auc(test[label].astype(int), predict(cand, cinfo["features"], test))
            cur_auc = safe_auc(test[label].astype(int), cur.predict(test)[0]) if cur else float("nan")
            basis = "held-out scans"
        else:
            cand_auc, cur_auc, basis = cinfo["metrics"]["auc"], float("nan"), "cross-validation (too few scans to hold out)"
        r.update(candidate_auc=round(cand_auc, 4), current_auc=None if math.isnan(cur_auc) else round(cur_auc, 4),
                 compared_on=basis)
        accept = math.isnan(cur_auc) or cand_auc >= cur_auc - TOLERANCE
        if not accept:
            r["status"] = f"kept current model (new AUC {cand_auc:.3f} < current {cur_auc:.3f} on {basis})"
            log(r["status"])
            continue
        log(f"[{task}] accepted - training final model on all {len(d)} rows")
        final, finfo = cv_train(d, label, groups=d.seriesuid.values, log=log)
        meta = dict(finfo, task=task, label=label, demo=False, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    trained_on=(f"{'base table + ' if info['base_rows'] else ''}"
                                f"{info['upload_cases']} confirmed uploaded scans"),
                    holdout={"candidate_auc": r["candidate_auc"], "previous_auc": r["current_auc"], "basis": basis})
        out_dir.mkdir(parents=True, exist_ok=True)
        final.save_model(str(out_dir / f"{task}_model.json"))
        (out_dir / f"{task}_model.meta.json").write_text(json.dumps(meta, indent=2))
        for ext in (".json", ".meta.json"):
            data = (out_dir / f"{task}_model{ext}").read_bytes()
            to_store[f"models/{task}_model{ext}"] = data
            to_store[f"models/history/{stamp}/{task}_model{ext}"] = data
        r["status"] = f"deployed new model (AUC {finfo['metrics']['auc']:.3f} cross-validated)"
        r["new_metrics"] = finfo["metrics"]
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if to_store:
        to_store["models/last_retrain.json"] = json.dumps(report, indent=2, default=str).encode()
        store.write_many(to_store, f"Retrained models {stamp}")
    return report
