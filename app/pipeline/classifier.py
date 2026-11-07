"""XGBoost models that run on the nodule feature table.

Two optional models live in models/ (each = <name>.json XGBoost booster + <name>.meta.json):

  nodule_model      task="nodule"      : real nodule vs. false positive (vessel, scar, ...)
                                         -> trained from LUNA16 candidates / annotations
  malignancy_model  task="malignancy"  : lung-cancer (malignant) vs. benign nodule
                                         -> needs malignancy labels (e.g. LIDC-IDRI ratings)

Use one or both. Train with training/train_xgboost.py. The .meta.json stores the feature list,
the decision threshold and the validation metrics so the report can show them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class Model:
    def __init__(self, name: str, model_dir: Path = MODEL_DIR):
        self.name = name
        self.meta = json.loads((model_dir / f"{name}.meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / f"{name}.json"))
        self.features = self.meta["features"]
        if self.booster.feature_names is None and self.booster.num_features() == len(self.features):
            self.booster.feature_names = list(self.features)
        best = self.booster.attr("best_iteration")        # sklearn early-stopping models predict with best_iteration
        self.iter_range = (0, int(best) + 1) if best is not None else (0, 0)
        env = os.environ.get(f"{name.upper()}_THRESHOLD")    # e.g. NODULE_MODEL_THRESHOLD=0.5
        self.threshold = float(env if env else self.meta.get("threshold", 0.5))

    def predict(self, df: pd.DataFrame):
        missing = [f for f in self.features if f not in df.columns]
        X = df.reindex(columns=self.features).astype(float)
        p = self.booster.predict(xgb.DMatrix(X.values, feature_names=self.features, missing=np.nan),
                                 iteration_range=self.iter_range)
        return p, missing

    def info(self):
        m = self.meta
        d = {k: m.get(k) for k in ("task", "label", "trained_on", "n_rows", "n_scans", "metrics", "demo", "created",
                                   "top_features", "source", "operating_points", "feature_set")}
        d["threshold"] = self.threshold
        return d


def load_models(*dirs: Path):
    """Load each model from the first directory that has it (e.g. retrained models first, bundled last)."""
    models = {}
    for name in ("nodule_model", "malignancy_model"):
        for d in [*dirs, MODEL_DIR]:
            d = Path(d)
            if (d / f"{name}.json").exists() and (d / f"{name}.meta.json").exists():
                models[name] = Model(name, d)
                break
    return models


def risk_tier(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.3:
        return "Intermediate"
    return "Low"


def classify(df: pd.DataFrame, models: dict) -> tuple[pd.DataFrame, dict]:
    """Adds p_nodule / is_nodule / p_malignant / ai_label columns. Returns (df, notes)."""
    notes = {"warnings": []}
    df = df.copy()
    if df.empty:
        return df, notes

    if "nodule_model" in models:
        m = models["nodule_model"]
        p, miss = m.predict(df)
        df["p_nodule"] = np.round(p, 4)
        df["is_nodule"] = (p >= m.threshold).astype(int)
        if miss:
            notes["warnings"].append(f"nodule_model expects features not produced by the extractor: {miss}")
    else:
        df["p_nodule"] = np.nan
        df["is_nodule"] = 1  # no false-positive filter -> every candidate is reported

    if "malignancy_model" in models:
        m = models["malignancy_model"]
        p, miss = m.predict(df)
        df["p_malignant"] = np.round(p, 4)
        df["ai_label"] = np.where(p >= m.threshold, "Suspicious for lung cancer", "Likely benign")
        df["ai_risk"] = [risk_tier(x) for x in p]
        if miss:
            notes["warnings"].append(f"malignancy_model expects features not produced by the extractor: {miss}")
    elif "nodule_model" in models:
        # only a nodule-vs-non-nodule model is available (plain LUNA16 labels)
        df["p_malignant"] = np.nan
        df["ai_label"] = np.where(df["is_nodule"] == 1, "Nodule detected", "Not a nodule")
        df["ai_risk"] = "n/a"
        notes["warnings"].append("No malignancy model loaded: results show nodule detection only. "
                                 "Train malignancy_model with malignancy labels to get cancer probability.")
    else:
        df["p_malignant"] = np.nan
        df["ai_label"] = "Not classified"
        df["ai_risk"] = "n/a"
        notes["warnings"].append("No XGBoost model found in models/. Run training/train_xgboost.py.")
    return df, notes
