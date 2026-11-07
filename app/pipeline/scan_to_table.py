"""CT image  ->  lungs  ->  nodule candidates  ->  feature table (pandas DataFrame).

Shared by the website and by the LUNA16 training-table builder, so training and inference use
exactly the same steps.
"""
from __future__ import annotations

import pandas as pd

from .features import FEATURE_COLUMNS, INFO_COLUMNS, compute_features, lung_bbox_info
from .lla_features import LLA_FEATURE_COLUMNS, lla_features
from .lungs import extract_lungs
from .nodules import detect_candidates


def scan_to_table(img, progress=lambda step, msg="": None):
    progress("lungs", "Converting to a 1 mm 3-D volume and extracting the lungs")
    L = extract_lungs(img)
    progress("candidates", "Searching the lungs for nodule candidates")
    cands = detect_candidates(L["array"], L["lung"])
    progress("features", f"Measuring {len(cands)} candidates (3-D shape, density, margin, location)")
    bb = lung_bbox_info(L["lung"], L["image"])
    rows = []
    for i, c in enumerate(cands, start=1):
        r = compute_features(c, L["array"], L["lung"], L["side_map"], L["image"], bb)
        r["candidate_id"] = i
        # LungLocAI-3D features: 32 mm cube resampled from the ORIGINAL scan, exactly as in the dataset
        r.update(lla_features(img, (r["world_x"], r["world_y"], r["world_z"])))
        rows.append(r)
    cols = INFO_COLUMNS + FEATURE_COLUMNS + LLA_FEATURE_COLUMNS
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return L, cands, df, bb
