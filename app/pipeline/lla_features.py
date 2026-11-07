"""LungLocAI-3D features: the exact 539-feature recipe of the published dataset.

Source: LungLocAI_dataset_builder.ipynb (cell 66), IEEE DataPort, DOI 10.21227/0x9w-1y93.
For each candidate a 32 x 32 x 32 mm cube is resampled at 1 mm from the ORIGINAL scan (linear interpolation,
-1000 HU outside the scan), clipped to [-1000, 400] HU, and described by:
  * mean / std / p10 / p50 / p90 / solid fraction (> -400 HU) inside spheres of radius 3, 6, 10 mm and the whole cube
  * centre (r <= 6 mm) vs shell (6-12 mm) contrast, mean gradient magnitude (r <= 6 mm and whole cube)
  * the cube averaged down to 8 x 8 x 8 (vox0 ... vox511)
Keep this file identical to the notebook - the LungLocAI-3D XGBoost model expects exactly these values in this order.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

PATCH = 32
zz, yy, xx = np.mgrid[:PATCH, :PATCH, :PATCH]
C = (PATCH - 1) / 2
R = np.sqrt((zz - C) ** 2 + (yy - C) ** 2 + (xx - C) ** 2)
REGIONS = {"r3": R <= 3, "r6": R <= 6, "r10": R <= 10, "all": R >= 0}
SHELL = (R > 6) & (R <= 12)

LLA_FEATURE_COLUMNS = (
    [f"{r}_{s}" for r in REGIONS for s in ("mean", "std", "p10", "p50", "p90", "solid")]
    + ["contrast_center_vs_shell", "grad_r6_mean", "grad_all_mean"]
    + [f"vox{i}" for i in range(512)]
)


def extract_patch(image: sitk.Image, center_mm) -> np.ndarray:
    ref = sitk.Image([PATCH] * 3, sitk.sitkFloat32)
    ref.SetSpacing((1.0, 1.0, 1.0))
    ref.SetDirection(image.GetDirection())
    d = np.array(image.GetDirection()).reshape(3, 3)
    ref.SetOrigin(tuple(np.array(center_mm) - d @ np.full(3, C)))
    out = sitk.Resample(image, ref, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)  # (z, y, x), HU


def patch_features(p: np.ndarray) -> dict:
    p = np.clip(p, -1000, 400)
    f = {}
    for name, m in REGIONS.items():
        v = p[m]
        f[f"{name}_mean"], f[f"{name}_std"] = v.mean(), v.std()
        f[f"{name}_p10"], f[f"{name}_p50"], f[f"{name}_p90"] = np.percentile(v, [10, 50, 90])
        f[f"{name}_solid"] = (v > -400).mean()
    f["contrast_center_vs_shell"] = f["r6_mean"] - p[SHELL].mean()
    g = np.sqrt(sum(a ** 2 for a in np.gradient(p)))
    f["grad_r6_mean"], f["grad_all_mean"] = g[R <= 6].mean(), g.mean()
    small = p.reshape(8, 4, 8, 4, 8, 4).mean(axis=(1, 3, 5))
    for i, v in enumerate(small.ravel()):
        f[f"vox{i}"] = v
    return f


def lla_features(image: sitk.Image, center_mm) -> dict:
    f = patch_features(extract_patch(image, center_mm))
    return {k: float(v) for k, v in f.items()}


assert len(LLA_FEATURE_COLUMNS) == 539
