"""3-D nodule  ->  one row of a table.

Every candidate nodule is described by the numeric columns in FEATURE_COLUMNS. The SAME function is
used (a) by the website at inference time and (b) by training/build_luna16_table.py when the
LungLocAI table is built from LUNA16, so the XGBoost model always sees identical features.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import ConvexHull
from skimage import measure

FEATURE_COLUMNS = [
    # size & shape
    "diameter_mm", "volume_mm3", "major_axis_mm", "minor_axis_mm", "elongation", "flatness",
    "axis_ratio", "surface_area_mm2", "sphericity", "compactness", "solidity",
    # density (Hounsfield units)
    "mean_hu", "std_hu", "min_hu", "max_hu", "median_hu", "p10_hu", "p90_hu",
    "skewness_hu", "kurtosis_hu", "entropy_hu",
    "solid_fraction", "ggo_fraction", "calcified_fraction",
    # margin / edge
    "edge_contrast_hu", "edge_gradient", "margin_irregularity",
    # location
    "pleura_dist_mm", "juxtapleural", "rel_z", "rel_x", "rel_y", "is_left",
]

INFO_COLUMNS = ["candidate_id", "world_x", "world_y", "world_z", "voxel_z", "voxel_y", "voxel_x",
                "side", "zone", "nodule_type", "core_solid_fraction"]


def _entropy(v):
    h, _ = np.histogram(v, bins=32, range=(-1000, 400))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def compute_features(cand: dict, arr: np.ndarray, lung: np.ndarray, side_map: np.ndarray,
                     image: sitk.Image, lung_bbox: dict, spacing: float = 1.0) -> dict:
    box = cand["box"]
    m = cand["mask"].astype(bool)
    # pad the box a little for edge measurements
    pad = 3
    lo = [max(s.start - pad, 0) for s in box]
    hi = [min(s.stop + pad, n) for s, n in zip(box, arr.shape)]
    pbox = tuple(slice(a, b) for a, b in zip(lo, hi))
    mm = np.zeros([b - a for a, b in zip(lo, hi)], bool)
    inner = tuple(slice(s.start - a, s.stop - a) for s, a in zip(box, lo))
    mm[inner] = m
    sub = arr[pbox].astype(np.float32)
    vals = sub[mm]

    nvox = int(mm.sum())
    vol = nvox * spacing ** 3
    diam = (6 * vol / np.pi) ** (1 / 3)

    coords = np.argwhere(mm).astype(float) * spacing
    if len(coords) >= 4:
        ev = np.sort(np.linalg.eigvalsh(np.cov((coords - coords.mean(0)).T) + np.eye(3) * 1e-3))[::-1]
    else:
        ev = np.array([1.0, 1.0, 1.0])
    major = 2 * np.sqrt(5 * ev[0]); minor = 2 * np.sqrt(5 * ev[2])

    # surface area by marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(np.pad(mm, 1).astype(np.float32), 0.5,
                                                    spacing=(spacing,) * 3)
        area = float(measure.mesh_surface_area(verts, faces))
    except Exception:
        area = float(np.pi * diam ** 2)
    sphericity = float(min(1.0, np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / max(area, 1e-6)))
    compactness = float(vol / (area ** 1.5 + 1e-6) * 36 ** 0.5 * np.pi ** 0.5)

    try:
        hull = ConvexHull(coords) if len(coords) >= 10 else None
        solidity = float(min(1.0, vol / hull.volume)) if hull and hull.volume > 0 else 1.0
    except Exception:
        solidity = 1.0

    # margin: shell around the nodule inside the lung
    lsub = lung[pbox]
    dil = ndi.binary_dilation(mm, iterations=2)
    shell = dil & ~mm & lsub
    shell_mean = float(sub[shell].mean()) if shell.any() else -850.0
    gmag = np.sqrt(sum(g ** 2 for g in np.gradient(sub, spacing)))
    boundary = mm & ~ndi.binary_erosion(mm)
    edge_grad = float(gmag[boundary].mean()) if boundary.any() else 0.0
    # irregularity: spread of boundary distances from the centroid (spiculated -> large)
    bc = np.argwhere(boundary).astype(float) * spacing
    if len(bc) > 4:
        rr = np.linalg.norm(bc - coords.mean(0), axis=1)
        irregular = float(rr.std() / (rr.mean() + 1e-6))
    else:
        irregular = 0.0

    # location
    cz, cy, cx = cand["centroid"]
    wx, wy, wz = image.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(cz)))
    zi = int(np.clip(round(cz), 0, arr.shape[0] - 1))
    yi = int(np.clip(round(cy), 0, arr.shape[1] - 1))
    xi = int(np.clip(round(cx), 0, arr.shape[2] - 1))
    side_code = int(side_map[zi, yi, xi])
    if side_code == 0:  # candidate sits on the mask edge; use the nearest labelled voxel
        nz = np.argwhere(side_map[pbox] > 0)
        if len(nz):
            near = nz[np.argmin(np.linalg.norm(nz - (np.array([cz, cy, cx]) - lo), axis=1))]
            side_code = int(side_map[pbox][tuple(near)])
    is_left = 1 if side_code == 2 else 0

    zmin_w, zmax_w = lung_bbox["world_z"]
    rel_z = float(np.clip((wz - zmin_w) / max(zmax_w - zmin_w, 1), 0, 1))   # 0 = base, 1 = apex
    rel_x = float(np.clip((cx - lung_bbox["x"][0]) / max(lung_bbox["x"][1] - lung_bbox["x"][0], 1), 0, 1))
    rel_y = float(np.clip((cy - lung_bbox["y"][0]) / max(lung_bbox["y"][1] - lung_bbox["y"][0], 1), 0, 1))
    pleura = float(cand.get("pleura_dist_mm", 0.0))
    juxta = 1 if pleura <= diam / 2 + 3 else 0

    solid_f = float((vals > -300).mean()) if len(vals) else 0.0
    ggo_f = float(((vals > -800) & (vals <= -300)).mean()) if len(vals) else 0.0
    calc_f = float((vals > 200).mean()) if len(vals) else 0.0
    # nodule type from interior voxels (edges are blurred by partial-volume effect)
    core = ndi.binary_erosion(mm)
    tv = sub[core] if core.sum() >= 20 else vals
    t_solid = float((tv > -300).mean()) if len(tv) else 0.0
    solid_d = diam * t_solid ** (1 / 3)
    if calc_f > 0.3 or float((tv > 200).mean()) > 0.5:
        ntype = "calcified"
    elif t_solid >= 0.8:
        ntype = "solid"
    elif t_solid >= 0.05 and solid_d >= 3.0:
        ntype = "part-solid"
    else:
        ntype = "ground-glass"

    row = {
        "diameter_mm": diam, "volume_mm3": vol, "major_axis_mm": float(major), "minor_axis_mm": float(minor),
        "elongation": float(np.sqrt(ev[0] / ev[1])), "flatness": float(np.sqrt(ev[1] / ev[2])),
        "axis_ratio": float(np.sqrt(ev[0] / ev[2])), "surface_area_mm2": area,
        "sphericity": sphericity, "compactness": compactness, "solidity": solidity,
        "mean_hu": float(vals.mean()), "std_hu": float(vals.std()), "min_hu": float(vals.min()),
        "max_hu": float(vals.max()), "median_hu": float(np.median(vals)),
        "p10_hu": float(np.percentile(vals, 10)), "p90_hu": float(np.percentile(vals, 90)),
        "skewness_hu": float(np.nan_to_num(stats.skew(vals))),
        "kurtosis_hu": float(np.nan_to_num(stats.kurtosis(vals))),
        "entropy_hu": _entropy(vals),
        "solid_fraction": solid_f, "ggo_fraction": ggo_f, "calcified_fraction": calc_f,
        "edge_contrast_hu": float(vals.mean() - shell_mean), "edge_gradient": edge_grad,
        "margin_irregularity": irregular,
        "pleura_dist_mm": pleura, "juxtapleural": juxta, "rel_z": rel_z, "rel_x": rel_x, "rel_y": rel_y,
        "is_left": is_left,
        # info (not fed to the model)
        "world_x": float(wx), "world_y": float(wy), "world_z": float(wz),
        "voxel_z": zi, "voxel_y": yi, "voxel_x": xi,
        "side": "Left" if is_left else "Right",
        "zone": "Upper" if rel_z > 2 / 3 else ("Middle" if rel_z > 1 / 3 else "Lower"),
        "nodule_type": ntype,
        "core_solid_fraction": t_solid,
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


def lung_bbox_info(lung: np.ndarray, image: sitk.Image) -> dict:
    z, y, x = np.where(lung)
    p0 = image.TransformIndexToPhysicalPoint((int(x.min()), int(y.min()), int(z.min())))
    p1 = image.TransformIndexToPhysicalPoint((int(x.max()), int(y.max()), int(z.max())))
    return {"x": [int(x.min()), int(x.max())], "y": [int(y.min()), int(y.max())],
            "z": [int(z.min()), int(z.max())], "world_z": sorted([p0[2], p1[2]])}
