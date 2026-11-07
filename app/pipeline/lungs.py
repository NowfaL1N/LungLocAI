"""Lung extraction.

Works for chest CT *and* full-body / extended-coverage CT:
  1. a coarse (2.5 mm) pass over the whole scan finds the body, the air inside it, and picks the
     two lungs out of all internal air pockets (rejecting bowel / stomach gas and outside air);
  2. the scan is cropped to the lung bounding box and resampled to 1 mm isotropic voxels
     (the 3-D volume used for everything downstream);
  3. a fine lung mask is rebuilt on the crop and "closed" so that nodules touching the pleura and
     vessels are inside the mask (otherwise juxtapleural nodules would be cut away).

Coordinates follow ITK / LUNA16 convention (LPS): +x = patient left, +y = posterior, +z = superior.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .ct_io import resample

AIR_HU = -400          # anything below this inside the body is air / lung
MIN_LUNG_ML = 150      # a single lung is never smaller than this


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _fill_holes_2d(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        if mask[z].any():
            out[z] = ndi.binary_fill_holes(mask[z])
    return out


def body_mask(arr: np.ndarray) -> np.ndarray:
    m = arr > -500
    m = ndi.binary_opening(m, iterations=1)
    m = _largest_cc(m)
    return _fill_holes_2d(m)


def _select_lung_components(arr: np.ndarray, body: np.ndarray, voxel_ml: float):
    """Return mask of lung components among internal air pockets."""
    air = (arr < AIR_HU) & body
    air = ndi.binary_opening(air, iterations=1)
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros_like(air), []
    idx = np.arange(1, n + 1)
    vol_ml = ndi.sum(air, lab, idx) * voxel_ml
    mean_hu = ndi.mean(arr, lab, idx)
    comps = []
    for i, v, mh in zip(idx, vol_ml, mean_hu):
        if v < MIN_LUNG_ML:
            continue
        zz = np.where((lab == i).any(axis=(1, 2)))[0]
        comps.append(dict(label=int(i), vol_ml=float(v), mean_hu=float(mh), zmin=int(zz.min()), zmax=int(zz.max())))
    if not comps:
        return np.zeros_like(air), []
    # Lung parenchyma ~ -900..-700 HU; pure bowel gas ~ -1000. Prefer big, lung-dense pockets.
    for c in comps:
        density_bonus = 1.0 if -960 < c["mean_hu"] < -500 else 0.35
        c["score"] = c["vol_ml"] * density_bonus
    comps.sort(key=lambda c: c["score"], reverse=True)
    keep = [comps[0]]
    for c in comps[1:3]:
        # second lung: comparable size, overlapping in z with the first one
        overlap = min(c["zmax"], keep[0]["zmax"]) - max(c["zmin"], keep[0]["zmin"])
        if c["vol_ml"] > 0.2 * keep[0]["vol_ml"] and overlap > 0.3 * (keep[0]["zmax"] - keep[0]["zmin"]):
            keep.append(c)
            break
    mask = np.isin(lab, [c["label"] for c in keep])
    return mask, keep


def extract_lungs(img: sitk.Image, out_spacing: float = 1.0, margin_mm: float = 12.0):
    """Return dict with cropped 1 mm volume, lung mask, and bookkeeping."""
    # ---- coarse pass over the whole scan --------------------------------------------------
    coarse_sp = 2.5
    coarse = resample(img, (coarse_sp,) * 3)
    carr = sitk.GetArrayFromImage(coarse)            # z, y, x
    cbody = body_mask(carr)
    clung, comps = _select_lung_components(carr, cbody, voxel_ml=coarse_sp ** 3 / 1000.0)
    if not clung.any():
        raise ValueError("No lungs were found in this scan (no large air-filled regions inside the body).")

    z, y, x = np.where(clung)
    m = int(np.ceil(margin_mm / coarse_sp))
    lo = np.maximum([z.min() - m, y.min() - m, x.min() - m], 0)
    hi = np.minimum([z.max() + m, y.max() + m, x.max() + m], np.array(carr.shape) - 1)

    scan_z_mm = carr.shape[0] * coarse_sp
    lung_z_mm = (z.max() - z.min() + 1) * coarse_sp
    full_body = scan_z_mm > 1.6 * lung_z_mm + 60  # scan reaches well past the thorax

    # coronal overview (min-intensity projection shows lungs nicely) for the report
    yc = int(np.median(y))                                       # coronal slice through the lungs
    overview = {
        "coronal_slice": carr[:, yc, :].astype(np.int16),
        "coronal_lung": clung[:, yc, :],
        "bbox_zx": [int(lo[0]), int(hi[0]), int(lo[2]), int(hi[2])],
        "spacing": coarse_sp,
    }

    # ---- crop + resample to 1 mm isotropic -------------------------------------------------
    origin = coarse.TransformIndexToPhysicalPoint((int(lo[2]), int(lo[1]), int(lo[0])))
    size_mm = (hi - lo + 1)[::-1] * coarse_sp         # x, y, z
    size = [int(np.ceil(s / out_spacing)) for s in size_mm]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing((out_spacing,) * 3)
    f.SetSize(size)
    f.SetOutputOrigin(origin)
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkLinear)
    f.SetDefaultPixelValue(-1024)
    crop = f.Execute(img)
    arr = sitk.GetArrayFromImage(crop).astype(np.int16)

    # ---- fine lung mask ------------------------------------------------------------------
    fbody = body_mask(arr)
    raw, _ = _select_lung_components(arr, fbody, voxel_ml=out_spacing ** 3 / 1000.0)
    if not raw.any():  # fall back to the coarse mask
        cm = sitk.GetImageFromArray(clung.astype(np.uint8))
        cm.CopyInformation(coarse)
        cm = sitk.Resample(cm, crop, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        raw = sitk.GetArrayFromImage(cm).astype(bool)

    # close each axial slice so pleural nodules / vessels are inside the lung field
    r = int(round(8 / out_spacing))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    lung = np.zeros_like(raw)
    for k in range(raw.shape[0]):
        if raw[k].any():
            s = ndi.binary_closing(np.pad(raw[k], r), structure=disk)[r:-r, r:-r]
            lung[k] = ndi.binary_fill_holes(s)
    lung &= fbody

    # left / right labelling (physical +x = patient left in LPS)
    lab, n = ndi.label(lung)
    sizes = ndi.sum(lung, lab, range(1, n + 1)) if n else []
    side_map = np.zeros(lung.shape, np.uint8)  # 1 = right lung, 2 = left lung
    big = [i + 1 for i, s in enumerate(sizes) if s * out_spacing ** 3 / 1000 > MIN_LUNG_ML * 0.5]
    dir_x = img.GetDirection()[0]
    if len(big) >= 2:
        cents = ndi.center_of_mass(lung, lab, big)
        order = np.argsort([c[2] * np.sign(dir_x or 1) for c in cents])
        side_map[lab == big[order[0]]] = 1
        side_map[lab == big[order[-1]]] = 2
    else:  # lungs touch (or only one) -> split at body midline
        xs = np.where(fbody.any(axis=(0, 1)))[0]
        mid = (xs.min() + xs.max()) // 2 if len(xs) else arr.shape[2] // 2
        left_is_high_x = (dir_x or 1) > 0
        side_map[:, :, :mid][lung[:, :, :mid]] = 2 if not left_is_high_x else 1
        side_map[:, :, mid:][lung[:, :, mid:]] = 1 if not left_is_high_x else 2

    lz = np.where(lung.any(axis=(1, 2)))[0]
    return {
        "image": crop,
        "array": arr,
        "lung": lung,
        "side_map": side_map,
        "body": fbody,
        "full_body": bool(full_body),
        "scan_length_mm": round(float(scan_z_mm), 1),
        "lung_length_mm": round(float(lung_z_mm), 1),
        "lung_volume_ml": round(float(lung.sum() * out_spacing ** 3 / 1000.0), 1),
        "lung_z_range": [int(lz.min()), int(lz.max())] if len(lz) else [0, arr.shape[0] - 1],
        "components": comps,
        "overview": overview,
    }
