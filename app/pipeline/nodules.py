"""Nodule candidate detection inside the extracted lung field (1 mm isotropic volume).

Classic, explainable approach (no GPU needed):
  * soft-tissue density voxels (> -600 HU) inside the lungs = vessels + airway walls + nodules
  * multi-scale morphological opening with exact balls (EDT based) of radius 1, 2, 3, 5 mm:
    thin vessels disappear, round structures survive -> nodules attached to vessels get isolated
  * each surviving blob is kept only if it is round enough (not a tube, not a pleural sheet)
    and 3-40 mm in diameter (LUNA16 annotates nodules >= 3 mm)
  * duplicates found at several scales are merged, keeping the roundest version
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

DENSE_HU = -600
MIN_DIAM_MM, MAX_DIAM_MM = 3.0, 40.0
MAX_ELONGATION = 3.0
MAX_CANDIDATES = 60


def _ball_open(mask: np.ndarray, r: float) -> np.ndarray:
    if r <= 0:
        return mask
    eroded = ndi.distance_transform_edt(mask) > r
    if not eroded.any():
        return eroded
    return ndi.distance_transform_edt(~eroded) <= r


def _shape(coords: np.ndarray):
    c = coords - coords.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T) + np.eye(3) * 1e-3))[::-1]
    elong = float(np.sqrt(ev[0] / ev[1]))
    flat = float(np.sqrt(ev[1] / ev[2]))
    return elong, flat, float(np.sqrt(ev[0] / ev[2]))


def detect_candidates(arr: np.ndarray, lung: np.ndarray, spacing: float = 1.0):
    """Return list of candidates: dict(mask_slices, mask, centroid, scale, ...)."""
    zs, ys, xs = np.where(lung)
    if len(zs) == 0:
        return []
    # work in the lung bounding box only (speed)
    bb = tuple(slice(max(a.min() - 2, 0), a.max() + 3) for a in (zs, ys, xs))
    sub = arr[bb]
    lsub = lung[bb]
    dense = (sub > DENSE_HU) & lsub
    dense = ndi.binary_fill_holes(dense)
    lung_edt = ndi.distance_transform_edt(lsub)

    found = []
    for r in (1.0, 2.0, 3.0, 5.0):
        op = _ball_open(dense, r / spacing)
        lab, n = ndi.label(op)
        if n == 0:
            continue
        objs = ndi.find_objects(lab)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            m = lab[sl] == i
            nvox = int(m.sum())
            vol = nvox * spacing ** 3
            diam = (6 * vol / np.pi) ** (1 / 3)
            if diam < MIN_DIAM_MM or diam > MAX_DIAM_MM:
                continue
            coords = np.argwhere(m).astype(float)
            if len(coords) < 8:
                continue
            elong, flat, ratio = _shape(coords)
            if elong > MAX_ELONGATION or ratio > 4.5:
                continue
            # pleural sheets / chest wall: candidate must be mostly inside the lung field
            cen = coords.mean(0) + np.array([s.start for s in sl])
            ci = tuple(np.clip(np.round(cen).astype(int), 0, np.array(lsub.shape) - 1))
            ext = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
            m_ext = lab[ext] == i
            shell = ndi.binary_dilation(m_ext, iterations=2) & ~m_ext
            in_lung_frac = float((shell & lsub[ext]).sum() / max(shell.sum(), 1))
            if in_lung_frac < 0.35:
                continue
            # roundness score used for deduplication
            score = (1.0 / elong) * (1.0 / flat) ** 0.5 * min(1.0, diam / 4.0)
            full = np.zeros(lsub.shape, bool)
            full[sl] = m
            found.append(dict(scale=r, centroid_sub=cen, diam=diam, score=score, mask_sub=full,
                              box_sub=sl, in_lung_frac=in_lung_frac,
                              pleura_dist=float(lung_edt[ci])))

    # merge duplicates across scales (keep the roundest; ties -> larger scale = cleaner shape)
    found.sort(key=lambda c: (c["score"], c["scale"]), reverse=True)
    kept = []
    for c in found:
        dup = False
        for k in kept:
            d = np.linalg.norm(c["centroid_sub"] - k["centroid_sub"])
            if d < max(c["diam"], k["diam"]) / 2.0 + 1.0:
                dup = True
                break
        if not dup:
            kept.append(c)
        if len(kept) >= MAX_CANDIDATES:
            break

    off = np.array([s.start for s in bb])
    out = []
    for c in kept:
        sl = c["box_sub"]
        gsl = tuple(slice(s.start + o, s.stop + o) for s, o in zip(sl, off))
        out.append(dict(
            box=gsl,                                   # slices into the cropped 1 mm volume
            mask=c["mask_sub"][sl],                    # boolean mask inside box
            centroid=(c["centroid_sub"] + off).tolist(),  # z, y, x voxel coords
            scale_mm=c["scale"],
            in_lung_frac=c["in_lung_frac"],
            pleura_dist_mm=c["pleura_dist"] * spacing,
        ))
    return out


def region_grow_at(arr: np.ndarray, lung: np.ndarray, center_zyx, radius_mm: float, spacing=1.0):
    """Segment a nodule around a known centre (used by the LUNA16 training-table builder
    so that every annotated nodule gets a row even if the detector missed it)."""
    c = np.round(center_zyx).astype(int)
    r = int(np.ceil(radius_mm * 1.6 / spacing)) + 3
    lo = np.maximum(c - r, 0)
    hi = np.minimum(c + r + 1, arr.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = arr[box]
    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1)
    dist = np.linalg.norm((grid - c) * spacing, axis=-1)
    m = (sub > DENSE_HU) & (dist <= radius_mm * 1.3)
    lab, n = ndi.label(m)
    if n == 0:
        m = dist <= max(radius_mm, 1.5)
    else:
        ci = tuple(np.clip(c - lo, 0, np.array(sub.shape) - 1))
        target = lab[ci] if lab[ci] else int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))) + 1
        m = lab == target
    coords = np.argwhere(m)
    cen = coords.mean(0) + lo if len(coords) else c.astype(float)
    ledt = ndi.distance_transform_edt(lung[box])
    ci = tuple(np.clip(np.round(cen - lo).astype(int), 0, np.array(sub.shape) - 1))
    return dict(box=box, mask=m, centroid=cen.tolist(), scale_mm=0.0, in_lung_frac=1.0,
                pleura_dist_mm=float(ledt[ci]) * spacing)
