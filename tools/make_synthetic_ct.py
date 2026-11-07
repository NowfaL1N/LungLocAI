"""Generate synthetic CT phantoms in LUNA16 format (.mhd/.raw + annotations.csv) for testing.

NOT medical data. Used only to (1) smoke-test the website end-to-end and (2) train a clearly
labelled DEMO model so the site works before the real LungLocAI model is plugged in.

    python tools/make_synthetic_ct.py --out data/synthetic --n 6
    python tools/make_synthetic_ct.py --out data/synthetic --n 1 --full-body
"""
from __future__ import annotations

import argparse
import csv
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def make_case(rng, full_body=False, sp_xy=1.0, sp_z=2.0):
    nx = ny = 300
    if full_body:
        z_len, chest_top, sp_z = 1300.0, 1050.0, 2.5
    else:
        z_len, chest_top = 340.0, 320.0
    nz = int(z_len / sp_z)
    yy = ((np.arange(ny) - ny / 2) * sp_xy)[:, None]
    xx = ((np.arange(nx) - nx / 2) * sp_xy)[None, :]
    lung_bot, lung_top = chest_top - 260, chest_top - 15

    vol = np.full((nz, ny, nx), -1000, np.int16)
    lungs = np.zeros((nz, ny, nx), bool)
    body = np.zeros((nz, ny, nx), bool)
    for k in range(nz):
        z = k * sp_z
        rx = 160 if (lung_bot - 250 < z < chest_top + 20) else 140
        if z > chest_top + 40: rx = 85
        if z < lung_bot - 250: rx = 120
        ry = rx * 0.68
        b = (xx / rx) ** 2 + (yy / ry) ** 2 <= 1
        body[k] = b
        sl = np.where(b, 35, -1000).astype(np.int16)
        if lung_bot - 20 < z < lung_top + 10:
            ring = b & ((xx / (rx - 8)) ** 2 + (yy / (ry - 8)) ** 2 > 1)
            if np.sin(z / 6.0) > 0.3:
                sl[ring] = 450
        sl[(xx ** 2 + (yy - ry + 30) ** 2 <= 14 ** 2) & b] = 500
        lz = (z - lung_bot) / (lung_top - lung_bot)
        if 0 <= lz <= 1:
            shrink = min(np.sqrt(1 - lz) * 1.15, 1.0)
            for side in (-1, 1):
                a_, b_ = 52 * shrink + 1e-3, 80 * shrink + 1e-3
                lm = ((xx - side * 62) / a_) ** 2 + ((yy + 5) / b_) ** 2 <= 1
                lungs[k] |= lm
            sl[lungs[k]] = -860
        hz = (z - lung_bot - 60) / 55
        if abs(hz) <= 1:
            sl[((xx + 10) / 45) ** 2 + ((yy + 25) / 40) ** 2 <= 1 - hz ** 2] = 40
        vol[k] = sl

    def box(c, r):
        iz, iy, ix = int(c[0] / sp_z), int(c[1] / sp_xy + ny / 2), int(c[2] / sp_xy + nx / 2)
        rz, rr = int(np.ceil(r / sp_z)) + 1, int(np.ceil(r / sp_xy)) + 1
        s = (slice(max(iz - rz, 0), min(iz + rz + 1, nz)), slice(max(iy - rr, 0), min(iy + rr + 1, ny)),
             slice(max(ix - rr, 0), min(ix + rr + 1, nx)))
        Z = (np.arange(s[0].start, s[0].stop) * sp_z)[:, None, None] - c[0]
        Y = ((np.arange(s[1].start, s[1].stop) - ny / 2) * sp_xy)[None, :, None] - c[1]
        X = ((np.arange(s[2].start, s[2].stop) - nx / 2) * sp_xy)[None, None, :] - c[2]
        return s, Z, Y, X

    # vessels
    for side in (-1, 1):
        for _ in range(22):
            p = np.array([lung_bot + 140 + rng.normal(0, 15), -5 + rng.normal(0, 8), side * 30.0])
            d = np.array([rng.normal(0, 1), rng.normal(0, 1), side * abs(rng.normal(1.2, 0.3))]); d /= np.linalg.norm(d)
            r = rng.uniform(1.0, 3.2)
            for _step in range(70):
                s, Z, Y, X = box(p, r)
                if any(q.stop <= q.start for q in s):
                    break
                g = (Z ** 2 + Y ** 2 + X ** 2) <= r ** 2
                g &= lungs[s]
                vol[s][g] = 30
                d = d + rng.normal(0, 0.18, 3); d /= np.linalg.norm(d)
                p = p + d * 2.0
                r = max(0.6, r * 0.985)

    if full_body:
        for _ in range(9):
            c = np.array([rng.uniform(lung_bot - 240, lung_bot - 60), rng.normal(0, 30), rng.normal(0, 60)])
            rad = rng.uniform(12, 32)
            s, Z, Y, X = box(c, rad)
            g = ((Z ** 2 + Y ** 2 + X ** 2) <= rad ** 2) & body[s]
            vol[s][g] = -1000

    nods = []
    n_nod = rng.integers(3, 7)
    lz_idx = np.argwhere(lungs[::2, ::4, ::4]) * np.array([2, 4, 4])
    tries = 0
    while len(nods) < n_nod and tries < 800:
        tries += 1
        iz, iy, ix = lz_idx[rng.integers(len(lz_idx))]
        c = np.array([iz * sp_z, (iy - ny / 2) * sp_xy, (ix - nx / 2) * sp_xy])
        kind = rng.choice(["benign_smooth", "calcified", "malignant_spiculated", "malignant_partsolid", "malignant_large"],
                          p=[0.3, 0.12, 0.25, 0.15, 0.18])
        d = {"benign_smooth": rng.uniform(4, 9), "calcified": rng.uniform(4, 8),
             "malignant_spiculated": rng.uniform(9, 22), "malignant_partsolid": rng.uniform(10, 20),
             "malignant_large": rng.uniform(16, 30)}[kind]
        r = d / 2
        if any(np.linalg.norm(c - n["c"]) < (r + n["r"] + 6) for n in nods):
            continue
        s, Z, Y, X = box(c, r * 2.2)
        dist = np.sqrt(Z ** 2 + Y ** 2 + X ** 2)
        local = dist <= r * 2.2
        if (lungs[s] & local).sum() < 0.9 * local.sum():
            continue
        sub = vol[s].astype(np.float32)
        if kind == "malignant_spiculated":
            dirs = rng.normal(0, 1, (14, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            rel = np.stack(np.broadcast_arrays(Z, Y, X), -1)
            shape = dist <= r * (0.85 + 0.15 * np.sin(4 * np.arctan2(Y, X) + 3 * Z / max(r, 1)))
            for dv in dirs:
                proj = rel @ dv
                perp = np.linalg.norm(rel - proj[..., None] * dv, axis=-1)
                shape |= (proj > 0) & (proj < r * 1.9) & (perp < 0.9) & local
            sub[shape] = rng.normal(35, 10) + rng.normal(0, 25, shape.sum())
        elif kind == "malignant_partsolid":
            m = (dist <= r) & lungs[s]
            sub[m] = -520 + rng.normal(0, 60, m.sum())
            sub[dist <= r * 0.5] = 25
        elif kind == "calcified":
            sub[dist <= r] = rng.uniform(250, 600)
        else:
            sub[dist <= r] = rng.normal(30, 8) + rng.normal(0, 18, (dist <= r).sum())
        vol[s] = sub.astype(np.int16)
        nods.append(dict(c=c, r=r, d=d, kind=kind, malignant=int(kind.startswith("malignant"))))

    for k in range(nz):  # noise + slight blur slice by slice (memory friendly)
        f = vol[k].astype(np.float32) + rng.normal(0, 18, (ny, nx)).astype(np.float32)
        f = ndi.gaussian_filter(f, 0.6)
        f[~body[k]] = -1000 + rng.normal(0, 8, (~body[k]).sum())
        vol[k] = np.clip(f, -1024, 3000).astype(np.int16)

    img = sitk.GetImageFromArray(vol)
    origin = (-nx / 2 * sp_xy, -ny / 2 * sp_xy, -z_len + 50)
    img.SetSpacing((sp_xy, sp_xy, sp_z))
    img.SetOrigin(origin)
    ann = []
    for n in nods:
        ann.append(dict(coordX=round(n["c"][2], 2), coordY=round(n["c"][1], 2), coordZ=round(n["c"][0] + origin[2], 2),
                        diameter_mm=round(n["d"], 2), malignant=n["malignant"], kind=n["kind"]))
    return img, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--full-body", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ann_path = out / "annotations.csv"
    new = not ann_path.exists()
    with open(ann_path, "a", newline="") as f:
        w = csv.DictWriter(f, ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "malignant", "kind"])
        if new:
            w.writeheader()
        for i in range(a.n):
            img, ann = make_case(rng, full_body=a.full_body)
            uid = ("synthetic.fullbody." if a.full_body else "synthetic.chest.") + uuid.UUID(int=int(rng.integers(1 << 62))).hex[-12:]
            sitk.WriteImage(img, str(out / f"{uid}.mhd"))
            for r in ann:
                w.writerow(dict(seriesuid=uid, **r))
            print(uid, len(ann), "nodules", img.GetSize())


if __name__ == "__main__":
    main()
