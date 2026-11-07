"""Export LIDC-IDRI radiologist malignancy ratings as a CSV the table builder understands.

LUNA16's annotations.csv says WHERE nodules are, not whether they are cancer. LUNA16 was derived
from LIDC-IDRI, where up to 4 radiologists rated each nodule's malignancy 1 (benign) .. 5 (malignant).
This script averages those ratings per nodule and writes:

    seriesuid,coordX,coordY,coordZ,diameter_mm,malignancy,n_readers

Requirements: `pip install pylidc` and a configured pylidc (it reads the LIDC XML + DICOM;
see https://pylidc.github.io/install.html). The LUNA16 .mhd files are used to convert voxel
centroids to world (mm) coordinates, so pass the same --luna-dir you use for the table builder.

    python training/make_lidc_malignancy_csv.py --luna-dir /content/LUNA16 --out data/lidc_malignancy.csv

Then:  python training/build_luna16_table.py ... --malignancy data/lidc_malignancy.csv
Mean rating >= 3.5 -> malignant (1), <= 2.5 -> benign (0), 3 -> excluded (indeterminate).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna-dir", required=True)
    ap.add_argument("--out", default="data/lidc_malignancy.csv")
    ap.add_argument("--min-readers", type=int, default=1)
    a = ap.parse_args()
    import pylidc as pl  # noqa: imported here so the rest of the project does not need it

    mhd = {p.stem: p for p in Path(a.luna_dir).rglob("*.mhd")}
    rows = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "malignancy", "n_readers"])
        for scan in pl.query(pl.Scan):
            uid = scan.series_instance_uid
            if uid not in mhd:
                continue
            img = sitk.ReadImage(str(mhd[uid]))
            # pylidc slices are ordered by increasing z like the LUNA16 volume; verify the extent matches
            if img.GetSize()[2] != len(scan.slice_zvals):
                print(f"skip {uid}: slice count mismatch")
                continue
            for cluster in scan.cluster_annotations():
                if len(cluster) < a.min_readers:
                    continue
                # pylidc centroid = (row, col, slice) with slices sorted by ascending z, same as LUNA16 .mhd
                r, c, k = np.array([ann.centroid for ann in cluster]).mean(0)
                x, y, z = img.TransformContinuousIndexToPhysicalPoint((float(c), float(r), float(k)))
                mal = float(np.mean([ann.malignancy for ann in cluster]))
                diam = float(np.mean([ann.diameter for ann in cluster]))
                w.writerow([uid, round(x, 2), round(y, 2), round(z, 2), round(diam, 2), round(mal, 2), len(cluster)])
                rows += 1
    print(f"wrote {rows} nodules to {a.out}")


if __name__ == "__main__":
    main()
