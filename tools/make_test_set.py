"""Rebuild the official LungLocAI synthetic test set (identical every time: fixed random seeds).

    python tools/make_test_set.py --out test_scans

Creates 7 scans (5 chest, 2 full-body) with 30 nodules at known positions, an ANSWER_KEY.csv,
annotations.csv (LUNA16 format) and test01 in two extra formats (MetaImage .mhd/.raw, DICOM .zip).
The GitHub release workflow runs this and attaches the zip to the release.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]

README = """LungLocAI TEST CT SCANS  (synthetic - computer-generated, not real patients)
===========================================================================

7 CT scans with 30 nodules at KNOWN positions (11 malignant, 19 benign). None of them were used
to train the model, so they are a fair test of the pipeline.

  test01_chest.nii.gz ... test05_chest.nii.gz    chest CT
  test06_fullbody.nii.gz, test07_fullbody.nii.gz full-body CT (1.3 m) - lungs must be auto-extracted
  other_formats/test01_chest.mhd + .raw          same scan as test01, LUNA16 format (select BOTH files)
  other_formats/test01_chest_DICOM.zip           same scan as test01, DICOM series (hospital format)

  ANSWER_KEY.csv     the truth: every nodule's side, zone, size and whether it is malignant
  annotations.csv    same truth in LUNA16 format (used by the scoring script)

HOW TO TEST
  1. Start the website and upload one .nii.gz file (just that one file)
  2. Compare the nodules the site reports with that scan's rows in ANSWER_KEY.csv
     (match by side + zone + size; "Position (mm)" on each card = world_x, world_y, world_z)

SCORE ALL 7 AUTOMATICALLY
  Put this folder inside the project as  test_scans/  and run
      python tools/evaluate.py --scans test_scans --annotations test_scans/annotations.csv
  (Windows: double-click evaluate_test_scans.bat)
"""


def write_dicom(img, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.20261106.11"
    D = img.GetDirection()
    tags = [("0008|0021", "20261106"), ("0008|0008", "DERIVED\\SECONDARY"), ("0020|000e", uid),
            ("0020|0037", "\\".join(map(str, (D[0], D[3], D[6], D[1], D[4], D[7])))),
            ("0008|103e", f"LungLocAI {name}"), ("0008|0060", "CT"), ("0028|1053", "1"), ("0028|1052", "0"),
            ("0010|0010", "SYNTHETIC^TEST01")]
    for i in range(img.GetDepth()):
        s = img[:, :, i]
        for k, v in tags:
            s.SetMetaData(k, v)
        s.SetMetaData("0020|0032", "\\".join(map(str, img.TransformIndexToPhysicalPoint((0, 0, i)))))
        s.SetMetaData("0020|0013", str(i + 1))
        s.SetMetaData("0018|0050", str(img.GetSpacing()[2]))
        w.SetFileName(str(folder / f"IM{i + 1:04d}.dcm"))
        w.Execute(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_scans")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "other_formats").mkdir(parents=True, exist_ok=True)
    src = Path(tempfile.mkdtemp())
    gen = str(ROOT / "tools" / "make_synthetic_ct.py")
    subprocess.run([sys.executable, gen, "--out", str(src), "--n", "5", "--seed", "501"], check=True)
    subprocess.run([sys.executable, gen, "--out", str(src), "--n", "2", "--full-body", "--seed", "601"], check=True)

    ann = pd.read_csv(src / "annotations.csv")
    uids = list(dict.fromkeys(ann.seriesuid))
    names = {u: f"test{i:02d}_{'fullbody' if 'fullbody' in u else 'chest'}" for i, u in enumerate(uids, 1)}
    ann["seriesuid"] = ann.seriesuid.map(names)
    for u, n in names.items():
        sitk.WriteImage(sitk.ReadImage(str(src / f"{u}.mhd")), str(out / f"{n}.nii.gz"), useCompression=True)
    ann.to_csv(out / "annotations.csv", index=False)

    rows = []
    for n, g in ann.groupby("seriesuid", sort=True):
        bot, top = (-460, -215) if "fullbody" in n else (-230, 15)
        for j, (_, r) in enumerate(g.sort_values("coordZ", ascending=False).iterrows(), 1):
            rel = (r.coordZ - bot) / (top - bot)
            rows.append(dict(
                scan_file=n + ".nii.gz", nodule=j, side="Right" if r.coordX < 0 else "Left",
                zone="Upper" if rel > 2 / 3 else ("Middle" if rel > 1 / 3 else "Lower"),
                diameter_mm=r.diameter_mm, truth="MALIGNANT (cancer)" if r.malignant else "benign",
                appearance=r.kind.replace("malignant_", "").replace("benign_", "")
                .replace("partsolid", "part-solid").replace("_", " "),
                world_x=r.coordX, world_y=r.coordY, world_z=r.coordZ))
    pd.DataFrame(rows).to_csv(out / "ANSWER_KEY.csv", index=False)

    first = [u for u, n in names.items() if n == "test01_chest"][0]
    img = sitk.ReadImage(str(src / f"{first}.mhd"))
    sitk.WriteImage(img, str(out / "other_formats" / "test01_chest.mhd"))
    dcm = src / "test01_chest_DICOM"
    write_dicom(img, dcm, "test01 chest")
    with zipfile.ZipFile(out / "other_formats" / "test01_chest_DICOM.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dcm.iterdir()):
            z.write(f, "test01_chest_DICOM/" + f.name)
    (out / "README.txt").write_text(README)
    shutil.rmtree(src, ignore_errors=True)
    print(f"Test set written to {out}: {len(names)} scans, {len(ann)} nodules "
          f"({int(ann.malignant.sum())} malignant)")


if __name__ == "__main__":
    main()
