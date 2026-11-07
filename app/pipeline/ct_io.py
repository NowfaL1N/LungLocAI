"""Load a CT scan from whatever the user uploaded and return a 3-D volume in Hounsfield units.

Supported inputs
  * LUNA16 / MetaImage   : .mhd + .raw   (upload both files, or a .zip containing them)
  * DICOM series         : .zip of .dcm files (or a folder of files without extension)
  * NIfTI                : .nii / .nii.gz
  * NRRD                 : .nrrd
  * NumPy                : .npy (assumed HU, 1 mm isotropic, z-y-x order)

Everything is returned as a SimpleITK image so spacing / origin / direction are preserved
(needed to report nodule positions in millimetres / world coordinates like LUNA16 does).
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

VOLUME_EXT = (".mhd", ".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".npy")


def _extract_zips(folder: Path) -> None:
    for z in list(folder.rglob("*.zip")):
        target = z.with_suffix("")
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                # protect against path traversal ("zip slip")
                dest = (target / m.filename).resolve()
                if not str(dest).startswith(str(target.resolve())):
                    continue
                zf.extract(m, target)
        z.unlink()


def _find_dicom_series(folder: Path):
    reader = sitk.ImageSeriesReader()
    best, best_n = None, 0
    dirs = {p.parent for p in folder.rglob("*") if p.is_file()}
    for d in dirs:
        try:
            for sid in reader.GetGDCMSeriesIDs(str(d)) or []:
                files = reader.GetGDCMSeriesFileNames(str(d), sid)
                if len(files) > best_n:
                    best, best_n = files, len(files)
        except Exception:
            continue
    return best


def load_ct(upload_dir: str | Path) -> tuple[sitk.Image, dict]:
    """Return (image_in_HU, info) from the files in upload_dir."""
    folder = Path(upload_dir)
    _extract_zips(folder)

    files = [p for p in folder.rglob("*") if p.is_file()]
    vols = [p for p in files if p.name.lower().endswith(VOLUME_EXT)]
    info: dict = {}

    if vols:
        # prefer .mhd (LUNA16 native) then nifti etc.
        vols.sort(key=lambda p: [e for e in VOLUME_EXT if p.name.lower().endswith(e)][0] != ".mhd")
        path = vols[0]
        if path.name.lower().endswith(".mhd"):
            ref = [l.split("=", 1)[1].strip() for l in path.read_text(errors="ignore").splitlines()
                   if l.strip().startswith("ElementDataFile")]
            if ref and ref[0] != "LOCAL" and not (path.parent / ref[0]).exists():
                raise ValueError(f"The .mhd header points to '{ref[0]}', which was not uploaded. "
                                 "Select both the .mhd and the .raw file (or upload them together in a .zip).")
        if path.name.lower().endswith(".npy"):
            arr = np.load(path).astype(np.int16)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((1.0, 1.0, 1.0))
        else:
            img = sitk.ReadImage(str(path))
        info["format"] = path.suffix.lstrip(".").upper() if not path.name.endswith(".nii.gz") else "NIFTI"
        info["source_file"] = path.name
        info["series_uid"] = path.name.replace(".mhd", "").replace(".nii.gz", "").replace(".nii", "")
    else:
        series = _find_dicom_series(folder)
        pics = [p for p in files if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")]
        if not series and pics:
            raise ValueError(
                f"'{pics[0].name}' is a picture of a CT, not a CT scan. The analysis needs the full 3-D scan "
                "(hundreds of slices with Hounsfield-unit values and mm spacing). Upload a LUNA16 .mhd together "
                "with its .raw file, a .zip of the DICOM (.dcm) series, or a NIfTI (.nii/.nii.gz) file."
            )
        if not series:
            raise ValueError(
                "Could not find a CT volume. Upload a LUNA16 .mhd + .raw pair, a NIfTI file, "
                "or a .zip of a DICOM series."
            )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        img = reader.Execute()
        info["format"] = "DICOM"
        info["source_file"] = f"{len(series)} DICOM slices"
        try:
            info["series_uid"] = reader.GetMetaData(0, "0020|000e").strip()
        except Exception:
            info["series_uid"] = "dicom-series"

    if img.GetDimension() == 4:  # some nifti exports carry a singleton 4th dim
        img = img[:, :, :, 0]
    if img.GetDimension() != 3:
        raise ValueError("Uploaded image is not a 3-D CT volume.")

    img = sitk.Cast(img, sitk.sitkInt16)
    arr = sitk.GetArrayViewFromImage(img)
    # Some exports store raw values (0..4095) instead of HU. Shift if clearly not HU.
    if arr.min() >= 0 and np.percentile(arr, 5) > 500:
        img = sitk.Cast(img - 1024, sitk.sitkInt16)
        info["hu_shifted"] = True

    size = img.GetSize()
    sp = img.GetSpacing()
    info.update(
        size_voxels=list(size),
        spacing_mm=[round(s, 3) for s in sp],
        extent_mm=[round(a * b, 1) for a, b in zip(size, sp)],
        origin=list(img.GetOrigin()),
    )
    return img, info


def resample(img: sitk.Image, spacing=(1.0, 1.0, 1.0), interp=sitk.sitkLinear, default=-1024) -> sitk.Image:
    """Resample to isotropic spacing -> this is the '3-D conversion' step (true-scale voxel cube)."""
    old_sp = np.array(img.GetSpacing())
    old_sz = np.array(img.GetSize())
    new_sp = np.array(spacing, dtype=float)
    new_sz = np.maximum(1, np.round(old_sz * old_sp / new_sp)).astype(int)
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(new_sp))
    f.SetSize([int(s) for s in new_sz])
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(interp)
    f.SetDefaultPixelValue(default)
    return f.Execute(img)
