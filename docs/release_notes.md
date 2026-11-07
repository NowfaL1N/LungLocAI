First release of **LungLocAI**: a web app that analyses a CT scan for lung nodules and estimates lung-cancer risk with XGBoost.

### Test CT scans (`LungLocAI_test_scans.zip`)
7 **synthetic** CT scans (computer-generated, not patients) with **30 nodules at known positions** (11 malignant, 19 benign), none used in training:
- `test01`–`test05_chest.nii.gz`: chest CT
- `test06`, `test07_fullbody.nii.gz`: full-body CT (lungs are auto-extracted)
- `other_formats/`: test01 as LUNA16 `.mhd`+`.raw` and as a DICOM `.zip`
- `ANSWER_KEY.csv`: the truth for every nodule; `annotations.csv`: same truth in LUNA16 format

Unzip it into the project folder (so you get `test_scans/`) and run `evaluate_test_scans.bat` (Windows) or
`python tools/evaluate.py --scans test_scans --annotations test_scans/annotations.csv`.

| Detection | | Lung-cancer classification | |
|---|---|---|---|
| Nodules found | 29 / 30 (96.7 %) | Cancers flagged | 10 / 11 |
| False positives / scan | 0.71 | Benign wrongly flagged | 0 / 19 |
| Size error | 0.8 mm | AUC | 0.99 |

These numbers show the pipeline works. They are not clinical accuracy. The bundled XGBoost models are **demo models** trained on synthetic data; train on LUNA16 with the included scripts before drawing conclusions.

⚠️ Research prototype, not a medical device.
