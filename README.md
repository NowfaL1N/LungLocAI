# LungLocAI: CT lung nodule detection and lung-cancer risk

**Upload a CT scan → the lungs are extracted (even from full-body CT) → nodules are found in 3-D →
each nodule becomes a row of measurements → XGBoost estimates whether it is a lung-cancer nodule →
a report a doctor can act on.**

🌐 **Live demo:** _LIVE_DEMO_URL_ &nbsp;·&nbsp; 🧪 **Test scans:** [Releases](../../releases) &nbsp;·&nbsp; 📓 **Train on LUNA16:** [`notebooks/`](notebooks)

> ⚠️ Research prototype for decision support. **Not a medical device.** A radiologist must confirm every finding.
> Do not upload scans that contain patient names or IDs.

<p align="center"><img src="docs/results.png" alt="LungLocAI results page" width="900"></p>

## What it does

| Step | What happens | Code |
|---|---|---|
| 1. 3-D volume | CT resampled to 1 mm isotropic voxels in Hounsfield units | `app/pipeline/ct_io.py` |
| 2. Lung extraction | Finds the body and internal air, picks the two lungs (rejects bowel gas and outside air), crops them out of a full-body scan, labels left/right | `app/pipeline/lungs.py` |
| 3. Nodule detection | Multi-scale 3-D ball opening (1, 2, 3, 5 mm) separates round nodules from vessels; shape and pleura filters | `app/pipeline/nodules.py` |
| 4. Feature table | 33 numbers per nodule: size, 3-D shape, density, margin, location → CSV | `app/pipeline/features.py` |
| 5. XGBoost | `nodule_model` (real nodule vs. false positive) + `malignancy_model` (lung cancer vs. benign) | `app/pipeline/classifier.py` |
| 6. Report | AI cancer probability, Lung-RADS-v2022-style size category, follow-up suggestion, triage priority, 3-plane close-ups, slice viewer, 3-D view, **PDF** | `app/report.py` |

<p align="center"><img src="docs/nodule.png" alt="Nodule card" width="900"></p>

**Accepted uploads:** DICOM series (`.zip` of `.dcm` files, the hospital format), NIfTI (`.nii` / `.nii.gz`),
LUNA16 / MetaImage (`.mhd` **and** `.raw` together), NRRD. Chest CT and full-body CT both work.
A photo or screenshot of a CT (JPG/PNG) can't be analysed: it has no 3-D data or density values.

## Run it on your computer

**Windows:** double-click `run.bat` (installs everything on first run, then opens http://localhost:8000).
**macOS / Linux:** `./run.sh`

```bash
# or by hand
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

**Docker:**
```bash
docker build -t lunglocai .
docker run -p 7860:7860 lunglocai     # http://localhost:7860
```

## Put it online for everyone (free, Hugging Face Spaces)

The site needs a Python server, so GitHub Pages can't host it. Hugging Face Spaces runs it for free
(2 vCPU, 16 GB RAM) and this repo deploys there automatically:

1. Create a free account at https://huggingface.co and click **New Space** → name `LungLocAI` →
   SDK **Docker** → *Blank* → **Create Space**.
2. Create an access token at https://huggingface.co/settings/tokens with **Write** permission.
3. In this GitHub repo: **Settings → Secrets and variables → Actions**
   - **Secrets** tab → *New repository secret*: `HF_TOKEN` = your token
   - **Variables** tab → *New repository variable*: `HF_SPACE` = `your-hf-username/LungLocAI`
4. **Actions** tab → *Deploy to Hugging Face Spaces* → **Run workflow**.

The site goes live at `https://huggingface.co/spaces/your-hf-username/LungLocAI` after the first build
(about 5 minutes), and every push to `main` updates it. Uploaded scans and results are deleted
automatically after 2 hours (`JOB_TTL_HOURS`), and uploads are capped at 1.5 GB (`MAX_UPLOAD_MB`).

The `Dockerfile` also works on Render, Railway, Google Cloud Run or any Docker host. Give it at least 4 GB RAM,
because full-body scans need it.

## Models: from demo to your LungLocAI dataset

The repo ships **DEMO models trained on synthetic phantoms**. The site and PDF say so in orange.
Train real ones on LUNA16 with the **same pipeline** so training and website features match exactly:

```bash
python training/build_luna16_table.py --luna-dir /path/LUNA16 --annotations /path/LUNA16/annotations.csv \
       --out data/lunglocai_table.csv --workers 2
python training/train_xgboost.py --table data/lunglocai_table.csv --task nodule
```
In Colab use [`notebooks/LungLocAI_train_on_LUNA16.ipynb`](notebooks/LungLocAI_train_on_LUNA16.ipynb) (resumable).
Put the resulting `models/*_model.json` + `.meta.json` in `models/` and commit. The live site then updates by itself.

**LUNA16 has no cancer labels.** Its annotations mark nodule vs. non-nodule. For the lung-cancer model,
take malignancy ratings from LIDC-IDRI:
```bash
pip install pylidc
python training/make_lidc_malignancy_csv.py --luna-dir /path/LUNA16 --out data/lidc_malignancy.csv
python training/build_luna16_table.py ... --malignancy data/lidc_malignancy.csv
python training/train_xgboost.py --table data/lunglocai_table.csv --task malignancy
```
Validation is grouped by scan (GroupKFold on `seriesuid`). AUC, sensitivity, specificity, the threshold
and the top features are stored in `.meta.json` and shown on the site and in the PDF.

## Measuring accuracy

`tools/evaluate.py` scores the whole pipeline (detection + XGBoost) against an answer key.
It works on the synthetic test set and on real LUNA16 scans:

```bash
python tools/evaluate.py --scans test_scans --annotations test_scans/annotations.csv
python tools/evaluate.py --scans /path/LUNA16/subset9 --annotations /path/LUNA16/annotations.csv
```

Results on the 7 synthetic test scans in [Releases](../../releases) (30 nodules, never used in training):

| Detection | | Lung-cancer classification | |
|---|---|---|---|
| Nodules found | 29 / 30 (96.7 %) | Cancers flagged | 10 / 11 (90.9 %) |
| False positives / scan | 0.71 | Benign wrongly flagged | 0 / 19 |
| Size error | 0.8 mm | AUC | 0.99 |

These numbers show the pipeline works. They are **not** clinical accuracy: run the same script on held-out
LUNA16 scans for that.

Make your own synthetic test scans: `python tools/make_synthetic_ct.py --out data/synthetic --n 5 [--full-body]`.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/analyze` | multipart `files` → `{job_id}` |
| GET | `/api/jobs/{id}` | status / result JSON |
| GET | `/api/jobs/{id}/report.pdf` | doctor report |
| GET | `/api/jobs/{id}/table.csv` | feature table + predictions |
| GET | `/api/jobs/{id}/slice?axis=axial&index=120` | PNG slice with nodule outlines |
| GET | `/api/jobs/{id}/scene` | 3-D meshes (lungs + nodules) |
| GET | `/api/model` | loaded models + metrics |

## Project layout

```
app/            FastAPI server, CT pipeline, report, web UI (static/)
models/         XGBoost models (*.json) + metadata (*.meta.json)
training/       build LUNA16 table, train XGBoost, LIDC malignancy export
tools/          synthetic CT generator, accuracy evaluation
notebooks/      Colab notebook for training on LUNA16
Dockerfile      container for Hugging Face Spaces / any Docker host
```

## Known limitations

* A nodule fused with the mediastinum, heart or diaphragm can fall outside the lung mask and be missed.
* Faint ground-glass nodules (below −600 HU) and the faint rim of part-solid nodules can be under-segmented.
* Spiculations thinner than ~2 mm are removed by the morphological opening, so diameter can be under-estimated.
* The Lung-RADS category is an automatic approximation, not a radiologist's reading.
* The demo models are trained on synthetic data. Replace them before drawing any conclusions.

## License

MIT, see [LICENSE](LICENSE). Research use only; not for clinical diagnosis.
