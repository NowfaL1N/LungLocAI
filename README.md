# LungLocAI: CT lung nodule detection and lung-cancer risk

**Upload a CT scan → the lungs are extracted (even from full-body CT) → nodules are found in 3-D →
each nodule becomes a row of measurements → XGBoost estimates whether it is a lung-cancer nodule →
a report a doctor can act on.**

🌐 **Live demo: https://lunglocai.gentlerock-8c108a22.eastasia.azurecontainerapps.io** (first load after idle ~30–60 s) &nbsp;·&nbsp; 🧪 **Test scans:** [Releases](../../releases) &nbsp;·&nbsp; 📓 **Train on LUNA16:** [`notebooks/`](notebooks)

> ⚠️ Research prototype for decision support. **Not a medical device.** A radiologist must confirm every finding.
> Do not upload scans that contain patient names or IDs.

<p align="center"><img src="docs/results.png" alt="LungLocAI results page" width="900"></p>

## Dataset: LungLocAI-3D

The project's dataset is published on IEEE DataPort:

> **LungLocAI-3D: Isotropic 3D Lung Nodule Candidate Patches and Engineered Features Derived from LUNA16**
> S. K. Thangavel, Nowfal N, Sree Hari R, Naveen Kumar M, IEEE DataPort, 2026.
> DOI: [10.21227/0x9w-1y93](https://doi.org/10.21227/0x9w-1y93) · License: CC BY 4.0

All 888 LUNA16 scans · 54,837 labelled candidates (1,557 nodules, 53,280 non-nodules) · 32×32×32 mm patches at
1 mm isotropic spacing (HU) · 539 engineered features per candidate · patient-level train/val/test split
(subsets 0–7 / 8 / 9) · XGBoost baseline test AUC 0.9725.
The data (~3 GB) is hosted on IEEE DataPort, not in this repository, because GitHub limits files to 100 MB.

If you use it, please also cite LUNA16 (Setio et al., *Medical Image Analysis* 42, 2017, doi:10.1016/j.media.2017.06.015)
and LIDC-IDRI (Armato et al., *Medical Physics* 38(2), 2011, doi:10.1118/1.3528204; TCIA doi:10.7937/K9/TCIA.2015.LO9QL9SX).

## What it does

| Step | What happens | Code |
|---|---|---|
| 1. 3-D volume | CT resampled to 1 mm isotropic voxels in Hounsfield units | `app/pipeline/ct_io.py` |
| 2. Lung extraction | Finds the body and internal air, picks the two lungs (rejects bowel gas and outside air), crops them out of a full-body scan, labels left/right | `app/pipeline/lungs.py` |
| 3. Nodule detection | Multi-scale 3-D ball opening (1, 2, 3, 5 mm) separates round nodules from vessels; shape and pleura filters | `app/pipeline/nodules.py` |
| 4. Feature table | per candidate: the 539 LungLocAI-3D features (32 mm cube) + 33 size/shape/density/location measures → CSV | `app/pipeline/lla_features.py`, `features.py` |
| 5. XGBoost | `nodule_model` = LungLocAI-3D model (real nodule vs. false positive, AUC 0.9725) + `malignancy_model` (lung cancer vs. benign) | `app/pipeline/classifier.py` |
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

## Put it online for everyone (Azure for Students, free)

The site needs a Python server (GitHub Pages can't run it). It runs on **Azure Container Apps**:
2 vCPU / 4 GB, scales to zero when idle, so normal use stays inside the monthly free grant
(180,000 vCPU-seconds + 360,000 GiB-seconds); Azure for Students adds $100 credit with no credit card.

Every push to `main` builds the Docker image with GitHub Actions and publishes it free at
`ghcr.io/nowfal1n/lunglocai` ([workflow](.github/workflows/azure.yml)).

**One-time setup (about 10 minutes):**
1. Activate **Azure for Students** with your college email: https://azure.microsoft.com/free/students
2. Open **Azure Cloud Shell**: https://shell.azure.com → choose **Bash** (if asked about storage, pick *No storage account required*).
3. Paste and press Enter:
   ```bash
   curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/setup.sh | bash
   ```
   It creates the app and prints the public link, e.g. `https://lunglocai.<random>.eastasia.azurecontainerapps.io`.
   (Azure for Students may block Indian regions; the script then tries the next allowed region automatically.)
4. *(Optional, enables auto-update on every push.)* If the script printed a JSON block, save it as a GitHub
   secret named `AZURE_CREDENTIALS` (Settings → Secrets and variables → Actions → New repository secret).
   Otherwise, update the live site after a push with
   `curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/update.sh | bash`.

To remove everything from Azure: `curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/delete.sh | bash`.

Uploaded scans and results are deleted automatically after 2 hours (`JOB_TTL_HOURS`), uploads are capped at
1.5 GB (`MAX_UPLOAD_MB`), and only one instance runs (`--max-replicas 1`) so each job stays on the
machine that processes it. After a period without visitors the first request takes ~30–60 s while the
container starts.

**Other hosts:** the same `Dockerfile` runs on Google Cloud Run, Oracle Cloud, any VPS
(`docker run -p 7860:7860 ghcr.io/nowfal1n/lunglocai`) or Hugging Face Spaces (Docker Spaces need a
PRO plan; set the `HF_TOKEN` secret and `HF_SPACE` variable to use [this workflow](.github/workflows/deploy-huggingface.yml)).
Give it at least 4 GB RAM, because full-body scans need it.

## Research storage and owner-only admin (optional)

When enabled, the upload page shows an **opt-in** checkbox. Only if the uploader ticks it, an **anonymised** copy of the
findings is saved to a **private** GitHub repository (default `NowfaL1N/LungLocAI-data`): the nodule feature table, AI
results, the PDF report and small nodule close-ups. The CT scan itself is never stored, and file names and DICOM UIDs are
removed. The app refuses to store anything if that repository is public.

The owner logs in at **`/admin`** to browse stored cases, open their reports, and label each nodule as *malignant*,
*benign*, *not a nodule* or *unsure*. **Retrain** then trains XGBoost on the base table plus the confirmed labels. The new
model goes live only if it scores at least as well as the current one on held-out scans, and every model is versioned
in the private repo.

Enable it (one time):
1. Create a **private** repo `LungLocAI-data` (with a README).
2. Create a fine-grained token at https://github.com/settings/personal-access-tokens/new: *Only select repositories* →
   `LungLocAI-data`, permission **Contents: Read and write**.
3. In Azure Cloud Shell: `curl -sL https://raw.githubusercontent.com/NowfaL1N/LungLocAI/main/azure/enable-storage.sh | bash`
   (asks for the token and an admin password; both are stored as Azure secrets).

Locally: `DATA_DIR=./research_data ADMIN_PASSWORD=... uvicorn app.main:app` uses a folder instead of GitHub.

## Models

| Model | What it decides | Features | Trained on | Status |
|---|---|---|---|---|
| `nodule_model` | real nodule vs. false positive | the 539 LungLocAI-3D features | LungLocAI-3D: 888 LUNA16 scans, 54,837 candidates | **test AUC 0.9725** (LUNA16 subset 9) |
| `malignancy_model` | lung cancer vs. benign | 33 shape/density/location measures | synthetic phantoms | **demo**: LungLocAI-3D / LUNA16 have no malignancy labels |

**Nodule model.** This is the published `xgboost_baseline.json`. For every candidate the website computes exactly the
dataset's features (`app/pipeline/lla_features.py`, copied from the dataset notebook): a 32 mm cube resampled at 1 mm from
the original scan, sphere statistics at 3/6/10 mm, centre-vs-shell contrast, gradients and the 8×8×8 downsampled patch.
Predictions use the model's best iteration (599), which reproduces the published test results exactly.
The website uses threshold **0.1** (sensitivity 0.80, specificity 0.980 on the test set) instead of 0.5 (sensitivity 0.58,
specificity 0.997), because a screening tool should miss fewer nodules. Change it with the environment variable
`NODULE_MODEL_THRESHOLD`.

**Cancer model.** It becomes real once you label stored cases as malignant/benign on `/admin` and press **Retrain**.
That trains from scratch on your confirmed labels; it never fine-tunes the demo. Malignancy ratings from LIDC-IDRI are
another source (`training/make_lidc_malignancy_csv.py`).

**Retraining the nodule model.** Without a base table, Retrain *fine-tunes* the LungLocAI-3D model with your confirmed
labels, so it keeps what it learned from 888 scans. To retrain on the dataset itself, make a compact table and upload it
on `/admin`:
```bash
python tools/make_base_table.py --metadata metadata.csv --features features.csv --out base_table.csv   # under 60 MB
```
In both cases a new model only goes live if it is at least as good as the current one on held-out scans.

Rebuilding everything from raw LUNA16 is also possible (`training/build_luna16_table.py`,
`training/train_xgboost.py`, `notebooks/`).

## Measuring accuracy

`tools/evaluate.py` scores the whole pipeline (detection + XGBoost) against an answer key.
It works on the synthetic test set and on real LUNA16 scans:

```bash
python tools/evaluate.py --scans test_scans --annotations test_scans/annotations.csv
python tools/evaluate.py --scans /path/LUNA16/subset9 --annotations /path/LUNA16/annotations.csv
```

Results on the 7 synthetic test scans in [Releases](../../releases) (30 nodules, never used in training):

| | Result |
|---|---|
| Nodules found | 29 / 30 (96.7 %) |
| Candidate-level AUC of the LungLocAI-3D model | 0.992 |
| False positives / scan at threshold 0.1 | 8.1 (1.1 at 0.5) |
| Size error | 0.8 mm |

The synthetic scans have unrealistically uniform vessels that a model trained on real CT has never seen, so the
false-positive count here is pessimistic. On real LUNA16 test data the published figure is about 1.2 false positives
per scan at 0.1. The synthetic set checks that the pipeline works; real accuracy comes from LUNA16 (`tools/evaluate.py`
on subset 9).

Rebuild the exact test set locally with `python tools/make_test_set.py --out test_scans`, or make new synthetic scans with `python tools/make_synthetic_ct.py --out data/synthetic --n 5 [--full-body]`.

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
Dockerfile      container image (Azure Container Apps / any Docker host)
azure/          one-command Azure setup, update, storage and delete scripts
app/storage.py  private research storage (GitHub repo or folder); app/retrain.py: label-based retraining
```

## Known limitations

* A nodule fused with the mediastinum, heart or diaphragm can fall outside the lung mask and be missed.
* Faint ground-glass nodules (below −600 HU) and the faint rim of part-solid nodules can be under-segmented.
* Spiculations thinner than ~2 mm are removed by the morphological opening, so diameter can be under-estimated.
* The Lung-RADS category is an automatic approximation, not a radiologist's reading.
* The demo models are trained on synthetic data. Replace them before drawing any conclusions.

## License

MIT, see [LICENSE](LICENSE). Research use only; not for clinical diagnosis.
