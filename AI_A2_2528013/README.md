# Musanze Harvest Intelligence Service

**Student full name:** Mahgoub Adil Ahmed Alhassan  
**Registration number:** 25/28013  
**Institutional username:** ug2528013@ines.ac.rw  
**Verification code:** `AI-A2-8013`  
**Repository URL:** [ADD YOUR REAL REPOSITORY URL IF REQUIRED]  
**Final commit hash:** [ADD THE REAL FINAL COMMIT HASH IF USING GIT]

## Purpose
This individual assignment demonstrates a reproducible synthetic-data pipeline for crop-yield regression, high-risk classification, farm clustering, and a validated FastAPI inference service. **All dataset records are synthetic**, generated from the assignment's specified formula; the results are not real-world agricultural validation.

## Environment
- Python 3.11 or newer recommended; the preparation environment used Python 3.13.
- Windows PowerShell commands are shown below; the project uses `pathlib` and should also work on Linux/macOS.
- Main libraries: NumPy, pandas, scikit-learn, Matplotlib, joblib, FastAPI, Uvicorn, pytest, ReportLab.

## Setup (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce all data, models, reports, and plots
```powershell
python run_all.py --student-id 25/28013 --seed 8013
```
The seed must equal the final four digits of the registration number. The pipeline regenerates the CSV, fingerprint, cleaned data, validation report, model metrics, model files, test predictions, plots, technical report, declaration draft, and test-log PDF.

To check the fingerprint across repeated runs, save the SHA256 line from `artifacts/data_fingerprint.txt`, rerun the command, and compare it. The dataset should be deterministic for the same seed and package versions.

## Prediction CLI
```powershell
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
```
The valid file returns a JSON prediction. The invalid example should fail with a useful message and nonzero exit code because rainfall is outside the allowed range.

## Automated tests
```powershell
python -m pytest -q
```
The tests exercise health, model info, valid prediction, missing fields, wrong types, out-of-range values, repeatability, extra fields, and synthetic-data disclosure.

## Run the API
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs` for Swagger documentation.

### Endpoints
- `GET /health`: service health and verification code.
- `GET /model-info`: tasks, selected model information, and synthetic-data status.
- `POST /predict`: validates a record and returns estimated yield, high-risk label, high-risk probability, cluster ID, and verification code.

Example valid JSON:
```json
{
  "rainfall_mm": 1000,
  "avg_temp_c": 20,
  "soil_moisture_pct": 50,
  "fertilizer_kg": 100,
  "field_size_ha": 2,
  "season": "A",
  "cooperative_support": 1
}
```

## Important artifacts
- `artifacts/regression_metrics.json` — baseline/candidate MAE, RMSE, R² and selected model.
- `artifacts/classification_metrics.json` — baseline/candidate classification metrics, confusion matrices and ROC-AUC where valid.
- `artifacts/clustering_report.json` — silhouette scores K=2…6, selected K, and profiles.
- `artifacts/test_predictions.csv` — held-out predictions, actual values, probabilities, errors, clusters, and verification code.
- `artifacts/data_fingerprint.txt` — SHA-256 of the generated raw CSV bytes.
- `artifacts/data_validation.json` — schema/range/missing/duplicate/leakage checks.
- `artifacts/plots/` — residuals, confusion matrix, and clustering plot.
- `models/` — saved fitted supervised pipelines and clustering preprocessing/model.
- `docs/AI_A2_2528013_REPORT.pdf` — seven-section technical report.
- `evidence/` — test log, declaration draft, and AI-use disclosure.

## Responsible use and limitations
The dataset is synthetic and based on a fixed formula. Metrics describe only a held-out sample from this generated dataset. The API must not be used to determine lending, insurance, eligibility, or real farm interventions. Model errors are possible; a qualified person must review any consequential decision. Clusters are not farm-quality scores.

## Student tasks still required before submission
1. Replace the full-name and institutional-username placeholders; add repository URL and actual commit hash if required.
2. Write your own interpretations of one regression error, one classification error, and one cluster profile in the report/reflection, and ensure you can explain the code.
3. Review and complete `evidence/AI_USE.md` honestly.
4. Check the official institutional declaration wording and sign only after reviewing the completed work.
5. Record the required **continuous, unedited MP4 demonstration of no more than two minutes** in your real environment. It must show your identity card, terminal clock, real run output, metrics/plot interpretation, valid and invalid prediction, API docs and validation error, tests, report, and declaration. A real personal demonstration cannot be fabricated from this project archive.
6. Create the final ZIP named `AI_A2_2528013.zip` with the extracted root folder `AI_A2_2528013` and upload it to the course platform.
