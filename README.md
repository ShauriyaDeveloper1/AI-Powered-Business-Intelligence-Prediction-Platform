# AI-Powered Business Intelligence & Predictive Analytics Platform

This workspace contains a full-stack platform built with:
- **Backend:** Python + FastAPI + scikit-learn + MySQL
- **Frontend:** HTML + CSS + JavaScript + Chart.js

## Features Implemented

- Admin login with secure password hashing and token-based authentication
- CSV upload with validation and automated ML pipeline execution
- Data preprocessing: missing value handling, categorical encoding, feature scaling, train/test split
- Model training + evaluation:
  - Logistic Regression (baseline classifier)
  - Random Forest Classifier (final classifier)
  - Linear Regression (baseline regressor)
  - Random Forest Regressor (final regressor)
- Metrics: Accuracy, Precision, Recall, F1-score, MSE, model comparison
- Prediction API with real-time frontend predictions
- Prediction history stored in MySQL
- K-Means clustering for customer segmentation
- Elbow-based cluster count selection
- Explainable AI:
  - Feature importance extraction
  - Prediction probability score (classification)
  - Auto-generated insight text
- Dashboard visualizations:
  - KPI cards
  - Model performance chart
  - Churn visualization
  - Cluster distribution + scatter
  - Feature importance chart

## Project Structure

- `backend/main.py` - FastAPI app and routes
- `backend/database.py` - MySQL schema/init and persistence functions
- `backend/ml.py` - preprocessing, training, clustering, explainability, prediction
- `backend/schemas.py` - request payload models
- `frontend/index.html` - dashboard UI
- `frontend/script.js` - dashboard client logic
- `frontend/style.css` - styling
- `start-platform.ps1` - one-command launcher

## 1) Prerequisites

- Python 3.10+
- MySQL 8+
- PowerShell (Windows)

## 2) Setup

From workspace root (`E:\AI Business`):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Create environment variables (PowerShell example):

```powershell
$env:AI_DB_HOST="localhost"
$env:AI_DB_PORT="3306"
$env:AI_DB_USER="root"
$env:AI_DB_PASSWORD="change-me"
$env:AI_DB_NAME="ai_platform"
$env:AI_ADMIN_USERNAME="admin"
$env:AI_ADMIN_PASSWORD="admin123"
$env:AI_TOKEN_TTL_HOURS="12"
```

## 3) Run Backend + Frontend

### Option A: One command launcher

```powershell
.\start-platform.ps1
```

This starts FastAPI on `http://127.0.0.1:8000` and opens `frontend/index.html`.

### Option B: Manual

```powershell
& .venv\Scripts\Activate.ps1
Set-Location backend
uvicorn main:app --reload
```

Then open `frontend/index.html` in your browser.

## 4) Login

Default admin credentials:
- Username: `admin`
- Password: `admin123`

(Or use your configured `AI_ADMIN_USERNAME` and `AI_ADMIN_PASSWORD`.)

## 5) API Endpoints

- `POST /auth/login`
- `POST /upload`
- `POST /predict`
- `GET /dashboard`
- `GET /history`
- `GET /clusters`
- `GET /feature-importance`
- `GET /insights`
- `GET /latest-upload`

FastAPI docs: `http://127.0.0.1:8000/docs`

## 6) Functional Flow

1. Login as admin
2. Upload CSV and select task type (`classification` or `regression`)
3. Backend preprocesses data and auto-trains models
4. Dashboard refreshes with KPIs, charts, clustering, and feature importance
5. Run real-time predictions from frontend
6. Predictions (and metadata) are stored in MySQL history

## Notes

- Model files are saved under `saved_models/` and auto-updated on each new dataset upload.
- The backend creates required MySQL tables on startup.
