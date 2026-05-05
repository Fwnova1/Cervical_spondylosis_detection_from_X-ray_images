# C-Spine Predictor Web App

Web app for cervical spine degeneration prediction using:
- YOLO keypoint detection
- a tabular classifier (joblib `.pkl`)
- FastAPI backend + React (Vite) frontend

## Project structure

- `backend/`: FastAPI API and prediction pipeline.
- `frontend/`: React UI for upload, prediction, and keypoint adjustment.
- `models/`: model files used by backend at runtime.

## Requirements

- Python 3.10+
- Node.js 18+

## Model files

Place these files in `models/` (repo root):

- `models/best.pt`
- `models/cervical_spondylosis_xgb.pkl`

The models can be found here: https://drive.google.com/drive/folders/1fO4c7YnIOq2E7oChOcuL1MUI9DnVob9C?usp=sharing.

## Important filename rule

For `/predict`, the backend parses demographics from the uploaded filename:

- gender is character at index `4` (must be `0` or `1`)
- age starts at index `5` (must be digits)

Example:
- `0001054.png` -> gender `0`, age `54`

If the filename format is invalid, the API returns `400`.

## Run backend (local, no Docker)

From repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Set environment variables (optional if using default `models/` paths):

```powershell
$env:YOLO_MODEL_PATH="C:/path/to/best.pt"
$env:CLASSIFIER_MODEL_PATH="C:/path/to/cervical_spondylosis_xgb.pkl"
$env:FRONTEND_ORIGIN="http://localhost:5173"
```

Start API:

```powershell
uvicorn backend.main:app --reload --port 8000
```

API docs:
- `http://localhost:8000/docs`

## Run frontend (local)

```powershell
cd frontend
npm install
npm run dev
```

Optional API base URL override:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
```

Open:
- `http://localhost:5173`

## API endpoints

- `GET /health`: health check
- `POST /predict`: upload image, auto-extract keypoints + demographics from filename
- `POST /predict/keypoints`: predict from manually adjusted keypoints + explicit age/gender
