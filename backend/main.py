import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import predict_from_keypoints, predict_image


app = FastAPI(title="C-Spine Prediction API", version="1.1.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KeypointPredictRequest(BaseModel):
    keypoints: list[list[float]] = Field(..., min_length=3)
    age: int = Field(..., ge=0)
    gender: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    tmp_dir = None
    tmp_path = None

    try:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        original_name = Path(image.filename).name
        tmp_dir = tempfile.mkdtemp(prefix="cspine_")
        tmp_path = os.path.join(tmp_dir, original_name)

        with open(tmp_path, "wb") as tmp:
            tmp.write(content)

        prediction_result = predict_image(tmp_path)
        pred = prediction_result["prediction"]
        return {
            "prediction": pred,
            "probability": prediction_result["probability"],
            "label": "Degenerative" if pred == 1 else "Normal",
            "age": prediction_result["age"],
            "gender": prediction_result["gender"],
            "filename": original_name,
            "features": prediction_result["features"],
            "keypoints": prediction_result["keypoints"],
            "overlay_image": prediction_result["overlay_image"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/predict/keypoints")
def predict_from_adjusted_keypoints(payload: KeypointPredictRequest):
    try:
        prediction_result = predict_from_keypoints(
            keypoints=payload.keypoints,
            age=payload.age,
            gender=payload.gender,
        )
        pred = prediction_result["prediction"]
        return {
            "prediction": pred,
            "probability": prediction_result["probability"],
            "label": "Degenerative" if pred == 1 else "Normal",
            "age": prediction_result["age"],
            "gender": prediction_result["gender"],
            "features": prediction_result["features"],
            "keypoints": prediction_result["keypoints"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
