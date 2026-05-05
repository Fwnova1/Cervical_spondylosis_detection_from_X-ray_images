import base64
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-8

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "best.pt"),
)
CLASSIFIER_MODEL_PATH = os.getenv(
    "CLASSIFIER_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "cervical_spondylosis_xgb.pkl"),
)


yolo_model = YOLO(YOLO_MODEL_PATH)
model_package = joblib.load(CLASSIFIER_MODEL_PATH)

clf_model = model_package["model"]
imputer = model_package.get("imputer")
threshold = model_package.get("best_threshold", model_package.get("threshold", 0.5))
feature_names = model_package["feature_names"]


def extract_keypoints_from_yolo(result) -> np.ndarray:
    if result.keypoints is None or result.keypoints.data is None or len(result.keypoints.data) == 0:
        raise ValueError("No keypoints detected by YOLO.")

    keypoints = result.keypoints.data[0].cpu().numpy()[:, :2].astype(float)
    h, w = result.orig_shape

    if np.nanmax(keypoints) <= 1.5:
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h

    if len(keypoints) < 3:
        raise ValueError(f"Not enough keypoints: expected >=3, got {len(keypoints)}")

    keypoints[[0, 1, 2]] = keypoints[[2, 0, 1]].copy()
    return keypoints


def extract_age_gender(image_path: str) -> Tuple[int, int]:
    name = Path(image_path).stem

    if len(name) < 6:
        raise ValueError(
            "Filename must contain at least 6 characters (gender at index 4, age from index 5)."
        )
    if not name[4].isdigit() or not name[5:].isdigit():
        raise ValueError(
            "Filename format invalid. Expected gender at index 4 and age digits after index 5."
        )

    gender = int(name[4])
    age = int(name[5:])

    if gender not in (0, 1):
        raise ValueError("Parsed gender must be 0 or 1.")
    if age < 0:
        raise ValueError("Parsed age must be >= 0.")

    return age, gender


def distance(p1, p2) -> float:
    return float(np.linalg.norm(p1 - p2))


def slope(p1, p2) -> float:
    return float((p2[1] - p1[1]) / (p2[0] - p1[0] + EPSILON))


def compute_features(keypoints: np.ndarray, age: int, gender: int) -> Dict[str, float]:
    kp = keypoints
    features: Dict[str, float] = {}

    centers = kp[::3]
    if len(centers) < 6:
        raise ValueError(f"Not enough vertebra centers: expected >=6, got {len(centers)}")

    discs = np.array(
        [
            distance(centers[0], centers[1]),
            distance(centers[1], centers[2]),
            distance(centers[2], centers[3]),
            distance(centers[3], centers[4]),
            distance(centers[4], centers[5]),
        ]
    )

    disc_cols = [
        "Disc height (final) | C2-3 disc height",
        "Disc height (final) | C3-4 disc height",
        "Disc height (final) | C4-5 disc height",
        "Disc height (final) | C5-6 disc height",
        "Disc height (final) | C6-7 disc height",
    ]

    for name, val in zip(disc_cols, discs):
        features[name] = float(val)

    features["min_disc_height"] = float(discs.min())
    features["mean_disc_height"] = float(discs.mean())
    features["std_disc_height"] = float(discs.std())

    features["ratio_C3_4"] = float(discs[1] / (discs[0] + EPSILON))
    features["ratio_C4_5"] = float(discs[2] / (discs[1] + EPSILON))
    features["ratio_C5_6"] = float(discs[3] / (discs[2] + EPSILON))
    features["ratio_C6_7"] = float(discs[4] / (discs[3] + EPSILON))

    try:
        slope_c2 = slope(kp[1], kp[2])
        slope_c7 = slope(kp[-2], kp[-1])
        angle_c2 = np.degrees(np.arctan(slope_c2))
        angle_c7 = np.degrees(np.arctan(slope_c7))
        features["cobb_diff"] = float(abs(angle_c2 - angle_c7))
    except Exception:
        features["cobb_diff"] = 0.0

    pairs = [
        (1, 3),
        (4, 6),
        (7, 9),
        (10, 12),
        (13, 15),
        (2, 4),
        (5, 7),
        (8, 10),
        (11, 13),
        (14, 16),
    ]

    instability = np.array(
        [abs(kp[a][0] - kp[b][0]) for a, b in pairs if a < len(kp) and b < len(kp)]
    )

    if instability.size == 0:
        features["total_instability"] = 0.0
        features["max_instability"] = 0.0
        features["instability_std"] = 0.0
        features["instability_range"] = 0.0
    else:
        features["total_instability"] = float(instability.sum())
        features["max_instability"] = float(instability.max())
        features["instability_std"] = float(instability.std())
        features["instability_range"] = float(instability.max() - instability.min())

    slope_pairs = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (19, 20)]
    slopes = np.array([slope(kp[a], kp[b]) for a, b in slope_pairs if a < len(kp) and b < len(kp)])

    if slopes.size == 0:
        features["slope_mean"] = 0.0
        features["slope_std"] = 0.0
        abs_slopes = np.array([0.0])
    else:
        features["slope_mean"] = float(slopes.mean())
        features["slope_std"] = float(slopes.std())
        abs_slopes = np.abs(slopes)

    features["endplate_mean"] = float(abs_slopes.mean())
    features["endplate_max"] = float(abs_slopes.max())
    features["endplate_std"] = float(abs_slopes.std())
    features["severe_endplate"] = int(abs_slopes.max() > abs_slopes.mean() + abs_slopes.std())

    max_disc = discs.max()
    grad = np.abs(np.diff(discs))

    features["degeneration_count"] = int(
        np.sum((discs < max_disc * 0.85) | (np.append(grad, 0) > grad.mean()))
    )
    features["severe_deg_count"] = int(
        np.sum((discs < max_disc * 0.70) | (np.append(grad, 0) > grad.mean() + grad.std()))
    )
    features["has_severe_deg"] = int(features["severe_deg_count"] > 0)

    features["Basic information | Age"] = int(age)
    features["gender"] = int(gender)
    features["age_disc_interaction"] = float(age / (features["min_disc_height"] + EPSILON))
    features["age_adjusted_deg"] = float(features["degeneration_count"] / (age + 1))
    features["age_weighted_disc"] = float(age * features["mean_disc_height"])

    features["upper_disc_mean"] = float(discs[:2].mean())
    features["middle_disc_mean"] = float(discs[2])
    features["lower_disc_mean"] = float(discs[3:].mean())
    features["upper_lower_ratio"] = float(features["upper_disc_mean"] / (features["lower_disc_mean"] + EPSILON))

    gradients = np.abs(np.diff(discs))
    features["max_disc_gradient"] = float(gradients.max())

    features["degeneration_score"] = float(
        features["degeneration_count"] + features["total_instability"] + features["severe_deg_count"]
    )

    features["disc_variation"] = float(features["std_disc_height"] / (features["mean_disc_height"] + EPSILON))
    features["degeneration_intensity"] = float(features["severe_deg_count"] * features["max_instability"])
    features["risk_score"] = float(
        features["degeneration_count"] * 0.4
        + features["total_instability"] * 0.3
        + features["age_disc_interaction"] * 0.3
    )

    return features


def render_keypoint_overlay(image_path: str, keypoints: np.ndarray) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is required to render keypoint overlay.") from exc

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to read image for keypoint overlay.")

    overlay = image.copy()

    for idx, (x, y) in enumerate(keypoints):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(overlay, pt, 4, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(idx + 1),
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for i in range(len(keypoints) - 1):
        p1 = (int(round(keypoints[i][0])), int(round(keypoints[i][1])))
        p2 = (int(round(keypoints[i + 1][0])), int(round(keypoints[i + 1][1])))
        cv2.line(overlay, p1, p2, (0, 170, 255), thickness=1, lineType=cv2.LINE_AA)

    rendered = cv2.addWeighted(overlay, 0.85, image, 0.15, 0.0)
    success, png_buffer = cv2.imencode(".png", rendered)
    if not success:
        raise ValueError("Failed to encode keypoint overlay image.")

    encoded = base64.b64encode(png_buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_keypoints(keypoints: Iterable[Iterable[float]]) -> np.ndarray:
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[1] != 2:
        raise ValueError("Keypoints must be a 2D array with shape [N, 2].")
    if kp.shape[0] < 3:
        raise ValueError("At least 3 keypoints are required.")
    if not np.isfinite(kp).all():
        raise ValueError("Keypoints contain invalid numeric values.")
    return kp


def run_classifier(features: Dict[str, float]) -> Tuple[int, float]:
    df = pd.DataFrame([features]).reindex(columns=feature_names)

    if imputer is not None:
        try:
            df_imputed = imputer.transform(df)
        except AttributeError:
            if hasattr(imputer, "statistics_") and imputer.statistics_ is not None:
                df_manual = df.copy()
                stats = np.asarray(imputer.statistics_, dtype=float)
                for idx, col in enumerate(df_manual.columns):
                    fill_val = stats[idx] if idx < len(stats) else np.nan
                    df_manual[col] = df_manual[col].fillna(fill_val)
                df_imputed = df_manual.to_numpy(dtype=float)
            else:
                df_imputed = df.fillna(0).to_numpy(dtype=float)
    else:
        df_imputed = df.fillna(0).to_numpy(dtype=float)

    prob = float(clf_model.predict_proba(df_imputed)[:, 1][0])
    pred = int(prob > threshold)
    return pred, prob


def predict_from_keypoints(keypoints: Iterable[Iterable[float]], age: int, gender: int) -> Dict[str, Any]:
    kp = normalize_keypoints(keypoints)
    features = compute_features(kp, age=age, gender=gender)
    pred, prob = run_classifier(features)

    return {
        "prediction": pred,
        "probability": prob,
        "features": features,
        "age": age,
        "gender": gender,
        "keypoints": [[float(x), float(y)] for x, y in kp],
    }


def predict_image(image_path: str) -> Dict[str, Any]:
    results = yolo_model(image_path)
    if not results:
        raise ValueError("YOLO returned no results.")

    keypoints = extract_keypoints_from_yolo(results[0])
    age, gender = extract_age_gender(image_path)
    prediction_result = predict_from_keypoints(keypoints, age=age, gender=gender)
    overlay_image = render_keypoint_overlay(image_path, keypoints)

    return {
        **prediction_result,
        "overlay_image": overlay_image,
    }
