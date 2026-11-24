
import sys
import time
import base64
import threading
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Project directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import Engine

engine = None
prediction_lock = threading.Lock()

# Load model only once
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    print("Loading Signify AI recognition engine...")
    engine = Engine()

    print("SIGNIFY AI MODEL READY!")
    print("Supported alphabets:", engine.model.classes_)
    print("Input features:", engine.model.n_features_in_)

    yield

    engine.det.close()


app = FastAPI(
    title="Signify AI",
    description="Real-time ASL Recognition API",
    version="1.0.0",
    lifespan=lifespan
)

# Connect React frontend to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve gesture images and evaluation reports
app.mount(
    "/assets",
    StaticFiles(directory=str(ROOT / "assets")),
    name="assets"
)

app.mount(
    "/reports",
    StaticFiles(directory=str(ROOT / "reports")),
    name="reports"
)


@app.get("/")
def home():
    return {
        "name": "Signify AI",
        "status": "online",
        "message": "Your Hands. Your Voice."
    }


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_loaded": engine is not None,
        "model": "Advanced SVM",
        "features": int(engine.model.n_features_in_),
        "classes": engine.model.classes_.tolist()
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):

    start_time = time.perf_counter()

    try:
        # Read uploaded camera frame
        contents = file.file.read(5 * 1024 * 1024 + 1)

        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image is too large"
            )

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image"
            )

        # Reduce processing size while preserving ratio
        h, w = image.shape[:2]

        if max(h, w) > 640:
            scale = 640 / max(h, w)

            image = cv2.resize(
                image,
                (
                    max(1, int(w * scale)),
                    max(1, int(h * scale))
                )
            )

        # Use existing trained recognition engine
        with prediction_lock:
            output, letter, confidence = engine.predict(image)

        # Encode image containing hand landmarks
        success, buffer = cv2.imencode(
            ".jpg",
            output,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Image encoding failed"
            )

        encoded_image = base64.b64encode(
            buffer
        ).decode("utf-8")

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "success": True,
            "detected": bool(letter),
            "letter": letter,
            "confidence": float(confidence),
            "confidence_percent": round(
                float(confidence) * 100, 2
            ),
            "latency_ms": round(latency, 2),
            "image": (
                "data:image/jpeg;base64,"
                + encoded_image
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        print("Prediction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Recognition failed"
        )
