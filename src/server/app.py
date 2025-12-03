import os
import tempfile
from io import BytesIO

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.inference.run_inference import load_artifacts, load_signal, predict_signal


def create_app() -> FastAPI:
    app = FastAPI(title="SpecView Inference API", version="1.0.0")

    # Enable CORS for local dev (adjust origins as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )_headers=["*"],
    )

    # Load artifacts once at startup
    model, scaler, label_encoder, _, device = load_artifacts(
        model_path=os.getenv("MODEL_PATH", "best_cnn1d.pth"),
        scaler_path=os.getenv("SCALER_PATH", "scaler.pkl"),
        encoder_path=os.getenv("ENCODER_PATH", "label_encoder.pkl"),
        sst_params_path=os.getenv("SST_PARAMS_PATH", "sst_params.json"),
        device=os.getenv("DEVICE", None),
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)):
        try:
            # Persist upload to a temporary file to reuse existing loader
            suffix = os.path.splitext(file.filename or "")[1].lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Convert to 1D signal and predict
            signal = load_signal(tmp_path)
            label, confidence = predict_signal(signal, model, scaler, label_encoder)
            return JSONResponse({"label": label, "confidence": float(confidence)})
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return app