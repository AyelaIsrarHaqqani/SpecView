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
    )