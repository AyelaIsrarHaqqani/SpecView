import os
import tempfile
from io import BytesIO

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.inference.run_inference import load_artifacts, load_signal, predict_signal
