#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides upload and conversion API. Run with: python server_pa.py
Server runs at http://localhost:8000

Improvements over initial version:
- Pipeline loaded once at startup (SAM3 model stays in GPU memory)
- /convert returns the DrawIO XML file directly (FileResponse)
- Upload size capped at MAX_UPLOAD_BYTES to prevent OOM
- Temp files cleaned up reliably
"""

import os
import sys
import logging
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from main import load_config, Pipeline

logger = logging.getLogger("edit-banana")

# ======================== constants ========================

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# ======================== singleton pipeline ========================

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Return the shared Pipeline instance (created on first call)."""
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = Pipeline(config)
        logger.info("Pipeline initialized")
    return _pipeline


# ======================== lifespan ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the pipeline so the first request doesn't pay model-load latency."""
    logger.info("Starting Edit Banana API — loading pipeline …")
    get_pipeline()
    logger.info("Pipeline ready")
    yield
    logger.info("Shutting down")


# ======================== app ========================

app = FastAPI(
    title="Edit Banana API",
    description="Image to editable DrawIO (XML) — upload a diagram image, get DrawIO XML.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "Edit Banana", "version": "1.1.0", "docs": "/docs"}


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload an image and get editable DrawIO XML.

    Supported formats: PNG, JPG, BMP, TIFF, WebP.
    Max file size: 20 MB.

    Returns the generated .drawio XML file directly as a download.
    """
    # --- validate extension ---
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # --- validate file size ---
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(contents) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB.",
        )

    # --- run pipeline ---
    pipeline = get_pipeline()
    config = pipeline.config
    output_dir = config.get("paths", {}).get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=output_dir) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result_path = pipeline.process_image(
            tmp_path,
            output_dir=output_dir,
            with_refinement=False,
            with_text=True,
        )

        if not result_path or not os.path.exists(result_path):
            raise HTTPException(500, "Conversion failed — no output generated.")

        # Return the actual DrawIO XML file as a download.
        download_name = Path(name).stem + ".drawio"
        return FileResponse(
            path=result_path,
            media_type="application/xml",
            filename=download_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Conversion error")
        raise HTTPException(500, f"Conversion failed: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
