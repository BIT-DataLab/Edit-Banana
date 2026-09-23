#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides upload and conversion API. Run with: python server_pa.py
Server runs at http://localhost:8000
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from modules.exceptions import EditBananaException, ErrorSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Edit Banana API",
    description="Image to editable DrawIO (XML) — upload a diagram image, get DrawIO XML.",
    version="1.1.0",
)


class APIErrorResponse:
    """Standardized API error response structure."""

    @staticmethod
    def create(
        message: str,
        error_code: Optional[str] = None,
        can_retry: bool = False,
        partial_results: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ) -> JSONResponse:
        """Create a structured error response."""
        content = {
            "success": False,
            "error": {
                "message": message,
                "code": error_code or "UNKNOWN_ERROR",
                "can_retry": can_retry,
            }
        }
        if partial_results:
            content["partial_results"] = partial_results
        return JSONResponse(content=content, status_code=status_code)


@app.get("/")
def root():
    return {"service": "Edit Banana", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    config_exists = os.path.exists(config_path)
    return {
        "status": "healthy" if config_exists else "degraded",
        "config_loaded": config_exists,
        "version": "1.1.0"
    }


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    resume: bool = False
):
    """
    Upload an image and get editable DrawIO XML. Supported: PNG, JPG, BMP, TIFF, WebP.

    Args:
        file: Image file to convert
        resume: If true, attempt to resume from previous checkpoint
    """
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        return APIErrorResponse.create(
            message=f"Unsupported format. Use one of: {', '.join(sorted(allowed))}.",
            error_code="UNSUPPORTED_FORMAT",
            can_retry=False,
            status_code=400
        )

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        return APIErrorResponse.create(
            message="Server not configured (missing config/config.yaml)",
            error_code="SERVER_NOT_CONFIGURED",
            can_retry=False,
            status_code=503
        )

    try:
        from main import load_config, Pipeline, PipelineResult
        import tempfile
        import shutil

        config = load_config()
        output_dir = config.get("paths", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            pipeline = Pipeline(config)
            result: PipelineResult = pipeline.process_image(
                tmp_path,
                output_dir=output_dir,
                with_refinement=False,
                with_text=True,
                resume_from_checkpoint=resume
            )

            if result.success:
                return {
                    "success": True,
                    "output_path": result.output_path,
                    "stages_completed": result.last_completed_stage + 1
                }
            else:
                # Build partial results for error response
                partial_results = None
                if result.partial_elements or result.partial_xml_fragments:
                    partial_results = {
                        "last_completed_stage": result.last_completed_stage,
                        "elements_found": len(result.partial_elements),
                        "fragments_generated": len(result.partial_xml_fragments),
                        "checkpoint_available": result.checkpoint_path is not None
                    }

                return APIErrorResponse.create(
                    message=result.error_message or "Conversion failed",
                    error_code=result.error_code or "CONVERSION_FAILED",
                    can_retry=result.can_retry,
                    partial_results=partial_results,
                    status_code=500 if not result.can_retry else 503
                )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        logger.exception("Unexpected error during conversion")
        return APIErrorResponse.create(
            message=str(e),
            error_code="INTERNAL_ERROR",
            can_retry=True,
            status_code=500
        )


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
