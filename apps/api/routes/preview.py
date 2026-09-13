"""
Preview API routes for Edit Banana.

Provides preview generation endpoint with rate limiting.
"""

import os
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from services.preview_generator import (
    PreviewGenerator,
    preview_store,
    generate_preview_id,
    get_preview_path,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["preview"])

# Rate limit configuration
RATE_LIMIT_MAX = 3  # max previews per IP per hour
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds


@router.post("/{job_id}/preview")
async def create_preview(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Generate a preview for a completed job.

    Args:
        job_id: The job ID to generate preview from

    Returns:
        Preview metadata including URL and expiration time

    Raises:
        429: Rate limit exceeded (3 previews/hour/IP)
        404: Job not found or no output available
        400: Job not completed yet
    """
    # Get client IP
    client_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP")
        or request.client.host
    )

    # Check rate limit
    preview_count = preview_store.get_ip_preview_count(client_ip, RATE_LIMIT_WINDOW)
    if preview_count >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Rate limit exceeded: {RATE_LIMIT_MAX} previews per hour",
                "limit": RATE_LIMIT_MAX,
                "window_seconds": RATE_LIMIT_WINDOW,
                "current_count": preview_count,
            },
        )

    # Find job output file
    # Get output directory from config or use default
    from main import load_config
    config = load_config()
    output_dir = config.get("paths", {}).get("output_dir", "./output")

    # Look for job output (drawio or image file)
    job_output_path = None
    job_dir = os.path.join(output_dir, job_id)

    if os.path.exists(job_dir):
        # Check for original uploaded image in job directory
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
            candidate = os.path.join(job_dir, f"input{ext}")
            if os.path.exists(candidate):
                job_output_path = candidate
                break

    # Fallback: check for any image in output directory
    if not job_output_path:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
            candidate = os.path.join(output_dir, f"{job_id}{ext}")
            if os.path.exists(candidate):
                job_output_path = candidate
                break

    if not job_output_path or not os.path.exists(job_output_path):
        raise HTTPException(
            status_code=404,
            detail="Job not found or no output available for preview",
        )

    # Generate preview
    preview_id = generate_preview_id(job_id, client_ip)
    preview_path = get_preview_path(output_dir, preview_id)

    # Check if preview already exists (re-use)
    existing = preview_store.get(preview_id)
    if existing and os.path.exists(preview_path):
        return {
            "success": True,
            "preview_id": preview_id,
            "preview_url": f"/api/v1/previews/{preview_id}",
            "expires_at": existing["expires_at"].isoformat(),
            "rate_limit": {
                "limit": RATE_LIMIT_MAX,
                "window_seconds": RATE_LIMIT_WINDOW,
                "current_count": preview_count + 1,
            },
        }

    # Generate new preview
    generator = PreviewGenerator(
        quality=50,
        blur_percent=30.0,
        watermark_text="EditBanana Preview",
    )

    try:
        generator.generate(job_output_path, preview_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}",
        )

    # Store preview metadata
    preview_store.add(preview_id, preview_path, client_ip)

    # Schedule cleanup in background
    background_tasks.add_task(_cleanup_expired_previews)

    return {
        "success": True,
        "preview_id": preview_id,
        "preview_url": f"/api/v1/previews/{preview_id}",
        "expires_at": preview_store.get(preview_id)["expires_at"].isoformat(),
        "rate_limit": {
            "limit": RATE_LIMIT_MAX,
            "window_seconds": RATE_LIMIT_WINDOW,
            "current_count": preview_count + 1,
        },
    }


@router.get("/api/v1/previews/{preview_id}")
async def get_preview(preview_id: str):
    """
    Serve a preview image.

    Args:
        preview_id: The preview ID

    Returns:
        JPEG image file

    Raises:
        404: Preview not found or expired
    """
    preview = preview_store.get(preview_id)
    if not preview:
        raise HTTPException(
            status_code=404,
            detail="Preview not found or expired",
        )

    preview_path = preview["path"]
    if not os.path.exists(preview_path):
        raise HTTPException(
            status_code=404,
            detail="Preview file not found",
        )

    return FileResponse(
        preview_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Preview-Id": preview_id,
        },
    )


@router.delete("/api/v1/previews/{preview_id}")
async def delete_preview(preview_id: str):
    """
    Delete a preview (manual cleanup).

    Args:
        preview_id: The preview ID to delete

    Returns:
        Success confirmation
    """
    preview = preview_store.get(preview_id)
    if preview:
        preview_store.delete(preview_id)

    return {"success": True, "message": "Preview deleted"}


def _cleanup_expired_previews():
    """Background task to cleanup expired previews."""
    count = preview_store.cleanup_expired()
    return count
