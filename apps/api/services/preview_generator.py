"""
Preview generation service for Edit Banana.

Generates low-res preview images with watermark and blur effects.
"""

import os
import io
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


class PreviewStore:
    """In-memory store for preview metadata with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self._previews: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def add(self, preview_id: str, file_path: str, ip: str):
        """Add a preview to the store."""
        self._previews[preview_id] = {
            "id": preview_id,
            "path": file_path,
            "ip": ip,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=self._ttl),
            "access_count": 0,
        }

    def get(self, preview_id: str) -> Optional[Dict[str, Any]]:
        """Get preview metadata if not expired."""
        preview = self._previews.get(preview_id)
        if not preview:
            return None

        if datetime.now() > preview["expires_at"]:
            self.delete(preview_id)
            return None

        preview["access_count"] += 1
        return preview

    def delete(self, preview_id: str):
        """Delete preview from store and remove file."""
        preview = self._previews.pop(preview_id, None)
        if preview and os.path.exists(preview["path"]):
            try:
                os.unlink(preview["path"])
            except Exception:
                pass

    def cleanup_expired(self) -> int:
        """Remove expired previews. Returns count cleaned."""
        now = datetime.now()
        expired = [
            pid for pid, p in self._previews.items()
            if now > p["expires_at"]
        ]
        for pid in expired:
            self.delete(pid)
        return len(expired)

    def get_ip_preview_count(self, ip: str, window_seconds: int = 3600) -> int:
        """Count previews created by IP within time window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        return sum(
            1 for p in self._previews.values()
            if p["ip"] == ip and p["created_at"] > cutoff
        )


# Global preview store instance
preview_store = PreviewStore(ttl_seconds=300)


class PreviewGenerator:
    """Generates watermarked preview images."""

    def __init__(
        self,
        quality: int = 50,
        blur_percent: float = 30.0,
        watermark_text: str = "EditBanana Preview",
    ):
        self.quality = quality
        self.blur_percent = blur_percent
        self.watermark_text = watermark_text

    def generate(
        self,
        source_path: str,
        output_path: str,
    ) -> str:
        """
        Generate a preview image from source.

        Args:
            source_path: Path to source image
            output_path: Path to save preview

        Returns:
            Path to generated preview
        """
        # Open source image
        with Image.open(source_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize to 50% for low-res
            width, height = img.size
            new_size = (width // 2, height // 2)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Add diagonal watermark
            img = self._add_watermark(img)

            # Blur bottom portion
            img = self._add_blur_overlay(img)

            # Save with reduced quality
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, "JPEG", quality=self.quality, optimize=True)

        return output_path

    def _add_watermark(self, img: Image.Image) -> Image.Image:
        """Add diagonal watermark across the image."""
        width, height = img.size

        # Create a transparent overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load a font, fallback to default
        try:
            # Use a larger font size relative to image
            font_size = max(width, height) // 15
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Calculate text size and position for diagonal
        bbox = draw.textbbox((0, 0), self.watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position diagonally - center of image, rotated 45 degrees
        center_x = width // 2
        center_y = height // 2

        # Draw multiple watermarks across the image
        positions = [
            (center_x, center_y),
            (center_x - width // 3, center_y - height // 3),
            (center_x + width // 3, center_y + height // 3),
            (center_x - width // 3, center_y + height // 3),
            (center_x + width // 3, center_y - height // 3),
        ]

        for x, y in positions:
            # Semi-transparent white text
            draw.text(
                (x - text_width // 2, y - text_height // 2),
                self.watermark_text,
                font=font,
                fill=(255, 255, 255, 64),  # White with 25% opacity
            )

        # Composite overlay onto image
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    def _add_blur_overlay(self, img: Image.Image) -> Image.Image:
        """Blur the bottom portion of the image."""
        width, height = img.size
        blur_height = int(height * (self.blur_percent / 100))

        if blur_height <= 0:
            return img

        # Split image
        top_region = img.crop((0, 0, width, height - blur_height))
        bottom_region = img.crop((0, height - blur_height, width, height))

        # Blur bottom region
        bottom_blurred = bottom_region.filter(ImageFilter.GaussianBlur(radius=10))

        # Create new image and paste regions
        result = Image.new("RGB", img.size)
        result.paste(top_region, (0, 0))
        result.paste(bottom_blurred, (0, height - blur_height))

        # Add "Pay to unlock" text overlay on blurred portion
        draw = ImageDraw.Draw(result)

        try:
            text_size = max(width, height) // 20
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", text_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_size)
            except Exception:
                font = ImageFont.load_default()

        # Draw text in center of blurred area
        text = "Pay $X to unlock full quality"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = (width - text_width) // 2
        text_y = height - blur_height // 2 - text_height // 2

        # Draw text with dark background for readability
        padding = 10
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill=(0, 0, 0, 180)
        )
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        return result


def generate_preview_id(job_id: str, ip: str) -> str:
    """Generate a unique preview ID from job_id and IP."""
    data = f"{job_id}:{ip}:{datetime.now().timestamp()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def get_preview_path(output_dir: str, preview_id: str) -> str:
    """Get the file path for a preview."""
    previews_dir = os.path.join(output_dir, "previews")
    return os.path.join(previews_dir, f"{preview_id}.jpg")
