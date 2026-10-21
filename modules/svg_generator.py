"""
Stage 8a: SVG Generator Module

Converts ElementInfo objects into SVG files.

Hybrid approach:
    - Known shapes (rectangle, ellipse, diamond, etc.) -> clean geometric SVG primitives
    - Complex/unknown shapes -> smoothed polygon SVG paths
    - Raster elements (icon, picture, photo) -> SVG with embedded base64 <image>
    - Text elements -> SVG <text> with editable text
    - Arrows -> SVG <line>/<polyline> with marker arrowheads

Output: individual SVG files per element + combined SVG with all elements grouped.

Usage:
    from modules.svg_generator import SVGGenerator

    generator = SVGGenerator()
    # Generate individual element SVG
    svg_string = generator.element_to_svg(element, image)
    # Generate combined SVG
    combined_svg = generator.generate_combined_svg(elements, image, canvas_w, canvas_h)
"""

import os
import io
import base64
import math
from typing import List, Optional, Tuple, Dict, Any
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from PIL import Image
import cv2

from .base import BaseProcessor, ProcessingContext
from .data_types import (
    ElementInfo,
    BoundingBox,
    ProcessingResult,
    LayerLevel,
    get_layer_level,
)

# Prompt lists are the single source of truth for element-type classification.
# Importing here means any new prompt auto-registers as its category —
# no silent "falls through to polygon outline" bugs (see heart-model regression).
from prompts.image import IMAGE_PROMPT
from prompts.shape import SHAPE_PROMPT
from prompts.arrow import ARROW_PROMPT


# ======================== Constants ========================

# SVG 1.1 header template (Canva requires SVG 1.1 profile)
SVG_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     version="1.1"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
"""
SVG_FOOTER = "</svg>\n"

# Arrowhead marker definition (reusable)
ARROW_MARKER_DEF = """  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7"
            refX="10" refY="3.5" orient="auto" fill="{color}">
      <polygon points="0 0, 10 3.5, 0 7" />
    </marker>
    <marker id="arrowhead-start" markerWidth="10" markerHeight="7"
            refX="0" refY="3.5" orient="auto" fill="{color}">
      <polygon points="10 0, 0 3.5, 10 7" />
    </marker>
  </defs>
"""


def _expand_forms(prompts):
    """
    For each prompt name, yield both the lowercase space form AND the
    lowercase underscore form. This protects against the two different
    element_type normalization conventions used downstream:
        - svg_generator.py uses  element.element_type.lower()         ("3d heart model")
        - vector_exporter.py uses ....lower().replace(" ", "_")        ("3d_heart_model")
    """
    out = set()
    for p in prompts:
        low = p.lower()
        out.add(low)
        out.add(low.replace(" ", "_"))
    return out


# Known geometric shape types that get clean SVG primitives.
# Derived from SHAPE_PROMPT + legacy container types.
GEOMETRIC_SHAPES = _expand_forms(SHAPE_PROMPT) | {
    'section_panel', 'title_bar',  # legacy container types (not in prompts)
    'actor',                        # legacy
}

# Raster/image element types that get embedded base64.
# Derived from IMAGE_PROMPT + legacy ElementType enum entries.
# CRITICAL: without IMAGE_PROMPT expansion, "3D heart model" / "MRI image"
# fall through to polygon outline rendering (blank white shape bug).
RASTER_TYPES = _expand_forms(IMAGE_PROMPT) | {
    'function_graph', 'image',  # legacy ElementType entries not in prompts
}

# Arrow/connector types. Derived from ARROW_PROMPT + legacy names.
ARROW_TYPES = _expand_forms(ARROW_PROMPT) | {
    'arrow', 'line', 'connector',  # legacy (overlap with prompts but explicit)
}

# Web-safe font stack for text elements
DEFAULT_FONT_FAMILY = "Arial, Helvetica, sans-serif"


class SVGGenerator:
    """
    Generates SVG files from ElementInfo objects.

    Hybrid strategy:
        - Geometric shapes -> clean <rect>, <ellipse>, <polygon> primitives
        - Complex shapes -> smoothed <path> from polygon contours
        - Raster elements -> <image> with base64 data
        - Text -> <text> with editable content
        - Arrows -> <polyline>/<line> with arrowhead markers
    """

    def __init__(self):
        self._log_prefix = "[SVGGenerator]"

    def _log(self, msg: str):
        print(f"{self._log_prefix} {msg}")

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def element_to_svg(
        self,
        element: ElementInfo,
        image: np.ndarray,
        standalone: bool = True,
        offset: Tuple[int, int] = (0, 0),
    ) -> str:
        """
        Convert a single ElementInfo to an SVG string.

        Args:
            element: The element to convert
            image: Original image as numpy array (BGR, for cropping rasters)
            standalone: If True, wrap in full SVG document; if False, return inner element only
            offset: (ox, oy) to subtract from coordinates (for section-relative positioning)

        Returns:
            SVG string
        """
        elem_type = element.element_type.lower()

        # Determine which renderer to use
        if elem_type in RASTER_TYPES:
            inner = self._raster_element_svg(element, image, offset)
        elif elem_type in ARROW_TYPES:
            inner = self._arrow_element_svg(element, image, offset)
        elif elem_type == 'text':
            inner = self._text_element_svg(element, offset)
        elif elem_type in GEOMETRIC_SHAPES:
            inner = self._geometric_shape_svg(element, offset)
        else:
            # Unknown type: try polygon fallback, or raster crop
            if element.polygon and len(element.polygon) >= 3:
                inner = self._polygon_shape_svg(element, offset)
            else:
                inner = self._raster_element_svg(element, image, offset)

        if not standalone:
            return inner

        # Wrap in standalone SVG document
        bbox = element.bbox
        w = bbox.width + 4  # small padding
        h = bbox.height + 4
        header = SVG_HEADER.format(width=w, height=h)

        # If arrow, include marker defs
        defs = ""
        if elem_type in ARROW_TYPES:
            stroke_color = element.stroke_color or "#000000"
            defs = ARROW_MARKER_DEF.format(color=stroke_color)

        # Re-offset inner content to (2,2) padding origin
        # We need to wrap in a <g> with translate
        ox, oy = offset
        tx = -bbox.x1 + ox + 2
        ty = -bbox.y1 + oy + 2

        return (
            header
            + defs
            + f'  <g transform="translate({tx},{ty})">\n'
            + inner
            + "  </g>\n"
            + SVG_FOOTER
        )

    def generate_combined_svg(
        self,
        elements: List[ElementInfo],
        image: np.ndarray,
        canvas_width: int,
        canvas_height: int,
    ) -> str:
        """
        Generate a single SVG containing ALL elements as named <g> groups.

        Elements are layered by their layer_level (background first, text on top).
        """
        header = SVG_HEADER.format(width=canvas_width, height=canvas_height)

        # Arrow marker defs
        defs = ARROW_MARKER_DEF.format(color="#000000")

        # Sort by layer level (low = bottom = rendered first in SVG)
        sorted_elems = sorted(elements, key=lambda e: e.layer_level)

        # Group by layer
        layer_names = {
            LayerLevel.BACKGROUND.value: "background",
            LayerLevel.BASIC_SHAPE.value: "shapes",
            LayerLevel.IMAGE.value: "images",
            LayerLevel.ARROW.value: "arrows",
            LayerLevel.TEXT.value: "text",
            LayerLevel.OTHER.value: "other",
        }

        body_parts = []
        current_layer = None
        for elem in sorted_elems:
            layer = elem.layer_level
            if layer != current_layer:
                if current_layer is not None:
                    body_parts.append("  </g>\n")
                layer_name = layer_names.get(layer, f"layer_{layer}")
                body_parts.append(
                    f'  <g id="layer-{layer_name}" data-layer="{layer_name}">\n'
                )
                current_layer = layer

            # Element group
            elem_type = elem.element_type.lower().replace(" ", "_")
            group_id = f"elem-{elem.id}-{elem_type}"
            body_parts.append(f'    <g id="{group_id}">\n')
            inner_svg = self.element_to_svg(elem, image, standalone=False)
            # Indent inner content
            for line in inner_svg.strip().split("\n"):
                body_parts.append(f"      {line}\n")
            body_parts.append("    </g>\n")

        # Close last layer group
        if current_layer is not None:
            body_parts.append("  </g>\n")

        return header + defs + "".join(body_parts) + SVG_FOOTER

    def crop_raster_element(
        self, element: ElementInfo, image: np.ndarray
    ) -> Optional[Image.Image]:
        """
        Crop a raster element from the image with transparent background.

        Uses the element's polygon mask if available, otherwise uses bbox crop.
        Returns a PIL Image with RGBA (transparent background).
        """
        bbox = element.bbox
        x1 = max(0, bbox.x1)
        y1 = max(0, bbox.y1)
        x2 = min(image.shape[1], bbox.x2)
        y2 = min(image.shape[0], bbox.y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2].copy()

        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # If polygon available, create alpha mask
        if element.polygon and len(element.polygon) >= 3:
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            # Offset polygon to crop coordinates
            pts = np.array(element.polygon, dtype=np.int32)
            pts[:, 0] -= x1
            pts[:, 1] -= y1
            cv2.fillPoly(mask, [pts], 255)
            # Create RGBA
            rgba = np.dstack([crop_rgb, mask])
        else:
            # No polygon — full opaque bbox crop
            alpha = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
            rgba = np.dstack([crop_rgb, alpha])

        return Image.fromarray(rgba, "RGBA")

    def save_raster_crop(
        self, element: ElementInfo, image: np.ndarray, output_path: str
    ) -> Optional[str]:
        """Crop and save raster element as PNG. Returns path or None."""
        pil_img = self.crop_raster_element(element, image)
        if pil_img is None:
            return None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_img.save(output_path, "PNG")
        return output_path

    # ================================================================
    #  PRIVATE: Shape Renderers
    # ================================================================

    def _geometric_shape_svg(
        self, element: ElementInfo, offset: Tuple[int, int] = (0, 0)
    ) -> str:
        """Render known geometric shapes as clean SVG primitives."""
        bbox = element.bbox
        ox, oy = offset
        x = bbox.x1 - ox
        y = bbox.y1 - oy
        w = bbox.width
        h = bbox.height

        fill = element.fill_color or "#ffffff"
        stroke = element.stroke_color or "#000000"
        sw = element.stroke_width or 1
        elem_type = element.element_type.lower()

        style = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'

        if elem_type in ('rectangle', 'section_panel', 'title_bar'):
            return f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" {style} />\n'

        elif elem_type in ('rounded_rectangle', 'rounded rectangle'):
            rx = min(10, w // 6, h // 6)
            return f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" {style} />\n'

        elif elem_type in ('ellipse', 'circle'):
            cx = x + w // 2
            cy = y + h // 2
            rx = w // 2
            ry = h // 2
            return f'    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" {style} />\n'

        elif elem_type == 'diamond':
            cx = x + w // 2
            cy = y + h // 2
            points = f"{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}"
            return f'    <polygon points="{points}" {style} />\n'

        elif elem_type == 'triangle':
            cx = x + w // 2
            points = f"{cx},{y} {x + w},{y + h} {x},{y + h}"
            return f'    <polygon points="{points}" {style} />\n'

        elif elem_type == 'hexagon':
            # Flat-top hexagon
            cx = x + w // 2
            cy = y + h // 2
            qw = w // 4
            points = (
                f"{x + qw},{y} {x + 3 * qw},{y} "
                f"{x + w},{cy} "
                f"{x + 3 * qw},{y + h} {x + qw},{y + h} "
                f"{x},{cy}"
            )
            return f'    <polygon points="{points}" {style} />\n'

        elif elem_type == 'parallelogram':
            skew = w // 5
            points = (
                f"{x + skew},{y} {x + w},{y} "
                f"{x + w - skew},{y + h} {x},{y + h}"
            )
            return f'    <polygon points="{points}" {style} />\n'

        elif elem_type == 'trapezoid':
            # Wider base at bottom (like encoder in ML figures)
            inset = w // 6
            points = (
                f"{x + inset},{y} {x + w - inset},{y} "
                f"{x + w},{y + h} {x},{y + h}"
            )
            return f'    <polygon points="{points}" {style} />\n'

        elif elem_type == 'square':
            # Force aspect ratio (take min dimension)
            s = min(w, h)
            return f'    <rect x="{x}" y="{y}" width="{s}" height="{s}" {style} />\n'

        elif elem_type == 'cloud':
            # Approximated cloud shape using a path with multiple arcs
            cx = x + w // 2
            cy = y + h // 2
            r1 = min(w, h) // 4
            path_d = (
                f"M{x + r1},{cy} "
                f"a{r1},{r1} 0 0,1 {r1 * 2},0 "
                f"a{r1},{r1} 0 0,1 {r1 * 2},0 "
                f"a{r1},{r1} 0 0,1 0,{r1 * 2} "
                f"a{r1},{r1} 0 0,1 -{r1 * 4},0 "
                f"a{r1},{r1} 0 0,1 0,-{r1 * 2} Z"
            )
            return f'    <path d="{path_d}" {style} />\n'

        elif elem_type in ('3d_cube', '3d cube', 'isometric_box', 'isometric box'):
            # Isometric 3D cube: front face + top face + side face
            depth = min(w, h) // 4
            # Front face
            svg = (
                f'    <rect x="{x}" y="{y + depth}" width="{w - depth}" height="{h - depth}" '
                f'{style} />\n'
            )
            # Top face (parallelogram)
            top_pts = (
                f"{x},{y + depth} "
                f"{x + depth},{y} "
                f"{x + w},{y} "
                f"{x + w - depth},{y + depth}"
            )
            svg += f'    <polygon points="{top_pts}" {style} />\n'
            # Right face (parallelogram)
            right_pts = (
                f"{x + w - depth},{y + depth} "
                f"{x + w},{y} "
                f"{x + w},{y + h - depth} "
                f"{x + w - depth},{y + h}"
            )
            svg += f'    <polygon points="{right_pts}" {style} />\n'
            return svg

        elif elem_type in (
            'color_swatch', 'color swatch',
            'small_colored_square', 'small colored square',
        ):
            # Small colored square (legend swatch) — solid-filled rect
            return f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" {style} />\n'

        elif elem_type in (
            'stack_of_rectangles', 'stack of rectangles',
            'layered_boxes', 'layered boxes',
        ):
            # Render as 3 overlapping rectangles to suggest a stack
            offset_step = min(w, h) // 12
            svg = ""
            # Back rectangle (offset up-right)
            svg += (
                f'    <rect x="{x + 2 * offset_step}" y="{y}" '
                f'width="{w - 2 * offset_step}" height="{h - 2 * offset_step}" '
                f'{style} />\n'
            )
            # Middle rectangle
            svg += (
                f'    <rect x="{x + offset_step}" y="{y + offset_step}" '
                f'width="{w - 2 * offset_step}" height="{h - 2 * offset_step}" '
                f'{style} />\n'
            )
            # Front rectangle
            svg += (
                f'    <rect x="{x}" y="{y + 2 * offset_step}" '
                f'width="{w - 2 * offset_step}" height="{h - 2 * offset_step}" '
                f'{style} />\n'
            )
            return svg

        elif elem_type == 'cylinder':
            # Approximate cylinder as rect + two ellipses
            ry_cap = min(15, h // 6)
            svg = ""
            # Body
            svg += f'    <rect x="{x}" y="{y + ry_cap}" width="{w}" height="{h - 2 * ry_cap}" fill="{fill}" stroke="none" />\n'
            # Side lines
            svg += f'    <line x1="{x}" y1="{y + ry_cap}" x2="{x}" y2="{y + h - ry_cap}" stroke="{stroke}" stroke-width="{sw}" />\n'
            svg += f'    <line x1="{x + w}" y1="{y + ry_cap}" x2="{x + w}" y2="{y + h - ry_cap}" stroke="{stroke}" stroke-width="{sw}" />\n'
            # Top ellipse
            cx_e = x + w // 2
            svg += f'    <ellipse cx="{cx_e}" cy="{y + ry_cap}" rx="{w // 2}" ry="{ry_cap}" {style} />\n'
            # Bottom ellipse (only bottom half visible)
            svg += f'    <ellipse cx="{cx_e}" cy="{y + h - ry_cap}" rx="{w // 2}" ry="{ry_cap}" {style} />\n'
            return svg

        # Fallback: use polygon if available
        if element.polygon and len(element.polygon) >= 3:
            return self._polygon_shape_svg(element, offset)

        # Last resort: rectangle
        return f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" {style} />\n'

    def _polygon_shape_svg(
        self, element: ElementInfo, offset: Tuple[int, int] = (0, 0)
    ) -> str:
        """Render element using its polygon contour as an SVG path with smoothing."""
        fill = element.fill_color or "#ffffff"
        stroke = element.stroke_color or "#000000"
        sw = element.stroke_width or 1
        ox, oy = offset

        polygon = element.polygon
        if not polygon or len(polygon) < 3:
            # Fallback to bbox rect
            return self._geometric_shape_svg(element, offset)

        # Smooth the polygon using Chaikin's algorithm (1 iteration)
        smoothed = self._chaikin_smooth(polygon, iterations=2)

        # Build SVG path
        path_d = self._polygon_to_svg_path(smoothed, ox, oy)

        style = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
        return f'    <path d="{path_d}" {style} />\n'

    def _arrow_element_svg(
        self, element: ElementInfo, image: np.ndarray, offset: Tuple[int, int] = (0, 0)
    ) -> str:
        """Render arrow/line/connector as SVG polyline with arrowhead markers."""
        ox, oy = offset
        stroke = element.stroke_color or "#000000"
        sw = element.stroke_width or 2

        # If we have explicit start/end points, use them
        if element.arrow_start and element.arrow_end:
            x1 = element.arrow_start[0] - ox
            y1 = element.arrow_start[1] - oy
            x2 = element.arrow_end[0] - ox
            y2 = element.arrow_end[1] - oy
            return (
                f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="{sw}" '
                f'marker-end="url(#arrowhead)" />\n'
            )

        # If we have vector_points, use polyline
        if element.vector_points and len(element.vector_points) >= 2:
            points = " ".join(
                f"{p[0] - ox},{p[1] - oy}" for p in element.vector_points
            )
            return (
                f'    <polyline points="{points}" '
                f'fill="none" stroke="{stroke}" stroke-width="{sw}" '
                f'marker-end="url(#arrowhead)" />\n'
            )

        # Fallback: use polygon contour as path, or crop as raster
        if element.polygon and len(element.polygon) >= 3:
            # Use polygon outline (no fill) as the arrow shape
            path_d = self._polygon_to_svg_path(element.polygon, ox, oy)
            return (
                f'    <path d="{path_d}" '
                f'fill="{stroke}" stroke="{stroke}" stroke-width="1" />\n'
            )

        # Last fallback: embed as raster crop
        return self._raster_element_svg(element, image, offset)

    def _text_element_svg(
        self, element: ElementInfo, offset: Tuple[int, int] = (0, 0)
    ) -> str:
        """Render text element as SVG <text> (editable)."""
        bbox = element.bbox
        ox, oy = offset

        # Get text content from processing_notes or source_prompt
        text_content = ""
        for note in element.processing_notes:
            if note.startswith("text:"):
                text_content = note[5:].strip()
                break
        if not text_content and element.source_prompt:
            text_content = element.source_prompt

        # Position at center of bbox
        x = bbox.x1 - ox + bbox.width // 2
        y = bbox.y1 - oy + bbox.height // 2

        # Estimate font size from bbox height
        font_size = max(8, min(48, int(bbox.height * 0.7)))

        fill = element.fill_color or "#000000"
        escaped_text = xml_escape(text_content)

        return (
            f'    <text x="{x}" y="{y}" '
            f'font-family="{DEFAULT_FONT_FAMILY}" font-size="{font_size}" '
            f'fill="{fill}" text-anchor="middle" dominant-baseline="central">'
            f'{escaped_text}</text>\n'
        )

    def _raster_element_svg(
        self, element: ElementInfo, image: np.ndarray, offset: Tuple[int, int] = (0, 0)
    ) -> str:
        """Embed raster crop as base64 <image> inside SVG."""
        bbox = element.bbox
        ox, oy = offset

        # Use existing base64 if available (from IconPictureProcessor)
        if element.base64:
            b64_data = element.base64
        else:
            # Crop and encode
            pil_img = self.crop_raster_element(element, image)
            if pil_img is None:
                # Empty placeholder
                return f'    <rect x="{bbox.x1 - ox}" y="{bbox.y1 - oy}" width="{bbox.width}" height="{bbox.height}" fill="#cccccc" stroke="#999999" />\n'
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64_data = base64.b64encode(buf.getvalue()).decode("ascii")

        x = bbox.x1 - ox
        y = bbox.y1 - oy

        return (
            f'    <image x="{x}" y="{y}" width="{bbox.width}" height="{bbox.height}" '
            f'href="data:image/png;base64,{b64_data}" />\n'
        )

    # ================================================================
    #  PRIVATE: Geometry Helpers
    # ================================================================

    @staticmethod
    def _chaikin_smooth(
        polygon: List[List[int]], iterations: int = 2
    ) -> List[List[float]]:
        """
        Chaikin's corner-cutting algorithm for polygon smoothing.
        Each iteration doubles the point count and rounds corners.
        """
        pts = [list(map(float, p)) for p in polygon]
        for _ in range(iterations):
            if len(pts) < 3:
                break
            new_pts = []
            n = len(pts)
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                # Q = 3/4 * P_i + 1/4 * P_{i+1}
                q = [0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]]
                # R = 1/4 * P_i + 3/4 * P_{i+1}
                r = [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]
                new_pts.append(q)
                new_pts.append(r)
            pts = new_pts
        return pts

    @staticmethod
    def _polygon_to_svg_path(
        polygon: List, ox: int = 0, oy: int = 0
    ) -> str:
        """Convert polygon points to SVG path d attribute string."""
        if not polygon:
            return ""

        parts = []
        for i, pt in enumerate(polygon):
            x = pt[0] - ox
            y = pt[1] - oy
            cmd = "M" if i == 0 else "L"
            parts.append(f"{cmd}{x:.1f},{y:.1f}")
        parts.append("Z")
        return " ".join(parts)

    @staticmethod
    def _image_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
        """Convert PIL image to base64 string."""
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("ascii")
