"""
Stage 8: Vector Exporter — Orchestrator Module

Coordinates SVG generation, raster cropping, section detection, PDF combination,
and manifest generation. This is the main entry point for the vector export pipeline.

Integrates into Edit Banana as Stage 8 (runs after XML Merge in Stage 7).

Reads from ProcessingContext:
    - context.elements (all detected ElementInfo objects with bbox, polygon, type, colors)
    - context.image_path (original image for raster cropping)
    - context.canvas_width / canvas_height
    - context.output_dir (base output directory for this image)

Produces:
    output/{image}/vectors/
        combined/      -> combined.svg + combined.pdf
        elements/      -> individual element SVGs
        rasters/       -> cropped PNGs for raster elements
        sections/      -> section-level SVGs (on demand)
        components/    -> grouped element SVGs (on demand)
        manifest.json  -> element index

Usage:
    from modules.vector_exporter import VectorExporter

    exporter = VectorExporter()
    result = exporter.process(context)

CLI flags (handled by main.py, passed via context.intermediate_results):
    --vector-level=granular|section|component|all (default: granular)
    --no-vectors (skip vector export entirely)
"""

import os
import json
import re
import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base import BaseProcessor, ProcessingContext
from .data_types import (
    ElementInfo,
    ProcessingResult,
    LayerLevel,
)
from .svg_generator import SVGGenerator, RASTER_TYPES, ARROW_TYPES, GEOMETRIC_SHAPES
from .pdf_combiner import PDFCombiner
from .section_detector import SectionDetector, Section


# Valid vector-level options
VECTOR_LEVELS = {"granular", "section", "component", "all"}
DEFAULT_VECTOR_LEVEL = "granular"


class VectorExporter(BaseProcessor):
    """
    Stage 8 orchestrator: exports editable vector assets from pipeline results.

    Inherits BaseProcessor for consistent interface with other pipeline stages.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._svg_gen = SVGGenerator()
        self._pdf_combiner = PDFCombiner()
        self._section_detector = SectionDetector()

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        Main entry point — called by Pipeline after Stage 7 (XML Merge).

        Reads vector_level from context.intermediate_results['vector_level'].
        Default is 'granular' (individual element SVGs + combined SVG/PDF).
        """
        self._log("Starting vector export...")

        # Get configuration from context
        vector_level = context.intermediate_results.get(
            "vector_level", DEFAULT_VECTOR_LEVEL
        )
        if vector_level not in VECTOR_LEVELS:
            self._log(f"Unknown vector level '{vector_level}', using 'granular'")
            vector_level = DEFAULT_VECTOR_LEVEL

        # Load image
        image = cv2.imread(context.image_path)
        if image is None:
            return ProcessingResult(
                success=False,
                error_message=f"Could not read image: {context.image_path}",
            )

        # Create output directories
        vectors_dir = os.path.join(context.output_dir, "vectors")
        dirs = self._create_output_dirs(vectors_dir, vector_level)

        # Get canvas dimensions
        canvas_w = context.canvas_width or image.shape[1]
        canvas_h = context.canvas_height or image.shape[0]

        # Filter to elements that have been processed
        elements = context.elements
        if not elements:
            self._log("No elements to export")
            return ProcessingResult(
                success=True,
                metadata={"exported_count": 0, "vector_dir": vectors_dir},
            )

        self._log(f"Exporting {len(elements)} elements (level={vector_level})")

        # Track manifest entries
        manifest_entries = []

        # ============================================================
        # GRANULAR: always runs — individual element SVGs + raster PNGs
        # ============================================================
        self._log("Exporting individual elements...")
        for elem in elements:
            entry = self._export_element(
                elem, image, dirs["elements"], dirs["rasters"]
            )
            if entry:
                manifest_entries.append(entry)

        self._log(f"  Exported {len(manifest_entries)} elements")

        # ============================================================
        # COMBINED SVG + PDF: always generated
        # ============================================================
        self._log("Generating combined SVG...")
        combined_svg = self._svg_gen.generate_combined_svg(
            elements, image, canvas_w, canvas_h
        )
        combined_svg_path = os.path.join(dirs["combined"], "combined.svg")
        with open(combined_svg_path, "w", encoding="utf-8") as f:
            f.write(combined_svg)
        self._log(f"  Saved: {combined_svg_path}")

        # PDF
        self._log("Generating combined PDF...")
        combined_pdf_path = os.path.join(dirs["combined"], "combined.pdf")
        if self._pdf_combiner.is_available:
            pdf_ok = self._pdf_combiner.svg_string_to_pdf(
                combined_svg, combined_pdf_path
            )
            if not pdf_ok:
                self._log("  PDF generation failed (SVG may have unsupported features)")
        else:
            self._log(
                "  PDF generation skipped (install svglib+reportlab: "
                "pip install svglib reportlab)"
            )

        # ============================================================
        # SECTION LEVEL: on demand
        # ============================================================
        sections = []
        if vector_level in ("section", "all"):
            self._log("Detecting sections...")
            sections = self._section_detector.detect_sections(context)
            if sections:
                self._export_sections(
                    sections, elements, image, dirs["sections"], canvas_w, canvas_h
                )
            else:
                self._log("  No sections detected")

        # ============================================================
        # COMPONENT LEVEL: on demand (smart grouping)
        # ============================================================
        if vector_level in ("component", "all"):
            self._log("Generating component groups...")
            self._export_components(
                elements, image, dirs["components"], canvas_w, canvas_h
            )

        # ============================================================
        # MANIFEST
        # ============================================================
        manifest = self._build_manifest(
            context, manifest_entries, sections, vector_level, canvas_w, canvas_h
        )
        manifest_path = os.path.join(vectors_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        self._log(f"Manifest saved: {manifest_path}")

        self._log("Vector export complete.")

        return ProcessingResult(
            success=True,
            elements=elements,
            metadata={
                "exported_count": len(manifest_entries),
                "vector_dir": vectors_dir,
                "combined_svg": combined_svg_path,
                "combined_pdf": combined_pdf_path
                if os.path.exists(combined_pdf_path)
                else None,
                "sections_count": len(sections),
                "manifest_path": manifest_path,
            },
        )

    # ================================================================
    #  Element Export
    # ================================================================

    def _export_element(
        self,
        element: ElementInfo,
        image: np.ndarray,
        elements_dir: str,
        rasters_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Export a single element as SVG + optional raster PNG."""
        elem_type = element.element_type.lower().replace(" ", "_")
        elem_id = f"{elem_type}_{element.id:03d}"

        # Generate SVG
        try:
            svg_string = self._svg_gen.element_to_svg(element, image, standalone=True)
        except Exception as e:
            self._log(f"  Failed to generate SVG for {elem_id}: {e}")
            return None

        # Save SVG
        svg_filename = f"{elem_id}.svg"
        svg_path = os.path.join(elements_dir, svg_filename)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_string)

        # Save raster PNG for image-type elements
        raster_path = None
        raster_filename = None
        if elem_type in RASTER_TYPES or element.base64:
            raster_filename = f"{elem_id}.png"
            raster_full_path = os.path.join(rasters_dir, raster_filename)
            saved = self._svg_gen.save_raster_crop(element, image, raster_full_path)
            if saved:
                raster_path = f"rasters/{raster_filename}"

        # Build manifest entry
        return {
            "id": elem_id,
            "element_id": element.id,
            "type": element.element_type,
            "svg_path": f"elements/{svg_filename}",
            "raster_path": raster_path,
            "bbox": {
                "x": element.bbox.x1,
                "y": element.bbox.y1,
                "w": element.bbox.width,
                "h": element.bbox.height,
            },
            "colors": {
                "fill": element.fill_color,
                "stroke": element.stroke_color,
            },
            "score": round(element.score, 4),
            "layer": LayerLevel(element.layer_level).name
            if element.layer_level in [lv.value for lv in LayerLevel]
            else "OTHER",
        }

    # ================================================================
    #  Section Export
    # ================================================================

    def _export_sections(
        self,
        sections: List[Section],
        elements: List[ElementInfo],
        image: np.ndarray,
        sections_dir: str,
        canvas_w: int,
        canvas_h: int,
    ):
        """Export each section as its own SVG with child elements."""
        # Build element lookup by ID
        elem_by_id = {e.id: e for e in elements}

        for section in sections:
            section_label = f"section_{section.label}"
            section_subdir = os.path.join(sections_dir, section_label)
            os.makedirs(section_subdir, exist_ok=True)

            # Get child elements
            child_elements = [
                elem_by_id[eid]
                for eid in section.child_element_ids
                if eid in elem_by_id
            ]

            if not child_elements:
                continue

            # Generate section SVG (elements positioned relative to section bbox)
            section_svg = self._svg_gen.generate_combined_svg(
                child_elements,
                image,
                section.bbox.width,
                section.bbox.height,
            )

            # Adjust viewBox to section coordinates
            section_svg = section_svg.replace(
                f'viewBox="0 0 {section.bbox.width} {section.bbox.height}"',
                f'viewBox="{section.bbox.x1} {section.bbox.y1} '
                f'{section.bbox.width} {section.bbox.height}"',
            )

            svg_path = os.path.join(section_subdir, f"{section_label}.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(section_svg)

            # Also export individual elements within section
            elements_subdir = os.path.join(section_subdir, "elements")
            os.makedirs(elements_subdir, exist_ok=True)
            for elem in child_elements:
                elem_type = elem.element_type.lower().replace(" ", "_")
                elem_filename = f"{elem_type}_{elem.id:03d}.svg"
                try:
                    elem_svg = self._svg_gen.element_to_svg(
                        elem, image, standalone=True
                    )
                    with open(
                        os.path.join(elements_subdir, elem_filename), "w", encoding="utf-8"
                    ) as f:
                        f.write(elem_svg)
                except Exception:
                    pass

            self._log(
                f"  Section {section.label}: {len(child_elements)} elements -> {svg_path}"
            )

    # ================================================================
    #  Component Export (Smart Grouping)
    # ================================================================

    def _export_components(
        self,
        elements: List[ElementInfo],
        image: np.ndarray,
        components_dir: str,
        canvas_w: int,
        canvas_h: int,
    ):
        """
        Group related elements into components and export each as SVG.

        Grouping heuristics:
            - Shape + overlapping text -> "labeled box" component
            - Sequential arrows -> "arrow chain" component
            - Nearby icons/pictures -> "icon group" component
        """
        used_ids = set()
        component_id = 0

        # Build spatial index
        shapes = [
            e for e in elements
            if e.element_type.lower() in GEOMETRIC_SHAPES
        ]
        texts = [
            e for e in elements
            if e.element_type.lower() == "text"
        ]
        arrows = [
            e for e in elements
            if e.element_type.lower() in ARROW_TYPES
        ]
        others = [
            e for e in elements
            if e.id not in {s.id for s in shapes}
            and e.id not in {t.id for t in texts}
            and e.id not in {a.id for a in arrows}
        ]

        # Heuristic 1: Shape + contained text = labeled box
        for shape in shapes:
            if shape.id in used_ids:
                continue

            group = [shape]
            used_ids.add(shape.id)

            # Find text elements whose center falls within this shape's bbox
            for text in texts:
                if text.id in used_ids:
                    continue
                tcx, tcy = text.bbox.center
                if (
                    shape.bbox.x1 <= tcx <= shape.bbox.x2
                    and shape.bbox.y1 <= tcy <= shape.bbox.y2
                ):
                    group.append(text)
                    used_ids.add(text.id)

            if len(group) >= 1:
                self._save_component_group(
                    group, image, components_dir, component_id, "labeled_box"
                )
                component_id += 1

        # Heuristic 2: Individual arrows
        for arrow in arrows:
            if arrow.id in used_ids:
                continue
            self._save_component_group(
                [arrow], image, components_dir, component_id, "arrow"
            )
            used_ids.add(arrow.id)
            component_id += 1

        # Heuristic 3: Remaining elements as individual components
        for elem in others + [t for t in texts if t.id not in used_ids]:
            if elem.id in used_ids:
                continue
            elem_type = elem.element_type.lower().replace(" ", "_")
            self._save_component_group(
                [elem], image, components_dir, component_id, elem_type
            )
            used_ids.add(elem.id)
            component_id += 1

        self._log(f"  Exported {component_id} components")

    def _save_component_group(
        self,
        elements: List[ElementInfo],
        image: np.ndarray,
        components_dir: str,
        component_id: int,
        component_type: str,
    ):
        """Save a group of elements as a single component SVG."""
        if not elements:
            return

        # Calculate bounding box of all elements in group
        x1 = min(e.bbox.x1 for e in elements)
        y1 = min(e.bbox.y1 for e in elements)
        x2 = max(e.bbox.x2 for e in elements)
        y2 = max(e.bbox.y2 for e in elements)
        w = x2 - x1 + 4  # padding
        h = y2 - y1 + 4

        # Generate combined SVG for this group
        svg = self._svg_gen.generate_combined_svg(elements, image, w, h)

        # Adjust viewBox to component coordinates
        svg = svg.replace(
            f'viewBox="0 0 {w} {h}"',
            f'viewBox="{x1 - 2} {y1 - 2} {w} {h}"',
        )

        filename = f"component_{component_id:03d}_{component_type}.svg"
        filepath = os.path.join(components_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg)

    # ================================================================
    #  Manifest
    # ================================================================

    def _build_manifest(
        self,
        context: ProcessingContext,
        elements: List[Dict],
        sections: List[Section],
        vector_level: str,
        canvas_w: int,
        canvas_h: int,
    ) -> Dict[str, Any]:
        """Build manifest.json content."""
        return {
            "source_image": os.path.basename(context.image_path),
            "image_size": {"width": canvas_w, "height": canvas_h},
            "extraction_date": datetime.now().isoformat(),
            "vector_level": vector_level,
            "levels_generated": self._levels_generated(vector_level),
            "total_elements": len(elements),
            "elements": elements,
            "sections": [s.to_dict() for s in sections] if sections else [],
        }

    @staticmethod
    def _levels_generated(vector_level: str) -> List[str]:
        if vector_level == "all":
            return ["granular", "section", "component"]
        return [vector_level]

    # ================================================================
    #  Directory Setup
    # ================================================================

    def _create_output_dirs(
        self, vectors_dir: str, vector_level: str
    ) -> Dict[str, str]:
        """Create output directory structure."""
        dirs = {
            "root": vectors_dir,
            "combined": os.path.join(vectors_dir, "combined"),
            "elements": os.path.join(vectors_dir, "elements"),
            "rasters": os.path.join(vectors_dir, "rasters"),
        }

        if vector_level in ("section", "all"):
            dirs["sections"] = os.path.join(vectors_dir, "sections")

        if vector_level in ("component", "all"):
            dirs["components"] = os.path.join(vectors_dir, "components")

        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        return dirs


