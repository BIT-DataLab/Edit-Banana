"""
Text restorer — main interface for diagram text to draw.io XML.

Converts text and formulas in diagram images to draw.io XML.

Usage:
    from modules.text import TextRestorer
    restorer = TextRestorer()
    xml_string = restorer.process("input.png")
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

from .ocr.local_ocr import LocalOCR
from .coord_processor import CoordProcessor
from .xml_generator import MxGraphXMLGenerator

from .processors.font_size import FontSizeProcessor
from .processors.font_family import FontFamilyProcessor
from .processors.style import StyleProcessor
from .processors.formula import FormulaProcessor


class TextRestorer:
    """
    Coordinates OCR, processors, and XML output for text restoration.
    Default: Tesseract OCR + optional Pix2Text for formulas.
    """

    def __init__(
        self,
        formula_engine: str = "pix2text",
        ocr_engine: str = "tesseract",
    ):
        """
        Args:
            formula_engine: Formula engine ('pix2text', 'none').
            ocr_engine: Layout/text OCR engine ('tesseract', 'paddleocr'). PaddleOCR often better for mixed CN/EN.
        """
        self.formula_engine = formula_engine
        self._ocr_engine = (ocr_engine or "tesseract").strip().lower()

        self._layout_ocr = None
        self._pix2text_ocr = None

        self.font_size_processor = FontSizeProcessor()
        self.font_family_processor = FontFamilyProcessor()
        self.style_processor = StyleProcessor()
        self.formula_processor = FormulaProcessor()

        self.timing = {
            "text_ocr": 0.0,
            "pix2text_ocr": 0.0,
            "processing": 0.0,
            "total": 0.0,
        }

    @property
    def layout_ocr(self):
        """Lazy-init layout OCR (tesseract or paddleocr); fallback to Tesseract if PaddleOCR fails."""
        if self._layout_ocr is None:
            if self._ocr_engine == "paddleocr":
                try:
                    from .ocr.paddle_ocr import PaddleOCRAdapter
                    self._layout_ocr = PaddleOCRAdapter()
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"PaddleOCR unavailable ({e}), falling back to Tesseract. See README for compatible install.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._layout_ocr = LocalOCR()
            else:
                self._layout_ocr = LocalOCR()
        return self._layout_ocr

    @property
    def pix2text_ocr(self):
        """Lazy-init Pix2Text OCR (None if pix2text not installed)."""
        from .ocr import Pix2TextOCR
        if Pix2TextOCR is None:
            return None
        if self._pix2text_ocr is None:
            self._pix2text_ocr = Pix2TextOCR()
        return self._pix2text_ocr
    
    def process(self, image_path: str) -> str:
        """
        Process image and return draw.io XML string.

        Args:
            image_path: Input image path.
        Returns:
            draw.io XML string.
        """
        image_path = Path(image_path)

        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        text_blocks = self.process_image(str(image_path))

        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for block in text_blocks:
            geo = block["geometry"]
            cell = generator.create_text_cell(
                text=block["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),
                height=max(geo["height"], 10),
                font_size=block.get("font_size", 12),
                is_latex=block.get("is_latex", False),
                rotation=geo.get("rotation", 0),
                font_weight=block.get("font_weight"),
                font_style=block.get("font_style"),
                font_color=block.get("font_color"),
                font_family=block.get("font_family")
            )
            text_cells.append(cell)
        
        return generator.generate_xml(text_cells)
    
    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process image and return list of text blocks."""
        total_start = time.time()
        image_path = Path(image_path)

        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # Step 1: OCR
        ocr_result, formula_result = self._run_ocr(str(image_path))

        # Step 2: Formula (layout OCR + Pix2Text)
        processing_start = time.time()

        if formula_result:
            logger.info("Formula refinement...")
            merged_blocks = self.formula_processor.merge_ocr_results(ocr_result, formula_result)
            text_blocks = self.formula_processor.to_dict_list(merged_blocks)
        else:
            text_blocks = self._ocr_result_to_dict_list(ocr_result)
        
        logger.info(f"   {len(text_blocks)} text blocks")

        # Step 3: Coord transform
        logger.info("Coord transform...")
        coord_processor = CoordProcessor(
            source_width=image_width,
            source_height=image_height
        )
        
        for block in text_blocks:
            polygon = block.get("polygon", [])
            if polygon:
                geometry = coord_processor.polygon_to_geometry(polygon)
                block["geometry"] = geometry
            else:
                block["geometry"] = {"x": 0, "y": 0, "width": 100, "height": 20, "rotation": 0}
        
        # Step 4: Font size
        logger.info("Font size...")
        text_blocks = self.font_size_processor.process(text_blocks)
        
        # Step 5: Font family
        logger.info("Font family...")
        global_font = self._detect_global_font(ocr_result)
        text_blocks = self.font_family_processor.process(text_blocks, global_font=global_font)

        # Step 6: Style (bold/color)
        logger.info("Style...")
        ocr_styles = getattr(ocr_result, "styles", [])
        text_blocks = self.style_processor.process(text_blocks, ocr_styles=ocr_styles)
        
        self.timing["processing"] = time.time() - processing_start
        self.timing["total"] = time.time() - total_start
        
        return text_blocks
    
    def restore(
        self,
        image_path: str,
        output_path: str = None,
        save_metadata: bool = True,
        save_debug_image: bool = True
    ) -> str:
        """Full pipeline: process image and write draw.io file."""
        image_path = Path(image_path)

        if output_path is None:
            output_path = image_path.with_suffix(".drawio")
        else:
            output_path = Path(output_path)
        
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        logger.info(f"Input: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Size: {image_width} x {image_height}")

        text_blocks = self.process_image(str(image_path))

        logger.info("Generating XML...")
        xml_start = time.time()
        
        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for block in text_blocks:
            geo = block["geometry"]
            cell = generator.create_text_cell(
                text=block["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),
                height=max(geo["height"], 10),
                font_size=block.get("font_size", 12),
                is_latex=block.get("is_latex", False),
                rotation=geo.get("rotation", 0),
                font_weight=block.get("font_weight"),
                font_style=block.get("font_style"),
                font_color=block.get("font_color"),
                font_family=block.get("font_family")
            )
            text_cells.append(cell)
        
        generator.save_to_file(text_cells, str(output_path))
        
        xml_time = time.time() - xml_start
        self.timing["total"] += xml_time
        
        # Save metadata
        if save_metadata:
            self._save_metadata(str(image_path), str(output_path), text_blocks, image_width, image_height)
        
        # Debug image
        if save_debug_image:
            debug_path = output_path.parent / "debug.png"
            self._generate_debug_image(str(image_path), str(debug_path))
        
        # Stats
        self._print_stats(text_blocks)
        
        return str(output_path)
    
    def _run_ocr(self, image_path: str):
        """Run OCR (Tesseract or PaddleOCR + optional Pix2Text for formulas)."""
        engine_label = "PaddleOCR" if self._ocr_engine == "paddleocr" else "Tesseract"
        logger.info(f"Text OCR ({engine_label})...")
        text_start = time.time()
        try:
            ocr_result = self.layout_ocr.analyze_image(image_path)
        except Exception as e:
            if self._ocr_engine == "paddleocr":
                import warnings
                warnings.warn(
                    f"PaddleOCR inference failed ({e!r}), falling back to Tesseract.",
                    UserWarning,
                    stacklevel=2,
                )
                self._layout_ocr = LocalOCR()
                ocr_result = self._layout_ocr.analyze_image(image_path)
            else:
                raise
        self.timing["text_ocr"] = time.time() - text_start
        logger.info(f"   {len(ocr_result.text_blocks)} text blocks ({self.timing['text_ocr']:.2f}s)")

        formula_result = None

        if self.formula_engine == "pix2text" and self.pix2text_ocr is not None:
            logger.info("Formula refinement...")
            refine_start = time.time()
            fixed_count = 0

            processed_indices = set()
            new_blocks_map = {}
            indices_to_remove = set()

            blocks = ocr_result.text_blocks
            i = 0
            while i < len(blocks):
                if i in processed_indices:
                    i += 1
                    continue
                
                # Current block
                curr_block = blocks[i]
                curr_poly = curr_block.polygon
                
                # Worth refining?
                if not self._should_refine_block(curr_block.text):
                    i += 1
                    continue
                
                # Look ahead for merge
                group_indices = [i]
                group_polygon = curr_poly
                
                j = i + 1
                while j < len(blocks):
                    next_block = blocks[j]
                    
                    # Distance check
                    if self._is_spatially_close(group_polygon, next_block.polygon):
                        if self._should_refine_block(next_block.text): 
                            group_indices.append(j)
                            group_polygon = self._merge_polygons(group_polygon, next_block.polygon)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                # Final region
                target_polygon = group_polygon
                
                # Call Pix2Text
                latex_text = self.pix2text_ocr.recognize_region(image_path, target_polygon)
                
                if latex_text and self.formula_processor.is_valid_formula(latex_text):
                    original_text_combined = " ".join([blocks[k].text for k in group_indices])
                    
                    if self._is_refinement_meaningful(original_text_combined, latex_text):
                        cleaned_latex = self.formula_processor.clean_latex(latex_text)
                        
                        import copy
                        new_block = copy.deepcopy(curr_block)
                        new_block.text = f"${cleaned_latex}$"
                        new_block.is_latex = True
                        new_block.polygon = target_polygon
                        new_block.font_family = "Latin Modern Math"
                        
                        if len(group_indices) > 1:
                            logger.info(f"   Refine [Merge {group_indices}]: '{original_text_combined}' -> '${cleaned_latex}$'")
                            indices_to_remove.update(group_indices)
                            new_blocks_map[i] = new_block
                        else:
                            logger.info(f"   Refine [{i}]: '{curr_block.text}' -> '${cleaned_latex}$'")
                            curr_block.text = f"${cleaned_latex}$"
                            curr_block.is_latex = True
                            curr_block.font_family = "Latin Modern Math"
                            fixed_count += 1
                        
                        processed_indices.update(group_indices)
                        i = j
                        continue
                
                i += 1
            
            if indices_to_remove:
                final_blocks = []
                for idx, block in enumerate(blocks):
                    if idx in new_blocks_map:
                        final_blocks.append(new_blocks_map[idx])
                        fixed_count += 1
                    elif idx not in indices_to_remove:
                        final_blocks.append(block)
                ocr_result.text_blocks = final_blocks

            self.timing["pix2text_ocr"] = time.time() - refine_start
            logger.info(f"   Refined {fixed_count} formula blocks ({self.timing['pix2text_ocr']:.2f}s)")

            formula_result = None

        elif self.formula_engine == "pix2text" and self.pix2text_ocr is None:
            logger.info("Skipping formula (pix2text not installed)")
        else:
            logger.info("Skipping formula")

        return ocr_result, formula_result

    def _should_refine_block(self, text: str) -> bool:
        """Whether to try refinement."""
        if not text: return False
        
        if '?' in text or '？' in text or '(?)' in text:
            return True
        
        words = text.split()
        if len(words) > 8: return False
        
        import re
        if re.match(r'^[a-zA-Z\s\-,.:!\\\'"]+$', text):
            if len(text) < 4: 
                return True
            return False 
            
        return True

    def _is_refinement_meaningful(self, original: str, new_latex: str) -> bool:
        """Whether refinement changed result meaningfully."""
        import re
        
        core_latex = re.sub(r'\\(mathbf|mathrm|textit|text|boldsymbol|mathcal|mathscr)\{([^\}]+)\}', r'\2', new_latex)
        core_latex = re.sub(r'\s|~', '', core_latex)
        core_latex = core_latex.replace('$', '')
        
        core_original = re.sub(r'\s', '', original)
        
        if core_latex == core_original:
            return False
            
        return True

    def _is_spatially_close(self, poly1, poly2) -> bool:
        """Whether two polygons are spatially close."""
        def get_bbox(p):
            xs, ys = [pt[0] for pt in p], [pt[1] for pt in p]
            return min(xs), min(ys), max(xs), max(ys)
        
        x1_min, y1_min, x1_max, y1_max = get_bbox(poly1)
        x2_min, y2_min, x2_max, y2_max = get_bbox(poly2)
        
        h1, h2 = y1_max - y1_min, y2_max - y2_min
        ref_h = max(h1, h2)
        
        y_overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
        is_y_aligned = y_overlap > -ref_h * 0.5 
        
        if is_y_aligned:
            x_dist = max(0, x2_min - x1_max) if x1_min < x2_min else max(0, x1_min - x2_max)
            if x_dist < ref_h * 1.2:
                h_ratio = min(h1, h2) / max(h1, h2)
                if h_ratio > 0.6:
                    return True

        x_overlap = min(x1_max, x2_max) - max(x1_min, x2_min)
        wmin = min(x1_max - x1_min, x2_max - x2_min)
        
        if x_overlap > wmin * 0.2: 
            y_dist = max(0, y2_min - y1_max) if y1_min < y2_min else max(0, y1_min - y2_max)
            if y_dist < ref_h * 0.5:
                return True
                
        return False

    def _merge_polygons(self, poly1, poly2):
        """Merge two polygons."""
        xs = [p[0] for p in poly1] + [p[0] for p in poly2]
        ys = [p[1] for p in poly1] + [p[1] for p in poly2]
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    
    def _ocr_result_to_dict_list(self, ocr_result) -> List[Dict[str, Any]]:
        """Convert OCR result to list of dicts."""
        return [
            {
                "text": block.text,
                "polygon": block.polygon,
                "confidence": getattr(block, "confidence", 1.0),
                "font_size_px": block.font_size_px,
                "is_latex": getattr(block, "is_latex", False),
                "font_family": getattr(block, "font_family", getattr(block, "font_name", None)),
                "font_weight": getattr(block, "font_weight", None),
                "font_style": getattr(block, "font_style", None),
                "font_color": getattr(block, "font_color", None),
                "is_bold": getattr(block, "is_bold", False),
                "is_italic": getattr(block, "is_italic", False),
                "spans": getattr(block, "spans", []),
            }
            for block in ocr_result.text_blocks
        ]

    def _detect_global_font(self, ocr_result) -> str:
        """Detect global dominant font."""
        if not ocr_result.text_blocks:
            return "Arial"

        def get_area(block):
            polygon = block.polygon
            if not polygon or len(polygon) < 4:
                return 0
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        best_block = max(ocr_result.text_blocks, key=get_area)
        font = getattr(best_block, "font_name", None)
        
        if font:
            logger.info(f"   Dominant font: {font}")
            return font
        
        return "Arial"
    
    def _save_metadata(self, image_path: str, output_path: str, text_blocks: List[Dict], 
                       image_width: int, image_height: int):
        """Save metadata."""
        import json
        from datetime import datetime
        
        metadata_path = Path(output_path).parent / "metadata.json"
        
        font_stats = {}
        for block in text_blocks:
            font = block.get("font_family", "unknown")
            font_stats[font] = font_stats.get(font, 0) + 1
        
        metadata = {
            "version": "3.0",
            "generated_at": datetime.now().isoformat(),
            "input": {"path": image_path, "width": image_width, "height": image_height},
            "output": {"drawio_path": output_path},
            "mode": f"local+{self.formula_engine}",
            "timing": self.timing,
            "statistics": {
                "total_cells": len(text_blocks),
                "text_cells": sum(1 for b in text_blocks if not b.get("is_latex")),
                "formula_cells": sum(1 for b in text_blocks if b.get("is_latex")),
                "fonts": font_stats
            },
            "text_blocks": [
                {
                    "id": i + 1,
                    "text": block["text"][:100],
                    "position": block["geometry"],
                    "style": {
                        "font_size": block.get("font_size"),
                        "font_family": block.get("font_family"),
                        "font_weight": block.get("font_weight"),
                        "font_color": block.get("font_color"),
                        "is_formula": block.get("is_latex", False)
                    }
                }
                for i, block in enumerate(text_blocks)
            ]
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   Metadata saved: {metadata_path}")
    
    def _generate_debug_image(self, image_path: str, output_path: str):
        """Generate debug image."""
        try:
            # Simple: copy image as debug
            from PIL import Image
            img = Image.open(image_path)
            img.save(output_path)
        except Exception as e:
            logger.warning(f"   Debug image failed: {e}")
    
    def _print_stats(self, text_blocks: List[Dict]):
        """Print stats."""
        logger.info(f"Time: Text OCR {self.timing['text_ocr']:.2f}s, "
                    f"Pix2Text {self.timing['pix2text_ocr']:.2f}s, "
                    f"Processing {self.timing['processing']:.2f}s, "
                    f"Total {self.timing['total']:.2f}s")
        logger.info(f"Done: {len(text_blocks)} text cells")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python restorer.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    restorer = TextRestorer()
    restorer.restore(image_path, output_path)
