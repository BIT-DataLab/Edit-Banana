"""
Stage 8b: PDF Combiner Module

Converts a combined SVG (or set of individual SVGs) into a single layered PDF.
Uses svglib + reportlab for pure-Python PDF generation (no system dependencies).

Fallback: if svglib is not installed, attempts CairoSVG. If neither is available,
skips PDF generation with a warning.

Usage:
    from modules.pdf_combiner import PDFCombiner

    combiner = PDFCombiner()
    combiner.svg_to_pdf("combined.svg", "output.pdf")
"""

import os
import warnings
from typing import Optional

# Try to import svglib + reportlab (primary)
_SVGLIB_AVAILABLE = False
_CAIROSVG_AVAILABLE = False

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import letter
    _SVGLIB_AVAILABLE = True
except ImportError:
    pass

if not _SVGLIB_AVAILABLE:
    try:
        import cairosvg
        _CAIROSVG_AVAILABLE = True
    except ImportError:
        pass


class PDFCombiner:
    """
    Converts SVG files to PDF.

    Priority:
        1. svglib + reportlab (pure Python, recommended)
        2. CairoSVG (requires cairo system library)
        3. Skip with warning

    Methods:
        svg_to_pdf(svg_path, pdf_path) -> bool
        svg_string_to_pdf(svg_string, pdf_path) -> bool
    """

    def __init__(self):
        self._log_prefix = "[PDFCombiner]"
        self._backend = self._detect_backend()

    def _log(self, msg: str):
        print(f"{self._log_prefix} {msg}")

    def _detect_backend(self) -> str:
        if _SVGLIB_AVAILABLE:
            return "svglib"
        elif _CAIROSVG_AVAILABLE:
            return "cairosvg"
        else:
            return "none"

    @property
    def is_available(self) -> bool:
        """Check if any PDF backend is available."""
        return self._backend != "none"

    def svg_to_pdf(self, svg_path: str, pdf_path: str) -> bool:
        """
        Convert an SVG file to PDF.

        Args:
            svg_path: Path to input SVG file
            pdf_path: Path to output PDF file

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(svg_path):
            self._log(f"SVG file not found: {svg_path}")
            return False

        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        if self._backend == "svglib":
            return self._svglib_convert(svg_path, pdf_path)
        elif self._backend == "cairosvg":
            return self._cairosvg_convert(svg_path, pdf_path)
        else:
            self._log(
                "No PDF backend available. Install svglib+reportlab: "
                "pip install svglib reportlab"
            )
            return False

    def svg_string_to_pdf(self, svg_string: str, pdf_path: str) -> bool:
        """
        Convert an SVG string to PDF by writing to temp file first.

        Args:
            svg_string: SVG content as string
            pdf_path: Path to output PDF file

        Returns:
            True if successful, False otherwise
        """
        # Write SVG to temporary file next to output
        tmp_svg = pdf_path.replace(".pdf", "_tmp.svg")
        try:
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(tmp_svg, "w", encoding="utf-8") as f:
                f.write(svg_string)
            result = self.svg_to_pdf(tmp_svg, pdf_path)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(tmp_svg):
                try:
                    os.remove(tmp_svg)
                except OSError:
                    pass

    # ================================================================
    #  Backend Implementations
    # ================================================================

    def _svglib_convert(self, svg_path: str, pdf_path: str) -> bool:
        """Convert SVG to PDF using svglib + reportlab."""
        try:
            drawing = svg2rlg(svg_path)
            if drawing is None:
                self._log(f"svglib could not parse: {svg_path}")
                return False
            renderPDF.drawToFile(drawing, pdf_path)
            self._log(f"PDF created (svglib): {pdf_path}")
            return True
        except Exception as e:
            self._log(f"svglib conversion failed: {e}")
            return False

    def _cairosvg_convert(self, svg_path: str, pdf_path: str) -> bool:
        """Convert SVG to PDF using CairoSVG."""
        try:
            import cairosvg
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            self._log(f"PDF created (cairosvg): {pdf_path}")
            return True
        except Exception as e:
            self._log(f"CairoSVG conversion failed: {e}")
            return False
