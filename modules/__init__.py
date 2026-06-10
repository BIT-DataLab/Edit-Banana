"""
Pipeline modules: segmentation, text extraction, shape/arrow handling, XML merge.
See project README for pipeline overview and config.
"""

from .base import BaseProcessor, ProcessingContext
from .data_types import (
    ElementInfo, 
    BoundingBox, 
    ProcessingResult, 
    XMLFragment,
    LayerLevel,
    get_layer_level,
)
from .sam3_info_extractor import Sam3InfoExtractor
from .xml_merger import XMLMerger

# Shape/icon processors
from .icon_picture_processor import IconPictureProcessor
from .basic_shape_processor import BasicShapeProcessor
from .metric_evaluator import MetricEvaluator
from .refinement_processor import RefinementProcessor

# Stage 8: Vector export
from .vector_exporter import VectorExporter
from .svg_generator import SVGGenerator
from .pdf_combiner import PDFCombiner
from .section_detector import SectionDetector

# Text (modules/text/); optional if ocr/coord_processor missing
try:
    from .text.restorer import TextRestorer
except Exception as e:
    import warnings
    warnings.warn(f"TextRestorer unavailable (missing deps): {e}. Pipeline will run with_text=False.")
    TextRestorer = None

__all__ = [
    # Base
    'BaseProcessor',
    'ProcessingContext',
    
    # Data types
    'ElementInfo',
    'BoundingBox',
    'ProcessingResult',
    'XMLFragment',
    'LayerLevel',
    'get_layer_level',
    
    # Text (first step)
    'TextRestorer',
    
    # Core
    'Sam3InfoExtractor',
    'XMLMerger',
    # Shape/icon
    'IconPictureProcessor',
    'BasicShapeProcessor',
    'MetricEvaluator',
    'RefinementProcessor',
    # Stage 8: Vector export
    'VectorExporter',
    'SVGGenerator',
    'PDFCombiner',
    'SectionDetector',
]
