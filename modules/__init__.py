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

# 图形处理模块
from .icon_picture_processor import IconPictureProcessor
from .basic_shape_processor import BasicShapeProcessor
from .arrow_processor import ArrowProcessor
from .metric_evaluator import MetricEvaluator
from .refinement_processor import RefinementProcessor

# 文字处理模块（已整合到 modules/text/）；依赖 ocr/coord_processor 等，缺失时可选跳过
try:
    from .text.restorer import TextRestorer
except Exception as e:
    import warnings
    warnings.warn(f"TextRestorer unavailable (missing deps): {e}. Pipeline will run with_text=False.")
    TextRestorer = None

__all__ = [
    # 基础类
    'BaseProcessor',
    'ProcessingContext',
    
    # 数据类型
    'ElementInfo',
    'BoundingBox',
    'ProcessingResult',
    'XMLFragment',
    'LayerLevel',
    'get_layer_level',
    
    # 文字处理模块（第一步）
    'TextRestorer',
    
    # 核心模块
    'Sam3InfoExtractor',
    'XMLMerger',
    
    # 图形处理模块
    'IconPictureProcessor',
    'BasicShapeProcessor',
    'ArrowProcessor',
    'MetricEvaluator',
    'RefinementProcessor',
]
