"""
图片转可编辑形式的模块化处理框架

模块划分:
    1. text (TextRestorer)    - 文字处理（OCR + 公式识别）【第一步】
    2. sam3_info_extractor   - SAM3提取结构化信息
    3. icon_picture_processor - Icon/Picture非基本图形处理（转base64，生成XML）
    4. basic_shape_processor  - 基本图形处理（取色、生成XML）
    5. arrow_processor        - 箭头处理（生成XML）
    6. xml_merger            - XML合并（只负责收集和排序，不生成样式）
    7. metric_evaluator      - 质量评估
    8. refinement_processor  - 表现不好的区域二次处理

重构说明：
    - 每个子模块负责生成自己的mxCell XML代码
    - 子模块设置 element.xml_fragment 和 element.layer_level
    - XMLMerger只负责收集、排序、合并，不负责生成样式
    - 文字处理模块（modules/text/）已整合，直接导入 TextRestorer

使用方式:
    from modules import Sam3InfoExtractor, XMLMerger, TextRestorer
    from modules.data_types import ElementInfo, XMLFragment, LayerLevel
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
