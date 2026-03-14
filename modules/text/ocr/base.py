"""
OCR 通用数据结构

TextBlock、OCRResult 供各 OCR 引擎（本地 Tesseract、Pix2Text 等）统一使用。
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """文本块数据结构"""
    text: str
    polygon: List[Tuple[float, float]]
    confidence: float = 1.0
    font_size_px: Optional[float] = None
    spans: List[dict] = field(default_factory=list)
    font_style: Optional[str] = None
    font_weight: Optional[str] = None
    font_name: Optional[str] = None
    font_color: Optional[str] = None
    background_color: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class OCRResult:
    """OCR 识别结果"""
    image_width: int
    image_height: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    styles: List[dict] = field(default_factory=list)
