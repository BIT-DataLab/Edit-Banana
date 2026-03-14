"""
OCR 数据源模块

包含：
    - LocalOCR: 本地 Tesseract OCR（默认，易安装）
    - Pix2TextOCR: Pix2Text 公式识别
    - TextBlock, OCRResult: 通用数据结构
"""

from .base import TextBlock, OCRResult
from .local_ocr import LocalOCR
from .pix2text import Pix2TextOCR, Pix2TextBlock, Pix2TextResult

__all__ = [
    "TextBlock",
    "OCRResult",
    "LocalOCR",
    "Pix2TextOCR",
    "Pix2TextBlock",
    "Pix2TextResult",
]
