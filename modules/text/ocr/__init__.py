"""
OCR 数据源模块

包含：
    - AzureOCR: Azure Document Intelligence OCR
    - Pix2TextOCR: Pix2Text 公式识别

这些是数据源层，一般不需要修改。
"""

from .azure import AzureOCR, TextBlock, OCRResult
from .pix2text import Pix2TextOCR, Pix2TextBlock, Pix2TextResult

__all__ = [
    'AzureOCR',
    'TextBlock',
    'OCRResult',
    'Pix2TextOCR',
    'Pix2TextBlock',
    'Pix2TextResult',
]
