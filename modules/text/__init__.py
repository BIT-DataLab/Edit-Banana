"""
文字处理模块

功能：
    将流程图图片中的文字和公式识别并转换为 draw.io XML 格式。

Pipeline 接口：
    from modules.text import TextRestorer
    
    restorer = TextRestorer()
    xml_string = restorer.process("input.png")  # 返回 XML 字符串

原始代码来自 flowchart_text 文件夹，已整合到 modules/text/ 下。
"""

from .restorer import TextRestorer
from .coord_processor import CoordProcessor
from .xml_generator import MxGraphXMLGenerator, TextCellData

__all__ = [
    'TextRestorer',
    'CoordProcessor', 
    'MxGraphXMLGenerator',
    'TextCellData',
]
