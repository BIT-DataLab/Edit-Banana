"""
Azure Document Intelligence OCR 模块
调用 Azure API 获取文字块的多边形坐标和初始样式
"""

import base64
import io
import math
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import AZURE_ENDPOINT, AZURE_API_KEY


@dataclass
class TextBlock:
    """文本块数据结构"""
    text: str
    polygon: list[tuple[float, float]]  # 8点多边形坐标 [(x1,y1), ..., (x4,y4)]
    confidence: float
    font_size_px: Optional[float] = None
    font_style: Optional[str] = None
    is_latex: bool = False
    latex_content: Optional[str] = None


@dataclass
class OCRResult:
    """OCR 识别结果"""
    image_width: int
    image_height: int
    text_blocks: list[TextBlock] = field(default_factory=list)


class AzureOCR:
    """Azure Document Intelligence OCR 客户端"""
    
    def __init__(self):
        if not AZURE_ENDPOINT or not AZURE_API_KEY:
            raise ValueError("Azure API 配置缺失，请检查 .env 文件")
        
        self.client = DocumentIntelligenceClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY),
            connection_verify=False
        )
    
    def _compress_image_if_needed(self, image_path: Path) -> tuple[bytes, int, int]:
        """
        如果图片太大，压缩它以满足 Azure API 限制
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple[bytes, int, int]: (压缩后的图片字节, 原始宽度, 原始高度)
        """
        # 读取原始图片获取尺寸
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            # 转换为 RGB（如果是 RGBA 或其他格式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 读取原始文件大小
            original_size = image_path.stat().st_size
            max_size = 4 * 1024 * 1024  # 4MB 限制
            
            if original_size <= max_size:
                # 文件不大，直接返回
                with open(image_path, "rb") as f:
                    return f.read(), original_width, original_height
            
            # 需要压缩：逐步降低质量直到文件大小合适
            output = io.BytesIO()
            quality = 95
            
            while quality > 20:
                output.seek(0)
                output.truncate(0)
                img.save(output, format='JPEG', quality=quality, optimize=True)
                compressed_size = output.tell()
                
                if compressed_size <= max_size:
                    break
                quality -= 10
            
            output.seek(0)
            return output.read(), original_width, original_height
    
    def analyze_image(self, image_path: str) -> OCRResult:
        """
        分析图像并提取文字块
        
        Args:
            image_path: 图像文件路径 (PNG, JPG, BMP, PDF)
            
        Returns:
            OCRResult: 包含所有文字块的识别结果
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 压缩图片（如果需要）并获取原始尺寸
        image_bytes, original_width, original_height = self._compress_image_if_needed(image_path)
        
        # 调用 Azure Document Intelligence API
        # 使用 prebuilt-read 模型，启用高分辨率 OCR
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=image_bytes),
            features=[
                DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,
                DocumentAnalysisFeature.STYLE_FONT
            ]
        )
        
        result = poller.result()
        
        # 使用原始图像尺寸（不是压缩后的）
        image_width = original_width
        image_height = original_height
        
        # 提取文字块（使用 lines 级别以获取完整短语）
        text_blocks = []
        
        if result.pages:
            for page in result.pages:
                if page.lines:
                    for line in page.lines:
                        # 提取多边形坐标
                        polygon = self._extract_polygon(line.polygon)
                        
                        # 计算字号（基于多边形高度）
                        font_size_px = self._estimate_font_size(polygon)
                        
                        # lines 没有直接的 confidence，使用默认值 1.0
                        # 或者从其包含的 words 中计算平均置信度
                        confidence = 1.0
                        
                        block = TextBlock(
                            text=line.content,
                            polygon=polygon,
                            confidence=confidence,
                            font_size_px=font_size_px
                        )
                        text_blocks.append(block)
        
        # 提取样式信息（如果有）
        if result.styles:
            self._apply_styles(text_blocks, result.styles)
        
        return OCRResult(
            image_width=int(image_width),
            image_height=int(image_height),
            text_blocks=text_blocks
        )
    
    def _extract_polygon(self, polygon_data) -> list[tuple[float, float]]:
        """
        从 API 返回的多边形数据提取坐标点
        Azure 返回的是 [x1, y1, x2, y2, x3, y3, x4, y4] 的扁平数组
        """
        if not polygon_data:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]
        
        points = []
        for i in range(0, len(polygon_data), 2):
            if i + 1 < len(polygon_data):
                points.append((polygon_data[i], polygon_data[i + 1]))
        
        # 确保有4个点
        while len(points) < 4:
            points.append((0, 0))
        
        return points[:4]
    
    def _estimate_font_size(self, polygon: list[tuple[float, float]]) -> float:
        """
        根据多边形估算字号（像素高度）
        
        对于普通横排文字：使用 Y 方向的高度
        对于竖排文字（旋转约90°）：使用 X 方向的宽度
        
        通用方法：使用多边形的"短边"长度，因为文字高度通常是短边
        """
        if len(polygon) < 4:
            return 12.0
        
        p0, p1, p2, p3 = polygon[:4]
        
        # 计算两条相邻边的长度
        # 边1: p0 -> p1 (通常是顶边/横向)
        edge1_len = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        # 边2: p0 -> p3 (通常是左边/纵向)  
        edge2_len = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
        
        # 文字的"高度"是较短的那条边
        # 因为文字框的宽度（文字长度方向）通常大于高度（字号方向）
        font_height = min(edge1_len, edge2_len)
        
        return font_height if font_height > 0 else 12.0
    
    def _apply_styles(self, text_blocks: list[TextBlock], styles) -> None:
        """
        将样式信息应用到文字块（如果 Azure 返回了样式）
        """
        # Azure 的 styles 包含字体、大小等信息
        # 根据 spans 匹配到对应的文字块
        for style in styles:
            if hasattr(style, 'font_style'):
                font_style = style.font_style
                # 这里可以根据 style.spans 来匹配具体的文字块
                # 简化实现：暂时跳过详细匹配


def analyze_image_with_azure(image_path: str) -> OCRResult:
    """
    便捷函数：使用 Azure OCR 分析图像
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        OCRResult: OCR 识别结果
    """
    ocr = AzureOCR()
    return ocr.analyze_image(image_path)


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        result = analyze_image_with_azure(sys.argv[1])
        print(f"图像尺寸: {result.image_width} x {result.image_height}")
        print(f"识别到 {len(result.text_blocks)} 个文字块")
        for block in result.text_blocks[:5]:
            print(f"  - '{block.text}' (置信度: {block.confidence:.2f})")

