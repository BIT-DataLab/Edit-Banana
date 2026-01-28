"""
Azure Document Intelligence OCR 模块

功能：
    调用本地部署的 Azure Document Intelligence Docker 容器进行 OCR 识别。
    支持识别文字内容、位置坐标、字体、加粗、斜体、颜色等样式信息。

本地容器部署：
    docker run -d --name azure-form-recognizer -p 5000:5000 --memory 16g --cpus 8 \
        -e Eula=accept -e Billing=<endpoint> -e ApiKey=<key> \
        mcr.microsoft.com/azure-cognitive-services/form-recognizer/layout-4.0:latest

返回信息：
    - text: 文字内容
    - polygon: 四边形坐标
    - font_name: 字体名称
    - font_weight: 字重（normal, bold）
    - font_style: 样式（normal, italic）
    - font_color: 颜色
"""

import io
import math
import time
import requests
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from PIL import Image


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


class AzureOCR:
    """
    Azure Document Intelligence OCR 客户端
    
    使用示例：
        ocr = AzureOCR()
        result = ocr.analyze_image("input.png")
        for block in result.text_blocks:
            print(f"{block.text} - 字体: {block.font_name}")
    """
    
    def __init__(self, endpoint: str = "http://localhost:5000"):
        """
        初始化 OCR 客户端
        
        Args:
            endpoint: 本地容器地址
        """
        self.endpoint = endpoint.rstrip('/')
        self.analyze_url = f"{self.endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze"
        self.api_version = "2024-11-30"
        
        # 验证容器
        try:
            resp = requests.get(f"{self.endpoint}/ready", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"容器未就绪: {resp.text}")
        except requests.RequestException as e:
            raise ConnectionError(f"无法连接到容器 {endpoint}: {e}")
    
    def analyze_image(self, image_path: str) -> OCRResult:
        """
        分析图像并提取文字
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            OCRResult: 识别结果
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        # 压缩图片
        image_bytes, width, height = self._compress_image(image_path)
        
        # Content-Type
        suffix = image_path.suffix.lower()
        content_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.bmp': 'image/bmp',
            '.tiff': 'image/tiff', '.pdf': 'application/pdf'
        }
        content_type = content_types.get(suffix, 'application/octet-stream')
        
        # 发送请求
        headers = {'Content-Type': content_type}
        params = {'api-version': self.api_version, 'features': 'styleFont'}
        
        resp = requests.post(self.analyze_url, headers=headers, params=params, 
                            data=image_bytes, timeout=120)
        
        if resp.status_code != 202:
            raise RuntimeError(f"分析请求失败: {resp.status_code} - {resp.text}")
        
        # 轮询结果
        operation_url = resp.headers.get('Operation-Location')
        if not operation_url:
            raise RuntimeError("未返回 Operation-Location")
        
        result = self._poll_result(operation_url)
        return self._parse_result(result, width, height)
    
    def _compress_image(self, image_path: Path) -> Tuple[bytes, int, int]:
        """压缩图片（API 限制 4MB）"""
        with Image.open(image_path) as img:
            width, height = img.size
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_size = image_path.stat().st_size
            if original_size <= 4 * 1024 * 1024:
                with open(image_path, "rb") as f:
                    return f.read(), width, height
            
            # 压缩
            output = io.BytesIO()
            quality = 95
            while quality > 20:
                output.seek(0)
                output.truncate(0)
                img.save(output, format='JPEG', quality=quality, optimize=True)
                if output.tell() <= 4 * 1024 * 1024:
                    break
                quality -= 10
            
            output.seek(0)
            return output.read(), width, height
    
    def _poll_result(self, operation_url: str, max_wait: int = 120) -> dict:
        """轮询等待结果"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            resp = requests.get(operation_url, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"获取结果失败: {resp.status_code}")
            
            result = resp.json()
            status = result.get('status')
            
            if status == 'succeeded':
                return result.get('analyzeResult', {})
            elif status == 'failed':
                raise RuntimeError(f"分析失败: {result.get('error', {}).get('message')}")
            
            time.sleep(2)
        
        raise TimeoutError(f"分析超时（{max_wait}秒）")
    
    def _parse_result(self, result: dict, width: int, height: int) -> OCRResult:
        """解析 API 返回结果"""
        text_blocks = []
        
        # 解析文字行
        for page in result.get('pages', []):
            for line in page.get('lines', []):
                polygon = self._extract_polygon(line.get('polygon', []))
                font_size_px = self._estimate_font_size(polygon)
                
                block = TextBlock(
                    text=line.get('content', ''),
                    polygon=polygon,
                    confidence=1.0,
                    font_size_px=font_size_px,
                    spans=line.get('spans', [])
                )
                text_blocks.append(block)
        
        # 应用样式
        styles = result.get('styles', [])
        self._apply_styles(text_blocks, styles)
        
        return OCRResult(
            image_width=width, 
            image_height=height, 
            text_blocks=text_blocks,
            styles=styles
        )
    
    def _extract_polygon(self, polygon_data: list) -> List[Tuple[float, float]]:
        """提取四边形坐标"""
        if not polygon_data:
            return [(0, 0)] * 4
        
        points = []
        for i in range(0, len(polygon_data), 2):
            if i + 1 < len(polygon_data):
                points.append((polygon_data[i], polygon_data[i + 1]))
        
        while len(points) < 4:
            points.append((0, 0))
        
        return points[:4]
    
    def _estimate_font_size(self, polygon: List[Tuple[float, float]]) -> float:
        """估算字号（取短边长度）"""
        if len(polygon) < 4:
            return 12.0
        
        p0, p1, p2, p3 = polygon[:4]
        edge1 = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        edge2 = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
        
        return min(edge1, edge2) if min(edge1, edge2) > 0 else 12.0
    
    def _apply_styles(self, text_blocks: List[TextBlock], styles: list) -> None:
        """将样式应用到文字块"""
        for style in styles:
            font_style = style.get('fontStyle')
            font_weight = style.get('fontWeight')
            font_name = style.get('similarFontFamily')
            font_color = style.get('color')
            background_color = style.get('backgroundColor')
            
            for style_span in style.get('spans', []):
                style_start = style_span.get('offset', 0)
                style_end = style_start + style_span.get('length', 0)
                
                for block in text_blocks:
                    for block_span in block.spans:
                        block_start = block_span.get('offset', 0)
                        block_end = block_start + block_span.get('length', 0)
                        
                        # 检查重叠
                        if max(style_start, block_start) < min(style_end, block_end):
                            if font_style:
                                block.font_style = font_style
                                block.is_italic = (font_style == 'italic')
                            if font_weight:
                                block.font_weight = font_weight
                                block.is_bold = (font_weight == 'bold')
                            if font_name:
                                block.font_name = font_name
                            if font_color:
                                block.font_color = font_color
                            if background_color:
                                block.background_color = background_color
                            break
