"""
字号计算器模块
实现 Cap-Height 算法计算字号
"""

import re
from dataclasses import dataclass
from typing import Optional
from statistics import mean, median

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import CAP_HEIGHT_RATIO, RENDER_RATIO


@dataclass
class FontSizeResult:
    """字号计算结果"""
    estimated_pt: float  # 估算的磅值
    cap_height_px: float  # 大写字母高度（像素）
    sample_count: int  # 采样数量
    confidence: float  # 置信度 [0, 1]


class FontCalculator:
    """
    字号计算器
    
    实现 Cap-Height 算法：
    - 锚点识别: 扫描文本内容，筛选出大写字母 (A-Z) 和数字 (0-9)
    - 高度计算: OCR 返回的高度近似于 cap-height
    - 转换公式: font_size_pt = (cap_height_px / CAP_HEIGHT_RATIO) / RENDER_RATIO
    
    参数说明见 config.py
    """
    
    def __init__(self, canvas_scale: float = 1.0, cap_height_ratio: float = None,
                 source_width: int = None, source_height: int = None):
        """
        初始化字号计算器
        
        Args:
            canvas_scale: 画布缩放比例
            cap_height_ratio: 大写字母高度占总字号的比例（默认 0.7）
            source_width: 源图像宽度（像素）- 保留参数兼容性
            source_height: 源图像高度（像素）- 保留参数兼容性
        """
        self.canvas_scale = canvas_scale
        self.cap_height_ratio = cap_height_ratio or CAP_HEIGHT_RATIO
        
        # 用于匹配大写字母和数字的正则表达式
        self.anchor_pattern = re.compile(r'[A-Z0-9]')
    
    def contains_anchor_chars(self, text: str) -> bool:
        """
        检查文本是否包含锚点字符（大写字母或数字）
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否包含锚点字符
        """
        return bool(self.anchor_pattern.search(text))
    
    def calculate_font_size(
        self, 
        text: str, 
        polygon_height_px: float,
        bbox_width: float = None,
        bbox_height: float = None,
        rotation: float = 0.0
    ) -> FontSizeResult:
        """
        根据文本内容和多边形高度计算字号（Cap-Height 算法）
        
        核心原理：
        1. OCR 返回的 polygon_height_px 近似于 cap-height（大写字母高度）
        2. 根据排版学标准：cap-height ≈ 0.7 × font-size
        3. 考虑 draw.io 的渲染特性：N pt 文字 ≈ N × 1.4 px
        
        公式: font_size_pt = (cap_height_px / 0.7) / 1.4
                          = cap_height_px / 0.98
                          ≈ cap_height_px
        
        Args:
            text: 文本内容
            polygon_height_px: 多边形短边长度（像素）- OCR 返回的字符高度
            bbox_width: 边界框宽度（可选）
            bbox_height: 边界框高度（可选）
            rotation: 旋转角度（度）
            
        Returns:
            FontSizeResult: 字号计算结果
        """
        has_anchors = self.contains_anchor_chars(text)
        
        # Cap-Height 算法
        # OCR 返回的高度近似于 cap-height（如果有大写字母/数字）
        if has_anchors:
            # 包含大写字母/数字
            # cap-height ≈ 0.7 × font-size → font_size_px = cap_height / 0.7
            font_size_px = polygon_height_px / self.cap_height_ratio
        else:
            # 只有小写字母，高度近似 x-height
            # x-height ≈ 0.5 × font-size → font_size_px = x_height / 0.5
            font_size_px = polygon_height_px / 0.5
        
        # 像素到 pt 的转换
        # draw.io 中 N pt 的文字渲染高度约为 N × RENDER_RATIO 像素
        # 所以 font_size_pt = font_size_px / RENDER_RATIO
        estimated_pt = font_size_px / RENDER_RATIO
        
        # 应用画布缩放（如果有）
        estimated_pt = estimated_pt * self.canvas_scale
        
        # 根据是否有锚点字符设置置信度
        if has_anchors:
            confidence = 0.9
            sample_count = len(self.anchor_pattern.findall(text))
        else:
            confidence = 0.7
            sample_count = 0
        
        return FontSizeResult(
            estimated_pt=round(estimated_pt, 1),
            cap_height_px=polygon_height_px,  # 保存原始 OCR 高度
            sample_count=sample_count,
            confidence=confidence
        )
    
    def calculate_batch_font_size(
        self, 
        text_blocks: list[tuple[str, float]]
    ) -> tuple[float, list[FontSizeResult]]:
        """
        批量计算字号，并返回统一的推荐字号
        
        设计要求：同一文本框内文字严禁出现因笔画差异导致的"字号抖动"
        因此需要计算一个稳定的统一字号
        
        Args:
            text_blocks: 列表，每项为 (文本内容, 多边形高度像素)
            
        Returns:
            tuple: (推荐统一字号, 各块的计算结果列表)
        """
        results = []
        anchor_sizes = []  # 有锚点字符的字号
        all_sizes = []
        
        for text, height_px in text_blocks:
            result = self.calculate_font_size(text, height_px)
            results.append(result)
            all_sizes.append(result.estimated_pt)
            
            if result.sample_count > 0:
                anchor_sizes.append(result.estimated_pt)
        
        # 优先使用有锚点字符的样本来确定统一字号
        if anchor_sizes:
            # 使用中位数避免异常值影响
            unified_size = median(anchor_sizes)
        elif all_sizes:
            unified_size = median(all_sizes)
        else:
            unified_size = 12.0  # 默认字号
        
        # 将统一字号量化到标准档位
        unified_size = self.quantize_font_size(unified_size)
        
        return unified_size, results
    
    def quantize_font_size(self, size: float) -> float:
        """
        将字号量化到标准档位
        
        常用字号档位: 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 48, 72
        
        Args:
            size: 原始估算字号
            
        Returns:
            float: 量化后的标准字号
        """
        standard_sizes = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 42, 48, 56, 72]
        
        # 找到最接近的标准字号
        closest = min(standard_sizes, key=lambda s: abs(s - size))
        return float(closest)
    
    def group_by_font_size(
        self, 
        text_blocks: list[tuple[str, float]],
        tolerance: float = 2.0
    ) -> dict[float, list[int]]:
        """
        按字号分组文本块
        
        Args:
            text_blocks: 文本块列表
            tolerance: 字号容差（磅）
            
        Returns:
            dict: {字号: [块索引列表]}
        """
        _, results = self.calculate_batch_font_size(text_blocks)
        
        groups: dict[float, list[int]] = {}
        
        for i, result in enumerate(results):
            quantized = self.quantize_font_size(result.estimated_pt)
            
            # 查找是否有相近的组
            found_group = None
            for group_size in groups.keys():
                if abs(group_size - quantized) <= tolerance:
                    found_group = group_size
                    break
            
            if found_group is not None:
                groups[found_group].append(i)
            else:
                groups[quantized] = [i]
        
        return groups


def create_calculator(canvas_scale: float = 1.0) -> FontCalculator:
    """
    便捷函数：创建字号计算器
    
    Args:
        canvas_scale: 画布缩放比例
        
    Returns:
        FontCalculator: 字号计算器实例
    """
    return FontCalculator(canvas_scale)


def px_to_pt(px: float, dpi: float = 72) -> float:
    """
    像素转换为磅值
    
    Args:
        px: 像素值
        dpi: DPI（默认 72）
        
    Returns:
        float: 磅值
    """
    # 1 英寸 = 72 磅
    # 像素 / DPI = 英寸
    # 英寸 × 72 = 磅
    return px / dpi * 72


if __name__ == "__main__":
    # 测试代码
    calculator = FontCalculator(canvas_scale=1.0)
    
    # 测试单个文本
    test_cases = [
        ("Hello World", 24),  # 包含大写字母
        ("ABC123", 18),       # 全是锚点字符
        ("hello", 20),        # 只有小写字母
        ("$x^2 + y^2$", 16),  # LaTeX 公式
    ]
    
    print("字号计算测试:")
    for text, height in test_cases:
        result = calculator.calculate_font_size(text, height)
        print(f"  '{text}' (高度{height}px): {result.estimated_pt}pt, "
              f"置信度: {result.confidence:.1f}")
    
    # 测试批量计算
    print("\n批量计算测试:")
    unified, results = calculator.calculate_batch_font_size(test_cases)
    print(f"统一推荐字号: {unified}pt")
    
    # 测试分组
    print("\n字号分组测试:")
    groups = calculator.group_by_font_size(test_cases)
    for size, indices in groups.items():
        print(f"  {size}pt: {indices}")

