"""
内容对齐模块
使用 IoU（交并比）算法匹配 Azure 和 Mistral 返回的文本块
"""

from dataclasses import dataclass
from typing import Optional
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import IOU_THRESHOLD
from src.ocr_azure import TextBlock, OCRResult
from src.ocr_mistral import MistralOCRResult


@dataclass
class AlignedTextBlock:
    """对齐后的文本块，融合了 Azure 和 Mistral 的结果"""
    text: str  # 最终文本（可能来自 Mistral 的 LaTeX）
    polygon: list[tuple[float, float]]  # 来自 Azure 的精确坐标
    confidence: float  # 来自 Azure 的置信度
    font_size_px: float  # 估算的字号
    is_latex: bool  # 是否为 LaTeX 公式
    original_azure_text: str  # Azure 原始识别文本
    latex_source: Optional[str] = None  # 如果是 LaTeX，来源文本


def calculate_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """
    从多边形计算边界框 (x_min, y_min, x_max, y_max)
    
    Args:
        polygon: 多边形顶点列表
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    if not polygon:
        return (0, 0, 0, 0)
    
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    """
    计算两个边界框的 IoU（交并比）
    
    Args:
        bbox1: 边界框1 (x_min, y_min, x_max, y_max)
        bbox2: 边界框2 (x_min, y_min, x_max, y_max)
        
    Returns:
        float: IoU 值 [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度（简单的字符匹配）
    
    Args:
        text1: 文本1
        text2: 文本2
        
    Returns:
        float: 相似度 [0, 1]
    """
    if not text1 or not text2:
        return 0.0
    
    # 移除空格和特殊字符进行比较
    clean1 = re.sub(r'[\s\$\\{}]', '', text1.lower())
    clean2 = re.sub(r'[\s\$\\{}]', '', text2.lower())
    
    if not clean1 or not clean2:
        return 0.0
    
    # 计算公共字符数
    common = sum(1 for c in clean1 if c in clean2)
    max_len = max(len(clean1), len(clean2))
    
    return common / max_len if max_len > 0 else 0.0


def contains_math_symbols(text: str) -> bool:
    """
    检查文本是否包含数学符号
    
    Args:
        text: 要检查的文本
        
    Returns:
        bool: 是否包含数学符号
    """
    math_symbols = ['$', '\\', '∑', '∫', '∂', '√', 'α', 'β', 'γ', 'δ', 
                    'θ', 'λ', 'μ', 'σ', 'π', '∞', '≤', '≥', '≠', '±',
                    '_', '^', 'frac', 'sqrt', 'sum', 'int']
    return any(sym in text for sym in math_symbols)


def align_ocr_results(
    azure_result: OCRResult,
    mistral_result: MistralOCRResult,
    iou_threshold: float = None
) -> list[AlignedTextBlock]:
    """
    对齐 Azure 和 Mistral 的 OCR 结果
    
    策略：
    1. 以 Azure 的文本块为基准（因为有精确坐标）
    2. 检查每个文本块是否包含数学符号
    3. 如果包含，尝试从 Mistral 结果中找到对应的 LaTeX 表示
    
    Args:
        azure_result: Azure OCR 结果
        mistral_result: Mistral OCR 结果
        iou_threshold: IoU 阈值（默认使用配置值）
        
    Returns:
        list[AlignedTextBlock]: 对齐后的文本块列表
    """
    if iou_threshold is None:
        iou_threshold = IOU_THRESHOLD
    
    aligned_blocks = []
    
    # 解析 Mistral 结果中的 LaTeX 公式
    mistral_latex_map = _extract_latex_mapping(mistral_result.raw_text)
    
    for block in azure_result.text_blocks:
        # 检查是否需要用 LaTeX 替换
        is_latex = False
        final_text = block.text
        latex_source = None
        
        # 检查 Azure 文本是否包含可能的数学内容
        if contains_math_symbols(block.text):
            # 尝试从 Mistral 结果中匹配
            matched_latex = _find_matching_latex(block.text, mistral_latex_map)
            if matched_latex:
                final_text = matched_latex
                is_latex = True
                latex_source = "mistral"
        
        # 如果 Mistral 识别到了 LaTeX 但 Azure 没有识别出数学符号
        # 检查文本相似度来匹配
        for latex in mistral_result.latex_blocks:
            if text_similarity(block.text, latex) > 0.5:
                final_text = f"${latex}$"
                is_latex = True
                latex_source = "mistral"
                break
        
        aligned_block = AlignedTextBlock(
            text=final_text,
            polygon=block.polygon,
            confidence=block.confidence,
            font_size_px=block.font_size_px or 12.0,
            is_latex=is_latex,
            original_azure_text=block.text,
            latex_source=latex_source
        )
        aligned_blocks.append(aligned_block)
    
    return aligned_blocks


def _extract_latex_mapping(text: str) -> dict[str, str]:
    """
    从 Mistral 文本中提取 LaTeX 公式及其上下文
    
    Returns:
        dict: {简化文本: LaTeX原文}
    """
    mapping = {}
    
    # 匹配块级公式 $$...$$
    block_pattern = r'\$\$(.*?)\$\$'
    for match in re.finditer(block_pattern, text, re.DOTALL):
        latex = match.group(1).strip()
        # 提取简化版本作为键
        simplified = re.sub(r'[\\\{\}\s]', '', latex)
        mapping[simplified.lower()] = f"$${latex}$$"
    
    # 匹配行内公式 $...$
    inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
    for match in re.finditer(inline_pattern, text):
        latex = match.group(1).strip()
        simplified = re.sub(r'[\\\{\}\s]', '', latex)
        mapping[simplified.lower()] = f"${latex}$"
    
    return mapping


def _find_matching_latex(text: str, latex_map: dict[str, str]) -> Optional[str]:
    """
    在 LaTeX 映射中查找匹配的公式
    """
    simplified_text = re.sub(r'[\s]', '', text.lower())
    
    for key, latex in latex_map.items():
        if text_similarity(simplified_text, key) > 0.6:
            return latex
    
    return None




def merge_text_blocks(blocks: list[AlignedTextBlock], line_threshold: float = 10.0) -> list[AlignedTextBlock]:
    """
    合并同一行的文本块（模拟段落合并）
    
    逻辑：
    1. 按垂直坐标 (y) 排序
    2. 遍历检查：
       如果两个块垂直距离很近 (abs(y1-y2) < threshold) 且水平没有太大间隙：
       -> 合并它们
    """
    if not blocks:
        return []

    # 按 y 坐标排序
    sorted_blocks = sorted(blocks, key=lambda b: calculate_bbox(b.polygon)[1])
    
    merged = []
    current_line = [sorted_blocks[0]]
    current_bbox = calculate_bbox(sorted_blocks[0].polygon) # y_min
    
    for i in range(1, len(sorted_blocks)):
        block = sorted_blocks[i]
        bbox = calculate_bbox(block.polygon)
        
        # 检查是否在同一行 (Y轴重叠或接近)
        # 这里用中心点或顶边差异
        y_diff = abs(bbox[1] - current_bbox[1])
        
        if y_diff < line_threshold:
            current_line.append(block)
        else:
            # 结束当前行，合并并添加到结果
            merged.extend(_merge_line_blocks(current_line))
            current_line = [block]
            current_bbox = bbox
            
    # 处理最后一行
    if current_line:
        merged.extend(_merge_line_blocks(current_line))
        
    return merged

def _merge_line_blocks(line_blocks: list[AlignedTextBlock]) -> list[AlignedTextBlock]:
    """合并单行内的文本块（水平排序后合并）"""
    if not line_blocks: return []
    
    # 按 x 坐标排序
    line_blocks.sort(key=lambda b: calculate_bbox(b.polygon)[0])
    
    final_blocks = []
    if not line_blocks: return []
    
    current = line_blocks[0]
    
    for i in range(1, len(line_blocks)):
        next_block = line_blocks[i]
        
        curr_box = calculate_bbox(current.polygon)
        next_box = calculate_bbox(next_block.polygon)
        
        # 检查水平距离
        x_dist = next_box[0] - curr_box[2]
        
        # 如果距离小于一定值（比如字号的 2 倍），则合并
        # 注意：这里我们假设是从左到右书写
        font_size = current.font_size_px
        if x_dist < font_size * 2.0:
            # 合并文本
            new_text = current.text + " " + next_block.text
            # 合并 Polygon (取并集外框作为新的矩形 Polygon)
            # 简化：直接用大矩形四个点
            x1 = min(curr_box[0], next_box[0])
            y1 = min(curr_box[1], next_box[1])
            x2 = max(curr_box[2], next_box[2])
            y2 = max(curr_box[3], next_box[3])
            new_poly = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
            
            # 创建新对象
            current = AlignedTextBlock(
                text=new_text,
                polygon=new_poly,
                confidence=min(current.confidence, next_block.confidence),
                font_size_px=(current.font_size_px + next_block.font_size_px)/2,
                is_latex=current.is_latex or next_block.is_latex,
                original_azure_text=current.original_azure_text + " " + next_block.original_azure_text
            )
        else:
            final_blocks.append(current)
            current = next_block
            
    final_blocks.append(current)
    return final_blocks


