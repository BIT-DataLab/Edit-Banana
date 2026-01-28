"""
公式处理器模块

功能：
    1. 使用 Pix2Text 识别数学公式
    2. 融合 Azure OCR 和 Pix2Text 结果
    3. 验证公式有效性
    4. 清洗 LaTeX 命令

负责人：[填写负责人姓名]

接口说明：
    输入：azure_result（Azure OCR 结果）和 pix2text_result（Pix2Text 结果）
    输出：合并后的 text_blocks 列表

使用示例：
    from processors.formula import FormulaProcessor
    
    processor = FormulaProcessor()
    merged_blocks = processor.merge_ocr_results(azure_result, pix2text_result)
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import field


@dataclass
class MergedBlock:
    """合并后的文本块"""
    text: str
    polygon: List[Tuple[float, float]]
    confidence: float = 1.0
    font_size_px: float = 12.0
    is_latex: bool = False
    source: str = "azure"  # "azure" or "pix2text"
    # 字体相关（从 Azure 继承）
    font_name: Optional[str] = None
    font_style: Optional[str] = None
    font_weight: Optional[str] = None
    font_color: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    spans: List[dict] = field(default_factory=list)


class FormulaProcessor:
    """
    公式处理器
    
    处理流程：
    1. 从 Pix2Text 结果提取有效公式
    2. 计算公式与 Azure 文字块的重叠
    3. 用公式替换被覆盖的文字块
    4. 保留未被覆盖的 Azure 文字块
    """
    
    # 数学符号指示器
    MATH_INDICATORS = [
        '\\frac', '\\sum', '\\int', '\\prod', '\\approx', '\\equiv',
        '\\neq', '\\leq', '\\geq', '=', '\\mathbb', '\\mathbf',
        '\\pi', '\\theta', '\\sigma', '_', '^', '\\sqrt',
        '_t', '_i', '_j', '_{'
    ]
    
    # 无效模式（Pix2Text 误识别）
    INVALID_PATTERNS = [
        '\\Updownarrow',
        '\\dagger', '\\hat{\\varUpsilon}',
        '\\underline', '\\underbrace', '\\overbrace',
        '\\varPsi', '\\vdots', '\\Im',
        '\\widehat{\\left\\{',  # 误识别的图形元素
    ]
    
    # 重复模式检测（如 \b=\b=\b= 或 \\Xi}\\Xi}）
    REPETITIVE_PATTERN = re.compile(r'(.{2,10}?)\1{3,}')
    
    # LaTeX 命令替换
    LATEX_REPLACEMENTS = [
        ('\\boldsymbol', '\\mathbf'),
        ('\\cfrac', '\\frac'),
        (r'\ ', ' '),
        ('\\mathrm{o l d}', '\\text{old}'),
        # 修复误识别的圈乘符号
        ('\\circledR', '\\otimes'),
        ('\\copyright', '\\otimes'),
        ('\\textcircled{x}', '\\otimes'),
        ('\\textcircled{\\times}', '\\otimes'),
        ('\\textcircled{r}', '\\otimes'),
        ('\\textcircled{R}', '\\otimes'),
    ]
    
    def __init__(self, overlap_threshold: float = 0.2, text_similarity_threshold: float = 0.8):
        """
        初始化公式处理器
        
        Args:
            overlap_threshold: 覆盖率阈值
            text_similarity_threshold: 文本相似度阈值
        """
        self.overlap_threshold = overlap_threshold
        self.text_similarity_threshold = text_similarity_threshold
    
    def merge_ocr_results(
        self,
        azure_result,
        pix2text_result
    ) -> List[MergedBlock]:
        """
        合并 Azure 和 Pix2Text 结果（主入口）
        
        Args:
            azure_result: Azure OCR 结果对象
            pix2text_result: Pix2Text 结果对象
            
        Returns:
            合并后的 MergedBlock 列表
        """
        # 无 Pix2Text 结果时直接转换 Azure
        if not pix2text_result:
            return self._convert_azure_only(azure_result)
        
        # 提取有效公式（兼容 Pix2Text 和 UniMERNet）
        formula_blocks = [
            b for b in pix2text_result.blocks
            if getattr(b, 'type', '') == 'formula' or 
               getattr(b, 'block_type', '') == 'formula' or 
               getattr(b, 'is_latex', False)
        ]
        
        # 详细调试：打印每个检测到的公式
        print(f"   公式引擎检测到 {len(formula_blocks)} 个公式块:")
        for i, f in enumerate(formula_blocks):
            is_valid = self.is_valid_formula(f.text)
            status = "✅" if is_valid else "❌"
            print(f"      {i+1}. {status} \"{f.text}\"")
        
        valid_formulas = [f for f in formula_blocks if self.is_valid_formula(f.text)]
        
        print(f"   {len(valid_formulas)}/{len(formula_blocks)} 个有效公式")
        
        merged_results: List[MergedBlock] = []
        azure_used_indices = set()
        
        # 处理每个公式
        for formula in valid_formulas:
            f_poly = formula.polygon
            f_box = self._polygon_to_bbox(f_poly)
            
            matched_indices = []
            
            # 查找与公式重叠的 Azure 块
            for i, azure_block in enumerate(azure_result.text_blocks):
                if i in azure_used_indices:
                    continue
                
                a_box = self._polygon_to_bbox(azure_block.polygon)
                ratio = self._calculate_overlap_ratio(f_box, a_box)
                text_match = self.text_similarity(azure_block.text, formula.text) > self.text_similarity_threshold
                
                if ratio > self.overlap_threshold or text_match:
                    matched_indices.append(i)
            
            # 标记已使用的 Azure 块
            if matched_indices:
                azure_used_indices.update(matched_indices)
            
            # 无论是否检测到重叠，都保留 Pix2Text 的公式（优先信任 Pix2Text）
            cleaned_text = self.clean_latex(formula.text)
            formula_height = f_box[3] - f_box[1]
            
            merged_results.append(MergedBlock(
                text=cleaned_text,
                polygon=f_poly,
                confidence=getattr(formula, 'score', 1.0),
                font_size_px=formula_height * 0.35,
                is_latex=True,
                source="pix2text"
            ))
        
        # 添加剩余的 Azure 块（保留字体信息）
        for i, azure_block in enumerate(azure_result.text_blocks):
            if i not in azure_used_indices:
                merged_results.append(MergedBlock(
                    text=azure_block.text,
                    polygon=azure_block.polygon,
                    confidence=getattr(azure_block, 'confidence', 1.0),
                    font_size_px=azure_block.font_size_px,
                    is_latex=False,
                    source="azure",
                    # 保留 Azure 的字体信息
                    font_name=getattr(azure_block, 'font_name', None),
                    font_style=getattr(azure_block, 'font_style', None),
                    font_weight=getattr(azure_block, 'font_weight', None),
                    font_color=getattr(azure_block, 'font_color', None),
                    is_bold=getattr(azure_block, 'is_bold', False),
                    is_italic=getattr(azure_block, 'is_italic', False),
                    spans=getattr(azure_block, 'spans', [])
                ))
        
        return merged_results
    
    def _convert_azure_only(self, azure_result) -> List[MergedBlock]:
        """仅转换 Azure 结果（保留字体信息）"""
        return [
            MergedBlock(
                text=b.text,
                polygon=b.polygon,
                confidence=getattr(b, 'confidence', 1.0),
                font_size_px=b.font_size_px,
                is_latex=False,
                source="azure",
                font_name=getattr(b, 'font_name', None),
                font_style=getattr(b, 'font_style', None),
                font_weight=getattr(b, 'font_weight', None),
                font_color=getattr(b, 'font_color', None),
                is_bold=getattr(b, 'is_bold', False),
                is_italic=getattr(b, 'is_italic', False),
                spans=getattr(b, 'spans', [])
            )
            for b in azure_result.text_blocks
        ]
    
    def is_valid_formula(self, latex_text: str) -> bool:
        """
        验证是否为有效公式
        
        过滤条件：
        1. 长度 >= 2
        2. 包含数学符号
        3. 不包含无效模式
        4. 不是纯粗体短文本
        5. 不是重复模式（如 \b=\b=\b=...）
        6. 不是纯 array 结构（通常是误识别的图形）
        """
        text = latex_text.strip().strip('$').strip()
        
        if len(text) < 2:
            return False
        
        # 扩展数学符号检测：包括上下标、希腊字母等
        extended_math_indicators = [
            '\\frac', '\\sum', '\\int', '\\prod', '\\approx', '\\equiv',
            '\\neq', '\\leq', '\\geq', '\\mathbb', '\\mathbf',
            '\\pi', '\\theta', '\\sigma', '\\alpha', '\\beta', '\\gamma',
            '\\epsilon', '\\varepsilon', '\\delta', '\\lambda', '\\mu',
            '\\sqrt', '\\tilde', '\\hat', '\\bar', '\\vec', '\\dot',
            '_', '^', '=', '+', '-', '\\times', '\\cdot',
            '\\left', '\\right', '\\langle', '\\rangle'
        ]
        
        has_math = any(ind in text for ind in extended_math_indicators)
        is_invalid = any(p in text for p in self.INVALID_PATTERNS)
        
        # 检测重复模式（如 \b=\b=\b= 重复）
        if self.REPETITIVE_PATTERN.search(text):
            return False
        
        # 纯 array/matrix 结构通常是误识别（如虚线分隔符）
        if '\\begin{array}' in text or '\\begin{matrix}' in text:
            # 如果只有 array/matrix 结构而没有实质数学内容，则无效
            content = re.sub(r'\\begin\{(array|matrix)\}.*?\\end\{\1\}', '', text, flags=re.DOTALL)
            content = re.sub(r'[\s\{\}\[\]\\]', '', content)
            if len(content) < 3:
                return False
        
        # 单字符通常不是公式（除非有特殊格式）
        if len(text) == 1:
            return False
        
        # 纯粗体短文本
        if text.startswith('\\mathbf{') and len(text) < 15:
            return False

        # 检查括号是否匹配 (解决 Extra close brace 错误)
        if text.count('{') != text.count('}'):
            return False
            
        return has_math and not is_invalid
    
    def clean_latex(self, latex_text: str) -> str:
        """
        清洗 LaTeX，替换不兼容的命令
        """
        text = latex_text
        for old, new in self.LATEX_REPLACEMENTS:
            text = text.replace(old, new)
        return text
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        改进：使用更严格的匹配，避免长文本与短公式误匹配
        """
        if not text1 or not text2:
            return 0.0
        
        # 清理特殊字符
        clean1 = re.sub(r'[\s\$\\{}\[\]]', '', text1.lower())
        clean2 = re.sub(r'[\s\$\\{}\[\]]', '', text2.lower())
        
        # 移除 LaTeX 命令
        clean1 = re.sub(r'[a-z]+bf|mathbf|frac|theta|pi|approx|sum|mathbb', '', clean1)
        clean2 = re.sub(r'[a-z]+bf|mathbf|frac|theta|pi|approx|sum|mathbb', '', clean2)
        
        if not clean1 or not clean2:
            return 0.0
        
        # 长度差异过大时，降低相似度
        len_ratio = min(len(clean1), len(clean2)) / max(len(clean1), len(clean2))
        if len_ratio < 0.3:  # 长度差异超过 3 倍
            return 0.0
        
        # 使用序列匹配而非字符集匹配
        # 计算最长公共子串比例
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, clean1, clean2).ratio()
        
        return ratio
    
    def _polygon_to_bbox(
        self,
        polygon: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """多边形转边界框 (x_min, y_min, x_max, y_max)"""
        if not polygon:
            return (0, 0, 0, 0)
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def _calculate_overlap_ratio(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """
        计算 box1 对 box2 的覆盖率
        
        返回 (交集面积) / (box2 面积)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / area2 if area2 > 0 else 0.0
    
    def to_dict_list(self, merged_blocks: List[MergedBlock]) -> List[Dict[str, Any]]:
        """
        将 MergedBlock 列表转换为字典列表
        
        供后续处理器使用
        """
        return [
            {
                "text": block.text,
                "polygon": block.polygon,
                "confidence": block.confidence,
                "font_size_px": block.font_size_px,
                "is_latex": block.is_latex,
                "source": block.source,
                # 字体信息
                "font_family": block.font_name,  # 使用 font_family 键名与后续处理器兼容
                "font_style": block.font_style,
                "font_weight": block.font_weight,
                "font_color": block.font_color,
                "is_bold": block.is_bold,
                "is_italic": block.is_italic,
                "spans": block.spans or []
            }
            for block in merged_blocks
        ]


if __name__ == "__main__":
    processor = FormulaProcessor()
    
    # 测试公式验证
    print("=== 公式验证测试 ===")
    test_formulas = [
        r'\frac{a}{b}',
        r'x^2 + y^2 = z^2',
        'hello',
        r'\mathbf{A}',
        r'\sum_{i=1}^{n} x_i'
    ]
    for formula in test_formulas:
        print(f"  '{formula}': {processor.is_valid_formula(formula)}")
    
    # 测试文本相似度
    print("\n=== 文本相似度测试 ===")
    print(f"  'x^2' vs '$x^2$': {processor.text_similarity('x^2', '$x^2$'):.2f}")
