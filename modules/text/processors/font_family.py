"""
字体处理器模块

功能：
    1. 从 Azure OCR 结果提取字体名称
    2. 字体名称标准化（如 ArialMT → Arial）
    3. 字体推测（当 OCR 未返回字体时，根据文本内容推测）
    4. 空间聚类统一字体

负责人：[填写负责人姓名]

接口说明：
    输入：text_blocks 列表，每个块包含 text 和可选的 font_family
    输出：text_blocks 列表，每个块填充 font_family 字段

使用示例：
    from processors.font_family import FontFamilyProcessor
    
    processor = FontFamilyProcessor()
    blocks = processor.process(text_blocks, global_font="Arial")
"""

import re
import copy
from typing import List, Dict, Any, Optional


class FontFamilyProcessor:
    """
    字体处理器
    
    处理流程：
    1. extract_from_ocr: 从 OCR 结果提取字体
    2. standardize: 标准化字体名称
    3. infer_from_text: 根据文本内容推测字体
    4. unify_by_clustering: 空间聚类统一字体
    """
    
    # 代码风格关键字
    CODE_KEYWORDS = ["id_", "code_", "0x", "struct", "func_", "var_", "ptr_", 
                     "def ", "class ", "import ", "__", "::", "{}"]
    
    # 字体标准化映射表
    FONT_MAPPING = {
        # 中文无衬线体
        "microsoft yahei": "Microsoft YaHei",
        "微软雅黑": "Microsoft YaHei",
        "simhei": "SimHei",
        "黑体": "SimHei",
        "dengxian": "DengXian",
        "等线": "DengXian",
        
        # 英文无衬线体
        "arial": "Arial",
        "calibri": "Calibri",
        "verdana": "Verdana",
        "helvetica": "Helvetica",
        "roboto": "Roboto",
        
        # 衬线体
        "simsun": "SimSun",
        "宋体": "SimSun",
        "times new roman": "Times New Roman",
        "times": "Times New Roman",
        "georgia": "Georgia",
        "yu mincho": "SimSun",
        "ms mincho": "SimSun",
        
        # 等宽体
        "courier new": "Courier New",
        "courier": "Courier New",
        "consolas": "Courier New",
        "monaco": "Courier New",
        "menlo": "Courier New",
    }
    
    # 字体类别关键词
    SERIF_KEYWORDS = ["baskerville", "garamond", "palatino", "didot", "bodoni"]
    SANS_KEYWORDS = ["segoe", "tahoma", "trebuchet", "lucida"]
    MONO_KEYWORDS = ["mono", "consolas", "menlo", "monaco", "courier"]
    
    def __init__(self, default_font: str = "Arial"):
        """
        初始化字体处理器
        
        Args:
            default_font: 默认字体
        """
        self.default_font = default_font
        self.font_cache = {}
    
    def process(
        self,
        text_blocks: List[Dict[str, Any]],
        global_font: str = None,
        unify: bool = True
    ) -> List[Dict[str, Any]]:
        """
        处理字体（主入口）
        
        Args:
            text_blocks: 文字块列表
            global_font: 全局主字体（从最大文字块识别）
            unify: 是否执行聚类统一
            
        Returns:
            处理后的文字块列表
        """
        global_font = global_font or self.default_font
        result = []
        
        for block in text_blocks:
            block = copy.copy(block)
            
            # 已有字体则标准化
            if block.get("font_family"):
                block["font_family"] = self.standardize(block["font_family"])
            else:
                # 推测字体
                block["font_family"] = self.infer_from_text(
                    block.get("text", ""),
                    is_bold=block.get("is_bold", False),
                    is_latex=block.get("is_latex", False),
                    default_font=global_font
                )
            
            result.append(block)
        
        # 聚类统一
        if unify and len(result) > 1:
            result = self.unify_by_clustering(result)
        
        return result
    
    def standardize(self, font_name: str) -> str:
        """
        标准化字体名称
        
        策略：
        1. 精确匹配映射表
        2. 模糊匹配（如 ArialMT → Arial）
        3. 归类未知字体
        """
        if not font_name:
            return self.default_font
        
        # 清理
        original = font_name.strip()
        main_font = original.split(',')[0].strip()
        clean_name = main_font.lower()
        
        # 精确匹配
        if clean_name in self.FONT_MAPPING:
            return self.FONT_MAPPING[clean_name]
        
        # 模糊匹配
        for key, value in self.FONT_MAPPING.items():
            if key in clean_name:
                return value
        
        # 归类未知字体
        if any(kw in clean_name for kw in self.SERIF_KEYWORDS):
            return "Times New Roman"
        if any(kw in clean_name for kw in self.SANS_KEYWORDS):
            return "Arial"
        if any(kw in clean_name for kw in self.MONO_KEYWORDS):
            return "Courier New"
        
        # 保留原始字体
        return main_font
    
    def infer_from_text(
        self,
        text: str,
        is_bold: bool = False,
        is_latex: bool = False,
        default_font: str = None
    ) -> str:
        """
        根据文本内容推测字体
        
        推测规则（按优先级）：
        1. LaTeX 公式 → Times New Roman
        2. 中文字符 → SimSun
        3. 代码特征 → Courier New
        4. 其他 → default_font
        """
        default_font = default_font or self.default_font
        
        # 缓存
        cache_key = f"{text}_{is_bold}_{is_latex}"
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font = default_font
        
        # 公式
        if is_latex:
            font = "Times New Roman"
        # 中文
        elif re.search(r'[\u4e00-\u9fff]', text):
            font = "SimSun"
        # 代码
        elif self._is_code_text(text):
            font = "Courier New"
        # 学术文本
        elif self._is_academic_text(text):
            font = "Times New Roman"
        
        self.font_cache[cache_key] = font
        return font
    
    def _is_code_text(self, text: str) -> bool:
        """检测是否为代码风格文本"""
        text_lower = text.lower()
        
        # 关键字匹配
        if any(kw in text_lower for kw in self.CODE_KEYWORDS):
            return True
        
        # 变量名风格（含下划线且为单词）
        if '_' in text and len(text.split()) == 1:
            return True
        
        return False
    
    def _is_academic_text(self, text: str) -> bool:
        """检测是否为学术文本"""
        academic_keywords = ['figure', 'table', 'equation', 'result', 
                            'method', 'data', 'analysis']
        
        # 含学术关键词且为完整句子
        if any(kw in text.lower() for kw in academic_keywords) and len(text) > 10:
            return True
        
        # 完整句子（含空格、标点、适当长度）
        if ' ' in text and len(text) > 15 and any(p in text for p in ['.', ',', ';']):
            return True
        
        return False
    
    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold: float = 0.5,
        horizontal_threshold: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        通过空间聚类统一字体
        
        算法：
        1. 并查集聚类：将空间相邻的文字块分组
        2. 组内统一：使用多数投票选择字体
        """
        if not text_blocks:
            return text_blocks
        
        n = len(text_blocks)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 提取边界框
        boxes = []
        for block in text_blocks:
            polygon = block.get('polygon', [])
            if polygon and len(polygon) >= 4:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                boxes.append({
                    'x_min': min(xs), 'y_min': min(ys),
                    'x_max': max(xs), 'y_max': max(ys),
                    'width': max(xs) - min(xs),
                    'height': max(ys) - min(ys),
                    'font_family': block.get('font_family', '')
                })
            else:
                geo = block.get('geometry', {})
                boxes.append({
                    'x_min': geo.get('x', 0),
                    'y_min': geo.get('y', 0),
                    'x_max': geo.get('x', 0) + geo.get('width', 100),
                    'y_max': geo.get('y', 0) + geo.get('height', 20),
                    'width': geo.get('width', 100),
                    'height': geo.get('height', 20),
                    'font_family': block.get('font_family', '')
                })
        
        # 聚类
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(boxes[i], boxes[j], vertical_threshold, horizontal_threshold):
                    union(i, j)
        
        # 分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 多数投票统一
        result = copy.deepcopy(text_blocks)
        
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            
            # 统计字体出现次数
            font_counts = {}
            for idx in group_indices:
                font = result[idx].get('font_family', '')
                if font:
                    font_counts[font] = font_counts.get(font, 0) + 1
            
            if font_counts:
                # 选择出现最多的字体
                winner_font = max(font_counts.items(), key=lambda x: x[1])[0]
                for idx in group_indices:
                    result[idx]['font_family'] = winner_font
        
        return result
    
    def _should_merge(
        self, 
        box1: Dict, 
        box2: Dict,
        vertical_threshold: float,
        horizontal_threshold: float
    ) -> bool:
        """判断两个文字块是否应该合并"""
        # 字体必须相同或其中一个为空
        font1 = box1.get('font_family', '')
        font2 = box2.get('font_family', '')
        if font1 and font2 and font1 != font2:
            return False
        
        avg_height = (box1['height'] + box2['height']) / 2
        avg_width = (box1['width'] + box2['width']) / 2
        
        # 计算间距
        vertical_gap = min(
            abs(box1['y_min'] - box2['y_max']),
            abs(box2['y_min'] - box1['y_max'])
        )
        horizontal_gap = min(
            abs(box1['x_min'] - box2['x_max']),
            abs(box2['x_min'] - box1['x_max'])
        )
        
        # 垂直/水平重叠
        y_overlap = not (box1['y_max'] < box2['y_min'] or box2['y_max'] < box1['y_min'])
        x_overlap = not (box1['x_max'] < box2['x_min'] or box2['x_max'] < box1['x_min'])
        
        # 同一行或同一列
        same_row = (y_overlap or vertical_gap < avg_height * vertical_threshold) and \
                   horizontal_gap < avg_width * horizontal_threshold
        same_col = (x_overlap or horizontal_gap < avg_width * 0.3) and \
                   vertical_gap < avg_height * 1.5
        
        return same_row or same_col


if __name__ == "__main__":
    processor = FontFamilyProcessor()
    
    # 测试标准化
    print("=== 字体标准化测试 ===")
    test_fonts = ["ArialMT", "Times New Roman", "微软雅黑", "Consolas"]
    for font in test_fonts:
        print(f"  {font} → {processor.standardize(font)}")
    
    # 测试推测
    print("\n=== 字体推测测试 ===")
    test_texts = ["Hello World", "你好世界", "def main():", "Figure 1. Results"]
    for text in test_texts:
        print(f"  '{text}' → {processor.infer_from_text(text)}")
