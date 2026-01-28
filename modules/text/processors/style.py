"""
样式处理器模块

功能：
    1. 从 Azure OCR 提取加粗、斜体、颜色等样式
    2. 空间聚类统一样式（解决同一行样式不一致问题）

负责人：[填写负责人姓名]

接口说明：
    输入：text_blocks 列表，每个块包含 spans 信息
    输出：text_blocks 列表，每个块填充 font_weight, font_style, font_color 字段

使用示例：
    from processors.style import StyleProcessor
    
    processor = StyleProcessor()
    blocks = processor.process(text_blocks, azure_styles)
"""

import copy
from typing import List, Dict, Any, Optional


class StyleProcessor:
    """
    样式处理器
    
    处理的样式属性：
    - font_weight: normal / bold
    - font_style: normal / italic
    - font_color: 十六进制颜色（如 #1d1d1d）
    - background_color: 背景颜色
    """
    
    def __init__(self):
        pass
    
    def process(
        self,
        text_blocks: List[Dict[str, Any]],
        azure_styles: List[Dict] = None,
        unify: bool = True
    ) -> List[Dict[str, Any]]:
        """
        处理样式（主入口）
        
        Args:
            text_blocks: 文字块列表
            azure_styles: Azure 返回的全局 styles 列表
            unify: 是否执行聚类统一
            
        Returns:
            处理后的文字块列表
        """
        azure_styles = azure_styles or []
        
        # 步骤 1: 提取样式
        result = self.extract_styles(text_blocks, azure_styles)
        
        # 步骤 2: 聚类统一
        if unify and len(result) > 1:
            result = self.unify_by_clustering(result)
        
        return result
    
    def extract_styles(
        self,
        text_blocks: List[Dict[str, Any]],
        azure_styles: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        从文字块和 Azure styles 中提取样式
        
        优先级：
        1. 文字块自身属性
        2. Azure styles 的 spans 匹配
        """
        result = []
        
        for block in text_blocks:
            block = copy.copy(block)
            
            # 提取样式
            styles = self._extract_block_styles(block, azure_styles)
            
            # 应用样式
            block["font_weight"] = "bold" if styles["is_bold"] else "normal"
            block["font_style"] = "italic" if styles["is_italic"] else "normal"
            block["is_bold"] = styles["is_bold"]
            block["is_italic"] = styles["is_italic"]
            
            if styles["color"]:
                block["font_color"] = styles["color"]
            if styles["background_color"]:
                block["background_color"] = styles["background_color"]
            
            result.append(block)
        
        return result
    
    def _extract_block_styles(
        self, 
        block: Dict[str, Any], 
        azure_styles: List[Dict]
    ) -> Dict[str, Any]:
        """
        提取单个文字块的样式
        """
        styles = {
            "is_bold": False,
            "is_italic": False,
            "color": None,
            "background_color": None
        }
        
        # 优先使用 block 自身属性
        if block.get("font_weight") == "bold" or block.get("is_bold"):
            styles["is_bold"] = True
        if block.get("font_style") == "italic" or block.get("is_italic"):
            styles["is_italic"] = True
        if block.get("font_color"):
            styles["color"] = block["font_color"]
        if block.get("background_color"):
            styles["background_color"] = block["background_color"]
        
        # 如果已有足够信息，直接返回
        has_info = styles["is_bold"] or styles["is_italic"] or styles["color"]
        if has_info or not azure_styles:
            return styles
        
        # 从 azure_styles 匹配
        block_spans = block.get("spans", [])
        if not block_spans:
            return styles
        
        block_offset = block_spans[0].get("offset", 0) if isinstance(block_spans[0], dict) else 0
        block_length = block_spans[0].get("length", 0) if isinstance(block_spans[0], dict) else 0
        
        for style in azure_styles:
            style_spans = style.get("spans", [])
            
            for span in style_spans:
                span_offset = span.get("offset", 0)
                span_length = span.get("length", 0)
                
                # 检查是否重叠
                if self._spans_overlap(block_offset, block_length, span_offset, span_length):
                    if style.get("fontWeight") == "bold":
                        styles["is_bold"] = True
                    if style.get("fontStyle") == "italic":
                        styles["is_italic"] = True
                    if style.get("color") and not styles["color"]:
                        styles["color"] = style["color"]
                    if style.get("backgroundColor") and not styles["background_color"]:
                        styles["background_color"] = style["backgroundColor"]
        
        return styles
    
    def _spans_overlap(
        self, 
        offset1: int, 
        length1: int, 
        offset2: int, 
        length2: int
    ) -> bool:
        end1 = offset1 + length1
        end2 = offset2 + length2
        return not (end1 <= offset2 or end2 <= offset1)
    
    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold: float = 0.8,  # 收紧阈值
        horizontal_threshold: float = 0.5  # 收紧阈值
    ) -> List[Dict[str, Any]]:
        """
        通过空间聚类统一样式
        
        注意：
        - 加粗/斜体 **不做** 聚类统一，保留 OCR 原始值
        - 只对颜色做聚类统一（同一区域的文字颜色通常一致）
        
        原因：
        - 加粗/斜体是文字本身的属性，不应被空间位置影响
        - 之前的问题：并查集传递性导致几乎所有块都被合并到一个组
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
        boxes = self._extract_boxes(text_blocks)
        
        # 聚类（使用更严格的条件）
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge_for_color(boxes[i], boxes[j], vertical_threshold, horizontal_threshold):
                    union(i, j)
        
        # 分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 只统一颜色
        result = copy.deepcopy(text_blocks)
        
        multi_groups = [g for g in groups.values() if len(g) > 1]
        if multi_groups:
            print(f"     发现 {len(multi_groups)} 个需要统一颜色的组（包含 {sum(len(g) for g in multi_groups)} 个文字块）")
        
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            
            # 只统计颜色
            color_counts = {}
            for idx in group_indices:
                blk = result[idx]
                color = blk.get("font_color")
                if color:
                    color_counts[color] = color_counts.get(color, 0) + 1
            
            # 颜色多数投票
            if color_counts:
                winner_color = max(color_counts.items(), key=lambda x: x[1])[0]
                for idx in group_indices:
                    result[idx]["font_color"] = winner_color
        
        # 加粗/斜体保留原始值，不做统一
        # （已经在 extract_styles 中设置好了）
        
        return result
    
    def _should_merge_for_color(
        self, 
        box1: Dict, 
        box2: Dict,
        vertical_threshold: float,
        horizontal_threshold: float
    ) -> bool:
        """
        判断两个文字块是否应该合并（用于颜色统一）
        
        使用更严格的条件：必须在同一行（垂直重叠）
        """
        avg_height = (box1['height'] + box2['height']) / 2
        avg_width = (box1['width'] + box2['width']) / 2
        
        vertical_gap = min(
            abs(box1['y_min'] - box2['y_max']),
            abs(box2['y_min'] - box1['y_max'])
        )
        horizontal_gap = min(
            abs(box1['x_min'] - box2['x_max']),
            abs(box2['x_min'] - box1['x_max'])
        )
        
        # 必须垂直重叠（真正的同一行）
        y_overlap = not (box1['y_max'] < box2['y_min'] or box2['y_max'] < box1['y_min'])
        
        # 同一行：垂直重叠 + 水平距离近
        same_row = y_overlap and horizontal_gap < avg_width * horizontal_threshold
        
        return same_row
    
    def _extract_boxes(self, text_blocks: List[Dict]) -> List[Dict]:
        """提取文字块的边界框"""
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
                    'height': max(ys) - min(ys)
                })
            else:
                geo = block.get('geometry', {})
                boxes.append({
                    'x_min': geo.get('x', 0),
                    'y_min': geo.get('y', 0),
                    'x_max': geo.get('x', 0) + geo.get('width', 100),
                    'y_max': geo.get('y', 0) + geo.get('height', 20),
                    'width': geo.get('width', 100),
                    'height': geo.get('height', 20)
                })
        return boxes
    
    def _should_merge(
        self, 
        box1: Dict, 
        box2: Dict,
        vertical_threshold: float,
        horizontal_threshold: float
    ) -> bool:
        """判断两个文字块是否应该合并"""
        avg_height = (box1['height'] + box2['height']) / 2
        avg_width = (box1['width'] + box2['width']) / 2
        
        vertical_gap = min(
            abs(box1['y_min'] - box2['y_max']),
            abs(box2['y_min'] - box1['y_max'])
        )
        horizontal_gap = min(
            abs(box1['x_min'] - box2['x_max']),
            abs(box2['x_min'] - box1['x_max'])
        )
        
        y_overlap = not (box1['y_max'] < box2['y_min'] or box2['y_max'] < box1['y_min'])
        x_overlap = not (box1['x_max'] < box2['x_min'] or box2['x_max'] < box1['x_min'])
        
        same_row = (y_overlap or vertical_gap < avg_height * 0.5) and \
                   horizontal_gap < avg_width * horizontal_threshold
        same_col = (x_overlap or horizontal_gap < avg_width * 0.3) and \
                   vertical_gap < avg_height * vertical_threshold
        
        return same_row or same_col


if __name__ == "__main__":
    processor = StyleProcessor()
    
    test_blocks = [
        {"text": "Hello", "font_weight": "bold", "font_color": "#ff0000"},
        {"text": "World", "is_bold": True},
        {"text": "Test", "font_style": "italic"},
    ]
    
    result = processor.process(test_blocks, unify=False)
    for block in result:
        print(f"{block['text']}: bold={block['is_bold']}, italic={block['is_italic']}, color={block.get('font_color')}")
