"""
坐标处理器模块
实现坐标归一化和基线锚定算法
"""

from dataclasses import dataclass
from typing import Optional
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# 注：不再使用固定的 CANVAS_WIDTH/CANVAS_HEIGHT，改用原图尺寸


@dataclass
class NormalizedCoords:
    """归一化后的坐标"""
    x: float  # 左上角 x
    y: float  # 左上角 y
    width: float  # 宽度
    height: float  # 高度
    baseline_y: float  # 基线 Y 坐标
    rotation: float  # 旋转角度（度）


class CoordProcessor:
    """坐标处理器"""
    
    def __init__(self, source_width: int, source_height: int,
                 canvas_width: int = None, canvas_height: int = None):
        """
        初始化坐标处理器
        
        Args:
            source_width: 原图宽度（像素）
            source_height: 原图高度（像素）
            canvas_width: 目标画布宽度（如果为 None，则使用原图宽度）
            canvas_height: 目标画布高度（如果为 None，则使用原图高度）
        """
        self.source_width = source_width
        self.source_height = source_height
        
        # 如果未指定目标画布尺寸，则使用原图尺寸（不缩放）
        self.canvas_width = canvas_width if canvas_width is not None else source_width
        self.canvas_height = canvas_height if canvas_height is not None else source_height
        
        # 计算缩放比例（保持宽高比）
        self.scale_x = self.canvas_width / source_width
        self.scale_y = self.canvas_height / source_height
        
        # 使用统一缩放比例以保持宽高比
        self.uniform_scale = min(self.scale_x, self.scale_y)
    
    def normalize_polygon(self, polygon: list[tuple[float, float]]) -> NormalizedCoords:
        """
        将多边形坐标归一化到目标画布
        
        核心思想：利用 OCR 多边形的中心点定位
        - 计算多边形中心点
        - draw.io 文本框中心点与 OCR 中心点对齐
        - 配合 align=center + verticalAlign=middle，文字自动居中显示
        
        Args:
            polygon: 原始多边形顶点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    
        Returns:
            NormalizedCoords: 归一化后的坐标
        """
        if len(polygon) < 4:
            return NormalizedCoords(0, 0, 0, 0, 0, 0)
        
        # 归一化各顶点
        normalized_points = [
            (p[0] * self.uniform_scale, p[1] * self.uniform_scale)
            for p in polygon
        ]
        
        # 提取各点
        p0, p1, p2, p3 = normalized_points[:4]
        
        # 计算旋转角度（基于顶部边的倾斜）
        rotation = self._calculate_rotation(p0, p1)
        
        # 计算 OCR 多边形的中心点（这是最重要的锚点！）
        center_x = sum(p[0] for p in normalized_points) / 4
        center_y = sum(p[1] for p in normalized_points) / 4
        
        # 计算两条相邻边的长度（文字的实际宽高）
        edge_top = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)  # 顶边
        edge_left = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)  # 左边
        
        # 判断是否是竖排文字（旋转角度接近 ±90°）
        is_vertical = abs(abs(rotation) - 90) < 15
        
        if is_vertical:
            # 竖排文字：
            # - edge_top (p0->p1) 是文字串长度（垂直方向）
            # - edge_left (p0->p3) 是单个字符高度（水平方向）
            # 
            # draw.io 机制：先放置 (width x height) 的框，再绕中心旋转
            # 旋转 -90° 后，视觉尺寸变成 (height x width)
            # 我们需要视觉尺寸 = (edge_left x edge_top) = (字符高度 x 文字串长度)
            # 所以 draw.io 的 width = edge_top, height = edge_left
            width = edge_top    # 文字串长度（旋转前的宽度）
            height = edge_left  # 字符高度（旋转前的高度）
        else:
            # 横排文字：正常使用
            width = edge_top    # 文字串长度
            height = edge_left  # 文字高度
        
        # 从中心点反推左上角坐标
        # draw.io 的 (x, y) 是左上角，中心点 = (x + width/2, y + height/2)
        x = center_x - width / 2
        y = center_y - height / 2
        
        # 计算基线（底部两点的 Y 坐标平均值）
        baseline_y = (p2[1] + p3[1]) / 2
        
        return NormalizedCoords(
            x=x,
            y=y,
            width=width,
            height=height,
            baseline_y=baseline_y,
            rotation=rotation
        )
    
    def _calculate_rotation(self, p0: tuple, p1: tuple) -> float:
        """
        计算文本的旋转角度（基于顶部边）
        
        Args:
            p0: 左上角点
            p1: 右上角点
            
        Returns:
            float: 旋转角度（度）
        """
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 小角度（< 2度）视为水平
        if abs(angle_deg) < 2:
            return 0.0
        
        return round(angle_deg, 1)
    
    def polygon_to_geometry(self, polygon: list[tuple[float, float]]) -> dict:
        """
        将多边形转换为 draw.io 的 geometry 格式
        
        Args:
            polygon: 原始多边形顶点
            
        Returns:
            dict: 包含 x, y, width, height 的字典
        """
        coords = self.normalize_polygon(polygon)
        
        return {
            "x": round(coords.x, 2),
            "y": round(coords.y, 2),
            "width": round(coords.width, 2),
            "height": round(coords.height, 2),
            "baseline_y": round(coords.baseline_y, 2),
            "rotation": coords.rotation
        }
    
    def anchor_to_baseline(self, geometry: dict) -> dict:
        """
        将几何信息锚定到基线
        
        在 draw.io 中，文本框的 y 坐标是左上角的位置
        为了让文本基线对齐，需要调整 y 坐标
        
        Args:
            geometry: 原始几何信息
            
        Returns:
            dict: 调整后的几何信息
        """
        # 基线通常在文本框底部约 15-20% 的位置（取决于字体）
        # 这里使用保守估计，假设基线在底部 18% 处
        baseline_offset_ratio = 0.18
        
        adjusted = geometry.copy()
        
        # 计算基线偏移量
        baseline_offset = geometry["height"] * baseline_offset_ratio
        
        # 调整 y 坐标，使基线位于目标位置
        adjusted["y_baseline_adjusted"] = geometry["baseline_y"] - geometry["height"] + baseline_offset
        
        return adjusted
    
    def batch_normalize(self, polygons: list[list[tuple[float, float]]]) -> list[dict]:
        """
        批量归一化多个多边形
        
        Args:
            polygons: 多边形列表
            
        Returns:
            list[dict]: 几何信息列表
        """
        return [self.polygon_to_geometry(p) for p in polygons]


def create_processor(source_width: int, source_height: int) -> CoordProcessor:
    """
    便捷函数：创建坐标处理器
    
    Args:
        source_width: 原图宽度
        source_height: 原图高度
        
    Returns:
        CoordProcessor: 坐标处理器实例
    """
    return CoordProcessor(source_width, source_height)


if __name__ == "__main__":
    # 测试代码
    processor = CoordProcessor(source_width=2000, source_height=1500)
    
    # 模拟一个矩形文本框的多边形
    # 左上、右上、右下、左下
    test_polygon = [
        (100, 200),   # 左上
        (300, 200),   # 右上
        (300, 250),   # 右下
        (100, 250)    # 左下
    ]
    
    result = processor.normalize_polygon(test_polygon)
    print(f"归一化结果:")
    print(f"  位置: ({result.x:.2f}, {result.y:.2f})")
    print(f"  尺寸: {result.width:.2f} x {result.height:.2f}")
    print(f"  基线 Y: {result.baseline_y:.2f}")
    print(f"  旋转: {result.rotation}°")
    
    geometry = processor.polygon_to_geometry(test_polygon)
    print(f"\nGeometry 格式: {geometry}")
    
    anchored = processor.anchor_to_baseline(geometry)
    print(f"基线锚定后: {anchored}")

