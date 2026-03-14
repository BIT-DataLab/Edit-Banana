"""
Coordinate processor: convert OCR polygon to draw.io geometry (x, y, width, height, rotation).
"""

import math
from dataclasses import dataclass


@dataclass
class NormalizedCoords:
    """Normalized geometry: x, y, width, height, baseline_y, rotation (degrees)."""
    x: float
    y: float
    width: float
    height: float
    baseline_y: float
    rotation: float


class CoordProcessor:
    """Convert source image coordinates to canvas geometry with optional scaling."""

    def __init__(self, source_width: int, source_height: int,
                 canvas_width: int = None, canvas_height: int = None):
        self.source_width = source_width
        self.source_height = source_height
        self.canvas_width = canvas_width if canvas_width is not None else source_width
        self.canvas_height = canvas_height if canvas_height is not None else source_height
        self.scale_x = self.canvas_width / source_width
        self.scale_y = self.canvas_height / source_height
        self.uniform_scale = min(self.scale_x, self.scale_y)

    def normalize_polygon(self, polygon: list[tuple[float, float]]) -> NormalizedCoords:
        """Normalize quadrilateral to canvas geometry."""
        if len(polygon) < 4:
            return NormalizedCoords(0, 0, 0, 0, 0, 0)
        
        # 缩放坐标
        normalized_points = [
            (p[0] * self.uniform_scale, p[1] * self.uniform_scale)
            for p in polygon
        ]
        
        p0, p1, p2, p3 = normalized_points[:4]
        rotation = self._calculate_rotation(p0, p1)
        center_x = sum(p[0] for p in normalized_points) / 4
        center_y = sum(p[1] for p in normalized_points) / 4
        edge_top = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        edge_left = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
        is_vertical = abs(abs(rotation) - 90) < 15
        
        if is_vertical:
            width = edge_top
            height = edge_left
        else:
            width = edge_top
            height = edge_left
        
        # 计算左上角坐标（draw.io 从左上角定位）
        x = center_x - width / 2
        y = center_y - height / 2
        
        # 计算基线位置
        baseline_y = (p2[1] + p3[1]) / 2
        
        return NormalizedCoords(
            x=x, y=y, width=width, height=height,
            baseline_y=baseline_y, rotation=rotation
        )
    
    def _calculate_rotation(self, p0: tuple, p1: tuple) -> float:
        """
        计算旋转角度
        
        通过上边（P0 到 P1）的方向向量计算旋转角度。
        水平向右为 0°，顺时针旋转为正角度。
        
        Args:
            p0: 左上角坐标
            p1: 右上角坐标
            
        Returns:
            旋转角度（度）
        """
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 小角度忽略（避免微小误差）
        if abs(angle_deg) < 2:
            return 0.0
        
        return round(angle_deg, 1)
    
    def polygon_to_geometry(self, polygon: list[tuple[float, float]]) -> dict:
        """
        将多边形转换为 draw.io geometry 格式
        
        Args:
            polygon: 四边形顶点坐标
            
        Returns:
            dict: {"x", "y", "width", "height", "baseline_y", "rotation"}
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
if __name__ == "__main__":
    # 测试代码
    processor = CoordProcessor(source_width=2000, source_height=1500)
    
    # 测试横排文本框
    test_polygon = [
        (100, 200),   # 左上
        (300, 200),   # 右上
        (300, 250),   # 右下
        (100, 250)    # 左下
    ]
    
    result = processor.normalize_polygon(test_polygon)
    print(f"横排文本归一化结果:")
    print(f"  位置: ({result.x:.2f}, {result.y:.2f})")
    print(f"  尺寸: {result.width:.2f} x {result.height:.2f}")
    print(f"  旋转: {result.rotation}°")
    
    geometry = processor.polygon_to_geometry(test_polygon)
    print(f"  Geometry: {geometry}")