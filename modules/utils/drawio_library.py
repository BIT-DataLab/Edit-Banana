"""
DrawIO 图元库与箭头样式检测

- ArrowAttributeDetector: 从图像/路径检测箭头属性（端点箭头、曲线类型等）
- build_arrow_style: 根据属性生成 DrawIO Edge 样式字符串
- 常量与辅助 API 供其它模块使用
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ---------- 常量（DrawIO 图元名） ----------
DRAWIO_BASIC_SHAPES = [
    "rectangle", "ellipse", "rhombus", "triangle", "hexagon", "parallelogram", "cylinder",
]
DRAWIO_ARROWS = ["classic", "block", "open", "diamond", "oval", "none"]
DRAWIO_ARROW_HEADS = ["classic", "block", "open", "diamond", "oval", "none"]
DRAWIO_EDGE_STYLES = ["orthogonalEdgeStyle", "straight", "entityRelationEdgeStyle", "loop", "elbowConnector"]
DRAWIO_DASH_PATTERNS = ["solid", "dashed", "dotted", "dashDot", "dashDotDot"]
DRAWIO_ARROW_SHAPES = ["classic", "block", "open", "diamond", "oval"]


class DrawIOLibrary:
    """DrawIO 图元库（占位，便于扩展）"""
    pass


class ArrowAttributeDetector:
    """
    从箭头裁剪图与路径点检测 DrawIO 箭头属性。
    供 ArrowProcessor 调用，返回 build_arrow_style 所需的 kwargs。
    """

    def __init__(self):
        pass

    def detect_all_attributes(
        self,
        image_crop: np.ndarray,
        path_points: Optional[List[List[int]]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        检测箭头属性。

        Args:
            image_crop: 箭头区域裁剪图 (H,W,3)
            path_points: 路径点 [[x,y], ...]，起点到终点
            mask: 可选 mask

        Returns:
            dict: start_arrow, start_fill, end_arrow, end_fill, stroke_color, stroke_width, ...
        """
        # 简单实现：默认单向箭头，末端 classic。上游会覆盖 start_arrow/start_fill、curve_type。
        color = self._sample_stroke_color(image_crop, path_points)
        return {
            "start_arrow": "none",
            "start_fill": False,
            "end_arrow": "classic",
            "end_fill": True,
            "stroke_color": color,
            "stroke_width": 2,
            "curve_type": "sharp",
        }

    def _sample_stroke_color(
        self,
        crop: np.ndarray,
        path_points: Optional[List[List[int]]],
    ) -> str:
        """从裁剪图粗略取色，返回 #RRGGBB"""
        try:
            h, w = crop.shape[:2]
            if crop.size == 0:
                return "#000000"
            # 取中心带区域中位数
            y0, y1 = max(0, h // 4), min(h, 3 * h // 4)
            x0, x1 = max(0, w // 4), min(w, 3 * w // 4)
            region = crop[y0:y1, x0:x1]
            if region.size == 0:
                return "#000000"
            r = int(np.median(region[:, :, 0]))
            g = int(np.median(region[:, :, 1]))
            b = int(np.median(region[:, :, 2]))
            from .color_utils import rgb_to_hex
            return rgb_to_hex(r, g, b)
        except Exception:
            return "#000000"


def build_arrow_style(
    start_arrow: str = "none",
    start_fill: bool = False,
    end_arrow: str = "classic",
    end_fill: bool = True,
    stroke_color: str = "#000000",
    stroke_width: int = 2,
    curve_type: str = "sharp",
    **kwargs: Any,
) -> str:
    """
    根据属性构建 DrawIO Edge 样式字符串。

    curve_type: sharp -> rounded=0; rounded -> rounded=1; curved -> curved=1, edgeStyle 可调。
    """
    parts = [
        "html=1",
        "edgeStyle=orthogonalEdgeStyle",
        "orthogonalLoop=1",
        "jettySize=auto",
        f"startArrow={start_arrow or 'none'}",
        "startFill=1" if start_fill else "startFill=0",
        f"endArrow={end_arrow or 'classic'}",
        "endFill=1" if end_fill else "endFill=0",
        f"strokeColor={stroke_color or '#000000'}",
        f"strokeWidth={stroke_width or 2}",
    ]
    if curve_type == "curved":
        parts.append("curved=1")
        # 可改为 edgeStyle=curved，这里保持正交以便 waypoints 生效
    elif curve_type == "rounded":
        parts.append("rounded=1")
    else:
        parts.append("rounded=0")
    return ";".join(parts)


def build_style_string(**attrs: Any) -> str:
    """通用样式键值转 DrawIO style 字符串"""
    return ";".join(f"{k}={v}" for k, v in attrs.items() if v is not None)


def get_drawio_style(element_type: str, **overrides: Any) -> str:
    """按图元类型返回默认 DrawIO 样式（可覆盖）"""
    base = {"shape": "rectangle", "strokeColor": "#000000", "fillColor": "#ffffff"}
    base.update(overrides)
    return build_style_string(**base)


def match_element_to_drawio(element_type: str) -> str:
    """将内部图元类型映射到 DrawIO shape 名"""
    m = {"rectangle": "rectangle", "ellipse": "ellipse", "triangle": "triangle", "arrow": "connector"}
    return m.get(element_type.lower(), "rectangle")


def detect_arrow_style(_image: np.ndarray, _path: Optional[List] = None) -> Dict[str, Any]:
    """占位：检测箭头样式"""
    return {"end_arrow": "classic", "start_arrow": "none", "curve_type": "sharp"}


def detect_arrow_attributes(_image: np.ndarray, _path: Optional[List] = None) -> Dict[str, Any]:
    """占位：同 detect_arrow_style"""
    return detect_arrow_style(_image, _path)


def get_all_arrow_head_types() -> List[str]:
    return list(DRAWIO_ARROW_HEADS)


def get_all_dash_patterns() -> List[str]:
    return list(DRAWIO_DASH_PATTERNS)


def get_all_edge_styles() -> List[str]:
    return list(DRAWIO_EDGE_STYLES)
