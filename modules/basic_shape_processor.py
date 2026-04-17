"""
任务3：基本图形处理模块

功能：
    - 处理rectangle、ellipse、diamond等基本图形
    - 从图片中提取填充色和描边色
    - 检测边框宽度
    - 用XML描述这些图形
    - 支持CV补充检测（检测SAM3遗漏的矩形/容器）
    - 输出XML片段

负责人：[已实现]
负责任务：任务3 - 基本图形类（取色，用XML描述）

使用示例：
    from modules import BasicShapeProcessor, ProcessingContext
    
    processor = BasicShapeProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # 从SAM3获取的元素
    
    result = processor.process(context)
    # 处理后的元素会包含 fill_color, stroke_color, xml_fragment 字段

接口说明：
    输入：
        - context.elements: ElementInfo列表，筛选出基本图形
        - context.image_path: 原始图片路径，用于取色
        
    输出：
        - 更新 element.fill_color: 填充颜色（十六进制）
        - 更新 element.stroke_color: 描边颜色（十六进制）
        - 更新 element.stroke_width: 描边宽度
        - 更新 element.xml_fragment: 该元素的XML片段
"""

import os
import cv2
import numpy as np
import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult, LayerLevel, get_layer_level


# ======================== DrawIO样式配置 ========================
DRAWIO_STYLES = {
    "rectangle": "rounded=0;whiteSpace=wrap;html=1;",
    "rounded rectangle": "rounded=1;whiteSpace=wrap;html=1;",
    "title_bar": "rounded=0;whiteSpace=wrap;html=1;fillColor=#E6E6E6;",
    "section_panel": "rounded=0;whiteSpace=wrap;html=1;dashed=1;dashPattern=1 1;",
    "container": "rounded=1;whiteSpace=wrap;html=1;",
    "diamond": "rhombus;whiteSpace=wrap;html=1;",
    "ellipse": "ellipse;whiteSpace=wrap;html=1;",
    "circle": "ellipse;whiteSpace=wrap;html=1;",
    "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;",
    "cloud": "ellipse;shape=cloud;whiteSpace=wrap;html=1;",
    "actor": "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;",
    "hexagon": "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;",
    "triangle": "triangle;whiteSpace=wrap;html=1;",
    "parallelogram": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;",
    "trapezoid": "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;",
    "square": "rounded=0;whiteSpace=wrap;html=1;aspect=fixed;",
}

# 支持矢量化的图形类型
VECTOR_TYPES = {
    "rectangle", "rounded_rectangle", "rounded rectangle",
    "diamond", "ellipse", "circle",
    "cylinder", "cloud", "actor",
    "hexagon", "triangle", "parallelogram",
    "title_bar", "section_panel", "container",
    "trapezoid", "square"
}


# ======================== 几何参数提取 ========================
def extract_geometric_params(image: np.ndarray, bbox: list, shape_type: str) -> dict:
    """
    针对特定形状提取几何参数（如平行四边形的倾斜度、圆柱体的顶部高度等）。
    返回参数字典，例如 {"size": 0.2, "direction": "south"}
    """
    params = {}
    x1, y1, x2, y2 = map(int, bbox)
    w_box, h_box = x2 - x1, y2 - y1
    
    if w_box <= 0 or h_box <= 0:
        return params

    # 提取 ROI 用于分析
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 通用预处理：获取轮廓
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            main_cnt = cnt
            
    if main_cnt is None:
        return params

    # 针对不同形状的分析
    if shape_type == "parallelogram":
        # 计算倾斜比例 size (0~1)
        epsilon = 0.02 * cv2.arcLength(main_cnt, True)
        approx = cv2.approxPolyDP(main_cnt, epsilon, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            pts = pts[np.argsort(pts[:, 1])]
            top_pts = pts[:2]
            bottom_pts = pts[2:]
            
            top_pts = top_pts[np.argsort(top_pts[:, 0])]
            bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
            
            tl, tr = top_pts[0], top_pts[1]
            bl, br = bottom_pts[0], bottom_pts[1]
            
            dx = abs(tl[0] - bl[0])
            size_val = dx / w_box if w_box > 0 else 0.2
            params["size"] = max(0.05, min(0.5, size_val))
            
    elif shape_type == "cylinder":
        params["size"] = max(10, int(w_box * 0.15))
        
    elif shape_type == "triangle":
        epsilon = 0.04 * cv2.arcLength(main_cnt, True)
        approx = cv2.approxPolyDP(main_cnt, epsilon, True)
        
        if len(approx) == 3:
            M = cv2.moments(main_cnt)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                rel_cy = cy / h_box
                if rel_cy > 0.55:
                    params["direction"] = "north"
                elif rel_cy < 0.45:
                    params["direction"] = "south"
                else:
                    cx = int(M["m10"] / M["m00"])
                    rel_cx = cx / w_box
                    if rel_cx > 0.55:
                        params["direction"] = "west"
                    elif rel_cx < 0.45:
                        params["direction"] = "east"

    return params


# ======================== IoU计算 ========================
def calculate_iou(box1: list, box2: list) -> float:
    """计算两个矩形框的 IoU"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


# ======================== 边框宽度检测 ========================
def calculate_stroke_width(image: np.ndarray, bbox: list, max_width: int = 8) -> int:
    """
    计算边框粗细 (Stroke Width)
    逻辑：沿四边向内扫描，寻找颜色突变点，多个采样点综合取中位数。
    
    优化：
    - 提高突变阈值（35），减少误检
    - 限制最大宽度（8像素），避免过粗
    - 大多数边框在 1-5 像素
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 1
    
    roi_h, roi_w = roi.shape[:2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    scan_limit = min(max_width, roi_w // 2 - 1, roi_h // 2 - 1)
    if scan_limit < 1:
        return 1
    
    detected_widths = []
    
    def scan_line(pixels, limit):
        if len(pixels) < limit + 1:
            return None
        diffs = np.abs(np.diff(pixels[:limit+2].astype(int)))
        threshold = 35  # 提高阈值，减少误检（原值20太敏感）
        candidates = np.where(diffs > threshold)[0]
        if len(candidates) > 0:
            return candidates[0] + 1
        return None

    num_samples = 5
    
    # Top Edge
    for i in range(1, num_samples + 1):
        x = int(roi_w * i / (num_samples + 1))
        col = roi_gray[:, x]
        w_val = scan_line(col, scan_limit)
        if w_val:
            detected_widths.append(w_val)
        
    # Bottom Edge
    for i in range(1, num_samples + 1):
        x = int(roi_w * i / (num_samples + 1))
        col = roi_gray[::-1, x]
        w_val = scan_line(col, scan_limit)
        if w_val:
            detected_widths.append(w_val)
        
    # Left Edge
    for i in range(1, num_samples + 1):
        y = int(roi_h * i / (num_samples + 1))
        row = roi_gray[y, :]
        w_val = scan_line(row, scan_limit)
        if w_val:
            detected_widths.append(w_val)

    # Right Edge
    for i in range(1, num_samples + 1):
        y = int(roi_h * i / (num_samples + 1))
        row = roi_gray[y, ::-1]
        w_val = scan_line(row, scan_limit)
        if w_val:
            detected_widths.append(w_val)
        
    if not detected_widths:
        return 1
        
    final_width = int(np.median(detected_widths))
    # 限制合理范围：大多数边框在 1-2 像素（降低上限以匹配原图）
    return max(1, min(final_width, 2))


# ======================== 颜色提取 ========================
def extract_style_colors(image: np.ndarray, bbox: list) -> tuple:
    """
    精细化取色逻辑：区分 边框区域(Stroke) 和 内部区域(Fill)
    
    优化策略：
    1. Fill: 采样边框内侧的"回"字形区域（避开中心可能的文字），使用K-Means聚类找主色
    2. Stroke: 提取边界框外围10%区域，取最暗的25%像素的均值作为边框色
    3. 同时返回检测到的边框宽度
    
    :param image: BGR格式的OpenCV图像
    :param bbox: [x1, y1, x2, y2]
    :return: (fill_color_hex, stroke_color_hex, stroke_width)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box, w_box = y2 - y1, x2 - x1
    
    # 截取ROI
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "#ffffff", "#000000", 1
    
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # --- 0. 检测边框宽度 ---
    max_w = min(15, w_box // 5, h_box // 5)
    stroke_width = calculate_stroke_width(image, bbox, max_width=max(1, max_w))
    
    # --- 1. 提取填充色 (Fill Color) ---
    # 优化：采样边框内侧的"回"字形区域，避开中心可能存在的文字
    s_w = int(stroke_width)
    border_padding = max(2, s_w + 2)
    sample_depth = max(5, min(20, w_box // 10, h_box // 10))
    
    fill_samples = []
    
    if w_box > 2 * (border_padding + sample_depth) and h_box > 2 * (border_padding + sample_depth):
        # Top strip (边框内侧上方)
        fill_samples.append(roi_rgb[border_padding:border_padding+sample_depth, border_padding:w_box-border_padding])
        # Bottom strip (边框内侧下方)
        fill_samples.append(roi_rgb[h_box-border_padding-sample_depth:h_box-border_padding, border_padding:w_box-border_padding])
        # Left strip (边框内侧左侧)
        fill_samples.append(roi_rgb[border_padding:h_box-border_padding, border_padding:border_padding+sample_depth])
        # Right strip (边框内侧右侧)
        fill_samples.append(roi_rgb[border_padding:h_box-border_padding, w_box-border_padding-sample_depth:w_box-border_padding])
    else:
        # Fallback: 区域太小，取中心区域
        margin_x = min(int(stroke_width + 2), w_box // 2 - 1)
        margin_y = min(int(stroke_width + 2), h_box // 2 - 1)
        if margin_x > 0 and margin_y > 0:
            fill_samples.append(roi_rgb[margin_y:h_box-margin_y, margin_x:w_box-margin_x])
        else:
            fill_samples.append(roi_rgb)

    # 合并所有采样像素
    if fill_samples:
        valid_samples = [s.reshape(-1, 3) for s in fill_samples if s.size > 0]
        if valid_samples:
            inner_pixels = np.concatenate(valid_samples)
        else:
            inner_pixels = roi_rgb.reshape(-1, 3)
    else:
        inner_pixels = roi_rgb.reshape(-1, 3)

    if inner_pixels.size == 0:
        inner_pixels = roi_rgb.reshape(-1, 3)

    # 使用K-Means聚类找主色（比中位数更准确）
    fill_rgb = np.median(inner_pixels, axis=0).astype(int)  # 默认用中位数
    
    if len(inner_pixels) > 200:
        try:
            # 降采样以提高速度
            if len(inner_pixels) > 2000:
                indices = np.random.choice(len(inner_pixels), 2000, replace=False)
                pixels_for_kmeans = inner_pixels[indices].astype(np.float32)
            else:
                pixels_for_kmeans = inner_pixels.astype(np.float32)
                
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 2  # 假设背景+前景杂噪
            _, labels, centers = cv2.kmeans(pixels_for_kmeans, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            dominant_idx = np.argmax(counts)
            fill_rgb = centers[dominant_idx].astype(int)
        except Exception:
            pass  # 保持中位数结果
    
    # --- 2. 提取描边色 (Stroke Color) ---
    border_w = max(2, stroke_width)
    
    top = roi_rgb[:border_w, :]
    bottom = roi_rgb[h_box-border_w:, :]
    left = roi_rgb[:, :border_w]
    right = roi_rgb[:, w_box-border_w:]
    
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)
    
    if border_pixels.size > 0:
        # 计算亮度 (Luminance): L = 0.299*R + 0.587*G + 0.114*B
        luminance = np.dot(border_pixels, [0.299, 0.587, 0.114])
        # 提取最暗的 25% 像素 (假设边框比背景深)
        dark_threshold = np.percentile(luminance, 25)
        darker_pixels = border_pixels[luminance <= dark_threshold]
        
        if len(darker_pixels) > 0:
            stroke_rgb = np.mean(darker_pixels, axis=0).astype(int)
        else:
            stroke_rgb = np.mean(border_pixels, axis=0).astype(int)
    else:
        stroke_rgb = np.array([0, 0, 0])

    # RGB -> Hex
    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    # 限制 stroke_width 在合理范围内（1-3），避免过粗的边框
    stroke_width = min(3, max(1, stroke_width))
    
    return rgb2hex(fill_rgb), rgb2hex(stroke_rgb), stroke_width


def extract_style_specific(image: np.ndarray, bbox: list, shape_type: str) -> dict:
    """
    针对不同基础形状的特定取色和边框算法。
    
    - 对于矩形类形状，使用动态边框宽度检测
    - 对于非矩形形状（椭圆、菱形等），使用Mask提取更准确的填充色
    """
    fill_hex, stroke_hex, stroke_w = extract_style_colors(image, bbox)
    
    # 针对非矩形形状，使用 Mask 提取更准确的填充色
    if shape_type in ["ellipse", "cloud", "circle", "diamond", "triangle", "hexagon"]:
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:
            h, w = roi.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if shape_type in ["ellipse", "cloud", "circle"]:
                cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
            elif shape_type == "diamond":
                pts = np.array([[w//2, 0], [w, h//2], [w//2, h], [0, h//2]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            elif shape_type == "triangle":
                pts = np.array([[w//2, 0], [w, h], [0, h]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            elif shape_type == "hexagon":
                pts = np.array([
                    [w//4, 0], [w*3//4, 0], 
                    [w, h//2], 
                    [w*3//4, h], [w//4, h], 
                    [0, h//2]
                ], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            # 腐蚀掉边缘区域
            kernel_size = max(3, stroke_w * 2 + 1)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel)
            
            if cv2.countNonZero(mask) > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                masked_pixels = roi_rgb[mask > 0]
                masked_pixels = masked_pixels.reshape(-1, 3)
                
                if masked_pixels.size > 0:
                    fill_rgb = np.median(masked_pixels, axis=0).astype(int)
                    fill_hex = "#{:02x}{:02x}{:02x}".format(*map(int, fill_rgb))

    geo_params = extract_geometric_params(image, bbox, shape_type)

    return {
        "fill_color": fill_hex,
        "stroke_color": stroke_hex,
        "stroke_width": stroke_w,
        "geo_params": geo_params
    }


# ======================== Mask精确取色 ========================
def extract_color_with_mask(image: np.ndarray, bbox: list, mask: np.ndarray,
                            shape_type: str = "unknown") -> dict:
    """
    使用SAM3提供的Mask进行精确取色
    
    Args:
        image: BGR格式的OpenCV图像
        bbox: [x1, y1, x2, y2] 边界框
        mask: SAM3提供的二值掩码 (full size or cropped)
        shape_type: 形状类型
        
    Returns:
        {
            'fill_color': '#xxxxxx',
            'stroke_color': '#xxxxxx', 
            'stroke_width': int,
            'geo_params': dict,
            'has_gradient': bool,
            'gradient_info': dict or None
        }
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    roi = image[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    h_roi, w_roi = roi.shape[:2]
    
    if h_roi == 0 or w_roi == 0:
        return {
            'fill_color': '#ffffff',
            'stroke_color': '#000000',
            'stroke_width': 1,
            'geo_params': {},
            'has_gradient': False,
            'gradient_info': None
        }
    
    # 处理Mask：确保和ROI尺寸一致
    if mask is not None and mask.size > 0:
        # 如果mask是全图尺寸，裁剪到ROI
        if mask.shape[0] == h_img and mask.shape[1] == w_img:
            mask_crop = mask[y1:y2, x1:x2]
        elif mask.shape[0] == h_roi and mask.shape[1] == w_roi:
            mask_crop = mask
        else:
            # 尺寸不匹配，resize
            mask_crop = cv2.resize(mask.astype(np.uint8), (w_roi, h_roi))
        
        # 二值化
        if mask_crop.max() > 1:
            mask_crop = (mask_crop > 127).astype(np.uint8)
        else:
            mask_crop = mask_crop.astype(np.uint8)
    else:
        # 没有mask，创建全1掩码
        mask_crop = np.ones((h_roi, w_roi), dtype=np.uint8)
    
    # =========== 1. 使用Mask精确提取填充色 ===========
    # 腐蚀Mask去除边框区域
    kernel_erode = np.ones((5, 5), np.uint8)
    inner_mask = cv2.erode(mask_crop, kernel_erode, iterations=2)
    
    # 提取内部像素
    if cv2.countNonZero(inner_mask) > 10:
        fill_pixels = roi_rgb[inner_mask > 0]
    else:
        fill_pixels = roi_rgb[mask_crop > 0] if cv2.countNonZero(mask_crop) > 0 else roi_rgb.reshape(-1, 3)
    
    if len(fill_pixels) > 0:
        fill_pixels = fill_pixels.reshape(-1, 3)
        
        # K-Means找主色（比中位数更准确）
        if len(fill_pixels) > 50:
            try:
                pixels_f32 = fill_pixels.astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = min(3, len(fill_pixels) // 20)
                k = max(2, k)
                _, labels, centers = cv2.kmeans(pixels_f32, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                
                # 选择占比最大的颜色
                counts = np.bincount(labels.flatten())
                dominant_idx = np.argmax(counts)
                fill_rgb = centers[dominant_idx].astype(int)
            except Exception:
                fill_rgb = np.median(fill_pixels, axis=0).astype(int)
        else:
            fill_rgb = np.median(fill_pixels, axis=0).astype(int)
    else:
        fill_rgb = np.array([255, 255, 255])
    
    # =========== 2. 使用Mask边缘提取描边色 ===========
    # 获取Mask边缘
    edge_mask = mask_crop - inner_mask
    edge_mask = np.maximum(edge_mask, 0).astype(np.uint8)
    
    # 如果边缘太薄，膨胀一下
    if cv2.countNonZero(edge_mask) < 50:
        kernel_edge = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask_crop, kernel_edge, iterations=1)
        edge_mask = dilated_mask - cv2.erode(mask_crop, kernel_edge, iterations=1)
        edge_mask = np.maximum(edge_mask, 0).astype(np.uint8)
    
    if cv2.countNonZero(edge_mask) > 5:
        stroke_pixels = roi_rgb[edge_mask > 0].reshape(-1, 3)
        
        # 取最暗的像素作为描边色
        if len(stroke_pixels) > 0:
            luminance = np.dot(stroke_pixels, [0.299, 0.587, 0.114])
            dark_threshold = np.percentile(luminance, 30)
            dark_pixels = stroke_pixels[luminance <= dark_threshold]
            
            if len(dark_pixels) > 0:
                stroke_rgb = np.mean(dark_pixels, axis=0).astype(int)
            else:
                stroke_rgb = np.mean(stroke_pixels, axis=0).astype(int)
        else:
            stroke_rgb = np.array([0, 0, 0])
    else:
        stroke_rgb = np.array([0, 0, 0])
    
    # =========== 3. 估算描边宽度 ===========
    # 基于Mask边缘厚度，限制在1-3范围内
    if cv2.countNonZero(edge_mask) > 0:
        # 计算边缘区域的平均厚度
        dist_transform = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        stroke_width = max(1, min(3, int(max_dist * 0.15)))  # 限制最大为3
    else:
        stroke_width = 1
    
    # =========== 4. 检测渐变 ===========
    has_gradient = False
    gradient_info = None
    
    if len(fill_pixels) > 100:
        # 将填充区域分为上下/左右两半，比较颜色差异
        coords = np.argwhere(inner_mask > 0 if cv2.countNonZero(inner_mask) > 10 else mask_crop > 0)
        if len(coords) > 20:
            mid_y = (coords[:, 0].min() + coords[:, 0].max()) // 2
            mid_x = (coords[:, 1].min() + coords[:, 1].max()) // 2
            
            # 上下分区
            top_coords = coords[coords[:, 0] < mid_y]
            bottom_coords = coords[coords[:, 0] >= mid_y]
            
            if len(top_coords) > 10 and len(bottom_coords) > 10:
                top_colors = roi_rgb[top_coords[:, 0], top_coords[:, 1]]
                bottom_colors = roi_rgb[bottom_coords[:, 0], bottom_coords[:, 1]]
                
                top_mean = np.mean(top_colors, axis=0)
                bottom_mean = np.mean(bottom_colors, axis=0)
                v_diff = np.linalg.norm(top_mean - bottom_mean)
                
                if v_diff > 35:
                    has_gradient = True
                    gradient_info = {
                        'direction': 'vertical',
                        'start_color': "#{:02x}{:02x}{:02x}".format(*top_mean.astype(int).clip(0, 255)),
                        'end_color': "#{:02x}{:02x}{:02x}".format(*bottom_mean.astype(int).clip(0, 255))
                    }
            
            # 左右分区（如果垂直没有渐变）
            if not has_gradient:
                left_coords = coords[coords[:, 1] < mid_x]
                right_coords = coords[coords[:, 1] >= mid_x]
                
                if len(left_coords) > 10 and len(right_coords) > 10:
                    left_colors = roi_rgb[left_coords[:, 0], left_coords[:, 1]]
                    right_colors = roi_rgb[right_coords[:, 0], right_coords[:, 1]]
                    
                    left_mean = np.mean(left_colors, axis=0)
                    right_mean = np.mean(right_colors, axis=0)
                    h_diff = np.linalg.norm(left_mean - right_mean)
                    
                    if h_diff > 35:
                        has_gradient = True
                        gradient_info = {
                            'direction': 'horizontal',
                            'start_color': "#{:02x}{:02x}{:02x}".format(*left_mean.astype(int).clip(0, 255)),
                            'end_color': "#{:02x}{:02x}{:02x}".format(*right_mean.astype(int).clip(0, 255))
                        }
    
    # =========== 5. 提取几何参数 ===========
    geo_params = extract_geometric_params(image, bbox, shape_type)
    
    # 格式化输出
    fill_color = "#{:02x}{:02x}{:02x}".format(*fill_rgb.clip(0, 255))
    stroke_color = "#{:02x}{:02x}{:02x}".format(*stroke_rgb.clip(0, 255))
    
    return {
        'fill_color': fill_color,
        'stroke_color': stroke_color,
        'stroke_width': stroke_width,
        'geo_params': geo_params,
        'has_gradient': has_gradient,
        'gradient_info': gradient_info
    }


# ======================== 样式统一 ========================
def unify_element_styles(elements: list) -> list:
    """
    统一相似大小和类型的基本图形的边框厚度。
    
    注意：参考 sam3_extractor.py 的简化逻辑，默认边框宽度为1，
    这里主要用于确保同类元素风格一致。
    """
    if not elements:
        return elements

    groups = {}
    
    for i, elem in enumerate(elements):
        shape_type = elem.get("_type", "rectangle")
        bbox = elem["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        diag = math.sqrt(w**2 + h**2)
        size_key = int(round(diag / 20))
        
        key = (shape_type, size_key)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
        
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        
        # 获取边框宽度，如果不存在则默认为1
        widths = []
        for i in indices:
            style = elements[i].get("_style", {})
            widths.append(style.get("stroke_width", 1))
        
        if not widths:
            continue
        median_width = int(np.median(widths))
        
        for i in indices:
            if "_style" not in elements[i]:
                elements[i]["_style"] = {}
            elements[i]["_style"]["stroke_width"] = median_width
            
    return elements


# ======================== CV矩形检测优化辅助函数 ========================
def _merge_nearby_lines(lines, threshold=10):
    """
    合并相近的平行线段，减少冗余
    
    Args:
        lines: 线段列表，格式为 [(y, x1, x2), ...] 或 [(x, y1, y2), ...]
        threshold: 合并阈值，位置差小于此值的线段会被合并
        
    Returns:
        合并后的线段列表
    """
    if not lines:
        return []
    
    merged = []
    used = set()
    
    for i, line in enumerate(lines):
        if i in used:
            continue
        
        pos, start, end = line  # y/x, x1/y1, x2/y2
        # 找到所有相近的线段
        group_pos = [pos]
        group_start = [start]
        group_end = [end]
        
        for j, other in enumerate(lines[i+1:], i+1):
            if j in used:
                continue
            o_pos, o_start, o_end = other
            if abs(o_pos - pos) < threshold:
                group_pos.append(o_pos)
                group_start.append(o_start)
                group_end.append(o_end)
                used.add(j)
        
        # 合并为一条线
        merged.append((
            int(np.mean(group_pos)),
            min(group_start),
            max(group_end)
        ))
        used.add(i)
    
    return merged


# ======================== CV结果验证 ========================
def _validate_cv_rectangle(cv2_image: np.ndarray, bbox: list, min_std: float = 8) -> bool:
    """
    验证CV检测到的矩形是否有效
    
    检查内容：
    1. 内部颜色是否有足够变化（排除纯色背景误检）
    2. 边框与内部是否有明显区别
    
    :param cv2_image: BGR图像
    :param bbox: [x1, y1, x2, y2]
    :param min_std: 最小颜色标准差
    :return: True=有效, False=可能是误检
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = cv2_image.shape[:2]
    
    # 边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 - x1 < 20 or y2 - y1 < 20:
        return False
    
    roi = cv2_image[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 检查1：内部是否有足够的颜色变化
    roi_h, roi_w = gray_roi.shape
    margin = max(3, min(roi_w, roi_h) // 10)
    
    if roi_h > 2 * margin and roi_w > 2 * margin:
        inner = gray_roi[margin:-margin, margin:-margin]
        inner_std = np.std(inner)
        
        # 如果内部颜色太均匀，可能是误检的背景区域
        if inner_std < min_std:
            return False
    
    # 检查2：边框与内部是否有对比度
    border_size = max(2, min(roi_w, roi_h) // 20)
    
    if roi_h > 2 * border_size and roi_w > 2 * border_size:
        border_top = gray_roi[:border_size, :].mean()
        border_bottom = gray_roi[-border_size:, :].mean()
        border_left = gray_roi[:, :border_size].mean()
        border_right = gray_roi[:, -border_size:].mean()
        border_mean = np.mean([border_top, border_bottom, border_left, border_right])
        
        inner_region = gray_roi[border_size:-border_size, border_size:-border_size]
        inner_mean = inner_region.mean()
        
        contrast = abs(border_mean - inner_mean)
        
        # 边框和内部需要有一定对比度
        if contrast < 5:
            return False
    
    return True


# ======================== CV矩形检测 ========================
def detect_rectangles_robust(cv2_image: np.ndarray, existing_elements: dict, config: dict = None) -> dict:
    """
    精准矩形检测（补充SAM3遗漏的矩形）
    
    采用保守策略：
    - 默认只启用可靠的检测方法（contour, nested_contour）
    - 提高检测门槛减少误检
    - 对检测结果进行内容验证
    
    :param cv2_image: BGR格式的OpenCV图像
    :param existing_elements: SAM3已识别的元素字典
    :param config: 配置参数字典
    :return: {"rectangles": [...], "containers": [...]}
    """
    default_config = {
        # 面积限制（提高门槛减少误检）
        "min_area": 5000,            # 提高最小面积（原3000）
        "min_area_ratio": 0.005,     # 最小面积占比
        "max_area_ratio": 0.5,
        
        # 去重阈值（更积极去重）
        "iou_threshold": 0.2,        # 降低IoU阈值（原0.3）
        "nms_threshold": 0.25,       # 降低NMS阈值（原0.3）
        
        # 形状验证（提高要求）
        "min_rectangularity": 0.7,   # 提高矩形度（原0.6）
        "border_contrast": 15,       # 提高边框对比度（原10）
        
        # 容器检测
        "container_threshold": 0.8,
        "min_contained": 3,
        
        # 启用的检测方法（保守模式：只启用可靠的方法）
        "enabled_methods": ["contour", "nested_contour"],
        # 完整模式可用: ["contour", "region", "low_contrast", "hough_lines", "nested_contour"]
        
        # 内容验证（CV结果需要通过验证）
        "validate_content": True,
        "min_content_std": 8,        # 内部颜色标准差阈值
    }
    cfg = {**default_config, **(config or {})}
    
    enabled_methods = set(cfg.get("enabled_methods", ["contour", "nested_contour"]))
    
    h, w = cv2_image.shape[:2]
    total_area = h * w
    max_area = total_area * cfg["max_area_ratio"]
    min_area = max(cfg["min_area"], int(total_area * cfg.get("min_area_ratio", 0)))
    
    # 收集SAM3已检测的bbox
    sam3_bboxes = []
    for elem_type, items in existing_elements.items():
        for item in items:
            sam3_bboxes.append({"bbox": item["bbox"], "type": elem_type})
    
    all_candidates = []
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    
    # 方法1：边缘轮廓检测（最可靠的方法）
    if "contour" in enabled_methods:
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < 100:
                continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if not (4 <= len(approx) <= 8):
                continue
            
            x, y, rw, rh = cv2.boundingRect(approx)
            bbox = [x, y, x+rw, y+rh]
            area = rw * rh
            
            if area < min_area or area > max_area:
                continue
            
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 4:
                continue
            
            cnt_area = cv2.contourArea(approx)
            rectangularity = cnt_area / area if area > 0 else 0
            if rectangularity < cfg["min_rectangularity"]:
                continue
            
            # 验证边框线
            border_w = max(3, min(8, rw // 15, rh // 15))
            
            if rw > 2 * border_w and rh > 2 * border_w:
                roi = gray[y:y+rh, x:x+rw]
                
                border_top = roi[:border_w, :].flatten()
                border_bottom = roi[-border_w:, :].flatten()
                border_left = roi[:, :border_w].flatten()
                border_right = roi[:, -border_w:].flatten()
                border_pixels = np.concatenate([border_top, border_bottom, border_left, border_right])
                
                inner = roi[border_w:-border_w, border_w:-border_w].flatten()
                
                if len(inner) > 0 and len(border_pixels) > 0:
                    border_mean = np.mean(border_pixels)
                    inner_mean = np.mean(inner)
                    contrast = abs(border_mean - inner_mean)
                    
                    if contrast < cfg["border_contrast"]:
                        continue
            
            is_rounded = rectangularity < 0.98
            
            all_candidates.append({
                "bbox": bbox,
                "area": area,
                "method": "contour",
                "score": rectangularity,
                "rectangularity": rectangularity,
                "is_rounded": is_rounded
            })
    
    # 方法2：区域颜色检测（容易误检，默认禁用）
    if "region" in enabled_methods:
        hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 50, 252])
        
        mask_region = cv2.inRange(hsv, lower_gray, upper_gray)
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close_region = np.ones((7, 7), np.uint8)
        
        mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_OPEN, kernel_open)
        mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, kernel_close_region)
        
        contours_region, _ = cv2.findContours(mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_region:
            peri = cv2.arcLength(cnt, True)
            if peri < 100:
                continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if not (4 <= len(approx) <= 12):
                continue
                
            x, y, rw, rh = cv2.boundingRect(approx)
            bbox = [x, y, x+rw, y+rh]
            area = rw * rh
            
            if area < min_area or area > max_area:
                continue
                
            if max(rw, rh) / max(1, min(rw, rh)) > 5:
                continue
                
            cnt_area = cv2.contourArea(approx)
            if area > 0 and cnt_area / area < 0.6:
                continue
                
            rect_ratio = cnt_area / area if area > 0 else 0
            is_rounded_region = False
            if 0.85 <= rect_ratio < 0.96:
                is_rounded_region = True
            elif len(approx) > 4 and rect_ratio < 0.96:
                is_rounded_region = True
                
            all_candidates.append({
                "bbox": bbox,
                "area": area,
                "method": "region",
                "score": rect_ratio,
                "rectangularity": rect_ratio,
                "is_rounded": is_rounded_region
            })

    # 方法3：低对比度框检测（容易误检，默认禁用，通过 enabled_methods 过滤）
    edges_low = cv2.Canny(gray, 10, 50)
    edges_low = cv2.dilate(edges_low, np.ones((3, 3), np.uint8), iterations=2)
    
    kernel_close = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges_low, cv2.MORPH_CLOSE, kernel_close)
    
    contours_low, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_low:
        peri = cv2.arcLength(cnt, True)
        if peri < 100:
            continue
        
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        
        if not (4 <= len(approx) <= 8):
            continue
        
        x, y, rw, rh = cv2.boundingRect(approx)
        bbox = [x, y, x+rw, y+rh]
        area = rw * rh
        
        max_area_expanded = total_area * 0.8
        if area < min_area or area > max_area_expanded:
            continue
        
        aspect = max(rw, rh) / max(1, min(rw, rh))
        if aspect > 5:
            continue
        
        cnt_area = cv2.contourArea(approx)
        rectangularity = cnt_area / area if area > 0 else 0
        if rectangularity < 0.55:
            continue
        
        # 浅色背景检查
        color_check_passed = False
        
        if rw > 15 and rh > 15:
            roi_bgr = cv2_image[y:y+rh, x:x+rw]
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            margin = max(5, min(rw // 12, rh // 12))
            if rw > 2 * margin and rh > 2 * margin:
                center_hsv = roi_hsv[margin:-margin, margin:-margin]
                s_channel = center_hsv[:, :, 1]
                median_saturation = np.median(s_channel)
                
                v_channel = center_hsv[:, :, 2]
                median_value = np.median(v_channel)
                
                if median_saturation < 75 and median_value > 150:
                    color_check_passed = True
        
        if not color_check_passed:
            continue
        
        is_rounded = rectangularity < 0.92
        
        all_candidates.append({
            "bbox": bbox,
            "area": area,
            "method": "low_contrast_gray",
            "score": rectangularity,
            "rectangularity": rectangularity,
            "is_rounded": is_rounded
        })
    
    # 方法4：霍夫线检测（检测虚线框、表格线等）
    edges_hough = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges_hough, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        # 分类水平线和垂直线
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi
            length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
            
            if length < 30:
                continue
            
            if abs(angle) < 15 or abs(angle) > 165:
                h_lines.append((min(y1_l, y2_l), min(x1_l, x2_l), max(x1_l, x2_l)))
            elif 75 < abs(angle) < 105:
                v_lines.append((min(x1_l, x2_l), min(y1_l, y2_l), max(y1_l, y2_l)))
        
        # 尝试从线段重建矩形
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            # 优化：合并相近线段，减少冗余
            h_lines = _merge_nearby_lines(h_lines, threshold=15)
            v_lines = _merge_nearby_lines(v_lines, threshold=15)
            
            # 优化：限制线段数量，控制复杂度上限
            MAX_LINES = 30
            h_lines = h_lines[:MAX_LINES]
            v_lines = v_lines[:MAX_LINES]
            
            h_lines.sort(key=lambda x: x[0])
            v_lines.sort(key=lambda x: x[0])
            
            tolerance = 15
            
            # 优化：提前终止，找到足够候选后停止
            MAX_HOUGH_CANDIDATES = 50
            hough_found = 0
            
            for i, h_top in enumerate(h_lines):
                if hough_found >= MAX_HOUGH_CANDIDATES:
                    break
                for h_bottom in h_lines[i+1:]:
                    if hough_found >= MAX_HOUGH_CANDIDATES:
                        break
                    rect_height = h_bottom[0] - h_top[0]
                    if rect_height < 30:
                        continue
                    
                    for j, v_left in enumerate(v_lines):
                        if hough_found >= MAX_HOUGH_CANDIDATES:
                            break
                        for v_right in v_lines[j+1:]:
                            if hough_found >= MAX_HOUGH_CANDIDATES:
                                break
                            rect_width = v_right[0] - v_left[0]
                            if rect_width < 30:
                                continue
                            
                            # 检查四条边是否能形成矩形
                            h_top_valid = (h_top[1] <= v_left[0] + tolerance and 
                                          h_top[2] >= v_right[0] - tolerance)
                            h_bottom_valid = (h_bottom[1] <= v_left[0] + tolerance and 
                                             h_bottom[2] >= v_right[0] - tolerance)
                            v_left_valid = (v_left[1] <= h_top[0] + tolerance and 
                                           v_left[2] >= h_bottom[0] - tolerance)
                            v_right_valid = (v_right[1] <= h_top[0] + tolerance and 
                                            v_right[2] >= h_bottom[0] - tolerance)
                            
                            if h_top_valid and h_bottom_valid and v_left_valid and v_right_valid:
                                bbox_h = [v_left[0], h_top[0], v_right[0], h_bottom[0]]
                                area_h = rect_width * rect_height
                                
                                if min_area <= area_h <= max_area:
                                    all_candidates.append({
                                        "bbox": bbox_h,
                                        "area": area_h,
                                        "method": "hough_lines",
                                        "score": 0.8,
                                        "rectangularity": 0.9,
                                        "is_rounded": False
                                    })
                                    hough_found += 1
    
    # 方法5：嵌套轮廓检测（检测容器框）
    _, binary_nest = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_nest, hierarchy_nest = cv2.findContours(binary_nest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy_nest is not None:
        hierarchy_nest = hierarchy_nest[0]
        
        for idx, cnt in enumerate(contours_nest):
            # 检查是否有子轮廓
            if hierarchy_nest[idx][2] == -1:  # 没有子轮廓
                continue
            
            # 计算子轮廓数量
            child_count = 0
            child_idx = hierarchy_nest[idx][2]
            while child_idx != -1:
                child_count += 1
                child_idx = hierarchy_nest[child_idx][0]
            
            if child_count < 2:  # 至少包含2个子元素
                continue
            
            peri = cv2.arcLength(cnt, True)
            if peri < 200:
                continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if not (4 <= len(approx) <= 10):
                continue
            
            x, y, rw, rh = cv2.boundingRect(approx)
            bbox_n = [x, y, x+rw, y+rh]
            area_n = rw * rh
            
            if area_n < min_area * 2 or area_n > max_area:  # 容器通常较大
                continue
            
            aspect_n = max(rw, rh) / max(1, min(rw, rh))
            if aspect_n > 4:
                continue
            
            cnt_area_n = cv2.contourArea(approx)
            rectangularity_n = cnt_area_n / area_n if area_n > 0 else 0
            
            if rectangularity_n < 0.5:
                continue
            
            all_candidates.append({
                "bbox": bbox_n,
                "area": area_n,
                "method": "nested_contour",
                "score": rectangularity_n,
                "rectangularity": rectangularity_n,
                "is_rounded": rectangularity_n < 0.9,
                "child_count": child_count
            })
    
    # 按方法过滤（只保留启用的方法的结果）
    method_mapping = {
        "contour": "contour",
        "region": "region", 
        "low_contrast_gray": "low_contrast",
        "hough_lines": "hough_lines",
        "nested_contour": "nested_contour"
    }
    
    all_candidates = [
        cand for cand in all_candidates 
        if method_mapping.get(cand["method"], cand["method"]) in enabled_methods
    ]
    
    # NMS去重
    all_candidates.sort(key=lambda x: x["area"], reverse=True)
    
    filtered_candidates = []
    validate_content = cfg.get("validate_content", True)
    min_content_std = cfg.get("min_content_std", 8)
    
    for cand in all_candidates:
        bbox = cand["bbox"]
        
        # 内容验证（过滤误检的背景区域）
        if validate_content:
            if not _validate_cv_rectangle(cv2_image, bbox, min_std=min_content_std):
                continue
        
        # 与SAM3结果对比
        is_dup_sam3 = False
        for sam3_item in sam3_bboxes:
            iou = calculate_iou(bbox, sam3_item["bbox"])
            if iou > cfg["iou_threshold"]:
                is_dup_sam3 = True
                break
        if is_dup_sam3:
            continue
        
        # NMS
        is_dup_nms = False
        for existing in filtered_candidates:
            iou = calculate_iou(bbox, existing["bbox"])
            if iou > cfg["nms_threshold"]:
                is_dup_nms = True
                break
        if is_dup_nms:
            continue
        
        filtered_candidates.append(cand)
    
    # 自动分层（判断谁是容器）
    all_bboxes_for_contain = [item["bbox"] for item in sam3_bboxes] + [c["bbox"] for c in filtered_candidates]
    
    for cand in filtered_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        contained_count = 0
        
        for other_bbox in all_bboxes_for_contain:
            if other_bbox == cand["bbox"]:
                continue
            ox1, oy1, ox2, oy2 = other_bbox
            if x1 <= ox1 and y1 <= oy1 and x2 >= ox2 and y2 >= oy2:
                contained_count += 1
            elif calculate_iou(cand["bbox"], other_bbox) > 0:
                inter_x1 = max(x1, ox1)
                inter_y1 = max(y1, oy1)
                inter_x2 = min(x2, ox2)
                inter_y2 = min(y2, oy2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    other_area = (ox2 - ox1) * (oy2 - oy1)
                    if other_area > 0 and inter_area / other_area > cfg["container_threshold"]:
                        contained_count += 1
        
        cand["contained_count"] = contained_count
        cand["is_container"] = contained_count >= cfg["min_contained"]
    
    # 颜色提取
    rectangles = []
    containers = []
    
    for cand in filtered_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        rw, rh = x2 - x1, y2 - y1
        
        # 填充色提取
        margin_x = max(3, int(rw * 0.25))
        margin_y = max(3, int(rh * 0.25))
        inner_x1, inner_y1 = x1 + margin_x, y1 + margin_y
        inner_x2, inner_y2 = x2 - margin_x, y2 - margin_y
        
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            inner_roi = cv2_image[inner_y1:inner_y2, inner_x1:inner_x2]
            inner_rgb = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2RGB)
            pixels = inner_rgb.reshape(-1, 3).astype(np.float32)
            
            if len(pixels) > 10:
                try:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                    counts = np.bincount(labels.flatten())
                    dominant_idx = np.argmax(counts)
                    fill_rgb = centers[dominant_idx].astype(int)
                except Exception:
                    fill_rgb = np.median(pixels, axis=0).astype(int)
            else:
                fill_rgb = np.median(pixels, axis=0).astype(int) if len(pixels) > 0 else np.array([255, 255, 255])
            
            fill_color = "#{:02x}{:02x}{:02x}".format(*np.clip(fill_rgb, 0, 255))
        else:
            fill_color = "#ffffff"
        
        # CV检测结果强制使用黑色细边框（风格统一）
        stroke_color = "#000000"
        
        result_item = {
            "bbox": cand["bbox"],
            "area": cand["area"],
            "fill_color": fill_color,
            "stroke_color": stroke_color,
            "score": cand["score"],
            "method": cand["method"],
            "contained_count": cand.get("contained_count", 0),
            "is_rounded": cand.get("is_rounded", False)
        }
        
        if cand["is_container"]:
            containers.append(result_item)
        else:
            rectangles.append(result_item)
    
    return {
        "rectangles": rectangles,
        "containers": containers
    }


# ======================== 基本图形处理器 ========================
class BasicShapeProcessor(BaseProcessor):
    """
    基本图形处理模块
    
    处理流程：
        1. 从context.elements中筛选基本图形
        2. 对每个图形提取填充色和描边色
        3. 生成XML片段
        4. 可选：运行CV补充检测遗漏的矩形
    """
    
    def __init__(self, config=None, enable_cv_detection: bool = True):
        """
        Args:
            config: 处理配置
            enable_cv_detection: 是否启用CV补充检测（检测SAM3遗漏的矩形）
        """
        super().__init__(config)
        self.enable_cv_detection = enable_cv_detection
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult
        """
        self._log("开始处理基本图形")
        
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        cv2_image = cv2.imread(context.image_path)
        if cv2_image is None:
            return ProcessingResult(
                success=False,
                error_message="无法读取图片"
            )
        
        # 筛选基本图形
        elements_to_process = self._get_elements_to_process(context.elements)
        
        # 计算画布面积，用于判断大面积元素
        canvas_area = context.canvas_width * context.canvas_height
        
        processed_count = 0
        for elem in elements_to_process:
            try:
                self._process_element(elem, cv2_image, canvas_area)
                processed_count += 1
            except Exception as e:
                elem.processing_notes.append(f"处理失败: {str(e)}")
                self._log(f"元素{elem.id}处理失败: {e}")
        
        # CV补充检测
        cv_added_count = 0
        if self.enable_cv_detection:
            cv_added_count = self._run_cv_detection(context, cv2_image)
        
        self._log(f"处理完成: {processed_count}个SAM3图形, {cv_added_count}个CV补充")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'processed_count': processed_count,
                'cv_added_count': cv_added_count,
                'total_to_process': len(elements_to_process)
            }
        )
    
    def _get_elements_to_process(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """筛选需要处理的基本图形"""
        return [
            e for e in elements
            if e.element_type.lower() in VECTOR_TYPES and e.fill_color is None
        ]
    
    def _process_element(self, elem: ElementInfo, cv2_image: np.ndarray, canvas_area: int = 0):
        """
        处理单个元素：提取颜色并生成XML
        
        优先使用SAM3提供的Mask进行精确取色
        
        Args:
            elem: 元素信息
            cv2_image: OpenCV格式的图像
            canvas_area: 画布总面积，用于判断大面积元素
        """
        elem_type = elem.element_type.lower()
        
        # 提取样式 - 优先使用Mask
        if elem.mask is not None and hasattr(elem.mask, 'shape') and elem.mask.size > 0:
            # 使用SAM3提供的Mask进行精确取色
            style_data = extract_color_with_mask(
            cv2_image,
                elem.bbox.to_list(), 
                elem.mask,
                elem_type
            )
            elem.processing_notes.append("使用Mask精确取色")
        else:
            # 降级：使用传统的bbox取色
            style_data = extract_style_specific(cv2_image, elem.bbox.to_list(), elem_type)
            elem.processing_notes.append("使用bbox取色(无Mask)")
        
        elem.fill_color = style_data["fill_color"]
        elem.stroke_color = style_data["stroke_color"]
        elem.stroke_width = style_data["stroke_width"]
        
        # 记录渐变信息（如果有）
        if style_data.get('has_gradient'):
            elem.processing_notes.append(f"检测到渐变: {style_data.get('gradient_info')}")
        
        # 设置层级 - 根据类型和面积判断
        elem_area = elem.bbox.area if elem.bbox else 0
        area_ratio = elem_area / canvas_area if canvas_area > 0 else 0
        
        # 大面积元素（>15%画布面积）或特定类型放到背景层
        if elem_type in {'section_panel', 'title_bar', 'container'} or area_ratio > 0.15:
            elem.layer_level = LayerLevel.BACKGROUND.value
            if area_ratio > 0.15:
                elem.processing_notes.append(f"大面积元素({area_ratio:.1%})，放入背景层")
        else:
            elem.layer_level = LayerLevel.BASIC_SHAPE.value
        
        # 生成XML片段
        elem.xml_fragment = self._generate_xml(elem, style_data)
        elem.processing_notes.append("BasicShapeProcessor处理完成")
    
    def _generate_xml(self, elem: ElementInfo, style_data: dict) -> str:
        """生成mxCell XML"""
        elem_type = elem.element_type.lower()
        
        # 获取基础样式
        base_style = DRAWIO_STYLES.get(elem_type, "rounded=0;whiteSpace=wrap;html=1;")
        
        # 动态应用几何参数
        geo_params = style_data.get("geo_params", {})
        if elem_type == "parallelogram" and "size" in geo_params:
            base_style += f"size={geo_params['size']:.2f};"
        elif elem_type == "cylinder" and "size" in geo_params:
            base_style += f"size={geo_params['size']};"
        elif elem_type == "triangle" and "direction" in geo_params:
            base_style += f"direction={geo_params['direction']};"
        
        # 构建完整样式
        fill_color = style_data["fill_color"]
        stroke_color = style_data["stroke_color"]
        stroke_width = style_data["stroke_width"]
        
        style = f"{base_style}fillColor={fill_color};strokeColor={stroke_color};strokeWidth={stroke_width};"
        
        # DrawIO的id必须从2开始（0和1是保留的根元素）
        cell_id = elem.id + 2
        
        return f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{elem.bbox.x1}" y="{elem.bbox.y1}" width="{elem.bbox.width}" height="{elem.bbox.height}" as="geometry"/>
</mxCell>'''
    
    def _run_cv_detection(self, context: ProcessingContext, cv2_image: np.ndarray) -> int:
        """运行CV补充检测"""
        # 构建SAM3元素字典格式
        sam3_elements = {}
        for elem in context.elements:
            elem_type = elem.element_type.lower()
            if elem_type not in sam3_elements:
                sam3_elements[elem_type] = []
            sam3_elements[elem_type].append({
                "bbox": elem.bbox.to_list(),
                "score": elem.score
            })
        
        # 运行检测
        h, w = cv2_image.shape[:2]
        cv_results = detect_rectangles_robust(cv2_image, sam3_elements, {
            "min_area_ratio": 0.07,
            "max_area_ratio": 0.95
        })
        
        added_count = 0
        start_id = max([e.id for e in context.elements], default=0) + 1
        
        # 添加检测到的矩形
        for item in cv_results["rectangles"]:
            new_elem = self._create_element_from_cv(item, start_id + added_count, "rectangle", cv2_image)
            context.elements.append(new_elem)
            added_count += 1
        
        # 添加检测到的容器
        for item in cv_results["containers"]:
            new_elem = self._create_element_from_cv(item, start_id + added_count, "container", cv2_image)
            context.elements.append(new_elem)
            added_count += 1
        
        return added_count
    
    def _create_element_from_cv(self, item: dict, elem_id: int, elem_type: str, cv2_image: np.ndarray) -> ElementInfo:
        """从CV检测结果创建ElementInfo"""
        bbox = BoundingBox.from_list(item["bbox"])
        
        # 判断是圆角还是直角
        actual_type = elem_type
        if item.get("is_rounded", False) and elem_type == "rectangle":
            actual_type = "rounded rectangle"
        
        elem = ElementInfo(
            id=elem_id,
            element_type=actual_type,
            bbox=bbox,
            score=item.get("score", 0.8),
            fill_color=item.get("fill_color"),
            stroke_color=item.get("stroke_color"),
            source_prompt="cv_detection"
        )
        
        # 设置层级
        if elem_type == "container":
            elem.layer_level = LayerLevel.BACKGROUND.value
        else:
            elem.layer_level = LayerLevel.BASIC_SHAPE.value
        
        # 提取填充色（使用优化的取色逻辑）
        style_data = extract_style_specific(cv2_image, item["bbox"], actual_type)
        elem.fill_color = item.get("fill_color") or style_data["fill_color"]
        
        # CV检测结果强制使用黑色细边框（风格统一、更清晰）
        elem.stroke_color = "#000000"
        elem.stroke_width = 1
        
        # 生成XML（使用强制的边框样式）
        style_data_for_xml = {
            "fill_color": elem.fill_color,
            "stroke_color": elem.stroke_color,
            "stroke_width": elem.stroke_width,
            "geo_params": style_data.get("geo_params", {})
        }
        elem.xml_fragment = self._generate_xml(elem, style_data_for_xml)
        elem.processing_notes.append(f"CV检测补充 (method={item.get('method', 'unknown')})")
        
        return elem


# ======================== 独立处理函数 ========================
def process_basic_shapes(image: np.ndarray, sam3_elements: dict) -> str:
    """
    处理所有基本图形（SAM3结果 + CV补充检测），生成DrawIO XML。
    
    :param image: 原始图像 (BGR)
    :param sam3_elements: SAM3提取的元素字典
    :return: 格式化的XML字符串
    """
    h, w = image.shape[:2]
    
    # 运行CV补充检测
    cv_results = detect_rectangles_robust(image, sam3_elements, {
        "min_area_ratio": 0.07,
        "max_area_ratio": 0.95
    })
    
    # 收集所有需要绘制的元素
    containers_list = []
    shapes_list = []
    
    # 来自 SAM3 的 container
    if "container" in sam3_elements:
        for item in sam3_elements["container"]:
            item_copy = item.copy()
            item_copy["_type"] = "container"
            item_copy["_source"] = "sam3"
            containers_list.append(item_copy)
            
    # 来自 CV 检测的 containers
    for item in cv_results["containers"]:
        item_copy = item.copy()
        item_copy["_type"] = "container"
        item_copy["_source"] = "cv"
        containers_list.append(item_copy)
        
    # 来自 SAM3 的其他形状
    for key, items in sam3_elements.items():
        if key in VECTOR_TYPES and key != "container":
            for item in items:
                item_copy = item.copy()
                item_copy["_type"] = key
                item_copy["_source"] = "sam3"
                shapes_list.append(item_copy)
                
    # 来自 CV 检测的 rectangles
    for item in cv_results["rectangles"]:
        item_copy = item.copy()
        item_copy["_type"] = "rectangle"
        item_copy["_source"] = "cv"
        shapes_list.append(item_copy)
    
    # 排序：面积大的在底层
    def calculate_element_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    containers_list.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)
    shapes_list.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)
    
    # 提取样式
    all_elements_ref = []
    
    def get_style_for_item(item):
        """获取元素样式，CV检测的强制使用黑色细边框"""
        style = extract_style_specific(image, item["bbox"], item["_type"])
        if item.get("_source") == "cv":
            style["stroke_width"] = 1
            style["stroke_color"] = "#000000"
            # 如果detect_rectangles_robust已经提取了填充色，优先使用
            if item.get("fill_color"):
                style["fill_color"] = item["fill_color"]
        return style
    
    for item in containers_list:
        item["_style"] = get_style_for_item(item)
        all_elements_ref.append(item)
        
    for item in shapes_list:
        item["_style"] = get_style_for_item(item)
        all_elements_ref.append(item)
        
    # 统一边框厚度
    unify_element_styles(all_elements_ref)
    
    # 构建XML结构
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "BasicShapes", "name": "Page-1"})
    mx_graph_model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(w), "dy": str(h), "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(w), "pageHeight": str(h),
        "background": "#ffffff"
    })
    root = ET.SubElement(mx_graph_model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    
    cell_id = 2
    
    def add_cell(item_list):
        nonlocal cell_id
        for item in item_list:
            x1, y1, x2, y2 = map(int, item["bbox"])
            width, height = x2 - x1, y2 - y1
            elem_type = item["_type"]
            
            style_data = item.get("_style")
            if not style_data:
                f_color, s_color, s_width = extract_style_colors(image, item["bbox"])
                style_data = {"fill_color": f_color, "stroke_color": s_color, "stroke_width": s_width}
                
            fill_color = style_data["fill_color"]
            stroke_color = style_data["stroke_color"]
            stroke_width = style_data["stroke_width"]
            
            if elem_type in ("rectangle", "rounded rectangle", "container"):
                is_rounded = item.get("is_rounded", elem_type == "rounded rectangle")
                rounded_val = "1" if is_rounded else "0"
                base_style = f"rounded={rounded_val};whiteSpace=wrap;html=1;"
            else:
                base_style = DRAWIO_STYLES.get(elem_type, "rounded=0;whiteSpace=wrap;html=1;")
                
                geo_params = style_data.get("geo_params", {})
                if elem_type == "parallelogram" and "size" in geo_params:
                    base_style += f"size={geo_params['size']:.2f};"
                elif elem_type == "cylinder" and "size" in geo_params:
                    base_style += f"size={geo_params['size']};"
                elif elem_type == "triangle" and "direction" in geo_params:
                    base_style += f"direction={geo_params['direction']};"
            
            style = f"{base_style}fillColor={fill_color};strokeColor={stroke_color};strokeWidth={stroke_width};"
            
            cell = ET.SubElement(root, "mxCell", {
                "id": str(cell_id),
                "parent": "1",
                "vertex": "1",
                "value": "",
                "style": style
            })
            ET.SubElement(cell, "mxGeometry", {
                "x": str(x1), "y": str(y1),
                "width": str(width), "height": str(height),
                "as": "geometry"
            })
            
            cell_id += 1

    add_cell(containers_list)
    add_cell(shapes_list)
    
    # 格式化XML
    rough_string = ET.tostring(mxfile, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return '\n'.join([
        line for line in reparsed.toprettyxml(indent="  ").split('\n')
        if line.strip() and not line.strip().startswith("<?xml")
    ])


# ======================== 快捷函数 ========================
def extract_shape_colors(elements: List[ElementInfo], 
                         image_path: str) -> List[ElementInfo]:
    """
    快捷函数 - 提取所有基本图形的颜色
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        
    Returns:
        处理后的元素列表
    """
    processor = BasicShapeProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=elements
    )
    
    result = processor.process(context)
    return result.elements
