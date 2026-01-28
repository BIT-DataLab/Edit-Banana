"""
任务7：质量评估模块（MetricEvaluator）

================================================================================
设计思路
================================================================================

【核心目标】
计算SAM3和其他检测器"检测到了多少"，以及"漏掉了多少"，用于决定是否需要
进行fallback补救（二次处理）。

这个模块回答的核心问题是：
    "把检测到的元素框掩掉后，剩下还有多少内容没覆盖？这些内容在哪里？"

【评分系统设计 - 满分100分】

核心指标：内容覆盖率（Content Coverage Score）

    计算公式：
        score = (covered_content_pixels / total_content_pixels) × 100
    
    关键概念 - 什么是"内容"（前景）vs "背景"：
        ❌ 背景：纯白、浅灰、浅色的大面积连续区域，这些不需要被检测
        ✅ 内容（前景）：图形、图标、文字、箭头、图片等需要被检测的元素
        
    内容识别策略（改进后）：
        1. 边缘检测（主要）：有边缘的地方才是真正的内容边界
        2. 灰度阈值（辅助）：灰度 < 240 的区域（更严格，排除浅色背景）
        3. 形态学去噪：去除小噪点
        4. 连通域过滤：去除面积太小的区域（可能是噪点）
        
    其中：
        - total_content_pixels: 原图中的前景内容像素（经过背景过滤）
        - covered_content_pixels: 已检测元素的bbox覆盖的前景内容像素
        
    含义解读：
        - 100分：所有前景内容都被检测到了（完美覆盖）
        - 90分：90%的内容被检测到，10%漏检（很好）
        - 70分：70%的内容被检测到，30%漏检（需要refinement）
        - <70分：大量内容漏检，refinement必要
        
    漏检率 = 100 - score（即有多少前景内容没被检测到）
    
    注意：背景不参与计算，所以浅色/白色背景不会影响分数

【问题区域检测 - 双通道策略】
使用双通道检测策略，目标是：
    - 电商长图：鞋子、人脸等商品主体，给出稳定的小框
    - 学术海报：左上小图、热力图、3D块等，每个版块给出清晰矩形

1. 细粒度通道（Fine Channel）：
   - 不做形态学操作，直接在未覆盖内容上做连通域分析
   - 用于检测：小图、图标、人脸、小子图等小目标
   - 参数：面积0.1%~15%，填充率>=20%，宽高比<=6

2. 粗粒度通道（Coarse Channel）：
   - 使用中等核(7×7)的闭操作合并相邻内容
   - 用于检测：版块、大图、分散的图形组
   - 参数：面积0.3%~25%，填充率>=30%，宽高比<=6

3. 小框优先NMS：
   - 关键创新：按面积从小到大排序处理
   - 保留小框，抑制被小框高度覆盖的大框
   - 避免把多个小目标误合并成一个大框

4. 去重过滤：
   - 与已检测元素IoU > 30% 的丢弃（避免重复）
   - 框内部被已覆盖区域占比 > 50% 的丢弃（说明大部分已识别）
   - 框内实际漏检内容 < 10% 的丢弃（说明内容太少）

【可选增强：边缘检测】
对于线条、虚线等稀疏内容，灰度阈值可能漏检。
开启 use_edge_detection 后会同时使用Canny边缘检测，并与灰度阈值取并集。

【与IMG2XML的对比和改进】
- 整合了 detect_missed_images（结构化分块）和 detect_missing_by_coverage（宽松过滤）的优点
- 简化配置参数，提供合理默认值（可通过config覆盖）
- 评分系统更直观：分数=覆盖率，一目了然
- 增加详细的 metrics 字典，便于调试和分析
- 提供可视化保存函数

================================================================================
接口说明
================================================================================

输入：
    - context.image_path: 原始图片路径
    - context.elements: 已检测的元素列表（包含bbox信息）
    - context.canvas_width/height: 画布尺寸

输出（ProcessingResult.metadata）：
    - overall_score: 总体评分（0-100，即覆盖率，越高越好）
    - missing_rate: 漏检率（0-100%，越低越好）
    - bad_regions: 问题区域列表，每个包含：
        - bbox: [x1, y1, x2, y2] - 问题区域的边界框
        - area: 面积（像素）
        - area_ratio: 占图片面积的比例
        - missing_pixels: 该区域内的漏检内容像素数
        - reason: 'uncovered_content'
        - channel: 'fine'（细粒度）或 'coarse'（粗粒度）
        - description: 可读的描述文本
    - metrics: 详细指标字典（用于调试）
    - needs_refinement: 是否建议进行二次处理

================================================================================
使用示例
================================================================================

    from modules import MetricEvaluator, ProcessingContext
    
    evaluator = MetricEvaluator()
    context = ProcessingContext(image_path="test.png", elements=[...])
    
    result = evaluator.process(context)
    
    print(f"覆盖率评分: {result.metadata['overall_score']}/100")
    print(f"漏检率: {result.metadata['missing_rate']}%")
    print(f"问题区域: {len(result.metadata['bad_regions'])}个")
    print(f"是否需要refinement: {result.metadata['needs_refinement']}")
    
    for region in result.metadata['bad_regions']:
        print(f"  位置: {region['bbox']}, 占比: {region['area_ratio']*100:.2f}%")
        print(f"    检测通道: {region['channel']}, 漏检像素: {region['missing_pixels']}")
    
    # 保存可视化结果
    evaluator.save_visualization(context, result.metadata['bad_regions'], "eval_result.png")
    evaluator.save_uncovered_mask(context, "uncovered_mask.png")

================================================================================
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class MetricEvaluator(BaseProcessor):
    """
    质量评估模块
    
    评估SAM3+其他检测器的覆盖效果，识别漏检区域。
    
    核心功能：
        1. 计算内容覆盖率（检测到的内容占总内容的比例）= 评分
        2. 识别问题区域（有内容但未被检测覆盖）
        3. 输出总体评分和问题区域列表供RefinementProcessor使用
    """

    DEFAULT_CONFIG = {
        # ===== 内容检测参数 =====
        # 注意：这里的"内容"指的是需要被检测的前景元素，不包括背景
        'content_threshold': 245,      # 灰度阈值，低于此值认为有内容
        'use_edge_detection': True,    # 是否使用边缘检测增强
        'edge_low_threshold': 30,      # Canny边缘检测低阈值（更敏感）
        'edge_high_threshold': 100,    # Canny边缘检测高阈值（更敏感）
        
        # ===== 背景过滤参数 =====
        'filter_background': True,     # 是否过滤背景区域
        'background_denoise_kernel': 2, # 去噪形态学核大小（更小，保留更多细节）
        'min_content_area': 30,        # 最小内容连通域面积（更小，保留更多细节）
        
        # ===== 细粒度通道参数（检测小目标：图标/人脸/小图）=====
        'fine_min_area_ratio': 0.0005,     # 最小面积比例 0.05%（更敏感）
        'fine_max_area_ratio': 0.20,       # 最大面积比例 20%
        'fine_min_fill_ratio': 0.15,       # 最小填充率（更宽松，检测稀疏内容）
        'fine_max_aspect_ratio': 8.0,      # 最大宽高比
        
        # ===== 粗粒度通道参数（检测版块/大图）=====
        'coarse_min_area_ratio': 0.002,    # 最小面积比例 0.2%（更敏感）
        'coarse_max_area_ratio': 0.30,     # 最大面积比例 30%
        'coarse_min_fill_ratio': 0.20,     # 最小填充率（更宽松）
        'coarse_max_aspect_ratio': 8.0,    # 最大宽高比
        'coarse_kernel_size': 5,           # 闭操作核大小（更小，避免过度合并）
        
        # ===== NMS和去重参数 =====
        'nms_iou_threshold': 0.3,          # 小框优先NMS的IoU阈值（更严格）
        'existing_iou_threshold': 0.5,     # 与已有元素去重的IoU阈值（更宽松，保留更多候选）
        'max_covered_ratio': 0.7,          # 候选框内最大已覆盖比例（更宽松）
        
        # ===== 评分阈值 =====
        'good_coverage_threshold': 95,     # 覆盖率>=95%才认为很好（更严格）
        'acceptable_threshold': 80,        # 覆盖率>=80%认为可接受
        
        # ===== 漏检内容最小比例 =====
        'min_missing_content_ratio': 0.05, # 候选框内至少5%是漏检内容才保留（更敏感）
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        # 合并用户配置和默认配置
        self.eval_config = {**self.DEFAULT_CONFIG, **(config or {})}
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 评估质量
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult，metadata中包含评估结果
        """
        self._log("开始质量评估")
        
        # 验证输入
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        # 加载原图
        cv2_image = cv2.imread(context.image_path)
        if cv2_image is None:
            return ProcessingResult(
                success=False,
                error_message="无法读取图片"
            )
        
        h, w = cv2_image.shape[:2]
        img_area = h * w
        
        # ========== 1. 创建内容掩码（识别有内容的区域） ==========
        content_mask = self._create_content_mask(cv2_image)
        total_content_pixels = int(np.sum(content_mask > 0))
        
        # ========== 2. 创建覆盖掩码（已检测元素覆盖的区域） ==========
        # 获取 OCR 文字 XML（如果有的话）
        text_xml = context.intermediate_results.get('text_xml', None)
        covered_mask, existing_bboxes = self._create_covered_mask(context.elements, h, w, text_xml)
        
        # ========== 3. 计算覆盖率评分 ==========
        # 内容区域中被覆盖的像素
        covered_content = cv2.bitwise_and(content_mask, covered_mask)
        covered_content_pixels = int(np.sum(covered_content > 0))
        
        # 像素级覆盖率（辅助指标）
        if total_content_pixels > 0:
            content_coverage = (covered_content_pixels / total_content_pixels) * 100
        else:
            content_coverage = 100.0  # 没有内容，认为完全覆盖
        
        missing_rate = 100.0 - content_coverage
        
        # ========== 4. 计算未覆盖内容掩码 ==========
        uncovered_content = cv2.bitwise_and(
            content_mask,
            cv2.bitwise_not(covered_mask)
        )
        
        # ========== 5. 识别问题区域（三重策略） ==========
        bad_regions = self._detect_bad_regions(
            cv2_image, content_mask, covered_mask, existing_bboxes, img_area, context.elements, context
        )
        
        # ========== 6. 计算真实评分（基于问题区域面积，去重） ==========
        # 创建问题区域掩码，去除重叠部分
        bad_region_mask = np.zeros((h, w), dtype=np.uint8)
        for region in bad_regions:
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                bad_region_mask[y1:y2, x1:x2] = 255
        
        # 计算去重后的实际问题区域面积
        actual_bad_region_pixels = int(np.sum(bad_region_mask > 0))
        total_bad_region_ratio = (actual_bad_region_pixels / img_area) * 100 if img_area > 0 else 0
        
        # 评分 = 100 - 问题区域总面积比例（去重后）
        overall_score = max(0, 100.0 - total_bad_region_ratio)
        
        # 是否需要refinement
        # 条件: 有问题区域就需要补救
        has_complex_image_regions = any(r.get('channel') == 'complex' for r in bad_regions)
        needs_refinement = len(bad_regions) > 0 or has_complex_image_regions
        
        # 构建详细指标（使用去重后的面积）
        total_bad_region_area = actual_bad_region_pixels
        metrics = {
            'overall_score': round(overall_score, 2),  # 最终评分（100 - 问题区域面积比例）
            'pixel_coverage': round(content_coverage, 2),  # 像素级覆盖率（辅助指标）
            'missing_rate': round(missing_rate, 2),
            'total_content_pixels': total_content_pixels,
            'covered_content_pixels': covered_content_pixels,
            'missing_content_pixels': total_content_pixels - covered_content_pixels,
            'image_area': img_area,
            'content_ratio': round(total_content_pixels / img_area * 100, 2),
            'element_count': len(context.elements),
            'bad_region_count': len(bad_regions),
            'total_bad_region_area': total_bad_region_area,
            'total_bad_region_ratio': round(total_bad_region_ratio, 2),  # 问题区域占图片面积的比例
        }
        
        self._log(f"评估完成: 评分={overall_score:.1f}, 问题区域={len(bad_regions)}个, 问题面积={total_bad_region_ratio:.1f}%")
        
        # ========== 6. 自动保存可视化和评估结果到 output_dir ==========
        if context.output_dir and os.path.exists(context.output_dir):
            # 保存未覆盖内容可视化
            uncovered_vis_path = os.path.join(context.output_dir, "metric_uncovered.png")
            self._save_uncovered_visualization(cv2_image, uncovered_content, covered_mask, bad_regions, uncovered_vis_path)
            
            # 保存评估分数到 JSON
            eval_json_path = os.path.join(context.output_dir, "metric_evaluation.json")
            self._save_evaluation_json(metrics, bad_regions, needs_refinement, overall_score, eval_json_path)
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width or w,
            canvas_height=context.canvas_height or h,
            metadata={
                'overall_score': round(overall_score, 2),
                'pixel_coverage': round(content_coverage, 2),
                'bad_region_ratio': round(total_bad_region_ratio, 2),
                'metrics': metrics,
                'bad_regions': bad_regions,
                'needs_refinement': needs_refinement,
            }
        )
    
    def _create_content_mask(self, cv2_image: np.ndarray) -> np.ndarray:
        """
        创建内容掩码 - 识别图片中真正需要被检测的前景内容
        
        关键改进：区分"前景内容"和"背景"
        - 背景：大面积连续的单色区域（白色、浅灰色、浅蓝色等）
        - 前景内容：图形、图标、文字、箭头等需要被检测的元素
        
        策略：
        1. 边缘检测（推荐）：有边缘的地方才是真正的内容边界
        2. 灰度阈值：低于阈值的非背景区域
        3. 形态学去噪：去除小噪点
        4. 连通域过滤：去除太小的连通域
        
        Returns:
            二值掩码，255表示有前景内容，0表示背景/无内容
        """
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # ========== 方法1：边缘检测（推荐开启，能更好地识别内容边界）==========
        # 边缘才是真正区分前景和背景的标志
        if self.eval_config.get('use_edge_detection', True):
            edges = cv2.Canny(
                gray,
                self.eval_config['edge_low_threshold'],
                self.eval_config['edge_high_threshold']
            )
            # 膨胀边缘，让边缘区域扩展成有效内容区域
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            edge_mask = edges_dilated
        else:
            edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # ========== 方法2：灰度阈值（作为补充）==========
        threshold = self.eval_config['content_threshold']
        content_by_gray = (gray < threshold).astype(np.uint8) * 255
        
        # ========== 合并两种方法 ==========
        if self.eval_config.get('use_edge_detection', True):
            # 边缘检测开启时：取交集或并集（推荐取并集，但边缘为主）
            # 这里使用并集，但主要依赖边缘检测的结果
            content_mask = cv2.bitwise_or(content_by_gray, edge_mask)
        else:
            content_mask = content_by_gray
        
        # ========== 背景过滤：去除噪点和小区域 ==========
        if self.eval_config.get('filter_background', True):
            # 1. 形态学去噪（开操作：先腐蚀后膨胀，去除小噪点）
            denoise_size = self.eval_config.get('background_denoise_kernel', 3)
            if denoise_size > 0:
                denoise_kernel = np.ones((denoise_size, denoise_size), np.uint8)
                content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, denoise_kernel)
            
            # 2. 连通域过滤：去除太小的连通域（可能是噪点）
            min_area = self.eval_config.get('min_content_area', 50)
            if min_area > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(content_mask, connectivity=8)
                filtered_mask = np.zeros_like(content_mask)
                for i in range(1, num_labels):  # 跳过背景（标签0）
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_area:
                        filtered_mask[labels == i] = 255
                content_mask = filtered_mask
        
        # 日志输出
        content_pixels = int(np.sum(content_mask > 0))
        content_ratio = content_pixels / (h * w) * 100
        self._log(f"内容检测: {content_pixels}px ({content_ratio:.1f}% of image)")
        
        return content_mask
    
    # 需要有 base64 图片才算真正覆盖的元素类型（复杂图片内容）
    IMAGE_CONTENT_TYPES = {
        'icon', 'picture', 'photo', 'chart', 'function_graph', 'screenshot', 
        'image', 'diagram', 'logo', 'heatmap', 'graph', 'line graph', 'bar graph',
        'pie chart', 'scatter plot', 'histogram'
    }
    
    # 基本矢量图形类型（有 XML 就算覆盖）
    VECTOR_SHAPE_TYPES = {
        'rectangle', 'rounded rectangle', 'circle', 'ellipse', 'diamond', 
        'triangle', 'cloud', 'arrow', 'line', 'connector', 'polygon',
        'section_panel', 'title_bar', 'background'
    }
    
    def _create_covered_mask(self, 
                              elements: List[ElementInfo],
                              height: int,
                              width: int,
                              text_xml: str = None) -> Tuple[np.ndarray, List[List[int]]]:
        """
        创建覆盖掩码 - 严格判断有效输出
        
        有效输出的定义：
        1. 对于图片类内容（热力图、chart等）：必须有 base64 才算覆盖
        2. 对于基本矢量图形：有 xml_fragment 就算覆盖
        3. 加上 OCR 识别的文字区域
        
        Returns:
            (covered_mask, existing_bboxes)
            - covered_mask: 二值掩码，255表示已覆盖，0表示未覆盖
            - existing_bboxes: 已有元素的bbox列表
        """
        covered_mask = np.zeros((height, width), dtype=np.uint8)
        existing_bboxes = []
        
        img_area = height * width
        
        valid_count = 0
        skipped_image_no_base64 = 0
        skipped_no_output = 0
        
        for elem in elements:
            elem_type = elem.element_type.lower()
            
            # 计算元素面积比例
            elem_area = (elem.bbox.x2 - elem.bbox.x1) * (elem.bbox.y2 - elem.bbox.y1)
            area_ratio = elem_area / img_area if img_area > 0 else 0
            
            # 判断是否为图片类内容
            is_image_content = elem_type in self.IMAGE_CONTENT_TYPES
            
            # 判断是否有有效输出
            if is_image_content:
                # 图片类内容：必须有 base64 才算覆盖
                has_valid_output = elem.base64 is not None
                if not has_valid_output:
                    skipped_image_no_base64 += 1
                    continue
            else:
                # 基本矢量图形：有 XML 或 base64 就算覆盖
                # 矩形、圆等都是有效的流程图元素，不再跳过"大面积基本图形"
                # 真正漏检的复杂图像内容由策略2检测
                has_valid_output = elem.has_xml() or elem.base64 is not None
                if not has_valid_output:
                    skipped_no_output += 1
                    continue
            
            x1 = max(0, min(width, elem.bbox.x1))
            y1 = max(0, min(height, elem.bbox.y1))
            x2 = max(0, min(width, elem.bbox.x2))
            y2 = max(0, min(height, elem.bbox.y2))
            
            if x2 > x1 and y2 > y1:
                covered_mask[y1:y2, x1:x2] = 255
                existing_bboxes.append([x1, y1, x2, y2])
                valid_count += 1
        
        # 从 text_xml 中提取文字区域
        text_count = 0
        if text_xml:
            text_bboxes = self._extract_text_bboxes_from_xml(text_xml, width, height)
            for bbox in text_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    covered_mask[y1:y2, x1:x2] = 255
                    existing_bboxes.append([x1, y1, x2, y2])
                    text_count += 1
        
        self._log(f"覆盖区域统计: 有效元素={valid_count}, 图片类无base64={skipped_image_no_base64}, 无输出={skipped_no_output}, OCR文字={text_count}")
        
        return covered_mask, existing_bboxes
    
    def _extract_text_bboxes_from_xml(self, text_xml: str, img_width: int, img_height: int) -> List[List[int]]:
        """
        从文字 XML 中提取所有文字元素的 bbox
        
        Args:
            text_xml: 文字处理生成的 XML 内容
            img_width, img_height: 图片尺寸
            
        Returns:
            文字 bbox 列表 [[x1, y1, x2, y2], ...]
        """
        import re
        
        bboxes = []
        
        # 匹配 mxGeometry 标签中的坐标
        # <mxGeometry x="100" y="200" width="50" height="20" as="geometry"/>
        pattern = r'<mxGeometry\s+x="([^"]+)"\s+y="([^"]+)"\s+width="([^"]+)"\s+height="([^"]+)"'
        
        for match in re.finditer(pattern, text_xml):
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                w = float(match.group(3))
                h = float(match.group(4))
                
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img_width, int(x + w))
                y2 = min(img_height, int(y + h))
                
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
            except (ValueError, IndexError):
                continue
        
        return bboxes
    
    def _detect_bad_regions(self,
                            cv2_image: np.ndarray,
                            content_mask: np.ndarray,
                            covered_mask: np.ndarray,
                            existing_bboxes: List[List[int]],
                            img_area: int,
                            elements: List[ElementInfo] = None,
                            context: ProcessingContext = None) -> List[Dict[str, Any]]:
        """
        识别问题区域 - 三重策略检测漏检区域
        
        策略：
        1. 双通道检测：基于内容掩码的细粒度/粗粒度连通域检测
        2. 复杂图像区域检测：检测高方差区域（热力图、照片等）是否有 base64 覆盖
        3. 小框优先NMS + 去重过滤
        
        Returns:
            问题区域列表
        """
        h, w = cv2_image.shape[:2]
        
        # 1. 有内容但未覆盖的区域
        uncovered_content = cv2.bitwise_and(
            content_mask,
            cv2.bitwise_not(covered_mask)
        )
        
        candidates = []
        
        # ===== 策略1: 细粒度通道 =====
        fine_candidates = self._detect_fine_channel(uncovered_content, img_area)
        candidates.extend([(box, 'fine') for box in fine_candidates])
        
        # ===== 策略2: 粗粒度通道 =====
        coarse_candidates = self._detect_coarse_channel(uncovered_content, img_area)
        candidates.extend([(box, 'coarse') for box in coarse_candidates])
        
        # ===== 策略3: 复杂图像区域检测 =====
        # 检测高方差区域（热力图、照片等），如果没有 base64 覆盖则标记为问题区域
        complex_candidates = self._detect_complex_image_regions(cv2_image, elements, img_area, context)
        candidates.extend([(box, 'complex') for box in complex_candidates])
        
        self._log(f"三重检测: 细粒度={len(fine_candidates)}, 粗粒度={len(coarse_candidates)}, 复杂图像={len(complex_candidates)}")
        
        # ===== 小框优先NMS =====
        nms_threshold = self.eval_config['nms_iou_threshold']
        candidates = self._nms_smallest_first(candidates, nms_threshold)
        
        self._log(f"NMS后: {len(candidates)}个候选")
        
        # ===== 与已有元素去重 + 覆盖比例过滤 =====
        bad_regions = self._filter_candidates(
            candidates, covered_mask, existing_bboxes, uncovered_content, img_area
        )
        
        # ===== 合并相邻的区域（距离小于图片短边的10%则合并）=====
        h, w = covered_mask.shape[:2]
        merge_distance = min(h, w) * 0.10  # 合并距离阈值
        bad_regions = self._merge_nearby_regions(bad_regions, merge_distance, img_area)
        
        # 按面积从大到小排序
        bad_regions.sort(key=lambda r: r['area'], reverse=True)
        
        return bad_regions
    
    def _detect_complex_image_regions(self, 
                                       cv2_image: np.ndarray, 
                                       elements: List[ElementInfo],
                                       img_area: int,
                                       context: ProcessingContext) -> List[List[int]]:
        """
        检测复杂图像区域（热力图、照片、图表等）
        
        三重策略：
        1. 检查"图片类但没有 base64"的元素
        2. 检测"完全没有被任何元素覆盖"的高复杂度区域（SAM3 漏检）
        3. 基于图像分析的补充检测
        
        Returns:
            问题区域 bbox 列表
        """
        h, w = cv2_image.shape[:2]
        complex_regions = []
        
        min_region_ratio = self.eval_config.get('complex_min_area_ratio', 0.002)  # 0.2%（更小）
        max_region_ratio = self.eval_config.get('complex_max_area_ratio', 0.30)   # 30%
        min_area = img_area * min_region_ratio
        max_area = img_area * max_region_ratio
        
        # ===== 策略1: 检测"图片类但没有 base64"的元素 =====
        # 注意：跳过面积超过 50% 的大元素（通常是 SAM3 把整图作为 diagram 检测的结果）
        # 这种情况下，图中的其他小组件（箭头、形状、文字）通常已经被正确处理了
        large_element_threshold = 0.50  # 50%
        if elements:
            for elem in elements:
                elem_type = elem.element_type.lower()
                if elem_type in self.IMAGE_CONTENT_TYPES and elem.base64 is None:
                    x1, y1 = max(0, elem.bbox.x1), max(0, elem.bbox.y1)
                    x2, y2 = min(w, elem.bbox.x2), min(h, elem.bbox.y2)
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / img_area
                    
                    # 跳过覆盖整图的大元素（如整图被检测为 diagram）
                    if area_ratio > large_element_threshold:
                        self._log(f"跳过大面积元素: {elem.id}({elem_type}), 面积={area_ratio*100:.1f}% > {large_element_threshold*100}%（可能是整图背景）")
                        continue
                    
                    if area >= min_area * 0.5:  # 正常大小的图片类元素
                        complex_regions.append([x1, y1, x2, y2])
                        self._log(f"检测到未处理的图片类元素: {elem.id}({elem_type}), 面积={area_ratio*100:.1f}%")
        
        # ===== 策略2: 检测没有被"实质内容"覆盖的高复杂度区域 =====
        # 需要排除的区域：
        # 1. 有 base64 图片的元素（真正的图片内容）
        # 2. 面积足够大的矢量图形（如大矩形，占比 > 1%）
        # 3. OCR 文字区域
        # 
        # 注意：小箭头、小圆等不算"覆盖"，因为它们不能代表复杂图像内容
        
        # 创建"已处理"掩码
        processed_mask = np.zeros((h, w), dtype=np.uint8)
        min_element_ratio = 0.01  # 至少 1% 面积的元素才算"覆盖"
        
        # 1. 有 base64 的元素（图片内容，无论大小都算覆盖）
        if elements:
            for elem in elements:
                if elem.base64 is not None:
                    x1, y1 = max(0, elem.bbox.x1), max(0, elem.bbox.y1)
                    x2, y2 = min(w, elem.bbox.x2), min(h, elem.bbox.y2)
                    if x2 > x1 and y2 > y1:
                        processed_mask[y1:y2, x1:x2] = 255
        
        # 2. 面积足够大的矢量图形（排除小箭头、container等）
        # Container 是布局容器，不代表实际内容，不应该算作"覆盖"
        SKIP_TYPES_FOR_COVERAGE = {'container', 'group', 'frame', 'background'}
        if elements:
            for elem in elements:
                if elem.has_xml() and elem.base64 is None:
                    # 跳过布局容器类型
                    if elem.element_type.lower() in SKIP_TYPES_FOR_COVERAGE:
                        continue
                    x1, y1 = max(0, elem.bbox.x1), max(0, elem.bbox.y1)
                    x2, y2 = min(w, elem.bbox.x2), min(h, elem.bbox.y2)
                    elem_area = (x2 - x1) * (y2 - y1)
                    elem_ratio = elem_area / img_area
                    # 只有面积足够大的矢量图形才算"覆盖"
                    if elem_ratio >= min_element_ratio and x2 > x1 and y2 > y1:
                        processed_mask[y1:y2, x1:x2] = 255
        
        # 3. OCR 文字区域
        # 这部分很重要：OCR 已经处理的文字区域不应该被 fallback 重复处理
        text_xml = context.intermediate_results.get('text_xml', '') if hasattr(context, 'intermediate_results') else ''
        if not text_xml and hasattr(context, 'output_dir') and context.output_dir:
            text_xml_path = os.path.join(context.output_dir, 'text_only.drawio')
            if os.path.exists(text_xml_path):
                with open(text_xml_path, 'r', encoding='utf-8') as f:
                    text_xml = f.read()
        
        if text_xml:
            text_bboxes = self._extract_text_bboxes_from_xml(text_xml, w, h)
            for bbox in text_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    # 稍微扩展一点 OCR 区域，避免边缘被检测
                    pad = 5
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                    processed_mask[y1:y2, x1:x2] = 255
        
        all_elements_mask = processed_mask
        
        
        # 计算图像复杂度（局部方差）
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        kernel_size = max(21, min(h, w) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0)
        
        # 边缘检测补充（边缘密度高的区域更可能是图像内容）
        edges = cv2.Canny(gray, 30, 100)
        edge_density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size))
        
        # 组合复杂度指标
        variance_norm = local_variance / (np.max(local_variance) + 1e-6)
        edge_norm = edge_density / (np.max(edge_density) + 1e-6)
        complexity = variance_norm * 0.6 + edge_norm * 0.4
        
        # 阈值化找高复杂度区域
        complexity_threshold = np.percentile(complexity, 75)  # 前 25% 复杂度
        high_complexity_mask = (complexity > complexity_threshold).astype(np.uint8) * 255
        
        # 形态学处理：适度闭操作，不要过度合并（保留独立区域如 4 个热力图）
        kernel_close = np.ones((15, 15), np.uint8)  # 更小的核，避免合并独立区域
        high_complexity_mask = cv2.morphologyEx(high_complexity_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 找出"高复杂度但没有元素覆盖"的区域
        uncovered_complex = cv2.bitwise_and(high_complexity_mask, cv2.bitwise_not(all_elements_mask))
        
        # 进一步形态学处理：先开操作去噪，再闭操作合并相邻区域
        kernel_open = np.ones((7, 7), np.uint8)  # 小核去噪
        uncovered_complex = cv2.morphologyEx(uncovered_complex, cv2.MORPH_OPEN, kernel_open)
        # 使用较大的核合并相邻区域（如同一个图表的多个部分）
        kernel_close2 = np.ones((51, 51), np.uint8)  # 较大核，合并相邻区域
        uncovered_complex = cv2.morphologyEx(uncovered_complex, cv2.MORPH_CLOSE, kernel_close2)
        
        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered_complex, connectivity=8)
        
        self._log(f"未覆盖复杂区域连通域: {num_labels - 1} 个")
        
        for i in range(1, num_labels):
            x, y, rw, rh, pixel_area = stats[i]
            bbox_area = rw * rh
            
            # 面积过滤
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 填充率检查
            fill_ratio = pixel_area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.15:  # 更宽松
                continue
            
            # 宽高比检查
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 8:
                continue
            
            new_bbox = [x, y, x + rw, y + rh]
            
            # 检查是否与已有区域重叠
            is_duplicate = False
            for existing in complex_regions:
                iou = calculate_iou(new_bbox, existing)
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                complex_regions.append(new_bbox)
                self._log(f"检测到未覆盖的复杂区域: ({x},{y})-({x+rw},{y+rh}), 面积={bbox_area/img_area*100:.1f}%")
        
        # ===== 策略3: 检测中等面积未覆盖内容区域（不依赖复杂度） =====
        # 热力图等渐变色图像边缘不多，复杂度检测可能漏掉
        # 直接检测"有内容但未被覆盖"的中等面积区域
        
        # 创建内容掩码（非白色/接近白色的区域）
        _, content_mask_simple = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 减去已处理区域（SAM3 元素 + OCR）
        uncovered_content = cv2.bitwise_and(content_mask_simple, cv2.bitwise_not(all_elements_mask))
        
        # 形态学处理：使用较小的核，避免连接独立区域
        # 开操作去噪
        kernel_open3 = np.ones((5, 5), np.uint8)
        uncovered_content = cv2.morphologyEx(uncovered_content, cv2.MORPH_OPEN, kernel_open3)
        # 小核闭操作，只连接非常近的碎片
        kernel_close3 = np.ones((11, 11), np.uint8)
        uncovered_content = cv2.morphologyEx(uncovered_content, cv2.MORPH_CLOSE, kernel_close3)
        
        # 连通域分析
        num_labels3, labels3, stats3, _ = cv2.connectedComponentsWithStats(uncovered_content, connectivity=8)
        
        # 面积阈值：检测 1% - 15% 的中等面积区域
        min_uncovered_threshold = 0.01  # 最小 1%
        max_uncovered_threshold = 0.15  # 最大 15%（避免检测整个边缘区域）
        
        for i in range(1, num_labels3):
            x, y, rw, rh, pixel_area = stats3[i]
            bbox_area = rw * rh
            area_ratio = bbox_area / img_area
            
            # 面积范围过滤
            if area_ratio < min_uncovered_threshold or area_ratio > max_uncovered_threshold:
                continue
            
            # 填充率检查（避免狭长的边缘区域）
            fill_ratio = pixel_area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.25:  # 更严格的填充率
                continue
            
            # 宽高比检查
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 4:  # 更严格的宽高比
                continue
            
            new_bbox = [x, y, x + rw, y + rh]
            
            # 检查是否与已有区域重叠
            is_duplicate = False
            for existing in complex_regions:
                iou = calculate_iou(new_bbox, existing)
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                complex_regions.append(new_bbox)
                self._log(f"检测到未覆盖区域: ({x},{y})-({x+rw},{y+rh}), 面积={area_ratio*100:.1f}%")
        
        return complex_regions
    
    def _merge_nearby_regions(self, 
                               regions: List[Dict], 
                               merge_distance: float,
                               img_area: int) -> List[Dict]:
        """
        合并相邻的小问题区域
        
        只对面积 < 3% 的小区域进行合并，大区域保持独立
        """
        if len(regions) <= 1:
            return regions
        
        # 分离大区域和小区域
        small_threshold = 0.03  # 3%
        large_regions = [r for r in regions if r['area_ratio'] >= small_threshold]
        small_regions = [r for r in regions if r['area_ratio'] < small_threshold]
        
        if len(small_regions) <= 1:
            return regions  # 没有足够的小区域需要合并
        
        def box_distance(box1, box2):
            """计算两个框之间的最小距离"""
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            if x2_1 < x1_2:
                dx = x1_2 - x2_1
            elif x2_2 < x1_1:
                dx = x1_1 - x2_2
            else:
                dx = 0
            
            if y2_1 < y1_2:
                dy = y1_2 - y2_1
            elif y2_2 < y1_1:
                dy = y1_1 - y2_2
            else:
                dy = 0
            
            return max(dx, dy)
        
        # 只对小区域使用并查集合并
        n = len(small_regions)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = box_distance(small_regions[i]['bbox'], small_regions[j]['bbox'])
                if dist < merge_distance:
                    union(i, j)
        
        # 按组合并小区域
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        merged_small = []
        for indices in groups.values():
            if len(indices) == 1:
                merged_small.append(small_regions[indices[0]])
            else:
                boxes = [small_regions[i]['bbox'] for i in indices]
                merged_box = [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes)
                ]
                merged_area = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                
                merged_small.append({
                    'bbox': merged_box,
                    'area': merged_area,
                    'area_ratio': round(merged_area / img_area, 4),
                    'missing_pixels': sum(small_regions[i]['missing_pixels'] for i in indices),
                    'reason': 'merged_regions',
                    'channel': 'merged',
                    'description': f'合并了{len(indices)}个相邻区域',
                })
        
        # 返回大区域 + 合并后的小区域
        return large_regions + merged_small
    
    def _merge_overlapping_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """合并重叠的边界框"""
        if not boxes:
            return []
        
        # 转换为 numpy 数组方便处理
        boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        merged = [boxes[0]]
        
        for box in boxes[1:]:
            last = merged[-1]
            # 检查是否重叠
            if (box[0] <= last[2] and box[1] <= last[3] and
                box[2] >= last[0] and box[3] >= last[1]):
                # 合并
                merged[-1] = [
                    min(last[0], box[0]),
                    min(last[1], box[1]),
                    max(last[2], box[2]),
                    max(last[3], box[3])
                ]
            else:
                merged.append(box)
        
        return merged
    
    def _detect_fine_channel(self, uncovered_content: np.ndarray, img_area: int) -> List[List[int]]:
        """
        细粒度通道：不做形态学操作，直接连通域分析
        用于检测小图、图标、人脸等小目标
        """
        min_area = img_area * self.eval_config['fine_min_area_ratio']
        max_area = img_area * self.eval_config['fine_max_area_ratio']
        min_fill = self.eval_config['fine_min_fill_ratio']
        max_aspect = self.eval_config['fine_max_aspect_ratio']
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(uncovered_content, connectivity=8)
        
        boxes = []
        for i in range(1, num_labels):
            x, y, rw, rh, cc_area = stats[i]
            if rw <= 0 or rh <= 0:
                continue
            
            bbox_area = rw * rh
            
            # 面积过滤
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 宽高比过滤
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > max_aspect:
                continue
            
            # 填充率过滤
            fill = cc_area / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue
            
            boxes.append([x, y, x + rw, y + rh])
        
        return boxes
    
    def _detect_coarse_channel(self, uncovered_content: np.ndarray, img_area: int) -> List[List[int]]:
        """
        粗粒度通道：使用闭操作合并相邻内容
        用于检测版块、大图、分散的图形组
        """
        min_area = img_area * self.eval_config['coarse_min_area_ratio']
        max_area = img_area * self.eval_config['coarse_max_area_ratio']
        min_fill = self.eval_config['coarse_min_fill_ratio']
        max_aspect = self.eval_config['coarse_max_aspect_ratio']
        kernel_size = self.eval_config['coarse_kernel_size']
        
        # 闭操作合并相邻内容
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(uncovered_content, cv2.MORPH_CLOSE, kernel)
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        boxes = []
        for i in range(1, num_labels):
            x, y, rw, rh, cc_area = stats[i]
            if rw <= 0 or rh <= 0:
                continue
            
            bbox_area = rw * rh
            
            # 面积过滤
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 宽高比过滤
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > max_aspect:
                continue
            
            # 填充率过滤
            fill = cc_area / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue
            
            boxes.append([x, y, x + rw, y + rh])
        
        return boxes
    
    def _nms_smallest_first(self, 
                            candidates: List[Tuple[List[int], str]], 
                            iou_threshold: float) -> List[Tuple[List[int], str]]:
        """
        小框优先NMS：保留小框，抑制被小框高度覆盖的大框
        
        逻辑：
        1. 按面积从小到大排序
        2. 依次处理每个框，保留最小的
        3. 用保留的小框去抑制与之高度重叠的大框
        
        这样可以避免把多个小目标误合并成一个大框
        """
        if not candidates:
            return []
        
        # 计算面积并排序
        boxes_with_area = [(box, channel, (box[2]-box[0])*(box[3]-box[1])) 
                           for box, channel in candidates]
        boxes_with_area.sort(key=lambda x: x[2])  # 按面积升序
        
        keep = []
        suppressed = [False] * len(boxes_with_area)
        
        for i, (box_i, channel_i, area_i) in enumerate(boxes_with_area):
            if suppressed[i]:
                continue
            
            # 保留当前最小的未被抑制的框
            keep.append((box_i, channel_i))
            
            # 用这个小框去抑制后面所有与之高度重叠的框（大的会被抑制）
            for j in range(i + 1, len(boxes_with_area)):
                if suppressed[j]:
                    continue
                
                box_j = boxes_with_area[j][0]
                if calculate_iou(box_i, box_j) > iou_threshold:
                    suppressed[j] = True
        
        return keep
    
    def _filter_candidates(self,
                           candidates: List[Tuple[List[int], str]],
                           covered_mask: np.ndarray,
                           existing_bboxes: List[List[int]],
                           uncovered_content: np.ndarray,
                           img_area: int) -> List[Dict[str, Any]]:
        """
        最终过滤：与已有元素去重 + 覆盖比例过滤
        
        对于 complex 通道（复杂图像检测），使用更宽松的过滤条件
        """
        iou_threshold = self.eval_config['existing_iou_threshold']
        max_covered_ratio = self.eval_config['max_covered_ratio']
        
        bad_regions = []
        
        for box, channel in candidates:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            # 对于复杂图像通道，使用更宽松的过滤条件
            is_complex_channel = (channel == 'complex')
            
            # 1. 与已有元素IoU过高则丢弃（复杂图像通道使用更高阈值）
            current_iou_threshold = 0.8 if is_complex_channel else iou_threshold
            if any(calculate_iou(box, eb) > current_iou_threshold for eb in existing_bboxes):
                self._log(f"过滤候选({channel}): IoU过高") if is_complex_channel else None
                continue
            
            # 2. 框内部若大比例像素已被覆盖（SAM/OCR），则丢弃
            # 对于复杂图像通道，跳过这个检查（因为可能被文字覆盖但图像内容没处理）
            if not is_complex_channel and x2 > x1 and y2 > y1:
                cover_ratio = float(np.mean(covered_mask[y1:y2, x1:x2] > 0))
                if cover_ratio > max_covered_ratio:
                    continue
            
            # 3. 计算该区域内的实际漏检内容像素数
            region_uncovered = uncovered_content[y1:y2, x1:x2]
            missing_pixels = int(np.sum(region_uncovered > 0))
            
            # 对于复杂图像通道，使用区域面积作为 missing_pixels（因为整个区域都是复杂图像）
            if is_complex_channel:
                missing_pixels = area
            
            # 如果漏检内容太少，跳过（复杂图像通道不检查）
            min_missing_ratio = self.eval_config.get('min_missing_content_ratio', 0.10)
            if not is_complex_channel and missing_pixels < area * min_missing_ratio:
                continue
            
            bad_regions.append({
                'bbox': [x1, y1, x2, y2],
                'area': area,
                'area_ratio': round(area / img_area, 4),
                'missing_pixels': missing_pixels,
                'reason': 'uncovered_content' if not is_complex_channel else 'complex_image_no_base64',
                'channel': channel,
                'description': f'区域({x1},{y1})-({x2},{y2})存在未识别的{"复杂图像内容" if is_complex_channel else "内容"} [{channel}通道]',
            })
        
        return bad_regions
    
    def _save_uncovered_visualization(self,
                                      cv2_image: np.ndarray,
                                      uncovered_content: np.ndarray,
                                      covered_mask: np.ndarray,
                                      bad_regions: List[Dict],
                                      output_path: str):
        """
        保存问题区域可视化图 - 重点突出需要 fallback 补救的区域
        
        显示：
        - 原图作为背景
        - 红色半透明填充 + 粗红框标记问题区域（需要 fallback）
        - 问题区域的详细标注
        
        Args:
            cv2_image: 原始图像
            uncovered_content: 未覆盖内容掩码（不再显示，因为大部分是噪点）
            covered_mask: 已覆盖区域掩码（不再显示）
            bad_regions: 问题区域列表
            output_path: 输出路径
        """
        h, w = cv2_image.shape[:2]
        
        # 创建输出图像（原图副本）
        result = cv2_image.copy()
        overlay = cv2_image.copy()
        
        # 1. 画问题区域（红色半透明填充 + 粗边框）
        for i, region in enumerate(bad_regions):
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 半透明红色填充
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            
            # 粗红色边框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # 标注：序号 + 面积 + 原因
            area_pct = region.get('area_ratio', 0) * 100
            channel = region.get('channel', 'unknown')
            reason = region.get('reason', 'unknown')
            
            # 显示简化的原因
            if 'complex' in channel:
                reason_short = "IMAGE_NO_BASE64"
            elif channel == 'fine':
                reason_short = "UNCOVERED_FINE"
            elif channel == 'coarse':
                reason_short = "UNCOVERED_COARSE"
            else:
                reason_short = channel.upper()
            
            label = f"#{i+1} {reason_short} ({area_pct:.1f}%)"
            
            # 背景框让文字更清晰
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 混合半透明层
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # 2. 添加图例（顶部）
        legend_bg = np.zeros((120, w, 3), dtype=np.uint8)
        legend_bg[:] = (40, 40, 40)  # 深灰色背景
        
        cv2.putText(legend_bg, f"METRIC EVALUATION - Problem Regions for Fallback", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(legend_bg, f"Total Bad Regions: {len(bad_regions)}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(legend_bg, f"RED = regions that need fallback (image content without base64 processing)", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # 拼接图例和结果图
        result = np.vstack([legend_bg, result])
        
        cv2.imwrite(output_path, result)
        self._log(f"保存问题区域可视化: {output_path}")
    
    def _save_evaluation_json(self,
                              metrics: Dict,
                              bad_regions: List[Dict],
                              needs_refinement: bool,
                              overall_score: float,
                              output_path: str):
        """
        保存评估结果到 JSON 文件
        
        Args:
            metrics: 详细指标
            bad_regions: 问题区域列表
            needs_refinement: 是否需要二次处理
            overall_score: 总体评分
            output_path: 输出路径
        """
        import json
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        evaluation_result = {
            'overall_score': round(float(overall_score), 2),
            'needs_refinement': bool(needs_refinement),
            'metrics': convert_to_native(metrics),
            'bad_regions': convert_to_native(bad_regions),
            'summary': {
                'score': f"{overall_score:.1f}/100",
                'bad_region_ratio': f"{metrics['total_bad_region_ratio']:.1f}%",
                'bad_region_count': int(metrics['bad_region_count']),
                'pixel_coverage': f"{metrics['pixel_coverage']:.1f}%",
                'element_count': int(metrics['element_count']),
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        self._log(f"保存评估结果: {output_path}")
    
    def save_visualization(self, 
                           context: ProcessingContext,
                           bad_regions: List[Dict],
                           output_path: str):
        """
        保存评估结果可视化图
        
        Args:
            context: 处理上下文
            bad_regions: 问题区域列表
            output_path: 输出路径
        """
        if not context.image_path or not os.path.exists(context.image_path):
            return
        
        img = cv2.imread(context.image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        # 1. 画已检测元素（蓝色）
        for elem in context.elements:
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
        
        # 2. 画问题区域（红色=粗粒度，绿色=细粒度）
        for region in bad_regions:
            x1, y1, x2, y2 = region['bbox']
            color = (0, 0, 255) if region.get('channel') == 'coarse' else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # 标注
            text = f"{region['area_ratio']*100:.1f}%"
            cv2.putText(img, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 3. 图例
        cv2.putText(img, "Blue: Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(img, "Red: Missing (coarse)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "Green: Missing (fine)", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
        self._log(f"保存可视化结果: {output_path}")
    
    def save_uncovered_mask(self,
                            context: ProcessingContext,
                            output_path: str,
                            bad_regions: List[Dict[str, Any]] = None):
        """
        保存问题区域可视化图
        
        显示：
        1. 检测到的问题区域（红色边框 + 半透明填充）
        2. 有 base64 的图片元素（绿色边框）
        3. 无 base64 的图片类元素（黄色边框）
        """
        if not context.image_path or not os.path.exists(context.image_path):
            return
        
        img = cv2.imread(context.image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        result = img.copy()
        overlay = img.copy()
        
        # 1. 画有 base64 的元素（绿色）和无 base64 的图片类元素（黄色）
        for elem in context.elements:
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            
            elem_type = elem.element_type.lower()
            is_image_type = elem_type in self.IMAGE_CONTENT_TYPES
            
            if elem.base64 is not None:
                # 有 base64 的元素：绿色
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 0), 2)
            elif is_image_type:
                # 图片类但无 base64：黄色（这是问题！）
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 255), 2)
        
        # 2. 画检测到的问题区域（红色粗框 + 半透明填充）
        if bad_regions:
            for i, region in enumerate(bad_regions):
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                
                # 半透明红色填充
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
                # 红色粗边框
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # 标注序号和面积
                area_pct = region.get('area_ratio', 0) * 100
                label = f"BAD #{i+1} ({area_pct:.1f}%)"
                cv2.putText(result, label, (x1 + 5, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 混合半透明层
        alpha = 0.25
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # 3. 图例
        legend_y = 40
        cv2.putText(result, f"GREEN: elements with base64 (OK)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        cv2.putText(result, f"YELLOW: image-type without base64 (PROBLEM)", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(result, f"RED: detected bad regions for fallback ({len(bad_regions or [])})", (10, legend_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, result)
        self._log(f"保存问题区域可视化: {output_path}")


# ======================== 快捷函数 ========================

def evaluate_result(elements: List[ElementInfo],
                    image_path: str,
                    canvas_width: int = 0,
                    canvas_height: int = 0,
                    config: Dict = None) -> Dict[str, Any]:
    """
    快捷函数 - 评估转换结果
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        canvas_width: 画布宽度（可选，会自动从图片获取）
        canvas_height: 画布高度（可选）
        config: 评估配置
        
    Returns:
        评估结果字典，包含：
        - overall_score: 总体分数（0-100，即覆盖率）
        - content_coverage: 内容覆盖率
        - missing_rate: 漏检率
        - bad_regions: 问题区域列表
        - metrics: 详细指标
        
    使用示例:
        result = evaluate_result(elements, "test.png")
        print(f"评分: {result['overall_score']}/100")
        print(f"覆盖率: {result['content_coverage']}%")
        print(f"漏检率: {result['missing_rate']}%")
        print(f"问题区域: {len(result['bad_regions'])}个")
        
        for region in result['bad_regions']:
            print(f"  - {region['bbox']}: {region['description']}")
    """
    evaluator = MetricEvaluator(config)
    context = ProcessingContext(
        image_path=image_path,
        elements=elements,
        canvas_width=canvas_width,
        canvas_height=canvas_height
    )
    
    result = evaluator.process(context)
    return result.metadata

def compute_content_coverage(image_path: str, 
                              bboxes: List[List[int]],
                              content_threshold: int = 245) -> Dict[str, float]:
    """
    计算内容覆盖率的简化函数
    
    Args:
        image_path: 图片路径
        bboxes: bbox列表，格式 [[x1,y1,x2,y2], ...]
        content_threshold: 内容检测阈值
        
    Returns:
        {'coverage': 覆盖率, 'missing': 漏检率}
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'coverage': 0, 'missing': 100}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 内容掩码
    content_mask = (gray < content_threshold).astype(np.uint8)
    total_content = np.sum(content_mask > 0)
    
    if total_content == 0:
        return {'coverage': 100, 'missing': 0}
    
    # 覆盖掩码
    covered_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 > x1 and y2 > y1:
            covered_mask[y1:y2, x1:x2] = 1
    
    # 覆盖的内容
    covered_content = np.sum(np.logical_and(content_mask, covered_mask))
    coverage = (covered_content / total_content) * 100
    
    return {
        'coverage': round(coverage, 2),
        'missing': round(100 - coverage, 2)
    }


# ======================== 渲染对比功能 ========================

def compare_with_rendered(original_path: str, 
                          rendered_path: str,
                          config: Dict = None) -> Dict[str, Any]:
    """
    对比原图和渲染后的图像，找出差异区域（遗漏的内容）
    
    Args:
        original_path: 原始图片路径
        rendered_path: DrawIO渲染后的图片路径
        config: 配置参数
            - diff_threshold: 差异阈值（默认30）
            - min_region_area: 最小区域面积（默认500）
            - merge_distance: 相邻区域合并距离（默认20）
    
    Returns:
        {
            'overall_similarity': 整体相似度 (0-100),
            'missing_regions': 差异区域列表 [{'bbox': [x1,y1,x2,y2], 'area': int, ...}],
            'diff_image_path': 差异可视化图路径（如果指定了output_path）
        }
    
    使用示例:
        result = compare_with_rendered("original.png", "rendered.png")
        print(f"相似度: {result['overall_similarity']}%")
        for region in result['missing_regions']:
            print(f"遗漏区域: {region['bbox']}")
    """
    default_config = {
        'diff_threshold': 30,
        'min_region_area': 500,
        'merge_distance': 20,
        'output_path': None,  # 差异可视化输出路径
    }
    cfg = {**default_config, **(config or {})}
    
    # 读取图像
    original = cv2.imread(original_path)
    rendered = cv2.imread(rendered_path)
    
    if original is None or rendered is None:
        return {
            'overall_similarity': 0,
            'missing_regions': [],
            'error': '无法读取图像'
        }
    
    # 确保尺寸一致
    if original.shape != rendered.shape:
        rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]))
    
    h, w = original.shape[:2]
    total_area = h * w
    
    # 计算差异
    diff = cv2.absdiff(original, rendered)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 阈值化找出显著差异区域
    _, diff_mask = cv2.threshold(diff_gray, cfg['diff_threshold'], 255, cv2.THRESH_BINARY)
    
    # 形态学处理：合并相邻区域
    kernel_size = cfg['merge_distance']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # 找连通域
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    missing_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg['min_region_area']:
            continue
        
        x, y, rw, rh = cv2.boundingRect(cnt)
        
        # 计算该区域的差异强度
        region_diff = diff_gray[y:y+rh, x:x+rw]
        diff_intensity = np.mean(region_diff)
        
        missing_regions.append({
            'bbox': [x, y, x+rw, y+rh],
            'area': int(rw * rh),
            'area_ratio': (rw * rh) / total_area,
            'diff_intensity': float(diff_intensity),
            'description': f'渲染差异区域 ({rw}x{rh})'
        })
    
    # 计算整体相似度
    diff_pixels = np.count_nonzero(diff_mask)
    similarity = max(0, 100 - (diff_pixels / total_area) * 100)
    
    # 可视化输出
    output_path = cfg.get('output_path')
    if output_path:
        vis = original.copy()
        for region in missing_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(output_path, vis)
    
    return {
        'overall_similarity': round(similarity, 2),
        'missing_regions': missing_regions,
        'diff_pixels': int(diff_pixels)
    }


def detect_missing_from_rendered_diff(original_path: str,
                                       rendered_path: str,
                                       output_dir: str = None) -> List[Dict]:
    """
    从渲染对比中检测遗漏区域，并裁剪保存
    
    这个函数会：
    1. 对比原图和渲染图
    2. 找出遗漏的区域
    3. 从原图裁剪这些区域
    4. 可选保存为单独的图片
    
    Args:
        original_path: 原始图片路径
        rendered_path: 渲染后的图片路径
        output_dir: 裁剪区域保存目录（可选）
    
    Returns:
        遗漏区域列表，每个包含:
        - bbox: 边界框
        - cropped_image: PIL Image对象
        - base64: base64编码的图像
    
    使用示例:
        missing = detect_missing_from_rendered_diff("original.png", "rendered.png")
        for i, region in enumerate(missing):
            # 可以直接用于生成XML
            base64_data = region['base64']
            bbox = region['bbox']
    """
    import base64
    from io import BytesIO
    
    # 检测差异区域
    result = compare_with_rendered(original_path, rendered_path, {
        'diff_threshold': 25,
        'min_region_area': 300,
        'merge_distance': 15
    })
    
    if not result.get('missing_regions'):
        return []
    
    # 读取原图
    original_pil = Image.open(original_path).convert("RGB")
    
    missing_elements = []
    
    for i, region in enumerate(result['missing_regions']):
        x1, y1, x2, y2 = region['bbox']
        
        # 裁剪
        cropped = original_pil.crop((x1, y1, x2, y2))
        
        # 转base64
        buffer = BytesIO()
        cropped.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        elem = {
            'bbox': region['bbox'],
            'area': region['area'],
            'area_ratio': region['area_ratio'],
            'diff_intensity': region['diff_intensity'],
            'cropped_image': cropped,
            'base64': b64_data,
            'description': region['description']
        }
        
        # 可选保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            crop_path = os.path.join(output_dir, f"missing_region_{i}.png")
            cropped.save(crop_path)
            elem['saved_path'] = crop_path
        
        missing_elements.append(elem)
    
    return missing_elements

