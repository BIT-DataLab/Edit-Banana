"""
任务8：二次处理模块（RefinementProcessor）- Fallback补救

================================================================================
设计思路
================================================================================

【核心目标】
对MetricEvaluator识别出的"问题区域"进行二次处理，将漏检的内容补救回来。
这是一个保守策略：既然SAM3和其他检测器没能识别出这些区域的具体类型，
我们就直接把这些区域作为图片(picture)裁剪下来，转成base64嵌入到最终XML中。

【为什么叫Fallback？】
- SAM3可能漏检某些复杂图像（热力图、3D渲染、人脸、商品照片等）
- OCR可能漏检某些艺术字、图片文字
- 这些漏检内容虽然不能被矢量化，但总比丢失好
- 直接作为picture贴过去，保证"所见即所得"

【处理策略】

1. 保守策略（默认）：
   - 直接裁剪问题区域原图
   - 转成PNG base64
   - 作为picture元素添加
   - 优点：快速、稳定、不会引入新错误
   - 缺点：无法进一步细分区域内的子元素

2. 激进策略（可选，未来扩展）：
   - 对问题区域再次调用SAM3进行细粒度分割
   - 或调用VLM获取更精确的prompt
   - 优点：可能获得更好的识别结果
   - 缺点：增加处理时间，可能引入新错误

【与IMG2XML的对应关系】
本模块对应IMG2XML中的以下逻辑：
- sam3_extractor.py::run_fallback_with_ocr() - 执行fallback检测
- 直接裁切漏检图像，不抠图，保留原始背景
- 给予中等置信度分数(0.5~0.6)

【元素层级】
新增的picture元素放在 LayerLevel.IMAGE (层级2)，
位于基本图形之上、箭头之下。

================================================================================
接口说明
================================================================================

输入：
    - context.image_path: 原始图片路径
    - context.elements: 现有元素列表
    - context.intermediate_results['bad_regions']: 问题区域列表（来自MetricEvaluator）
      每个region包含: bbox, area, area_ratio, missing_pixels, channel, description

输出（ProcessingResult）：
    - success: 是否处理成功
    - elements: 更新后的元素列表（包含新增的picture元素）
    - metadata:
        - new_elements_count: 新增元素数量
        - regions_processed: 处理的区域数量
        - regions_skipped: 跳过的区域数量（可能因为太小/重叠等）

================================================================================
使用示例
================================================================================

    from modules import RefinementProcessor, ProcessingContext
    
    # 1. 先用MetricEvaluator获取问题区域
    eval_result = evaluator.process(context)
    bad_regions = eval_result.metadata['bad_regions']
    
    # 2. 如果有问题区域，进行refinement
    if bad_regions:
        context.intermediate_results['bad_regions'] = bad_regions
        processor = RefinementProcessor()
        result = processor.process(context)
        print(f"新增 {result.metadata['new_elements_count']} 个元素")
    
    # 3. 新元素已自动添加到 context.elements 中

================================================================================
配置参数
================================================================================

    - min_region_area: 最小区域面积（像素），太小的区域跳过
    - min_region_ratio: 最小区域面积比例（相对于图片），太小的跳过
    - default_confidence: 新元素的默认置信度分数
    - expand_margin: 裁剪时向外扩展的边距（像素），避免边缘切割

================================================================================
"""

import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult, LayerLevel


class RefinementProcessor(BaseProcessor):
    """
    二次处理模块（Fallback补救）
    
    对MetricEvaluator识别出的问题区域进行二次处理。
    
    核心策略（保守）：
        - 直接裁剪问题区域作为picture
        - 转成base64嵌入XML
        - 保证"所见即所得"，不丢失任何内容
    
    可选扩展（TODO）：
        - 激进策略：对问题区域再次调用SAM3
        - VLM增强：获取更精确的prompt
        - 局部高清化：提升小图质量
    """
    
    # 默认配置参数
    DEFAULT_CONFIG = {
        'min_region_area': 100,           # 最小区域面积（像素），太小跳过
        'min_region_ratio': 0.0005,       # 最小区域面积比例(0.05%)，太小跳过
        'default_confidence': 0.5,        # 新元素的默认置信度
        'expand_margin': 5,               # 裁剪时向外扩展的边距（像素）- 增加边距避免切边
        'skip_if_mostly_white': True,     # 是否跳过大部分为白色的区域
        'white_threshold': 0.95,          # 白色像素占比阈值 - 放宽到95%，只跳过真正纯白区域
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        # 合并用户配置和默认配置
        self.refine_config = {**self.DEFAULT_CONFIG, **(config or {})}
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 对问题区域进行Fallback补救
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult，新增的元素会被添加到context.elements中
        """
        self._log("开始二次处理（Fallback补救）")
        
        # 获取问题区域
        bad_regions = context.intermediate_results.get('bad_regions', [])
        
        if not bad_regions:
            self._log("没有问题区域需要处理")
            return ProcessingResult(
                success=True,
                elements=context.elements,
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                metadata={
                    'new_elements_count': 0,
                    'regions_processed': 0,
                    'regions_skipped': 0
                }
            )
        
        # 加载原图
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        original_image = Image.open(context.image_path).convert("RGB")
        img_width, img_height = original_image.size
        img_area = img_width * img_height
        
        # 加载CV2图像用于白色检测
        cv2_image = None
        if self.refine_config.get('skip_if_mostly_white', True):
            cv2_image = cv2.imread(context.image_path)
        
        # 处理问题区域
        new_elements = []
        skipped_count = 0
        start_id = len(context.elements)
        
        # 获取配置参数
        min_area = self.refine_config.get('min_region_area', 100)
        min_ratio = self.refine_config.get('min_region_ratio', 0.0005)
        
        for i, region in enumerate(bad_regions):
            try:
                bbox = region.get('bbox', [])
                if len(bbox) != 4:
                    skipped_count += 1
                    continue
                
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                
                # 面积过滤
                if area < min_area or area < img_area * min_ratio:
                    self._log(f"  区域{i}面积太小({area}px)，跳过")
                    skipped_count += 1
                    continue
                
                # 白色内容过滤（可选）
                if cv2_image is not None and self._is_mostly_white(cv2_image, bbox):
                    self._log(f"  区域{i}大部分为白色，跳过")
                    skipped_count += 1
                    continue
                
                # 处理该区域
                elem = self._process_region(
                    region,
                    original_image,
                    start_id + len(new_elements),
                    img_width,
                    img_height
                )
                if elem:
                    new_elements.append(elem)
                else:
                    skipped_count += 1
                    
            except Exception as e:
                self._log(f"区域{i}处理失败: {e}")
                skipped_count += 1
        
        # 合并到现有元素
        context.elements.extend(new_elements)
        
        self._log(f"二次处理完成: 新增{len(new_elements)}个元素，跳过{skipped_count}个")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'new_elements_count': len(new_elements),
                'regions_processed': len(bad_regions),
                'regions_skipped': skipped_count
            }
        )
    
    def _is_mostly_white(self, cv2_image: np.ndarray, bbox: List[int]) -> bool:
        """
        检查区域是否大部分为白色
        
        用于过滤那些实际上没什么内容的"空白"区域
        """
        x1, y1, x2, y2 = bbox
        h, w = cv2_image.shape[:2]
        
        # 边界检查
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        
        if x2 <= x1 or y2 <= y1:
            return True
        
        # 裁剪区域并检查白色像素比例
        region = cv2_image[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # 灰度>245认为是白色
        white_pixels = np.sum(gray > 245)
        total_pixels = gray.size
        
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 1.0
        threshold = self.refine_config.get('white_threshold', 0.85)
        
        return white_ratio > threshold
    
    def _process_region(self,
                        region: Dict[str, Any],
                        original_image: Image.Image,
                        element_id: int,
                        img_width: int,
                        img_height: int) -> Optional[ElementInfo]:
        """
        处理单个问题区域
        
        当前策略（保守）：直接裁剪原图区域，转成picture元素
        
        Args:
            region: 问题区域信息（来自MetricEvaluator）
            original_image: PIL原图
            element_id: 分配给新元素的ID
            img_width, img_height: 图片尺寸
            
        Returns:
            ElementInfo 或 None（如果处理失败）
        """
        bbox = region.get('bbox', [])
        if len(bbox) != 4:
            return None
        
        x1, y1, x2, y2 = bbox
        
        # 边距扩展（可选，避免边缘切割）
        margin = self.refine_config.get('expand_margin', 2)
        if margin > 0:
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_width, x2 + margin)
            y2 = min(img_height, y2 + margin)
        
        # 裁剪区域
        cropped = original_image.crop((x1, y1, x2, y2))
        
        # 转base64
        base64_str = self._image_to_base64(cropped)
        
        # 获取配置的置信度
        confidence = self.refine_config.get('default_confidence', 0.5)
        
        # 构建处理备注
        channel = region.get('channel', 'unknown')
        area_ratio = region.get('area_ratio', 0) * 100
        missing_pixels = region.get('missing_pixels', 0)
        
        notes = [
            f"Fallback补救: 检测通道={channel}",
            f"区域占比={area_ratio:.2f}%, 漏检像素={missing_pixels}",
            region.get('description', '')
        ]
        
        # 创建元素
        element = ElementInfo(
            id=element_id,
            element_type='picture',  # 作为图片处理
            bbox=BoundingBox(x1, y1, x2, y2),
            score=confidence,
            base64=base64_str,
            layer_level=LayerLevel.IMAGE.value,  # 图片层级
            source_prompt='refinement_fallback',
            processing_notes=[n for n in notes if n]  # 过滤空字符串
        )
        
        # 生成 XML 片段
        self._generate_xml_fragment(element)
        
        return element
    
    def _generate_xml_fragment(self, element: ElementInfo):
        """
        为 fallback 元素生成 XML 片段
        
        Args:
            element: 元素信息
        """
        x1 = element.bbox.x1
        y1 = element.bbox.y1
        width = element.bbox.x2 - element.bbox.x1
        height = element.bbox.y2 - element.bbox.y1
        
        # DrawIO 图片样式
        style = (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;"
            f"image=data:image/png,{element.base64};"
        )
        
        # DrawIO的id必须从2开始（0和1是保留的根元素）
        cell_id = element.id + 2
        
        element.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64"""
        import io
        import base64
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def save_visualization(self, 
                           context: ProcessingContext,
                           new_elements: List[ElementInfo],
                           output_path: str):
        """
        保存refinement结果可视化图
        
        Args:
            context: 处理上下文
            new_elements: 新增的元素列表
            output_path: 输出路径
        """
        if not context.image_path or not os.path.exists(context.image_path):
            return
        
        img = cv2.imread(context.image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        # 1. 画原有元素（蓝色细框）
        for elem in context.elements:
            if elem not in new_elements:
                x1 = max(0, min(w, elem.bbox.x1))
                y1 = max(0, min(h, elem.bbox.y1))
                x2 = max(0, min(w, elem.bbox.x2))
                y2 = max(0, min(h, elem.bbox.y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 1)
        
        # 2. 画新增元素（红色粗框 + 标注）
        for i, elem in enumerate(new_elements):
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 标注序号
            text = f"NEW-{i}"
            cv2.putText(img, text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 3. 图例
        cv2.putText(img, f"Original: {len(context.elements) - len(new_elements)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.putText(img, f"New (Fallback): {len(new_elements)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, img)
        self._log(f"保存refinement可视化结果: {output_path}")


# ======================== 快捷函数 ========================

def refine_bad_regions(elements: List[ElementInfo],
                       bad_regions: List[Dict],
                       image_path: str,
                       config: Dict = None) -> List[ElementInfo]:
    """
    快捷函数 - 处理问题区域
    
    Args:
        elements: 现有元素列表
        bad_regions: 问题区域列表（来自MetricEvaluator）
        image_path: 原始图片路径
        config: 可选配置参数
        
    Returns:
        更新后的元素列表（包含新增的fallback元素）
        
    使用示例:
        # 基本用法
        elements = refine_bad_regions(elements, bad_regions, "test.png")
        
        # 自定义配置
        elements = refine_bad_regions(elements, bad_regions, "test.png", {
            'default_confidence': 0.6,
            'skip_if_mostly_white': False
        })
    """
    processor = RefinementProcessor(config)
    context = ProcessingContext(
        image_path=image_path,
        elements=elements.copy()
    )
    context.intermediate_results['bad_regions'] = bad_regions
    
    result = processor.process(context)
    return result.elements

def evaluate_and_refine(elements: List[ElementInfo],
                        image_path: str,
                        eval_config: Dict = None,
                        refine_config: Dict = None) -> Dict[str, Any]:
    """
    一键执行评估+补救的便捷函数
    
    这个函数整合了MetricEvaluator和RefinementProcessor的完整流程。
    
    Args:
        elements: 已检测的元素列表
        image_path: 原始图片路径
        eval_config: MetricEvaluator配置（可选）
        refine_config: RefinementProcessor配置（可选）
        
    Returns:
        字典包含：
        - elements: 最终元素列表（包含原有+新增）
        - evaluation: 评估结果字典
        - refinement: 补救结果字典（如果执行了的话）
        
    使用示例:
        result = evaluate_and_refine(elements, "test.png")
        
        print(f"评分: {result['evaluation']['overall_score']}/100")
        print(f"新增元素: {result['refinement']['new_elements_count']}")
        print(f"最终元素数: {len(result['elements'])}")
    """
    from .metric_evaluator import MetricEvaluator
    
    # 1. 评估
    evaluator = MetricEvaluator(eval_config)
    context = ProcessingContext(
        image_path=image_path,
        elements=elements.copy()
    )
    eval_result = evaluator.process(context)
    
    result = {
        'evaluation': eval_result.metadata,
        'refinement': None,
        'elements': context.elements
    }
    
    # 2. 如果需要refinement且有问题区域
    bad_regions = eval_result.metadata.get('bad_regions', [])
    if eval_result.metadata.get('needs_refinement', False) and bad_regions:
        context.intermediate_results['bad_regions'] = bad_regions
        processor = RefinementProcessor(refine_config)
        refine_result = processor.process(context)
        
        result['refinement'] = refine_result.metadata
        result['elements'] = context.elements
    
    return result


def refine_from_rendered_comparison(elements: List[ElementInfo],
                                     original_path: str,
                                     rendered_path: str,
                                     config: Dict = None) -> Dict[str, Any]:
    """
    基于渲染对比进行补救
    
    流程：
    1. 对比原图和渲染后的图像
    2. 找出差异区域（遗漏的内容）
    3. 从原图裁剪这些区域
    4. 作为picture元素添加
    
    Args:
        elements: 现有元素列表
        original_path: 原始图片路径
        rendered_path: DrawIO渲染后的图片路径
        config: 配置参数
            - diff_threshold: 差异阈值（默认30）
            - min_region_area: 最小区域面积（默认300）
            - expand_margin: 裁剪扩展边距（默认5）
    
    Returns:
        {
            'elements': 更新后的元素列表,
            'comparison': 对比结果,
            'new_count': 新增元素数量
        }
    
    使用示例:
        result = refine_from_rendered_comparison(
            elements, 
            "original.png", 
            "rendered.png"
        )
        print(f"相似度: {result['comparison']['overall_similarity']}%")
        print(f"新增: {result['new_count']}个元素")
        final_elements = result['elements']
    """
    from .metric_evaluator import compare_with_rendered
    import io
    import base64
    
    default_config = {
        'diff_threshold': 30,
        'min_region_area': 300,
        'expand_margin': 5,
        'default_confidence': 0.4  # 渲染对比补救的置信度较低
    }
    cfg = {**default_config, **(config or {})}
    
    # 1. 对比原图和渲染图
    comparison = compare_with_rendered(original_path, rendered_path, {
        'diff_threshold': cfg['diff_threshold'],
        'min_region_area': cfg['min_region_area'],
        'merge_distance': 15
    })
    
    missing_regions = comparison.get('missing_regions', [])
    
    if not missing_regions:
        return {
            'elements': elements,
            'comparison': comparison,
            'new_count': 0
        }
    
    # 2. 读取原图
    original = Image.open(original_path).convert("RGB")
    img_w, img_h = original.size
    
    # 3. 处理每个遗漏区域
    new_elements = []
    start_id = max([e.id for e in elements], default=0) + 1
    margin = cfg['expand_margin']
    
    for i, region in enumerate(missing_regions):
        x1, y1, x2, y2 = region['bbox']
        
        # 扩展边距
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_w, x2 + margin)
        y2 = min(img_h, y2 + margin)
        
        # 裁剪
        cropped = original.crop((x1, y1, x2, y2))
        
        # 转base64
        buffer = io.BytesIO()
        cropped.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 创建元素
        element = ElementInfo(
            id=start_id + i,
            element_type='picture',
            bbox=BoundingBox(x1, y1, x2, y2),
            score=cfg['default_confidence'],
            base64=b64_data,
            layer_level=LayerLevel.IMAGE.value,
            source_prompt='rendered_comparison_fallback',
            processing_notes=[
                f"渲染对比补救",
                f"差异强度: {region.get('diff_intensity', 0):.1f}",
                region.get('description', '')
            ]
        )
        new_elements.append(element)
    
    # 4. 合并
    all_elements = elements + new_elements
    
    return {
        'elements': all_elements,
        'comparison': comparison,
        'new_count': len(new_elements)
    }

