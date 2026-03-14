"""
SAM3 info extractor: extract diagram elements (shapes, arrows, icons, background) from images.
Prompt groups and thresholds are loaded from config.yaml.
"""

import os
import sys
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import threading
from prompts.arrow import ARROW_PROMPT
from prompts.background import BACKGROUND_PROMPT
from prompts.shape import SHAPE_PROMPT
from prompts.image import IMAGE_PROMPT

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, BoundingBox, ProcessingResult


# ======================== 提示词分组枚举 ========================
class PromptGroup(Enum):
    """提示词分组"""
    IMAGE = "image"          # 图片类（需要转base64）
    ARROW = "arrow"          # 箭头类（需要方向检测）
    BASIC_SHAPE = "shape"    # 基本图形（需要取色矢量化）
    BACKGROUND = "background"  # 背景/容器类


@dataclass
class PromptGroupConfig:
    """提示词组配置"""
    name: str                           # 组名
    prompts: List[str] = field(default_factory=list)  # 该组的提示词
    score_threshold: float = 0.5        # 置信度阈值
    min_area: int = 100                 # 最小面积
    priority: int = 1                   # 去重优先级（越高越优先保留）
    description: str = ""               # 描述
    
    def add_prompt(self, prompt: str):
        """添加提示词"""
        if prompt not in self.prompts:
            self.prompts.append(prompt)
    
    def remove_prompt(self, prompt: str):
        """移除提示词"""
        if prompt in self.prompts:
            self.prompts.remove(prompt)


# ======================== 配置加载器 ========================
class ConfigLoader:
    """从config.yaml加载词组配置"""
    
    _instance = None
    _config = None
    _config_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config_path(cls) -> str:
        """获取配置文件路径"""
        if cls._config_path is None:
            cls._config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "config.yaml"
            )
        return cls._config_path
    
    @classmethod
    def load_config(cls, force_reload: bool = False) -> dict:
        """加载配置文件"""
        if cls._config is None or force_reload:
            config_path = cls.get_config_path()
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
            else:
                print(f"[ConfigLoader] Config not found: {config_path}, using defaults")
                cls._config = cls._get_default_config()
        return cls._config
    
    @classmethod
    def _get_default_config(cls) -> dict:
        """获取默认配置（当config.yaml不存在时使用）"""
        return {
            'sam3': {
                'checkpoint_path': '',
                'bpe_path': '',
            },
            'prompt_groups': {
                'image': {
                    'name': '图片类',
                    'prompts': ['icon', 'picture', 'logo', 'chart'],
                    'score_threshold': 0.5,
                    'min_area': 100,
                    'priority': 2,
                },
                'arrow': {
                    'name': '箭头类',
                    'prompts': ['arrow', 'line', 'connector'],
                    'score_threshold': 0.45,
                    'min_area': 50,
                    'priority': 4,
                },
                'shape': {
                    'name': '基本图形',
                    'prompts': ['rectangle', 'rounded rectangle', 'diamond', 'ellipse'],
                    'score_threshold': 0.5,
                    'min_area': 200,
                    'priority': 3,
                },
                'background': {
                    'name': '背景容器',
                    'prompts': ['section_panel', 'title bar', 'container'],
                    'score_threshold': 0.25,  # 降低阈值以检测更多背景色块
                    'min_area': 500,
                    'priority': 1,
                },
            },
            'text_filter': {
                'blacklist': ['text', 'word', 'label'],
                'keywords': ['text', 'word'],
            },
            'deduplication': {
                'iou_threshold': 0.7,
                'arrow_iou_threshold': 0.85,
            },
        }
    
    @classmethod
    def get_prompt_groups(cls) -> Dict[PromptGroup, PromptGroupConfig]:
        """从配置文件加载词组配置"""
        config = cls.load_config()
        prompt_groups_config = config.get('prompt_groups', {})
        
        result = {}
        
        # 映射配置键到枚举
        key_to_enum = {
            'image': PromptGroup.IMAGE,
            'arrow': PromptGroup.ARROW,
            'shape': PromptGroup.BASIC_SHAPE,
            'background': PromptGroup.BACKGROUND,
        }
        prompt_mapping = {
            'image': IMAGE_PROMPT,
            'arrow': ARROW_PROMPT,
            'shape': SHAPE_PROMPT,
            'background': BACKGROUND_PROMPT,
        }
        
        
        for key, enum_val in key_to_enum.items():
            if key in prompt_groups_config:
                # 从映射关系获取提示词
                prompts = prompt_mapping.get(key, [])
                # 从config.yaml读取其他配置（阈值、面积、优先级等）
                group_cfg = prompt_groups_config.get(key, {})
                result[enum_val] = PromptGroupConfig(
                    name=group_cfg.get('name', key),
                    prompts=prompts,
                    score_threshold=group_cfg.get('score_threshold', 0.5),
                    min_area=group_cfg.get('min_area', 100),
                    priority=group_cfg.get('priority', 1),
                    description=group_cfg.get('description', ''),
                )
        
        return result
    
    @classmethod
    def get_text_filter(cls) -> dict:
        """获取文字过滤配置"""
        config = cls.load_config()
        return config.get('text_filter', {'blacklist': [], 'keywords': []})
    
    @classmethod
    def get_deduplication_config(cls) -> dict:
        """获取去重配置"""
        config = cls.load_config()
        return config.get('deduplication', {
            'iou_threshold': 0.7,
            'arrow_iou_threshold': 0.85,
        })
    
    @classmethod
    def get_drawio_styles(cls) -> dict:
        """获取DrawIO样式配置"""
        config = cls.load_config()
        return config.get('drawio_styles', {})
    
    @classmethod
    def get_sam3_config(cls) -> dict:
        """获取SAM3配置"""
        config = cls.load_config()
        return config.get('sam3', {})


# ======================== SAM3模型封装 ========================
class SAM3Model(ModelWrapper):
    """SAM3模型封装"""
    
    def __init__(self, checkpoint_path: str, bpe_path: str, device: str = None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        
        # 图像状态缓存
        self._state_cache = OrderedDict()
        self._max_cache_size = 3
        self._cache_lock = threading.Lock()
    
    def load(self):
        """加载SAM3模型"""
        if self._is_loaded:
            return
            
        print(f"[SAM3Model] 加载模型中... (设备: {self.device})")
        
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        self._model = build_sam3_image_model(
            bpe_path=self.bpe_path,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            device=self.device
        )
        self._processor = Sam3Processor(self._model)
        self._is_loaded = True
        
        print("[SAM3Model] 模型加载完成！")
    
    def predict(self, image_path: str, prompts: List[str], 
                score_threshold: float = 0.5,
                min_area: int = 100) -> List[Dict[str, Any]]:
        """
        SAM3推理
        
        Args:
            image_path: 图片路径
            prompts: 提示词列表
            score_threshold: 置信度阈值
            min_area: 最小面积阈值
            
        Returns:
            元素列表
        """
        if not self._is_loaded:
            self.load()
        
        state, pil_image = self._get_image_state(image_path)
        
        results = []
        for prompt in prompts:
            self._processor.reset_all_prompts(state)
            result_state = self._processor.set_text_prompt(prompt=prompt, state=state)
            
            masks = result_state.get("masks", [])
            boxes = result_state.get("boxes", [])
            scores = result_state.get("scores", [])
            
            num_masks = masks.shape[0] if (isinstance(masks, torch.Tensor) and masks.dim() > 0) else len(masks)
            
            for i in range(num_masks):
                score = scores[i]
                score_val = score.item() if hasattr(score, 'item') else float(score)
                
                if score_val < score_threshold:
                    continue
                
                # 提取bbox
                box = boxes[i]
                bbox = box.cpu().numpy().tolist() if isinstance(box, torch.Tensor) else box
                bbox = [int(coord) for coord in bbox]
                
                # 检查面积
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < min_area:
                    continue
                
                # 提取mask
                mask = masks[i]
                binary_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask.squeeze()
                binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255
                
                # 提取polygon
                polygon = self._extract_polygon(binary_mask, min_area)
                
                if polygon:
                    results.append({
                        'prompt': prompt,
                        'bbox': bbox,
                        'score': score_val,
                        'mask': binary_mask,
                        'polygon': polygon,
                        'area': area
                    })
        
        return results
    
    def _get_image_state(self, image_path: str):
        """获取或创建图像状态（LRU缓存）"""
        with self._cache_lock:
            if image_path in self._state_cache:
                self._state_cache.move_to_end(image_path)
                cache_item = self._state_cache[image_path]
                return cache_item["state"], cache_item["pil_image"]
        
        pil_image = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(pil_image)
        
        cache_item = {"state": state, "pil_image": pil_image}
        
        with self._cache_lock:
            if image_path in self._state_cache:
                self._state_cache.move_to_end(image_path)
                return self._state_cache[image_path]["state"], self._state_cache[image_path]["pil_image"]
            
            self._state_cache[image_path] = cache_item
            
            if len(self._state_cache) > self._max_cache_size:
                self._state_cache.popitem(last=False)
        
        return state, pil_image
    
    def _extract_polygon(self, binary_mask: np.ndarray, 
                         min_area: int = 100, 
                         epsilon_factor: float = 0.02) -> List[List[int]]:
        """从mask提取多边形轮廓"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            return approx.reshape(-1, 2).tolist()
        
        return []
    
    def clear_cache(self):
        """清空图像缓存"""
        with self._cache_lock:
            self._state_cache.clear()


# ======================== SAM3信息提取器 ========================
class Sam3InfoExtractor(BaseProcessor):
    """Extract diagram elements via SAM3; prompt groups and thresholds from config."""

    def __init__(self, config=None, checkpoint_path: str = None, bpe_path: str = None):
        super().__init__(config)
        
        # 从配置文件加载词组（不再硬编码）
        self.prompt_groups = ConfigLoader.get_prompt_groups()
        self.text_filter = ConfigLoader.get_text_filter()
        self.dedup_config = ConfigLoader.get_deduplication_config()
        
        # 加载SAM3模型配置
        sam3_config = ConfigLoader.get_sam3_config()
        self._checkpoint_path = checkpoint_path or sam3_config.get('checkpoint_path', '')
        self._bpe_path = bpe_path or sam3_config.get('bpe_path', '')
        
        self._sam3_model: Optional[SAM3Model] = None
        self._current_image_path: Optional[str] = None

    def reload_config(self):
        """Reload config from disk."""
        ConfigLoader.load_config(force_reload=True)
        self.prompt_groups = ConfigLoader.get_prompt_groups()
        self.text_filter = ConfigLoader.get_text_filter()
        self.dedup_config = ConfigLoader.get_deduplication_config()
        self._log("Config reloaded")

    def load_model(self):
        """Load SAM3 model."""
        if self._sam3_model is None:
            self._sam3_model = SAM3Model(
                checkpoint_path=self._checkpoint_path,
                bpe_path=self._bpe_path
            )
        if not self._sam3_model.is_loaded:
            self._sam3_model.load()
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 分组提取图片中的所有元素
        
        Args:
            context: 处理上下文，需要包含 image_path
            
        Returns:
            ProcessingResult: 包含所有提取的ElementInfo
        """
        self._log(f"开始处理: {context.image_path}")
        
        # 保存当前图像路径（供去重分析使用）
        self._current_image_path = context.image_path
        
        self.load_model()
        
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        
        all_elements = []
        group_stats = {}
        process_order = [
            PromptGroup.BACKGROUND,
            PromptGroup.BASIC_SHAPE, 
            PromptGroup.IMAGE,
            PromptGroup.ARROW
        ]
        
        for group_type in process_order:
            if group_type not in self.prompt_groups:
                continue
                
            group_config = self.prompt_groups[group_type]
            
            if not group_config.prompts:
                continue
                
            self._log(f"  处理组 [{group_config.name}]: {len(group_config.prompts)}个提示词")
            
            raw_results = self._sam3_model.predict(
                context.image_path,
                group_config.prompts,
                score_threshold=group_config.score_threshold,
                min_area=group_config.min_area
            )
            raw_results = self._filter_text_elements(raw_results)
            
            elements = self._convert_to_elements(
                raw_results, 
                start_id=len(all_elements),
                source_group=group_type.value,
                group_priority=group_config.priority
            )
            
            all_elements.extend(elements)
            group_stats[group_config.name] = len(elements)
            
            self._log(f"    提取到 {len(elements)} 个元素")
        
        # 组间去重
        all_elements = self._deduplicate_cross_groups(all_elements)
        
        # 过滤被大图完全包含的小元素
        all_elements = self._filter_contained_elements(all_elements)
        
        context.elements = all_elements
        
        result = ProcessingResult(
            success=True,
            elements=all_elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'group_stats': group_stats,
                'total_before_dedup': sum(group_stats.values()),
                'total_after_dedup': len(all_elements),
                'groups_processed': list(group_stats.keys())
            }
        )
        
        self._log(f"Done: {len(all_elements)} elements (before dedup: {sum(group_stats.values())})")
        return result

    def extract_by_group(self, context: ProcessingContext, 
                         group_type: PromptGroup) -> ProcessingResult:
        """Extract only the given prompt group."""
        self._log(f"Extract group [{group_type.value}]: {context.image_path}")
        
        self.load_model()
        
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        
        if group_type not in self.prompt_groups:
            return ProcessingResult(
                success=False,
                error_message=f"未知的组类型: {group_type}"
            )
        
        group_config = self.prompt_groups[group_type]
        
        raw_results = self._sam3_model.predict(
            context.image_path,
            group_config.prompts,
            score_threshold=group_config.score_threshold,
            min_area=group_config.min_area
        )
        raw_results = self._filter_text_elements(raw_results)
        elements = self._convert_to_elements(
            raw_results,
            start_id=0,
            source_group=group_type.value,
            group_priority=group_config.priority
        )
        
        elements = self._deduplicate_within_group(elements)
        
        context.elements = elements
        
        return ProcessingResult(
            success=True,
            elements=elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'group': group_type.value,
                'prompts_used': group_config.prompts,
                'element_count': len(elements)
            }
        )
    
    def extract_with_custom_prompts(self, context: ProcessingContext,
                                    prompts: List[str],
                                    score_threshold: float = 0.5,
                                    min_area: int = 100) -> ProcessingResult:
        """
        使用自定义提示词提取（不使用分组）
        
        Args:
            context: 处理上下文
            prompts: 自定义提示词列表
            score_threshold: 置信度阈值
            min_area: 最小面积
        """
        self._log(f"自定义提取: {prompts}")
        
        self.load_model()
        
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        
        raw_results = self._sam3_model.predict(
            context.image_path,
            prompts,
            score_threshold=score_threshold,
            min_area=min_area
        )
        
        elements = self._convert_to_elements(
            raw_results,
            start_id=0,
            source_group="custom",
            group_priority=2
        )
        
        elements = self._deduplicate_within_group(elements)
        
        context.elements = elements
        
        return ProcessingResult(
            success=True,
            elements=elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'prompts_used': prompts,
                'element_count': len(elements)
            }
        )
    
    def _filter_text_elements(self, raw_results: List[Dict]) -> List[Dict]:
        """Filter out text-type elements by blacklist/keywords."""
        blacklist = set(self.text_filter.get('blacklist', []))
        keywords = self.text_filter.get('keywords', [])
        
        filtered = []
        for item in raw_results:
            prompt = item['prompt'].lower()
            if prompt in blacklist:
                continue
            is_text = False
            for kw in keywords:
                if kw in prompt:
                    is_text = True
                    break
            
            if not is_text:
                filtered.append(item)
        
        return filtered
    
    def _convert_to_elements(self, raw_results: List[Dict], 
                             start_id: int = 0,
                             source_group: str = "",
                             group_priority: int = 1) -> List[ElementInfo]:
        """将原始结果转换为ElementInfo列表"""
        elements = []
        
        for i, item in enumerate(raw_results):
            bbox = BoundingBox.from_list(item['bbox'])
            
            element = ElementInfo(
                id=start_id + i,
                element_type=item['prompt'],
                bbox=bbox,
                score=item['score'],
                polygon=item['polygon'],
                mask=item['mask'],
                source_prompt=item['prompt']
            )
            
            element.processing_notes.append(f"source_group={source_group}")
            element.processing_notes.append(f"area={item.get('area', bbox.area)}")
            element._group_priority = group_priority
            element._source_group = source_group
            
            elements.append(element)
        
        return elements
    
    def _deduplicate_within_group(self, elements: List[ElementInfo], 
                                  iou_threshold: float = None) -> List[ElementInfo]:
        """组内去重"""
        if not elements:
            return elements
        
        if iou_threshold is None:
            iou_threshold = self.dedup_config.get('iou_threshold', 0.7) + 0.15  # 组内阈值稍高
        
        sorted_elements = sorted(elements, key=lambda x: x.score, reverse=True)
        
        keep = []
        dropped = set()
        
        for i, elem_i in enumerate(sorted_elements):
            if i in dropped:
                continue
            
            keep.append(elem_i)
            
            for j in range(i + 1, len(sorted_elements)):
                if j in dropped:
                    continue
                
                iou = self._calculate_iou(
                    elem_i.bbox.to_list(),
                    sorted_elements[j].bbox.to_list()
                )
                
                if iou > iou_threshold:
                    dropped.add(j)
        
        for i, elem in enumerate(keep):
            elem.id = i
        
        return keep
    
    def _analyze_region_complexity(self, image_path: str, bbox: List[int]) -> dict:
        """Analyze region complexity (texture, border) for shape vs image classification."""
        try:
            cv2_image = cv2.imread(image_path)
            x1, y1, x2, y2 = bbox
            roi = cv2_image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {'classification': 'unknown', 'is_complex': False}
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 计算拉普拉斯方差（纹理/边缘丰富度）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 计算标准差（颜色变化）
            std_dev = np.std(gray)
            
            # 检测边缘
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / edges.size
            
            # 检测是否有清晰的矩形边框
            h, w = roi.shape[:2]
            border_size = max(3, min(10, w // 20, h // 20))
            
            # 采样边框区域
            top_edge = gray[:border_size, :].flatten()
            bottom_edge = gray[-border_size:, :].flatten()
            left_edge = gray[:, :border_size].flatten()
            right_edge = gray[:, -border_size:].flatten()
            
            border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
            inner_margin = border_size + 2
            if w > 2 * inner_margin and h > 2 * inner_margin:
                inner = gray[inner_margin:-inner_margin, inner_margin:-inner_margin].flatten()
            else:
                inner = gray.flatten()
            
            # 边框和内部的对比度
            border_mean = np.mean(border_pixels)
            inner_mean = np.mean(inner)
            border_contrast = abs(border_mean - inner_mean)
            
            has_clear_border = border_contrast > 25 and edge_ratio > 0.03
            
            # 分类判断
            is_complex = laplacian_var > 800 or std_dev > 55
            
            if is_complex and not has_clear_border:
                classification = 'image_only'  # 真实图片（照片、图表）
            elif has_clear_border and not is_complex:
                classification = 'shape_only'  # 基础图形
            elif has_clear_border and is_complex:
                classification = 'shape_with_content'  # 图形容器+内容
            else:
                classification = 'image_fallback'  # 兜底当图片
            
            return {
                'laplacian_var': laplacian_var,
                'std_dev': std_dev,
                'edge_ratio': edge_ratio,
                'is_complex': is_complex,
                'has_clear_border': has_clear_border,
                'border_contrast': border_contrast,
                'classification': classification
            }
            
        except Exception as e:
            return {'classification': 'unknown', 'is_complex': False, 'error': str(e)}
    
    def _deduplicate_cross_groups(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """
        跨组去重（智能版）
        
        规则：
        1. 优先保留 priority 高的组
        2. 同优先级时，保留 score 高的
        3. 箭头与其他元素重叠时特殊处理
        4. 【新增】基础图形和图片类重叠时，分析图像复杂度决定保留策略
        """
        if not elements:
            return elements
        
        iou_threshold = self.dedup_config.get('iou_threshold', 0.7)
        arrow_iou_threshold = self.dedup_config.get('arrow_iou_threshold', 0.85)
        shape_image_iou_threshold = self.dedup_config.get('shape_image_iou_threshold', 0.6)
        
        sorted_elements = sorted(
            elements,
            key=lambda x: (getattr(x, '_group_priority', 1), x.score),
            reverse=True
        )
        
        keep = []
        dropped = set()
        
        for i, elem_i in enumerate(sorted_elements):
            if i in dropped:
                continue
            
            keep.append(elem_i)
            
            for j in range(i + 1, len(sorted_elements)):
                if j in dropped:
                    continue
                
                elem_j = sorted_elements[j]
                
                group_i = getattr(elem_i, '_source_group', '')
                group_j = getattr(elem_j, '_source_group', '')
                
                effective_threshold = iou_threshold
                if group_i == 'arrow' or group_j == 'arrow':
                    effective_threshold = arrow_iou_threshold
                
                iou = self._calculate_iou(
                    elem_i.bbox.to_list(),
                    elem_j.bbox.to_list()
                )
                
                if iou < 0.1:
                    continue  # 无重叠
                
                # 【新增】基础图形和图片类重叠的智能判断
                if iou > shape_image_iou_threshold:
                    is_shape_image_overlap = (
                        (group_i == 'shape' and group_j == 'image') or
                        (group_i == 'image' and group_j == 'shape')
                    )
                    # Optional: use _analyze_region_complexity for shape vs image
                    # if is_shape_image_overlap:
                    #     analysis = self._analyze_region_complexity(
                    #         self._current_image_path,
                    #         elem_i.bbox.to_list()
                    #     )
                        
                    #     classification = analysis.get('classification', 'unknown')
                        
                    #     if classification == 'image_only':
                    #         # 真实图片：保留图片类，丢弃图形类
                    #         if group_i == 'shape':
                    #             # elem_i是图形，应该丢弃它，保留elem_j（图片）
                    #             keep.remove(elem_i)
                    #             keep.append(elem_j)
                    #             dropped.add(j)
                    #         else:
                    #             # elem_i是图片，保留
                    #             dropped.add(j)
                    #     elif classification == 'shape_only':
                    #         # 基础图形：保留图形类，丢弃图片类
                    #         if group_i == 'image':
                    #             # elem_i是图片，应该丢弃它，保留elem_j（图形）
                    #             keep.remove(elem_i)
                    #             keep.append(elem_j)
                    #             dropped.add(j)
                    #         else:
                    #             # elem_i是图形，保留
                    #             dropped.add(j)
                    #     elif classification == 'shape_with_content':
                    #         # 图形容器+内容：两者都保留（不去重）
                    #         # 标记为层叠关系
                    #         elem_i.processing_notes.append(f"与{elem_j.id}层叠")
                    #         elem_j.processing_notes.append(f"与{elem_i.id}层叠")
                    #         continue
                    #     else:
                    #         # 兜底：当图片处理，保留图片类
                    #         if group_i == 'shape':
                    #             keep.remove(elem_i)
                    #             keep.append(elem_j)
                    #         dropped.add(j)
                    #     continue
                    if is_shape_image_overlap:
                        # Prefer image over shape when overlapping
                        if group_i == 'shape':
                            if elem_i in keep:
                                keep.remove(elem_i)
                            if elem_j not in keep:
                                keep.append(elem_j)
                            dropped.add(j)
                            break  # elem_i 已移除，退出内层循环
                        else:
                            # elem_i 是 image，保留它，丢弃 shape (elem_j)
                            dropped.add(j)
                        continue
                
                # 标准去重逻辑
                if iou > effective_threshold:
                    dropped.add(j)
        
        for i, elem in enumerate(keep):
            elem.id = i
        
        return keep
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU of two boxes."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_contained_elements(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """
        过滤被大图完全包含的小元素
        
        规则：
        1. 如果小元素被图片类大元素包含 > 85%，只保留大元素
        2. 图片类：icon, picture, logo, chart, function_graph
        3. 这样可以避免大图里的小箭头/小图形被单独提取
        """
        IMAGE_TYPES = {'icon', 'picture', 'logo', 'chart', 'function_graph', 'image'}
        
        if not elements:
            return elements
        
        to_remove = set()
        
        for i, elem_i in enumerate(elements):
            if i in to_remove:
                continue
            
            bbox_i = elem_i.bbox.to_list()
            area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
            type_i = elem_i.element_type.lower()
            
            for j, elem_j in enumerate(elements):
                if i == j or j in to_remove:
                    continue
                
                bbox_j = elem_j.bbox.to_list()
                area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                type_j = elem_j.element_type.lower()
                
                if area_i > area_j:
                    containment = self._calculate_containment(bbox_i, bbox_j)
                    if containment > 0.85 and type_i in IMAGE_TYPES:
                        to_remove.add(j)
                        self._log(f"Filter {elem_j.id}({type_j}): contained by {elem_i.id}({type_i}) {containment:.0%}")
                elif area_j > area_i:
                    containment = self._calculate_containment(bbox_j, bbox_i)
                    if containment > 0.85 and type_j in IMAGE_TYPES:
                        to_remove.add(i)
                        self._log(f"Filter {elem_i.id}({type_i}): contained by {elem_j.id}({type_j}) {containment:.0%}")
                        break
        
        result = [e for i, e in enumerate(elements) if i not in to_remove]
        
        # 重新编号
        for i, elem in enumerate(result):
            elem.id = i
        
        if to_remove:
            self._log(f"完全包含过滤: 移除了 {len(to_remove)} 个被大图包含的小元素")
        
        return result
    
    def _calculate_containment(self, box_outer: List[int], box_inner: List[int]) -> float:
        """
        计算 box_inner 被 box_outer 包含的比例
        
        返回值范围 [0, 1]：
        - 1.0 表示完全包含
        - 0.0 表示无重叠
        """
        x1 = max(box_outer[0], box_inner[0])
        y1 = max(box_outer[1], box_inner[1])
        x2 = min(box_outer[2], box_inner[2])
        y2 = min(box_outer[3], box_inner[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        inner_area = (box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1])
        
        return inter_area / inner_area if inner_area > 0 else 0.0
    
    def save_visualization(self, context: ProcessingContext, output_path: str):
        """Save visualization (one color per group)."""
        cv2_image = cv2.imread(context.image_path)
        GROUP_COLORS = {
            'image': (0, 255, 0),
            'arrow': (255, 0, 0),
            'shape': (0, 0, 255),
            'background': (255, 255, 0),
            'custom': (128, 0, 128),
        }
        DEFAULT_COLOR = (128, 128, 128)
        
        image = cv2_image.copy()
        overlay = cv2_image.copy()
        
        for elem in context.elements:
            group = getattr(elem, '_source_group', '')
            color = GROUP_COLORS.get(group, DEFAULT_COLOR)
            
            points = np.array(elem.polygon, dtype=np.int32)
            
            if points.size > 0:
                cv2.fillPoly(overlay, [points], color)
            
            x1, y1, x2, y2 = elem.bbox.to_list()
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{elem.element_type}"
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(output_path, result)
        
        self._log(f"可视化已保存: {output_path}")
    
    def save_metadata(self, context: ProcessingContext, output_path: str):
        """保存元数据JSON"""
        import json
        
        metadata = {
            'image_path': context.image_path,
            'image_size': {
                'width': context.canvas_width,
                'height': context.canvas_height
            },
            'total_elements': len(context.elements),
            'by_group': {},
            'by_type': {},
            'elements': []
        }
        
        for elem in context.elements:
            group = getattr(elem, '_source_group', 'unknown')
            if group not in metadata['by_group']:
                metadata['by_group'][group] = []
            
            elem_type = elem.element_type
            if elem_type not in metadata['by_type']:
                metadata['by_type'][elem_type] = []
            
            elem_data = elem.to_dict()
            elem_data['source_group'] = group
            
            metadata['by_group'][group].append(elem_data)
            metadata['by_type'][elem_type].append(elem_data)
            metadata['elements'].append(elem_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self._log(f"Metadata saved: {output_path}")

    def get_all_prompts(self) -> Dict[str, List[str]]:
        """Return all prompt groups."""
        return {
            group.value: config.prompts.copy()
            for group, config in self.prompt_groups.items()
        }
    
    def get_group_config(self, group_type: PromptGroup) -> Optional[PromptGroupConfig]:
        """获取指定组的配置"""
        return self.prompt_groups.get(group_type)
    
    def add_prompts_to_group(self, group_type: PromptGroup, prompts: List[str]):
        """向指定组添加提示词（运行时）"""
        if group_type in self.prompt_groups:
            for p in prompts:
                self.prompt_groups[group_type].add_prompt(p)
    
    def remove_prompts_from_group(self, group_type: PromptGroup, prompts: List[str]):
        """Remove prompts from a group at runtime."""
        if group_type in self.prompt_groups:
            for p in prompts:
                self.prompt_groups[group_type].remove_prompt(p)
    
    def set_group_threshold(self, group_type: PromptGroup,
                            score_threshold: float = None,
                            min_area: int = None):
        """Set group thresholds at runtime."""
        if group_type in self.prompt_groups:
            if score_threshold is not None:
                self.prompt_groups[group_type].score_threshold = score_threshold
            if min_area is not None:
                self.prompt_groups[group_type].min_area = min_area
    
    def print_prompt_groups(self):
        """打印当前词库配置"""
        print("\n" + "="*60)
        print("当前SAM3提示词词库配置 (从 config.yaml 加载)")
        print("="*60)
        
        for group_type, config in self.prompt_groups.items():
            print(f"\n[{config.name}] ({group_type.value})")
            print(f"  置信度阈值: {config.score_threshold}")
            print(f"  最小面积: {config.min_area}")
            print(f"  优先级: {config.priority}")
            print(f"  提示词 ({len(config.prompts)}个):")
            for p in config.prompts:
                print(f"    - {p}")
        
        print("\n" + "="*60)
        print(f"配置文件路径: {ConfigLoader.get_config_path()}")
        print("="*60)


# ======================== 快捷函数 ========================
def extract_elements(image_path: str, 
                     groups: List[PromptGroup] = None) -> ProcessingResult:
    """
    快捷函数 - 一行代码提取元素
    
    Args:
        image_path: 图片路径
        groups: 要处理的组列表（默认全部）
        
    Returns:
        ProcessingResult
        
    使用示例:
        # 提取所有元素
        result = extract_elements("test.png")
        
        # 只提取图片和箭头
        result = extract_elements("test.png", groups=[PromptGroup.IMAGE, PromptGroup.ARROW])
    """
    extractor = Sam3InfoExtractor()
    context = ProcessingContext(image_path=image_path)
    
    if groups:
        all_elements = []
        for group in groups:
            result = extractor.extract_by_group(context, group)
            all_elements.extend(result.elements)
        
        for i, elem in enumerate(all_elements):
            elem.id = i
        
        return ProcessingResult(
            success=True,
            elements=all_elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height
        )
    
    return extractor.process(context)

def extract_with_prompts(image_path: str, 
                         prompts: List[str],
                         score_threshold: float = 0.5) -> ProcessingResult:
    """Convenience: extract with custom prompts only."""
    extractor = Sam3InfoExtractor()
    context = ProcessingContext(image_path=image_path)
    
    return extractor.extract_with_custom_prompts(
        context, 
        prompts,
        score_threshold=score_threshold
    )

