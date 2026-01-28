"""
任务2：Icon/Picture非基本图形处理模块

功能：
    - 处理icon、picture、logo、chart、function_graph等非基本图形
    - 使用RMBG-2.0进行背景移除
    - 将处理后的图片转换为base64
    - 生成XML片段

负责人：[已实现]
负责任务：任务2 - Icon、picture、函数图等非基本图形类

使用示例：
    from modules import IconPictureProcessor, ProcessingContext
    
    processor = IconPictureProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # 从SAM3获取的元素
    
    result = processor.process(context)
    # 处理后的元素会包含 base64 和 xml_fragment 字段

接口说明：
    输入：
        - context.elements: ElementInfo列表，筛选出需要处理的非基本图形
        - context.image_path: 原始图片路径，用于裁剪
        
    输出：
        - 更新 element.base64: 处理后的base64编码图片
        - 更新 element.has_transparency: 是否已去除背景
        - 更新 element.xml_fragment: 该元素的XML片段
"""

import os
import io
import base64
from typing import Optional, List
from PIL import Image
import numpy as np
import cv2
from prompts.image import IMAGE_PROMPT
# ONNX Runtime（可选依赖）
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[IconPictureProcessor] Warning: onnxruntime not available, RMBG disabled")

# 超分模型（可选依赖）
try:
    import torch
    from spandrel import ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False
    print("[IconPictureProcessor] Warning: spandrel/torch not available, Upscale disabled")

from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, ProcessingResult, LayerLevel


# ======================== RMBG-2.0 模型封装 ========================
class RMBGModel(ModelWrapper):
    """
    RMBG-2.0 背景移除模型封装
    
    基于 ONNX Runtime 实现，支持 CUDA 加速
    
    使用示例：
        model = RMBGModel(model_path)
        model.load()
        
        # 一行调用去除背景
        rgba_image = model.remove_background(pil_image)
    """
    
    # 模型固定输入尺寸
    INPUT_SIZE = (1024, 1024)
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path or self._get_default_path()
        self._session = None
        self._input_name = None
        self._output_name = None
    
    def _get_default_path(self) -> str:
        """获取默认模型路径"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "rmbg", "model.onnx"
        )
    
    def load(self):
        """
        加载RMBG-2.0 ONNX模型
        支持自动降级：CUDA失败时自动回退到CPU
        """
        if self._is_loaded:
            return
        
        if not ONNX_AVAILABLE:
            print("[RMBGModel] Warning: onnxruntime not available, using fallback mode")
            self._is_loaded = True
            return
        
        if not os.path.exists(self.model_path):
            print(f"[RMBGModel] Warning: Model file not found at {self.model_path}, using fallback mode")
            self._is_loaded = True
            return
        
        # 配置 ONNX Runtime 选项，屏蔽警告
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # ERROR only
        session_options.enable_profiling = False
        
        # 获取可用的 providers
        available_providers = ort.get_available_providers()
        
        # 尝试加载顺序：先 CUDA，失败则 CPU
        providers_to_try = [
            (['CUDAExecutionProvider', 'CPUExecutionProvider'], "CUDA+CPU"),
            (['CPUExecutionProvider'], "CPU only"),
        ]
        
        for providers, name in providers_to_try:
            # 过滤出可用的 providers
            valid_providers = [p for p in providers if p in available_providers]
            if not valid_providers:
                continue
            
            try:
                print(f"[RMBGModel] Trying to load with {name} ({valid_providers})...")
                self._session = ort.InferenceSession(
                    self.model_path,
                    providers=valid_providers,
                    sess_options=session_options
                )
                
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._providers = valid_providers
                
                self._is_loaded = True
                print(f"[RMBGModel] Model loaded successfully with {name}")
                return
                
            except Exception as e:
                print(f"[RMBGModel] Failed to load with {name}: {e}")
                # 继续尝试下一个配置
                continue
        
        # 所有尝试都失败，使用 fallback 模式
        print("[RMBGModel] Warning: All loading attempts failed, using fallback mode (no background removal)")
        self._is_loaded = True
    
    def _preprocess(self, img: np.ndarray) -> tuple:
        """
        图片预处理：缩放、归一化、转CHW格式
        
        Args:
            img: RGB格式的numpy数组
            
        Returns:
            (preprocessed_image, original_size)
        """
        # RMBG-2.0 要求 BGR 格式
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        
        # 缩放到模型输入尺寸
        img_resized = cv2.resize(img_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 转 CHW 格式 (HWC -> CHW)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # 增加 batch 维度 (3, 1024, 1024) -> (1, 3, 1024, 1024)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, (h, w)
    
    def _postprocess(self, pred: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        后处理：提取 alpha 通道并还原到原图尺寸
        
        Args:
            pred: 模型输出
            original_size: (height, width)
            
        Returns:
            alpha通道 (uint8, 0-255)
        """
        # 移除 batch 维度，提取 alpha 通道 (1, 1, 1024, 1024) -> (1024, 1024)
        alpha = pred[0, 0, :, :]
        
        # 缩放回原图尺寸
        alpha_resized = cv2.resize(alpha, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 255] 并转 8 位
        alpha_resized = (alpha_resized * 255).astype(np.uint8)
        
        return alpha_resized
    
    def predict(self, image: Image.Image) -> Image.Image:
        """
        背景移除推理
        支持自动降级：GPU推理失败时自动回退到CPU
        
        Args:
            image: PIL图像（RGB格式）
            
        Returns:
            去除背景后的RGBA图像
        """
        if not self._is_loaded:
            self.load()
        
        # 如果模型未成功加载，返回 fallback 结果
        if self._session is None:
            return image.convert("RGBA")
        
        # 转换为 numpy 数组
        img = np.array(image)
        
        # 预处理
        img_input, original_size = self._preprocess(img)
        
        # 尝试推理（带错误处理）
        try:
            pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
        except Exception as e:
            # GPU 推理失败，尝试重新加载为 CPU 模式
            if hasattr(self, '_providers') and 'CUDAExecutionProvider' in self._providers:
                print(f"[RMBGModel] GPU inference failed (OOM), switching to CPU...")
                
                try:
                    # 释放当前 session
                    self._session = None
                    
                    # 重新创建 CPU-only session
                    session_options = ort.SessionOptions()
                    session_options.log_severity_level = 3
                    
                    self._session = ort.InferenceSession(
                        self.model_path,
                        providers=['CPUExecutionProvider'],
                        sess_options=session_options
                    )
                    self._providers = ['CPUExecutionProvider']
                    
                    # 重试推理
                    pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
                    print("[RMBGModel] CPU inference successful")
                    
                except Exception as e2:
                    print(f"[RMBGModel] CPU inference also failed: {e2}")
                    print("[RMBGModel] Falling back to no background removal")
                    return image.convert("RGBA")
            else:
                print(f"[RMBGModel] Inference failed: {e}, using fallback (no background removal)")
                return image.convert("RGBA")
        
        # 后处理得到 alpha 通道
        alpha = self._postprocess(pred, original_size)
        
        # 合并 alpha 通道到原图（生成 RGBA 图片）
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = alpha
        
        # 转换为 PIL 图片
        return Image.fromarray(img_rgba)
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """predict的别名，更语义化"""
        return self.predict(image)
    
    def unload(self):
        """释放模型资源"""
        self._session = None
        self._is_loaded = False

# ======================== 超分模型 Realesrgan 封装 ========================
class UpscaleModel(ModelWrapper):
    """
    超分模型封装（基于 spandrel 模型加载）
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        fp16: bool = False,
    ):
        super().__init__()
        self.model_path = model_path or self._get_default_path()
        self.device = device
        self.fp16 = fp16
        self.scale = 1
        self._model = None
        self._device: Optional["torch.device"] = None
        self._use_half = False

    def _resolve_device(self) -> "torch.device":
        if not SPANDREL_AVAILABLE:
            raise ImportError("spandrel/torch 未安装")
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        # allow explicit strings like "cpu" or "cuda:1"
        return torch.device(self.device)
    
    def  _get_default_path(self) -> str:
        """获取默认超分模型路径"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "RealESRGAN_x4plus_anime_6B.pth"
        )

    def load(self):
        """加载超分模型，失败时静默降级"""
        if self._is_loaded:
            return

        if not SPANDREL_AVAILABLE:
            print("[UpscaleModel] spandrel/torch 未安装，跳过超分")
            self._is_loaded = True
            return

        if not os.path.exists(self.model_path):
            print(f"[UpscaleModel] 模型文件不存在: {self.model_path}，跳过超分")
            self._is_loaded = True
            return

        try:
            self._device = self._resolve_device()
            descriptor = ModelLoader().load_from_file(str(self.model_path))
            self.scale = descriptor.scale
            self._model = descriptor.model
            self._model.eval().to(self._device)
            self._use_half = bool(self.fp16 and descriptor.supports_half and self._device.type == "cuda")
            if self._use_half:
                self._model.half()
            self._is_loaded = True
            print("[UpscaleModel] 模型加载成功")
        except Exception as e:
            print(f"[UpscaleModel] 加载或初始化失败，跳过超分: {e}")
            self._model = None
            self._device = None
            self._use_half = False
            self._is_loaded = True

    def upscale(self, image: Image.Image) -> Image.Image:
        if not self._is_loaded:
            self.load()
        if not SPANDREL_AVAILABLE:
            return image
        if self._model is None or self._device is None:
            return image

        rgb = image.convert("RGB")
        img = np.array(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        tensor = tensor.half() if self._use_half else tensor.float()

        with torch.no_grad():
            out = self._model(tensor)

        out = out.float().clamp(0, 1)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def unload(self):
        """释放模型资源"""
        self._model = None
        self._device = None
        self._use_half = False
        self._is_loaded = False

# ======================== Icon/Picture处理器 ========================
class IconPictureProcessor(BaseProcessor):
    """
    Icon/Picture处理模块
    
    处理流程：
        1. 从context.elements中筛选需要处理的元素
        2. 根据元素bbox从原图裁剪
        3. 使用RMBG去除背景（icon类）或保留背景（picture类）
        4. 转换为base64编码
        5. 生成XML片段
        6. 更新元素信息
    """
    
    # # 需要去背景的类型（抠图）
    RMBG_TYPES = {"icon", "logo", "symbol", "emoji", "button"}
    

    # 保留背景的类型（直接裁剪）
    KEEP_BG_TYPES = {
        "picture", "photo", "chart", "function_graph", "screenshot", "image", "diagram",
        "graph", "line graph", "bar graph", "heatmap", "scatter plot", "histogram", "pie chart"
    }
    
    # 元素面积占图片面积的最大比例阈值（超过此比例的元素会被跳过）
    MAX_AREA_RATIO = 0.75

    
    def __init__(
        self,
        config=None,
        rmbg_model_path: str = None,
        upscale_enabled: bool = True,
        upscale_model_path: Optional[str] = None,
        upscale_device: str = "auto",
    ):
        super().__init__(config)
        self._rmbg_model: Optional[RMBGModel] = None
        self._rmbg_model_path = rmbg_model_path
        self._upscale_enabled = upscale_enabled
        self._upscale_model_path = upscale_model_path
        self._upscale_device = upscale_device
        self._upscale_model: Optional[UpscaleModel] = None

    def load_rmbg_model(self):
        """加载RMBG模型"""
        if self._rmbg_model is None:
            self._rmbg_model = RMBGModel(self._rmbg_model_path)
        if not self._rmbg_model.is_loaded:
            self._rmbg_model.load()

    def load_model(self):
        """兼容旧接口：加载RMBG模型"""
        self.load_rmbg_model()

    def load_upscale_model(self):
        """加载超分模型（可选依赖，失败时自动跳过）"""
        if not self._upscale_enabled:
            return
        if self._upscale_model is not None:
            return
        if not SPANDREL_AVAILABLE:
            self._log("超分模型依赖未安装，已跳过")
            return

        model_path = self._upscale_model_path
        try:
            self._upscale_model = UpscaleModel(
                model_path=model_path,
                device=self._upscale_device,
            )
            self._upscale_model.load()
        except Exception as e:
            self._upscale_model = None
            self._log(f"超分模型不可用，已跳过: {e}")
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult
        """
        self._log("开始处理Icon/Picture元素")
        
        # 检查原图是否已经超分过（来自Pipeline预处理）
        was_upscaled = context.intermediate_results.get('was_upscaled', False)
        if was_upscaled:
            self._log("原图已超分，跳过元素级超分")
            self._skip_element_upscale = True
        else:
            self._skip_element_upscale = False
        
        # 加载模型
        self.load_rmbg_model()
        if not self._skip_element_upscale:
            self.load_upscale_model()
        
        # 加载原图
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        original_image = Image.open(context.image_path).convert("RGB")
        cv2_image = cv2.imread(context.image_path)
        
        # 筛选需要处理的元素
        elements_to_process = self._get_elements_to_process(context.elements)
        
        self._log(f"需要处理的元素数量: {len(elements_to_process)}")
        
        processed_count = 0
        rmbg_count = 0
        keep_bg_count = 0
        
        for elem in elements_to_process:
            try:
                is_rmbg = self._process_element(elem, original_image)
                processed_count += 1
                if is_rmbg:
                    rmbg_count += 1
                else:
                    keep_bg_count += 1
            except Exception as e:
                elem.processing_notes.append(f"处理失败: {str(e)}")
                self._log(f"元素{elem.id}处理失败: {e}")
        
        self._log(f"处理完成: {processed_count}/{len(elements_to_process)}个元素 (抠图:{rmbg_count}, 保留背景:{keep_bg_count})")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'processed_count': processed_count,
                'total_to_process': len(elements_to_process),
                'rmbg_count': rmbg_count,
                'keep_bg_count': keep_bg_count
            }
        )
    
    def _get_elements_to_process(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """筛选需要处理的元素"""
        all_types = IMAGE_PROMPT
        print(all_types)
        return [
            e for e in elements
            if e.element_type.lower() in all_types and e.base64 is None
        ]
    
    def _process_element(self, elem: ElementInfo, original_image: Image.Image) -> bool:
        """
        处理单个元素
        
        Args:
            elem: 元素信息
            original_image: PIL原图
            
        Returns:
            bool: 是否使用了RMBG抠图
        """
        elem_type = elem.element_type.lower()
        
        # 裁剪区域（向内收缩以避免截取边框外的内容）
        # shrink_margin > 0 表示向内收缩，< 0 表示向外扩展
        shrink_margin = 0
        img_w, img_h = original_image.size
        
        # 计算收缩后的边界，确保不会收缩过多导致区域太小
        orig_w = elem.bbox.x2 - elem.bbox.x1
        orig_h = elem.bbox.y2 - elem.bbox.y1
        # 最多收缩到原尺寸的10%
        max_shrink = min(orig_w * 0.1, orig_h * 0.1, shrink_margin)
        actual_shrink = int(max_shrink)
        
        x1 = max(0, elem.bbox.x1 + actual_shrink)
        y1 = max(0, elem.bbox.y1 + actual_shrink)
        x2 = min(img_w, elem.bbox.x2 - actual_shrink)
        y2 = min(img_h, elem.bbox.y2 - actual_shrink)
        
        # 确保裁剪区域有效
        if x2 <= x1 or y2 <= y1:
            # 收缩过多，回退到原始边界
            x1, y1 = elem.bbox.x1, elem.bbox.y1
            x2, y2 = elem.bbox.x2, elem.bbox.y2
        
        cropped = original_image.crop((x1, y1, x2, y2))

        # 元素级超分（仅在原图未超分时执行）
        if not getattr(self, '_skip_element_upscale', False) and self._upscale_model is not None:
            try:
                cropped = self._upscale_model.upscale(cropped)
                elem.processing_notes.append("元素级超分完成")
            except Exception as e:
                elem.processing_notes.append(f"超分失败，已跳过: {str(e)}")
                self._log(f"元素{elem.id}超分失败: {e}")
        
        is_rmbg = False
        
        # 根据类型决定是否去背景
        if elem_type in self.RMBG_TYPES:
            # 去除背景
            processed = self._rmbg_model.remove_background(cropped)
            elem.has_transparency = True
            is_rmbg = True
        else:
            # 保留背景
            processed = cropped.convert("RGBA")
            elem.has_transparency = False
        
        # 转base64
        elem.base64 = self._image_to_base64(processed)
        
        # 更新bbox（因为加了padding）
        elem.bbox.x1 = x1
        elem.bbox.y1 = y1
        elem.bbox.x2 = x2
        elem.bbox.y2 = y2
        
        # 生成XML片段
        self._generate_xml(elem)
        
        elem.processing_notes.append(f"IconPictureProcessor处理完成 (RMBG={is_rmbg})")
        
        return is_rmbg
    
    def _generate_xml(self, elem: ElementInfo):
        """
        生成图片元素的XML片段
        """
        x1 = elem.bbox.x1
        y1 = elem.bbox.y1
        width = elem.bbox.x2 - elem.bbox.x1
        height = elem.bbox.y2 - elem.bbox.y1
        
        # DrawIO 图片样式
        style = (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;"
            f"image=data:image/png,{elem.base64};"
        )
        
        # DrawIO的id必须从2开始（0和1是保留的根元素）
        cell_id = elem.id + 2
        
        elem.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
        
        # 设置层级
        elem.layer_level = LayerLevel.IMAGE.value
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ======================== 图像复杂度分析 ========================
def calculate_image_complexity(image_arr: np.ndarray) -> tuple:
    """
    计算图像丰富度/复杂度
    用于自动判断元素是否应该作为图片处理
    
    Args:
        image_arr: BGR格式的numpy数组
        
    Returns:
        (laplacian_variance, std_deviation)
        - laplacian_variance: 拉普拉斯方差（纹理/边缘丰富度）
        - std_deviation: 标准差（对比度/颜色变化）
    """
    if image_arr.size == 0:
        return 0.0, 0.0
    
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    
    # 拉普拉斯方差 (纹理/边缘丰富度)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 标准差 (对比度/颜色变化)
    std_dev = np.std(gray)
    
    return laplacian_var, std_dev


def is_complex_image(image_arr: np.ndarray, laplacian_threshold: float = 800, std_threshold: float = 50) -> bool:
    """
    判断图像是否为复杂图像（应作为picture处理）
    
    Args:
        image_arr: BGR格式的numpy数组
        laplacian_threshold: 拉普拉斯方差阈值
        std_threshold: 标准差阈值
        
    Returns:
        bool: 是否为复杂图像
    """
    l_var, s_dev = calculate_image_complexity(image_arr)
    return l_var > laplacian_threshold or s_dev > std_threshold


# ======================== 快捷函数 ========================
def process_icons_pictures(elements: List[ElementInfo], 
                           image_path: str) -> List[ElementInfo]:
    """
    快捷函数 - 处理所有icon和picture元素
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        
    Returns:
        处理后的元素列表
        
    使用示例:
        elements = process_icons_pictures(elements, "test.png")
    """
    processor = IconPictureProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=elements
    )
    
    result = processor.process(context)
    return result.elements
