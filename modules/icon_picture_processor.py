"""
Icon/Picture processor for non-basic shapes (icons, pictures, logos, charts, etc.).

- Uses RMBG-2.0 for background removal on icon-like types
- Converts crops to base64 and generates XML fragments

Usage:
    from modules import IconPictureProcessor, ProcessingContext
    processor = IconPictureProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # from SAM3
    result = processor.process(context)
    # Elements get base64 and xml_fragment
"""

import os
import io
import base64
from typing import Optional, List
from PIL import Image
import numpy as np
import cv2
from prompts.image import IMAGE_PROMPT
# ONNX Runtime (optional, for RMBG)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[IconPictureProcessor] Warning: onnxruntime not available, RMBG disabled")

from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, ProcessingResult, LayerLevel


# ======================== RMBG-2.0 model wrapper ========================
class RMBGModel(ModelWrapper):
    """
    RMBG-2.0 background-removal model (ONNX Runtime, CUDA if available).

    Example:
        model = RMBGModel(model_path)
        model.load()
        rgba_image = model.remove_background(pil_image)
    """

    INPUT_SIZE = (1024, 1024)

    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path or self._get_default_path()
        self._session = None
        self._input_name = None
        self._output_name = None

    def _get_default_path(self) -> str:
        """Default model path under models/rmbg/."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "rmbg", "model.onnx"
        )
    
    def load(self):
        """Load RMBG-2.0 ONNX model; fallback to CPU if CUDA fails."""
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
        
        # ONNX Runtime options
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # ERROR only
        session_options.enable_profiling = False
        
        # Available providers
        available_providers = ort.get_available_providers()
        
        # Try CUDA then CPU
        providers_to_try = [
            (['CUDAExecutionProvider', 'CPUExecutionProvider'], "CUDA+CPU"),
            (['CPUExecutionProvider'], "CPU only"),
        ]
        
        for providers, name in providers_to_try:
            # Filter valid providers
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
                # Try next config
                continue
        
        # All attempts failed, use fallback
        print("[RMBGModel] Warning: All loading attempts failed, using fallback mode (no background removal)")
        self._is_loaded = True
    
    def _preprocess(self, img: np.ndarray) -> tuple:
        """Preprocess: scale, normalize, HWC->CHW. img: RGB numpy.
            
        Returns:
            (preprocessed_image, original_size)
        """
        # RMBG-2.0 expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        
        # Scale to model input size
        img_resized = cv2.resize(img_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dim
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, (h, w)
    
    def _postprocess(self, pred: np.ndarray, original_size: tuple) -> np.ndarray:
        """Extract alpha and resize to original. Returns alpha uint8."""
        # Remove batch, get alpha
        alpha = pred[0, 0, :, :]
        
        # Resize to original
        alpha_resized = cv2.resize(alpha, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # To uint8
        alpha_resized = (alpha_resized * 255).astype(np.uint8)
        
        return alpha_resized
    
    def predict(self, image: Image.Image) -> Image.Image:
        """Background removal; fallback to CPU if GPU fails. Returns RGBA PIL."""
        if not self._is_loaded:
            self.load()
        
        # Model not loaded: return fallback
        if self._session is None:
            return image.convert("RGBA")
        
        # To numpy
        img = np.array(image)
        
        # Preprocess
        img_input, original_size = self._preprocess(img)
        
        try:
            pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
        except Exception as e:
            # GPU failed, try CPU
            if hasattr(self, '_providers') and 'CUDAExecutionProvider' in self._providers:
                print(f"[RMBGModel] GPU inference failed (OOM), switching to CPU...")
                
                try:
                    # Release session
                    self._session = None
                    
                    # New CPU session
                    session_options = ort.SessionOptions()
                    session_options.log_severity_level = 3
                    
                    self._session = ort.InferenceSession(
                        self.model_path,
                        providers=['CPUExecutionProvider'],
                        sess_options=session_options
                    )
                    self._providers = ['CPUExecutionProvider']
                    
                    # Retry
                    pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
                    print("[RMBGModel] CPU inference successful")
                    
                except Exception as e2:
                    print(f"[RMBGModel] CPU inference also failed: {e2}")
                    print("[RMBGModel] Falling back to no background removal")
                    return image.convert("RGBA")
            else:
                print(f"[RMBGModel] Inference failed: {e}, using fallback (no background removal)")
                return image.convert("RGBA")
        
        # Postprocess alpha
        alpha = self._postprocess(pred, original_size)
        
        # Merge alpha -> RGBA
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = alpha
        
        # To PIL
        return Image.fromarray(img_rgba)
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Alias for predict."""
        return self.predict(image)
    
    def unload(self):
        """Release model resources."""
        self._session = None
        self._is_loaded = False

# ======================== Icon/Picture processor ========================
class IconPictureProcessor(BaseProcessor):
    """Process icon/picture elements: filter, crop, optional RMBG, base64, XML fragments."""

    # Types that use RMBG for background removal; others keep original crop
    RMBG_TYPES = {"icon", "logo", "symbol", "emoji", "button", "arrow"}
    

    # Types that keep background (crop only)
    KEEP_BG_TYPES = {
        "picture", "photo", "chart", "function_graph", "screenshot", "image", "diagram",
        "graph", "line graph", "bar graph", "heatmap", "scatter plot", "histogram", "pie chart"
    }
    
    # Max element area ratio (skip if element area > this fraction of image)
    MAX_AREA_RATIO = 0.75

    
    def __init__(
        self,
        config=None,
        rmbg_model_path: str = None,
    ):
        super().__init__(config)
        self._rmbg_model: Optional[RMBGModel] = None
        self._rmbg_model_path = rmbg_model_path

    def load_rmbg_model(self):
        """Load RMBG model."""
        if self._rmbg_model is None:
            self._rmbg_model = RMBGModel(self._rmbg_model_path)
        if not self._rmbg_model.is_loaded:
            self._rmbg_model.load()

    def load_model(self):
        """Alias: load RMBG model."""
        self.load_rmbg_model()

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process icon/picture elements in context."""
        self._log("Processing Icon/Picture elements")
        self.load_rmbg_model()

        # Load image
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="Invalid image path"
            )
        
        original_image = Image.open(context.image_path).convert("RGB")
        cv2_image = cv2.imread(context.image_path)
        
        # Filter elements to process
        elements_to_process = self._get_elements_to_process(context.elements)
        
        self._log(f"Elements to process: {len(elements_to_process)}")
        
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
                elem.processing_notes.append(f"Failed: {str(e)}")
                self._log(f"Element {elem.id} failed: {e}")
        
        self._log(f"Done: {processed_count}/{len(elements_to_process)} (RMBG:{rmbg_count}, keep_bg:{keep_bg_count})")
        
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
        """Filter elements to process (icons, arrows, etc.; arrows treated as icon crop).

        NOTE: IMAGE_PROMPT contains mixed-case strings (e.g. "3D heart model",
        "MRI image", "CT scan image"). Comparing `.lower()` against the raw
        set caused those detections to be silently skipped — no base64 was
        generated and downstream SVG rendering fell back to a plain polygon
        outline. Always normalize both sides.
        """
        all_types = {t.lower() for t in IMAGE_PROMPT} | {"arrow", "line", "connector"}
        return [
            e for e in elements
            if e.element_type.lower() in all_types and e.base64 is None
        ]
    
    def _process_element(self, elem: ElementInfo, original_image: Image.Image) -> bool:
        """Process one element. Returns True if RMBG was used."""
        elem_type = elem.element_type.lower()
        
        # Crop (shrink_margin: positive = shrink in)
        shrink_margin = 0
        img_w, img_h = original_image.size
        
        # Shrink bounds
        orig_w = elem.bbox.x2 - elem.bbox.x1
        orig_h = elem.bbox.y2 - elem.bbox.y1
        # Cap shrink to 10% of size
        max_shrink = min(orig_w * 0.1, orig_h * 0.1, shrink_margin)
        actual_shrink = int(max_shrink)
        
        x1 = max(0, elem.bbox.x1 + actual_shrink)
        y1 = max(0, elem.bbox.y1 + actual_shrink)
        x2 = min(img_w, elem.bbox.x2 - actual_shrink)
        y2 = min(img_h, elem.bbox.y2 - actual_shrink)
        
        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            # Fallback to original bounds
            x1, y1 = elem.bbox.x1, elem.bbox.y1
            x2, y2 = elem.bbox.x2, elem.bbox.y2
        
        cropped = original_image.crop((x1, y1, x2, y2))

        is_rmbg = False
        
        # RMBG or keep background by type
        if elem_type in self.RMBG_TYPES:
            processed = self._rmbg_model.remove_background(cropped)
            elem.has_transparency = True
            is_rmbg = True
        else:
            processed = cropped.convert("RGBA")
            elem.has_transparency = False
        
        # To base64
        elem.base64 = self._image_to_base64(processed)
        
        # Update bbox for padding
        elem.bbox.x1 = x1
        elem.bbox.y1 = y1
        elem.bbox.x2 = x2
        elem.bbox.y2 = y2
        
        # XML fragment
        self._generate_xml(elem)
        
        elem.processing_notes.append(f"IconPictureProcessor done (RMBG={is_rmbg})")
        
        return is_rmbg
    
    def _generate_xml(self, elem: ElementInfo):
        """
        Generate XML fragment for image element.
        """
        x1 = elem.bbox.x1
        y1 = elem.bbox.y1
        width = elem.bbox.x2 - elem.bbox.x1
        height = elem.bbox.y2 - elem.bbox.y1
        
        # DrawIO image style
        style = (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;"
            f"image=data:image/png,{elem.base64};"
        )
        
        # DrawIO ids start at 2 (0,1 reserved)
        cell_id = elem.id + 2
        
        elem.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
        
        # Layer
        elem.layer_level = LayerLevel.IMAGE.value
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ======================== Image complexity ========================
def calculate_image_complexity(image_arr: np.ndarray) -> tuple:
    """Compute image complexity (for picture vs icon). Returns (laplacian_variance, std_deviation)."""
    if image_arr.size == 0:
        return 0.0, 0.0
    
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance (texture/edge)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Std dev (contrast/color variation)
    std_dev = np.std(gray)
    
    return laplacian_var, std_dev


def is_complex_image(image_arr: np.ndarray, laplacian_threshold: float = 800, std_threshold: float = 50) -> bool:
    """Whether image is complex enough to treat as picture."""
    l_var, s_dev = calculate_image_complexity(image_arr)
    return l_var > laplacian_threshold or s_dev > std_threshold


# ======================== Convenience ========================
def process_icons_pictures(elements: List[ElementInfo], 
                           image_path: str) -> List[ElementInfo]:
    """Process all icon/picture elements. Example: process_icons_pictures(elements, 'test.png')."""
    processor = IconPictureProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=elements
    )
    
    result = processor.process(context)
    return result.elements
