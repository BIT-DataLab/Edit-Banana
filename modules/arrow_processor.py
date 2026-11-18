"""
Arrow processing module.

Converts arrow/connector elements from SAM3 masks into DrawIO edges.
Vector path from skeleton, or fallback to image when needed.
"""

import io
import base64
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image

try:
    from skimage.morphology import skeletonize
    SKELETONIZE_AVAILABLE = True
except ImportError:
    SKELETONIZE_AVAILABLE = False
    print("[ArrowProcessor] Warning: skimage not available, skeletonize disabled")

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult, LayerLevel
from .utils import ArrowAttributeDetector, build_arrow_style


class ArrowProcessor(BaseProcessor):
    """Arrow processor: vector path (skeleton) or image fallback for DrawIO edges."""
    PADDING = 15

    def __init__(self, config=None):
        super().__init__(config)
        self._arrow_detector = ArrowAttributeDetector()

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process arrow/connector elements."""
        self._log("Processing arrows")
        arrows = [e for e in context.elements
                  if e.element_type.lower() in {'arrow', 'line', 'connector'}]
        if not arrows:
            self._log("No arrow elements found")
            return ProcessingResult(
                success=True,
                elements=context.elements,
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                metadata={'arrows_processed': 0, 'total_arrows': 0}
            )

        pil_image = Image.open(context.image_path).convert("RGB")
        full_image_np = np.array(pil_image)
        img_h, img_w = full_image_np.shape[:2]

        processed_count = 0
        vector_count = 0
        image_count = 0
        for arrow in arrows:
            try:
                if self._process_arrow(arrow, full_image_np, img_w, img_h):
                    processed_count += 1
                if arrow.vector_points:
                    vector_count += 1
                elif arrow.base64:
                    image_count += 1
            except Exception as e:
                arrow.processing_notes.append(f"Error: {str(e)}")
                self._log(f"Arrow {arrow.id} failed: {e}")
        self._log(f"Done: {processed_count}/{len(arrows)} arrows (vector:{vector_count}, image:{image_count})")
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'arrows_processed': processed_count,
                'total_arrows': len(arrows),
                'vector_arrows': vector_count,
                'image_arrows': image_count
            }
        )

    def _process_arrow(self, arrow: ElementInfo, full_image_np: np.ndarray,
                      img_w: int, img_h: int) -> bool:
        """Process one arrow: path from mask or fallback to image."""
        x1, y1, x2, y2 = arrow.bbox.to_list()
        tip = ((x1 + x2) // 2, (y1 + y2) // 2)

        arrow_color = self._extract_arrow_color(arrow, full_image_np)
        if arrow_color is None:
            return self._fallback_to_image(arrow, full_image_np, img_w, img_h)

        vector_points = None
        if arrow.mask is not None:
            vector_points = self._extract_path_from_mask(arrow, full_image_np, tip)

        if vector_points and len(vector_points) >= 2:
            arrow.vector_points = vector_points
            arrow.arrow_start = tuple(vector_points[0])
            arrow.arrow_end = tuple(vector_points[-1])
            p_x1, p_y1 = max(0, x1 - 5), max(0, y1 - 5)
            p_x2, p_y2 = min(img_w, x2 + 5), min(img_h, y2 + 5)
            small_crop = full_image_np[p_y1:p_y2, p_x1:p_x2]
            local_path = [[pt[0] - p_x1, pt[1] - p_y1] for pt in vector_points]
            arrow_attrs = self._arrow_detector.detect_all_attributes(
                small_crop, path_points=local_path, mask=None
            )
            arrow_attrs['start_arrow'] = 'none'
            arrow_attrs['start_fill'] = False
            arrow_attrs['curve_type'] = self._detect_curve_type(vector_points)
            self._generate_vector_xml(arrow, arrow_attrs)
            arrow.processing_notes.append(f"Vector path: {len(vector_points)} points")
            return True
        return self._fallback_to_image(arrow, full_image_np, img_w, img_h)

    def _extract_arrow_color(self, arrow: ElementInfo,
                             full_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract stroke color from mask or bbox region."""
        x1, y1, x2, y2 = arrow.bbox.to_list()
        h_img, w_img = full_image.shape[:2]
        pad = 5
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        roi = full_image[y1_p:y2_p, x1_p:x2_p]
        if arrow.mask is not None:
            try:
                mask = arrow.mask
                if mask.shape[0] >= y2_p and mask.shape[1] >= x2_p:
                    mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
                    mask_binary = mask_roi > 127
                    if np.count_nonzero(mask_binary) > 0:
                        return np.median(roi[mask_binary], axis=0).astype(np.uint8)
            except Exception:
                pass
        if roi.size > 0:
            pixels = roi.reshape(-1, 3)
            luminance = np.dot(pixels, [0.299, 0.587, 0.114])
            return pixels[np.argmin(luminance)].astype(np.uint8)
        return None

    def _extract_path_from_mask(self, arrow: ElementInfo,
                                full_image: np.ndarray,
                                tip: Tuple[int, int]) -> Optional[List[List[int]]]:
        """Extract path from mask: skeleton -> extreme points -> ordered path -> simplify."""
        if arrow.mask is None:
            return None
        x1, y1, x2, y2 = arrow.bbox.to_list()
        h_img, w_img = full_image.shape[:2]
        pad = 10
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        mask = arrow.mask
        if mask.shape[0] < y2_p or mask.shape[1] < x2_p:
            return None
        mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
        mask_binary = (mask_roi > 127).astype(np.uint8)
        if np.count_nonzero(mask_binary) < 10:
            return None

        if not SKELETONIZE_AVAILABLE:
            return self._fallback_extract_from_mask(arrow, full_image, tip)

        skeleton = skeletonize(mask_binary > 0)
        skel_points = np.argwhere(skeleton)
        if len(skel_points) < 2:
            return None

        extreme_endpoints = self._find_extreme_points(skel_points)
        if len(extreme_endpoints) < 2:
            return None
        global_endpoints = [
            [x1_p + ep[1], y1_p + ep[0]] for ep in extreme_endpoints
        ]
        ordered_path = self._extract_ordered_skeleton_path(
            skeleton, extreme_endpoints[0], extreme_endpoints[1]
        )
        if ordered_path and len(ordered_path) >= 2:
            global_path = [[x1_p + p[1], y1_p + p[0]] for p in ordered_path]
        else:
            global_path = global_endpoints
        global_path = self._orient_to_tip_simple(global_path, tip)

        if len(global_path) > 3:
            global_path = self._douglas_peucker_simplify(global_path, epsilon=3.0)
        return global_path if len(global_path) >= 2 else None

    def _find_extreme_points(self, skel_points: np.ndarray) -> List[Tuple[int, int]]:
        """Two endpoints by axis span (min/max row or column)."""
        if len(skel_points) < 2:
            return []
        min_y_idx = np.argmin(skel_points[:, 0])
        max_y_idx = np.argmax(skel_points[:, 0])
        min_x_idx = np.argmin(skel_points[:, 1])
        max_x_idx = np.argmax(skel_points[:, 1])
        y_span = skel_points[max_y_idx, 0] - skel_points[min_y_idx, 0]
        x_span = skel_points[max_x_idx, 1] - skel_points[min_x_idx, 1]
        if y_span >= x_span:
            return [tuple(skel_points[min_y_idx]), tuple(skel_points[max_y_idx])]
        return [tuple(skel_points[min_x_idx]), tuple(skel_points[max_x_idx])]

    def _extract_ordered_skeleton_path(self, skeleton: np.ndarray,
                                      start: Tuple[int, int],
                                      end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """BFS path from start to end on skeleton."""
        from collections import deque
        h, w = skeleton.shape
        if not (0 <= start[0] < h and 0 <= start[1] < w and skeleton[start[0], start[1]]):
            return None
        if not (0 <= end[0] < h and 0 <= end[1] < w and skeleton[end[0], end[1]]):
            return None
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            r, c = current
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc)
                    if (0 <= nr < h and 0 <= nc < w and skeleton[nr, nc] and neighbor not in visited):
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        if len(visited) > 2:
            return list(visited)
        return None

    def _orient_to_tip_simple(self, path: List[List[int]],
                              tip: Tuple[int, int]) -> List[List[int]]:
        """Ensure path end is closer to tip than start."""
        if len(path) < 2:
            return path
        start_d2 = (path[0][0] - tip[0])**2 + (path[0][1] - tip[1])**2
        end_d2 = (path[-1][0] - tip[0])**2 + (path[-1][1] - tip[1])**2
        return list(reversed(path)) if start_d2 < end_d2 else path

    def _douglas_peucker_simplify(self, path: List[List[int]], epsilon: float = 2.0) -> List[List[int]]:
        """Simplify path with Douglas-Peucker."""
        if len(path) < 3:
            return path
        path_array = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(path_array, epsilon, closed=False)
        result = [[int(p[0][0]), int(p[0][1])] for p in simplified]
        return result if len(result) >= 2 else path

    def _detect_curve_type(self, path: List[List[int]]) -> str:
        """Simple: 2 points -> sharp, >6 -> curved, else rounded."""
        if not path or len(path) < 2:
            return 'sharp'
        if len(path) == 2:
            return 'sharp'
        if len(path) > 6:
            return 'curved'
        return 'rounded'

    def _fallback_extract_from_mask(self, arrow: ElementInfo,
                                    full_image: np.ndarray,
                                    tip: Tuple[int, int]) -> Optional[List[List[int]]]:
        """When skeletonize unavailable: bbox center line to tip."""
        if arrow.mask is None:
            return None
        x1, y1, x2, y2 = arrow.bbox.to_list()
        cx = (x1 + x2) // 2
        if abs(tip[1] - y1) < abs(tip[1] - y2):
            return [[cx, y2], [tip[0], tip[1]]]
        return [[cx, y1], [tip[0], tip[1]]]

    def _fallback_to_image(self, arrow: ElementInfo, full_image_np: np.ndarray,
                           img_w: int, img_h: int) -> bool:
        """Vector failed: render arrow as image cell."""
        x1, y1, x2, y2 = arrow.bbox.to_list()
        pad = self.PADDING
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(img_w, x2 + pad)
        y2_p = min(img_h, y2 + pad)
        cropped = full_image_np[y1_p:y2_p, x1_p:x2_p]
        mask_crop = None
        if arrow.mask is not None:
            try:
                if arrow.mask.shape[0] >= y2_p and arrow.mask.shape[1] >= x2_p:
                    mask_crop = arrow.mask[y1_p:y2_p, x1_p:x2_p]
            except Exception:
                pass
        processed = self._process_arrow_image(cropped, mask_crop)
        arrow.base64 = self._image_to_base64(processed)
        arrow.bbox = BoundingBox(x1_p, y1_p, x2_p, y2_p)
        arrow.arrow_start = (x1_p, (y1_p + y2_p) // 2)
        arrow.arrow_end = (x2_p, (y1_p + y2_p) // 2)
        self._generate_image_xml(arrow)
        arrow.processing_notes.append("Fallback to image")
        return True

    def _process_arrow_image(self, cropped_np: np.ndarray,
                             mask_crop: Optional[np.ndarray]) -> Image.Image:
        """Crop with white background outside mask."""
        pil_image = Image.fromarray(cropped_np)
        if mask_crop is not None and np.count_nonzero(mask_crop) > 0:
            mask_binary = (mask_crop > 127).astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask_binary, kernel, iterations=2)
            result = np.ones_like(cropped_np) * 255
            mask_3ch = np.stack([mask_dilated] * 3, axis=-1) > 0
            result[mask_3ch] = cropped_np[mask_3ch]
            pil_image = Image.fromarray(result.astype(np.uint8))
        return pil_image

    def _generate_vector_xml(self, arrow: ElementInfo,
                            arrow_attrs: Dict[str, Any] = None):
        """Emit DrawIO edge XML for vector arrow."""
        if not arrow.vector_points or len(arrow.vector_points) < 2:
            return
        points = arrow.vector_points
        start, end = points[0], points[-1]
        style = build_arrow_style(**arrow_attrs) if arrow_attrs else (
            "html=1;edgeStyle=orthogonalEdgeStyle;endArrow=classic;rounded=0;strokeWidth=2;strokeColor=#000000;orthogonalLoop=1;jettySize=auto;"
        )
        cell_id = arrow.id + 2
        waypoints_xml = ""
        if len(points) > 2:
            waypoints_xml = '<Array as="points">\n'
            for pt in points[1:-1]:
                waypoints_xml += f'              <mxPoint x="{pt[0]}" y="{pt[1]}"/>\n'
            waypoints_xml += '            </Array>'
        arrow.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" edge="1" style="{style}">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{start[0]}" y="{start[1]}" as="sourcePoint"/>
            <mxPoint x="{end[0]}" y="{end[1]}" as="targetPoint"/>
            {waypoints_xml}
          </mxGeometry>
        </mxCell>'''
        arrow.layer_level = LayerLevel.ARROW.value

    def _generate_image_xml(self, arrow: ElementInfo, arrow_attrs: Dict[str, Any] = None):
        """Emit DrawIO image cell XML for arrow fallback."""
        if not arrow.base64:
            return
        x1, y1, x2, y2 = arrow.bbox.to_list()
        width, height = x2 - x1, y2 - y1
        cell_id = arrow.id + 2
        style = f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png;base64,{arrow.base64}"
        arrow.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
          <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
        </mxCell>'''
        arrow.layer_level = LayerLevel.ARROW.value

    def _image_to_base64(self, image: Image.Image) -> str:
        """PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
