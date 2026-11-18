"""
Stage 8c: Section Detector Module

Identifies major panels/sections in scientific figures (e.g., panel (a), (b), (c)).

Hybrid approach:
    1. Use SAM3 background elements (section_panel, title_bar) from existing segmentation
    2. Validate/refine with OpenCV line detection for panel dividers
    3. Assign elements to sections based on spatial overlap

Usage:
    from modules.section_detector import SectionDetector

    detector = SectionDetector()
    sections = detector.detect_sections(context)
    # Returns list of Section objects with bbox + child element IDs
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from .base import ProcessingContext
from .data_types import ElementInfo, BoundingBox


@dataclass
class Section:
    """Represents a detected panel/section of the figure."""
    id: int
    label: str                                # e.g., "a", "b", "c"
    bbox: BoundingBox                         # Section bounding box
    child_element_ids: List[int] = field(default_factory=list)  # IDs of elements inside
    confidence: float = 1.0                   # Detection confidence
    source: str = "sam3"                      # Detection method: "sam3", "line", "merged"

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this section."""
        return (
            self.bbox.x1 <= x <= self.bbox.x2
            and self.bbox.y1 <= y <= self.bbox.y2
        )

    def contains_bbox(self, other: BoundingBox, overlap_threshold: float = 0.5) -> bool:
        """Check if another bbox is mostly inside this section."""
        # Calculate intersection
        ix1 = max(self.bbox.x1, other.x1)
        iy1 = max(self.bbox.y1, other.y1)
        ix2 = min(self.bbox.x2, other.x2)
        iy2 = min(self.bbox.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        other_area = other.area
        if other_area == 0:
            return False

        return (intersection_area / other_area) >= overlap_threshold

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "bbox": self.bbox.to_list(),
            "child_element_ids": self.child_element_ids,
            "confidence": self.confidence,
            "source": self.source,
        }


class SectionDetector:
    """
    Detects major panels/sections in scientific figures.

    Strategy:
        1. Look for SAM3 background elements (section_panel, title_bar)
           with large area — these are likely panel boundaries
        2. Use OpenCV to detect strong horizontal/vertical lines
           that divide the image into panels
        3. Merge results from both methods
        4. Assign labels (a, b, c, ...) based on spatial ordering
        5. Assign child elements to sections
    """

    def __init__(
        self,
        min_section_area_ratio: float = 0.03,
        line_threshold: int = 100,
        merge_overlap_threshold: float = 0.6,
    ):
        """
        Args:
            min_section_area_ratio: Minimum section area as fraction of image area
            line_threshold: HoughLinesP threshold for line detection
            merge_overlap_threshold: IoU threshold for merging SAM3 + line detections
        """
        self._min_section_area_ratio = min_section_area_ratio
        self._line_threshold = line_threshold
        self._merge_overlap_threshold = merge_overlap_threshold
        self._log_prefix = "[SectionDetector]"

    def _log(self, msg: str):
        print(f"{self._log_prefix} {msg}")

    def detect_sections(
        self, context: ProcessingContext
    ) -> List[Section]:
        """
        Detect sections from ProcessingContext.

        Args:
            context: Pipeline context with elements and image info

        Returns:
            List of Section objects with child_element_ids populated
        """
        image = cv2.imread(context.image_path)
        if image is None:
            self._log(f"Could not read image: {context.image_path}")
            return []

        canvas_area = context.canvas_width * context.canvas_height
        if canvas_area == 0:
            canvas_area = image.shape[0] * image.shape[1]

        # Step 1: Get sections from SAM3 background elements
        sam3_sections = self._sections_from_sam3(
            context.elements, canvas_area
        )
        self._log(f"SAM3 sections: {len(sam3_sections)}")

        # Step 2: Get sections from line detection
        line_sections = self._sections_from_lines(
            image, canvas_area
        )
        self._log(f"Line-detected sections: {len(line_sections)}")

        # Step 3: Merge
        if sam3_sections and not line_sections:
            sections = sam3_sections
        elif line_sections and not sam3_sections:
            sections = line_sections
        elif sam3_sections and line_sections:
            sections = self._merge_sections(sam3_sections, line_sections)
        else:
            self._log("No sections detected")
            return []

        # Step 4: Assign labels (a, b, c, ...) by top-left ordering
        sections = self._assign_labels(sections)

        # Step 5: Assign child elements to sections
        self._assign_elements_to_sections(sections, context.elements)

        self._log(f"Final sections: {len(sections)}")
        for s in sections:
            self._log(
                f"  Section {s.label}: bbox={s.bbox.to_list()}, "
                f"children={len(s.child_element_ids)}, source={s.source}"
            )

        return sections

    # ================================================================
    #  Step 1: SAM3-based section detection
    # ================================================================

    def _sections_from_sam3(
        self, elements: List[ElementInfo], canvas_area: int
    ) -> List[Section]:
        """Extract sections from SAM3 background/panel elements."""
        sections = []
        min_area = canvas_area * self._min_section_area_ratio

        for elem in elements:
            elem_type = elem.element_type.lower()
            if elem_type not in ('section_panel', 'title_bar', 'panel', 'container'):
                # Also check source_group
                if not (hasattr(elem, 'source_prompt') and elem.source_prompt
                        and elem.source_prompt.lower() in ('panel', 'container', 'filled region', 'background')):
                    continue

            if elem.bbox.area < min_area:
                continue

            sections.append(
                Section(
                    id=len(sections),
                    label="",
                    bbox=elem.bbox,
                    confidence=elem.score,
                    source="sam3",
                )
            )

        return sections

    # ================================================================
    #  Step 2: Line-based section detection
    # ================================================================

    def _sections_from_lines(
        self, image: np.ndarray, canvas_area: int
    ) -> List[Section]:
        """
        Detect dividing lines (horizontal/vertical) and infer section bounding boxes.

        Only returns sections if strong full-width or full-height lines are found.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines with HoughLinesP
        min_line_length = min(w, h) * 0.3  # At least 30% of image dimension
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._line_threshold,
            minLineLength=int(min_line_length),
            maxLineGap=10,
        )

        if lines is None:
            return []

        # Classify lines as horizontal or vertical
        h_lines = []  # y-coordinates of horizontal dividers
        v_lines = []  # x-coordinates of vertical dividers

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Horizontal line (angle close to 0 or 180)
            if (angle < 5 or angle > 175) and length > w * 0.5:
                h_lines.append((y1 + y2) // 2)

            # Vertical line (angle close to 90)
            if 85 < angle < 95 and length > h * 0.3:
                v_lines.append((x1 + x2) // 2)

        # Cluster nearby lines (within 20px)
        h_dividers = self._cluster_values(h_lines, threshold=20)
        v_dividers = self._cluster_values(v_lines, threshold=20)

        if not h_dividers and not v_dividers:
            return []

        # Build grid of sections from dividers
        h_bounds = [0] + sorted(h_dividers) + [h]
        v_bounds = [0] + sorted(v_dividers) + [w]

        sections = []
        min_area = canvas_area * self._min_section_area_ratio

        for i in range(len(h_bounds) - 1):
            for j in range(len(v_bounds) - 1):
                y1 = h_bounds[i]
                y2 = h_bounds[i + 1]
                x1 = v_bounds[j]
                x2 = v_bounds[j + 1]
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                if bbox.area >= min_area:
                    sections.append(
                        Section(
                            id=len(sections),
                            label="",
                            bbox=bbox,
                            confidence=0.7,
                            source="line",
                        )
                    )

        return sections

    # ================================================================
    #  Step 3: Merge sections from multiple sources
    # ================================================================

    def _merge_sections(
        self, sam3_sections: List[Section], line_sections: List[Section]
    ) -> List[Section]:
        """Merge SAM3 and line-detected sections, preferring SAM3 when they overlap."""
        merged = list(sam3_sections)  # Start with SAM3 sections

        for ls in line_sections:
            # Check if any SAM3 section significantly overlaps
            has_overlap = False
            for ss in sam3_sections:
                iou = self._bbox_iou(ls.bbox, ss.bbox)
                if iou > self._merge_overlap_threshold:
                    has_overlap = True
                    break
            if not has_overlap:
                ls.id = len(merged)
                merged.append(ls)

        return merged

    # ================================================================
    #  Step 4 & 5: Label assignment and element mapping
    # ================================================================

    def _assign_labels(self, sections: List[Section]) -> List[Section]:
        """Assign (a), (b), (c) labels based on top-to-bottom, left-to-right order."""
        # Sort by (y_center, x_center) — row-major order
        sections.sort(key=lambda s: (s.bbox.center[1], s.bbox.center[0]))

        labels = "abcdefghijklmnopqrstuvwxyz"
        for i, section in enumerate(sections):
            section.id = i
            section.label = labels[i] if i < len(labels) else f"s{i}"

        return sections

    def _assign_elements_to_sections(
        self, sections: List[Section], elements: List[ElementInfo]
    ):
        """Assign each element to the section that contains its center."""
        for elem in elements:
            cx, cy = elem.bbox.center
            for section in sections:
                if section.contains_point(cx, cy):
                    section.child_element_ids.append(elem.id)
                    break  # Each element belongs to at most one section

    # ================================================================
    #  Helpers
    # ================================================================

    @staticmethod
    def _cluster_values(values: List[int], threshold: int = 20) -> List[int]:
        """Cluster nearby values and return cluster centers."""
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters = [[sorted_vals[0]]]
        for v in sorted_vals[1:]:
            if v - clusters[-1][-1] <= threshold:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [int(np.mean(c)) for c in clusters]

    @staticmethod
    def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
        """Calculate Intersection over Union of two bboxes."""
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = a.area + b.area - intersection
        if union == 0:
            return 0.0
        return intersection / union
