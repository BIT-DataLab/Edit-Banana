"""
Arrow connector module.

Automatically associates arrows with their source and target shapes
based on spatial proximity, directional alignment, and visual relationships.
"""

import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .data_types import ElementInfo, BoundingBox, ProcessingResult
from .base import BaseProcessor, ProcessingContext


@dataclass
class ConnectionCandidate:
    """Candidate connection between an arrow endpoint and a shape."""
    arrow_id: int
    shape_id: int
    endpoint_type: str  # 'start' or 'end'
    distance: float
    confidence: float
    shape_center: Tuple[float, float]
    endpoint: Tuple[float, float]


@dataclass
class ArrowConnection:
    """Final connection between an arrow and its source/target shapes."""
    arrow_id: int
    source_id: Optional[int]  # Shape ID at arrow start
    target_id: Optional[int]  # Shape ID at arrow end
    source_confidence: float
    target_confidence: float


class ArrowConnector(BaseProcessor):
    """
    Connect arrows to their source and target shapes.

    Algorithm:
    1. For each arrow, get its start and end points
    2. Calculate distance from each endpoint to all shape boundaries
    3. Find nearest shape within threshold for each endpoint
    4. Score connections based on distance and directional alignment
    5. Handle ambiguous cases (multiple arrows near one shape)
    """

    DEFAULT_DISTANCE_THRESHOLD = 50.0  # pixels
    DIRECTIONAL_WEIGHT = 0.3  # Weight for directional alignment in confidence

    def __init__(self, config=None, distance_threshold: float = None):
        super().__init__(config)
        self.distance_threshold = distance_threshold or self.DEFAULT_DISTANCE_THRESHOLD

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process all arrows and connect them to shapes."""
        self._log("Connecting arrows to shapes")

        # Separate arrows and shapes
        arrows = [e for e in context.elements
                  if e.element_type.lower() in {'arrow', 'line', 'connector'}]
        shapes = [e for e in context.elements
                  if e.element_type.lower() not in {'arrow', 'line', 'connector', 'text'}]

        if not arrows:
            self._log("No arrows found")
            return ProcessingResult(
                success=True,
                elements=context.elements,
                metadata={'arrows_connected': 0, 'total_arrows': 0}
            )

        if not shapes:
            self._log("No shapes found for arrow connection")
            return ProcessingResult(
                success=True,
                elements=context.elements,
                metadata={'arrows_connected': 0, 'total_arrows': len(arrows)}
            )

        # Find connections for all arrows
        connections = self._find_connections(arrows, shapes)

        # Apply connections to elements
        connected_count = self._apply_connections(context, connections)

        self._log(f"Connected {connected_count}/{len(arrows)} arrows")

        return ProcessingResult(
            success=True,
            elements=context.elements,
            metadata={
                'arrows_connected': connected_count,
                'total_arrows': len(arrows),
                'connections': [self._connection_to_dict(c) for c in connections]
            }
        )

    def _find_connections(self, arrows: List[ElementInfo],
                         shapes: List[ElementInfo]) -> List[ArrowConnection]:
        """Find shape connections for all arrows."""
        connections = []

        for arrow in arrows:
            source_id, source_conf = self._find_nearest_shape(
                arrow, shapes, 'start'
            )
            target_id, target_conf = self._find_nearest_shape(
                arrow, shapes, 'end'
            )

            connections.append(ArrowConnection(
                arrow_id=arrow.id,
                source_id=source_id,
                target_id=target_id,
                source_confidence=source_conf,
                target_confidence=target_conf
            ))

        return connections

    def _find_nearest_shape(self, arrow: ElementInfo,
                           shapes: List[ElementInfo],
                           endpoint_type: str) -> Tuple[Optional[int], float]:
        """
        Find the nearest shape to an arrow endpoint.

        Returns:
            Tuple of (shape_id, confidence) or (None, 0.0) if no shape within threshold
        """
        # Get endpoint coordinates
        if endpoint_type == 'start':
            endpoint = arrow.arrow_start
        else:
            endpoint = arrow.arrow_end

        if endpoint is None:
            return None, 0.0

        endpoint = (float(endpoint[0]), float(endpoint[1]))

        # Find all candidates within threshold
        candidates = []
        for shape in shapes:
            distance = self._distance_to_shape(endpoint, shape)
            if distance <= self.distance_threshold:
                # Calculate confidence based on distance (closer = higher confidence)
                confidence = self._calculate_confidence(distance, endpoint, shape)
                candidates.append((shape.id, distance, confidence))

        if not candidates:
            return None, 0.0

        # Sort by distance and return nearest
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0], candidates[0][2]

    def _distance_to_shape(self, point: Tuple[float, float],
                          shape: ElementInfo) -> float:
        """
        Calculate distance from a point to a shape's boundary.

        Returns:
            Distance in pixels (negative if strictly inside shape, 0 on boundary,
            positive if outside). This allows sorting to prefer containment.
        """
        bbox = shape.bbox
        px, py = point

        # Check if point is strictly inside bounding box (not on boundary)
        if bbox.x1 < px < bbox.x2 and bbox.y1 < py < bbox.y2:
            # Return negative distance to nearest edge (indicates inside)
            return -min(px - bbox.x1, bbox.x2 - px, py - bbox.y1, bbox.y2 - py)

        # Check if point is on boundary or inside (inclusive)
        if bbox.x1 <= px <= bbox.x2 and bbox.y1 <= py <= bbox.y2:
            return 0.0

        # Calculate distance to nearest edge
        dx = max(bbox.x1 - px, 0, px - bbox.x2)
        dy = max(bbox.y1 - py, 0, py - bbox.y2)

        return math.sqrt(dx * dx + dy * dy)

    def _calculate_confidence(self, distance: float,
                             endpoint: Tuple[float, float],
                             shape: ElementInfo) -> float:
        """
        Calculate connection confidence score.

        Factors:
        - Distance (closer = higher confidence)
        - Directional alignment (arrow pointing toward shape center)
        """
        # Distance confidence (exponential decay)
        dist_conf = math.exp(-distance / (self.distance_threshold / 2))

        # Directional confidence (optional enhancement)
        # For now, just use distance-based confidence
        confidence = dist_conf

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _apply_connections(self, context: ProcessingContext,
                          connections: List[ArrowConnection]) -> int:
        """Apply connections to arrow elements."""
        connected_count = 0

        # Build id -> element mapping
        element_map = {e.id: e for e in context.elements}

        for conn in connections:
            arrow = element_map.get(conn.arrow_id)
            if not arrow:
                continue

            # Store connection metadata
            arrow.metadata = arrow.metadata or {}
            arrow.metadata['arrow_connection'] = {
                'source_id': conn.source_id,
                'target_id': conn.target_id,
                'source_confidence': conn.source_confidence,
                'target_confidence': conn.target_confidence
            }

            # Add processing notes
            if conn.source_id is not None:
                arrow.processing_notes.append(
                    f"Connected to source shape {conn.source_id} "
                    f"(confidence: {conn.source_confidence:.2f})"
                )
            else:
                arrow.processing_notes.append("No source shape found")

            if conn.target_id is not None:
                arrow.processing_notes.append(
                    f"Connected to target shape {conn.target_id} "
                    f"(confidence: {conn.target_confidence:.2f})"
                )
            else:
                arrow.processing_notes.append("No target shape found")

            if conn.source_id is not None or conn.target_id is not None:
                connected_count += 1

        return connected_count

    def _connection_to_dict(self, conn: ArrowConnection) -> Dict[str, Any]:
        """Convert connection to dictionary for serialization."""
        return {
            'arrow_id': conn.arrow_id,
            'source_id': conn.source_id,
            'target_id': conn.target_id,
            'source_confidence': conn.source_confidence,
            'target_confidence': conn.target_confidence
        }
