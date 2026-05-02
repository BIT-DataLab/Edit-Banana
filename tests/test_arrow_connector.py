"""
Tests for ArrowConnector module.

Test cases cover:
- Single arrow → single shape
- Multiple arrows → one shape
- Arrow between two shapes
- Ambiguous connection handling
- Confidence threshold edge cases
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_types import ElementInfo, BoundingBox
from modules.base import ProcessingContext
from modules.arrow_connector import ArrowConnector, ArrowConnection


class TestArrowConnector:
    """Test suite for ArrowConnector class."""

    @pytest.fixture
    def connector(self):
        """Create a fresh ArrowConnector instance."""
        return ArrowConnector(distance_threshold=50.0)

    @pytest.fixture
    def context(self):
        """Create an empty ProcessingContext."""
        return ProcessingContext(image_path="dummy.png")

    def create_shape(self, shape_id: int, x1: int, y1: int, x2: int, y2: int,
                     shape_type: str = "rectangle") -> ElementInfo:
        """Helper to create a shape ElementInfo."""
        return ElementInfo(
            id=shape_id,
            element_type=shape_type,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            score=0.9
        )

    def create_arrow(self, arrow_id: int, start: tuple, end: tuple,
                     arrow_type: str = "arrow") -> ElementInfo:
        """Helper to create an arrow ElementInfo."""
        # Calculate bbox from start and end points
        x1 = min(start[0], end[0])
        y1 = min(start[1], end[1])
        x2 = max(start[0], end[0])
        y2 = max(start[1], end[1])

        return ElementInfo(
            id=arrow_id,
            element_type=arrow_type,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            arrow_start=start,
            arrow_end=end,
            score=0.85
        )

    def test_no_arrows(self, connector, context):
        """Test processing when no arrows are present."""
        # Only shapes, no arrows
        context.elements = [
            self.create_shape(1, 100, 100, 200, 200),
            self.create_shape(2, 300, 300, 400, 400),
        ]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['arrows_connected'] == 0
        assert result.metadata['total_arrows'] == 0

    def test_no_shapes(self, connector, context):
        """Test processing when arrows exist but no shapes."""
        context.elements = [
            self.create_arrow(1, (150, 150), (350, 350)),
        ]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['arrows_connected'] == 0
        assert result.metadata['total_arrows'] == 1

    def test_single_arrow_to_single_shape(self, connector, context):
        """Test connecting a single arrow to a nearby shape."""
        # Shape at (100, 100) to (200, 200)
        shape = self.create_shape(1, 100, 100, 200, 200)

        # Arrow ending near shape boundary
        # Start far away, end near the shape's left edge
        arrow = self.create_arrow(2, (300, 150), (105, 150))

        context.elements = [shape, arrow]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['total_arrows'] == 1
        assert result.metadata['arrows_connected'] == 1

        # Check arrow has connection metadata
        arrow_elem = result.elements[1]
        assert 'arrow_connection' in arrow_elem.metadata
        assert arrow_elem.metadata['arrow_connection']['target_id'] == 1
        assert arrow_elem.metadata['arrow_connection']['source_id'] is None

    def test_arrow_between_two_shapes(self, connector, context):
        """Test arrow connecting source and target shapes."""
        # Source shape on the left
        source_shape = self.create_shape(1, 100, 100, 200, 200)

        # Target shape on the right
        target_shape = self.create_shape(2, 400, 100, 500, 200)

        # Arrow from source to target
        arrow = self.create_arrow(3, (200, 150), (400, 150))

        context.elements = [source_shape, target_shape, arrow]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['arrows_connected'] == 1

        arrow_elem = result.elements[2]
        conn_meta = arrow_elem.metadata['arrow_connection']
        assert conn_meta['source_id'] == 1
        assert conn_meta['target_id'] == 2
        assert conn_meta['source_confidence'] > 0
        assert conn_meta['target_confidence'] > 0

    def test_multiple_arrows_to_one_shape(self, connector, context):
        """Test multiple arrows connecting to the same shape."""
        target_shape = self.create_shape(1, 200, 200, 300, 300)

        # Three arrows pointing to the shape from different directions
        arrow1 = self.create_arrow(2, (100, 250), (195, 250))  # From left
        arrow2 = self.create_arrow(3, (250, 100), (250, 195))  # From top
        arrow3 = self.create_arrow(4, (400, 250), (305, 250))  # From right

        context.elements = [target_shape, arrow1, arrow2, arrow3]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['total_arrows'] == 3
        assert result.metadata['arrows_connected'] == 3

        # All arrows should connect to shape 1
        for elem in result.elements[1:]:
            assert elem.metadata['arrow_connection']['target_id'] == 1

    def test_floating_arrow_no_connection(self, connector, context):
        """Test arrow too far from any shape doesn't connect."""
        shape = self.create_shape(1, 100, 100, 200, 200)

        # Arrow far away (beyond 50px threshold)
        arrow = self.create_arrow(2, (500, 500), (600, 600))

        context.elements = [shape, arrow]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['arrows_connected'] == 0

        arrow_elem = result.elements[1]
        conn_meta = arrow_elem.metadata['arrow_connection']
        assert conn_meta['source_id'] is None
        assert conn_meta['target_id'] is None

    def test_ambiguous_connection_picks_nearest(self, connector, context):
        """Test that ambiguous connections pick the nearest shape."""
        # Two shapes close to each other
        shape1 = self.create_shape(1, 100, 100, 200, 200)
        shape2 = self.create_shape(2, 180, 180, 280, 280)  # Overlapping region

        # Arrow endpoint in overlapping region - closer to shape2
        arrow = self.create_arrow(3, (50, 50), (200, 200))

        context.elements = [shape1, shape2, arrow]

        result = connector.process(context)

        assert result.success is True
        arrow_elem = result.elements[2]
        conn_meta = arrow_elem.metadata['arrow_connection']

        # Should connect to the nearest shape (shape2 in this case)
        # Distance to shape1 center (150,150): ~71px
        # Distance to shape2 center (230,230): ~42px
        assert conn_meta['target_id'] == 2

    def test_arrow_inside_shape_zero_distance(self, connector, context):
        """Test arrow endpoint inside shape has zero distance."""
        shape = self.create_shape(1, 100, 100, 300, 300)

        # Arrow ending inside the shape
        arrow = self.create_arrow(2, (400, 200), (200, 200))

        context.elements = [shape, arrow]

        result = connector.process(context)

        assert result.success is True
        arrow_elem = result.elements[1]
        conn_meta = arrow_elem.metadata['arrow_connection']

        # Should connect with high confidence (endpoint inside shape)
        assert conn_meta['target_id'] == 1
        assert conn_meta['target_confidence'] == 1.0  # Zero distance = max confidence

    def test_connection_confidence_threshold(self, connector, context):
        """Test that distance affects confidence score."""
        shape = self.create_shape(1, 100, 100, 200, 200)

        # Arrow at edge of threshold (50px)
        arrow = self.create_arrow(2, (300, 150), (250, 150))  # 50px from shape

        context.elements = [shape, arrow]

        result = connector.process(context)

        arrow_elem = result.elements[1]
        conn_meta = arrow_elem.metadata['arrow_connection']

        # Should still connect (at threshold)
        assert conn_meta['target_id'] == 1
        # Confidence should be lower due to distance
        assert 0 < conn_meta['target_confidence'] < 1.0

    def test_custom_distance_threshold(self, context):
        """Test connector with custom distance threshold."""
        connector = ArrowConnector(distance_threshold=100.0)

        shape = self.create_shape(1, 100, 100, 200, 200)

        # Arrow at 75px distance (would miss with 50px threshold)
        arrow = self.create_arrow(2, (300, 150), (275, 150))

        context.elements = [shape, arrow]

        result = connector.process(context)

        # Should connect with larger threshold
        assert result.metadata['arrows_connected'] == 1

    def test_line_type_element(self, connector, context):
        """Test that 'line' type elements are also processed as arrows."""
        shape = self.create_shape(1, 100, 100, 200, 200)
        line = self.create_arrow(2, (300, 150), (105, 150), arrow_type="line")

        context.elements = [shape, line]

        result = connector.process(context)

        assert result.success is True
        assert result.metadata['total_arrows'] == 1

    def test_processing_notes_added(self, connector, context):
        """Test that processing notes are added to arrow elements."""
        shape = self.create_shape(1, 100, 100, 200, 200)
        arrow = self.create_arrow(2, (300, 150), (105, 150))

        context.elements = [shape, arrow]

        result = connector.process(context)

        arrow_elem = result.elements[1]
        notes = arrow_elem.processing_notes

        # Should have notes about connection
        assert len(notes) >= 2
        assert any("source" in note.lower() for note in notes)
        assert any("target" in note.lower() for note in notes)

    def test_connection_serialization(self, connector, context):
        """Test that connections can be serialized to dict."""
        shape = self.create_shape(1, 100, 100, 200, 200)
        arrow = self.create_arrow(2, (300, 150), (105, 150))

        context.elements = [shape, arrow]

        result = connector.process(context)

        connections = result.metadata['connections']
        assert len(connections) == 1

        conn_dict = connections[0]
        assert conn_dict['arrow_id'] == 2
        assert conn_dict['target_id'] == 1
        assert 'source_confidence' in conn_dict
        assert 'target_confidence' in conn_dict


class TestDistanceCalculation:
    """Test distance calculation methods."""

    def test_point_inside_bbox(self):
        """Test distance is negative when point is strictly inside bbox."""
        connector = ArrowConnector()
        shape = ElementInfo(
            id=1,
            element_type="rectangle",
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            score=0.9
        )

        # Point strictly inside - distance is negative (distance to nearest edge)
        distance = connector._distance_to_shape((150, 150), shape)
        assert distance == -50.0  # 50px to nearest edge, negative for inside

    def test_point_outside_bbox(self):
        """Test distance calculation for point outside bbox."""
        connector = ArrowConnector()
        shape = ElementInfo(
            id=1,
            element_type="rectangle",
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            score=0.9
        )

        # Point to the right of bbox
        distance = connector._distance_to_shape((250, 150), shape)
        assert distance == 50.0  # 250 - 200 = 50

    def test_point_diagonal_to_bbox(self):
        """Test distance for point diagonally outside bbox."""
        connector = ArrowConnector()
        shape = ElementInfo(
            id=1,
            element_type="rectangle",
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            score=0.9
        )

        # Point diagonally outside (bottom-right corner)
        import math
        distance = connector._distance_to_shape((250, 250), shape)
        expected = math.sqrt(50**2 + 50**2)  # sqrt(2500 + 2500) = sqrt(5000) ≈ 70.71
        assert abs(distance - expected) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
