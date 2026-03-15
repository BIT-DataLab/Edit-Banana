"""
Tests for partial results handling.

Covers:
- PartialResultState serialization/deserialization
- PartialResultsHandler save/load operations
- Partial XML generation
- save_partial_results and load_partial_results convenience functions
- Edge cases: missing files, corrupted data
"""

import pytest
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.core.partial_results import (
    PartialResultState,
    PartialResultsHandler,
    save_partial_results,
    load_partial_results,
)
from modules.exceptions import EditBananaException, ProcessingPartialResultError


class TestPartialResultState:
    """Test PartialResultState dataclass."""

    def test_basic_creation(self):
        """Test creating a PartialResultState."""
        state = PartialResultState(
            image_path="/path/to/image.png",
            canvas_width=800,
            canvas_height=600,
        )
        assert state.image_path == "/path/to/image.png"
        assert state.canvas_width == 800
        assert state.canvas_height == 600
        assert state.elements == []
        assert state.xml_fragments == []

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        state = PartialResultState(
            image_path="/path/to/image.png",
            original_image_path="/original/image.png",
            elements=[{"id": 1, "type": "shape"}],
            canvas_width=800,
            canvas_height=600,
            xml_fragments=['<mxCell id="2" />'],
            intermediate_results={"upscaled": True},
            failed_stage="xml_generation",
            error_message="Something failed",
            error_code="XML_ERROR",
            completed_stages=["segmentation", "ocr"],
        )

        data = state.to_dict()

        assert data['image_path'] == "/path/to/image.png"
        assert data['original_image_path'] == "/original/image.png"
        assert data['elements'] == [{"id": 1, "type": "shape"}]
        assert data['canvas_width'] == 800
        assert data['canvas_height'] == 600
        assert data['xml_fragments'] == ['<mxCell id="2" />']
        assert data['intermediate_results'] == {"upscaled": True}
        assert data['failed_stage'] == "xml_generation"
        assert data['error_message'] == "Something failed"
        assert data['error_code'] == "XML_ERROR"
        assert data['completed_stages'] == ["segmentation", "ocr"]
        assert 'timestamp' in data

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'image_path': "/path/to/image.png",
            'original_image_path': "/original/image.png",
            'elements': [{"id": 1}],
            'canvas_width': 800,
            'canvas_height': 600,
            'xml_fragments': ['<mxCell id="2" />'],
            'intermediate_results': {"key": "value"},
            'failed_stage': "stage_name",
            'error_message': "Error",
            'error_code': "ERROR_CODE",
            'timestamp': "2024-01-01T00:00:00",
            'completed_stages': ["stage1"],
        }

        state = PartialResultState.from_dict(data)

        assert state.image_path == "/path/to/image.png"
        assert state.original_image_path == "/original/image.png"
        assert state.elements == [{"id": 1}]
        assert state.canvas_width == 800
        assert state.canvas_height == 600
        assert state.xml_fragments == ['<mxCell id="2" />']
        assert state.intermediate_results == {"key": "value"}
        assert state.failed_stage == "stage_name"
        assert state.error_message == "Error"
        assert state.error_code == "ERROR_CODE"
        assert state.timestamp == "2024-01-01T00:00:00"
        assert state.completed_stages == ["stage1"]

    def test_from_dict_with_defaults(self):
        """Test deserialization with missing optional fields."""
        data = {
            'image_path': "/path/to/image.png",
        }

        state = PartialResultState.from_dict(data)

        assert state.image_path == "/path/to/image.png"
        assert state.original_image_path is None
        assert state.elements == []
        assert state.canvas_width == 0
        assert state.canvas_height == 0
        assert state.xml_fragments == []
        assert state.intermediate_results == {}
        assert state.failed_stage is None
        assert state.error_message is None
        assert state.error_code is None
        assert state.completed_stages == []

    def test_serialize_intermediate_results(self):
        """Test serialization of intermediate results with non-JSON types."""
        state = PartialResultState(
            image_path="/path/to/image.png",
            intermediate_results={
                "text_xml": "<xml>content</xml>",
                "was_upscaled": True,
                "upscale_factor": 2.0,
                "original_image_path": "/original.png",
                "complex_object": set([1, 2, 3]),  # Non-JSON serializable
            }
        )

        data = state.to_dict()

        assert data['intermediate_results']['text_xml'] == "<xml>content</xml>"
        assert data['intermediate_results']['was_upscaled'] is True
        assert data['intermediate_results']['upscale_factor'] == 2.0
        assert data['intermediate_results']['original_image_path'] == "/original.png"
        assert data['intermediate_results']['complex_object'] == "<set>"


class TestPartialResultsHandler:
    """Test PartialResultsHandler class."""

    def setup_method(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_handler_initialization(self):
        """Test handler initialization."""
        handler = PartialResultsHandler(self.temp_dir)
        assert handler.output_dir == self.temp_dir
        assert handler.state_file == os.path.join(self.temp_dir, 'partial_state.json')

    def test_load_state_no_file(self):
        """Test loading state when file doesn't exist."""
        handler = PartialResultsHandler(self.temp_dir)
        state = handler.load_state()
        assert state is None

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        handler = PartialResultsHandler(self.temp_dir)

        # Create a state manually and save it
        state = PartialResultState(
            image_path="/path/to/image.png",
            canvas_width=800,
            canvas_height=600,
            failed_stage="test_stage",
            error_message="Test error",
        )

        # Save state directly
        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2)

        # Load it back
        loaded_state = handler.load_state()

        assert loaded_state is not None
        assert loaded_state.image_path == "/path/to/image.png"
        assert loaded_state.canvas_width == 800
        assert loaded_state.canvas_height == 600
        assert loaded_state.failed_stage == "test_stage"
        assert loaded_state.error_message == "Test error"

    def test_generate_partial_xml_no_state(self):
        """Test generating partial XML when no state exists."""
        handler = PartialResultsHandler(self.temp_dir)
        xml = handler.generate_partial_xml()
        assert xml is None

    def test_generate_partial_xml(self):
        """Test generating partial XML from saved state."""
        handler = PartialResultsHandler(self.temp_dir)

        # Create and save a state with XML fragments
        state = PartialResultState(
            image_path="/path/to/image.png",
            xml_fragments=[
                '<mxCell id="2" value="Shape 1" />',
                '<mxCell id="3" value="Shape 2" />',
            ],
        )

        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2)

        # Generate partial XML
        xml = handler.generate_partial_xml()

        assert xml is not None
        assert '<mxfile host="EditBanana" version="partial">' in xml
        assert '<diagram name="Partial Result">' in xml
        assert '<mxCell id="2" value="Shape 1" />' in xml
        assert '<mxCell id="3" value="Shape 2" />' in xml
        assert '</mxfile>' in xml

    def test_save_partial_xml(self):
        """Test saving partial XML to file."""
        handler = PartialResultsHandler(self.temp_dir)

        # Create and save a state
        state = PartialResultState(
            image_path="/path/to/image.png",
            xml_fragments=['<mxCell id="2" />'],
        )

        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2)

        # Save partial XML
        xml_path = handler.save_partial_xml()

        assert xml_path is not None
        assert os.path.exists(xml_path)
        assert xml_path.endswith('partial_result.drawio')

        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '<mxfile' in content

    def test_get_summary_no_state(self):
        """Test getting summary when no state exists."""
        handler = PartialResultsHandler(self.temp_dir)
        summary = handler.get_summary()

        assert 'error' in summary
        assert summary['error'] == 'No partial state found'

    def test_get_summary_with_state(self):
        """Test getting summary with saved state."""
        handler = PartialResultsHandler(self.temp_dir)

        # Create and save a state
        state = PartialResultState(
            image_path="/path/to/image.png",
            completed_stages=["segmentation", "ocr"],
            failed_stage="xml_generation",
            error_message="XML generation failed",
            error_code="XML_ERROR",
            elements=[{"id": 1}, {"id": 2}],
            xml_fragments=['<mxCell id="2" />'],
        )

        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2)

        summary = handler.get_summary()

        assert summary['image_path'] == "/path/to/image.png"
        assert summary['completed_stages'] == ["segmentation", "ocr"]
        assert summary['failed_stage'] == "xml_generation"
        assert summary['error_message'] == "XML generation failed"
        assert summary['error_code'] == "XML_ERROR"
        assert summary['elements_processed'] == 2
        assert summary['xml_fragments_generated'] == 1
        assert 'timestamp' in summary


class TestConvenienceFunctions:
    """Test save_partial_results and load_partial_results functions."""

    def setup_method(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_partial_results_no_file(self):
        """Test loading partial results when no file exists."""
        context = load_partial_results(self.temp_dir)
        assert context is None

    def test_load_partial_results_invalid_dir(self):
        """Test loading partial results with invalid directory."""
        non_existent_dir = os.path.join(self.temp_dir, "does_not_exist")
        context = load_partial_results(non_existent_dir)
        assert context is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_corrupted_json_file(self):
        """Test handling of corrupted JSON file."""
        handler = PartialResultsHandler(self.temp_dir)

        # Write corrupted JSON
        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            f.write("{ not valid json }")

        # Should raise JSON decode error
        with pytest.raises(json.JSONDecodeError):
            handler.load_state()

    def test_empty_json_file(self):
        """Test handling of empty JSON file."""
        handler = PartialResultsHandler(self.temp_dir)

        # Write empty file
        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            f.write("")

        # Should raise JSON decode error
        with pytest.raises(json.JSONDecodeError):
            handler.load_state()

    def test_partial_result_state_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated if not provided."""
        state = PartialResultState(image_path="/path/to/image.png")

        assert state.timestamp is not None
        assert len(state.timestamp) > 0

    def test_partial_xml_with_no_fragments(self):
        """Test generating partial XML with no fragments."""
        handler = PartialResultsHandler(self.temp_dir)

        # Create state with no fragments
        state = PartialResultState(
            image_path="/path/to/image.png",
            xml_fragments=[],
        )

        state_path = os.path.join(self.temp_dir, 'partial_state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2)

        xml = handler.generate_partial_xml()

        # Should still generate valid XML structure
        assert xml is not None
        assert '<mxfile' in xml
        assert '</mxfile>' in xml


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
