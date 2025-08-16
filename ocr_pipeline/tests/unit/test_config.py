"""
Unit tests for configuration management.
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.ocr_pipeline.config import Settings, get_settings, update_settings, OCREngine, TextExtractionMode


class TestSettings:
    """Test the Settings class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.ocr_engine == OCREngine.TESSERACT
        assert settings.extraction_mode == TextExtractionMode.AUTO
        assert settings.preserve_layout is True
        assert settings.ocr_confidence_threshold == 60.0
        assert settings.max_workers >= 1
        assert settings.log_level == "INFO"
    
    def test_settings_validation(self):
        """Test settings validation."""
        # Test confidence threshold validation
        settings = Settings(ocr_confidence_threshold=150.0)
        assert settings.ocr_confidence_threshold == 100.0  # Should be clamped
        
        settings = Settings(ocr_confidence_threshold=-10.0)
        assert settings.ocr_confidence_threshold == 0.0  # Should be clamped
        
        # Test max_workers validation (should not exceed CPU count)
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        settings = Settings(max_workers=cpu_count + 10)
        assert settings.max_workers <= cpu_count
    
    def test_path_resolution(self):
        """Test path resolution to absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                project_root=Path(temp_dir),
                pdf_input_dir="test_input",
                ocr_output_dir="test_output"
            )
            
            assert settings.pdf_input_dir.is_absolute()
            assert settings.ocr_output_dir.is_absolute()
            assert str(settings.pdf_input_dir).startswith(temp_dir)
            assert str(settings.ocr_output_dir).startswith(temp_dir)
    
    def test_environment_variables(self, monkeypatch):
        """Test environment variable loading."""
        # Set environment variables
        monkeypatch.setenv("OCR_ENGINE", "easyocr")
        monkeypatch.setenv("OCR_EXTRACTION_MODE", "ocr")
        monkeypatch.setenv("OCR_PRESERVE_LAYOUT", "false")
        monkeypatch.setenv("OCR_LOG_LEVEL", "DEBUG")
        
        settings = Settings()
        
        assert settings.ocr_engine == OCREngine.EASYOCR
        assert settings.extraction_mode == TextExtractionMode.OCR
        assert settings.preserve_layout is False
        assert settings.log_level == "DEBUG"


class TestConfigurationFunctions:
    """Test configuration utility functions."""
    
    def test_get_settings(self):
        """Test get_settings function."""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    def test_update_settings(self):
        """Test update_settings function."""
        original_engine = get_settings().ocr_engine
        
        # Update settings
        new_settings = update_settings(ocr_engine=OCREngine.EASYOCR)
        assert new_settings.ocr_engine == OCREngine.EASYOCR
        
        # Verify the global settings were updated
        current_settings = get_settings()
        assert current_settings.ocr_engine == OCREngine.EASYOCR
        
        # Restore original settings
        update_settings(ocr_engine=original_engine)
    
    def test_settings_persistence(self):
        """Test that settings changes persist."""
        # Change a setting
        update_settings(max_workers=2)
        
        # Get settings again and verify change persisted
        settings = get_settings()
        assert settings.max_workers == 2
    
    def test_invalid_enum_values(self):
        """Test handling of invalid enum values."""
        with pytest.raises(ValueError):
            Settings(ocr_engine="invalid_engine")
        
        with pytest.raises(ValueError):
            Settings(extraction_mode="invalid_mode")


class TestEnums:
    """Test enum definitions."""
    
    def test_ocr_engine_enum(self):
        """Test OCREngine enum values."""
        assert OCREngine.TESSERACT == "tesseract"
        assert OCREngine.EASYOCR == "easyocr"
        assert OCREngine.PADDLEOCR == "paddleocr"
        assert OCREngine.PYMUPDF == "pymupdf"
    
    def test_text_extraction_mode_enum(self):
        """Test TextExtractionMode enum values."""
        assert TextExtractionMode.AUTO == "auto"
        assert TextExtractionMode.DIGITAL == "digital"
        assert TextExtractionMode.OCR == "ocr"
        assert TextExtractionMode.HYBRID == "hybrid"
    
    def test_enum_string_conversion(self):
        """Test enum string conversions."""
        engine = OCREngine.TESSERACT
        assert str(engine) == "tesseract"
        
        mode = TextExtractionMode.AUTO
        assert str(mode) == "auto"