"""
Unit tests for utility functions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import hashlib
from PIL import Image
import numpy as np
import cv2

from src.ocr_pipeline.utils import (
    create_directory_if_not_exists,
    get_file_hash,
    get_pdf_files,
    clean_text,
    preprocess_image,
    pil_to_cv2,
    cv2_to_pil,
    format_processing_time,
    estimate_pdf_type,
    safe_filename,
    cleanup_temp_files
)


class TestDirectoryUtils:
    """Test directory utility functions."""
    
    def test_create_directory_if_not_exists(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_subdir" / "nested"
            
            # Directory shouldn't exist initially
            assert not test_dir.exists()
            
            # Create directory
            create_directory_if_not_exists(test_dir)
            
            # Should exist now
            assert test_dir.exists()
            assert test_dir.is_dir()
            
            # Should not raise error if called again
            create_directory_if_not_exists(test_dir)
    
    def test_create_directory_with_string_path(self):
        """Test directory creation with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = str(Path(temp_dir) / "string_path")
            
            create_directory_if_not_exists(test_dir)
            assert Path(test_dir).exists()


class TestFileUtils:
    """Test file utility functions."""
    
    def test_get_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            test_content = "Test file content for hashing"
            f.write(test_content)
            f.flush()
            
            # Calculate hash
            file_hash = get_file_hash(f.name)
            
            # Verify it's a valid MD5 hash
            assert len(file_hash) == 32
            assert all(c in '0123456789abcdef' for c in file_hash)
            
            # Calculate expected hash
            expected_hash = hashlib.md5(test_content.encode()).hexdigest()
            assert file_hash == expected_hash
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_get_pdf_files(self):
        """Test PDF file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            (temp_path / "document1.pdf").touch()
            (temp_path / "document2.PDF").touch()  # Test case insensitive
            (temp_path / "not_pdf.txt").touch()
            (temp_path / "document3.pdf").touch()
            
            # Get PDF files
            pdf_files = get_pdf_files(temp_path)
            
            # Should find 3 PDF files
            assert len(pdf_files) == 3
            
            # All should be PDF files
            for pdf_file in pdf_files:
                assert pdf_file.suffix.lower() == '.pdf'
    
    def test_get_pdf_files_nonexistent_directory(self):
        """Test PDF file discovery with non-existent directory."""
        nonexistent_dir = Path("/path/that/does/not/exist")
        pdf_files = get_pdf_files(nonexistent_dir)
        assert pdf_files == []


class TestTextUtils:
    """Test text processing utilities."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        dirty_text = "   This   is    a   test   \n\n\n   with   extra   spaces   \n\n"
        cleaned = clean_text(dirty_text)
        
        assert "This is a test" in cleaned
        assert "with extra spaces" in cleaned
        # Should not have excessive whitespace
        assert "   " not in cleaned
    
    def test_clean_text_empty(self):
        """Test cleaning empty or None text."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text("   \n\n   ") == ""
    
    def test_clean_text_preserves_structure(self):
        """Test that cleaning preserves basic structure."""
        text_with_structure = "Line 1\nLine 2\n\nLine 3"
        cleaned = clean_text(text_with_structure)
        
        lines = cleaned.split('\n')
        assert "Line 1" in lines
        assert "Line 2" in lines
        assert "Line 3" in lines


class TestImageUtils:
    """Test image processing utilities."""
    
    def test_pil_to_cv2_conversion(self, sample_text_image):
        """Test PIL to OpenCV conversion."""
        cv2_image = pil_to_cv2(sample_text_image)
        
        assert isinstance(cv2_image, np.ndarray)
        assert len(cv2_image.shape) == 3  # Should have 3 channels (BGR)
        assert cv2_image.shape[2] == 3
        assert cv2_image.shape[:2] == sample_text_image.size[::-1]  # Height, Width
    
    def test_cv2_to_pil_conversion(self, sample_text_image):
        """Test OpenCV to PIL conversion."""
        # Convert to CV2 and back
        cv2_image = pil_to_cv2(sample_text_image)
        pil_image = cv2_to_pil(cv2_image)
        
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == sample_text_image.size
        assert pil_image.mode == "RGB"
    
    def test_preprocess_image(self, sample_text_image):
        """Test image preprocessing."""
        cv2_image = pil_to_cv2(sample_text_image)
        
        # Test with enhancement enabled
        processed = preprocess_image(cv2_image, enhance_contrast=True, denoise=True)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 2  # Should be grayscale
        assert processed.dtype == np.uint8
        
        # Test with enhancement disabled
        processed_simple = preprocess_image(cv2_image, enhance_contrast=False, denoise=False)
        assert isinstance(processed_simple, np.ndarray)
    
    def test_preprocess_grayscale_image(self, sample_text_image):
        """Test preprocessing already grayscale image."""
        # Convert to grayscale first
        gray_image = cv2.cvtColor(pil_to_cv2(sample_text_image), cv2.COLOR_BGR2GRAY)
        
        processed = preprocess_image(gray_image)
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 2


class TestFormatUtils:
    """Test formatting utility functions."""
    
    def test_format_processing_time(self):
        """Test processing time formatting."""
        # Test seconds
        assert "seconds" in format_processing_time(45.5)
        assert "45.50" in format_processing_time(45.5)
        
        # Test minutes
        assert "minutes" in format_processing_time(120)
        assert "2.0" in format_processing_time(120)
        
        # Test hours
        assert "hours" in format_processing_time(7200)
        assert "2.0" in format_processing_time(7200)
    
    def test_estimate_pdf_type(self):
        """Test PDF type estimation."""
        # Test scanned (no text)
        assert estimate_pdf_type("", 1) == "scanned"
        assert estimate_pdf_type("  ", 1) == "scanned"
        
        # Test scanned (very little text)
        assert estimate_pdf_type("a few words", 1) == "scanned"
        
        # Test mixed (moderate text)
        medium_text = "Some text " * 20  # About 200 characters
        assert estimate_pdf_type(medium_text, 1) == "mixed"
        
        # Test digital (lots of text)
        long_text = "Lots of text content " * 100  # About 2000 characters
        assert estimate_pdf_type(long_text, 1) == "digital"
    
    def test_safe_filename(self):
        """Test safe filename generation."""
        # Test with invalid characters
        unsafe_name = 'file<>:"/\\|?*.txt'
        safe_name = safe_filename(unsafe_name)
        
        # Should not contain invalid characters
        invalid_chars = '<>:"/\\|?*'
        assert not any(char in safe_name for char in invalid_chars)
        
        # Test with valid name
        valid_name = "valid_filename.txt"
        assert safe_filename(valid_name) == valid_name
        
        # Test empty name
        assert safe_filename("") == "output"
        assert safe_filename("   ") == "output"


class TestCleanupUtils:
    """Test cleanup utility functions."""
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some files
            old_file = temp_path / "old_file.tmp"
            new_file = temp_path / "new_file.tmp"
            
            old_file.touch()
            new_file.touch()
            
            # Make old file actually old (modify timestamp)
            import time
            old_time = time.time() - (25 * 3600)  # 25 hours ago
            import os
            os.utime(old_file, (old_time, old_time))
            
            # Cleanup files older than 24 hours
            cleanup_temp_files(temp_path, max_age_hours=24)
            
            # Old file should be deleted, new file should remain
            assert not old_file.exists()
            assert new_file.exists()
    
    def test_cleanup_nonexistent_directory(self):
        """Test cleanup with non-existent directory."""
        # Should not raise exception
        cleanup_temp_files("/path/that/does/not/exist")