"""
Integration tests for the complete OCR pipeline.
"""

import pytest
from pathlib import Path
import tempfile
import time

from src.ocr_pipeline import TextExtractor, PDFProcessor, OCREngine
from src.ocr_pipeline.config import update_settings, TextExtractionMode


class TestPipelineIntegration:
    """Test the complete pipeline integration."""
    
    @pytest.mark.slow
    def test_complete_pipeline(self, test_settings, temp_pdf_file):
        """Test the complete extraction pipeline."""
        extractor = TextExtractor()
        
        # Test extraction
        result = extractor.extract_from_pdf(temp_pdf_file)
        
        # Verify results
        assert result.filename == temp_pdf_file.name
        assert result.filepath == temp_pdf_file
        assert result.total_pages > 0
        assert len(result.extracted_text) > 0
        assert result.processing_time > 0
        assert result.extraction_method in ["digital", "ocr", "hybrid"]
        assert len(result.errors) == 0
    
    def test_batch_processing(self, test_settings):
        """Test batch processing functionality."""
        import fitz
        
        # Create multiple test PDFs
        pdf_files = []
        for i in range(3):
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), f"Test PDF {i+1}\nContent for document {i+1}")
            
            pdf_path = test_settings.pdf_input_dir / f"test_doc_{i+1}.pdf"
            doc.save(pdf_path)
            doc.close()
            pdf_files.append(pdf_path)
        
        # Process batch
        extractor = TextExtractor()
        results = extractor.extract_batch(pdf_files)
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.filename == f"test_doc_{i+1}.pdf"
            assert len(result.extracted_text) > 0
            assert f"Test PDF {i+1}" in result.extracted_text
        
        # Cleanup
        for pdf_file in pdf_files:
            if pdf_file.exists():
                pdf_file.unlink()
    
    def test_different_extraction_modes(self, test_settings, temp_pdf_file):
        """Test different extraction modes."""
        extractor = TextExtractor()
        
        # Test auto mode
        update_settings(extraction_mode=TextExtractionMode.AUTO)
        result_auto = extractor.extract_from_pdf(temp_pdf_file)
        
        # Test digital mode
        update_settings(extraction_mode=TextExtractionMode.DIGITAL)
        result_digital = extractor.extract_from_pdf(temp_pdf_file)
        
        # Verify both work
        assert len(result_auto.extracted_text) > 0
        assert len(result_digital.extracted_text) > 0
        
        # Digital should be faster (typically)
        # Note: This might not always be true in tests due to overhead
        assert result_digital.extraction_method == "digital"
    
    @pytest.mark.slow
    def test_ocr_fallback(self, test_settings, sample_text_image):
        """Test OCR fallback functionality."""
        # Save image as a temporary file for testing
        temp_image_path = test_settings.temp_dir / "test_image.png"
        sample_text_image.save(temp_image_path)
        
        # Test OCR engine directly
        ocr_engine = OCREngine()
        
        if ocr_engine.get_available_engines():
            result = ocr_engine.extract_text_from_image(sample_text_image)
            
            assert "text" in result
            assert "confidence" in result
            assert "engine_used" in result
            assert len(result["errors"]) == 0
        else:
            pytest.skip("No OCR engines available")
        
        # Cleanup
        if temp_image_path.exists():
            temp_image_path.unlink()


class TestErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_invalid_pdf_file(self, test_settings):
        """Test handling of invalid PDF files."""
        # Create a non-PDF file with .pdf extension
        fake_pdf = test_settings.pdf_input_dir / "fake.pdf"
        fake_pdf.write_text("This is not a PDF file")
        
        extractor = TextExtractor()
        result = extractor.extract_from_pdf(fake_pdf)
        
        # Should handle gracefully
        assert result.extraction_method == "failed"
        assert len(result.errors) > 0
        assert result.extracted_text == ""
        
        # Cleanup
        fake_pdf.unlink()
    
    def test_nonexistent_pdf_file(self, test_settings):
        """Test handling of non-existent PDF files."""
        nonexistent_pdf = test_settings.pdf_input_dir / "does_not_exist.pdf"
        
        extractor = TextExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_pdf(nonexistent_pdf)
    
    def test_corrupted_pdf_handling(self, test_settings):
        """Test handling of corrupted PDF files."""
        # Create a file that looks like PDF but is corrupted
        corrupted_pdf = test_settings.pdf_input_dir / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"%PDF-1.4\n" + b"corrupted content" * 100)
        
        extractor = TextExtractor()
        result = extractor.extract_from_pdf(corrupted_pdf)
        
        # Should handle gracefully
        assert result.extraction_method == "failed"
        assert len(result.errors) > 0
        
        # Cleanup
        corrupted_pdf.unlink()


class TestPerformance:
    """Performance tests for the pipeline."""
    
    def test_processing_time_tracking(self, test_settings, temp_pdf_file):
        """Test that processing time is tracked correctly."""
        extractor = TextExtractor()
        
        start_time = time.time()
        result = extractor.extract_from_pdf(temp_pdf_file)
        end_time = time.time()
        
        # Processing time should be reasonable
        assert result.processing_time > 0
        assert result.processing_time < (end_time - start_time) + 1  # Allow some overhead
    
    @pytest.mark.slow
    def test_large_document_handling(self, test_settings):
        """Test handling of larger documents."""
        import fitz
        
        # Create a larger PDF
        doc = fitz.open()
        
        # Add multiple pages with content
        for page_num in range(5):
            page = doc.new_page()
            text = f"Page {page_num + 1}\n" + "Line of text. " * 50
            page.insert_text((50, 50), text, fontsize=12)
        
        large_pdf = test_settings.pdf_input_dir / "large_test.pdf"
        doc.save(large_pdf)
        doc.close()
        
        # Process the large PDF
        extractor = TextExtractor()
        result = extractor.extract_from_pdf(large_pdf)
        
        # Verify processing
        assert result.total_pages == 5
        assert len(result.extracted_text) > 1000  # Should have substantial content
        assert "Page 1" in result.extracted_text
        assert "Page 5" in result.extracted_text
        
        # Cleanup
        large_pdf.unlink()


class TestConfigurationImpact:
    """Test how different configurations affect pipeline behavior."""
    
    def test_layout_preservation_setting(self, test_settings, temp_pdf_file):
        """Test layout preservation setting impact."""
        extractor = TextExtractor()
        
        # Test with layout preservation
        update_settings(preserve_layout=True)
        result_with_layout = extractor.extract_from_pdf(temp_pdf_file)
        
        # Test without layout preservation
        update_settings(preserve_layout=False)
        result_without_layout = extractor.extract_from_pdf(temp_pdf_file)
        
        # Both should work but might have different formatting
        assert len(result_with_layout.extracted_text) > 0
        assert len(result_without_layout.extracted_text) > 0
    
    def test_confidence_threshold_setting(self, test_settings, sample_text_image):
        """Test OCR confidence threshold setting."""
        ocr_engine = OCREngine()
        
        if not ocr_engine.get_available_engines():
            pytest.skip("No OCR engines available")
        
        # Test with low threshold
        update_settings(ocr_confidence_threshold=10.0)
        result_low = ocr_engine.extract_text_from_image(sample_text_image)
        
        # Test with high threshold
        update_settings(ocr_confidence_threshold=90.0)
        result_high = ocr_engine.extract_text_from_image(sample_text_image)
        
        # Both should work, but high threshold might have fewer results
        assert "text" in result_low
        assert "text" in result_high