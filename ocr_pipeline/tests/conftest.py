"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.ocr_pipeline.config import Settings, update_settings


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings for the entire test session."""
    original_settings = Settings()
    
    # Use temporary directories for testing
    temp_dir = Path(tempfile.mkdtemp())
    
    test_settings = update_settings(
        pdf_input_dir=temp_dir / "test_pdfs",
        ocr_output_dir=temp_dir / "test_output",
        temp_dir=temp_dir / "test_temp",
        log_dir=temp_dir / "test_logs",
        log_level="DEBUG",
        log_to_file=False,  # Don't create log files during testing
        max_workers=1,  # Single-threaded for testing
        enable_image_preprocessing=False,  # Faster tests
    )
    
    # Create test directories
    test_settings.pdf_input_dir.mkdir(parents=True, exist_ok=True)
    test_settings.ocr_output_dir.mkdir(parents=True, exist_ok=True)
    test_settings.temp_dir.mkdir(parents=True, exist_ok=True)
    test_settings.log_dir.mkdir(parents=True, exist_ok=True)
    
    yield test_settings
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Restore original settings
    update_settings(**original_settings.dict())


@pytest.fixture
def sample_text_image():
    """Create a sample image with text for OCR testing."""
    # Create a white image
    width, height = 400, 200
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    
    # Add some text
    try:
        # Try to use a standard font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    text = "Sample OCR Text\nLine 2 with numbers: 12345\nSpecial chars: @#$%"
    draw.text((10, 10), text, fill="black", font=font)
    
    return image


@pytest.fixture
def sample_complex_layout_image():
    """Create a sample image with complex layout for testing."""
    width, height = 600, 800
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        heading_font = ImageFont.truetype("arial.ttf", 16)
        body_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = heading_font = body_font = ImageFont.load_default()
    
    # Title
    draw.text((200, 20), "Document Title", fill="black", font=title_font)
    
    # Heading
    draw.text((50, 80), "Section 1: Introduction", fill="black", font=heading_font)
    
    # Body text
    body_text = [
        "This is a sample paragraph with multiple lines of text.",
        "It demonstrates layout preservation capabilities of the OCR system.",
        "The text should maintain proper spacing and structure."
    ]
    
    y_pos = 120
    for line in body_text:
        draw.text((50, y_pos), line, fill="black", font=body_font)
        y_pos += 20
    
    # List items
    draw.text((50, 200), "• First bullet point", fill="black", font=body_font)
    draw.text((50, 220), "• Second bullet point", fill="black", font=body_font)
    draw.text((50, 240), "• Third bullet point", fill="black", font=body_font)
    
    # Two-column layout
    draw.text((50, 300), "Left Column", fill="black", font=heading_font)
    draw.text((350, 300), "Right Column", fill="black", font=heading_font)
    
    left_text = ["Left content line 1", "Left content line 2", "Left content line 3"]
    right_text = ["Right content line 1", "Right content line 2", "Right content line 3"]
    
    y_pos = 330
    for left_line, right_line in zip(left_text, right_text):
        draw.text((50, y_pos), left_line, fill="black", font=body_font)
        draw.text((350, y_pos), right_line, fill="black", font=body_font)
        y_pos += 20
    
    return image


@pytest.fixture
def temp_pdf_file(test_settings):
    """Create a temporary PDF file for testing."""
    import fitz  # PyMuPDF
    
    # Create a simple PDF with text
    doc = fitz.open()  # Create new PDF
    page = doc.new_page()
    
    # Add some text
    text = """
    Test PDF Document
    
    This is a test PDF file created for unit testing.
    It contains multiple paragraphs and formatting.
    
    Section 1: Introduction
    This section introduces the test document.
    
    Section 2: Content
    • First bullet point
    • Second bullet point
    • Third bullet point
    
    Conclusion
    This concludes the test document.
    """
    
    page.insert_text((50, 50), text, fontsize=12)
    
    # Save to temporary file
    pdf_path = test_settings.pdf_input_dir / "test_document.pdf"
    doc.save(pdf_path)
    doc.close()
    
    yield pdf_path
    
    # Cleanup
    if pdf_path.exists():
        pdf_path.unlink()


@pytest.fixture
def mock_ocr_result():
    """Mock OCR result for testing."""
    return {
        "text": "Sample extracted text from OCR",
        "confidence": 95.5,
        "word_confidences": [98.0, 93.0, 97.0, 92.0, 96.0],
        "bounding_boxes": [
            (10, 10, 60, 30),
            (70, 10, 130, 30),
            (140, 10, 180, 30),
            (190, 10, 230, 30),
            (240, 10, 280, 30)
        ],
        "processing_time": 0.5,
        "engine_used": "tesseract",
        "errors": []
    }


@pytest.fixture
def sample_layout_data():
    """Sample layout data for testing."""
    return {
        "page_number": 1,
        "page_rect": {"x0": 0, "y0": 0, "x1": 612, "y1": 792},
        "blocks": [
            {
                "type": "text",
                "bbox": [50, 50, 400, 100],
                "lines": [
                    {
                        "bbox": [50, 50, 400, 70],
                        "spans": [
                            {
                                "text": "Document Title",
                                "bbox": [50, 50, 200, 70],
                                "font": "Arial-Bold",
                                "size": 18.0,
                                "flags": 16,  # Bold flag
                                "color": 0
                            }
                        ]
                    }
                ]
            },
            {
                "type": "text",
                "bbox": [50, 120, 500, 200],
                "lines": [
                    {
                        "bbox": [50, 120, 500, 140],
                        "spans": [
                            {
                                "text": "This is a paragraph of body text with normal formatting.",
                                "bbox": [50, 120, 500, 140],
                                "font": "Arial",
                                "size": 12.0,
                                "flags": 0,
                                "color": 0
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup minimal logging for tests."""
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests