"""
OCR Pipeline Package

A comprehensive OCR pipeline for extracting text from digital and scanned PDFs
while maintaining the original text structure and layout.

Supports:
- Digital PDFs (direct text extraction)
- Scanned PDFs (OCR processing)
- Layout preservation
- Multiple OCR engines
- Batch processing
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .pdf_processor import PDFProcessor
from .text_extractor import TextExtractor
from .ocr_engine import OCREngine
from .layout_preserving import LayoutPreserver

__all__ = [
    "PDFProcessor",
    "TextExtractor", 
    "OCREngine",
    "LayoutPreserver",
]