# OCR Pipeline

A comprehensive OCR pipeline for extracting text from digital and scanned PDFs while maintaining the original text structure and layout.

## Features

- **Multi-format Support**: Handles both digital and scanned PDFs
- **Multiple OCR Engines**: Supports Tesseract, EasyOCR, and PaddleOCR with automatic fallback
- **Layout Preservation**: Maintains document structure, formatting, and spatial relationships
- **Intelligent Processing**: Automatically detects PDF type and chooses optimal extraction method
- **Batch Processing**: Process multiple PDFs efficiently
- **Configuration Management**: Flexible settings with environment variable support
- **Comprehensive Logging**: Detailed logging with performance metrics
- **Error Handling**: Robust error handling with graceful degradation

## Installation

### Prerequisites

1. **Python 3.8+** is required
2. **Tesseract OCR** (optional but recommended):
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

### Install Dependencies

```bash
# Clone or download the project
cd ocr_pipeline

# Install Python dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For better performance, you may want to install additional packages:

```bash
# For advanced image processing
pip install opencv-python scikit-image

# For machine learning-based column detection
pip install scikit-learn

# For AI-powered text extraction (experimental)
pip install transformers torch
```

## Quick Start

### Basic Usage

```python
from src.ocr_pipeline import TextExtractor
from pathlib import Path

# Initialize the extractor
extractor = TextExtractor()

# Extract text from a single PDF
result = extractor.extract_from_pdf("path/to/your/document.pdf")

print(f"Extracted {len(result.extracted_text)} characters")
print(f"Processing time: {result.processing_time:.2f} seconds")
print(f"Method used: {result.extraction_method}")
```

### Batch Processing

```python
from src.ocr_pipeline.utils import get_pdf_files
from src.ocr_pipeline import TextExtractor

# Get all PDFs from a directory
pdf_files = get_pdf_files("pdfs/")

# Process all PDFs
extractor = TextExtractor()
results = extractor.extract_batch(pdf_files)

for result in results:
    print(f"Processed: {result.filename}")
    print(f"Characters: {len(result.extracted_text)}")
    print(f"Confidence: {sum(result.confidence_scores)/len(result.confidence_scores):.1f}%")
```

### Command Line Usage

Run the existing script:

```bash
python run.py
```

Or use the new modular approach:

```bash
python -m src.ocr_pipeline.main --input-dir pdfs/ --output-dir output/
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
OCR_ENGINE=tesseract
OCR_EXTRACTION_MODE=auto
OCR_PRESERVE_LAYOUT=true
OCR_IMAGE_DPI=300
OCR_LOG_LEVEL=INFO
OCR_MAX_WORKERS=4
```

### Configuration File

You can also modify settings in `src/ocr_pipeline/config.py`:

```python
from src.ocr_pipeline.config import update_settings

# Update settings programmatically
update_settings(
    ocr_engine="easyocr",
    extraction_mode="hybrid",
    image_dpi=400
)
```

## Extraction Modes

1. **Auto Mode** (default): Automatically detects PDF type and chooses the best method
2. **Digital Mode**: Fast text extraction for born-digital PDFs
3. **OCR Mode**: Forces OCR processing for scanned documents
4. **Hybrid Mode**: Tries digital extraction first, falls back to OCR

## OCR Engines

### Tesseract (Default)
- Fast and reliable
- Good for standard documents
- Extensive language support

### EasyOCR
- Better accuracy for complex layouts
- Supports many languages out of the box
- GPU acceleration available

### PaddleOCR
- Excellent for preserving layout structure
- High accuracy for tables and forms
- Good multilingual support

## Project Structure

```
ocr_pipeline/
├── src/
│   └── ocr_pipeline/
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── pdf_processor.py       # PDF analysis and processing
│       ├── text_extractor.py      # Main text extraction logic
│       ├── ocr_engine.py          # OCR engine interface
│       ├── layout_preserving.py   # Layout analysis
│       ├── logging_config.py      # Logging setup
│       └── utils.py               # Utility functions
├── tests/                         # Test suite
├── logs/                          # Log files
├── temp/                          # Temporary processing files
├── pdfs/                          # Input PDF directory
├── output/                        # Output directory
├── requirements.txt               # Python dependencies
├── run.py                         # Legacy main script
└── README.md                      # This file
```

## Advanced Features

### Layout Analysis

```python
from src.ocr_pipeline import LayoutPreserver, PDFProcessor

# Analyze document layout
processor = PDFProcessor()
layout_analyzer = LayoutPreserver()

pdf_info = processor.analyze_pdf("document.pdf")
layout = layout_analyzer.analyze_layout(pdf_info.pages[0])

# Get structured text
structured_text = layout_analyzer.reconstruct_document_structure(layout)
```

### Performance Monitoring

```python
from src.ocr_pipeline.logging_config import log_performance

# Performance logging is automatic, but you can add custom metrics
log_performance("custom_operation", duration=1.23, custom_metric=42)
```

### Custom OCR Engine

```python
from src.ocr_pipeline import OCREngine

# Initialize with specific engine
ocr = OCREngine()

# Test engine availability
available_engines = ocr.get_available_engines()
print(f"Available engines: {available_engines}")

# Use specific engine
result = ocr.extract_text_from_image(image, engine="easyocr")
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src/ocr_pipeline --cov-report=html
```

## Performance Tips

1. **Choose the Right Engine**: 
   - Use Tesseract for speed
   - Use EasyOCR for accuracy
   - Use PaddleOCR for layout preservation

2. **Optimize Image DPI**: 
   - 300 DPI is usually sufficient
   - Higher DPI increases processing time significantly

3. **Use Batch Processing**: 
   - Process multiple files together for better efficiency

4. **Enable Preprocessing**: 
   - Image enhancement can improve OCR accuracy

## Troubleshooting

### Common Issues

1. **"No OCR engines available"**
   - Install at least one OCR engine: `pip install pytesseract`
   - For Tesseract, ensure the binary is in your PATH

2. **Poor OCR accuracy**
   - Try different OCR engines
   - Increase image DPI
   - Enable image preprocessing

3. **Memory issues with large PDFs**
   - Reduce batch size
   - Lower image DPI
   - Process pages individually

### Debug Mode

Enable debug logging:

```env
OCR_LOG_LEVEL=DEBUG
```

Or programmatically:

```python
from src.ocr_pipeline.config import update_settings
update_settings(log_level="DEBUG")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 1.0.0
- Initial release with multi-engine OCR support
- Layout preservation capabilities
- Batch processing
- Comprehensive configuration system
- Performance monitoring and logging