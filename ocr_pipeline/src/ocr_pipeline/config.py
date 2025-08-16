"""
Configuration management for the OCR Pipeline.

Uses Pydantic for type validation and settings management.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum


class OCREngine(str, Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    PYMUPDF = "pymupdf"  # For digital PDFs


class TextExtractionMode(str, Enum):
    """Text extraction modes for different PDF types."""
    AUTO = "auto"  # Automatically detect digital vs scanned
    DIGITAL = "digital"  # Direct text extraction
    OCR = "ocr"  # Force OCR processing
    HYBRID = "hybrid"  # Try digital first, fallback to OCR


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    pdf_input_dir: Path = Field(default="pdfs")
    ocr_output_dir: Path = Field(default="output/ocr")
    temp_dir: Path = Field(default="temp")
    log_dir: Path = Field(default="logs")
    
    # Processing settings
    ocr_engine: OCREngine = Field(default=OCREngine.TESSERACT)
    extraction_mode: TextExtractionMode = Field(default=TextExtractionMode.AUTO)
    preserve_layout: bool = Field(default=True)
    
    # OCR settings
    tesseract_config: str = Field(default="--oem 3 --psm 6")
    tesseract_lang: str = Field(default="eng")
    ocr_confidence_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    
    # Image preprocessing
    enable_image_preprocessing: bool = Field(default=True)
    image_dpi: int = Field(default=300, ge=150, le=600)
    enhance_contrast: bool = Field(default=True)
    denoise_image: bool = Field(default=True)
    
    # Output settings
    output_format: Literal["txt", "json", "both"] = Field(default="txt")
    include_metadata: bool = Field(default=True)
    include_page_numbers: bool = Field(default=True)
    
    # Performance settings
    max_workers: int = Field(default=4, ge=1, le=16)
    batch_size: int = Field(default=10, ge=1)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_to_file: bool = Field(default=True)
    log_rotation: str = Field(default="10 MB")
    
    @field_validator("pdf_input_dir", "ocr_output_dir", "temp_dir", "log_dir")
    def make_absolute_paths(cls, v, info):
        """Convert relative paths to absolute paths."""
        if isinstance(v, str):
            v = Path(v)
        
        if not v.is_absolute():
            project_root = info.data.get("project_root", Path(__file__).parent.parent.parent)
            v = project_root / v
        
        return v
    
    @field_validator("max_workers")
    def validate_max_workers(cls, v, info):
        """Ensure max_workers doesn't exceed CPU count."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if v > cpu_count:
            return cpu_count
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "OCR_"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global settings
    settings = Settings(**{**settings.model_dump(), **kwargs})
    return settings