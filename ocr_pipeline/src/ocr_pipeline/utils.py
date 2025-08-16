"""
Utility functions for the OCR Pipeline.

Common helper functions used across the pipeline components.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Tuple
from datetime import datetime
import json
import re

from loguru import logger
from PIL import Image
import cv2
import numpy as np


def create_directory_if_not_exists(path: Union[str, Path]) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    path = Path(path)
    if not path.exists():
        logger.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_pdf_files(directory: Union[str, Path]) -> List[Path]:
    """
    Get all PDF files from a directory.
    
    Args:
        directory: Directory to search for PDF files
        
    Returns:
        List of PDF file paths
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return []
    
    # Use case-insensitive pattern and remove duplicates
    pdf_files = list(directory.glob("*.[Pp][Dd][Ff]"))
    # Remove duplicates by converting to set and back (preserves order in Python 3.7+)
    pdf_files = list(dict.fromkeys(pdf_files))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and normalizing.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines at the beginning and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    return '\n'.join(lines)


def preprocess_image(image: np.ndarray, enhance_contrast: bool = True, 
                    denoise: bool = True) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image: Input image as numpy array
        enhance_contrast: Whether to enhance contrast
        denoise: Whether to apply denoising
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    
    # Denoise
    if denoise:
        image = cv2.fastNlMeansDenoising(image)
    
    # Threshold to binary
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image array
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL format.
    
    Args:
        cv2_image: OpenCV image array
        
    Returns:
        PIL Image object
    """
    color_converted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_converted)


def save_json_metadata(metadata: dict, output_path: Union[str, Path]) -> None:
    """
    Save metadata as JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.debug(f"Metadata saved to: {output_path}")


def format_processing_time(seconds: float) -> str:
    """
    Format processing time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def estimate_pdf_type(text_content: str, page_count: int) -> str:
    """
    Estimate if PDF is digital or scanned based on extracted text.
    
    Args:
        text_content: Extracted text content
        page_count: Number of pages in PDF
        
    Returns:
        PDF type estimation: "digital", "scanned", or "mixed"
    """
    if not text_content or len(text_content.strip()) < 10:
        return "scanned"
    
    # Calculate text density (characters per page)
    text_density = len(text_content) / page_count if page_count > 0 else 0
    
    # Digital PDFs typically have high text density
    if text_density > 1000:
        return "digital"
    elif text_density > 100:
        return "mixed"
    else:
        return "scanned"


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Trim and remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    return safe_name or "output"


def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> None:
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age of files to keep (in hours)
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
    
    for file_path in temp_dir.rglob("*"):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            try:
                file_path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")