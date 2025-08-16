"""
Text Extraction module for the OCR Pipeline.

Handles text extraction from both digital and scanned PDFs with fallback mechanisms.
"""

import fitz
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from loguru import logger
from PIL import Image

from .config import get_settings, TextExtractionMode
from .pdf_processor import PDFProcessor, PDFInfo
from .ocr_engine import OCREngine
from .utils import clean_text, safe_filename


@dataclass
class ExtractionResult:
    """Result of text extraction from a PDF."""
    filename: str
    filepath: Path
    total_pages: int
    extracted_text: str
    extraction_method: str
    processing_time: float
    metadata: Dict[str, Any]
    page_results: List[Dict[str, Any]]
    confidence_scores: List[float]
    errors: List[str]


class TextExtractor:
    """
    Main text extraction class that orchestrates the extraction process.
    
    Supports multiple extraction modes and fallback mechanisms.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.pdf_processor = PDFProcessor()
        self.ocr_engine = OCREngine()
        
    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from a PDF using the configured extraction mode.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractionResult object with extracted text and metadata
        """
        start_time = datetime.now()
        pdf_path = Path(pdf_path)
        
        logger.info(f"Starting text extraction: {pdf_path.name}")
        
        # Analyze PDF first
        pdf_info = self.pdf_processor.analyze_pdf(pdf_path)
        
        # Choose extraction method based on mode and PDF type
        if self.settings.extraction_mode == TextExtractionMode.AUTO:
            extraction_method = self._choose_extraction_method(pdf_info)
        else:
            extraction_method = self.settings.extraction_mode.value
        
        logger.info(f"Using extraction method: {extraction_method}")
        
        # Extract text based on chosen method
        try:
            if extraction_method == "digital":
                result = self._extract_digital(pdf_info)
            elif extraction_method == "ocr":
                result = self._extract_ocr(pdf_info)
            elif extraction_method == "hybrid":
                result = self._extract_hybrid(pdf_info)
            else:
                raise ValueError(f"Unknown extraction method: {extraction_method}")
            
            # Clean and post-process text
            result.extracted_text = clean_text(result.extracted_text)
            
            # Update timing
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.extraction_method = extraction_method
            
            logger.info(f"Text extraction completed: {pdf_path.name} "
                       f"({len(result.extracted_text)} characters, "
                       f"{result.processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed for {pdf_path.name}: {e}")
            logger.info(f"Attempting fallback to hybrid extraction method")
            
            # Try fallback to hybrid mode (digital + OCR)
            try:
                result = self._extract_hybrid(pdf_info)
                result.processing_time = (datetime.now() - start_time).total_seconds()
                result.extraction_method = f"{extraction_method}_fallback_to_hybrid"
                logger.info(f"Fallback extraction successful: {len(result.extracted_text)} characters")
                return result
            except Exception as fallback_e:
                logger.error(f"Fallback extraction also failed: {fallback_e}")
                # Return error result
                return ExtractionResult(
                    filename=pdf_path.name,
                    filepath=pdf_path,
                    total_pages=pdf_info.page_count,
                    extracted_text="",
                    extraction_method="failed",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    metadata=pdf_info.metadata,
                page_results=[],
                confidence_scores=[],
                errors=[str(e)]
            )
    
    def _choose_extraction_method(self, pdf_info: PDFInfo) -> str:
        """
        Automatically choose the best extraction method based on PDF analysis.
        
        Args:
            pdf_info: PDF information object
            
        Returns:
            Chosen extraction method
        """
        if pdf_info.estimated_type == "digital":
            return "digital"
        elif pdf_info.estimated_type == "scanned":
            return "ocr"
        else:  # mixed
            return "hybrid"
    
    def _extract_digital(self, pdf_info: PDFInfo) -> ExtractionResult:
        """
        Extract text from digital PDF using direct text extraction.
        
        Args:
            pdf_info: PDF information object
            
        Returns:
            ExtractionResult object
        """
        logger.debug(f"Extracting text digitally from: {pdf_info.filename}")
        
        doc = self.pdf_processor.open_pdf(pdf_info.filepath)
        
        try:
            extracted_text = []
            page_results = []
            
            for page_num in range(pdf_info.page_count):
                page = doc[page_num]
                
                # Extract text with layout preservation if enabled
                if self.settings.preserve_layout:
                    page_text = page.get_text("layout", sort=True)
                else:
                    page_text = page.get_text()
                
                # Add page header if enabled
                if self.settings.include_page_numbers:
                    page_header = f"\n===== Page {page_num + 1} =====\n"
                    page_text = page_header + page_text
                
                extracted_text.append(page_text)
                
                page_results.append({
                    "page_number": page_num + 1,
                    "text_length": len(page_text),
                    "method": "digital",
                    "confidence": 100.0,  # Digital extraction is always confident
                    "errors": []
                })
            
            full_text = "\n".join(extracted_text)
            
            return ExtractionResult(
                filename=pdf_info.filename,
                filepath=pdf_info.filepath,
                total_pages=pdf_info.page_count,
                extracted_text=full_text,
                extraction_method="digital",
                processing_time=0.0,  # Will be updated by caller
                metadata=pdf_info.metadata,
                page_results=page_results,
                confidence_scores=[100.0] * pdf_info.page_count,
                errors=[]
            )
            
        finally:
            doc.close()
    
    def _extract_ocr(self, pdf_info: PDFInfo) -> ExtractionResult:
        """
        Extract text using OCR from all pages.
        
        Args:
            pdf_info: PDF information object
            
        Returns:
            ExtractionResult object
        """
        logger.debug(f"Extracting text via OCR from: {pdf_info.filename}")
        
        doc = self.pdf_processor.open_pdf(pdf_info.filepath)
        
        try:
            extracted_text = []
            page_results = []
            confidence_scores = []
            errors = []
            
            for page_num in range(pdf_info.page_count):
                page = doc[page_num]
                
                try:
                    # Convert page to image
                    image = self.pdf_processor.convert_page_to_image(
                        page, dpi=self.settings.image_dpi
                    )
                    
                    # Perform OCR
                    ocr_result = self.ocr_engine.extract_text_from_image(image)
                    
                    page_text = ocr_result["text"]
                    confidence = ocr_result["confidence"]
                    
                    # Add page header if enabled
                    if self.settings.include_page_numbers:
                        page_header = f"\n===== Page {page_num + 1} =====\n"
                        page_text = page_header + page_text
                    
                    extracted_text.append(page_text)
                    confidence_scores.append(confidence)
                    
                    page_results.append({
                        "page_number": page_num + 1,
                        "text_length": len(page_text),
                        "method": "ocr",
                        "confidence": confidence,
                        "errors": []
                    })
                    
                except Exception as e:
                    error_msg = f"OCR failed for page {page_num + 1}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    
                    extracted_text.append("")
                    confidence_scores.append(0.0)
                    
                    page_results.append({
                        "page_number": page_num + 1,
                        "text_length": 0,
                        "method": "ocr",
                        "confidence": 0.0,
                        "errors": [error_msg]
                    })
            
            full_text = "\n".join(extracted_text)
            
            return ExtractionResult(
                filename=pdf_info.filename,
                filepath=pdf_info.filepath,
                total_pages=pdf_info.page_count,
                extracted_text=full_text,
                extraction_method="ocr",
                processing_time=0.0,  # Will be updated by caller
                metadata=pdf_info.metadata,
                page_results=page_results,
                confidence_scores=confidence_scores,
                errors=errors
            )
            
        finally:
            doc.close()
    
    def _extract_hybrid(self, pdf_info: PDFInfo) -> ExtractionResult:
        """
        Extract text using hybrid approach (digital first, OCR fallback).
        
        Args:
            pdf_info: PDF information object
            
        Returns:
            ExtractionResult object
        """
        logger.debug(f"Extracting text using hybrid method from: {pdf_info.filename}")
        
        doc = self.pdf_processor.open_pdf(pdf_info.filepath)
        
        try:
            extracted_text = []
            page_results = []
            confidence_scores = []
            errors = []
            
            for page_num in range(pdf_info.page_count):
                page = doc[page_num]
                page_info = pdf_info.pages[page_num]
                
                try:
                    # Try digital extraction first
                    # Updated for newer PyMuPDF versions - "layout" option no longer exists
                    if self.settings.preserve_layout:
                        digital_text = page.get_text("text", sort=True)
                    else:
                        digital_text = page.get_text()
                    
                    # Check if digital extraction was successful
                    text_density = len(digital_text.strip()) / (page_info.width * page_info.height)
                    
                    if text_density > 0.005 and len(digital_text.strip()) > 20:
                        # Digital extraction successful
                        page_text = digital_text
                        method = "digital"
                        confidence = 100.0
                        
                    else:
                        # Fall back to OCR
                        logger.debug(f"Falling back to OCR for page {page_num + 1}")
                        
                        image = self.pdf_processor.convert_page_to_image(
                            page, dpi=self.settings.image_dpi
                        )
                        
                        ocr_result = self.ocr_engine.extract_text_from_image(image)
                        page_text = ocr_result["text"]
                        method = "ocr"
                        confidence = ocr_result["confidence"]
                    
                    # Add page header if enabled
                    if self.settings.include_page_numbers:
                        page_header = f"n===== Page {page_num + 1} =====\n"
                        page_text = page_header + page_text
                    
                    extracted_text.append(page_text)
                    confidence_scores.append(confidence)
                    
                    page_results.append({
                        "page_number": page_num + 1,
                        "text_length": len(page_text),
                        "method": method,
                        "confidence": confidence,
                        "errors": []
                    })
                    
                except Exception as e:
                    error_msg = f"Hybrid extraction failed for page {page_num + 1}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    
                    extracted_text.append("")
                    confidence_scores.append(0.0)
                    
                    page_results.append({
                        "page_number": page_num + 1,
                        "text_length": 0,
                        "method": "failed",
                        "confidence": 0.0,
                        "errors": [error_msg]
                    })
            
            full_text = "\n".join(extracted_text)
            
            return ExtractionResult(
                filename=pdf_info.filename,
                filepath=pdf_info.filepath,
                total_pages=pdf_info.page_count,
                extracted_text=full_text,
                extraction_method="hybrid",
                processing_time=0.0,  # Will be updated by caller
                metadata=pdf_info.metadata,
                page_results=page_results,
                confidence_scores=confidence_scores,
                errors=errors
            )
            
        finally:
            doc.close()
    
    def extract_batch(self, pdf_paths: List[Union[str, Path]]) -> List[ExtractionResult]:
        """
        Extract text from multiple PDFs in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of ExtractionResult objects
        """
        logger.info(f"Starting batch extraction for {len(pdf_paths)} PDFs")
        
        results = []
        for pdf_path in pdf_paths:
            try:
                result = self.extract_from_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch extraction failed for {pdf_path}: {e}")
                # Add error result
                results.append(ExtractionResult(
                    filename=Path(pdf_path).name,
                    filepath=Path(pdf_path),
                    total_pages=0,
                    extracted_text="",
                    extraction_method="failed",
                    processing_time=0.0,
                    metadata={},
                    page_results=[],
                    confidence_scores=[],
                    errors=[str(e)]
                ))
        
        logger.info(f"Batch extraction completed: {len(results)} results")
        return results