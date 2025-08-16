"""
OCR Engine module for the OCR Pipeline.

Provides a unified interface for multiple OCR engines with preprocessing capabilities.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import tempfile

from loguru import logger
from PIL import Image
import numpy as np
import cv2

from .config import get_settings, OCREngine as OCREngineType
from .utils import preprocess_image, pil_to_cv2, cv2_to_pil


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    word_confidences: List[float]
    bounding_boxes: List[Tuple[int, int, int, int]]
    processing_time: float
    engine_used: str
    errors: List[str]


class OCREngine:
    """
    Unified OCR engine that supports multiple OCR backends.
    
    Supports Tesseract, EasyOCR, and PaddleOCR with automatic fallback.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._init_engines()
    
    def _init_engines(self):
        """Initialize available OCR engines."""
        self.available_engines = []
        
        # Map Tesseract language codes to other engines
        lang_mapping = {
            'eng': 'en',
            'chi_sim': 'ch_sim',
            'chi_tra': 'ch_tra',
            'fra': 'fr',
            'deu': 'de',
            'spa': 'es',
            'ita': 'it',
            'jpn': 'ja',
            'kor': 'ko',
            'ara': 'ar',
            'rus': 'ru',
            'tha': 'th',
            'vie': 'vi',
            'por': 'pt',
            'nld': 'nl',
            'pol': 'pl'
        }
        
        # Get EasyOCR/PaddleOCR compatible language code
        easyocr_lang = lang_mapping.get(self.settings.tesseract_lang, self.settings.tesseract_lang)
        
        # Initialize primary engine first (fast startup)
        primary_engine = self.settings.ocr_engine.value
        
        # Initialize Tesseract (primary)
        if primary_engine == "tesseract" or primary_engine not in ["easyocr", "paddleocr"]:
            try:
                import pytesseract
                self.tesseract = pytesseract
                self.available_engines.append("tesseract")
                logger.debug("Tesseract OCR initialized (primary)")
            except ImportError:
                logger.warning("Tesseract not available - install pytesseract")
                self.tesseract = None
        else:
            self.tesseract = None
        
        # Only initialize backup engines if primary fails
        if not self.available_engines:
            # Initialize EasyOCR as backup
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader([easyocr_lang])
                self.available_engines.append("easyocr")
                logger.debug(f"EasyOCR initialized as backup")
            except ImportError:
                logger.warning("EasyOCR not available - install easyocr")
                self.easyocr_reader = None
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.easyocr_reader = None
        else:
            self.easyocr_reader = None
        
        # Skip PaddleOCR for faster startup (causes 30+ sec delay)
        self.paddleocr = None
        logger.debug("PaddleOCR initialization skipped for faster startup")
        
        if not self.available_engines:
            logger.error("No OCR engines available! Install at least one: pytesseract, easyocr, paddleocr")
    
    def extract_text_from_image(self, image: Union[Image.Image, np.ndarray], 
                              engine: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from an image using the specified or default OCR engine.
        
        Args:
            image: PIL Image or numpy array
            engine: OCR engine to use ("tesseract", "easyocr", "paddleocr")
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.available_engines:
            raise RuntimeError("No OCR engines available")
        
        # Use default engine if not specified
        if engine is None:
            engine = self.settings.ocr_engine.value
        
        # Ensure engine is available
        if engine not in self.available_engines:
            logger.warning(f"Requested engine {engine} not available, using {self.available_engines[0]}")
            engine = self.available_engines[0]
        
        # Convert image format if needed
        if isinstance(image, np.ndarray):
            image = cv2_to_pil(image)
        
        # Preprocess image if enabled
        if self.settings.enable_image_preprocessing:
            cv2_image = pil_to_cv2(image)
            processed_image = preprocess_image(
                cv2_image,
                enhance_contrast=self.settings.enhance_contrast,
                denoise=self.settings.denoise_image
            )
            image = cv2_to_pil(processed_image)
        
        # Extract text using specified engine
        try:
            if engine == "tesseract":
                return self._extract_with_tesseract(image)
            elif engine == "easyocr":
                return self._extract_with_easyocr(image)
            elif engine == "paddleocr":
                return self._extract_with_paddleocr(image)
            else:
                raise ValueError(f"Unknown OCR engine: {engine}")
        
        except Exception as e:
            logger.error(f"OCR extraction failed with {engine}: {e}")
            # Try fallback engines
            for fallback_engine in self.available_engines:
                if fallback_engine != engine:
                    try:
                        logger.info(f"Trying fallback engine: {fallback_engine}")
                        if fallback_engine == "tesseract":
                            return self._extract_with_tesseract(image)
                        elif fallback_engine == "easyocr":
                            return self._extract_with_easyocr(image)
                        elif fallback_engine == "paddleocr":
                            return self._extract_with_paddleocr(image)
                    except Exception as fallback_error:
                        logger.warning(f"Fallback engine {fallback_engine} also failed: {fallback_error}")
                        continue
            
            # All engines failed
            return {
                "text": "",
                "confidence": 0.0,
                "word_confidences": [],
                "bounding_boxes": [],
                "processing_time": 0.0,
                "engine_used": engine,
                "errors": [str(e)]
            }
    
    def _extract_with_tesseract(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using Tesseract OCR."""
        import time
        start_time = time.time()
        
        try:
            # Get detailed data for confidence scores and bounding boxes
            data = self.tesseract.image_to_data(
                image,
                config=self.settings.tesseract_config,
                lang=self.settings.tesseract_lang,
                output_type=self.tesseract.Output.DICT
            )
            
            # Extract text
            text = self.tesseract.image_to_string(
                image,
                config=self.settings.tesseract_config,
                lang=self.settings.tesseract_lang
            )
            
            # Calculate confidence scores
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract bounding boxes for words with good confidence
            bounding_boxes = []
            word_confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > self.settings.ocr_confidence_threshold:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bounding_boxes.append((x, y, x + w, y + h))
                    word_confidences.append(float(conf))
            
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "confidence": avg_confidence,
                "word_confidences": word_confidences,
                "bounding_boxes": bounding_boxes,
                "processing_time": processing_time,
                "engine_used": "tesseract",
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise
    
    def _extract_with_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        import time
        start_time = time.time()
        
        try:
            # Convert PIL image to numpy array for EasyOCR
            img_array = np.array(image)
            
            # Run EasyOCR
            results = self.easyocr_reader.readtext(img_array)
            
            # Process results
            text_parts = []
            bounding_boxes = []
            word_confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > (self.settings.ocr_confidence_threshold / 100.0):
                    text_parts.append(text)
                    word_confidences.append(confidence * 100.0)  # Convert to percentage
                    
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            full_text = " ".join(text_parts)
            avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "word_confidences": word_confidences,
                "bounding_boxes": bounding_boxes,
                "processing_time": processing_time,
                "engine_used": "easyocr",
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            raise
    
    def _extract_with_paddleocr(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        import time
        start_time = time.time()
        
        try:
            # Convert PIL image to numpy array for PaddleOCR
            img_array = np.array(image)
            
            # Run PaddleOCR
            results = self.paddleocr.ocr(img_array, cls=True)
            
            # Process results
            text_parts = []
            bounding_boxes = []
            word_confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    
                    if confidence > (self.settings.ocr_confidence_threshold / 100.0):
                        text_parts.append(text)
                        word_confidences.append(confidence * 100.0)  # Convert to percentage
                        
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            full_text = "\\n".join(text_parts)  # PaddleOCR preserves line structure better
            avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "word_confidences": word_confidences,
                "bounding_boxes": bounding_boxes,
                "processing_time": processing_time,
                "engine_used": "paddleocr",
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            raise
    
    def extract_text_with_layout(self, image: Union[Image.Image, np.ndarray], 
                                preserve_structure: bool = True) -> Dict[str, Any]:
        """
        Extract text while preserving layout structure.
        
        Args:
            image: PIL Image or numpy array
            preserve_structure: Whether to preserve spatial structure
            
        Returns:
            Dictionary with structured text and layout information
        """
        # Use the best available engine for layout preservation
        if "paddleocr" in self.available_engines:
            engine = "paddleocr"  # PaddleOCR is best for layout
        elif "easyocr" in self.available_engines:
            engine = "easyocr"
        else:
            engine = "tesseract"
        
        result = self.extract_text_from_image(image, engine)
        
        if preserve_structure and result["bounding_boxes"]:
            # Sort text by spatial position (top to bottom, left to right)
            text_with_positions = list(zip(
                result["text"].split(),
                result["bounding_boxes"],
                result["word_confidences"]
            ))
            
            # Sort by y-coordinate first, then x-coordinate
            text_with_positions.sort(key=lambda x: (x[1][1], x[1][0]))
            
            # Reconstruct text with preserved layout
            structured_text = self._reconstruct_layout(text_with_positions)
            result["text"] = structured_text
            result["layout_preserved"] = True
        else:
            result["layout_preserved"] = False
        
        return result
    
    def _reconstruct_layout(self, text_with_positions: List[Tuple[str, Tuple[int, int, int, int], float]]) -> str:
        """
        Reconstruct text layout based on bounding box positions.
        
        Args:
            text_with_positions: List of (text, bbox, confidence) tuples
            
        Returns:
            Text with preserved layout structure
        """
        if not text_with_positions:
            return ""
        
        # Group words by approximate line (y-coordinate)
        lines = []
        current_line = []
        current_y = None
        line_threshold = 10  # Pixels tolerance for same line
        
        for text, bbox, confidence in text_with_positions:
            y_center = (bbox[1] + bbox[3]) / 2
            
            if current_y is None or abs(y_center - current_y) <= line_threshold:
                current_line.append((text, bbox, confidence))
                current_y = y_center if current_y is None else current_y
            else:
                # New line
                if current_line:
                    lines.append(current_line)
                current_line = [(text, bbox, confidence)]
                current_y = y_center
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Reconstruct text with proper spacing
        reconstructed_lines = []
        for line in lines:
            # Sort words in line by x-coordinate
            line.sort(key=lambda x: x[1][0])
            
            # Add appropriate spacing between words
            line_text = ""
            prev_x2 = None
            
            for text, bbox, confidence in line:
                x1 = bbox[0]
                
                if prev_x2 is not None:
                    # Calculate spacing based on distance
                    distance = x1 - prev_x2
                    if distance > 50:  # Large gap - multiple spaces/tab
                        line_text += "\\t"
                    elif distance > 20:  # Medium gap - multiple spaces
                        line_text += "  "
                    else:  # Small gap - single space
                        line_text += " "
                
                line_text += text
                prev_x2 = bbox[2]
            
            reconstructed_lines.append(line_text.strip())
        
        return "\\n".join(reconstructed_lines)
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        return self.available_engines.copy()
    
    def test_engine(self, engine: str) -> bool:
        """
        Test if an OCR engine is working properly.
        
        Args:
            engine: Engine name to test
            
        Returns:
            True if engine is working, False otherwise
        """
        if engine not in self.available_engines:
            return False
        
        try:
            # Create a simple test image with text
            test_image = Image.new("RGB", (200, 50), color="white")
            
            # Add some text using PIL (if available)
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(test_image)
                draw.text((10, 10), "Test OCR", fill="black")
            except:
                pass  # Text drawing failed, but we can still test OCR
            
            # Test the engine
            result = self.extract_text_from_image(test_image, engine)
            return len(result["errors"]) == 0
            
        except Exception as e:
            logger.error(f"Engine test failed for {engine}: {e}")
            return False