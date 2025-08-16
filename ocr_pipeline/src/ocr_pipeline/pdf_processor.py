"""
PDF Processing module for the OCR Pipeline.

Handles PDF file operations, page extraction, and coordinate processing.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
from PIL import Image
import numpy as np

from .config import get_settings
from .utils import get_file_hash, estimate_pdf_type


@dataclass
class PageInfo:
    """Information about a PDF page."""
    page_number: int
    width: float
    height: float
    rotation: int
    text_content: str
    has_images: bool
    is_scanned: bool


@dataclass
class PDFInfo:
    """Information about a PDF document."""
    filename: str
    filepath: Path
    file_hash: str
    page_count: int
    metadata: Dict[str, Any]
    estimated_type: str  # "digital", "scanned", "mixed"
    processing_time: float
    pages: List[PageInfo]


class PDFProcessor:
    """
    PDF processor that handles both digital and scanned PDFs.
    
    Provides methods for text extraction, image extraction, and page analysis.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
    def open_pdf(self, pdf_path: Union[str, Path]) -> fitz.Document:
        """
        Open a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Opened PDF document
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF can't be opened
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            logger.debug(f"Opened PDF: {pdf_path}")
            return doc
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise
    
    def analyze_pdf(self, pdf_path: Union[str, Path]) -> PDFInfo:
        """
        Analyze a PDF document and extract metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDF information object
        """
        start_time = datetime.now()
        pdf_path = Path(pdf_path)
        
        logger.info(f"Analyzing PDF: {pdf_path.name}")
        
        doc = self.open_pdf(pdf_path)
        
        try:
            # Extract basic metadata
            metadata = doc.metadata
            page_count = len(doc)
            file_hash = get_file_hash(pdf_path)
            
            # Analyze each page
            pages = []
            full_text = ""
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_info = self._analyze_page(page, page_num + 1)
                pages.append(page_info)
                full_text += page_info.text_content + "\n"
            
            # Estimate PDF type
            estimated_type = estimate_pdf_type(full_text, page_count)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            pdf_info = PDFInfo(
                filename=pdf_path.name,
                filepath=pdf_path,
                file_hash=file_hash,
                page_count=page_count,
                metadata=metadata,
                estimated_type=estimated_type,
                processing_time=processing_time,
                pages=pages
            )
            
            logger.info(f"PDF analysis complete: {pdf_path.name} "
                       f"({page_count} pages, {estimated_type} type)")
            
            return pdf_info
            
        finally:
            doc.close()
    
    def _analyze_page(self, page: fitz.Page, page_number: int) -> PageInfo:
        """
        Analyze a single PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            
        Returns:
            Page information object
        """
        # Get page dimensions and rotation
        rect = page.rect
        rotation = page.rotation
        
        # Extract text content
        # Updated for newer PyMuPDF versions - "layout" option no longer exists
        if self.settings.preserve_layout:
            text_content = page.get_text("text", sort=True)
        else:
            text_content = page.get_text()
        
        # Check for images
        image_list = page.get_images()
        has_images = len(image_list) > 0
        
        # Estimate if page is scanned (low text density + images usually indicates scan)
        text_density = len(text_content.strip()) / (rect.width * rect.height) if rect.width * rect.height > 0 else 0
        is_scanned = text_density < 0.01 and has_images
        
        return PageInfo(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            rotation=rotation,
            text_content=text_content,
            has_images=has_images,
            is_scanned=is_scanned
        )
    
    def extract_text_from_page(self, page: fitz.Page, mode: str = "layout") -> str:
        """
        Extract text from a single page.
        
        Args:
            page: PyMuPDF page object
            mode: Text extraction mode ("layout", "text", "dict", "rawdict")
            
        Returns:
            Extracted text
        """
        try:
            if mode == "layout":
                return page.get_text("layout", sort=True)
            else:
                return page.get_text(mode)
        except Exception as e:
            logger.error(f"Failed to extract text from page: {e}")
            return ""
    
    def extract_images_from_page(self, page: fitz.Page, page_number: int, 
                               output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Extract images from a single page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            output_dir: Optional directory to save images
            
        Returns:
            List of image information dictionaries
        """
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    
                    image_info = {
                        "page_number": page_number,
                        "image_index": img_index,
                        "width": pix.width,
                        "height": pix.height,
                        "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                        "data": img_data
                    }
                    
                    # Save image if output directory provided
                    if output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        img_filename = f"page_{page_number:03d}_img_{img_index:03d}.png"
                        img_path = output_dir / img_filename
                        
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        
                        image_info["saved_path"] = img_path
                    
                    images.append(image_info)
                
                pix = None  # Free memory
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_number}: {e}")
        
        return images
    
    def convert_page_to_image(self, page: fitz.Page, dpi: int = 300) -> Image.Image:
        """
        Convert a PDF page to PIL Image.
        
        Args:
            page: PyMuPDF page object
            dpi: Resolution for conversion
            
        Returns:
            PIL Image object
        """
        # Calculate zoom factor from DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        pix = None  # Free memory
        return image
    
    def get_page_layout_info(self, page: fitz.Page) -> Dict[str, Any]:
        """
        Extract detailed layout information from a page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Layout information dictionary
        """
        try:
            # Get text blocks with position information
            blocks = page.get_text("dict")
            
            layout_info = {
                "page_number": page.number + 1,
                "page_rect": {
                    "x0": page.rect.x0,
                    "y0": page.rect.y0,
                    "x1": page.rect.x1,
                    "y1": page.rect.y1
                },
                "blocks": [],
                "fonts": set(),
                "font_sizes": set()
            }
            
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    block_info = {
                        "type": "text",
                        "bbox": block["bbox"],
                        "lines": []
                    }
                    
                    for line in block["lines"]:
                        line_info = {
                            "bbox": line["bbox"],
                            "spans": []
                        }
                        
                        for span in line["spans"]:
                            layout_info["fonts"].add(span["font"])
                            layout_info["font_sizes"].add(span["size"])
                            
                            span_info = {
                                "text": span["text"],
                                "bbox": span["bbox"],
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],
                                "color": span["color"]
                            }
                            line_info["spans"].append(span_info)
                        
                        block_info["lines"].append(line_info)
                    
                    layout_info["blocks"].append(block_info)
                
                else:  # Image block
                    layout_info["blocks"].append({
                        "type": "image",
                        "bbox": block["bbox"]
                    })
            
            # Convert sets to lists for JSON serialization
            layout_info["fonts"] = list(layout_info["fonts"])
            layout_info["font_sizes"] = list(layout_info["font_sizes"])
            
            return layout_info
            
        except Exception as e:
            logger.error(f"Failed to extract layout info: {e}")
            return {}


# Import required at the end to avoid circular imports
import io