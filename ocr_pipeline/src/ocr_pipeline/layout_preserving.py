"""
Layout Preserving module for the OCR Pipeline.

Handles advanced layout analysis and text structure preservation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

from loguru import logger
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import get_settings
from .utils import create_directory_if_not_exists


class ElementType(Enum):
    """Types of document elements."""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    ANNOTATION = "annotation"


@dataclass
class TextElement:
    """A structured text element with layout information."""
    text: str
    element_type: ElementType
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    page_number: int
    reading_order: int


@dataclass
class LayoutStructure:
    """Complete layout structure of a document."""
    elements: List[TextElement]
    page_width: float
    page_height: float
    columns: int
    reading_flow: str  # "ltr", "rtl", "ttb"
    margins: Dict[str, float]


class LayoutPreserver:
    """
    Advanced layout analysis and preservation for maintaining document structure.
    
    Analyzes document layout and preserves spatial relationships between text elements.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def analyze_layout(self, page_data: Dict[str, Any], page_image: Optional[Image.Image] = None) -> LayoutStructure:
        """
        Analyze the layout structure of a page.
        
        Args:
            page_data: Page data from PDF processor or OCR engine
            page_image: Optional page image for visual analysis
            
        Returns:
            LayoutStructure object with analyzed layout
        """
        logger.debug("Analyzing page layout structure")
        
        # Extract basic page dimensions
        if "page_rect" in page_data:
            page_width = page_data["page_rect"]["x1"] - page_data["page_rect"]["x0"]
            page_height = page_data["page_rect"]["y1"] - page_data["page_rect"]["y0"]
        elif page_image:
            page_width, page_height = page_image.size
        else:
            page_width, page_height = 612, 792  # Default letter size
        
        # Analyze text elements
        elements = self._extract_text_elements(page_data)
        
        # Classify element types
        elements = self._classify_elements(elements, page_width, page_height)
        
        # Determine reading order
        elements = self._determine_reading_order(elements)
        
        # Analyze columns and flow
        columns = self._detect_columns(elements, page_width)
        reading_flow = self._detect_reading_flow(elements)
        
        # Calculate margins
        margins = self._calculate_margins(elements, page_width, page_height)
        
        return LayoutStructure(
            elements=elements,
            page_width=page_width,
            page_height=page_height,
            columns=columns,
            reading_flow=reading_flow,
            margins=margins
        )
    
    def _extract_text_elements(self, page_data: Dict[str, Any]) -> List[TextElement]:
        """Extract text elements from page data."""
        elements = []
        
        if "blocks" in page_data:
            # PDF processor data format
            for block in page_data.get("blocks", []):
                if block.get("type") == "text":
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            element = TextElement(
                                text=span.get("text", ""),
                                element_type=ElementType.PARAGRAPH,  # Will be classified later
                                bbox=tuple(span.get("bbox", [0, 0, 0, 0])),
                                confidence=100.0,  # PDF text is always confident
                                font_size=span.get("size", 12.0),
                                font_name=span.get("font", ""),
                                is_bold=bool(span.get("flags", 0) & 2**4),
                                is_italic=bool(span.get("flags", 0) & 2**1),
                                page_number=page_data.get("page_number", 1),
                                reading_order=0  # Will be determined later
                            )
                            elements.append(element)
        
        elif "bounding_boxes" in page_data:
            # OCR engine data format
            texts = page_data.get("text", "").split()
            bboxes = page_data.get("bounding_boxes", [])
            confidences = page_data.get("word_confidences", [])
            
            for i, (text, bbox) in enumerate(zip(texts, bboxes)):
                confidence = confidences[i] if i < len(confidences) else 0.0
                
                element = TextElement(
                    text=text,
                    element_type=ElementType.PARAGRAPH,  # Will be classified later
                    bbox=bbox,
                    confidence=confidence,
                    font_size=12.0,  # Default, could be estimated from bbox height
                    font_name="unknown",
                    is_bold=False,
                    is_italic=False,
                    page_number=page_data.get("page_number", 1),
                    reading_order=0  # Will be determined later
                )
                elements.append(element)
        
        return elements
    
    def _classify_elements(self, elements: List[TextElement], page_width: float, page_height: float) -> List[TextElement]:
        """Classify text elements by type based on formatting and position."""
        
        # Calculate font size statistics
        font_sizes = [elem.font_size for elem in elements if elem.font_size > 0]
        if font_sizes:
            avg_font_size = np.mean(font_sizes)
            large_font_threshold = avg_font_size * 1.5
            small_font_threshold = avg_font_size * 0.8
        else:
            avg_font_size = 12.0
            large_font_threshold = 18.0
            small_font_threshold = 10.0
        
        # Classify each element
        for element in elements:
            # Header/Footer detection (top 10% or bottom 10% of page)
            y_center = (element.bbox[1] + element.bbox[3]) / 2
            if y_center < page_height * 0.1:
                element.element_type = ElementType.HEADER
            elif y_center > page_height * 0.9:
                element.element_type = ElementType.FOOTER
            
            # Title detection (large font, near top, centered)
            elif (element.font_size > large_font_threshold and 
                  y_center < page_height * 0.3 and
                  self._is_centered(element, page_width)):
                element.element_type = ElementType.TITLE
            
            # Heading detection (large font or bold)
            elif (element.font_size > large_font_threshold or 
                  element.is_bold):
                element.element_type = ElementType.HEADING
            
            # List item detection (starts with bullet or number)
            elif self._is_list_item(element.text):
                element.element_type = ElementType.LIST_ITEM
            
            # Small text might be captions or annotations
            elif element.font_size < small_font_threshold:
                if self._is_near_image_area(element, elements):
                    element.element_type = ElementType.CAPTION
                else:
                    element.element_type = ElementType.ANNOTATION
            
            # Default to paragraph
            else:
                element.element_type = ElementType.PARAGRAPH
        
        return elements
    
    def _is_centered(self, element: TextElement, page_width: float, tolerance: float = 0.1) -> bool:
        """Check if an element is horizontally centered."""
        element_center = (element.bbox[0] + element.bbox[2]) / 2
        page_center = page_width / 2
        return abs(element_center - page_center) / page_width < tolerance
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text appears to be a list item."""
        text = text.strip()
        
        # Check for bullet points
        bullet_chars = ["•", "·", "◦", "▪", "▫", "‣", "⁃"]
        if any(text.startswith(char) for char in bullet_chars):
            return True
        
        # Check for numbered lists
        import re
        number_patterns = [
            r"^\d+\.",  # 1. 2. 3.
            r"^\d+\)",  # 1) 2) 3)
            r"^[a-zA-Z]\.",  # a. b. c.
            r"^[a-zA-Z]\)",  # a) b) c)
            r"^[ivxlcdm]+\.",  # i. ii. iii. (roman numerals)
        ]
        
        for pattern in number_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_near_image_area(self, element: TextElement, all_elements: List[TextElement]) -> bool:
        """Check if element is near an image area (simplified heuristic)."""
        # This is a simplified implementation
        # In a more advanced version, you'd check against actual image locations
        
        # Look for isolated small text (potential captions)
        nearby_elements = [
            elem for elem in all_elements 
            if elem != element and self._elements_nearby(element, elem, threshold=50)
        ]
        
        # If very few nearby elements, might be a caption
        return len(nearby_elements) < 3
    
    def _elements_nearby(self, elem1: TextElement, elem2: TextElement, threshold: float = 30) -> bool:
        """Check if two elements are spatially close."""
        # Calculate distance between element centers
        center1 = ((elem1.bbox[0] + elem1.bbox[2]) / 2, (elem1.bbox[1] + elem1.bbox[3]) / 2)
        center2 = ((elem2.bbox[0] + elem2.bbox[2]) / 2, (elem2.bbox[1] + elem2.bbox[3]) / 2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < threshold
    
    def _determine_reading_order(self, elements: List[TextElement]) -> List[TextElement]:
        """Determine the reading order of elements."""
        
        # Sort by spatial position (top-to-bottom, left-to-right for LTR languages)
        def sort_key(elem):
            # Primary sort: y-coordinate (top to bottom)
            # Secondary sort: x-coordinate (left to right)
            return (elem.bbox[1], elem.bbox[0])
        
        elements.sort(key=sort_key)
        
        # Assign reading order numbers
        for i, element in enumerate(elements):
            element.reading_order = i
        
        return elements
    
    def _detect_columns(self, elements: List[TextElement], page_width: float) -> int:
        """Detect the number of columns in the layout."""
        if not elements:
            return 1
        
        # Group elements by approximate x-position
        x_positions = [(elem.bbox[0] + elem.bbox[2]) / 2 for elem in elements]
        
        # Use clustering to find column centers
        from sklearn.cluster import KMeans
        
        # Try different numbers of columns (1-4)
        best_columns = 1
        best_score = float('inf')
        
        for n_cols in range(1, min(5, len(set(x_positions)) + 1)):
            try:
                kmeans = KMeans(n_clusters=n_cols, random_state=42, n_init=10)
                labels = kmeans.fit_predict(np.array(x_positions).reshape(-1, 1))
                score = kmeans.inertia_
                
                if score < best_score:
                    best_score = score
                    best_columns = n_cols
            except:
                continue
        
        return best_columns
    
    def _detect_reading_flow(self, elements: List[TextElement]) -> str:
        """Detect the reading flow direction."""
        # For now, assume left-to-right (could be enhanced for RTL languages)
        return "ltr"
    
    def _calculate_margins(self, elements: List[TextElement], page_width: float, page_height: float) -> Dict[str, float]:
        """Calculate page margins based on content placement."""
        if not elements:
            return {"top": 0, "bottom": 0, "left": 0, "right": 0}
        
        # Find content boundaries
        min_x = min(elem.bbox[0] for elem in elements)
        max_x = max(elem.bbox[2] for elem in elements)
        min_y = min(elem.bbox[1] for elem in elements)
        max_y = max(elem.bbox[3] for elem in elements)
        
        return {
            "top": min_y,
            "bottom": page_height - max_y,
            "left": min_x,
            "right": page_width - max_x
        }
    
    def reconstruct_document_structure(self, layout: LayoutStructure) -> str:
        """
        Reconstruct text with preserved document structure.
        
        Args:
            layout: LayoutStructure object
            
        Returns:
            Structured text representation
        """
        # Group elements by type and reading order
        grouped_elements = {}
        for element in sorted(layout.elements, key=lambda x: x.reading_order):
            elem_type = element.element_type
            if elem_type not in grouped_elements:
                grouped_elements[elem_type] = []
            grouped_elements[elem_type].append(element)
        
        # Reconstruct with structure markers
        structured_text = []
        
        # Add headers
        if ElementType.HEADER in grouped_elements:
            structured_text.append("=== HEADER ===")
            for elem in grouped_elements[ElementType.HEADER]:
                structured_text.append(elem.text)
            structured_text.append("")
        
        # Add title
        if ElementType.TITLE in grouped_elements:
            for elem in grouped_elements[ElementType.TITLE]:
                structured_text.append(f"# {elem.text}")
            structured_text.append("")
        
        # Add main content (headings, paragraphs, lists)
        content_types = [ElementType.HEADING, ElementType.PARAGRAPH, ElementType.LIST_ITEM]
        
        for element in sorted(layout.elements, key=lambda x: x.reading_order):
            if element.element_type in content_types:
                if element.element_type == ElementType.HEADING:
                    structured_text.append(f"## {element.text}")
                elif element.element_type == ElementType.LIST_ITEM:
                    structured_text.append(f"  • {element.text}")
                else:  # PARAGRAPH
                    structured_text.append(element.text)
        
        # Add captions and annotations
        if ElementType.CAPTION in grouped_elements:
            structured_text.append("")
            structured_text.append("=== CAPTIONS ===")
            for elem in grouped_elements[ElementType.CAPTION]:
                structured_text.append(f"[Caption: {elem.text}]")
        
        # Add footer
        if ElementType.FOOTER in grouped_elements:
            structured_text.append("")
            structured_text.append("=== FOOTER ===")
            for elem in grouped_elements[ElementType.FOOTER]:
                structured_text.append(elem.text)
        
        return "\\n".join(structured_text)
    
    def export_layout_analysis(self, layout: LayoutStructure, output_path: Path) -> None:
        """
        Export layout analysis to JSON file.
        
        Args:
            layout: LayoutStructure object
            output_path: Path to save the analysis
        """
        create_directory_if_not_exists(output_path.parent)
        
        # Convert to serializable format
        analysis_data = {
            "page_dimensions": {
                "width": layout.page_width,
                "height": layout.page_height
            },
            "layout_properties": {
                "columns": layout.columns,
                "reading_flow": layout.reading_flow,
                "margins": layout.margins
            },
            "elements": [
                {
                    "text": elem.text,
                    "type": elem.element_type.value,
                    "bbox": elem.bbox,
                    "confidence": elem.confidence,
                    "font_size": elem.font_size,
                    "font_name": elem.font_name,
                    "is_bold": elem.is_bold,
                    "is_italic": elem.is_italic,
                    "page_number": elem.page_number,
                    "reading_order": elem.reading_order
                }
                for elem in layout.elements
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Layout analysis exported to: {output_path}")
    
    def visualize_layout(self, layout: LayoutStructure, page_image: Image.Image, 
                        output_path: Optional[Path] = None) -> Image.Image:
        """
        Create a visual representation of the layout analysis.
        
        Args:
            layout: LayoutStructure object
            page_image: Original page image
            output_path: Optional path to save the visualization
            
        Returns:
            Image with layout overlay
        """
        # Create a copy of the image for drawing
        viz_image = page_image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        # Color mapping for different element types
        colors = {
            ElementType.TITLE: "red",
            ElementType.HEADING: "blue",
            ElementType.PARAGRAPH: "green",
            ElementType.LIST_ITEM: "orange",
            ElementType.CAPTION: "purple",
            ElementType.HEADER: "cyan",
            ElementType.FOOTER: "magenta",
            ElementType.ANNOTATION: "yellow"
        }
        
        # Draw bounding boxes for each element
        for element in layout.elements:
            color = colors.get(element.element_type, "black")
            
            # Draw bounding box
            draw.rectangle(element.bbox, outline=color, width=2)
            
            # Add label
            label = f"{element.element_type.value}_{element.reading_order}"
            draw.text((element.bbox[0], element.bbox[1] - 15), label, fill=color)
        
        # Save if path provided
        if output_path:
            create_directory_if_not_exists(output_path.parent)
            viz_image.save(output_path)
            logger.info(f"Layout visualization saved to: {output_path}")
        
        return viz_image


# Try to import sklearn for clustering, provide fallback if not available
try:
    from sklearn.cluster import KMeans
except ImportError:
    logger.warning("scikit-learn not available - column detection will be simplified")
    
    class KMeans:
        def __init__(self, n_clusters, random_state=None, n_init=10):
            self.n_clusters = n_clusters
            self.inertia_ = 0
        
        def fit_predict(self, X):
            # Simple fallback: just return cluster 0 for all points
            return [0] * len(X)