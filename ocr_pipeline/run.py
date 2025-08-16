#!/usr/bin/env python3
"""
Legacy OCR Pipeline Runner

This is the original run.py script, now refactored to use the new modular OCR pipeline.
For more advanced features and options, use: python -m src.ocr_pipeline.main

This script maintains compatibility with the original behavior while leveraging
the improved architecture underneath.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.ocr_pipeline import TextExtractor
    from src.ocr_pipeline.config import get_settings, update_settings
    from src.ocr_pipeline.utils import get_pdf_files, create_directory_if_not_exists, safe_filename
    from src.ocr_pipeline.logging_config import setup_logging, get_logger
except ImportError as e:
    print(f"Error importing OCR pipeline modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


# --- Configuration (Legacy compatibility) ---
# These paths match the original script for backward compatibility
PDF_INPUT_DIR = project_root / "pdfs"
OCR_OUTPUT_DIR = project_root / "output" / "ocr"


def setup_legacy_environment():
    """Set up the environment to match the original script behavior."""
    # Configure settings to match legacy behavior
    update_settings(
        pdf_input_dir=PDF_INPUT_DIR,
        ocr_output_dir=OCR_OUTPUT_DIR,
        preserve_layout=True,  # Original script used layout preservation
        include_page_numbers=True,  # Original script included page headers
        extraction_mode="digital",  # Original script was digital-only
        log_level="INFO",
        log_to_file=False  # Original script only used console output
    )
    
    # Setup simple logging for legacy compatibility
    setup_logging()
    
    return get_logger("legacy")


def process_all_pdfs_in_directory(input_dir, output_dir):
    """
    Legacy function that processes all PDFs in a directory.
    
    Now uses the new modular architecture while maintaining the same interface
    and behavior as the original script.
    """
    logger = get_logger("legacy")
    
    print("--- Starting OCR Pipeline V2 (Legacy Mode) ---")
    print("Note: This is the legacy interface. For advanced features, use:")
    print("      python -m src.ocr_pipeline.main --help")
    print()
    
    # Ensure paths are Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Get PDF files using the new utility
    pdf_files = get_pdf_files(input_dir)
    
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {input_dir}")
        print("Please create the directory and place your PDFs inside.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")
    
    # Create the text extractor
    extractor = TextExtractor()
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}...")
        
        try:
            # Extract text using the new pipeline
            result = extractor.extract_from_pdf(pdf_file)
            
            # Create output filename (matching original format)
            output_filename = safe_filename(pdf_file.stem) + "_ocr.txt"
            output_path = output_dir / output_filename
            
            # Format the output to match the original script
            formatted_output = format_legacy_output(result)
            
            # Save the text file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            
            print(f"Successfully extracted text to: {output_path}")
            print(f"  - Pages processed: {result.total_pages}")
            print(f"  - Characters extracted: {len(result.extracted_text):,}")
            print(f"  - Processing time: {result.processing_time:.2f} seconds")
            print(f"  - Method used: {result.extraction_method}")
            
            # Log any warnings
            if result.errors:
                for error in result.errors:
                    print(f"  - Warning: {error}")
            
        except Exception as e:
            print(f"[ERROR] Could not process {pdf_file.name}. Reason: {e}")
            logger.error(f"Processing failed for {pdf_file.name}: {e}", exc_info=True)
            # Print the full traceback for debugging
            import traceback
            print(f"Full error details: {traceback.format_exc()}")
    
    print("\n--- Pipeline finished. ---")
    print(f"Output files saved to: {output_dir}")


def format_legacy_output(extraction_result):
    """
    Format the extraction result to match the original script's output format.
    
    Args:
        extraction_result: ExtractionResult object from the new pipeline
        
    Returns:
        Formatted text string matching the legacy format
    """
    if not extraction_result.page_results:
        # If no page-level results, format the entire text with a simple header
        return f"==Start of OCR for {extraction_result.filename}==\n{extraction_result.extracted_text}\n==End of OCR for {extraction_result.filename}=="
    
    # Reconstruct with page headers similar to the original
    formatted_pages = []
    
    # Split the text by page if page numbers are included
    text_lines = extraction_result.extracted_text.split('\n')
    current_page_lines = []
    current_page_num = 1
    
    for line in text_lines:
        if line.strip().startswith("===== Page ") and line.strip().endswith(" ====="):
            # Found a page marker - save previous page and start new one
            if current_page_lines:
                page_text = '\n'.join(current_page_lines).strip()
                if page_text:
                    page_header = f"==Start of OCR for page {current_page_num}=="
                    page_footer = f"==End of OCR for page {current_page_num}=="
                    formatted_pages.append(f"{page_header}\n{page_text}\n{page_footer}")
                    current_page_num += 1
                current_page_lines = []
        else:
            current_page_lines.append(line)
    
    # Handle the last page
    if current_page_lines:
        page_text = '\n'.join(current_page_lines).strip()
        if page_text:
            page_header = f"==Start of OCR for page {current_page_num}=="
            page_footer = f"==End of OCR for page {current_page_num}=="
            formatted_pages.append(f"{page_header}\n{page_text}\n{page_footer}")
    
    # If no pages were found, use the entire text
    if not formatted_pages:
        page_header = f"==Start of OCR for page 1=="
        page_footer = f"==End of OCR for page 1=="
        return f"{page_header}\n{extraction_result.extracted_text}\n{page_footer}"
    
    return "\n\n".join(formatted_pages)


def create_directory_if_not_exists_legacy(path):
    """
    Legacy function for directory creation.
    
    Maintained for backward compatibility.
    """
    create_directory_if_not_exists(Path(path))
    print(f"Output directory ready: {path}")


def main():
    """Main function for the legacy script."""
    try:
        # Setup the environment
        logger = setup_legacy_environment()
        logger.info("Starting legacy OCR pipeline")
        
        # Ensure the output directory exists before we start
        create_directory_if_not_exists_legacy(OCR_OUTPUT_DIR)
        
        # Run the main processing function
        process_all_pdfs_in_directory(PDF_INPUT_DIR, OCR_OUTPUT_DIR)
        
        logger.info("Legacy OCR pipeline completed successfully")
        
    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()