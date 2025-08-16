"""
Main entry point for the OCR Pipeline.

Provides command-line interface and orchestrates the complete extraction process.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import time
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm

from .config import get_settings, update_settings, TextExtractionMode, OCREngine
from .text_extractor import TextExtractor
from .utils import get_pdf_files, create_directory_if_not_exists, format_processing_time, safe_filename
from .logging_config import setup_logging, get_logger, log_extraction_result


class OCRPipelineApp:
    """
    Main application class for the OCR Pipeline.
    
    Handles command-line interface, batch processing, and result output.
    """
    
    def __init__(self):
        self.console = Console()
        self.settings = get_settings()
        self.logger = get_logger("main")
        self.extractor = TextExtractor()
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the OCR pipeline with command-line arguments.
        
        Args:
            args: Command-line arguments (if None, uses sys.argv)
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Setup logging
            setup_logging()
            
            # Parse arguments
            parsed_args = self.parse_arguments(args)
            
            # Update settings from arguments
            self.update_settings_from_args(parsed_args)
            
            # Display startup information
            self.display_startup_info(parsed_args)
            
            # Get PDF files to process
            pdf_files = self.get_pdf_files(parsed_args.input_dir)
            
            if not pdf_files:
                self.console.print(f"[yellow]No PDF files found in {parsed_args.input_dir}[/yellow]")
                return 0
            
            # Process PDFs
            results = self.process_pdfs(pdf_files, parsed_args.output_dir)
            
            # Display results summary
            self.display_results_summary(results)
            
            # Save results
            if parsed_args.save_metadata:
                self.save_metadata(results, parsed_args.output_dir)
            
            self.logger.info("OCR Pipeline completed successfully")
            return 0
            
        except KeyboardInterrupt:
            self.console.print("\\n[yellow]Processing interrupted by user[/yellow]")
            return 1
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/red]")
            return 1
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="OCR Pipeline - Extract text from PDFs while preserving structure",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                                    # Process pdfs/ directory
  %(prog)s --input-dir /path/to/pdfs         # Process specific directory
  %(prog)s --output-dir /path/to/output      # Custom output directory
  %(prog)s --engine easyocr                  # Use EasyOCR engine
  %(prog)s --mode hybrid                     # Use hybrid extraction mode
  %(prog)s --dpi 400                         # High resolution OCR
  %(prog)s --workers 8                       # Use 8 parallel workers
            """
        )
        
        # Input/Output
        parser.add_argument(
            "--input-dir", "-i",
            type=Path,
            default=self.settings.pdf_input_dir,
            help=f"Input directory containing PDF files (default: {self.settings.pdf_input_dir})"
        )
        
        parser.add_argument(
            "--output-dir", "-o",
            type=Path,
            default=self.settings.ocr_output_dir,
            help=f"Output directory for extracted text (default: {self.settings.ocr_output_dir})"
        )
        
        # Processing options
        parser.add_argument(
            "--engine",
            type=str,
            choices=["tesseract", "easyocr", "paddleocr"],
            default=self.settings.ocr_engine.value,
            help=f"OCR engine to use (default: {self.settings.ocr_engine.value})"
        )
        
        parser.add_argument(
            "--mode",
            type=str,
            choices=["auto", "digital", "ocr", "hybrid"],
            default=self.settings.extraction_mode.value,
            help=f"Text extraction mode (default: {self.settings.extraction_mode.value})"
        )
        
        parser.add_argument(
            "--dpi",
            type=int,
            default=self.settings.image_dpi,
            help=f"Image DPI for OCR processing (default: {self.settings.image_dpi})"
        )
        
        parser.add_argument(
            "--confidence",
            type=float,
            default=self.settings.ocr_confidence_threshold,
            help=f"OCR confidence threshold (default: {self.settings.ocr_confidence_threshold})"
        )
        
        parser.add_argument(
            "--workers",
            type=int,
            default=self.settings.max_workers,
            help=f"Number of parallel workers (default: {self.settings.max_workers})"
        )
        
        # Output options
        parser.add_argument(
            "--no-layout",
            action="store_true",
            help="Disable layout preservation"
        )
        
        parser.add_argument(
            "--no-page-numbers",
            action="store_true",
            help="Don't include page numbers in output"
        )
        
        parser.add_argument(
            "--output-format",
            choices=["txt", "json", "both"],
            default=self.settings.output_format,
            help=f"Output format (default: {self.settings.output_format})"
        )
        
        parser.add_argument(
            "--save-metadata",
            action="store_true",
            help="Save extraction metadata to JSON files"
        )
        
        # Logging and debugging
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default=self.settings.log_level,
            help=f"Logging level (default: {self.settings.log_level})"
        )
        
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Quiet mode (minimal output)"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose mode (detailed output)"
        )
        
        # Version
        parser.add_argument(
            "--version",
            action="version",
            version="OCR Pipeline 1.0.0"
        )
        
        return parser.parse_args(args)
    
    def update_settings_from_args(self, args: argparse.Namespace) -> None:
        """Update settings based on command-line arguments."""
        updates = {}
        
        # Map arguments to settings
        if hasattr(args, 'engine'):
            updates['ocr_engine'] = OCREngine(args.engine)
        
        if hasattr(args, 'mode'):
            updates['extraction_mode'] = TextExtractionMode(args.mode)
        
        if hasattr(args, 'dpi'):
            updates['image_dpi'] = args.dpi
        
        if hasattr(args, 'confidence'):
            updates['ocr_confidence_threshold'] = args.confidence
        
        if hasattr(args, 'workers'):
            updates['max_workers'] = args.workers
        
        if hasattr(args, 'no_layout'):
            updates['preserve_layout'] = not args.no_layout
        
        if hasattr(args, 'no_page_numbers'):
            updates['include_page_numbers'] = not args.no_page_numbers
        
        if hasattr(args, 'output_format'):
            updates['output_format'] = args.output_format
        
        if hasattr(args, 'log_level'):
            updates['log_level'] = args.log_level
        
        # Apply updates
        if updates:
            update_settings(**updates)
            self.settings = get_settings()  # Refresh settings
    
    def display_startup_info(self, args: argparse.Namespace) -> None:
        """Display startup information."""
        if args.quiet:
            return
        
        self.console.print(Panel.fit(
            "[bold blue]OCR Pipeline v1.0.0[/bold blue]\\n"
            "Text extraction from PDFs with structure preservation",
            title="🔍 OCR Pipeline",
            border_style="blue"
        ))
        
        if args.verbose:
            config_table = Table(title="Configuration")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="magenta")
            
            config_table.add_row("Input Directory", str(args.input_dir))
            config_table.add_row("Output Directory", str(args.output_dir))
            config_table.add_row("OCR Engine", args.engine)
            config_table.add_row("Extraction Mode", args.mode)
            config_table.add_row("Image DPI", str(args.dpi))
            config_table.add_row("Confidence Threshold", f"{args.confidence}%")
            config_table.add_row("Workers", str(args.workers))
            config_table.add_row("Layout Preservation", str(not args.no_layout))
            
            self.console.print(config_table)
    
    def get_pdf_files(self, input_dir: Path) -> List[Path]:
        """Get list of PDF files to process."""
        pdf_files = get_pdf_files(input_dir)
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir}")
        else:
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        return pdf_files
    
    def process_pdfs(self, pdf_files: List[Path], output_dir: Path) -> List:
        """Process all PDF files and return results."""
        create_directory_if_not_exists(output_dir)
        
        results = []
        total_start_time = time.time()
        
        # Use Rich progress bar for better UX
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("Processing PDFs...", total=len(pdf_files))
            
            for pdf_file in pdf_files:
                # Update progress
                progress.update(main_task, description=f"Processing {pdf_file.name}")
                
                try:
                    # Extract text
                    result = self.extractor.extract_from_pdf(pdf_file)
                    results.append(result)
                    
                    # Save output
                    self.save_extraction_result(result, output_dir)
                    
                    # Log result
                    avg_confidence = (
                        sum(result.confidence_scores) / len(result.confidence_scores)
                        if result.confidence_scores else None
                    )
                    
                    log_extraction_result(
                        filename=result.filename,
                        pages=result.total_pages,
                        characters=len(result.extracted_text),
                        method=result.extraction_method,
                        duration=result.processing_time,
                        confidence=avg_confidence
                    )
                    
                    self.logger.info(
                        f"Processed {pdf_file.name}: "
                        f"{len(result.extracted_text)} chars, "
                        f"{result.processing_time:.2f}s, "
                        f"method: {result.extraction_method}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {pdf_file.name}: {e}")
                    # Add error result
                    from .text_extractor import ExtractionResult
                    error_result = ExtractionResult(
                        filename=pdf_file.name,
                        filepath=pdf_file,
                        total_pages=0,
                        extracted_text="",
                        extraction_method="failed",
                        processing_time=0.0,
                        metadata={},
                        page_results=[],
                        confidence_scores=[],
                        errors=[str(e)]
                    )
                    results.append(error_result)
                
                progress.advance(main_task)
        
        total_time = time.time() - total_start_time
        self.logger.info(f"Batch processing completed in {format_processing_time(total_time)}")
        
        return results
    
    def save_extraction_result(self, result, output_dir: Path) -> None:
        """Save extraction result to file(s)."""
        base_name = safe_filename(result.filename.replace('.pdf', ''))
        
        # Save text file
        if self.settings.output_format in ["txt", "both"]:
            txt_path = output_dir / f"{base_name}_extracted.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result.extracted_text)
        
        # Save JSON file
        if self.settings.output_format in ["json", "both"]:
            import json
            json_path = output_dir / f"{base_name}_result.json"
            
            result_data = {
                "filename": result.filename,
                "total_pages": result.total_pages,
                "extracted_text": result.extracted_text,
                "extraction_method": result.extraction_method,
                "processing_time": result.processing_time,
                "metadata": result.metadata,
                "page_results": result.page_results,
                "confidence_scores": result.confidence_scores,
                "errors": result.errors,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
    
    def save_metadata(self, results: List, output_dir: Path) -> None:
        """Save batch processing metadata."""
        import json
        
        metadata = {
            "batch_summary": {
                "total_files": len(results),
                "successful": len([r for r in results if r.extraction_method != "failed"]),
                "failed": len([r for r in results if r.extraction_method == "failed"]),
                "total_pages": sum(r.total_pages for r in results),
                "total_characters": sum(len(r.extracted_text) for r in results),
                "total_processing_time": sum(r.processing_time for r in results),
                "average_confidence": sum(
                    sum(r.confidence_scores) / len(r.confidence_scores)
                    for r in results if r.confidence_scores
                ) / len([r for r in results if r.confidence_scores]) if results else 0,
                "timestamp": datetime.now().isoformat()
            },
            "file_results": [
                {
                    "filename": r.filename,
                    "pages": r.total_pages,
                    "characters": len(r.extracted_text),
                    "method": r.extraction_method,
                    "processing_time": r.processing_time,
                    "errors": r.errors
                }
                for r in results
            ],
            "settings": self.settings.dict()
        }
        
        metadata_path = output_dir / "batch_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Batch metadata saved to {metadata_path}")
    
    def display_results_summary(self, results: List) -> None:
        """Display processing results summary."""
        if not results:
            return
        
        successful = [r for r in results if r.extraction_method != "failed"]
        failed = [r for r in results if r.extraction_method == "failed"]
        
        # Summary table
        summary_table = Table(title="Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Total Files", str(len(results)))
        summary_table.add_row("Successful", str(len(successful)))
        summary_table.add_row("Failed", str(len(failed)))
        summary_table.add_row("Total Pages", str(sum(r.total_pages for r in results)))
        summary_table.add_row("Total Characters", f"{sum(len(r.extracted_text) for r in results):,}")
        summary_table.add_row("Total Time", format_processing_time(sum(r.processing_time for r in results)))
        
        if successful:
            avg_time = sum(r.processing_time for r in successful) / len(successful)
            summary_table.add_row("Avg Time per File", format_processing_time(avg_time))
        
        self.console.print(summary_table)
        
        # Show failures if any
        if failed:
            self.console.print("\\n[red]Failed Files:[/red]")
            for result in failed:
                self.console.print(f"  • {result.filename}: {result.errors[0] if result.errors else 'Unknown error'}")


def main():
    """Main entry point for command-line usage."""
    app = OCRPipelineApp()
    exit_code = app.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()