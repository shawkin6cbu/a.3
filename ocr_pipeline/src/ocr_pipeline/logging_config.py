"""
Logging configuration for the OCR Pipeline.

Provides structured logging with file rotation and console output.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings
from .utils import create_directory_if_not_exists


class LoggingConfig:
    """
    Centralized logging configuration for the OCR Pipeline.
    
    Provides both file and console logging with structured formats.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.console = Console()
        self._configured = False
    
    def setup_logging(self) -> None:
        """
        Set up logging configuration with file and console handlers.
        """
        if self._configured:
            return
        
        # Remove default logger
        logger.remove()
        
        # Create log directory
        create_directory_if_not_exists(self.settings.log_dir)
        
        # Console logging with Rich
        if self.settings.log_level in ["DEBUG", "INFO"]:
            logger.add(
                RichHandler(
                    console=self.console,
                    show_time=True,
                    show_path=False,
                    markup=True,
                    rich_tracebacks=True
                ),
                level=self.settings.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True
            )
        else:
            # Simple console for WARNING/ERROR
            logger.add(
                sys.stderr,
                level=self.settings.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                colorize=False
            )
        
        # File logging if enabled
        if self.settings.log_to_file:
            log_file = self.settings.log_dir / "ocr_pipeline.log"
            
            logger.add(
                log_file,
                level="DEBUG",  # Always log everything to file
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
                rotation=self.settings.log_rotation,
                retention="1 month",
                compression="zip",
                encoding="utf-8",
                enqueue=True,  # Thread-safe logging
                serialize=False
            )
            
            # Separate error log
            error_log_file = self.settings.log_dir / "errors.log"
            logger.add(
                error_log_file,
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
                rotation="100 MB",
                retention="3 months",
                compression="zip",
                encoding="utf-8",
                enqueue=True
            )
        
        # Performance log for timing information
        perf_log_file = self.settings.log_dir / "performance.log"
        logger.add(
            perf_log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            filter=lambda record: "performance" in record["extra"],
            rotation="50 MB",
            retention="1 month",
            encoding="utf-8"
        )
        
        self._configured = True
        logger.info("Logging system initialized")
    
    def get_logger(self, name: Optional[str] = None) -> "loguru.Logger":
        """
        Get a logger instance with optional name.
        
        Args:
            name: Logger name (defaults to caller's module)
            
        Returns:
            Configured logger instance
        """
        if not self._configured:
            self.setup_logging()
        
        if name:
            return logger.bind(name=name)
        else:
            return logger
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional metadata
        """
        metadata = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        logger.bind(performance=True, **metadata).info(
            f"PERF | {operation} | {duration:.3f}s | {kwargs}"
        )
    
    def log_extraction_result(self, filename: str, pages: int, characters: int, 
                            method: str, duration: float, confidence: float = None) -> None:
        """
        Log extraction results for analytics.
        
        Args:
            filename: PDF filename
            pages: Number of pages processed
            characters: Number of characters extracted
            method: Extraction method used
            duration: Processing duration
            confidence: Average confidence score
        """
        metadata = {
            "filename": filename,
            "pages": pages,
            "characters": characters,
            "method": method,
            "duration_seconds": duration,
            "characters_per_second": characters / duration if duration > 0 else 0,
            "pages_per_second": pages / duration if duration > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if confidence is not None:
            metadata["avg_confidence"] = confidence
        
        logger.bind(extraction_result=True, **metadata).info(
            f"EXTRACTION | {filename} | {pages}p | {characters}c | {method} | {duration:.2f}s"
        )
    
    def log_error_with_context(self, error: Exception, context: dict) -> None:
        """
        Log an error with additional context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        logger.bind(**context).error(
            f"Error occurred: {type(error).__name__}: {error}",
            exc_info=error
        )


# Global logging instance
_logging_config = None


def get_logger(name: Optional[str] = None) -> "loguru.Logger":
    """
    Get a configured logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        Configured logger
    """
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    return _logging_config.get_logger(name)


def setup_logging() -> None:
    """Initialize the logging system."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    _logging_config.setup_logging()


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    _logging_config.log_performance(operation, duration, **kwargs)


def log_extraction_result(filename: str, pages: int, characters: int, 
                         method: str, duration: float, confidence: float = None) -> None:
    """Log extraction results."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    _logging_config.log_extraction_result(filename, pages, characters, method, duration, confidence)


def log_error_with_context(error: Exception, context: dict) -> None:
    """Log error with context."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    
    _logging_config.log_error_with_context(error, context)