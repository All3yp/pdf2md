"""pdf2md - convert PDF papers to clean Markdown."""

from __future__ import annotations

from pdf2md.converter import ConversionResult, convert_directory, convert_pdf_to_markdown
from pdf2md.figures import FigureOptions
from pdf2md.vision import VisionClassifier

__version__ = "0.1.0"

__all__ = [
	"ConversionResult",
	"FigureOptions",
	"VisionClassifier",
	"convert_directory",
	"convert_pdf_to_markdown",
	"__version__",
]
