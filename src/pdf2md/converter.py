"""Turn a PDF paper into Markdown: inline tables, filtered figures."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from pdf2md.figures import (
	FigureDecision,
	FigureOptions,
	cluster_flags,
	content_verdict,
	geometry_rejection,
	image_placements,
	image_stats,
	render_pixmap,
)
from pdf2md.tables import detect_tables
from pdf2md.text import extract_repeated_margins, frequent_xrefs, join_lines, normalize_line
from pdf2md.vision import VisionClassifier

SAVE_DPI = 220
STATS_DPI = 72
VISION_DPI = 120

_PDF_ERRORS = (
	OSError,
	pymupdf.FileDataError,
	getattr(pymupdf, "EmptyFileError", pymupdf.FileDataError),
)


@dataclass
class ConversionResult:
	"""Summary of converting a single PDF."""

	markdown_path: Path
	figures: int = 0
	tables: int = 0
	skipped: int = 0
	skipped_reasons: dict[str, int] = field(default_factory=dict)


def _page_text(page: pymupdf.Page, ocr: bool, ocr_lang: str) -> tuple[str, bool]:
	text = page.get_text("text", sort=True)
	if ocr and len(re.sub(r"\W", "", text)) < 40:
		try:
			textpage = page.get_textpage_ocr(language=ocr_lang, dpi=300, full=True)
			return page.get_text("text", textpage=textpage, sort=True), True
		except RuntimeError:
			print(
				"Aviso: OCR indisponível na página "
				f"{page.number + 1}; mantendo o conteúdo extraível.",
				file=sys.stderr,
			)
	return text, False


def _text_blocks(page: pymupdf.Page, text: str, ocr_used: bool) -> list[dict[str, Any]]:
	blocks = page.get_text("dict", sort=True)["blocks"]
	if not ocr_used:
		return [block for block in blocks if block.get("type") == 0]
	return [
		{
			"type": 0,
			"bbox": (0, 0, page.rect.width, page.rect.height),
			"lines": [
				{"spans": [{"text": line}]}
				for line in text.splitlines()
				if normalize_line(line)
			],
		}
	]


def _block_paragraph(block: dict[str, Any], repeated_lines: set[str]) -> str:
	line_values: list[str] = []
	for line in block.get("lines", []):
		value = join_lines([span.get("text", "") for span in line.get("spans", [])])
		if value and value not in repeated_lines and not re.fullmatch(r"\d+", value):
			line_values.append(value)
	return join_lines(line_values)


def _mostly_inside_any(
	bbox: pymupdf.Rect, rects: list[pymupdf.Rect], fraction: float = 0.5
) -> bool:
	area = bbox.get_area()
	if area <= 0:
		return False
	return any((bbox & rect).get_area() > fraction * area for rect in rects)


def _document_title(document: pymupdf.Document, pdf_path: Path) -> str:
	metadata_title = normalize_line(document.metadata.get("title") or "")
	fallback_title = re.sub(r"^\d+[_-]?", "", pdf_path.stem).replace("_", " ")
	has_title = metadata_title and metadata_title.lower() != "untitled"
	title = metadata_title if has_title else fallback_title
	return title.strip() or pdf_path.stem


def _classify(
	placement: Any,
	page: pymupdf.Page,
	*,
	table_rects: list[pymupdf.Rect],
	repeated_xrefs: set[int],
	clustered: bool,
	options: FigureOptions,
	vision: VisionClassifier | None,
) -> FigureDecision:
	reason = geometry_rejection(
		placement,
		page,
		table_rects=table_rects,
		repeated_xrefs=repeated_xrefs,
		options=options,
	)
	if reason:
		return FigureDecision(keep=False, reason=reason)

	stats = image_stats(render_pixmap(page, placement.bbox, STATS_DPI))
	decision = content_verdict(placement, stats, clustered=clustered, options=options)
	if not decision.ambiguous:
		return decision

	if vision is not None:
		png = render_pixmap(page, placement.bbox, VISION_DPI).tobytes("png")
		is_junk, label = vision.classify(png)
		if is_junk:
			return FigureDecision(keep=False, reason=f"vision:{label}", stats=stats)
		return FigureDecision(keep=True, stats=stats)
	return FigureDecision(keep=True, stats=stats)


def convert_pdf_to_markdown(
	pdf_path: Path,
	output_dir: Path,
	*,
	ocr: bool = True,
	ocr_lang: str = "eng",
	options: FigureOptions | None = None,
	vision: VisionClassifier | None = None,
	trace: bool = False,
) -> ConversionResult:
	"""Convert one PDF, writing ``<stem>.md`` and figure crops under ``<stem>_assets/``."""
	options = options or FigureOptions()
	pdf_path = Path(pdf_path)
	output_dir = Path(output_dir)
	markdown_path = output_dir / f"{pdf_path.stem}.md"
	assets_dir = output_dir / f"{pdf_path.stem}_assets"

	result = ConversionResult(markdown_path=markdown_path)

	with pymupdf.open(pdf_path) as document:
		pages = list(document)
		repeated_lines = extract_repeated_margins(pages)
		repeated_xrefs = frequent_xrefs(pages)
		lines_out = [f"# {_document_title(document, pdf_path)}\n"]
		image_count = 0

		for page in pages:
			tables = detect_tables(page)
			table_rects = [pymupdf.Rect(*table.bbox) for table in tables]
			result.tables += len(tables)

			placements = image_placements(page)
			clustered = cluster_flags(placements, options)
			text, ocr_used = _page_text(page, ocr, ocr_lang)

			elements: list[tuple[float, float, str]] = []
			for block in _text_blocks(page, text, ocr_used):
				bbox = pymupdf.Rect(*block["bbox"])
				if _mostly_inside_any(bbox, table_rects):
					continue
				paragraph = _block_paragraph(block, repeated_lines)
				if paragraph:
					elements.append((bbox.y0, bbox.x0, paragraph + "\n"))
			for table in tables:
				elements.append((table.bbox[1], table.bbox[0], table.markdown + "\n"))

			for index, placement in enumerate(placements):
				decision = _classify(
					placement,
					page,
					table_rects=table_rects,
					repeated_xrefs=repeated_xrefs,
					clustered=clustered[index],
					options=options,
					vision=vision,
				)
				if not decision.keep:
					result.skipped += 1
					reason = decision.reason or "ambiguous"
					result.skipped_reasons[reason] = result.skipped_reasons.get(reason, 0) + 1
					if trace and decision.reason:
						print(
							f"  página {page.number + 1}: ignorado ({decision.reason})",
							file=sys.stderr,
						)
					continue
				image_count += 1
				name = f"page-{page.number + 1:03d}-figure-{image_count:02d}.png"
				assets_dir.mkdir(parents=True, exist_ok=True)
				png = render_pixmap(page, placement.bbox, SAVE_DPI).tobytes("png")
				(assets_dir / name).write_bytes(png)
				elements.append(
					(
						placement.bbox[1],
						placement.bbox[0],
						f"\n![Figura da página {page.number + 1}]({assets_dir.name}/{name})\n",
					)
				)

			elements.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
			lines_out.extend(payload for _, _, payload in elements)

		result.figures = image_count

	output_dir.mkdir(parents=True, exist_ok=True)
	markdown_path.write_text("\n".join(lines_out).strip() + "\n", encoding="utf-8")
	return result


def convert_directory(
	input_dir: Path,
	output_dir: Path,
	*,
	ocr: bool = True,
	ocr_lang: str = "eng",
	options: FigureOptions | None = None,
	vision: VisionClassifier | None = None,
	trace: bool = False,
) -> int:
	"""Convert every PDF under ``input_dir`` recursively; skip bad files and continue."""
	pdf_paths = sorted(Path(input_dir).rglob("*.pdf"))
	converted = 0
	empty = 0
	invalid = 0
	for pdf_path in pdf_paths:
		if pdf_path.stat().st_size == 0:
			empty += 1
			print(f"VAZIO {pdf_path}: arquivo tem 0 bytes; baixe o PDF novamente.", file=sys.stderr)
			continue
		try:
			result = convert_pdf_to_markdown(
				pdf_path,
				output_dir,
				ocr=ocr,
				ocr_lang=ocr_lang,
				options=options,
				vision=vision,
				trace=trace,
			)
			converted += 1
			print(
				f"OK  {pdf_path} -> {result.markdown_path} "
				f"[{result.tables} tabelas, {result.figures} figuras, {result.skipped} ignoradas]"
			)
		except _PDF_ERRORS as error:
			invalid += 1
			print(f"ERRO {pdf_path}: {error}", file=sys.stderr)
	print(f"Convertidos: {converted}/{len(pdf_paths)} PDFs ({empty} vazios, {invalid} inválidos)")
	return converted
