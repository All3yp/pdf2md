"""Detect tables on a page and render them as GitHub-flavored Markdown tables."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class DetectedTable:
	"""A table found on a page, already rendered to Markdown."""

	bbox: tuple[float, float, float, float]
	markdown: str
	row_count: int
	col_count: int


def _cell(value: object) -> str:
	if value is None:
		return ""
	text = re.sub(r"\s+", " ", str(value)).strip()
	return text.replace("|", "\\|")


def render_markdown(rows: Sequence[Sequence[object]]) -> str:
	"""Render a matrix of cells as a GFM table (first row = header)."""
	cleaned = [[_cell(cell) for cell in row] for row in rows]
	width = max((len(row) for row in cleaned), default=0)
	if width == 0:
		return ""
	for row in cleaned:
		row.extend([""] * (width - len(row)))

	header, *body = cleaned
	lines = [
		"| " + " | ".join(header) + " |",
		"| " + " | ".join(["---"] * width) + " |",
	]
	lines.extend("| " + " | ".join(row) + " |" for row in body)
	return "\n".join(lines)


def detect_tables(
	page: pymupdf.Page,
	*,
	min_rows: int = 2,
	min_cols: int = 2,
	min_chars: int = 12,
	min_fill: float = 0.6,
) -> list[DetectedTable]:
	"""Return meaningful tables on ``page`` as Markdown, ignoring trivial hits.

	``find_tables`` also fires on figures, captions and multi-column body text;
	those are sparse (lots of empty cells), so a fill-ratio gate drops them —
	which matters because a false positive both deletes the surrounding text
	and emits a garbage table.
	"""
	try:
		finder = page.find_tables()
	except Exception:
		return []

	tables: list[DetectedTable] = []
	for table in finder.tables:
		rows = table.extract()
		if len(rows) < min_rows:
			continue
		col_count = max((len(row) for row in rows), default=0)
		if col_count < min_cols:
			continue
		cells = [cell for row in rows for cell in row]
		chars = sum(len(str(cell)) for cell in cells if cell)
		if chars < min_chars:
			continue
		filled = sum(1 for cell in cells if cell and str(cell).strip())
		if filled / max(1, len(cells)) < min_fill:
			continue
		markdown = render_markdown(rows)
		if not markdown:
			continue
		tables.append(
			DetectedTable(
				bbox=(table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3]),
				markdown=markdown,
				row_count=len(rows),
				col_count=col_count,
			)
		)
	return tables


def rect(bbox: tuple[float, float, float, float]) -> pymupdf.Rect:
	return pymupdf.Rect(*bbox)
