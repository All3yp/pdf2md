"""Text normalization and running-header/footer helpers shared across the pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import pymupdf


def normalize_line(value: str) -> str:
	"""Collapse whitespace and normalize unicode, dropping soft hyphens."""
	value = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
	return re.sub(r"\s+", " ", value).strip()


def join_lines(lines: Iterable[str]) -> str:
	"""Join wrapped lines into a paragraph, healing end-of-line hyphenation."""
	result = ""
	for line in lines:
		line = normalize_line(line)
		if not line:
			continue
		if result.endswith("-") and len(result) > 1 and result[-2].isalnum() and line[0].islower():
			result = result[:-1] + line
		elif result:
			result += " " + line
		else:
			result = line
	return result


def extract_repeated_margins(pages: list[pymupdf.Page], *, ratio: float = 0.6) -> set[str]:
	"""Return margin lines (running headers/footers) that repeat across pages."""
	if len(pages) < 3:
		return set()

	counts: Counter[str] = Counter()
	for page in pages:
		height = page.rect.height
		margin_lines = {
			normalize_line(line)
			for block in page.get_text("blocks")
			if block[1] < height * 0.08 or block[3] > height * 0.92
			for line in block[4].splitlines()
			if normalize_line(line)
		}
		counts.update(margin_lines)
	return {line for line, count in counts.items() if count >= max(3, len(pages) * ratio)}


def frequent_xrefs(pages: list[pymupdf.Page], *, ratio: float = 0.5) -> set[int]:
	"""Return image xrefs that reappear on many pages (logotypes, watermarks)."""
	if len(pages) < 3:
		return set()

	counts: Counter[int] = Counter()
	for page in pages:
		seen: set[int] = set()
		for info in page.get_image_info(xrefs=True):
			xref = int(info.get("xref") or 0)
			if xref and xref not in seen:
				seen.add(xref)
				counts[xref] += 1
	threshold = max(3, len(pages) * ratio)
	return {xref for xref, count in counts.items() if count >= threshold}
