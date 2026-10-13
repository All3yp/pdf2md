from pathlib import Path

import pymupdf
import pytest

from pdf2md.cli import main
from pdf2md.converter import convert_directory, convert_pdf_to_markdown


def _image_png(width: int, height: int, gray: int) -> bytes:
	pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
	pixmap.clear_with(gray)
	return pixmap.tobytes("png")


def _make_article(path: Path) -> None:
	document = pymupdf.open()
	page = document.new_page(width=400, height=600)
	page.insert_text((72, 72), "Readable article heading")
	page.insert_text((72, 96), "The article body contains readable text.")
	page.insert_image(pymupdf.Rect(40, 160, 360, 360), stream=_image_png(320, 200, 255))
	page.insert_image(pymupdf.Rect(300, 400, 372, 490), stream=_image_png(72, 90, 90))
	document.save(path)
	document.close()


def _make_table_pdf(path: Path) -> None:
	document = pymupdf.open()
	page = document.new_page(width=400, height=300)
	x0, y0, cw, ch = 50, 50, 90, 40
	for r in range(4):
		page.draw_line((x0, y0 + r * ch), (x0 + 3 * cw, y0 + r * ch))
	for c in range(4):
		page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * ch))
	for r in range(3):
		for c in range(3):
			page.insert_text((x0 + c * cw + 6, y0 + r * ch + 22), f"R{r}C{c}")
	document.save(path)
	document.close()


def test_convert_keeps_content_figure_and_drops_author_photo(tmp_path: Path) -> None:
	pdf_path = tmp_path / "article.pdf"
	_make_article(pdf_path)

	result = convert_pdf_to_markdown(pdf_path, tmp_path / "markdown", ocr=False)
	markdown = result.markdown_path.read_text(encoding="utf-8")

	assert "Readable article heading" in markdown
	assert "article body contains readable text" in markdown
	assert "![Figura" in markdown
	assert result.figures == 1
	assert "author photo" in result.skipped_reasons
	assert len(list((tmp_path / "markdown").glob("*_assets/*.png"))) == 1


def test_convert_turns_table_into_markdown_table(tmp_path: Path) -> None:
	pdf_path = tmp_path / "table.pdf"
	_make_table_pdf(pdf_path)

	result = convert_pdf_to_markdown(pdf_path, tmp_path / "markdown", ocr=False)
	markdown = result.markdown_path.read_text(encoding="utf-8")

	assert result.tables == 1
	assert "| R0C0 | R0C1 | R0C2 |" in markdown
	assert "| --- | --- | --- |" in markdown
	assert markdown.count("R0C0") == 1


def test_convert_directory_skips_empty_pdf(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	source = tmp_path / "source"
	source.mkdir()
	(source / "empty.pdf").touch()

	assert convert_directory(source, tmp_path / "markdown", ocr=False) == 0
	assert "VAZIO" in capsys.readouterr().err


def test_cli_fails_when_all_pdfs_are_empty(tmp_path: Path) -> None:
	source = tmp_path / "source"
	source.mkdir()
	(source / "empty.pdf").touch()

	with pytest.raises(SystemExit, match="1"):
		main([str(source), "--output", str(tmp_path / "markdown"), "--no-ocr"])
