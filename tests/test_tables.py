import pymupdf

from pdf2md.tables import detect_tables, render_markdown


def _draw_table(
	page: pymupdf.Page, x0: int, y0: int, cols: int, rows: int, cw: int = 90, ch: int = 40
) -> None:
	for r in range(rows + 1):
		page.draw_line((x0, y0 + r * ch), (x0 + cols * cw, y0 + r * ch))
	for c in range(cols + 1):
		page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * ch))
	for r in range(rows):
		for c in range(cols):
			page.insert_text((x0 + c * cw + 6, y0 + r * ch + 22), f"R{r}C{c}")


def test_render_markdown_basic() -> None:
	markdown = render_markdown([["A", "B"], ["1", "2"]])
	assert markdown.splitlines() == ["| A | B |", "| --- | --- |", "| 1 | 2 |"]


def test_render_markdown_escapes_pipes_and_pads_ragged_rows() -> None:
	markdown = render_markdown([["a|b", "c"], ["d"]])
	assert "a\\|b" in markdown
	assert markdown.splitlines()[2] == "| d |  |"


def test_detect_tables_finds_drawn_grid() -> None:
	document = pymupdf.open()
	page = document.new_page(width=400, height=300)
	_draw_table(page, 50, 50, cols=3, rows=3)

	tables = detect_tables(page)
	document.close()

	assert len(tables) == 1
	assert tables[0].row_count == 3
	assert tables[0].col_count == 3
	assert "| R0C0 | R0C1 | R0C2 |" in tables[0].markdown


def test_detect_tables_rejects_sparse_grid() -> None:
	document = pymupdf.open()
	page = document.new_page(width=400, height=300)
	x0, y0, cw, ch = 40, 40, 60, 40
	for r in range(6):
		page.draw_line((x0, y0 + r * ch), (x0 + 5 * cw, y0 + r * ch))
	for c in range(6):
		page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 5 * ch))
	page.insert_text((x0 + 5, y0 + 20), "only one filled cell")

	tables = detect_tables(page)
	document.close()

	assert tables == []
