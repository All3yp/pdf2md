from pdf2md.text import join_lines, normalize_line


def test_join_lines_removes_line_break_hyphenation() -> None:
	assert join_lines(["inter-", "national readable text"]) == "international readable text"


def test_join_lines_keeps_hyphen_before_capital() -> None:
	assert join_lines(["Figure-", "Caption"]) == "Figure- Caption"


def test_normalize_line_collapses_whitespace_and_soft_hyphens() -> None:
	assert normalize_line("  a\u00ad  b\t c ") == "a b c"
