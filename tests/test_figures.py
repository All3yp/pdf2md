import pymupdf

from pdf2md.figures import (
	FigureOptions,
	ImagePlacement,
	cluster_flags,
	content_verdict,
	geometry_rejection,
	image_stats,
)


def _pixmap(width: int, height: int, gray: int) -> pymupdf.Pixmap:
	pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
	pixmap.clear_with(gray)
	return pixmap


def _white_with_color_patch(width: int, height: int) -> pymupdf.Pixmap:
	"""A white-background portrait with a colorful center (like a studio headshot)."""
	pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
	pixmap.clear_with(255)
	for y in range(height // 5, height * 4 // 5):
		for x in range(width // 5, width * 4 // 5):
			pixmap.set_pixel(x, y, (200, 120, 90))
	return pixmap


def _placement(x0: float, y0: float, x1: float, y1: float, xref: int = 1) -> ImagePlacement:
	return ImagePlacement(bbox=(x0, y0, x1, y1), xref=xref, width=int(x1 - x0), height=int(y1 - y0))


def _geometry(
	placement: ImagePlacement, page: pymupdf.Page, *, repeated: set[int] | None = None
) -> str | None:
	return geometry_rejection(
		placement,
		page,
		table_rects=[],
		repeated_xrefs=repeated or set(),
		options=FigureOptions(),
	)


def test_image_stats_white_vs_dark() -> None:
	white = image_stats(_pixmap(40, 40, 255))
	dark = image_stats(_pixmap(40, 40, 90))

	assert white.white_fraction == 1.0
	assert white.colorful_fraction == 0.0
	assert dark.white_fraction == 0.0
	assert dark.mean_saturation == 0.0


def test_geometry_rejection_drops_tiny_and_full_page() -> None:
	page = pymupdf.open().new_page(width=400, height=600)

	assert _geometry(_placement(10, 10, 20, 20), page) is not None
	assert _geometry(_placement(0, 0, 400, 600), page) is not None
	assert _geometry(_placement(50, 50, 350, 250), page) is None


def test_geometry_rejection_drops_repeated_xref() -> None:
	page = pymupdf.open().new_page(width=400, height=600)

	assert _geometry(_placement(50, 50, 350, 250, xref=7), page, repeated={7}) is not None


def test_content_verdict_flags_small_portrait_photo() -> None:
	placement = _placement(0, 0, 72, 90)
	stats = image_stats(_pixmap(72, 90, 90))

	decision = content_verdict(placement, stats, clustered=False, options=FigureOptions())
	assert decision.keep is False
	assert decision.reason == "author photo"


def test_content_verdict_flags_white_background_portrait_photo() -> None:
	"""A studio headshot on a white background is still photographic, not a chart."""
	placement = _placement(0, 0, 72, 90)
	stats = image_stats(_white_with_color_patch(72, 90))

	assert stats.white_fraction >= FigureOptions().photo_white_max
	decision = content_verdict(placement, stats, clustered=False, options=FigureOptions())
	assert decision.keep is False
	assert decision.reason == "author photo"


def test_content_verdict_keeps_line_art_figure() -> None:
	placement = _placement(0, 0, 320, 200)
	stats = image_stats(_pixmap(320, 200, 255))

	decision = content_verdict(placement, stats, clustered=False, options=FigureOptions())
	assert decision.keep is True


def test_cluster_flags_similar_portrait_images() -> None:
	placements = [
		_placement(0, 0, 72, 90),
		_placement(200, 0, 272, 90),
		_placement(0, 300, 300, 500),
	]
	flags = cluster_flags(placements, FigureOptions())

	assert flags[0] is True
	assert flags[1] is True
	assert flags[2] is False
