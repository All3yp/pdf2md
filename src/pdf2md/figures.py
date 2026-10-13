"""Rasterize figures that are part of the article, dropping non-article junk.

Junk categories handled here (all best-effort, geometry + pixel statistics):

* tiny inline images (math glyphs, bullet icons),
* full-page backgrounds,
* images that sit inside a detected table,
* logotypes/watermarks reused across many pages,
* author photos (small, portrait, photographic) and other decorative photos.

Genuinely ambiguous images can be deferred to the optional vision classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf


@dataclass(frozen=True)
class ImagePlacement:
	"""Where an embedded image is drawn on a page."""

	bbox: tuple[float, float, float, float]
	xref: int
	width: int
	height: int

	@property
	def box_width(self) -> float:
		return self.bbox[2] - self.bbox[0]

	@property
	def box_height(self) -> float:
		return self.bbox[3] - self.bbox[1]

	@property
	def aspect(self) -> float:
		return self.box_width / max(1e-6, self.box_height)

	@property
	def min_side(self) -> float:
		return min(self.box_width, self.box_height)


@dataclass(frozen=True)
class ImageStats:
	"""Pixel statistics used to tell photographs from line art."""

	white_fraction: float
	color_count: int
	mean_saturation: float
	colorful_fraction: float


@dataclass
class FigureOptions:
	"""Tunable thresholds for the junk filters."""

	min_area: float = 1600.0
	min_side: float = 24.0
	max_area_ratio: float = 0.75
	table_overlap: float = 0.3
	photo_white_max: float = 0.35
	photo_colorful_min: float = 0.12
	portrait_min_ratio: float = 0.6
	portrait_max_ratio: float = 0.95
	small_side_max: float = 170.0
	cluster_tolerance: float = 0.15


@dataclass
class FigureDecision:
	"""Outcome of classifying one image."""

	keep: bool
	reason: str = ""
	ambiguous: bool = False
	stats: ImageStats | None = None
	extra: dict[str, object] = field(default_factory=dict)


def image_placements(page: pymupdf.Page) -> list[ImagePlacement]:
	"""Return embedded-image placements on ``page`` (bbox + intrinsic size + xref)."""
	placements: list[ImagePlacement] = []
	for info in page.get_image_info(xrefs=True):
		bbox = info.get("bbox")
		if not bbox:
			continue
		placements.append(
			ImagePlacement(
				bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
				xref=int(info.get("xref") or 0),
				width=int(info.get("width") or 0),
				height=int(info.get("height") or 0),
			)
		)
	return placements


def render_pixmap(
	page: pymupdf.Page, bbox: tuple[float, float, float, float], dpi: int
) -> pymupdf.Pixmap:
	"""Render a page region to an RGB pixmap."""
	clip = pymupdf.Rect(*bbox) & page.rect
	return page.get_pixmap(clip=clip, dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)


def image_stats(pixmap: pymupdf.Pixmap) -> ImageStats:
	"""Compute white fraction, color diversity, saturation and colorfulness."""
	data = pixmap.samples
	if not data:
		return ImageStats(1.0, 0, 0.0, 0.0)

	channels = pixmap.n
	stride = channels * 3
	total = 0
	white = 0
	colorful = 0
	saturation = 0.0
	colors: set[int] = set()
	for index in range(0, len(data) - channels, stride):
		r = data[index]
		g = data[index + 1]
		b = data[index + 2]
		total += 1
		if r > 235 and g > 235 and b > 235:
			white += 1
		high = max(r, g, b)
		low = min(r, g, b)
		if high - low > 40:
			colorful += 1
		saturation += high - low
		colors.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4))

	if total == 0:
		return ImageStats(1.0, 0, 0.0, 0.0)
	return ImageStats(
		white_fraction=white / total,
		color_count=len(colors),
		mean_saturation=saturation / total / 255.0,
		colorful_fraction=colorful / total,
	)


def cluster_flags(placements: list[ImagePlacement], options: FigureOptions) -> list[bool]:
	"""Flag placements that are part of a grid of >= 2 similarly-sized portrait images."""
	flags = [False] * len(placements)
	pairs = [(i, p) for i, p in enumerate(placements) if _is_portrait(p, options)]
	for index, left in pairs:
		for other_index, right in pairs:
			if index == other_index:
				continue
			tolerance = options.cluster_tolerance
			width_ok = abs(left.box_width - right.box_width) <= tolerance * left.box_width
			height_ok = abs(left.box_height - right.box_height) <= tolerance * left.box_height
			if width_ok and height_ok:
				flags[index] = True
				break
	return flags


def _is_portrait(placement: ImagePlacement, options: FigureOptions) -> bool:
	return options.portrait_min_ratio <= placement.aspect <= options.portrait_max_ratio


def geometry_rejection(
	placement: ImagePlacement,
	page: pymupdf.Page,
	*,
	table_rects: list[pymupdf.Rect],
	repeated_xrefs: set[int],
	options: FigureOptions,
) -> str | None:
	"""Return a skip reason based on geometry alone, or ``None`` to keep considering it."""
	bbox = pymupdf.Rect(*placement.bbox)
	area = bbox.get_area()
	if area <= 0:
		return "degenerate box"
	if area < options.min_area:
		return "tiny image (icon or inline glyph)"
	page_area = page.rect.width * page.rect.height
	if page_area > 0 and area > page_area * options.max_area_ratio:
		return "covers most of the page"
	if placement.min_side < options.min_side:
		return "thin sliver"
	for table_rect in table_rects:
		if (bbox & table_rect).get_area() > options.table_overlap * area:
			return "inside a detected table"
	if placement.xref and placement.xref in repeated_xrefs:
		return "repeated image (logo or watermark)"
	return None


def content_verdict(
	placement: ImagePlacement,
	stats: ImageStats,
	*,
	clustered: bool,
	options: FigureOptions,
) -> FigureDecision:
	"""Classify an image by pixel content after it passed the geometry filters."""
	photo_like = (
		stats.white_fraction < options.photo_white_max
		or stats.colorful_fraction > options.photo_colorful_min
	)
	portrait = _is_portrait(placement, options)
	small = placement.min_side <= options.small_side_max

	if small and photo_like and (portrait or clustered):
		return FigureDecision(keep=False, reason="author photo", stats=stats)
	if small and photo_like:
		return FigureDecision(keep=False, reason="", ambiguous=True, stats=stats)
	return FigureDecision(keep=True, reason="", stats=stats)
