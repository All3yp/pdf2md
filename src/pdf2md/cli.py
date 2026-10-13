"""Command-line interface for pdf2md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf2md.converter import convert_directory, convert_pdf_to_markdown
from pdf2md.figures import FigureOptions
from pdf2md.vision import VisionClassifier


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="pdf2md",
		description=(
			"Converte PDFs em Markdown: tabelas viram tabelas Markdown e imagens "
			"que não são do artigo (fotos de autor, logos, ícones) são descartadas."
		),
	)
	parser.add_argument("input", type=Path, help="Um PDF ou uma pasta com PDFs")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("output/markdown"),
		help="Pasta dos arquivos .md",
	)
	parser.add_argument(
		"--no-ocr", action="store_true", help="Desativa OCR em páginas sem texto extraível"
	)
	parser.add_argument("--ocr-lang", default="eng", help="Idioma do Tesseract (eng, por, ...)")
	parser.add_argument(
		"--vision",
		action="store_true",
		help="Usa o classificador de visão opcional (VISION_*) em imagens ambíguas",
	)
	parser.add_argument(
		"--trace", action="store_true", help="Mostra no stderr cada imagem ignorada e o motivo"
	)
	parser.add_argument(
		"--min-image-area",
		type=float,
		default=None,
		help="Área mínima em pt² para manter uma imagem",
	)
	return parser


def _make_options(args: argparse.Namespace) -> FigureOptions:
	options = FigureOptions()
	if args.min_image_area is not None:
		options.min_area = args.min_image_area
	return options


def _build_vision(args: argparse.Namespace) -> VisionClassifier | None:
	if not args.vision:
		return None
	vision = VisionClassifier.from_env()
	if vision is None:
		print(
			"Aviso: --vision ativado, mas VISION_BASE_URL/VISION_MODEL não estão "
			"definidos; seguindo sem visão.",
			file=sys.stderr,
		)
	return vision


def main(argv: list[str] | None = None) -> None:
	try:
		from dotenv import load_dotenv

		load_dotenv()
	except ImportError:
		pass

	parser = build_parser()
	args = parser.parse_args(argv)
	options = _make_options(args)
	vision = _build_vision(args)

	if args.input.is_dir():
		pdf_count = len(list(args.input.rglob("*.pdf")))
		converted = convert_directory(
			args.input,
			args.output,
			ocr=not args.no_ocr,
			ocr_lang=args.ocr_lang,
			options=options,
			vision=vision,
			trace=args.trace,
		)
		if pdf_count and converted == 0:
			raise SystemExit(1)
		return

	if args.input.suffix.lower() == ".pdf":
		result = convert_pdf_to_markdown(
			args.input,
			args.output,
			ocr=not args.no_ocr,
			ocr_lang=args.ocr_lang,
			options=options,
			vision=vision,
			trace=args.trace,
		)
		print(
			f"OK  {args.input} -> {result.markdown_path} "
			f"[{result.tables} tabelas, {result.figures} figuras, {result.skipped} ignoradas]"
		)
		return

	parser.error(f"Entrada inválida: {args.input} (use um .pdf ou uma pasta)")


if __name__ == "__main__":
	main()
