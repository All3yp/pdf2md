# pdf2md

Converte PDFs de artigos em Markdown, individualmente ou em lote.

## Recursos

- Converte tabelas em Markdown sem duplicar seu conteúdo como texto.
- Descarta detecções de tabela com baixo preenchimento.
- Filtra fotos de autores, ícones, fundos, logos repetidos e imagens dentro de tabelas.
- Salva figuras em PNG e inclui referências no Markdown.
- Usa Tesseract para OCR em páginas sem texto extraível.
- Oferece classificação de imagens por visão, desativada por padrão.

## Instalação

```bash
uv sync
```

## Uso

```bash
# Converter uma pasta, incluindo subpastas
uv run pdf2md data-pdfs --output output/markdown

# Converter um arquivo
uv run pdf2md artigo.pdf --output output/markdown

# Registrar imagens descartadas e seus motivos no stderr
uv run pdf2md data-pdfs --output output/markdown --trace
```

Estrutura de saída:

```text
output/markdown/
├── artigo.md
└── artigo_assets/
    ├── page-003-figure-01.png
    └── ...
```

## Opções

| Flag | Descrição |
|------|-----------|
| `--output DIR` | Pasta de saída. Padrão: `output/markdown` |
| `--no-ocr` | Desativa OCR |
| `--ocr-lang LANG` | Idioma do Tesseract: `eng`, `por` etc. |
| `--trace` | Registra imagens descartadas e seus motivos |
| `--vision` | Classifica imagens ambíguas com um modelo de visão |
| `--min-image-area N` | Área mínima das imagens, em pt² |

## Classificador de visão

O classificador é usado apenas para imagens ambíguas, como fotos que podem ser de autores ou figuras do artigo. Aceita endpoints compatíveis com a API OpenAI.

Crie o arquivo de configuração:

```bash
cp .env.example .env
```

Configure um provedor no `.env`:

```dotenv
# Provedores: nvidia, openrouter, ollama, lmstudio ou openai
VISION_PROVIDER=openrouter
VISION_API_KEY=sk-or-...
VISION_MODEL=google/gemma-3-27b-it:free

# URL direta: substitui o endpoint do provedor
# VISION_BASE_URL=http://localhost:11434/v1
```

Exemplo para NVIDIA NIM:

```dotenv
VISION_PROVIDER=nvidia
VISION_API_KEY=nvapi-xxxxxxxx
VISION_MODEL=meta/llama-3.2-90b-vision-instruct
VISION_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
```

Parâmetros adicionais:

- `VISION_TEMPERATURE`
- `VISION_MAX_TOKENS`
- `VISION_TOP_P`
- `VISION_EXTRA_BODY`: campos extras do provedor em JSON.

Se o modelo usar tokens de raciocínio, aumente `VISION_MAX_TOKENS` para reservar espaço para a resposta.

Execute com `--vision`:

```bash
uv run pdf2md data-pdfs --output output/markdown --vision
```

Sem modelo ou endpoint configurado, a conversão emite um aviso e usa apenas as heurísticas.

## Ajuste das heurísticas

Configure os limiares com `pdf2md.figures.FigureOptions`:

```python
from pathlib import Path
from pdf2md import FigureOptions, convert_pdf_to_markdown

options = FigureOptions(min_area=2500, small_side_max=150)
convert_pdf_to_markdown(Path("artigo.pdf"), Path("out"), options=options)
```

## Desenvolvimento

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```