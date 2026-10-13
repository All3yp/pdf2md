"""Optional vision classifier for ambiguous images (opt-in, OpenAI-compatible).

Any OpenAI-compatible vision endpoint works. A few are preset by name via
``VISION_PROVIDER`` (nvidia, openrouter, ollama, lmstudio, openai), and any
endpoint can be given explicitly with ``VISION_BASE_URL``. Provider-specific
request fields (e.g. NVIDIA's ``chat_template_kwargs``) go through
``VISION_EXTRA_BODY`` as a JSON object. See ``.env.example``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

PROVIDERS: dict[str, str] = {
	"nvidia": "https://integrate.api.nvidia.com/v1",
	"openrouter": "https://openrouter.ai/api/v1",
	"ollama": "http://localhost:11434/v1",
	"lmstudio": "http://localhost:1234/v1",
	"openai": "https://api.openai.com/v1",
}

JUNK_LABELS = {"author_photo", "logo", "icon", "decorative", "watermark", "qr_code", "signature"}
CONTENT_LABELS = {"content", "figure", "chart", "diagram", "table", "graph", "map", "plot"}

PROMPT = (
	"You are given one image cropped from a scientific paper PDF. "
	"Decide whether it is part of the article's content or not. "
	"Reply with exactly ONE word from this list and nothing else: "
	"content (a real figure, chart, diagram, table, plot, map or scientific image), "
	"author_photo, logo, icon, decorative, watermark, signature."
)


@dataclass
class VisionClassifier:
	"""Thin wrapper over an OpenAI-compatible chat endpoint with image input."""

	base_url: str
	api_key: str
	model: str
	timeout: float = 30.0
	temperature: float = 0.0
	max_tokens: int = 16
	top_p: float | None = None
	extra_body: dict[str, Any] | None = None

	@classmethod
	def from_env(cls) -> VisionClassifier | None:
		"""Build from VISION_* env vars, or ``None`` when unconfigured."""
		base_url = os.environ.get("VISION_BASE_URL", "").strip()
		if not base_url:
			provider = os.environ.get("VISION_PROVIDER", "").strip().lower()
			base_url = PROVIDERS.get(provider, "")
		model = os.environ.get("VISION_MODEL", "").strip()
		if not base_url or not model:
			return None
		top_p_raw = os.environ.get("VISION_TOP_P", "").strip()
		return cls(
			base_url=base_url,
			api_key=os.environ.get("VISION_API_KEY", "").strip() or "dummy",
			model=model,
			timeout=float(os.environ.get("VISION_TIMEOUT", "30")),
			temperature=float(os.environ.get("VISION_TEMPERATURE", "0")),
			max_tokens=int(os.environ.get("VISION_MAX_TOKENS", "16")),
			top_p=float(top_p_raw) if top_p_raw else None,
			extra_body=parse_extra_body(os.environ.get("VISION_EXTRA_BODY", "")),
		)

	def classify(self, png_bytes: bytes) -> tuple[bool, str]:
		"""Return ``(is_junk, label)``; failures fall back to keeping the image."""
		request: dict[str, Any] = {
			"model": self.model,
			"temperature": self.temperature,
			"max_tokens": self.max_tokens,
			"messages": [
				{
					"role": "user",
					"content": [
						{"type": "text", "text": PROMPT},
						{"type": "image_url", "image_url": {"url": _data_url(png_bytes)}},
					],
				}
			],
		}
		if self.top_p is not None:
			request["top_p"] = self.top_p
		if self.extra_body:
			request["extra_body"] = self.extra_body

		try:
			from openai import OpenAI

			client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
			response = client.chat.completions.create(**request)
			answer = (response.choices[0].message.content or "").strip().lower()
		except Exception as error:
			return False, f"vision error: {error}"

		label = _first_label(answer)
		return label in JUNK_LABELS, label or answer[:40]


def parse_extra_body(raw: str) -> dict[str, Any] | None:
	"""Parse ``VISION_EXTRA_BODY`` (a JSON object) into request extra fields."""
	raw = raw.strip()
	if not raw:
		return None
	try:
		parsed = json.loads(raw)
	except json.JSONDecodeError as error:
		print(f"Aviso: VISION_EXTRA_BODY inválido ({error}); ignorando.", file=sys.stderr)
		return None
	if not isinstance(parsed, dict):
		print("Aviso: VISION_EXTRA_BODY precisa ser um objeto JSON; ignorando.", file=sys.stderr)
		return None
	return parsed


def _data_url(png_bytes: bytes) -> str:
	return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _first_label(answer: str) -> str:
	tokens: list[str] = re.findall(r"[a-z_]+", answer)
	for token in tokens:
		if token in JUNK_LABELS or token in CONTENT_LABELS:
			return token
	return tokens[0] if tokens else ""
