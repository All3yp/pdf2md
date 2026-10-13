from types import SimpleNamespace

import openai
import pytest

from pdf2md.vision import VisionClassifier, _first_label, parse_extra_body

_ENV_KEYS = (
	"VISION_PROVIDER",
	"VISION_BASE_URL",
	"VISION_API_KEY",
	"VISION_MODEL",
	"VISION_TIMEOUT",
	"VISION_TEMPERATURE",
	"VISION_MAX_TOKENS",
	"VISION_TOP_P",
	"VISION_EXTRA_BODY",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
	for key in _ENV_KEYS:
		monkeypatch.delenv(key, raising=False)


def test_from_env_returns_none_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
	_clear(monkeypatch)
	assert VisionClassifier.from_env() is None


def test_from_env_uses_nvidia_preset(monkeypatch: pytest.MonkeyPatch) -> None:
	_clear(monkeypatch)
	monkeypatch.setenv("VISION_PROVIDER", "nvidia")
	monkeypatch.setenv("VISION_API_KEY", "nvapi-x")
	monkeypatch.setenv("VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")

	classifier = VisionClassifier.from_env()

	assert classifier is not None
	assert classifier.base_url == "https://integrate.api.nvidia.com/v1"


def test_from_env_base_url_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
	_clear(monkeypatch)
	monkeypatch.setenv("VISION_PROVIDER", "nvidia")
	monkeypatch.setenv("VISION_BASE_URL", "http://localhost:9999/v1")
	monkeypatch.setenv("VISION_MODEL", "m")

	classifier = VisionClassifier.from_env()

	assert classifier is not None
	assert classifier.base_url == "http://localhost:9999/v1"


def test_from_env_reads_sampling_and_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
	_clear(monkeypatch)
	monkeypatch.setenv("VISION_BASE_URL", "https://x/v1")
	monkeypatch.setenv("VISION_MODEL", "m")
	monkeypatch.setenv("VISION_TEMPERATURE", "1")
	monkeypatch.setenv("VISION_MAX_TOKENS", "4096")
	monkeypatch.setenv("VISION_TOP_P", "0.95")
	monkeypatch.setenv("VISION_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')

	classifier = VisionClassifier.from_env()

	assert classifier is not None
	assert (classifier.temperature, classifier.max_tokens, classifier.top_p) == (1.0, 4096, 0.95)
	assert classifier.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_parse_extra_body_rejects_invalid(capsys: pytest.CaptureFixture[str]) -> None:
	assert parse_extra_body("") is None
	assert parse_extra_body("{not json") is None
	assert parse_extra_body("[1, 2]") is None
	assert "VISION_EXTRA_BODY" in capsys.readouterr().err


def test_first_label_picks_known_label() -> None:
	assert _first_label("author_photo") == "author_photo"
	assert _first_label("The answer is content.") == "content"
	assert _first_label("") == ""


def test_classify_passes_extra_body_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
	captured: dict[str, object] = {}

	class FakeCompletions:
		def create(self, **kwargs: object) -> object:
			captured.update(kwargs)
			message = SimpleNamespace(content="author_photo")
			return SimpleNamespace(choices=[SimpleNamespace(message=message)])

	class FakeClient:
		def __init__(self, **kwargs: object) -> None:
			self.chat = SimpleNamespace(completions=FakeCompletions())

	monkeypatch.setattr(openai, "OpenAI", FakeClient)

	classifier = VisionClassifier(
		base_url="https://integrate.api.nvidia.com/v1",
		api_key="k",
		model="m",
		max_tokens=256,
		extra_body={"chat_template_kwargs": {"enable_thinking": False}},
	)
	is_junk, label = classifier.classify(b"png-bytes")

	assert is_junk is True
	assert label == "author_photo"
	assert captured["max_tokens"] == 256
	assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
	messages = captured["messages"]
	assert isinstance(messages, list)
	assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_classify_keeps_image_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
	class Boom:
		def __init__(self, **kwargs: object) -> None:
			raise RuntimeError("boom")

	monkeypatch.setattr(openai, "OpenAI", Boom)

	classifier = VisionClassifier(base_url="https://x/v1", api_key="k", model="m")
	is_junk, reason = classifier.classify(b"png")

	assert is_junk is False
	assert "vision error" in reason
