"""AI provider abstraction.

ONE common interface (`AIProvider.complete()` / `.test_connection()`) used
everywhere else in the app (website analysis, opportunity extraction,
eligibility interpretation, explanations). All provider-specific request/
response shaping -- and ONLY that -- lives in this file, in one adapter
class per provider family, so no other module needs to know how any
specific vendor's API works.

None of these adapters have been exercised against a real, working API key
in this build (no credentials were available in this environment). The HTTP
request shapes are written to match each vendor's published API docs as of
this writing, but "written correctly" and "tested against the live
service" are different claims -- be explicit about that distinction if
asked. `test_connection()` makes a REAL, minimal HTTP call and honestly
reports whatever the provider actually returns (including a real
authentication failure for a bad/placeholder key) -- it does not simulate
success.
"""
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("grantsetu.ai")

REQUEST_TIMEOUT = 20.0

# OpenAI-compatible providers differ only in base_url and (optionally)
# default model -- one adapter class handles all of them.
OPENAI_COMPATIBLE_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "together": {"base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3-8b-chat-hf"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"},
}

SUPPORTED_PROVIDERS = list(OPENAI_COMPATIBLE_DEFAULTS) + ["anthropic", "gemini", "cohere"]


@dataclass
class AIResult:
    ok: bool
    text: str | None = None
    error: str | None = None


class AIProvider:
    """Common interface every adapter implements."""

    name = "base"

    def complete(self, prompt: str, max_tokens: int = 700) -> AIResult:
        raise NotImplementedError

    def test_connection(self) -> AIResult:
        """A cheap, real call used by the "test connection" button. Default
        implementation just runs a tiny completion; adapters may override
        with a lighter-weight endpoint if their API has one."""
        return self.complete("Reply with the single word: ok", max_tokens=5)


class OpenAICompatibleProvider(AIProvider):
    """Covers OpenAI, DeepSeek, Groq, Mistral, Together AI, OpenRouter, and
    any other OpenAI-compatible chat-completions API (via base_url override)."""

    def __init__(self, provider_key: str, api_key: str, model: str | None = None, base_url: str | None = None):
        defaults = OPENAI_COMPATIBLE_DEFAULTS.get(provider_key, {})
        self.name = provider_key
        self.api_key = api_key
        self.model = model or defaults.get("model", "gpt-4o-mini")
        self.base_url = (base_url or defaults.get("base_url", "https://api.openai.com/v1")).rstrip("/")

    def complete(self, prompt: str, max_tokens: int = 700) -> AIResult:
        try:
            # Floor, never lower a caller's own higher value -- some models
            # served through this shared adapter (e.g. Groq's
            # openai/gpt-oss-20b) are reasoning models that spend part of
            # their output budget on internal reasoning tokens before any
            # visible text, so a low caller-supplied limit (e.g.
            # test_connection()'s max_tokens=5) can exhaust the budget with
            # zero visible output. 700 mirrors this class's own default.
            max_tokens = max(max_tokens, 700)
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return AIResult(ok=True, text=text)
        except Exception as exc:  # noqa: BLE001 -- must never crash the caller; always fall back
            logger.warning("%s completion failed: %s", self.name, exc)
            return AIResult(ok=False, error=str(exc))


class AnthropicProvider(AIProvider):
    name = "anthropic"
    API_URL = "https://api.anthropic.com/v1/messages"
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key
        self.model = model or "claude-3-5-haiku-latest"
        self.url = (base_url.rstrip("/") + "/messages") if base_url else self.API_URL

    def complete(self, prompt: str, max_tokens: int = 700) -> AIResult:
        try:
            response = httpx.post(
                self.url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            text = "".join(block.get("text", "") for block in data.get("content", []))
            return AIResult(ok=True, text=text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("anthropic completion failed: %s", exc)
            return AIResult(ok=False, error=str(exc))


class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key
        self.model = model or "gemini-3.6-flash"
        self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def complete(self, prompt: str, max_tokens: int = 700) -> AIResult:
        try:
            response = httpx.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates") or []
            parts = (candidates[0].get("content") or {}).get("parts") if candidates else None
            if not parts or "text" not in parts[0]:
                finish_reason = candidates[0].get("finishReason") if candidates else None
                if finish_reason == "MAX_TOKENS":
                    # The model stopped (e.g. on internal reasoning/thinking
                    # tokens for a "thinking"-capable model) before emitting
                    # any visible text -- a real, honest failure, not a
                    # parsing bug, so it must not be reported as one.
                    raise ValueError(
                        "Gemini stopped before producing any text (finishReason=MAX_TOKENS) -- "
                        "the maxOutputTokens limit was reached before a response was generated. "
                        "Try again with a higher max_tokens."
                    )
                raise ValueError(f"Unexpected Gemini response shape (no candidates[0].content.parts[0].text): {data}")
            text = parts[0]["text"]
            return AIResult(ok=True, text=text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gemini completion failed: %s", exc)
            return AIResult(ok=False, error=str(exc))

    def test_connection(self) -> AIResult:
        # Overrides the base class's max_tokens=5 -- too low for this
        # model: it can consume its whole maxOutputTokens budget on
        # internal reasoning before emitting visible text, hitting
        # finishReason=MAX_TOKENS with empty content (see complete() above)
        # even though the connection and key are actually fine.
        return self.complete("Reply with the single word: ok", max_tokens=64)


class CohereProvider(AIProvider):
    name = "cohere"
    API_URL = "https://api.cohere.com/v1/chat"

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key
        self.model = model or "command-r"
        self.url = (base_url.rstrip("/") + "/chat") if base_url else self.API_URL

    def complete(self, prompt: str, max_tokens: int = 700) -> AIResult:
        try:
            response = httpx.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "message": prompt, "max_tokens": max_tokens},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("text", "")
            return AIResult(ok=True, text=text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cohere completion failed: %s", exc)
            return AIResult(ok=False, error=str(exc))


def build_provider(provider_key: str, api_key: str, model: str | None = None, base_url: str | None = None) -> AIProvider:
    if provider_key in OPENAI_COMPATIBLE_DEFAULTS:
        return OpenAICompatibleProvider(provider_key, api_key, model=model, base_url=base_url)
    if provider_key == "anthropic":
        return AnthropicProvider(api_key, model=model, base_url=base_url)
    if provider_key == "gemini":
        return GeminiProvider(api_key, model=model, base_url=base_url)
    if provider_key == "cohere":
        return CohereProvider(api_key, model=model, base_url=base_url)
    raise ValueError(f"Unsupported AI provider '{provider_key}'. Supported: {SUPPORTED_PROVIDERS}")
