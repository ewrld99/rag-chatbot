from __future__ import annotations

from typing import Final


MODEL_OPTIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "google/gemma-4-26b-a4b-it",
        "label": "Google Gemma 4 26B A4B (OpenRouter)",
        "description": "Primary Gemma 4 26B model via OpenRouter.",
    },
    {
        "id": "google/gemma-4-26b-a4b-it:free",
        "label": "Google Gemma 4 26B A4B Free (OpenRouter)",
        "description": "Free-tier Gemma 4 26B model via OpenRouter.",
    },
    {
        "id": "llama3.2:3b",
        "label": "Llama 3.2 3B Local",
        "description": "Primary local Ollama model for grounded answers.",
    },
    {
        "id": "gemma3:4b",
        "label": "Gemma 3 4B Local",
        "description": "Local fallback model for grounded answers on CPU-only systems.",
    },
    {
        "id": "qwen3.5:4b",
        "label": "Qwen 3.5 4B Local",
        "description": "Optional local thinking model; slower and more demanding on CPU.",
    },
    {
        "id": "gemini-3.6-flash",
        "label": "Gemini 3.6 Flash",
        "description": "Default free-tier Gemini Flash model for grounded answers.",
    },
    {
        "id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B Versatile",
        "description": "Groq fallback for grounded multilingual answers.",
    },
)

SUPPORTED_MODEL_IDS: Final[tuple[str, ...]] = tuple(
    option["id"] for option in MODEL_OPTIONS
)
DEFAULT_MODEL: Final[str] = "google/gemma-4-26b-a4b-it"
DEFAULT_ANSWER_MODEL_ORDER: Final[tuple[str, ...]] = (
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-26b-a4b-it:free",
    "llama3.2:3b",
    "gemma3:4b",
    "gemini-3.6-flash",
    "llama-3.3-70b-versatile",
    "qwen3.5:4b",
)
DEFAULT_UTILITY_MODEL_ORDER: Final[tuple[str, ...]] = (
    "llama-3.1-8b-instant",
    "qwen2.5:1.5b",
    "gemma3:4b",
    "llama-3.3-70b-versatile",
)
UTILITY_MODEL_IDS: Final[frozenset[str]] = frozenset(DEFAULT_UTILITY_MODEL_ORDER)
GROUNDING_VERIFICATION_MODEL_ORDER: Final[tuple[str, ...]] = (
    "llama-3.1-8b-instant",
    "gemma3:4b",
    "llama-3.3-70b-versatile",
    "qwen2.5:1.5b",
)
MODEL_PREFERENCES: Final[tuple[str, ...]] = ("auto", *SUPPORTED_MODEL_IDS)


def validate_model_preference(value: str | None) -> str:
    normalized = (value or "auto").strip()
    if normalized not in MODEL_PREFERENCES:
        raise ValueError("Unsupported answer model preference.")
    return normalized


def parse_model_csv(
    value: str | None,
    fallback: tuple[str, ...],
) -> list[str]:
    raw_models = (value or "").split(",")
    models: list[str] = []
    for raw_model in raw_models:
        model = raw_model.strip()
        if model in SUPPORTED_MODEL_IDS and model not in models:
            models.append(model)
    return models or list(fallback)


def parse_utility_model_csv(value: str | None) -> list[str]:
    models = [
        model.strip()
        for model in (value or "").split(",")
        if model.strip() in UTILITY_MODEL_IDS
    ]
    return list(dict.fromkeys(models)) or list(DEFAULT_UTILITY_MODEL_ORDER)


def model_option(model_id: str) -> dict[str, str]:
    for option in MODEL_OPTIONS:
        if option["id"] == model_id:
            return dict(option)
    raise ValueError(f"Unsupported generation model: {model_id}")


def model_provider(model_id: str) -> str:
    if model_id.startswith("google/") or "/" in model_id:
        return "openrouter"
    if model_id.startswith("gemma") or ":" in model_id:
        return "ollama"
    if model_id.startswith("gemini-"):
        return "gemini"
    if model_id.startswith("llama-"):
        return "groq"
    if model_id in SUPPORTED_MODEL_IDS:
        return "groq"
    return "unknown"
