from __future__ import annotations

from typing import Final


MODEL_OPTIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B Versatile",
        "description": "Recommended default for grounded multilingual answers.",
    },
    {
        "id": "openai/gpt-oss-120b",
        "label": "GPT-OSS 120B",
        "description": "High-capacity fallback for grounded answers.",
    },
    {
        "id": "qwen/qwen3.6-27b",
        "label": "Qwen 3.6 27B",
        "description": "Balanced quality and speed.",
    },
    {
        "id": "openai/gpt-oss-20b",
        "label": "GPT-OSS 20B",
        "description": "Fast model for shorter answers.",
    },
    {
        "id": "llama-3.1-8b-instant",
        "label": "Llama 3.1 8B Instant",
        "description": "Fastest lightweight fallback.",
    },
)

SUPPORTED_MODEL_IDS: Final[tuple[str, ...]] = tuple(
    option["id"] for option in MODEL_OPTIONS
)
DEFAULT_MODEL: Final[str] = "llama-3.3-70b-versatile"
DEFAULT_ANSWER_MODEL_ORDER: Final[tuple[str, ...]] = SUPPORTED_MODEL_IDS
DEFAULT_UTILITY_MODEL_ORDER: Final[tuple[str, ...]] = (
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
)
MODEL_PREFERENCES: Final[tuple[str, ...]] = ("auto", *SUPPORTED_MODEL_IDS)


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


def model_option(model_id: str) -> dict[str, str]:
    for option in MODEL_OPTIONS:
        if option["id"] == model_id:
            return dict(option)
    raise ValueError(f"Unsupported generation model: {model_id}")
