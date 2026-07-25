from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.services.model_catalog import (
    DEFAULT_ANSWER_MODEL_ORDER,
    DEFAULT_MODEL,
    DEFAULT_UTILITY_MODEL_ORDER,
    MODEL_OPTIONS,
    SUPPORTED_MODEL_IDS,
    parse_model_csv,
)


class InvalidModelPreference(ValueError):
    pass


class ModelRouter:
    """Builds an allowlisted, deterministic candidate list for each LLM call."""

    _ANSWER_OPERATION_PREFIXES = (
        "document_answer",
        "student_support_answer",
        "conversational_answer",
        "clarification_request",
    )

    def __init__(self, settings_service: Any | None = None) -> None:
        self.settings_service = settings_service

    @property
    def selection_enabled(self) -> bool:
        if self.settings_service is not None:
            return bool(self.settings_service.generation_user_selection_enabled)
        return settings.GROQ_USER_MODEL_SELECTION_ENABLED

    @property
    def allowed_models(self) -> list[str]:
        if self.settings_service is not None:
            configured = list(self.settings_service.generation_allowed_models)
        else:
            configured = parse_model_csv(
                settings.GROQ_ALLOWED_MODELS,
                DEFAULT_ANSWER_MODEL_ORDER,
            )

        allowed = [
            model
            for model in configured
            if model in SUPPORTED_MODEL_IDS
        ]
        return allowed or list(DEFAULT_ANSWER_MODEL_ORDER)

    @property
    def default_model(self) -> str:
        if self.settings_service is not None:
            configured = self.settings_service.generation_default_model
        else:
            configured = settings.GROQ_MODEL
        return configured if configured in self.allowed_models else self.allowed_models[0]

    def normalize_preference(self, preference: str | None) -> str:
        normalized = (preference or "auto").strip()
        if not self.selection_enabled:
            return "auto"
        if normalized == "auto":
            return normalized
        if normalized not in self.allowed_models:
            raise InvalidModelPreference(
                "The selected model is not enabled for this application."
            )
        return normalized

    def candidates(self, operation: str, preference: str | None = None) -> list[str]:
        normalized = self.normalize_preference(preference)
        allowed = self.allowed_models

        if self._is_answer_operation(operation):
            configured = self._answer_order()
            ordered = [self.default_model, *configured, *allowed]
            if normalized != "auto":
                ordered.insert(0, normalized)
        else:
            ordered = [*self._utility_order(), *allowed]

        return self._deduplicate_allowed(ordered, allowed)

    def public_policy(self) -> dict[str, Any]:
        labels = {option["id"]: option for option in MODEL_OPTIONS}
        return {
            "default": "auto",
            "selection_enabled": self.selection_enabled,
            "models": [
                {
                    **labels[model],
                    "is_default": model == self.default_model,
                }
                for model in self.allowed_models
            ],
        }

    def _answer_order(self) -> list[str]:
        if self.settings_service is not None:
            configured = list(self.settings_service.generation_answer_model_order)
        else:
            configured = parse_model_csv(
                settings.GROQ_ANSWER_MODEL_ORDER,
                DEFAULT_ANSWER_MODEL_ORDER,
            )
        return configured

    def _utility_order(self) -> list[str]:
        if self.settings_service is not None:
            configured = list(self.settings_service.generation_utility_model_order)
        else:
            configured = parse_model_csv(
                settings.GROQ_UTILITY_MODEL_ORDER,
                DEFAULT_UTILITY_MODEL_ORDER,
            )
        return configured

    @classmethod
    def _is_answer_operation(cls, operation: str) -> bool:
        return operation.startswith(cls._ANSWER_OPERATION_PREFIXES)

    @staticmethod
    def _deduplicate_allowed(
        candidates: list[str],
        allowed: list[str],
    ) -> list[str]:
        result: list[str] = []
        for model in candidates:
            if model in allowed and model not in result:
                result.append(model)
        return result
