from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import Generic, NoReturn, TypeVar

from app.core.logging import format_log_event
from app.services.generation_resilience import (
    GenerationResilience,
    GenerationUnavailableError,
    generation_resilience,
)
from app.services.model_catalog import model_provider
from app.services.model_router import ModelRouter


logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ModelExecutionResult(Generic[T]):
    value: T
    requested_model: str
    selected_model: str
    fallback_used: bool


class ModelFailoverService:
    """Executes an operation against healthy models in deterministic order."""

    def __init__(
        self,
        router: ModelRouter,
        resilience: GenerationResilience = generation_resilience,
    ) -> None:
        self.router = router
        self.resilience = resilience

    def execute(
        self,
        operation: str,
        preference: str | None,
        callback: Callable[[str], T],
    ) -> ModelExecutionResult[T]:
        requested_model = self.router.normalize_preference(preference)
        candidates = self.router.candidates(operation, requested_model)
        failures: list[GenerationUnavailableError] = []
        logger.info(
            format_log_event(
                "Generation model candidates",
                operation=operation,
                requested_model=requested_model,
                candidates=candidates,
            )
        )

        for index, model in enumerate(candidates):
            try:
                value = self.resilience.call(
                    operation,
                    lambda model=model: callback(model),
                    circuit_key=f"{model_provider(model)}:{model}",
                    model=model,
                )
                return ModelExecutionResult(
                    value=value,
                    requested_model=requested_model,
                    selected_model=model,
                    fallback_used=index > 0,
                )
            except GenerationUnavailableError as exc:
                failures.append(exc)
                logger.info(
                    format_log_event(
                        "Generation model unavailable; trying fallback",
                        operation=operation,
                        model=model,
                    )
                )

        self._raise_exhausted(failures)

    async def execute_async(
        self,
        operation: str,
        preference: str | None,
        callback: Callable[[str], Awaitable[T]],
    ) -> ModelExecutionResult[T]:
        requested_model = self.router.normalize_preference(preference)
        candidates = self.router.candidates(operation, requested_model)
        failures: list[GenerationUnavailableError] = []
        logger.info(
            format_log_event(
                "Generation model candidates",
                operation=operation,
                requested_model=requested_model,
                candidates=candidates,
            )
        )

        for index, model in enumerate(candidates):
            try:
                value = await self.resilience.call_async(
                    operation,
                    lambda model=model: callback(model),
                    circuit_key=f"{model_provider(model)}:{model}",
                    model=model,
                )
                return ModelExecutionResult(
                    value=value,
                    requested_model=requested_model,
                    selected_model=model,
                    fallback_used=index > 0,
                )
            except GenerationUnavailableError as exc:
                failures.append(exc)
                logger.info(
                    format_log_event(
                        "Generation model unavailable; trying fallback",
                        operation=operation,
                        model=model,
                    )
                )

        self._raise_exhausted(failures)

    @staticmethod
    def _raise_exhausted(
        failures: list[GenerationUnavailableError],
    ) -> NoReturn:
        retry_values = [
            failure.retry_after
            for failure in failures
            if failure.retry_after is not None
        ]
        retry_after = min(retry_values) if retry_values else None
        error_id = failures[-1].error_id if failures else None
        logger.warning(
            format_log_event(
                "Generation model failover exhausted",
                failures=len(failures),
                retry_after=retry_after,
                error_id=error_id,
            )
        )
        raise GenerationUnavailableError(
            retry_after=retry_after,
            error_id=error_id,
        )
