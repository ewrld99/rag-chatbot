from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy.orm import joinedload
from starlette.concurrency import run_in_threadpool

from app.db.models import DocumentChunk
from app.services.conversation_context import generation_history
from app.services.generation_resilience import GenerationUnavailableError
from app.services.generation_service import GenerationService
from app.services.grounding_service import GroundingService
from app.services.intent_router import IntentDecision, IntentRouter
from app.services.model_router import ModelRouter
from app.services.query_normalization import (
    FALLBACK_ALIAS_EXPANSIONS,
    likely_swahili_query,
    metadata_search_text,
    significant_tokens,
    token_set,
    tokenize,
)
from app.services.retrieval_service import RetrievalService
from app.services.source_service import format_source_records
from app.core.config import settings
from app.core.logging import format_log_event


logger = logging.getLogger(__name__)


class _LatencyTrace:
    def __init__(self, operation: str, query: str) -> None:
        self.operation = operation
        self.query_preview = query[:120]
        self.started_at = time.perf_counter()
        self.stage_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.stage_ms[name] = round((time.perf_counter() - started_at) * 1000, 1)

    def mark(self, name: str) -> None:
        self.stage_ms[name] = round((time.perf_counter() - self.started_at) * 1000, 1)

    def log(self, **extra: Any) -> None:
        logger.info(
            format_log_event(
                "RAG latency",
                operation=self.operation,
                total_ms=round((time.perf_counter() - self.started_at) * 1000, 1),
                stages=self.stage_ms,
                extra=extra,
                query=self.query_preview,
            )
        )


class RAGPipeline:
    _CURRICULUM_METADATA_MAX_ROWS = 80
    _COMPLEX_GENERATION_QUERY_RE = re.compile(
        r"\b(?:calculate|calculated|calculating|calculation|compute|computed|"
        r"formula|equation|gpa|cgpa|grade point average|classification|"
        r"regulation|regulations|policy|policies|requirement|requirements|"
        r"hesabu|kuhesabu|inahesabiwa|wastani|alama|kanuni|sera|mahitaji)\b",
        re.IGNORECASE,
    )
    _PROCEDURE_GENERATION_QUERY_RE = re.compile(
        r"\b(?:how to|how do|process|procedure|steps?|apply|submit|"
        r"postpone|defer|deferment|register|appeal|jinsi|utaratibu|taratibu|"
        r"hatua|omba|maombi|wasilisha|tuma|sajili|jisajili|rufaa|"
        r"ahirisha|kuahirisha|kughairi|kughairisha|kusitisha)\b",
        re.IGNORECASE,
    )
    _CURRICULUM_LIST_QUERY_RE = re.compile(
        r"\b(?:course|courses|unit|units|module|modules|subject|subjects|"
        r"curriculum|kozi|somo|masomo|moduli|programu|mtaala)\b",
        re.IGNORECASE,
    )
    _ALMANAC_DATE_QUERY_RE = re.compile(
        r"\b(?:when|date|start|starts|starting|begin|begins|end|ends|finish|finishes|"
        r"examination|examinations|exam|exams|semester|supplementary|special|"
        r"lini|tarehe|anza|unaanza|kuanza|mwisho|malizika|mtihani|mitihani|"
        r"muhula|nyongeza|maalum)\b",
        re.IGNORECASE,
    )
    _ALMANAC_EVENT_STOPWORDS = {
        "academic", "all", "and", "are", "calendar", "date", "day", "does",
        "event", "for", "is", "of", "on", "program", "programme", "programmes",
        "programs", "the", "university", "when",
    }
    _CURRICULUM_STOPWORDS = {
        "what", "which", "list", "show", "give", "tell", "are", "the", "for",
        "student", "students", "course", "courses", "unit", "units", "module",
        "modules", "subject", "subjects", "semester", "year", "first", "second",
        "third", "fourth", "fifth", "one", "two", "three", "four", "five",
        "orodhesha", "onyesha", "toa", "nipe", "eleza", "taja", "zipi", "ipi",
        "kozi", "somo", "masomo", "moduli", "programu", "mtaala", "mwanafunzi",
        "wanafunzi", "mwaka", "muhula", "wa", "kwa", "katika", "kwanza",
        "pili", "tatu", "nne", "tano", "shahada", "chuo", "udom",
    }

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generator: GenerationService | None = None,
        intent_router: IntentRouter | None = None,
    ):
        self.retrieval_service = retrieval_service
        if generator is None:
            model_router = ModelRouter(retrieval_service.runtime_settings)
            generator = GenerationService(model_router=model_router)
        self.generator = generator
        self.intent_router = intent_router or IntentRouter(
            generator=self.generator,
            session_factory=retrieval_service.session_factory,
        )
        self._last_scored_documents: List[Tuple[Document, float]] = []

    def _is_empty_context(self, context: str) -> bool:
        return not context or context.strip() == "No relevant context found."

    def _retrieval_query(
        self,
        decision: IntentDecision,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        if (
            (
                getattr(decision, "conversation", None) is not None
                and decision.conversation.is_follow_up is True
            )
            or decision.source.startswith("rule:document_follow_up")
        ) and decision.standalone_query:
            return decision.standalone_query
        if chat_history and self.generator.is_history_dependent_query(query):
            rewritten_query = self.generator.rewrite_query(
                query,
                chat_history,
                user_profile=user_profile,
            )
            if rewritten_query:
                return rewritten_query
        if decision.intent == "UDOM_DOCUMENT_SEARCH" and likely_swahili_query(query):
            rewritten_query = self.generator.rewrite_query(
                query,
                chat_history,
                user_profile=user_profile,
            )
            if rewritten_query:
                return rewritten_query
        if decision.standalone_query:
            return decision.standalone_query
        return self.generator.rewrite_query(query, chat_history, user_profile=user_profile)

    @staticmethod
    def _generation_query(
        decision: IntentDecision,
        query: str,
        retrieval_query: str,
    ) -> str:
        if (
            getattr(decision, "conversation", None) is not None
            and decision.conversation.is_follow_up is True
        ) or decision.source.startswith("rule:document_follow_up"):
            return retrieval_query
        return query

    def _non_document_response(
        self,
        decision: IntentDecision,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | None:
        if decision.intent == "CONVERSATIONAL":
            answer = self.generator.generate_conversational(
                query,
                chat_history,
                user_profile=user_profile,
            )
            return self._base_response(query, answer, decision, include_model=True)

        if decision.intent == "STUDENT_SUPPORT":
            answer = self.generator.generate_student_support(
                query,
                chat_history,
                user_profile=user_profile,
            )
            return self._base_response(query, answer, decision, include_model=True)

        if decision.intent == "CLARIFY":
            answer = self.generator.generate_clarification_request(
                query,
                reason=decision.reason,
            )
            return self._base_response(query, answer, decision, include_model=True)

        if decision.intent == "OUT_OF_SCOPE":
            answer = self.generator.generate_out_of_scope(
                query,
                chat_history,
                user_profile=user_profile,
            )
            return self._base_response(query, answer, decision, include_model=True)

        return None

    def _arbitrate_student_support_route(
        self,
        decision: IntentDecision,
        query: str,
        chat_history: Optional[List[Dict[str, str]]],
        user_profile: Optional[Dict[str, Any]],
    ) -> tuple[IntentDecision, str | None, Dict[str, Any] | None]:
        if not self._should_probe_documents(query, decision):
            return decision, None, None

        retrieval_query = self._retrieval_query(
            decision,
            query,
            chat_history,
            user_profile=user_profile,
        )
        retrieval = self._retrieve_document_context(
            retrieval_query,
            filters=decision.filters,
        )
        confidence = retrieval["confidence"]
        if not self._retrieval_supports_route_override(confidence):
            self._last_scored_documents = []
            return decision, None, None

        promoted = self.intent_router.promote_to_document_search(
            decision,
            query,
            float(confidence.get("confidence") or 0.0),
        )
        logger.info(
            "Routing promoted by retrieval | from=%s to=%s tier=%s confidence=%.4f query=%s",
            decision.intent,
            promoted.intent,
            confidence.get("tier"),
            float(confidence.get("confidence") or 0.0),
            retrieval_query[:120],
        )
        return promoted, retrieval_query, retrieval

    def _should_probe_documents(
        self,
        query: str,
        decision: IntentDecision,
    ) -> bool:
        probe = getattr(self.intent_router, "should_probe_documents", None)
        return bool(callable(probe) and probe(query, decision))

    def _should_demote_to_student_support(
        self,
        query: str,
        decision: IntentDecision,
    ) -> bool:
        demote = getattr(self.intent_router, "should_demote_to_student_support", None)
        return bool(callable(demote) and demote(query, decision))

    @staticmethod
    def _retrieval_supports_route_override(confidence: Dict[str, Any]) -> bool:
        if not confidence.get("sufficient") or confidence.get("tier") != "high":
            return False
        if int(confidence.get("lexical_overlap") or 0) < 2:
            return False
        matched_sources = set(confidence.get("matched_sources") or [])
        sparse_rank = confidence.get("best_sparse_rank")
        return "sparse" in matched_sources or (
            isinstance(sparse_rank, int) and sparse_rank <= 10
        )

    def _base_response(
        self,
        query: str,
        answer: str,
        decision: IntentDecision,
        *,
        include_model: bool = False,
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "query": query,
            "answer": answer,
            "sources": [],
            "context_used": False,
            "routing": decision.to_dict(),
            "conversation": self._conversation_payload(decision),
        }
        if include_model:
            response.update(self.generator.model_metadata())
        return response

    @staticmethod
    def _conversation_payload(decision: IntentDecision) -> Dict[str, Any]:
        conversation = getattr(decision, "conversation", None)
        return conversation.to_dict() if conversation is not None else {}

    def _non_document_debug_response(
        self,
        decision: IntentDecision,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | None:
        response = self._non_document_response(decision, query, chat_history, user_profile)
        if response is None:
            return None

        context_preview = {
            "CONVERSATIONAL": "Conversational intent handled without retrieval.",
            "STUDENT_SUPPORT": "Student support intent handled without retrieval.",
            "CLARIFY": "Clarification requested before retrieval.",
            "OUT_OF_SCOPE": "Out-of-scope request rejected.",
        }.get(decision.intent, "Request handled without retrieval.")

        return self._debug_response(
            query,
            str(response["answer"]),
            decision,
            [],
            context_preview,
        )

    async def _yield_timed_stream(
        self,
        trace: _LatencyTrace,
        token_stream: AsyncGenerator[Any, None],
        *,
        status_messages: Optional[List[str]] = None,
        status_interval_seconds: float = 8.0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        first_token_seen = False
        started_at = time.perf_counter()
        messages = status_messages or []
        status_index = 0
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for token in token_stream:
                    await queue.put(("token", token))
                await queue.put(("done", None))
            except Exception as exc:
                await queue.put(("error", exc))

        producer = asyncio.create_task(produce())
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        queue.get(),
                        timeout=status_interval_seconds if messages else None,
                    )
                except asyncio.TimeoutError:
                    message = messages[min(status_index, len(messages) - 1)]
                    status_index += 1
                    yield {"type": "status", "message": message}
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    raise payload

                if isinstance(payload, dict) and payload.get("type") in {"stream", "replace"}:
                    event = dict(payload)
                else:
                    event = {"type": "stream", "token": str(payload)}

                has_visible_answer = bool(
                    (event.get("type") == "stream" and event.get("token"))
                    or (event.get("type") == "replace" and event.get("answer"))
                )
                if has_visible_answer and not first_token_seen:
                    trace.stage_ms["generation_to_first_token_ms"] = round(
                        (time.perf_counter() - started_at) * 1000,
                        1,
                    )
                    trace.mark("first_token_ms")
                    first_token_seen = True
                yield event
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except asyncio.CancelledError:
                    pass

        if not first_token_seen:
            trace.stage_ms["generation_to_first_token_ms"] = round(
                (time.perf_counter() - started_at) * 1000,
                1,
            )
            trace.mark("first_token_ms")
        trace.stage_ms["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 1)

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> Dict[str, Any]:
        trace = _LatencyTrace("chat.run", query)
        self._last_scored_documents = []
        self.generator.set_model_preference(model_preference)
        model_history = generation_history(chat_history)
        with trace.stage("intent_ms"):
            decision = self.intent_router.classify(query, chat_history=chat_history, user_profile=user_profile)

        retrieval_query: str | None = None
        retrieval: Dict[str, Any] | None = None
        if self._should_probe_documents(query, decision):
            with trace.stage("routing_retrieval_ms"):
                decision, retrieval_query, retrieval = self._arbitrate_student_support_route(
                    decision,
                    query,
                    model_history,
                    user_profile,
                )

        if decision.intent in {
            "CONVERSATIONAL",
            "STUDENT_SUPPORT",
            "CLARIFY",
            "OUT_OF_SCOPE",
        }:
            with trace.stage("generation_ms"):
                response = self._non_document_response(
                    decision,
                    query,
                    model_history,
                    user_profile,
                )
            trace.log(intent=decision.intent, source=decision.source)
            assert response is not None
            return response

        if retrieval_query is None or retrieval is None:
            with trace.stage("query_rewrite_ms"):
                retrieval_query = self._retrieval_query(
                    decision,
                    query,
                    model_history,
                    user_profile=user_profile,
                )
            with trace.stage("retrieval_ms"):
                retrieval = self._retrieve_document_context(
                    retrieval_query,
                    filters=decision.filters,
                )
        generation_query = self._generation_query(decision, query, retrieval_query)

        if self._is_empty_context(retrieval["context"]) or not retrieval["confidence"]["sufficient"]:
            # Before issuing a cold refusal, check whether the query is broad enough
            # to receive a useful student-support coaching answer instead.
            if self._should_demote_to_student_support(query, decision):
                demoted = self.intent_router.demote_to_student_support(decision, query)
                logger.info(
                    "Routing demoted by empty retrieval | from=%s to=%s query=%s",
                    decision.intent,
                    demoted.intent,
                    retrieval_query[:120],
                )
                with trace.stage("generation_ms"):
                    response = self._non_document_response(
                        demoted, query, model_history, user_profile
                    )
                trace.log(
                    intent=demoted.intent,
                    source=demoted.source,
                    docs=len(retrieval["documents"]),
                    confidence=retrieval["confidence"],
                    answered=True,
                    demotion="student_support",
                )
                assert response is not None
                return response
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                confidence=retrieval["confidence"],
                answered=False,
            )
            with trace.stage("generation_ms"):
                answer = self.generator.generate_document_refusal(
                    query,
                    model_history,
                    user_profile=user_profile,
                    reason=str(retrieval["confidence"].get("reason") or ""),
                )
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "context_used": False,
                "routing": decision.to_dict(),
                "conversation": self._conversation_payload(decision),
                "retrieval_query": retrieval_query,
                "retrieval_confidence": retrieval["confidence"],
                **self.generator.model_metadata(),
            }

        direct_curriculum = self._direct_curriculum_course_answer(generation_query, retrieval["documents"])
        if direct_curriculum is not None:
            with trace.stage("source_filter_ms"):
                sources = self._format_sources(direct_curriculum["documents"])
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                generation_docs=len(direct_curriculum["documents"]),
                generation_context_chars=0,
                generation_context_budget="curriculum_direct",
                sources=len(sources),
                confidence=retrieval["confidence"],
                grounding=direct_curriculum["grounding"],
                selected_model=None,
                fallback_used=False,
                answered=True,
            )
            return {
                "query": query,
                "answer": direct_curriculum["answer"],
                "sources": sources,
                "context_used": True,
                "routing": decision.to_dict(),
                "conversation": self._conversation_payload(decision),
                "retrieval_query": retrieval_query,
                "retrieval_confidence": retrieval["confidence"],
                "grounding": direct_curriculum["grounding"],
                **self.generator.model_metadata(),
            }

        direct_almanac = self._direct_almanac_event_answer(generation_query, retrieval["documents"])
        if direct_almanac is not None:
            with trace.stage("source_filter_ms"):
                sources = self._format_sources(direct_almanac["documents"])
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                generation_docs=len(direct_almanac["documents"]),
                generation_context_chars=0,
                generation_context_budget="almanac_direct",
                sources=len(sources),
                confidence=retrieval["confidence"],
                grounding=direct_almanac["grounding"],
                selected_model=None,
                fallback_used=False,
                answered=True,
            )
            return {
                "query": query,
                "answer": direct_almanac["answer"],
                "sources": sources,
                "context_used": True,
                "routing": decision.to_dict(),
                "conversation": self._conversation_payload(decision),
                "retrieval_query": retrieval_query,
                "retrieval_confidence": retrieval["confidence"],
                "grounding": direct_almanac["grounding"],
                **self.generator.model_metadata(),
            }

        generation_documents, generation_context, generation_budget = self._compact_generation_context(
            self._generation_context_documents(retrieval["documents"], generation_query),
            generation_query,
        )
        candidate_sources = self._format_sources(retrieval["documents"])
        try:
            with trace.stage("generation_ms"):
                generation_result = self.generator.generate_response(
                    generation_query,
                    generation_context,
                    model_history,
                    user_profile=user_profile,
                    documents=generation_documents,
                )
        except GenerationUnavailableError as exc:
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                generation_docs=len(generation_documents),
                generation_context_chars=len(generation_context),
                generation_context_budget=generation_budget,
                confidence=retrieval["confidence"],
                answered=False,
                error="generation_unavailable",
            )
            raise exc.attach_sources(candidate_sources)

        with trace.stage("source_filter_ms"):
            grounded_documents = self.generator.grounding.filter_documents(
                retrieval["documents"],
                generation_result["evidence_ids"],
            )
            sources = self._format_sources(grounded_documents)
        trace.log(
            intent=decision.intent,
            source=decision.source,
            docs=len(retrieval["documents"]),
            generation_docs=len(generation_documents),
            generation_context_chars=len(generation_context),
            generation_context_budget=generation_budget,
            sources=len(sources),
            confidence=retrieval["confidence"],
            grounding=generation_result["grounding"],
            selected_model=generation_result["selected_model"],
            fallback_used=generation_result["fallback_used"],
            answered=generation_result["grounding"].get("status") != "refused",
        )

        return {
            "query": query,
            "answer": generation_result["answer"],
            "sources": sources,
            "context_used": generation_result["context_used"],
            "routing": decision.to_dict(),
            "conversation": self._conversation_payload(decision),
            "retrieval_query": retrieval_query,
            "retrieval_confidence": retrieval["confidence"],
            "grounding": generation_result["grounding"],
            "requested_model": generation_result["requested_model"],
            "selected_model": generation_result["selected_model"],
            "fallback_used": generation_result["fallback_used"],
        }

    async def stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> AsyncGenerator[str, None]:
        """Backward-compatible verified text stream without replacement events."""
        answer = ""
        async for event in self.stream_events(
            query,
            chat_history,
            user_profile,
            model_preference,
        ):
            if event["type"] == "stream":
                answer += str(event["token"])
            elif event["type"] == "replace":
                answer = str(event.get("answer") or "")
        if answer:
            yield answer

    async def stream_events(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream answer text followed by server-validated source records."""
        trace = _LatencyTrace("chat.stream", query)
        self._last_scored_documents = []
        self.generator.set_model_preference(model_preference)
        model_history = generation_history(chat_history)
        with trace.stage("intent_ms"):
            decision = await run_in_threadpool(self.intent_router.classify, query, chat_history, user_profile)

        retrieval_query: str | None = None
        retrieval: Dict[str, Any] | None = None
        if self._should_probe_documents(query, decision):
            with trace.stage("routing_retrieval_ms"):
                decision, retrieval_query, retrieval = await run_in_threadpool(
                    self._arbitrate_student_support_route,
                    decision,
                    query,
                    model_history,
                    user_profile,
                )
        yield {
            "type": "routing",
            "routing": decision.to_dict(),
            "conversation": self._conversation_payload(decision),
        }

        if decision.intent == "CONVERSATIONAL":
            async for event in self._yield_timed_stream(
                trace,
                self.generator.stream_conversational(
                    query,
                    model_history,
                    user_profile=user_profile,
                ),
            ):
                yield event
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            trace.log(intent=decision.intent, source=decision.source)
            return
        if decision.intent == "STUDENT_SUPPORT":
            async for event in self._yield_timed_stream(
                trace,
                self.generator.stream_student_support(
                    query,
                    model_history,
                    user_profile=user_profile,
                ),
            ):
                yield event
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            trace.log(intent=decision.intent, source=decision.source)
            return
        if decision.intent == "CLARIFY":
            async for event in self._yield_timed_stream(
                trace,
                self.generator.stream_clarification_request(
                    query,
                    reason=decision.reason,
                ),
            ):
                yield event
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            trace.log(intent=decision.intent, source=decision.source)
            return
        if decision.intent == "OUT_OF_SCOPE":
            async for event in self._yield_timed_stream(
                trace,
                self.generator.stream_out_of_scope(
                    query,
                    model_history,
                    user_profile=user_profile,
                ),
            ):
                yield event
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            trace.log(intent=decision.intent, source=decision.source)
            return

        if retrieval_query is None or retrieval is None:
            with trace.stage("query_rewrite_ms"):
                yield {"type": "status", "message": "Preparing a focused search query..."}
                retrieval_query = await run_in_threadpool(
                    self._retrieval_query,
                    decision,
                    query,
                    model_history,
                    user_profile,
                )
            with trace.stage("retrieval_ms"):
                yield {"type": "status", "message": "Searching official documents..."}
                retrieval = await run_in_threadpool(
                    self._retrieve_document_context,
                    retrieval_query,
                    decision.filters,
                )
        generation_query = self._generation_query(decision, query, retrieval_query)

        if self._is_empty_context(retrieval["context"]) or not retrieval["confidence"]["sufficient"]:
            # Before issuing a cold refusal, check whether the query is broad enough
            # to receive a useful student-support coaching answer instead.
            if self._should_demote_to_student_support(query, decision):
                demoted = self.intent_router.demote_to_student_support(decision, query)
                logger.info(
                    "Routing demoted by empty retrieval | from=%s to=%s query=%s",
                    decision.intent,
                    demoted.intent,
                    retrieval_query[:120],
                )
                yield {
                    "type": "routing",
                    "routing": demoted.to_dict(),
                    "conversation": self._conversation_payload(demoted),
                }
                async for event in self._yield_timed_stream(
                    trace,
                    self.generator.stream_student_support(
                        query,
                        model_history,
                        user_profile=user_profile,
                    ),
                ):
                    yield event
                yield {"type": "model", **self.generator.model_metadata()}
                yield {"type": "sources", "sources": []}
                trace.log(
                    intent=demoted.intent,
                    source=demoted.source,
                    docs=len(retrieval["documents"]),
                    confidence=retrieval["confidence"],
                    answered=True,
                    demotion="student_support",
                )
                return
            async for event in self._yield_timed_stream(
                trace,
                self.generator.stream_document_refusal(
                    query,
                    model_history,
                    user_profile=user_profile,
                    reason=str(retrieval["confidence"].get("reason") or ""),
                ),
            ):
                yield event
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                confidence=retrieval["confidence"],
                answered=False,
            )
            return

        direct_curriculum = self._direct_curriculum_course_answer(generation_query, retrieval["documents"])
        if direct_curriculum is not None:
            trace.mark("first_token_ms")
            trace.stage_ms["generation_to_first_token_ms"] = 0.0
            trace.stage_ms["generation_ms"] = 0.0
            yield {"type": "stream", "token": direct_curriculum["answer"]}
            with trace.stage("source_filter_ms"):
                sources = self._format_sources(direct_curriculum["documents"])
            yield {"type": "sources", "sources": sources}
            yield {"type": "model", **self.generator.model_metadata()}
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                generation_docs=len(direct_curriculum["documents"]),
                generation_context_chars=0,
                generation_context_budget="curriculum_direct",
                sources=len(sources),
                confidence=retrieval["confidence"],
                grounding=direct_curriculum["grounding"],
                selected_model=None,
                fallback_used=False,
                answered=True,
            )
            return

        direct_almanac = self._direct_almanac_event_answer(generation_query, retrieval["documents"])
        if direct_almanac is not None:
            trace.mark("first_token_ms")
            trace.stage_ms["generation_to_first_token_ms"] = 0.0
            trace.stage_ms["generation_ms"] = 0.0
            yield {"type": "stream", "token": direct_almanac["answer"]}
            with trace.stage("source_filter_ms"):
                sources = self._format_sources(direct_almanac["documents"])
            yield {"type": "sources", "sources": sources}
            yield {"type": "model", **self.generator.model_metadata()}
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                generation_docs=len(direct_almanac["documents"]),
                generation_context_chars=0,
                generation_context_budget="almanac_direct",
                sources=len(sources),
                confidence=retrieval["confidence"],
                grounding=direct_almanac["grounding"],
                selected_model=None,
                fallback_used=False,
                answered=True,
            )
            return

        generation_documents, generation_context, generation_budget = self._compact_generation_context(
            self._generation_context_documents(retrieval["documents"], generation_query),
            generation_query,
        )
        candidate_sources = self._format_sources(retrieval["documents"])
        try:
            yield {"type": "status", "message": "Drafting a grounded answer..."}
            async for event in self._yield_timed_stream(
                trace,
                self.generator.stream_generate(
                generation_query,
                generation_context,
                model_history,
                    user_profile=user_profile,
                    documents=generation_documents,
                ),
                status_messages=[
                    "Checking the draft against retrieved sources...",
                    "Local model is still working on the grounded answer...",
                    "Finalizing the verified response...",
                ],
            ):
                yield event
        except GenerationUnavailableError as exc:
            trace.log(
                intent=decision.intent,
                source=decision.source,
                docs=len(retrieval["documents"]),
                generation_docs=len(generation_documents),
                generation_context_chars=len(generation_context),
                generation_context_budget=generation_budget,
                confidence=retrieval["confidence"],
                answered=False,
                error="generation_unavailable",
            )
            raise exc.attach_sources(candidate_sources)

        with trace.stage("source_filter_ms"):
            grounding = self.generator.grounding_metadata()
            grounded_documents = self.generator.grounding.filter_documents(
                retrieval["documents"],
                grounding["evidence_ids"],
            )
            sources = self._format_sources(grounded_documents)
        yield {"type": "sources", "sources": sources}
        yield {"type": "model", **self.generator.model_metadata()}
        trace.log(
            intent=decision.intent,
            source=decision.source,
            docs=len(retrieval["documents"]),
            generation_docs=len(generation_documents),
            generation_context_chars=len(generation_context),
            generation_context_budget=generation_budget,
            sources=len(sources),
            confidence=retrieval["confidence"],
            grounding=grounding,
            selected_model=self.generator.model_metadata()["selected_model"],
            fallback_used=self.generator.model_metadata()["fallback_used"],
            answered=grounding.get("status") != "refused",
        )

    def run_debug(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> Dict[str, Any]:
        self._last_scored_documents = []
        self.generator.set_model_preference(model_preference)
        model_history = generation_history(chat_history)
        decision = self.intent_router.classify(query, chat_history=chat_history, user_profile=user_profile)

        decision, retrieval_query, retrieval = self._arbitrate_student_support_route(
            decision,
            query,
            model_history,
            user_profile,
        )

        response = self._non_document_debug_response(decision, query, model_history, user_profile)
        if response is not None:
            return response

        if retrieval_query is None or retrieval is None:
            retrieval_query = self._retrieval_query(
                decision,
                query,
                model_history,
                user_profile=user_profile,
            )
            retrieval = self._retrieve_document_context(
                retrieval_query,
                filters=decision.filters,
            )
        generation_query = self._generation_query(decision, query, retrieval_query)
        sources = self._format_sources(retrieval["documents"])
        grounding = self.generator.grounding_metadata()

        if self._is_empty_context(retrieval["context"]) or not retrieval["confidence"]["sufficient"]:
            answer = self.generator.generate_document_refusal(
                query,
                model_history,
                user_profile=user_profile,
                reason=str(retrieval["confidence"].get("reason") or ""),
            )
        else:
            direct_curriculum = self._direct_curriculum_course_answer(generation_query, retrieval["documents"])
            if direct_curriculum is not None:
                answer = direct_curriculum["answer"]
                grounding = direct_curriculum["grounding"]
                sources = self._format_sources(direct_curriculum["documents"])
            else:
                direct_almanac = self._direct_almanac_event_answer(
                    generation_query,
                    retrieval["documents"],
                )
                if direct_almanac is not None:
                    answer = direct_almanac["answer"]
                    grounding = direct_almanac["grounding"]
                    sources = self._format_sources(direct_almanac["documents"])
                else:
                    generation_documents, generation_context, _generation_budget = self._compact_generation_context(
                        self._generation_context_documents(retrieval["documents"], generation_query),
                        generation_query,
                    )
                    generation_result = self.generator.generate_response(
                        generation_query,
                        generation_context,
                        model_history,
                        user_profile=user_profile,
                        documents=generation_documents,
                    )
                    answer = generation_result["answer"]
                    grounding = generation_result["grounding"]
                    grounded_documents = self.generator.grounding.filter_documents(
                        retrieval["documents"],
                        generation_result["evidence_ids"],
                    )
                    sources = self._format_sources(grounded_documents)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "conversation": self._conversation_payload(decision),
            "debug": {
                "routing": decision.to_dict(),
                "retrieval_query": retrieval_query,
                "retrieval_confidence": retrieval["confidence"],
                "num_docs": len(retrieval["documents"]),
                "documents": retrieval["debug_documents"],
                "context_preview": retrieval["context"][:500],
                "grounding": grounding,
                "fallback_used": self.generator.model_metadata()["fallback_used"],
            },
        }

    def _retrieve_document_context(self, query: str, filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
        scored_docs = self.retrieval_service.retrieve_with_scores(query, filters=filters or None)
        scored_docs = self._supplement_curriculum_documents(query, scored_docs)
        self._last_scored_documents = list(scored_docs)
        documents = [doc for doc, _score in scored_docs]
        context = self.retrieval_service.format_context(documents)
        confidence = self._assess_retrieval(scored_docs, query)
        debug_documents = [
            {"content": doc.page_content[:200], "metadata": doc.metadata, "score": score}
            for doc, score in scored_docs
        ]
        return {
            "scored_docs": scored_docs,
            "documents": documents,
            "context": context,
            "confidence": confidence,
            "debug_documents": debug_documents,
        }

    def last_retrieval(self) -> List[Tuple[Document, float]]:
        """Return the retrieval result produced by the latest pipeline run."""
        return list(self._last_scored_documents)

    def _compact_generation_context(self, documents: List[Document], query: str) -> Tuple[List[Document], str, str]:
        budget = self._generation_context_budget(query)
        max_docs = max(1, budget["max_docs"])
        max_chars = max(1000, budget["max_chars"])
        max_chars_per_doc = max(300, budget["max_chars_per_doc"])
        compact_documents: List[Document] = []

        for document in documents[:max_docs]:
            content = " ".join(str(document.page_content or "").split())
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc].rsplit(" ", 1)[0] + " ...[truncated]"
            compact_documents.append(
                Document(
                    page_content=content,
                    metadata=dict(document.metadata or {}),
                )
            )

        context = self.retrieval_service.format_context(
            compact_documents,
            max_chars=max_chars,
        )
        return compact_documents, context, budget["tier"]

    def _generation_context_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        if self.generator._is_procedure_query(query):
            return self._procedure_context_documents(documents, query)
        if self.generator._is_enumeration_query(query):
            return self._enumeration_context_documents(documents, query)
        if self.generator._CURRICULUM_LIST_QUERY_RE.search(query or ""):
            return self._curriculum_context_documents(documents, query)
        return documents

    def _curriculum_context_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        """Promote contiguous chunks around the strongest course/curriculum hit to capture full tables."""
        anchor = next(
            (
                document
                for document in documents
                if document.metadata.get("source_type") == "document"
                and not document.metadata.get("neighbor_expansion")
                and isinstance(document.metadata.get("chunk_index"), int)
            ),
            None,
        )
        if anchor is None:
            return documents

        try:
            document_id = UUID(str(anchor.metadata.get("document_id")))
            anchor_index = int(anchor.metadata["chunk_index"])
        except (TypeError, ValueError, KeyError):
            return documents

        window = 3
        with self.retrieval_service.database_session() as db:
            rows = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.chunk_index.between(
                        max(0, anchor_index - window),
                        anchor_index + window,
                    ),
                )
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )
        if not rows:
            return documents

        promoted: list[Document] = []
        promoted_ids: set[str] = set()
        for row in rows:
            chunk_id = str(row.id)
            metadata = {
                **dict(anchor.metadata or {}),
                **dict(row.metadata_ or {}),
                "chunk_id": chunk_id,
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "page_number": row.page_number,
                "section_expansion": True,
            }
            promoted.append(Document(page_content=row.chunk_text, metadata=metadata))
            promoted_ids.add(chunk_id)

        promoted.extend(
            document
            for document in documents
            if str(document.metadata.get("chunk_id")) not in promoted_ids
        )
        return promoted

    def _enumeration_context_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        """Promote the strongest list-bearing section for policy questions."""
        query_terms = set(significant_tokens(query))
        lowered_query = (query or "").lower()
        if "proper" in query_terms or "allowed" in query_terms:
            query_terms.add("appropriate")
        if "not allowed" in lowered_query or "forbidden" in query_terms:
            query_terms.update({"inappropriate", "prohibited"})

        generic_terms = {
            "code", "codes", "dress", "dressing", "rule", "rules",
            "requirement", "requirements", "student", "students",
        }
        specific_terms = query_terms - generic_terms
        topic_match = self.generator._ENUMERATION_TOPIC_RE.search(query or "")
        topic_phrase = topic_match.group(0).lower() if topic_match else ""

        candidates: list[tuple[tuple[int, int, int, int, int], Document]] = []
        for rank, document in enumerate(documents):
            metadata = document.metadata or {}
            if (
                metadata.get("source_type") != "document"
                or metadata.get("neighbor_expansion")
                or not isinstance(metadata.get("chunk_index"), int)
            ):
                continue

            content = str(document.page_content or "")
            content_terms = token_set(content)
            list_item_count = len(
                self.generator._EVIDENCE_LIST_ITEM_RE.findall(content)
            )
            topic_position = content.lower().rfind(topic_phrase) if topic_phrase else -1
            list_items_after_topic = (
                len(self.generator._EVIDENCE_LIST_ITEM_RE.findall(content[topic_position:]))
                if topic_position >= 0
                else 0
            )
            score = (
                len(specific_terms & content_terms),
                len(query_terms & content_terms),
                min(list_items_after_topic, 12),
                min(list_item_count, 12),
                -rank,
            )
            candidates.append((score, document))

        if not candidates:
            return documents

        _score, anchor = max(candidates, key=lambda item: item[0])
        try:
            document_id = UUID(str(anchor.metadata.get("document_id")))
            anchor_index = int(anchor.metadata["chunk_index"])
        except (TypeError, ValueError, KeyError):
            return documents

        window = max(1, settings.GENERATION_PROCEDURE_SECTION_WINDOW)
        preferred_indexes = [anchor_index]
        preferred_indexes.extend(anchor_index + offset for offset in range(1, window + 1))
        preferred_indexes.extend(anchor_index - offset for offset in range(1, window + 1))
        preferred_indexes = [index for index in preferred_indexes if index >= 0]

        with self.retrieval_service.database_session() as db:
            rows = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.chunk_index.in_(preferred_indexes),
                )
                .all()
            )
        if not rows:
            return documents

        rows_by_index = {row.chunk_index: row for row in rows}
        promoted: list[Document] = []
        promoted_ids: set[str] = set()
        for index in preferred_indexes:
            row = rows_by_index.get(index)
            if row is None:
                continue
            chunk_id = str(row.id)
            metadata = {
                **dict(anchor.metadata or {}),
                **dict(row.metadata_ or {}),
                "chunk_id": chunk_id,
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "page_number": row.page_number,
                "section_expansion": True,
                "section_expansion_kind": "enumeration",
            }
            metadata.pop("neighbor_expansion", None)
            metadata.pop("neighbor_of_chunk_id", None)
            metadata.pop("neighbor_offset", None)
            promoted.append(Document(page_content=row.chunk_text, metadata=metadata))
            promoted_ids.add(chunk_id)

        promoted.extend(
            document
            for document in documents
            if str(document.metadata.get("chunk_id")) not in promoted_ids
        )
        return promoted

    def _procedure_context_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        """Promote contiguous chunks around the strongest procedure hit."""
        if not self.generator._is_procedure_query(query):
            return documents

        anchor = next(
            (
                document
                for document in documents
                if document.metadata.get("source_type") == "document"
                and not document.metadata.get("neighbor_expansion")
                and isinstance(document.metadata.get("chunk_index"), int)
            ),
            None,
        )
        if anchor is None:
            return documents

        try:
            document_id = UUID(str(anchor.metadata.get("document_id")))
            anchor_index = int(anchor.metadata["chunk_index"])
        except (TypeError, ValueError, KeyError):
            return documents

        window = max(1, settings.GENERATION_PROCEDURE_SECTION_WINDOW)
        with self.retrieval_service.database_session() as db:
            rows = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.chunk_index.between(
                        max(0, anchor_index - window),
                        anchor_index + window,
                    ),
                )
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )
        if not rows:
            return documents

        promoted: list[Document] = []
        promoted_ids: set[str] = set()
        for row in rows:
            chunk_id = str(row.id)
            metadata = {
                **dict(anchor.metadata or {}),
                **dict(row.metadata_ or {}),
                "chunk_id": chunk_id,
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "page_number": row.page_number,
                "section_expansion": True,
            }
            metadata.pop("neighbor_expansion", None)
            metadata.pop("neighbor_of_chunk_id", None)
            metadata.pop("neighbor_offset", None)
            promoted.append(Document(page_content=row.chunk_text, metadata=metadata))
            promoted_ids.add(chunk_id)

        promoted.extend(
            document
            for document in documents
            if str(document.metadata.get("chunk_id")) not in promoted_ids
        )
        return promoted

    def _direct_curriculum_course_answer(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, Any] | None:
        if not self._CURRICULUM_LIST_QUERY_RE.search(query or ""):
            return None

        rows: list[Document] = []
        seen_codes: set[str] = set()
        for document in documents:
            metadata = document.metadata or {}
            if metadata.get("category") != "curriculum" or metadata.get("record_type") != "course":
                continue
            if not metadata.get("curriculum_metadata_match"):
                continue

            course_code = str(metadata.get("course_code") or "").strip()
            course_title = str(metadata.get("course_title") or "").strip()
            if not course_code or not course_title or course_code in seen_codes:
                continue
            seen_codes.add(course_code)
            rows.append(document)

        if not rows:
            return None

        rows.sort(key=self._curriculum_course_sort_key)
        first_metadata = rows[0].metadata or {}
        programme = str(first_metadata.get("programme") or "the requested programme").strip()
        year = str(first_metadata.get("year_of_study") or "").strip()
        semester = str(first_metadata.get("semester") or "").strip()
        heading_parts = [programme]
        if year:
            heading_parts.append(f"Year {year}")
        if semester:
            heading_parts.append(f"Semester {semester}")

        lines = [
            f"Here are the courses for {', '.join(heading_parts)}:",
            "",
            "| Course Code | Course Title | Status | Credits |",
            "| --- | --- | --- | --- |",
        ]
        for document in rows:
            metadata = document.metadata or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        self._markdown_cell(metadata.get("course_code")),
                        self._markdown_cell(metadata.get("course_title")),
                        self._markdown_cell(metadata.get("status")),
                        self._markdown_cell(metadata.get("credits")),
                    ]
                )
                + " |"
            )

        grounding_service = GroundingService()
        evidence_ids = list(
            dict.fromkeys(grounding_service.evidence_id(document) for document in rows)
        )
        grounding = {
            "coverage": "full",
            "evidence_ids": evidence_ids,
            "claim_count": len(rows),
            "supported_claim_count": len(rows),
            "repaired": False,
            "status": "grounded",
        }
        return {
            "answer": "\n".join(lines),
            "documents": rows,
            "grounding": grounding,
        }

    def _direct_almanac_event_answer(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, Any] | None:
        if not self._ALMANAC_DATE_QUERY_RE.search(query or ""):
            return None

        for document in documents:
            metadata = document.metadata or {}
            if metadata.get("category") != "academic_calendar" or metadata.get("record_type") != "almanac_event":
                continue

            event_date = str(metadata.get("date") or "").strip()
            activity = self._almanac_activity(document.page_content)
            if not event_date or not activity:
                continue
            if not self._almanac_event_matches_query(query, activity, metadata):
                continue

            answer = f"{activity} is on {event_date}."
            grounding_service = GroundingService()
            evidence_id = grounding_service.evidence_id(document)
            grounding = {
                "coverage": "full",
                "evidence_ids": [evidence_id],
                "claim_count": 1,
                "supported_claim_count": 1,
                "repaired": False,
                "status": "grounded",
            }
            return {
                "answer": answer,
                "documents": [document],
                "grounding": grounding,
            }

        return None

    @staticmethod
    def _almanac_activity(content: str) -> str:
        match = re.search(r"(?im)^Activity:\s*(.+)$", str(content or ""))
        if not match:
            return ""
        return " ".join(match.group(1).strip(" .").split())

    @classmethod
    def _almanac_event_matches_query(
        cls,
        query: str,
        activity: str,
        metadata: Dict[str, Any],
    ) -> bool:
        query_text = cls._normalize_almanac_match_text(query)
        activity_text = cls._normalize_almanac_match_text(activity)
        query_terms = set(tokenize(query_text)) - cls._ALMANAC_EVENT_STOPWORDS
        activity_terms = set(tokenize(activity_text))

        if cls._asks_for_start(query_text) and not cls._asks_for_start(activity_text):
            return False
        if cls._asks_for_end(query_text) and not cls._asks_for_end(activity_text):
            return False
        if cls._mentions_examinations(query_text) and not (
            cls._mentions_examinations(activity_text)
            or metadata.get("event_type") == "examination"
        ):
            return False
        if cls._mentions_supplementary_or_special(query_text) and not cls._mentions_supplementary_or_special(activity_text):
            return False
        if cls._mentions_semester_two(query_text) and not cls._mentions_semester_two(activity_text):
            return False
        if cls._mentions_semester_one(query_text) and not cls._mentions_semester_one(activity_text):
            return False
        if "nondegree" in query_terms and "nondegree" not in activity_terms:
            return False
        if "degree" in query_terms and "degree" not in activity_terms:
            return False

        matched_terms = query_terms & activity_terms
        required_terms = {
            term
            for term in query_terms
            if term not in {
                "begin", "begins", "beginning", "end", "ends", "examination",
                "examinations", "exam", "exams", "finish", "finishes", "start",
                "starts", "starting",
            }
        }
        if required_terms:
            return len(matched_terms & required_terms) >= min(2, len(required_terms))
        return bool(matched_terms)

    @staticmethod
    def _normalize_almanac_match_text(value: str) -> str:
        text = str(value or "").casefold()
        text = re.sub(r"\bsemester\s+(?:ii|2|two|second)\b", "semester_two", text)
        text = re.sub(r"\b(?:muhula|semester)\s+(?:wa\s+)?(?:pili|2)\b", "semester_two", text)
        text = re.sub(r"\bsemester\s+(?:i|1|one|first)\b", "semester_one", text)
        text = re.sub(r"\b(?:muhula|semester)\s+(?:wa\s+)?(?:kwanza|1)\b", "semester_one", text)
        text = re.sub(r"\bnon\s*-\s*degree\b", "nondegree", text)
        text = re.sub(r"\b(?:zisizo\s+za\s+)?shahada\b", "degree", text)
        text = text.replace("programmes", "programs").replace("programme", "program")
        text = text.replace("examinations", "examination").replace("exams", "exam")
        text = text.replace("mitihani", "examination").replace("mtihani", "exam")
        text = text.replace("tarehe", "date").replace("lini", "when")
        text = text.replace("kuanza", "start").replace("unaanza", "start").replace("anza", "start")
        text = text.replace("mwisho", "end").replace("malizika", "end")
        text = text.replace("nyongeza", "supplementary").replace("maalum", "special")
        return text.replace("_", " ")

    @staticmethod
    def _asks_for_start(text: str) -> bool:
        return bool(re.search(r"\b(?:start|starts|starting|begin|begins|beginning|anza|kuanza|unaanza)\b", text))

    @staticmethod
    def _asks_for_end(text: str) -> bool:
        return bool(re.search(r"\b(?:end|ends|ending|finish|finishes|finishing|mwisho|malizika)\b", text))

    @staticmethod
    def _mentions_examinations(text: str) -> bool:
        return bool(re.search(r"\b(?:examination|exam|mtihani|mitihani)\b", text))

    @staticmethod
    def _mentions_supplementary_or_special(text: str) -> bool:
        return bool(re.search(r"\b(?:supplementary|special|nyongeza|maalum)\b", text))

    @staticmethod
    def _mentions_semester_two(text: str) -> bool:
        return "semester two" in text

    @staticmethod
    def _mentions_semester_one(text: str) -> bool:
        return "semester one" in text

    @staticmethod
    def _curriculum_course_sort_key(document: Document) -> tuple[int, str]:
        metadata = document.metadata or {}
        try:
            chunk_index = int(metadata.get("chunk_index"))
        except (TypeError, ValueError):
            chunk_index = 0
        return (chunk_index, str(metadata.get("course_code") or ""))

    @staticmethod
    def _markdown_cell(value: Any) -> str:
        text = " ".join(str(value or "").split())
        return text.replace("|", "\\|")

    def _generation_context_budget(self, query: str) -> Dict[str, Any]:
        if self._COMPLEX_GENERATION_QUERY_RE.search(query or ""):
            return {
                "tier": "complex",
                "max_docs": settings.GENERATION_CONTEXT_MAX_DOCS,
                "max_chars": settings.GENERATION_CONTEXT_MAX_CHARS,
                "max_chars_per_doc": settings.GENERATION_CONTEXT_MAX_CHARS_PER_DOC,
            }
        if self._PROCEDURE_GENERATION_QUERY_RE.search(query or ""):
            return {
                "tier": "procedure",
                "max_docs": settings.GENERATION_PROCEDURE_CONTEXT_MAX_DOCS,
                "max_chars": settings.GENERATION_PROCEDURE_CONTEXT_MAX_CHARS,
                "max_chars_per_doc": settings.GENERATION_PROCEDURE_CONTEXT_MAX_CHARS_PER_DOC,
            }
        if self._CURRICULUM_LIST_QUERY_RE.search(query or ""):
            return {
                "tier": "curriculum",
                "max_docs": max(
                    settings.GENERATION_CONTEXT_MAX_DOCS,
                    self._CURRICULUM_METADATA_MAX_ROWS,
                ),
                "max_chars": max(settings.GENERATION_CONTEXT_MAX_CHARS, 12000),
                "max_chars_per_doc": min(settings.GENERATION_CONTEXT_MAX_CHARS_PER_DOC, 900),
            }
        return {
            "tier": "standard",
            "max_docs": min(settings.GENERATION_CONTEXT_MAX_DOCS, 5),
            "max_chars": min(settings.GENERATION_CONTEXT_MAX_CHARS, 5200),
            "max_chars_per_doc": min(settings.GENERATION_CONTEXT_MAX_CHARS_PER_DOC, 1000),
        }

    def _assess_retrieval(self, scored_docs: List[Tuple[Document, float]], query: str) -> Dict[str, Any]:
        if not scored_docs:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "No chunks were retrieved.",
                "top_score": 0.0,
                "top_rrf_score": 0.0,
                "lexical_overlap": 0,
                "matched_sources": [],
                "tier": "low",
            }

        central_docs = [(doc, score) for doc, score in scored_docs if not doc.metadata.get("neighbor_expansion")] or scored_docs
        query_terms = self._expanded_query_terms(query)
        top_doc, top_score = central_docs[0]
        top_rrf = self._safe_float(top_doc.metadata.get("rrf_score"), default=top_score)
        top_sources = self._sources(top_doc)
        reranker = str(top_doc.metadata.get("reranker") or "rrf")

        top_candidates = central_docs[:5]
        overlaps = [self._lexical_overlap(query_terms, doc) for doc, _score in top_candidates]
        best_overlap = max(overlaps) if overlaps else 0
        has_hybrid_match = any({"dense", "sparse"}.issubset(self._sources(doc)) for doc, _score in top_candidates)
        best_sparse_rank = self._best_rank(top_candidates, "sparse_rank")
        best_dense_rank = self._best_rank(top_candidates, "dense_rank")
        sparse_strong = best_sparse_rank is not None and best_sparse_rank <= 10
        # Dense-strong no longer requires lexical overlap: a high-quality vector
        # match (rank <= 8) is strong signal even when the user paraphrases.
        dense_strong = best_dense_rank is not None and best_dense_rank <= 8

        # Detect sparse-only degraded mode: the embedding service was down so
        # every result carries dense_retrieval_degraded=True in its metadata.
        # In this mode we widen thresholds because:
        #   - has_hybrid_match is always False (no dense retrieval ran)
        #   - RRF scores are lower (only sparse contributes, max ≈ 1/(60+1))
        #   - sparse_strong is the only strong signal available
        degraded_mode = all(
            doc.metadata.get("dense_retrieval_degraded") for doc, _score in top_candidates
        ) if top_candidates else False

        if reranker == "jina":
            sufficient = top_score >= 0.10 or has_hybrid_match or sparse_strong or dense_strong
        elif reranker == "local_lexical":
            if degraded_mode:
                # Lexical reranker scores against sparse-only results: accept
                # rank-1 result with any overlap or a decent rerank score.
                sufficient = top_score >= 0.6 or sparse_strong or best_overlap >= 1
            else:
                sufficient = top_score >= 1.0 or has_hybrid_match or sparse_strong or best_overlap >= 2
        else:
            # RRF path.
            if degraded_mode:
                # Sparse-only: max possible RRF score is 1/(60+1) ≈ 0.0164.
                # Accept if FTS found a strong match (rank ≤ 10) or any result
                # has lexical overlap with the query.
                sufficient = sparse_strong or best_overlap >= 1 or top_rrf >= 0.010
            else:
                sufficient = (
                    has_hybrid_match
                    or sparse_strong
                    or dense_strong
                    or (top_rrf >= 0.013 and best_overlap > 0)
                    or (top_rrf >= 0.015 and best_dense_rank is not None and best_dense_rank <= 15)
                )

        confidence = self._confidence_value(
            top_score=top_score,
            top_rrf=top_rrf,
            best_overlap=best_overlap,
            has_hybrid_match=has_hybrid_match,
            sparse_strong=sparse_strong,
            dense_strong=dense_strong,
            reranker=reranker,
        )
        if not sufficient:
            tier = "low"
        elif confidence >= 0.72 and (
            has_hybrid_match or (sparse_strong and dense_strong)
        ):
            tier = "high"
        else:
            tier = "medium"

        reasons: list[str] = []
        if has_hybrid_match:
            reasons.append("matched by both dense and sparse retrieval")
        if sparse_strong:
            reasons.append(f"sparse rank {best_sparse_rank}")
        if dense_strong:
            reasons.append(f"dense rank {best_dense_rank} with lexical overlap")
        if best_overlap:
            reasons.append(f"{best_overlap} query terms overlapped retrieved text")
        if not reasons:
            reasons.append("top chunks had weak lexical/retrieval signals")

        return {
            "sufficient": bool(sufficient),
            "confidence": round(confidence, 4),
            "reason": "; ".join(reasons),
            "top_score": round(float(top_score), 6),
            "top_rrf_score": round(float(top_rrf), 6),
            "lexical_overlap": best_overlap,
            "matched_sources": sorted(top_sources),
            "tier": tier,
            "reranker": reranker,
            "best_dense_rank": best_dense_rank,
            "best_sparse_rank": best_sparse_rank,
        }

    def _expanded_query_terms(self, query: str) -> set[str]:
        terms = set(significant_tokens(query))
        if not terms:
            terms = {term for term in tokenize(query) if len(term) > 2}
        try:
            aliases = self.retrieval_service.alias_expansions(query)
        except Exception:
            aliases = {}
        for term, values in aliases.items():
            terms.update(tokenize(term))
            for value in values:
                terms.update(tokenize(value))
        return terms

    def _lexical_overlap(self, query_terms: set[str], doc: Document) -> int:
        haystack = f"{doc.page_content} {metadata_search_text(doc.metadata)}"
        return len(query_terms & token_set(haystack))

    def _sources(self, doc: Document) -> set[str]:
        return {str(source) for source in (doc.metadata.get("retrieval_sources") or [])}

    def _best_rank(self, docs: List[Tuple[Document, float]], key: str) -> int | None:
        ranks = [doc.metadata.get(key) for doc, _score in docs if isinstance(doc.metadata.get(key), int)]
        return min(ranks) if ranks else None

    def _confidence_value(
        self,
        top_score: float,
        top_rrf: float,
        best_overlap: int,
        has_hybrid_match: bool,
        sparse_strong: bool,
        dense_strong: bool,
        reranker: str,
    ) -> float:
        confidence = 0.2
        if reranker == "jina":
            confidence = max(confidence, min(0.95, float(top_score)))
        elif reranker == "local_lexical":
            confidence = max(confidence, min(0.95, float(top_score) / 6.0))
        else:
            confidence = max(confidence, min(0.8, float(top_rrf) * 25.0))
        if has_hybrid_match:
            confidence += 0.2
        if sparse_strong:
            confidence += 0.15
        if dense_strong:
            confidence += 0.1
        confidence += min(0.2, best_overlap * 0.04)
        return min(1.0, confidence)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _supplement_curriculum_documents(
        self,
        query: str,
        scored_docs: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        constraints = self._curriculum_constraints_from_query(query)
        if not constraints or not constraints.get("programme"):
            return scored_docs

        metadata = DocumentChunk.metadata_
        with self.retrieval_service.database_session() as db:
            db_query = (
                db.query(DocumentChunk)
                .options(joinedload(DocumentChunk.document))
                .filter(metadata["category"].astext == "curriculum")
                .filter(metadata["record_type"].astext == "course")
                .filter(metadata["programme"].astext == constraints["programme"])
            )
            if constraints.get("year_of_study"):
                db_query = db_query.filter(metadata["year_of_study"].astext == constraints["year_of_study"])
            if constraints.get("semester"):
                db_query = db_query.filter(metadata["semester"].astext == constraints["semester"])

            rows = (
                db_query.order_by(DocumentChunk.chunk_index.asc())
                .limit(self._CURRICULUM_METADATA_MAX_ROWS)
                .all()
            )
        if not rows:
            return scored_docs

        supplements: List[Tuple[Document, float]] = []
        for row in rows:
            chunk_id = str(row.id)
            row_metadata: Dict[str, Any] = dict(
                cast(Dict[str, Any], row.metadata_ or {})
            )
            row_metadata.update(
                {
                    "document_id": str(row.document_id),
                    "chunk_id": chunk_id,
                    "chunk_index": row.chunk_index,
                    "source": row.document.filename if row.document else None,
                    "source_type": "document",
                    "document_title": row.document.title if row.document else None,
                    "source_url": row.document.source_url if row.document else None,
                    "document_status": row.document.status if row.document else None,
                    "retrieval_sources": ["metadata"],
                    "curriculum_metadata_match": True,
                }
            )
            supplements.append((Document(page_content=row.chunk_text, metadata=row_metadata), 1.0))

        if not supplements:
            return scored_docs

        supplement_ids = {str(doc.metadata.get("chunk_id")) for doc, _score in supplements}
        remaining_scored_docs = [
            (doc, score)
            for doc, score in scored_docs
            if str(doc.metadata.get("chunk_id")) not in supplement_ids
        ]

        logger.info(
            format_log_event(
                "Curriculum metadata supplement",
                matched=len(supplements),
                programme=constraints.get("programme"),
                year=constraints.get("year_of_study"),
                semester=constraints.get("semester"),
                query=query[:120],
            )
        )
        return supplements + remaining_scored_docs

    def _curriculum_constraints_from_query(self, query: str) -> Dict[str, str]:
        if not self._CURRICULUM_LIST_QUERY_RE.search(query or ""):
            return {}

        year_of_study = self._year_of_study_from_query(query)
        semester = self._semester_from_query(query)
        programme = self._programme_from_query(query)
        if not programme:
            return {}

        constraints = {"programme": programme}
        if year_of_study:
            constraints["year_of_study"] = year_of_study
        if semester:
            constraints["semester"] = semester
        return constraints

    def _programme_from_query(self, query: str) -> str | None:
        query_tokens = [
            token
            for token in significant_tokens(query)
            if token not in self._CURRICULUM_STOPWORDS and len(token) > 2
        ]
        if not query_tokens:
            return None

        metadata = DocumentChunk.metadata_
        with self.retrieval_service.database_session() as db:
            rows = (
                db.query(
                    metadata["programme"].astext.label("programme"),
                    metadata["programme_acronym"].astext.label("programme_acronym"),
                )
                .filter(metadata["category"].astext == "curriculum")
                .filter(metadata["record_type"].astext == "course")
                .filter(metadata["programme"].astext.isnot(None))
                .distinct()
                .all()
            )

        lowered_query = (query or "").lower()
        query_token_set = set(query_tokens)
        query_acronyms = {
            acronym.lower()
            for acronym in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", query or "")
        }

        # 1. Exact phrase & clean field matching (e.g. "software engineering" -> "Bachelor of Science in Software Engineering")
        phrase_matches = []
        for row in rows:
            prog_name = (row.programme or "").strip()
            prog_lower = prog_name.lower()
            clean_field = re.sub(
                r"\b(?:bachelor|master|diploma|certificate|science|arts|degree|of|in|with|honours|bsc|msc|ba|ma|diploma)\b",
                "",
                prog_lower,
                flags=re.I,
            )
            clean_field = re.sub(r"[\(\)]", "", clean_field).strip()

            if prog_lower and prog_lower in lowered_query:
                phrase_matches.append((len(prog_lower), prog_name))
            elif clean_field and len(clean_field) >= 4 and clean_field in lowered_query:
                phrase_matches.append((len(clean_field), prog_name))

        if phrase_matches:
            phrase_matches.sort(key=lambda item: item[0], reverse=True)
            return phrase_matches[0][1]

        # 2. Acronym matching
        exact_acronym_matches = []
        for row in rows:
            programme = row.programme or ""
            acronym_tokens = set(tokenize(row.programme_acronym or ""))
            if query_token_set & acronym_tokens or query_acronyms & acronym_tokens:
                programme_tokens = set(tokenize(programme))
                exact_acronym_matches.append(
                    (
                        bool((query_token_set | query_acronyms) & programme_tokens),
                        programme,
                    )
                )
        if exact_acronym_matches:
            exact_acronym_matches.sort(reverse=True)
            if exact_acronym_matches[0][0]:
                return exact_acronym_matches[0][1]

        alias_tokens = self._programme_alias_tokens_from_query(query)
        if alias_tokens:
            for row in rows:
                programme = row.programme or ""
                haystack_tokens = set(tokenize(f"{programme} {row.programme_acronym or ''}"))
                if alias_tokens <= haystack_tokens:
                    return programme

        if query_acronyms:
            return None

        # 3. Specific token scoring (ignore generic words like bachelor, science, degree)
        generic_tokens = {
            "bachelor", "master", "diploma", "certificate", "science", "arts",
            "degree", "programme", "program", "student", "students", "udom",
            "university", "dodoma", "course", "courses", "year", "semester",
            "one", "two", "first", "second", "third", "fourth", "bsc", "msc", "ba", "ma",
        }
        specific_query_tokens = set(query_tokens) - generic_tokens
        if not specific_query_tokens:
            specific_query_tokens = set(query_tokens)

        best_programme: str | None = None
        best_score = 0
        for row in rows:
            programme = row.programme or ""
            haystack_tokens = set(tokenize(f"{programme} {row.programme_acronym or ''}"))
            score = sum(1 for token in specific_query_tokens if token in haystack_tokens)
            if score > best_score:
                best_score = score
                best_programme = programme

        return best_programme if best_score >= 1 else None

    def _programme_alias_tokens_from_query(self, query: str) -> set[str]:
        query_tokens = set(tokenize(query))
        alias_expansions: dict[str, list[str]] = {}
        try:
            alias_expansions.update(self.retrieval_service.alias_expansions(query))
        except Exception:
            pass
        for term, aliases in FALLBACK_ALIAS_EXPANSIONS.items():
            values = [term, *aliases]
            if query_tokens & set(tokenize(" ".join(values))):
                alias_expansions.setdefault(term, aliases)

        best_tokens: set[str] = set()
        for term, aliases in alias_expansions.items():
            term_tokens = set(tokenize(term))
            all_alias_tokens = set(tokenize(" ".join([term, *aliases])))
            if not (query_tokens & (term_tokens | all_alias_tokens)):
                continue
            for alias in aliases:
                tokens = {
                    token
                    for token in significant_tokens(alias)
                    if token not in self._CURRICULUM_STOPWORDS and len(token) > 2
                }
                if len(tokens) > len(best_tokens):
                    best_tokens = tokens
        return best_tokens

    @staticmethod
    def _year_of_study_from_query(query: str) -> str | None:
        lowered = (query or "").lower()
        patterns = [
            (r"\b(?:year\s+one|first\s+year|1st\s+year|year\s+1|1\s+year|mwaka\s+(?:wa\s+)?kwanza|mwaka\s+1|1\s+mwaka)\b", "1"),
            (r"\b(?:year\s+two|second\s+year|2nd\s+year|year\s+2|2\s+year|mwaka\s+(?:wa\s+)?pili|mwaka\s+2|2\s+mwaka)\b", "2"),
            (r"\b(?:year\s+three|third\s+year|3rd\s+year|year\s+3|3\s+year|mwaka\s+(?:wa\s+)?tatu|mwaka\s+3|3\s+mwaka)\b", "3"),
            (r"\b(?:year\s+four|fourth\s+year|4th\s+year|year\s+4|4\s+year|mwaka\s+(?:wa\s+)?nne|mwaka\s+4|4\s+mwaka)\b", "4"),
            (r"\b(?:year\s+five|fifth\s+year|5th\s+year|year\s+5|5\s+year|mwaka\s+(?:wa\s+)?tano|mwaka\s+5|5\s+mwaka)\b", "5"),
        ]
        return next((value for pattern, value in patterns if re.search(pattern, lowered)), None)

    @staticmethod
    def _semester_from_query(query: str) -> str | None:
        lowered = (query or "").lower()
        patterns = [
            (r"\b(?:semester\s+one|semester\s+1|sem\s+one|sem\s+1|first\s+semester|1st\s+semester|muhula\s+(?:wa\s+)?kwanza|semester\s+(?:ya\s+|wa\s+)?kwanza|muhula\s+1)\b", "1"),
            (r"\b(?:semester\s+two|semester\s+2|sem\s+two|sem\s+2|second\s+semester|2nd\s+semester|muhula\s+(?:wa\s+)?pili|semester\s+(?:ya\s+|wa\s+)?pili|muhula\s+2)\b", "2"),
        ]
        return next((value for pattern, value in patterns if re.search(pattern, lowered)), None)

    def _debug_response(
        self,
        query: str,
        answer: str,
        decision: IntentDecision,
        documents: List[Dict[str, Any]],
        context_preview: str,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": answer,
            "conversation": self._conversation_payload(decision),
            "debug": {
                "routing": decision.to_dict(),
                "retrieval_query": decision.standalone_query,
                "retrieval_confidence": None,
                "num_docs": len(documents),
                "documents": documents,
                "context_preview": context_preview,
                "fallback_used": False,
            },
        }

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        return format_source_records(documents)
