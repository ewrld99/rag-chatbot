from __future__ import annotations

import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from starlette.concurrency import run_in_threadpool

from app.services.alias_expansion_service import AliasExpansionService
from app.services.conversation_service import static_conversational_response
from app.services.generation_resilience import GenerationUnavailableError
from app.services.generation_service import GenerationService
from app.services.intent_router import IntentDecision, IntentRouter
from app.services.model_router import ModelRouter
from app.services.query_normalization import significant_tokens, tokenize
from app.services.retrieval_service import RetrievalService
from app.services.settings_service import SettingsService
from app.services.source_service import format_source_records


class RAGPipeline:
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
        model_router = ModelRouter(SettingsService(retrieval_service.db))
        self.generator = GenerationService(model_router=model_router)
        self.intent_router = IntentRouter(retrieval_service.db, self.generator)
        self.document_refusal = GenerationService.DOCUMENT_REFUSAL
        self.out_of_scope_response = (
            "I can help with official University of Dodoma document questions and "
            "general university student support. This question is outside that scope."
        )

    def _is_empty_context(self, context: str) -> bool:
        return not context or context.strip() == "No relevant context found."

    def _retrieval_query(
        self,
        decision: IntentDecision,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        if decision.standalone_query:
            return decision.standalone_query
        return self.generator.rewrite_query(query, chat_history, user_profile=user_profile)

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> Dict[str, Any]:
        self.generator.set_model_preference(model_preference)
        decision = self.intent_router.classify(query, chat_history=chat_history, user_profile=user_profile)

        if decision.intent == "CONVERSATIONAL":
            answer = static_conversational_response(query)
            if answer is None:
                answer = self.generator.generate_conversational(
                    query,
                    chat_history,
                    user_profile=user_profile,
                )
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "context_used": False,
                "routing": decision.to_dict(),
                **self.generator.model_metadata(),
            }
        if decision.intent == "STUDENT_SUPPORT":
            answer = self.generator.generate_student_support(
                query,
                chat_history,
                user_profile=user_profile,
            )
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "context_used": False,
                "routing": decision.to_dict(),
                **self.generator.model_metadata(),
            }
        if decision.intent == "CLARIFY":
            answer = self.generator.generate_clarification_request(
                query,
                reason=decision.reason,
            )
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "context_used": False,
                "routing": decision.to_dict(),
                **self.generator.model_metadata(),
            }
        if decision.intent == "OUT_OF_SCOPE":
            return {
                "query": query,
                "answer": self.out_of_scope_response,
                "sources": [],
                "context_used": False,
                "routing": decision.to_dict(),
            }

        retrieval_query = self._retrieval_query(decision, query, chat_history, user_profile=user_profile)
        retrieval = self._retrieve_document_context(retrieval_query, filters=decision.filters)

        if self._is_empty_context(retrieval["context"]) or not retrieval["confidence"]["sufficient"]:
            return {
                "query": query,
                "answer": self.document_refusal,
                "sources": [],
                "context_used": False,
                "routing": decision.to_dict(),
                "retrieval_query": retrieval_query,
                "retrieval_confidence": retrieval["confidence"],
            }

        candidate_sources = self._format_sources(retrieval["documents"])
        try:
            generation_result = self.generator.generate_response(
                query,
                retrieval["context"],
                chat_history,
                user_profile=user_profile,
                documents=retrieval["documents"],
            )
        except GenerationUnavailableError as exc:
            raise exc.attach_sources(candidate_sources)

        grounded_documents = self.generator.grounding.filter_documents(
            retrieval["documents"],
            generation_result["evidence_ids"],
        )
        sources = self._format_sources(grounded_documents)

        return {
            "query": query,
            "answer": generation_result["answer"],
            "sources": sources,
            "context_used": generation_result["context_used"],
            "routing": decision.to_dict(),
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
        """Backward-compatible text-only stream."""
        async for event in self.stream_events(
            query,
            chat_history,
            user_profile,
            model_preference,
        ):
            if event["type"] == "stream":
                yield str(event["token"])

    async def stream_events(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream answer text followed by server-validated source records."""
        self.generator.set_model_preference(model_preference)
        decision = await run_in_threadpool(self.intent_router.classify, query, chat_history, user_profile)

        if decision.intent == "CONVERSATIONAL":
            answer = static_conversational_response(query)
            if answer is not None:
                yield {"type": "stream", "token": answer}
            else:
                async for token in self.generator.stream_conversational(
                    query,
                    chat_history,
                    user_profile=user_profile,
                ):
                    yield {"type": "stream", "token": token}
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            return
        if decision.intent == "STUDENT_SUPPORT":
            async for token in self.generator.stream_student_support(query, chat_history, user_profile=user_profile):
                yield {"type": "stream", "token": token}
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            return
        if decision.intent == "CLARIFY":
            async for token in self.generator.stream_clarification_request(query, reason=decision.reason):
                yield {"type": "stream", "token": token}
            yield {"type": "model", **self.generator.model_metadata()}
            yield {"type": "sources", "sources": []}
            return
        if decision.intent == "OUT_OF_SCOPE":
            yield {"type": "stream", "token": self.out_of_scope_response}
            yield {"type": "sources", "sources": []}
            return

        retrieval_query = await run_in_threadpool(
            self._retrieval_query,
            decision,
            query,
            chat_history,
            user_profile,
        )
        retrieval = await run_in_threadpool(
            self._retrieve_document_context,
            retrieval_query,
            decision.filters,
        )

        if self._is_empty_context(retrieval["context"]) or not retrieval["confidence"]["sufficient"]:
            yield {"type": "stream", "token": self.document_refusal}
            yield {"type": "sources", "sources": []}
            return

        candidate_sources = self._format_sources(retrieval["documents"])
        try:
            async for token in self.generator.stream_generate(
                query,
                retrieval["context"],
                chat_history,
                user_profile=user_profile,
                documents=retrieval["documents"],
            ):
                yield {"type": "stream", "token": token}
        except GenerationUnavailableError as exc:
            raise exc.attach_sources(candidate_sources)

        grounding = self.generator.grounding_metadata()
        grounded_documents = self.generator.grounding.filter_documents(
            retrieval["documents"],
            grounding["evidence_ids"],
        )
        yield {"type": "sources", "sources": self._format_sources(grounded_documents)}
        yield {"type": "model", **self.generator.model_metadata()}

    def run_debug(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        model_preference: str = "auto",
    ) -> Dict[str, Any]:
        self.generator.set_model_preference(model_preference)
        decision = self.intent_router.classify(query, chat_history=chat_history, user_profile=user_profile)

        if decision.intent == "CONVERSATIONAL":
            answer = static_conversational_response(query)
            if answer is None:
                answer = self.generator.generate_conversational(
                    query,
                    chat_history,
                    user_profile=user_profile,
                )
            return self._debug_response(
                query,
                answer,
                decision,
                [],
                "Conversational intent handled without retrieval.",
            )
        if decision.intent == "STUDENT_SUPPORT":
            answer = self.generator.generate_student_support(query, chat_history, user_profile=user_profile)
            return self._debug_response(query, answer, decision, [], "Student support intent handled without retrieval.")
        if decision.intent == "CLARIFY":
            answer = self.generator.generate_clarification_request(query, reason=decision.reason)
            return self._debug_response(query, answer, decision, [], "Clarification requested before retrieval.")
        if decision.intent == "OUT_OF_SCOPE":
            return self._debug_response(query, self.out_of_scope_response, decision, [], "Out-of-scope request rejected.")

        retrieval_query = self._retrieval_query(decision, query, chat_history, user_profile=user_profile)
        retrieval = self._retrieve_document_context(retrieval_query, filters=decision.filters)

        if self._is_empty_context(retrieval["context"]) or not retrieval["confidence"]["sufficient"]:
            answer = self.document_refusal
        else:
            generation_result = self.generator.generate_response(
                query,
                retrieval["context"],
                chat_history,
                user_profile=user_profile,
                documents=retrieval["documents"],
            )
            answer = generation_result["answer"]

        return {
            "query": query,
            "answer": answer,
            "debug": {
                "routing": decision.to_dict(),
                "retrieval_query": retrieval_query,
                "retrieval_confidence": retrieval["confidence"],
                "num_docs": len(retrieval["documents"]),
                "documents": retrieval["debug_documents"],
                "context_preview": retrieval["context"][:500],
                "grounding": self.generator.grounding_metadata(),
                "fallback_used": self.generator.model_metadata()["fallback_used"],
            },
        }

    def _retrieve_document_context(self, query: str, filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
        scored_docs = self.retrieval_service.retrieve_with_scores(query, filters=filters or None)
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
        dense_strong = best_dense_rank is not None and best_dense_rank <= 8 and best_overlap > 0

        if reranker == "jina":
            sufficient = top_score >= 0.15 or has_hybrid_match or sparse_strong or dense_strong
        elif reranker == "local_lexical":
            sufficient = top_score >= 1.0 or has_hybrid_match or sparse_strong or best_overlap >= 2
        else:
            sufficient = has_hybrid_match or sparse_strong or (top_rrf >= 0.016 and best_overlap > 0)

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
            aliases = AliasExpansionService(self.retrieval_service.db).get_expansions(query)
        except Exception:
            aliases = {}
        for term, values in aliases.items():
            terms.update(tokenize(term))
            for value in values:
                terms.update(tokenize(value))
        return terms

    def _lexical_overlap(self, query_terms: set[str], doc: Document) -> int:
        safe_metadata_keys = ("source", "document_title", "source_url", "category")
        metadata_text = " ".join(str(doc.metadata.get(key, "")) for key in safe_metadata_keys)
        haystack = f"{doc.page_content} {metadata_text}".lower()
        haystack_terms = set(re.findall(r"[a-z0-9]+", haystack))
        return len(query_terms & haystack_terms)

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
