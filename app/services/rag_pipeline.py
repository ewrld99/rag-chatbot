import re
from typing import Dict, Any, List, AsyncGenerator, Optional
from starlette.concurrency import run_in_threadpool

from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


class RAGPipeline:
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
        self.generator = GenerationService()
        self.document_refusal = GenerationService.DOCUMENT_REFUSAL
        self.out_of_domain_response = (
            "I am an AI assistant for the University of Dodoma. "
            "I can only help with university-related questions."
        )

    def _is_empty_context(self, context: str) -> bool:
        return not context or context.strip() == "No relevant context found."

    def _should_fallback(self, answer: str) -> bool:
        return answer.strip() == self.document_refusal

    def _retrieval_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Use the LLM to rewrite the query based on chat history.
        """
        return self.generator.rewrite_query(query, chat_history, user_profile=user_profile)

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run document-grounded RAG first, then fall back to general AI if the
        uploaded documents cannot answer.
        """

        intent = self.generator.classify_intent(query, chat_history, user_profile=user_profile)

        if intent == "conversational":
            return {
                "query": query,
                "answer": self.generator.generate_conversational(query, chat_history, user_profile=user_profile),
                "sources": [],
                "context_used": False
            }
        elif intent == "out_of_domain":
            return {
                "query": query,
                "answer": self.out_of_domain_response,
                "sources": [],
                "context_used": False
            }

        retrieval_query = self._retrieval_query(query, chat_history, user_profile=user_profile)
        retrieval_result = self.retrieval_service.get_context(retrieval_query)
        context = retrieval_result["context"]
        documents = retrieval_result["documents"]

        if self._is_empty_context(context):
            return {
                "query": query,
                "answer": self.document_refusal,
                "sources": [],
                "context_used": False
            }

        generation_result = self.generator.generate_response(query, context, chat_history, user_profile=user_profile)
        answer = generation_result["answer"]

        return {
            "query": query,
            "answer": answer,
            "sources": self._format_sources(documents),
            "context_used": generation_result["context_used"]
        }

    async def stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming pipeline for WebSocket or real-time UI.

        The document answer is buffered so users do not see the document-only
        refusal before the fallback answer.
        """

        intent = await run_in_threadpool(self.generator.classify_intent, query, chat_history, user_profile)

        if intent == "conversational":
            async for token in self.generator.stream_conversational(query, chat_history, user_profile=user_profile):
                yield token
            return
        elif intent == "out_of_domain":
            yield self.out_of_domain_response
            return

        retrieval_query = self._retrieval_query(query, chat_history, user_profile=user_profile)
        retrieval_result = await run_in_threadpool(self.retrieval_service.get_context, retrieval_query)
        context = retrieval_result["context"]

        if self._is_empty_context(context):
            yield self.document_refusal
            return

        async for token in self.generator.stream_generate(query, context, chat_history, user_profile=user_profile):
            yield token

    def run_debug(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns detailed internal pipeline data for debugging.
        """

        intent = self.generator.classify_intent(query, chat_history, user_profile=user_profile)

        if intent == "conversational":
            return {
                "query": query,
                "answer": self.generator.generate_conversational(query, chat_history),
                "debug": {
                    "num_docs": 0,
                    "documents": [],
                    "context_preview": "Conversational intent handled by LLM.",
                    "fallback_used": False
                }
            }
        elif intent == "out_of_domain":
            return {
                "query": query,
                "answer": self.out_of_domain_response,
                "debug": {
                    "num_docs": 0,
                    "documents": [],
                    "context_preview": "Out of domain intent rejected.",
                    "fallback_used": False
                }
            }

        retrieval_query = self._retrieval_query(query, chat_history, user_profile=user_profile)
        scored_docs = self.retrieval_service.retrieve_with_scores(retrieval_query)

        documents = []
        debug_info = []

        for doc, score in scored_docs:
            documents.append(doc)
            debug_info.append({
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
                "score": score
            })

        context = self.retrieval_service.format_context(documents)

        if self._is_empty_context(context):
            answer = self.document_refusal
            fallback_used = False
        else:
            answer = self.generator.generate(query, context, chat_history, user_profile=user_profile)
            fallback_used = False

        return {
            "query": query,
            "answer": answer,
            "debug": {
                "num_docs": len(documents),
                "documents": debug_info,
                "context_preview": context[:500],
                "fallback_used": fallback_used
            }
        }

    def _format_sources(self, documents: List) -> List[Dict[str, Any]]:
        """
        Formats sources for API response.
        """

        sources = []

        for doc in documents:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return sources
