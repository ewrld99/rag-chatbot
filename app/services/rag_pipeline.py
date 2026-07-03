import re
from typing import Dict, Any, List, AsyncGenerator, Optional

from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


class RAGPipeline:
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
        self.generator = GenerationService()
        self.document_refusal = GenerationService.DOCUMENT_REFUSAL
        self.greeting_response = (
            "Hello! How can I help you today? "
            "You can ask me anything, and I'll do my best to help."
        )

    def _is_greeting(self, query: str) -> bool:
        normalized = re.sub(r"[^a-z\s]", " ", query.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)

        greeting_patterns = (
            r"^(hi|hello|hey|hii|hiya)$",
            r"^(hi|hello|hey|hii|hiya)\s+(there|bot|assistant)$",
            r"^good\s+(morning|afternoon|evening)$",
            r"^(howdy|greetings)$",
        )

        return any(re.match(pattern, normalized) for pattern in greeting_patterns)

    def _is_empty_context(self, context: str) -> bool:
        return not context or context.strip() == "No relevant context found."

    def _should_fallback(self, answer: str) -> bool:
        return answer.strip() == self.document_refusal

    def _retrieval_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Use the LLM to rewrite the query based on chat history.
        """
        return self.generator.rewrite_query(query, chat_history)

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run document-grounded RAG first, then fall back to general AI if the
        uploaded documents cannot answer.
        """

        if self._is_greeting(query):
            return {
                "query": query,
                "answer": self.greeting_response,
                "sources": [],
                "context_used": False
            }

        retrieval_query = self._retrieval_query(query, chat_history)
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

        generation_result = self.generator.generate_response(query, context, chat_history)
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
    ) -> AsyncGenerator[str, None]:
        """
        Streaming pipeline for WebSocket or real-time UI.

        The document answer is buffered so users do not see the document-only
        refusal before the fallback answer.
        """

        if self._is_greeting(query):
            yield self.greeting_response
            return

        retrieval_query = self._retrieval_query(query, chat_history)
        retrieval_result = self.retrieval_service.get_context(retrieval_query)
        context = retrieval_result["context"]

        if self._is_empty_context(context):
            yield self.document_refusal
            return

        async for token in self.generator.stream_generate(query, context, chat_history):
            yield token

    def run_debug(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Returns detailed internal pipeline data for debugging.
        """

        if self._is_greeting(query):
            return {
                "query": query,
                "answer": self.greeting_response,
                "debug": {
                    "num_docs": 0,
                    "documents": [],
                    "context_preview": "Greeting handled without document retrieval.",
                    "fallback_used": False
                }
            }

        retrieval_query = self._retrieval_query(query, chat_history)
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
            answer = self.generator.generate(query, context, chat_history)
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
