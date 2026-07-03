from typing import Dict, Any, AsyncGenerator, List, Optional
from groq import Groq
from app.core.config import settings


class GenerationService:
    DOCUMENT_REFUSAL = "I don't have enough information from the provided documents."

    def __init__(self):
        # ✅ Validate API key early
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in environment variables")

        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    # ---------------------------------------
    # 1. System Prompt (STRICT CONTROL)
    # ---------------------------------------
    def system_prompt(self) -> str:
        """
        Strong grounding with XML boundaries and mandatory citations to prevent hallucination.
        """
        return (
            "You are a retrieval-augmented AI assistant.\n"
            "You MUST answer ONLY using the provided <documents>.\n\n"
            "RULES:\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT include inline citations (e.g., [Doc 1]) in your answer.\n"
            "- Instead, you MUST add a 'Sources:' section at the very bottom of your response.\n"
            "- In the 'Sources:' section, list the actual document names from the 'source' attribute (e.g., 'Sources: curriculum.pdf') that you used to answer the question.\n"
            "- If the answer is missing from the provided documents, say exactly:\n"
            f"\"{self.DOCUMENT_REFUSAL}\"\n"
            "- Be concise, factual, and strictly adhere to the context."
        )

    # (Fallback system prompt removed)

    def _history_messages(self, chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        messages = []

        for item in (chat_history or [])[-12:]:
            role = item.get("role")
            content = item.get("content", "").strip()

            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        return messages

    # ---------------------------------------
    # 2. User Prompt (CLEAN INPUT)
    # ---------------------------------------
    def user_prompt(self, query: str, context: str) -> str:
        """
        Clean separation of context and question using XML tags.
        """
        return f"""
<documents>
{context}
</documents>

QUESTION:
{query}
""".strip()

    # ---------------------------------------
    # 3. Generate Response (CORE FIX)
    # ---------------------------------------
    def generate(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Calls Groq LLM using structured messages.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": self.user_prompt(query, context)}
                ],
                temperature=0.1,   # ✅ Lower = more factual
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")

    # (generate_fallback removed)

    # ---------------------------------------
    # 4. Structured Response
    # ---------------------------------------
    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        API-friendly structured output.
        """

        answer = self.generate(query, context, chat_history)

        return {
            "query": query,
            "answer": answer,
            "context_used": bool(context and context.strip()),
        }

    # ---------------------------------------
    # 5. Query Rewriting (Contextualization)
    # ---------------------------------------
    def rewrite_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Rewrites conversational queries into standalone search queries for dense retrieval.
        """
        if not chat_history:
            return query

        history_msgs = self._history_messages(chat_history)[-6:]
        if not history_msgs:
            return query
            
        system = (
            "Given a chat history and the latest user question, formulate a standalone query "
            "that can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed, otherwise return it as is."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *history_msgs,
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return query  # Fallback to raw query on failure

    # ---------------------------------------
    # 5. Streaming Response (FIXED + INSIDE CLASS)
    # ---------------------------------------
    async def stream_generate(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams response tokens from Groq.
        """

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": self.user_prompt(query, context)}
                ],
                temperature=0.1,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            yield f"[ERROR]: {str(e)}"

    # (stream_fallback removed)
