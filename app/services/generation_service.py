from typing import Dict, Any, AsyncGenerator, List, Optional
from groq import Groq
from app.core.config import settings


class GenerationService:

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
        Strong grounding with XML boundaries, but allows general knowledge fallback.
        """
        return (
            "You are a helpful retrieval-augmented AI assistant for the University of Dodoma (UDOM).\n"
            "You have been provided with some <documents> as context.\n\n"
            "RULES:\n"
            "- If the <documents> contain the answer, use them and add a 'Sources:' section at the bottom listing the document names from the 'source' attribute.\n"
            "- If the <documents> DO NOT contain the answer, you MUST use your own general knowledge to provide a helpful and relevant answer to the user.\n"
            "- Do not hallucinate UDOM-specific policies if you aren't sure. Be concise and factual."
        )

    def fallback_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant for the University of Dodoma (UDOM).\n"
            "The user asked a question, but you do not have specific institutional documents containing the answer.\n"
            "You should answer the user's question using your general knowledge.\n"
            "If you do not know the answer, politely say so. DO NOT hallucinate UDOM-specific policies."
        )

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

    # ---------------------------------------
    # Fallback Generation
    # ---------------------------------------
    def generate_fallback(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.fallback_system_prompt()},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Groq API error (fallback): {str(e)}")

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

    async def stream_fallback(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.fallback_system_prompt()},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"[ERROR]: {str(e)}"
