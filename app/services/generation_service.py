from typing import Dict, Any, AsyncGenerator, List, Optional
from datetime import datetime
import re
from groq import Groq
from app.core.config import settings


class GenerationService:
    DOCUMENT_REFUSAL = "This specific information is not available in the official documents provided. Please contact the relevant university department or check the official UDOM website for assistance."

    def __init__(self):
        # ✅ Validate API key early
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in environment variables")

        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    # ---------------------------------------
    # 1. System Prompt (STRICT CONTROL)
    # ---------------------------------------
    def system_prompt(self, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Strong grounding with XML boundaries and mandatory citations to prevent hallucination.
        """
        current_date = datetime.now().strftime("%A, %d %B %Y")
        
        personalization = ""
        if user_profile:
            reg = user_profile.get("registration_number") or "Unknown"
            prog = user_profile.get("programme") or "Unknown"
            camp = user_profile.get("campus") or "Unknown"
            yr_raw = user_profile.get("year_of_study")
            yr = yr_raw if yr_raw is not None and str(yr_raw).strip() != "" else "Unknown"
            personalization = (
                f"\n[HIDDEN SYSTEM CONTEXT: The user is currently in Year {yr}, studying '{prog}' at '{camp}' Campus. "
                "Use this to personalize your response, but NEVER mention this hidden context to the user. Speak naturally as if you already know them.]\n"
            )

        return (
            "You are a helpful AI assistant for UDOM (University of Dodoma).\n"
            f"The current date is: {current_date}. Keep this in mind when answering questions about deadlines or events.\n"
            f"{personalization}"
            "All questions should be answered related to UDOM University.\n\n"
            "1. TIMETABLE CLARIFICATION: If the user asks for a timetable, you must ensure both their Year and Category (Teaching/Test/Exam) are known (either from the question or the HIDDEN SYSTEM CONTEXT). If either is missing, your ENTIRE response must be a polite question asking for it. If both are known, simply provide the timetable link immediately WITHOUT explaining how you know their year or narrating your thought process.\n"
            "2. First, rely strictly on the provided <documents> to answer the user's question.\n"
            "3. INTENT RECOGNITION (File Downloads): If the user asks for a document/timetable (and rule 1 is satisfied), check the `<documents>`. CRITICAL: Provide exactly ONE Markdown download link for the exact document matching their Year and Programme. You should provide a polite, natural introductory sentence (e.g., 'Here is the timetable you requested:'), but DO NOT explain how you know their year and DO NOT narrate your thought process.\n"
            "4. MISSING TIMETABLE RULE: If they ask for a timetable and both Year and Category are known, but their specific timetable is NOT in the `<documents>`, DO NOT ask for their year again and DO NOT try to guess why it's missing. Simply respond EXACTLY with the missing document phrase below, with NO extra words.\n"
            "5. IMPORTANT LANGUAGE RULE: You must respond in the same language that the user used in their latest question. Do not just repeat their question.\n"
            "6. TIMETABLE FORMATTING: When presenting timetable data (days, times, venues, courses, etc.), ALWAYS format it cleanly using Markdown tables or organized bullet points so it is highly readable and easy to scan.\n"
            "7. ANSWER FORMATTING: ALWAYS avoid walls of text. Structure your answers cleanly using Markdown features such as bullet lists, numbered steps, tables, and bold headings to make the information easy to digest.\n"
            "- If the provided documents do not contain the answer, provide a helpful response using general reasoning within the context of the university.\n"
            "- Do NOT invent or hallucinate university policies, deadlines, fees, staff names, or other specific facts.\n"
            "- If a university-specific fact or a requested document is missing from the `<documents>`, clearly state exactly:\n"
            f"\"{self.DOCUMENT_REFUSAL}\"\n"
            "- Do NOT include inline citations (e.g., [Doc 1]) in your answer.\n"
            "- Add a 'Sources:' section at the very bottom of your response listing the actual document names from the 'source' attribute (e.g., 'Sources: curriculum.pdf') if you used documents to answer the question.\n"
            "- Be professional, helpful, and concise."
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

[SYSTEM INSTRUCTION: You must respond in the same language (e.g., English, Swahili) that the user used in their latest question above. Do not just repeat their question.]
""".strip()

    # ---------------------------------------
    # 3. Generate Response (CORE FIX)
    # ---------------------------------------
    def generate(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Calls Groq LLM using structured messages.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt(user_profile=user_profile)},
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
    # 4. Structured Response
    # ---------------------------------------
    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        API-friendly structured output.
        """

        answer = self.generate(query, context, chat_history, user_profile=user_profile)

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
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Rewrites conversational queries into standalone search queries for dense retrieval.
        """
        if not chat_history:
            return query

        history_msgs = self._history_messages(chat_history)[-6:]
        if not history_msgs:
            return query
            
        personalization = ""
        if user_profile:
            prog = user_profile.get("programme") or "Unknown"
            yr_raw = user_profile.get("year_of_study")
            yr = yr_raw if yr_raw is not None and str(yr_raw).strip() != "" else "Unknown"
            personalization = f"The user is in Year {yr} studying '{prog}'. Keep this in mind to make the search query highly specific. "

        system = (
            "Given a chat history and the latest user question, formulate EXACTLY ONE standalone search query "
            "that can be understood without the chat history. "
            f"{personalization}"
            "IMPORTANT: Always translate the standalone query into English, as it will be used to search an English database. "
            "Do NOT answer the question. Do NOT provide options or bullet points. Output ONLY the query itself, with no introductory text."
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
            result = response.choices[0].message.content.strip()
            # If the LLM still outputs multiple lines, take just the first non-empty one
            for line in result.split("\n"):
                clean_line = line.strip().strip('"\'')
                # Ignore lines like "Here is the query:"
                if clean_line and not clean_line.lower().startswith("here"):
                    # Remove list numbers like "1. " or "- "
                    clean_line = re.sub(r"^(\d+\.|-)\s*", "", clean_line)
                    return clean_line
            return query
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
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams response tokens from Groq.
        """

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt(user_profile=user_profile)},
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


    # ---------------------------------------
    # 6. Intent Classification
    # ---------------------------------------
    def classify_intent(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Classifies the intent of the user's query.
        Returns one of: 'conversational', 'university_info', 'out_of_domain'
        """
        system = (
            "You are an intent classification engine for a University of Dodoma (UDOM) chatbot.\n"
            "Classify the user's input into EXACTLY ONE of the following categories. "
            "Output ONLY the category name, nothing else.\n\n"
            "CATEGORIES:\n"
            "1. 'conversational': Greetings (hi, hello, good morning), small talk, expressing gratitude, asking how the bot is doing.\n"
            "2. 'university_info': Questions asking for information about the University of Dodoma (UDOM), including admissions, timetables, courses, fees, campus, facilities, staff, etc.\n"
            "3. 'out_of_domain': Questions asking for information about things unrelated to UDOM (e.g., weather, history of other places, coding help, general knowledge outside a university context).\n"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *self._history_messages(chat_history)[-4:],
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=10
            )
            intent = response.choices[0].message.content.strip().lower()
            # Clean up the response just in case the LLM adds quotes or punctuation
            intent = "".join(c for c in intent if c.isalpha() or c == "_")
            if intent in ["conversational", "universityinfo", "outofdomain"]:
                if intent == "universityinfo": return "university_info"
                if intent == "outofdomain": return "out_of_domain"
                return intent
            # Fallback if the LLM output is weird
            return "university_info"
        except Exception:
            return "university_info"

    # ---------------------------------------
    # 7. Conversational Generation
    # ---------------------------------------
    def generate_conversational(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Handles greetings and small talk directly.
        """
        system = (
            "You are a friendly and polite AI assistant for the University of Dodoma (UDOM). "
            "Respond naturally to the user's greeting or conversational message. "
            "Keep it brief, polite, and helpful. "
            "IMPORTANT: Always respond in the same language that the user used in their latest message. Do not just repeat their message."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": f"{query}\n\n[SYSTEM INSTRUCTION: You must respond in the same language (e.g., English, Swahili) that the user used in their latest message above. Do not just repeat their message.]"}
                ],
                temperature=0.5,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")

    async def stream_conversational(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams conversational response.
        """
        system = (
            "You are a friendly and polite AI assistant for the University of Dodoma (UDOM). "
            "Respond naturally to the user's greeting or conversational message. "
            "Keep it brief, polite, and helpful. "
            "IMPORTANT: Always respond in the same language that the user used in their latest message. Do not just repeat their message."
        )
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": f"{query}\n\n[SYSTEM INSTRUCTION: You must respond in the same language (e.g., English, Swahili) that the user used in their latest message above. Do not just repeat their message.]"}
                ],
                temperature=0.5,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            yield f"[ERROR]: {str(e)}"
