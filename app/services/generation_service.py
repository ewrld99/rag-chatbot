from typing import Dict, Any, AsyncGenerator, List, Optional
from datetime import datetime
import re
from groq import Groq, AsyncGroq
from app.core.config import settings


class GenerationService:
    DOCUMENT_REFUSAL = "This specific information is not available in the official documents provided. Please contact the relevant university department or check the official UDOM website for assistance."
    CONVERSATIONAL_SYSTEM_PROMPT = (
        "You are a friendly and polite AI assistant for the University of Dodoma (UDOM). "
        "Respond naturally to the user's greeting or conversational message. "
        "Keep it brief, polite, and helpful. "
        "IMPORTANT: Always respond in the same language that the user used in their latest message. Do not just repeat their message."
    )

    def __init__(self):
        # ✅ Validate API key early
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in environment variables")

        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.async_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    # ---------------------------------------
    # 1. System Prompt (STRICT CONTROL)
    # ---------------------------------------
    def system_prompt(self, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Strong grounding with XML boundaries and mandatory citations to prevent hallucination.
        """
        current_date = datetime.now().strftime("%A, %d %B %Y")
        
        # ── Generic personalization (shown for all topics) ─────────────────
        personalization = ""
        if user_profile:
            details = []
            yr_raw = user_profile.get("year_of_study")
            if yr_raw is not None and str(yr_raw).strip() and str(yr_raw).strip().lower() != "unknown":
                details.append(f"in Year {yr_raw}")
            
            prog = user_profile.get("programme")
            if prog is not None and str(prog).strip():
                details.append(f"studying '{prog}'")
                
            camp = user_profile.get("campus")
            if camp is not None and str(camp).strip():
                details.append(f"at '{camp}' Campus")
            
            if details:
                details_str = ", ".join(details)
                personalization = (
                    f"\n[STUDENT CONTEXT — MANDATORY RULES:\n"
                    f"  • This student is: {details_str}.\n"
                    f"  • CURRICULUM FILTER: For questions about courses, subjects, units, modules, timetable, or semester schedule — respond ONLY with information that matches the student's year and programme. Do NOT list courses or content from other years or programmes.\n"
                    f"  • NEVER explicitly reveal this context to the student. Speak naturally as if you already know them.]\n"
                )

        return (
            "You are a helpful AI assistant for UDOM (University of Dodoma).\n"
            f"The current date is: {current_date}. Keep this in mind when answering questions about deadlines or events.\n"
            f"{personalization}"
            "All questions should be answered related to UDOM University.\n\n"
            "1. First, rely strictly on the provided <documents> to answer the user's question.\n"
            "2. IMPORTANT LANGUAGE RULE: You must respond in the same language that the user used in their latest question. Do not just repeat their question.\n"
            "3. ANSWER FORMATTING: ALWAYS avoid walls of text. Structure your answers cleanly using Markdown features such as bullet lists, numbered steps, tables, and bold headings to make the information easy to digest.\n"
            "4. DOCUMENT GROUNDING: If the provided documents do not contain the answer to a factual or policy question "
            "(including disciplinary consequences, penalties, rules, or procedures), you MUST respond "
            "with the exact refusal phrase below. Do NOT fill gaps with generic university-disciplinary "
            "knowledge from outside the documents, even if it sounds plausible.\n"
            f"\"{self.DOCUMENT_REFUSAL}\"\n"
            "5. GENERAL REASONING FALLBACK: 'General reasoning' is ONLY permitted for non-factual, non-policy questions "
            "(e.g. study tips, general encouragement) — never for anything resembling a rule, "
            "consequence, deadline, fee, or procedure.\n"
            "6. CITATIONS: Do NOT include inline citations (e.g., [Doc 1]) in your answer.\n"
            "7. SOURCES SECTION: If you used documents to answer the question, add a 'Sources:' section at the very bottom of your response. List the `name` attributes of the `<document>` tags you relied on as bullet points. Do NOT format them as links.\n"
            "8. TONE: Be professional, helpful, and concise."
        )


    def _history_messages(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
        limit: int = 12,
    ) -> List[Dict[str, str]]:
        """Returns the last `limit` valid conversation turns."""
        messages = []

        for item in (chat_history or [])[-limit:]:
            role = item.get("role")
            content = item.get("content", "").strip()

            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        return messages

    # ---------------------------------------
    # 2. Dynamic Token Limit
    # ---------------------------------------
    def _get_max_tokens(self, query: str) -> int:
        """
        Returns a higher token budget for queries that are likely to produce
        long list-style answers (course lists, module tables, etc.).
        """
        list_keywords = {
            "list", "courses", "course", "subjects", "subject",
            "units", "unit", "timetable", "semester", "all",
            "modules", "module", "curriculum", "schedule",
            "classes", "class", "lecture", "lectures",
        }
        if any(kw in query.lower() for kw in list_keywords):
            return 1000
        return 500

    # ---------------------------------------
    # 3. User Prompt (CLEAN INPUT)
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
                max_tokens=self._get_max_tokens(query)  # ✅ Dynamic: 1000 for lists, 500 otherwise
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

        Two modes:
        1. No chat history  — deterministically append year/programme from user_profile
           for curriculum queries (no LLM call needed, always reliable).
        2. With chat history — call the LLM rewriter to resolve pronouns/references
           and include profile context for curriculum queries.
        """
        is_curriculum = any(
            kw in query.lower()
            for kw in {
                "course", "courses", "subject", "subjects", "unit", "units",
                "timetable", "semester", "curriculum", "module", "modules",
                "schedule", "classes", "class", "lecture", "lectures",
            }
        )

        # --- Mode 1: No history — direct string enrichment, no LLM ---
        if not chat_history:
            if user_profile and is_curriculum:
                enrichment_parts = []
                yr_raw = user_profile.get("year_of_study")
                prog = user_profile.get("programme")
                if yr_raw is not None and str(yr_raw).strip() and str(yr_raw).lower() not in ("unknown", "none"):
                    enrichment_parts.append(f"Year {yr_raw}")
                if prog and str(prog).strip():
                    enrichment_parts.append(str(prog).strip())
                if enrichment_parts:
                    enrichment = " ".join(enrichment_parts)
                    # Only append if not already present in query
                    if enrichment.lower() not in query.lower():
                        return f"{query} {enrichment}"
            return query

        # --- Mode 2: Has history — use LLM to resolve references ---
        history_msgs = self._history_messages(chat_history, limit=6)
        if not history_msgs:
            # History present but empty after filtering, apply same direct enrichment
            if user_profile and is_curriculum:
                enrichment_parts = []
                yr_raw = user_profile.get("year_of_study")
                prog = user_profile.get("programme")
                if yr_raw is not None and str(yr_raw).strip() and str(yr_raw).lower() not in ("unknown", "none"):
                    enrichment_parts.append(f"Year {yr_raw}")
                if prog and str(prog).strip():
                    enrichment_parts.append(str(prog).strip())
                if enrichment_parts:
                    enrichment = " ".join(enrichment_parts)
                    if enrichment.lower() not in query.lower():
                        return f"{query} {enrichment}"
            return query
            
        personalization = ""
        if user_profile:
            details = []
            yr_raw = user_profile.get("year_of_study")
            if yr_raw is not None and str(yr_raw).strip() and str(yr_raw).strip().lower() != "unknown":
                details.append(f"in Year {yr_raw}")
            
            prog = user_profile.get("programme")
            if prog is not None and str(prog).strip():
                details.append(f"studying '{prog}'")
                
            camp = user_profile.get("campus")
            if camp is not None and str(camp).strip():
                details.append(f"at '{camp}' Campus")
                
            if details:
                details_str = " ".join(details)
                personalization = (
                    f"The user is {details_str}. "
                    "CURRICULUM RULE: If the user asks about courses, subjects, units, modules, timetable, semester schedule, or any curriculum content — "
                    "ALWAYS include their year and programme in the standalone search query (e.g. 'Year 2 BSc Computer Science Semester 1 courses'). "
                    "For general university rules, fees, policies, or other non-curriculum questions, do NOT include their personal details. "
                )

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
                    {"role": "user", "content": f"Chat History:\n{history_msgs}\n\nLatest Query: {query}"}
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
    # 6. Streaming Response
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
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt(user_profile=user_profile)},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": self.user_prompt(query, context)}
                ],
                temperature=0.1,
                max_tokens=self._get_max_tokens(query),  # ✅ Dynamic: 1000 for lists, 500 otherwise
                stream=True
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            yield f"[ERROR]: {str(e)}"


    # ---------------------------------------
    # 7. Intent Classification
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
            "2. 'university_info': Questions asking for information about the University of Dodoma (UDOM), including admissions, courses, fees, campus, facilities, staff, etc.\n"
            "3. 'out_of_domain': Questions asking for information about things unrelated to UDOM (e.g., weather, history of other places, coding help, general knowledge outside a university context).\n"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *self._history_messages(chat_history, limit=4),
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=10
            )
            intent = response.choices[0].message.content.strip().lower()
            # Clean up the response just in case the LLM adds quotes or punctuation
            intent = re.sub(r'[^a-z_]', '', intent)
            if intent in ["conversational", "university_info", "out_of_domain"]:
                return intent
            # Fallback if the LLM output is weird
            return "university_info"
        except Exception:
            return "university_info"

    # ---------------------------------------
    # 8. Conversational Generation
    # ---------------------------------------
    def generate_conversational(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Handles greetings and small talk directly.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.CONVERSATIONAL_SYSTEM_PROMPT},
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
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams conversational response.
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.CONVERSATIONAL_SYSTEM_PROMPT},
                    *self._history_messages(chat_history),
                    {"role": "user", "content": f"{query}\n\n[SYSTEM INSTRUCTION: You must respond in the same language (e.g., English, Swahili) that the user used in their latest message above. Do not just repeat their message.]"}
                ],
                temperature=0.5,
                stream=True
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            yield f"[ERROR]: {str(e)}"
