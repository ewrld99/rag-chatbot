"""
UDOM Timetable Fetcher Service
---------------------------------
Emulates the UDOM timetable download website (https://ratiba.udom.ac.tz/downloads/index)
to programmatically scrape, download, and ingest timetable PDFs into the RAG pipeline.

Flow:
  1. GET the page → extract CSRF token + session cookies
  2. AJAX calls → dynamically populate cascading dropdowns (year → semester → category → option → data)
  3. POST the form with all selections → receive PDF bytes in response
  4. Feed the PDF bytes through the existing chunking/embedding ingestion pipeline
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.db.models import DocumentModel, DocumentChunk
from app.services.embedding_service import get_embeddings, EmbeddingServiceError
from app.services.settings_service import SettingsService
from app.utils.chunking import split_text
from app.utils.loaders import load_pdf, load_pdf_timetable

logger = logging.getLogger(__name__)

BASE_URL = "https://ratiba.udom.ac.tz"
DOWNLOAD_INDEX_URL = f"{BASE_URL}/downloads/index"
DOWNLOAD_URL = f"{BASE_URL}/downloads/download"

# Real UDOM AJAX endpoints (verified via browser network inspection)
AJAX_SEMESTERS_URL = f"{BASE_URL}/downloads/fetch-semesters"
AJAX_CATEGORIES_URL = f"{BASE_URL}/downloads/fetch-categories"
AJAX_OPTS_URL = f"{BASE_URL}/downloads/opt"        # "option" dropdown (By Programme / By Course / etc.)
AJAX_DATA_URL = f"{BASE_URL}/downloads/data"        # final programme/course/room list

# Default HTTP headers to mimic a real browser
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": DOWNLOAD_INDEX_URL,
}


class TimetableFetchError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class TimetableFetcherService:
    """
    Scrapes the UDOM timetable portal and ingests PDFs into the RAG pipeline.
    Uses httpx for HTTP requests (sync) and BeautifulSoup for HTML parsing.
    """

    def __init__(self, db: Session):
        self.db = db
        self.client = httpx.Client(
            headers=BROWSER_HEADERS,
            follow_redirects=True,
            timeout=30.0,
        )
        self._csrf_token: Optional[str] = None

    def close(self):
        self.client.close()

    # ------------------------------------------------------------------
    # 1.  Session Bootstrapping — fetch fresh CSRF token + cookies
    # ------------------------------------------------------------------
    def _bootstrap_session(self) -> str:
        """
        GET the downloads index page, extract CSRF token and session cookies.
        Returns the CSRF token string.
        """
        try:
            response = self.client.get(DOWNLOAD_INDEX_URL)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise TimetableFetchError(
                f"Could not connect to ratiba.udom.ac.tz: {e}", status_code=503
            )

        soup = BeautifulSoup(response.text, "html.parser")
        csrf_meta = soup.find("meta", {"name": "csrf-token"})
        if not csrf_meta:
            raise TimetableFetchError(
                "CSRF token not found on UDOM timetable page. The site may have changed.",
                status_code=502,
            )

        self._csrf_token = csrf_meta["content"]
        logger.info("Bootstrapped UDOM session, CSRF token acquired.")
        return self._csrf_token

    def _ensure_session(self) -> str:
        """Return cached CSRF token or bootstrap a fresh one."""
        if not self._csrf_token:
            return self._bootstrap_session()
        return self._csrf_token

    # ------------------------------------------------------------------
    # 2.  Cascading Dropdown Helpers
    # ------------------------------------------------------------------
    def get_years(self) -> List[Dict[str, str]]:
        """
        Return static academic year options.
        These rarely change — update as needed.
        """
        return [
            {"value": "11", "label": "2024/2025"},
            {"value": "12", "label": "2025/2026"},
        ]

    def get_semesters(self, year: str) -> List[Dict[str, str]]:
        """Fetch available semesters for the given academic year."""
        csrf = self._ensure_session()
        try:
            response = self.client.get(
                AJAX_SEMESTERS_URL,
                params={
                    "_csrf-backend": csrf,
                    "year": year,
                    "semester": "",
                    "type": "",
                    "option": "",
                    "data[]": "",
                },
            )
            response.raise_for_status()
            return self._parse_select_options(response.text)
        except Exception as e:
            logger.warning("Failed to fetch semesters from UDOM, using static fallback: %s", e)
            return [
                {"value": "3366", "label": "SEMESTER ONE"},
                {"value": "3368", "label": "SEMESTER TWO"},
            ]

    def get_categories(self, year: str, semester: str) -> List[Dict[str, str]]:
        """Fetch timetable categories for a given year + semester."""
        csrf = self._ensure_session()
        try:
            response = self.client.get(
                AJAX_CATEGORIES_URL,
                params={
                    "_csrf-backend": csrf,
                    "year": year,
                    "semester": semester,
                    "type": "",
                    "option": "",
                    "data[]": "",
                },
            )
            response.raise_for_status()
            return self._parse_select_options(response.text)
        except Exception as e:
            logger.warning("Failed to fetch categories from UDOM, using static fallback: %s", e)
            return [
                {"value": "1", "label": "Teaching"},
                {"value": "2", "label": "Test"},
                {"value": "3", "label": "Examination"},
            ]

    def get_option_types(self, year: str, semester: str, category: str) -> List[Dict[str, str]]:
        """Fetch the 'option for download' list (By Programme / By Course / By Venue)."""
        csrf = self._ensure_session()
        try:
            response = self.client.get(
                AJAX_OPTS_URL,
                params={
                    "_csrf-backend": csrf,
                    "year": year,
                    "semester": semester,
                    "type": category,
                    "option": "",
                    "data[]": "",
                },
            )
            response.raise_for_status()
            return self._parse_select_options(response.text)
        except Exception as e:
            logger.warning("Failed to fetch option types from UDOM, using static fallback: %s", e)
            return [
                {"value": "room", "label": "By Venue"},
                {"value": "course", "label": "By Course"},
                {"value": "programme", "label": "By Programme"},
            ]

    def get_data_options(self, year: str, semester: str, category: str, option: str) -> List[Dict[str, str]]:
        """
        Fetch the final data list (specific programmes, courses, venues, or instructors)
        based on the selected option type.
        """
        csrf = self._ensure_session()
        try:
            response = self.client.get(
                AJAX_DATA_URL,
                params={
                    "_csrf-backend": csrf,
                    "year": year,
                    "semester": semester,
                    "type": category,
                    "option": option,
                    "data[]": "",
                },
            )
            response.raise_for_status()
            results = self._parse_select_options(response.text)
            if not results:
                raise TimetableFetchError(
                    f"No {option} options returned by UDOM. Check that your year/semester/category combination is valid.",
                    status_code=422,
                )
            return results
        except TimetableFetchError:
            raise
        except Exception as e:
            raise TimetableFetchError(
                f"Failed to retrieve {option} list from UDOM: {e}", status_code=502
            )

    # ------------------------------------------------------------------
    # 3.  PDF Download
    # ------------------------------------------------------------------
    def download_pdf(
        self,
        year: str,
        semester: str,
        category: str,
        option: str,
        data: List[str],
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> bytes:
        """
        Submit the timetable download form and return raw PDF bytes.
        Always refreshes the CSRF token right before submitting.
        """
        # Refresh session to get a fresh CSRF token
        if progress_cb:
            progress_cb("Connecting to UDOM Timetable system...", 5)
        self._bootstrap_session()

        form_data: Dict[str, Any] = {
            "_csrf-backend": self._csrf_token,
            "year": year,
            "semester": semester,
            "type": category,
            "option": option,
        }

        # The site accepts either a single data value or multiple data[]
        if len(data) == 1:
            form_data["data"] = data[0]
        else:
            form_data["data[]"] = data

        try:
            if progress_cb:
                progress_cb("Downloading PDF...", 10)
            response = self.client.post(DOWNLOAD_URL, data=form_data)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise TimetableFetchError(f"Network error downloading timetable: {e}", status_code=503)
        except httpx.HTTPStatusError as e:
            raise TimetableFetchError(
                f"UDOM server rejected the download request (HTTP {e.response.status_code}). "
                "Check that your selections are valid.",
                status_code=502,
            )

        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            raise TimetableFetchError(
                "UDOM did not return a PDF file. The site may require different form values, "
                "or the combination (year/semester/category/option/data) is invalid.",
                status_code=422,
            )

        logger.info("Downloaded timetable PDF: %d bytes", len(response.content))
        return response.content

    # ------------------------------------------------------------------
    # 4.  Ingestion into RAG Pipeline
    # ------------------------------------------------------------------
    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        document_label: str,
        admin_id: Optional[int] = None,
        strategy: str = "timetable",
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Save PDF bytes to a temp file, extract text, chunk, embed, and persist
        as a DocumentModel using the same pipeline as the upload endpoint.

        strategy options:
          - "timetable" : row-per-sentence extraction (best for timetable grids)
          - "auto"      : pdfplumber with markdown tables (good general purpose)
          - "fast"      : pypdf raw text (fastest, no table awareness)
        """
        # Write to temp file so pdfplumber can open it
        tmp_path = os.path.join("uploads", f"timetable_{uuid4().hex}.pdf")
        try:
            with open(tmp_path, "wb") as f:
                f.write(pdf_bytes)

            if progress_cb:
                progress_cb("Extracting text from PDF...", 30)

            if strategy == "timetable":
                # Row-per-sentence extraction — keeps time/course/venue/lecturer together
                text = load_pdf_timetable(tmp_path)
            elif strategy == "fast":
                from app.utils.loaders import load_pdf_fast
                text = load_pdf_fast(tmp_path)
            else:
                # "auto" — pdfplumber with markdown tables
                text = load_pdf(tmp_path)

        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise TimetableFetchError(f"Failed to extract text from timetable PDF: {e}")

        if not text or not text.strip():
            os.remove(tmp_path)
            raise TimetableFetchError("No text could be extracted from the timetable PDF.")

        if progress_cb:
            progress_cb("Chunking text...", 50)

        if strategy == "timetable":
            # Each line is already a complete, self-contained sentence — keep them as chunks
            raw_chunks = [line.strip() for line in text.splitlines() if line.strip()]
            chunks = raw_chunks if raw_chunks else split_text(text, chunk_size=800, overlap=0)
        else:
            settings_svc = SettingsService(self.db)
            chunks = split_text(text, chunk_size=settings_svc.chunk_size, overlap=settings_svc.chunk_overlap)

        if not chunks:
            os.remove(tmp_path)
            raise TimetableFetchError("Text could not be chunked.")

        if progress_cb:
            progress_cb(f"Generating embeddings for {len(chunks)} chunks...", 70)

        try:
            embeddings = get_embeddings(chunks, self.db)
        except EmbeddingServiceError as e:
            os.remove(tmp_path)
            raise TimetableFetchError(str(e), status_code=e.status_code)

        if len(embeddings) != len(chunks):
            os.remove(tmp_path)
            raise TimetableFetchError("Embedding count mismatch — check embedding service.")

        if progress_cb:
            progress_cb("Saving document and chunks to database...", 90)

        # Persist document record
        document = DocumentModel(
            title=document_label,
            filename=os.path.basename(tmp_path),
            file_path=tmp_path,
            category="timetable",
            uploaded_by=admin_id,
            status="active",
        )
        self.db.add(document)
        self.db.flush()  # get document.id

        self.db.bulk_insert_mappings(
            DocumentChunk,
            [
                {
                    "document_id": document.id,
                    "chunk_text": chunk,
                    "embedding": embedding,
                    "chunk_index": i,
                    "metadata": {"source_url": f"http://localhost:8000/uploads/{os.path.basename(tmp_path)}"},  # stored in metadata_ column
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        )
        self.db.commit()

        logger.info(
            "Ingested timetable '%s': %d chunks stored (doc_id=%s)",
            document_label,
            len(chunks),
            document.id,
        )

        return {
            "document_id": str(document.id),
            "title": document_label,
            "chunks": len(chunks),
            "file_path": tmp_path,
        }

    # ------------------------------------------------------------------
    # 5.  Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_select_options(html: str) -> List[Dict[str, str]]:
        """Parse <option value="...">Label</option> tags from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        options = []
        for opt in soup.find_all("option"):
            value = opt.get("value", "").strip()
            label = opt.get_text(strip=True)
            if value:
                options.append({"value": value, "label": label})
        return options
