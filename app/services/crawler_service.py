import logging
import time
import hashlib
import os
import asyncio
from types import SimpleNamespace
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import markdownify

from sqlalchemy.orm import Session
from sqlalchemy import case, update
from app.db.models import (
    CrawlerJob,
    CrawlerQueue,
    DocumentModel,
    ExternalLinkModel,
    FileOperation,
    User,
)
from app.db.session import SessionLocal
from app.services.settings_service import SettingsService
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.file_storage_service import FileStorageService
from app.services.storage_reconciler import create_file_operation, reconcile_file_operations

logger = logging.getLogger(__name__)

# Extensions to ignore
IGNORED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".avi", ".mov",
    ".zip", ".rar", ".tar", ".gz", ".7z", ".exe", ".iso", ".css", ".js"
}
ANNOUNCEMENT_KEYWORDS = ("announcement", "announcements", "news", "blog", "article", "event", "admission")
ANNOUNCEMENT_RECENT_LIMITS = {
    "/announcements": 10,
    "/blog/index": 5,
}
FULL_CRAWLER_ANNOUNCEMENT_OWNED_PATH_PREFIXES = (
    "/announcements",
    "/blog",
)
FULL_CRAWLER_BLOCKED_PATH_PREFIXES = (
    "/login",
    "/logout",
    "/admin",
    "/search",
    "/profile",
    "/staff",
    "/gallery",
    "/university_documents",
)
FULL_CRAWLER_BLOCKED_DOMAINS = {
    "application.udom.ac.tz",
    "catalog.udom.ac.tz",
    "coi.udom.ac.tz",
    "eresources.udom.ac.tz",
    "his.udom.ac.tz",
    "mail.udom.ac.tz",
    "maoni.udom.ac.tz",
    "meetings.udom.ac.tz",
    "oas.udom.ac.tz",
    "opras.udom.ac.tz",
    "portal.udom.ac.tz",
    "projects.udom.ac.tz",
    "ratiba.udom.ac.tz",
    "repository.udom.ac.tz",
    "ruhusa.udom.ac.tz",
    "sr2.udom.ac.tz",
    "staffsr.udom.ac.tz",
    "udomasa.udom.ac.tz",
}
ANNOUNCEMENT_LISTING_BLOCKED_PATH_PREFIXES = (
    "/site/",
    "/about",
    "/contact",
    "/staff",
    "/administration",
    "/directorate",
    "/college",
    "/schools",
    "/programmes",
)
NOT_FOUND_PATTERNS = (
    "404",
    "page not found",
    "the page you requested could not be found",
    "requested page could not be found",
    "oops page not found",
)

def is_valid_udom_url(url: str, allowlist: list, blocklist: list) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().split(':')[0]
    
    # Must end with udom.ac.tz
    if not domain.endswith("udom.ac.tz"):
        return False
        
    if domain == "storage.udom.ac.tz":
        return True
        
    # Check blocklist first
    for blocked in blocklist:
        if "/" in blocked:
            if blocked in url:
                return False
        else:
            if domain == blocked or domain.endswith("." + blocked):
                return False
            
    # If allowlist is empty, assume all udom domains are allowed.
    if not allowlist:
        return True
        
    for allowed in allowlist:
        if domain == allowed or domain.endswith("." + allowed):
            return True
            
    return False

def get_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalized_path(url: str) -> str:
    path = urlparse(url).path.rstrip("/").lower()
    return path or "/"

class CrawlerService:
    def __init__(self, db: Session):
        runtime_settings = SettingsService(db)
        self.settings = SimpleNamespace(
            crawler_allowlist=tuple(runtime_settings.crawler_allowlist),
            crawler_blocklist=tuple(runtime_settings.crawler_blocklist),
            crawler_max_age_days=runtime_settings.crawler_max_age_days,
        )
        # Ensure we have a system user for crawler uploads
        system_user = db.query(User).filter(User.username == "system_crawler").first()
        if not system_user:
            system_user = User(username="system_crawler", password_hash="dummy")
            db.add(system_user)
            db.commit()

        # Self-healing migration for link_text column
        from sqlalchemy import inspect, text
        inspector = inspect(db.bind)
        columns = [c['name'] for c in inspector.get_columns('crawler_queue')]
        if 'link_text' not in columns:
            try:
                db.execute(text("ALTER TABLE crawler_queue ADD COLUMN link_text TEXT"))
                db.commit()
                logger.info("Successfully added link_text column to crawler_queue")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to add link_text column: {e}")

    def process_external_link(self, url: str, found_on: str, db: Session):
        exists = db.query(ExternalLinkModel).filter(ExternalLinkModel.url == url).first()
        if not exists:
            ext_link = ExternalLinkModel(url=url, found_on=found_on)
            db.add(ext_link)
            try:
                db.commit()
            except Exception:
                db.rollback()

    def process_pdf(self, url: str, content: bytes, source_url: str, title_hint: str = "", db: Session = None):
        file_hash = get_content_hash(content.hex())
        
        # Check if we already indexed this PDF based on content hash. 
        # This prevents duplicates from S3 pre-signed URLs which change on every crawl due to signatures.
        doc_by_hash = db.query(DocumentModel).filter(
            (DocumentModel.content_hash == file_hash)
            | (DocumentModel.quality_report["source_content_hash"].astext == file_hash)
        ).first()
        if doc_by_hash:
            logger.info(f"PDF already exists with same content hash. Skipping {url}.")
            return

        # Fallback check based on source_url
        doc = db.query(DocumentModel).filter(DocumentModel.source_url == url).first()
        
        if doc and (
            doc.content_hash == file_hash
            or dict(doc.quality_report or {}).get("source_content_hash") == file_hash
        ):
            logger.info(f"PDF {url} has not changed. Skipping.")
            return

        filename = os.path.basename(urlparse(url).path)
        if title_hint:
            import re
            safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title_hint).strip()
            if safe_title:
                filename = f"{safe_title[:50].replace(' ', '_')}.pdf"
        if not filename.lower().endswith(".pdf"):
            filename = "document.pdf"

        storage = FileStorageService()
        staged = storage.stage_bytes(content, filename)
        old_file_path = doc.file_path if doc else None
        if doc is None:
            sys_user = db.query(User).filter(User.username == "system_crawler").first()
            doc = DocumentModel(
                title=staged.filename,
                filename=staged.filename,
                file_path=staged.target_path,
                source_url=url,
                category="pdf",
                uploaded_by=sys_user.id if sys_user else None,
                last_crawled_at=datetime.utcnow(),
                status="processing",
                indexing_status="processing",
                storage_state="staged",
            )
            db.add(doc)
            db.flush()
        operation = create_file_operation(
            db,
            document_id=doc.id,
            operation="promote",
            source_path=staged.source_path,
            target_path=staged.target_path,
        )
        operation.status = "held"
        db.commit()

        try:
            ingestion = DocumentIngestionService(db)
            prepared = ingestion.prepare_file(
                staged.source_path,
                "pdf",
                strategy="auto",
                current_document_id=doc.id,
                base_metadata={
                    "source_url": url,
                    "crawled_at": datetime.utcnow().isoformat(),
                },
            )
            embeddings = ingestion.embed(prepared)
            storage.promote(staged.source_path, staged.target_path)

            operation = db.query(FileOperation).filter(FileOperation.id == operation.id).first()
            doc = db.query(DocumentModel).filter(DocumentModel.id == doc.id).first()
            ingestion.apply(doc, prepared, embeddings, persist_review_chunks=not bool(doc.chunks))
            report = dict(doc.quality_report or {})
            report["source_content_hash"] = file_hash
            doc.quality_report = report
            doc.title = staged.filename
            doc.filename = staged.filename
            doc.file_path = staged.target_path
            doc.last_crawled_at = datetime.utcnow()
            doc.storage_state = "ready"
            doc.storage_error = None
            operation.status = "completed"
            db.commit()

            if old_file_path and old_file_path != staged.target_path:
                cleanup = create_file_operation(
                    db,
                    document_id=None,
                    operation="delete",
                    source_path=old_file_path,
                    target_path=storage.trash_path(doc.id, os.path.basename(old_file_path)),
                )
                db.commit()
                reconcile_file_operations(storage=storage)
        except Exception as exc:
            db.rollback()
            if storage.resolve(staged.target_path).exists() and not storage.resolve(staged.source_path).exists():
                storage.remove(staged.target_path)
            try:
                retained = storage.retain_failed(staged.source_path, doc.id, staged.filename)
            except Exception:
                retained = None
            operation = db.query(FileOperation).filter(FileOperation.id == operation.id).first()
            if operation is not None:
                operation.status = "cancelled"
                operation.last_error = str(exc)[:2000]
            current = db.query(DocumentModel).filter(DocumentModel.id == doc.id).first()
            if current is not None and old_file_path is None:
                current.file_path = retained or current.file_path
                current.storage_state = "ready" if retained else "missing"
                current.storage_error = str(exc)[:2000]
                current.status = "embedding_failed"
                current.indexing_status = "failed"
            db.commit()
            raise

    def _embed_and_save(self, url: str, title: str, text: str, content_hash: str, category: str, db: Session, doc_metadata: dict | None = None, file_path: str = None):
        doc = db.query(DocumentModel).filter(DocumentModel.source_url == url).first()
        base_meta = {
            "source_url": url,
            "crawled_at": datetime.utcnow().isoformat(),
            **(doc_metadata or {}),
        }
        ingestion = DocumentIngestionService(db)
        if file_path and os.path.exists(file_path):
            prepared = ingestion.prepare_file(
                file_path,
                category,
                strategy="auto",
                current_document_id=doc.id if doc else None,
                base_metadata=base_meta,
            )
        else:
            prepared = ingestion.prepare_text(
                text,
                file_type=category,
                current_document_id=doc.id if doc else None,
                base_metadata=base_meta,
                extraction_method="crawler_html",
            )
        embeddings = ingestion.embed(prepared)

        if not doc:
            sys_user = db.query(User).filter(User.username == "system_crawler").first()
            doc = DocumentModel(
                title=title,
                filename=title,
                file_path=file_path,
                source_url=url,
                category=category,
                uploaded_by=sys_user.id if sys_user else None,
                last_crawled_at=datetime.utcnow(),
                status="processing",
                indexing_status="processing",
                storage_state="ready" if file_path else "not_applicable",
            )
            db.add(doc)
            db.flush()
        else:
            doc.title = title
            doc.filename = title
            if file_path:
                doc.file_path = file_path
            doc.last_crawled_at = datetime.utcnow()

        ingestion.apply(
            doc,
            prepared,
            embeddings,
            persist_review_chunks=not bool(doc.chunks),
        )
        report = dict(doc.quality_report or {})
        report["source_content_hash"] = content_hash
        doc.quality_report = report
        db.commit()

    async def fetch_page(self, client, url: str):
        try:
            if "X-Amz-Signature" in url:
                import urllib.request
                import ssl
                import asyncio
                
                def _sync_fetch():
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    headers_dict = dict(client.headers)
                    if "host" in headers_dict:
                        del headers_dict["host"]
                    req = urllib.request.Request(url, headers=headers_dict)
                    with urllib.request.urlopen(req, context=ctx, timeout=30.0) as res:
                        content = res.read()
                        class DummyResponse:
                            def __init__(self, content, headers):
                                self.content = content
                                self.text = content.decode('utf-8', errors='replace')
                                self.headers = headers
                            def raise_for_status(self):
                                pass
                        return DummyResponse(content, res.info())
                return await asyncio.to_thread(_sync_fetch)

            response = await client.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {type(e).__name__} - {e}")
            raise  # Let the actual error bubble up to the queue result

    def _process_pdf_sync(self, url, content, source_url, link_text):
        with SessionLocal() as db:
            self.process_pdf(url, content, source_url, link_text, db)
            
    def _process_html_sync(self, url, title, html_content, external_links):
        markdown_text = markdownify.markdownify(html_content, heading_style="ATX", strip=['a', 'img']).strip()
        if not markdown_text:
            return
        if self._is_not_found_page(title, markdown_text):
            logger.info("Skipping not-found page %s", url)
            return
            
        text_hash = get_content_hash(markdown_text)
        
        with SessionLocal() as db:
            # Process external links found in this doc
            for ext_url in external_links:
                self.process_external_link(ext_url, url, db)
                
            doc = db.query(DocumentModel).filter(DocumentModel.source_url == url).first()
            if doc and doc.content_hash == text_hash:
                logger.info(f"Page {url} has not changed. Skipping.")
                doc.last_crawled_at = datetime.utcnow()
                db.commit()
            else:
                logger.info(f"Indexing new/updated page {url}")
                self._embed_and_save(url, title, markdown_text, text_hash, "webpage", db)

    def _is_not_found_page(self, title: str, text: str) -> bool:
        title_clean = " ".join(str(title or "").lower().split())
        text_clean = " ".join(str(text or "").lower().split())
        text_preview = text_clean[:1200]

        if title_clean in {"404", "404 not found", "not found", "page not found"}:
            return True
        if any(pattern in title_clean for pattern in ("404", "page not found")):
            return True
        if "not found" in title_clean and len(text_clean) < 2500:
            return True
        if any(pattern in text_preview for pattern in NOT_FOUND_PATTERNS) and len(text_clean) < 2500:
            return True
        return False

    def _announcement_listing_limit(self, url: str) -> int | None:
        return ANNOUNCEMENT_RECENT_LIMITS.get(_normalized_path(url))

    def _is_allowed_crawl_link(self, url: str, allowlist: list, blocklist: list) -> bool:
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext in IGNORED_EXTENSIONS:
            return False
        if not url.startswith("http"):
            return False
        return is_valid_udom_url(url, allowlist, blocklist)

    def _extract_recent_listing_links(
        self,
        soup: BeautifulSoup,
        listing_url: str,
        limit: int,
        allowlist: list,
        blocklist: list,
    ) -> list[tuple[str, str]]:
        listing_soup = BeautifulSoup(str(soup), "html.parser")
        for tag in listing_soup(["nav", "footer", "header", "script", "style", "aside", "form"]):
            tag.decompose()

        listing_path = _normalized_path(listing_url)
        seen: set[str] = set()
        links: list[tuple[str, str]] = []

        for a_tag in listing_soup.find_all("a", href=True):
            full_url = urljoin(listing_url, a_tag["href"]).split("#")[0]
            if not self._is_allowed_crawl_link(full_url, allowlist, blocklist):
                continue
            parsed = urlparse(full_url)
            if _normalized_path(full_url) == listing_path:
                continue
            normalized_path = _normalized_path(full_url)
            if normalized_path.startswith(ANNOUNCEMENT_LISTING_BLOCKED_PATH_PREFIXES):
                continue
            normalized_url = parsed._replace(query="").geturl()
            if normalized_url in seen:
                continue
            seen.add(normalized_url)
            links.append((full_url, a_tag.get_text(strip=True)))
            if len(links) >= limit:
                break

        logger.info(
            "Selected recent listing links | listing=%s selected=%d limit=%d",
            listing_url,
            len(links),
            limit,
        )
        return links

    def _extract_pdf_links(
        self,
        soup: BeautifulSoup,
        source_url: str,
        allowlist: list,
        blocklist: list,
    ) -> list[tuple[str, str]]:
        seen: set[str] = set()
        links: list[tuple[str, str]] = []
        for a_tag in soup.find_all("a", href=True):
            full_url = urljoin(source_url, a_tag["href"]).split("#")[0]
            parsed = urlparse(full_url)
            if os.path.splitext(parsed.path)[1].lower() != ".pdf":
                continue
            if not self._is_allowed_crawl_link(full_url, allowlist, blocklist):
                continue
            normalized_url = parsed._replace(query="").geturl()
            if normalized_url in seen:
                continue
            seen.add(normalized_url)
            links.append((full_url, a_tag.get_text(strip=True)))
        return links

    def _should_skip_full_crawler_url(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(":")[0]
        path = _normalized_path(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext == ".pdf":
            return True
        if domain in FULL_CRAWLER_BLOCKED_DOMAINS:
            return True
        if path.startswith(FULL_CRAWLER_ANNOUNCEMENT_OWNED_PATH_PREFIXES):
            return True
        if path.startswith(FULL_CRAWLER_BLOCKED_PATH_PREFIXES):
            return True
        return False

    async def _process_single_url_task(self, client, url: str, link_text: str, job_type: str, allowlist: list, blocklist: list):
        max_age_days = self.settings.crawler_max_age_days
        if job_type == "full" and self._should_skip_full_crawler_url(url):
            logger.info("Skipping full-crawler excluded URL %s", url)
            return []
        
        # 1. Check URL for embedded dates (e.g. doc1410_8_2026_05_12...)
        if max_age_days > 0:
            import re
            match = re.search(r'(20\d{2})[-_/\\](0[1-9]|1[0-2])[-_/\\](0[1-9]|[12]\d|3[01])', url)
            if match:
                year, month, day = map(int, match.groups())
                try:
                    url_date = datetime(year, month, day)
                    if (datetime.utcnow() - url_date).days > max_age_days:
                        logger.info(f"Skipping old announcement (URL date {url_date.date()}): {url}")
                        return []
                except ValueError:
                    pass

        # Process a single URL, return list of new discovered URLs or raise Exception if failed
        response = await self.fetch_page(client, url)
        if not response:
            raise Exception("Failed to fetch")
            
        # 2. Check HTTP Last-Modified header
        if max_age_days > 0:
            last_mod = response.headers.get("Last-Modified")
            if last_mod:
                try:
                    from email.utils import parsedate_to_datetime
                    last_mod_date = parsedate_to_datetime(last_mod).replace(tzinfo=None)
                    if (datetime.utcnow() - last_mod_date).days > max_age_days:
                        logger.info(f"Skipping old announcement (Last-Modified {last_mod_date.date()}): {url}")
                        return []
                except Exception:
                    pass
            
        content_type = response.headers.get("Content-Type", "").lower()
        parsed_url = urlparse(url)
        url_ext = os.path.splitext(parsed_url.path)[1].lower()

        if "application/pdf" in content_type or url_ext == ".pdf":
            if job_type == "full":
                logger.info("Skipping PDF during full crawler %s", url)
                return []
            await asyncio.to_thread(self._process_pdf_sync, url, response.content, url, link_text)
            return []
            
        if "text/html" not in content_type:
            return []
            
        soup = BeautifulSoup(response.text, "html.parser")
        new_urls = []
        external_links_to_save = []

        if job_type == "announcements":
            listing_limit = self._announcement_listing_limit(url)
            if listing_limit is not None:
                new_urls = self._extract_recent_listing_links(
                    soup,
                    url,
                    listing_limit,
                    allowlist,
                    blocklist,
                )
            else:
                new_urls = self._extract_pdf_links(soup, url, allowlist, blocklist)
        else:
            # Extract links
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                a_text = a_tag.get_text(strip=True)
                full_url = urljoin(url, href).split("#")[0]

                parsed_href = urlparse(full_url)
                ext = os.path.splitext(parsed_href.path)[1].lower()

                if ext in IGNORED_EXTENSIONS:
                    continue

                if not full_url.startswith("http"):
                    continue

                domain = parsed_href.netloc.lower().split(':')[0]
                if not domain.endswith("udom.ac.tz") and domain != "storage.udom.ac.tz":
                    external_links_to_save.append(full_url)
                    continue

                if job_type == "full" and self._should_skip_full_crawler_url(full_url):
                    continue

                if is_valid_udom_url(full_url, allowlist, blocklist):
                    new_urls.append((full_url, a_text))
                
        for tag in soup(["nav", "footer", "header", "script", "style", "aside", "form", "img", "svg", "video", "audio", "iframe"]):
            tag.decompose()
            
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        html_content = str(soup.body) if soup.body else str(soup)
        
        await asyncio.to_thread(self._process_html_sync, url, title, html_content, external_links_to_save)
        return new_urls

    @staticmethod
    def _initialize_crawl_job(
        start_urls: list[str],
        max_pages: int,
        job_type: str,
        reset: bool,
    ) -> bool:
        with SessionLocal() as db:
            job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
            if not job:
                job = CrawlerJob(
                    job_type=job_type,
                    status="running",
                    max_pages=max_pages,
                    crawled_count=0,
                )
                db.add(job)
            else:
                if job.status == "running" and not reset:
                    return False
                job.status = "running"
                job.max_pages = max_pages
                if reset:
                    job.crawled_count = 0
                    db.query(CrawlerQueue).filter_by(job_type=job_type).delete()
            if not reset:
                db.execute(
                    update(CrawlerQueue)
                    .where(
                        CrawlerQueue.job_type == job_type,
                        CrawlerQueue.status == "processing",
                    )
                    .values(status="pending")
                )
            for url in start_urls:
                exists = db.query(CrawlerQueue.id).filter_by(
                    job_type=job_type,
                    url=url,
                ).first()
                if not exists:
                    db.add(
                        CrawlerQueue(
                            job_type=job_type,
                            url=url,
                            status="pending",
                            link_text="",
                        )
                    )
            db.commit()
            return True

    @staticmethod
    def _claim_crawl_batch(job_type: str, batch_size: int) -> list[tuple[str, str]]:
        with SessionLocal() as db:
            job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
            if not job or job.status != "running" or job.crawled_count >= job.max_pages:
                return []
            pending_query = db.query(CrawlerQueue).filter_by(
                job_type=job_type,
                status="pending",
            )
            if job_type == "announcements":
                pending_query = pending_query.order_by(
                    case(
                        (CrawlerQueue.url.ilike("%.pdf%"), 0),
                        *[
                            (CrawlerQueue.url.ilike(f"%{keyword}%"), 1)
                            for keyword in ANNOUNCEMENT_KEYWORDS
                        ],
                        else_=2,
                    ),
                    CrawlerQueue.id.asc(),
                )
            else:
                pending_query = pending_query.order_by(CrawlerQueue.id.asc())
            pending = pending_query.with_for_update(skip_locked=True).limit(batch_size).all()
            if not pending:
                return []
            result = [(item.url, item.link_text or "") for item in pending]
            for item in pending:
                item.status = "processing"
            job.current_url = result[0][0]
            db.commit()
            return result

    @staticmethod
    def _save_crawl_progress(
        job_type: str,
        success_urls: list[str],
        failed_urls: list[str],
        new_urls: dict[str, str],
    ) -> None:
        with SessionLocal() as db:
            if success_urls:
                db.execute(
                    update(CrawlerQueue)
                    .where(
                        CrawlerQueue.job_type == job_type,
                        CrawlerQueue.url.in_(success_urls),
                    )
                    .values(status="completed")
                )
            if failed_urls:
                db.execute(
                    update(CrawlerQueue)
                    .where(
                        CrawlerQueue.job_type == job_type,
                        CrawlerQueue.url.in_(failed_urls),
                    )
                    .values(status="failed")
                )
            existing_urls = set(
                row[0]
                for row in db.query(CrawlerQueue.url).filter(
                    CrawlerQueue.job_type == job_type,
                    CrawlerQueue.url.in_(list(new_urls)),
                )
            ) if new_urls else set()
            db.add_all(
                [
                    CrawlerQueue(
                        job_type=job_type,
                        url=url,
                        status="pending",
                        link_text=link_text,
                    )
                    for url, link_text in new_urls.items()
                    if url not in existing_urls
                ]
            )
            job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
            if job:
                job.crawled_count += len(success_urls)
            db.commit()

    @staticmethod
    def _finish_crawl_job(job_type: str) -> None:
        with SessionLocal() as db:
            job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
            if job:
                job.status = "idle"
                job.current_url = ""
                job.last_run = datetime.utcnow()
            db.commit()

    async def run_crawler(self, start_urls: list, max_pages: int = 100, job_type: str = "full", reset: bool = True):
        allowlist = self.settings.crawler_allowlist
        blocklist = self.settings.crawler_blocklist

        started = await asyncio.to_thread(
            self._initialize_crawl_job,
            start_urls,
            max_pages,
            job_type,
            reset,
        )
        if not started:
            logger.warning("Crawler %s is already running. Skipping start.", job_type)
            return

        batch_size = 3
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        async with httpx.AsyncClient(verify=False, headers=headers) as client:
            try:
                while True:
                    batch_data = await asyncio.to_thread(
                        self._claim_crawl_batch,
                        job_type,
                        batch_size,
                    )
                    if not batch_data:
                        break
                    batch_urls = [url for url, _link_text in batch_data]
                    
                    # 2. Process batch concurrently
                    logger.info(f"[{job_type}] Processing batch of {len(batch_urls)} URLs...")
                    tasks = [self._process_single_url_task(client, url, link_text, job_type, allowlist, blocklist) for url, link_text in batch_data]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 3. Analyze results
                    new_urls_dict = {} # url -> link_text
                    success_urls = []
                    failed_urls = []
                    
                    for url, result in zip(batch_urls, results):
                        if isinstance(result, Exception) or result is None:
                            logger.error(f"[{job_type}] Failed URL: {url} -> {result}")
                            failed_urls.append(url)
                        else:
                            success_urls.append(url)
                            for new_url, text in result:
                                if new_url not in new_urls_dict:
                                    new_urls_dict[new_url] = text
                                    
                    await asyncio.to_thread(
                        self._save_crawl_progress,
                        job_type,
                        success_urls,
                        failed_urls,
                        new_urls_dict,
                    )

            except Exception as e:
                logger.error(f"Crawler {job_type} encountered a fatal error: {e}")
            finally:
                await asyncio.to_thread(self._finish_crawl_job, job_type)
