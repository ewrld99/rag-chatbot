import logging
import time
import hashlib
import os
import uuid
import asyncio
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import markdownify

from sqlalchemy.orm import Session
from sqlalchemy import case, update
from app.db.models import DocumentModel, DocumentChunk, ExternalLinkModel, User, CrawlerJob, CrawlerQueue
from app.db.session import SessionLocal
from app.services.settings_service import SettingsService
from app.utils.chunking import split_text
from app.services.embedding_service import get_embeddings
from app.utils.loaders import load_document

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
        self.db = db
        self.settings = SettingsService(db)
        # Ensure we have a system user for crawler uploads
        self.system_user = self.db.query(User).filter(User.username == "system_crawler").first()
        if not self.system_user:
            self.system_user = User(username="system_crawler", password_hash="dummy")
            self.db.add(self.system_user)
            self.db.commit()
            self.db.refresh(self.system_user)

        # Self-healing migration for link_text column
        from sqlalchemy import inspect, text
        inspector = inspect(self.db.bind)
        columns = [c['name'] for c in inspector.get_columns('crawler_queue')]
        if 'link_text' not in columns:
            try:
                self.db.execute(text("ALTER TABLE crawler_queue ADD COLUMN link_text TEXT"))
                self.db.commit()
                logger.info("Successfully added link_text column to crawler_queue")
            except Exception as e:
                self.db.rollback()
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
        doc_by_hash = db.query(DocumentModel).filter(DocumentModel.content_hash == file_hash).first()
        if doc_by_hash:
            logger.info(f"PDF already exists with same content hash. Skipping {url}.")
            return

        # Fallback check based on source_url
        doc = db.query(DocumentModel).filter(DocumentModel.source_url == url).first()
        
        if doc and doc.content_hash == file_hash:
            logger.info(f"PDF {url} has not changed. Skipping.")
            return

        if doc and doc.filename:
            # Overwrite existing file to prevent duplicates
            filename = doc.filename
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", filename)
        else:
            filename = os.path.basename(urlparse(url).path)
            if title_hint:
                import re
                safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title_hint).strip()
                if safe_title:
                    filename = f"{safe_title[:50].replace(' ', '_')}.pdf"
                    
            if not filename.lower().endswith(".pdf"):
                filename = "document.pdf"
                
            # Use UUID to prevent race conditions during concurrent crawler runs
            name_without_ext = os.path.splitext(filename)[0]
            filename = f"{name_without_ext}_{uuid.uuid4().hex[:8]}.pdf"
            
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", filename)

        with open(file_path, "wb") as f:
            f.write(content)
            
        text = load_document(file_path, "pdf", "fast")
        if not text or not text.strip():
            logger.warning(f"Could not extract text from {url}")
            return
            
        self._embed_and_save(url, filename, text, file_hash, "pdf", db, file_path=file_path)

    def _embed_and_save(self, url: str, title: str, text: str, content_hash: str, category: str, db: Session, doc_metadata: dict | None = None, file_path: str = None):
        local_settings = SettingsService(db)
        chunk_size = local_settings.chunk_size
        chunk_overlap = local_settings.chunk_overlap
        
        chunks = split_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            return
            
        embeddings = get_embeddings(chunks, db)
        
        doc = db.query(DocumentModel).filter(DocumentModel.source_url == url).first()
        if not doc:
            sys_user = db.query(User).filter(User.username == "system_crawler").first()
            doc = DocumentModel(
                title=title,
                filename=title,
                file_path=file_path,
                source_url=url,
                category=category,
                uploaded_by=sys_user.id if sys_user else None,
                content_hash=content_hash,
                last_crawled_at=datetime.utcnow()
            )
            db.add(doc)
            db.flush()
        else:
            doc.title = title
            doc.filename = title
            if file_path:
                doc.file_path = file_path
            doc.content_hash = content_hash
            doc.last_crawled_at = datetime.utcnow()
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            db.flush()

        # Merge crawler-specific fields with any caller-supplied metadata
        # (e.g. programme, year from future admin tooling).
        base_meta = {"source_url": url, "crawled_at": datetime.utcnow().isoformat()}
        if doc_metadata:
            base_meta.update(doc_metadata)

        new_chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                chunk_text=chunk,
                embedding=embedding,
                metadata_=base_meta
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        db.add_all(new_chunks)
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

    async def run_crawler(self, start_urls: list, max_pages: int = 100, job_type: str = "full", reset: bool = True):
        allowlist = self.settings.crawler_allowlist
        blocklist = self.settings.crawler_blocklist
        
        # Initialize or resume Job in DB
        with SessionLocal() as db:
            job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
            if not job:
                job = CrawlerJob(job_type=job_type, status="running", max_pages=max_pages, crawled_count=0)
                db.add(job)
            else:
                if job.status == "running" and not reset:
                    logger.warning(f"Crawler {job_type} is already running. Skipping start.")
                    return
                # Force reset if reset=True
                job.status = "running"
                job.max_pages = max_pages
                if reset:
                    job.crawled_count = 0
                    db.query(CrawlerQueue).filter_by(job_type=job_type).delete()
            
            if not reset:
                # Reset stuck 'processing' urls to 'pending'
                db.execute(update(CrawlerQueue).where(CrawlerQueue.job_type == job_type, CrawlerQueue.status == 'processing').values(status='pending'))
            
            # Insert start_urls (ignore duplicates by fetching existing first)
            for url in start_urls:
                existing = db.query(CrawlerQueue).filter_by(job_type=job_type, url=url).first()
                if not existing:
                    db.add(CrawlerQueue(job_type=job_type, url=url, status='pending', link_text=""))
            db.commit()

        batch_size = 3
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        async with httpx.AsyncClient(verify=False, headers=headers) as client:
            try:
                while True:
                    # 1. Check job status and fetch a batch
                    with SessionLocal() as db:
                        job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
                        if not job or job.status != "running" or job.crawled_count >= job.max_pages:
                            break
                        
                        pending_query = db.query(CrawlerQueue).filter_by(job_type=job_type, status='pending')
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
                        pending_batch = pending_query.limit(batch_size).all()
                        if not pending_batch:
                            break
                            
                        batch_data = [(item.url, item.link_text or "") for item in pending_batch]
                        batch_urls = [item.url for item in pending_batch]
                        
                        for item in pending_batch:
                            item.status = 'processing'
                        
                        # Also update current_url so frontend knows we are working
                        job.current_url = batch_urls[0]
                        db.commit()
                    
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
                                    
                    # 4. Save progress
                    with SessionLocal() as db:
                        if success_urls:
                            db.execute(update(CrawlerQueue).where(CrawlerQueue.job_type == job_type, CrawlerQueue.url.in_(success_urls)).values(status='completed'))
                        if failed_urls:
                            db.execute(update(CrawlerQueue).where(CrawlerQueue.job_type == job_type, CrawlerQueue.url.in_(failed_urls)).values(status='failed'))
                            
                        # Insert newly discovered links
                        for new_url, link_text in new_urls_dict.items():
                            existing = db.query(CrawlerQueue).filter_by(job_type=job_type, url=new_url).first()
                            if not existing:
                                db.add(CrawlerQueue(job_type=job_type, url=new_url, status='pending', link_text=link_text))
                                    
                        job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
                        if job:
                            job.crawled_count += len(success_urls)
                        db.commit()

            except Exception as e:
                logger.error(f"Crawler {job_type} encountered a fatal error: {e}")
            finally:
                # Job Finished safely
                with SessionLocal() as db:
                    job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
                    if job:
                        job.status = "idle"
                        job.current_url = ""
                        job.last_run = datetime.utcnow()
                    db.commit()
