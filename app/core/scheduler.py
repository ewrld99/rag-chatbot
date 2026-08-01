import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.crawler_service import CrawlerService

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

async def run_full_crawler(reset: bool = True):
    mode = "fresh" if reset else "resume"
    logger.info("Starting Full Weekly Crawler (%s)...", mode)
    db = SessionLocal()
    try:
        svc = CrawlerService(db)
        await svc.run_crawler(["https://www.udom.ac.tz/"], max_pages=5000, job_type="full", reset=reset)
    except Exception as e:
        logger.error(f"Full Crawler failed: {e}")
    finally:
        db.close()
    logger.info("Full Weekly Crawler finished.")

async def run_announcement_crawler():
    logger.info("Starting Hourly Announcement Crawler...")
    db = SessionLocal()
    try:
        svc = CrawlerService(db)
        start_urls = [
            "https://www.udom.ac.tz/announcements",
            "https://www.udom.ac.tz/blog/index",
        ]
        # Crawl listings, recent announcement/blog detail pages, and PDFs found there.
        await svc.run_crawler(start_urls, max_pages=30, job_type="announcements")
    except Exception as e:
        logger.error(f"Announcement Crawler failed: {e}")
    finally:
        db.close()
    logger.info("Hourly Announcement Crawler finished.")

def start_scheduler():
    if not settings.CRAWLER_ENABLED:
        logger.info("Background crawler scheduler disabled by CRAWLER_ENABLED=false.")
        return

    # Full crawler runs every week (e.g., Sunday at 2 AM)
    scheduler.add_job(run_full_crawler, 'cron', day_of_week='sun', hour=2, minute=0, id='full_crawler')
    
    # Announcement crawler runs every hour
    scheduler.add_job(run_announcement_crawler, 'interval', hours=1, id='announcement_crawler')
    
    scheduler.start()
    logger.info("Background scheduler started.")

def stop_scheduler():
    if not scheduler.running:
        return

    scheduler.shutdown()
    logger.info("Background scheduler stopped.")
