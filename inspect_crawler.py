import os
import sys

# Ensure the app module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.db.session import SessionLocal
from app.db.models import CrawlerJob, CrawlerQueue
from sqlalchemy import func

def inspect_db():
    db = SessionLocal()
    try:
        jobs = db.query(CrawlerJob).all()
        print("--- Crawler Jobs ---")
        for job in jobs:
            print(f"Type: {job.job_type}, Status: {job.status}, Crawled: {job.crawled_count}/{job.max_pages}")
            
        print("\n--- Crawler Queue Status Counts ---")
        for job in jobs:
            counts = db.query(CrawlerQueue.status, func.count(CrawlerQueue.id)).filter_by(job_type=job.job_type).group_by(CrawlerQueue.status).all()
            print(f"Job Type: {job.job_type}")
            for status, count in counts:
                print(f"  {status}: {count}")
    finally:
        db.close()

if __name__ == "__main__":
    inspect_db()
