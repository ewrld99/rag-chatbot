"""
One-time migration: Insert rrf_k into system_settings if it does not already exist.
Run from the project root: python scripts/add_rrf_k.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.orm import Session
from app.db.session import engine
from app.db.models import SystemSetting


def run():
    with Session(engine) as session:
        existing = session.query(SystemSetting).filter_by(key="rrf_k").first()
        if existing:
            print(f"rrf_k already exists in system_settings (value={existing.value!r}). Nothing to do.")
            return

        setting = SystemSetting(
            key="rrf_k",
            value="60",
            description="RRF smoothing constant — higher values reduce top-rank influence",
            category="retrieval",
            is_editable=True,
        )
        session.add(setting)
        session.commit()
        print("✅  rrf_k inserted into system_settings (value='60', category='retrieval').")


if __name__ == "__main__":
    run()
