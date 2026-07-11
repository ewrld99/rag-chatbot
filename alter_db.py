import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.session import SessionLocal
from sqlalchemy import text

def alter_table():
    db = SessionLocal()
    try:
        # Add link_text column to crawler_queue table
        db.execute(text("ALTER TABLE crawler_queue ADD COLUMN link_text TEXT;"))
        db.commit()
        print("Column link_text added successfully.")
    except Exception as e:
        db.rollback()
        print(f"Error adding column: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    alter_table()
