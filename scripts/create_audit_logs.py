import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import init_db

if __name__ == "__main__":
    print("Creating missing tables in the database...")
    init_db()
    print("✅ Done. The audit_logs table is now ready.")
