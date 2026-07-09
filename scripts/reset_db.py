import os
import sys
import subprocess
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.db.session import engine, init_db

def reset_db():
    print("WARNING: This will delete ALL data, users, and documents in the database.")
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    print("Dropping the public schema to clear all tables and data...")
    with engine.connect() as conn:
        # Drop and recreate the public schema (wipes all tables, triggers, types)
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        
        # Ensure extensions are re-created since they might be dropped
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
        
        conn.commit()

    print("Database cleared successfully.")
    
    print("Re-creating all base tables and triggers...")
    init_db()

    print("Stamping Alembic migrations to current head...")
    # Because init_db() creates all tables defined in models.py, we just tell Alembic
    # that the DB is already fully up to date to prevent duplicate creation errors.
    try:
        subprocess.run(["alembic", "stamp", "head"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error stamping alembic migrations: {e}")
        sys.exit(1)

    print("Database schema successfully reset and fully up to date.")

if __name__ == "__main__":
    reset_db()
