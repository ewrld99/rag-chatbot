"""Import retrieval aliases from CSV.

CSV columns:
    term, aliases, category, weight, is_active

aliases can be either a JSON array or a semicolon/comma-separated string.
Example:
    sr,"sr2; student records; student portal",portal,1.0,true

Usage:
    python scripts/import_retrieval_aliases.py aliases.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func

from app.db.models import RetrievalAlias
from app.db.session import SessionLocal, engine, run_sql_migrations
from app.services.alias_expansion_service import AliasExpansionService


def parse_aliases(raw: str | None) -> list[str]:
    if not raw:
        return []

    value = raw.strip()
    if not value:
        return []

    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("aliases JSON must be an array")
        return [str(item).strip() for item in parsed if str(item).strip()]

    delimiter = ";" if ";" in value else ","
    return [item.strip() for item in value.split(delimiter) if item.strip()]


def parse_bool(raw: str | None, default: bool = True) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "active"}


def parse_weight(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return 1.0
    return float(raw)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    term = str(row.get("term", "")).strip().lower()
    if not term:
        raise ValueError("term is required")

    return {
        "term": term,
        "aliases": parse_aliases(row.get("aliases")),
        "category": str(row.get("category") or "").strip() or None,
        "weight": parse_weight(row.get("weight")),
        "is_active": parse_bool(row.get("is_active"), default=True),
    }


def import_aliases(csv_path: Path, dry_run: bool = False) -> tuple[int, int]:
    with engine.begin() as conn:
        run_sql_migrations(conn)

    created = 0
    updated = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"term", "aliases"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

        with SessionLocal() as db:
            for line_number, row in enumerate(reader, start=2):
                try:
                    data = normalize_row(row)
                except Exception as exc:
                    raise ValueError(f"Invalid row {line_number}: {exc}") from exc

                existing = (
                    db.query(RetrievalAlias)
                    .filter(func.lower(RetrievalAlias.term) == data["term"])
                    .first()
                )
                if existing:
                    existing.aliases = data["aliases"]
                    existing.category = data["category"]
                    existing.weight = data["weight"]
                    existing.is_active = data["is_active"]
                    updated += 1
                else:
                    db.add(RetrievalAlias(**data))
                    created += 1

            if dry_run:
                db.rollback()
            else:
                db.commit()
                AliasExpansionService.clear_cache()

    return created, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Import retrieval aliases from CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    created, updated = import_aliases(args.csv_path, dry_run=args.dry_run)
    mode = "would create/update" if args.dry_run else "created/updated"
    print(f"{mode}: {created}/{updated}")


if __name__ == "__main__":
    main()
