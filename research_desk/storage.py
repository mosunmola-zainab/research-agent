import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    workflow TEXT NOT NULL,
    model TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    finished_utc TEXT,
    status TEXT NOT NULL,
    recent_query TEXT,
    foundational_query TEXT,
    candidate_count INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    openalex_id TEXT PRIMARY KEY,
    doi TEXT UNIQUE,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    publication_date TEXT,
    source_url TEXT NOT NULL,
    venue TEXT,
    first_run_id INTEGER NOT NULL REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    openalex_id TEXT NOT NULL UNIQUE REFERENCES papers(openalex_id),
    category TEXT NOT NULL,
    cited_by_count INTEGER,
    description TEXT NOT NULL,
    why_worth_reading TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def start_run(connection: sqlite3.Connection, workflow: str, model: str) -> int:
    with connection:
        cursor = connection.execute(
            "INSERT INTO runs (workflow, model, started_utc, status) VALUES (?, ?, ?, 'running')",
            (workflow, model, utc_now()),
        )
    return cursor.lastrowid


def finish_run(
    connection: sqlite3.Connection,
    run_id: int,
    status: str,
    recent_query: str | None = None,
    foundational_query: str | None = None,
    candidate_count: int | None = None,
    error: str | None = None,
) -> None:
    with connection:
        connection.execute(
            """UPDATE runs SET finished_utc = ?, status = ?, recent_query = ?, foundational_query = ?,
               candidate_count = ?, error = ? WHERE id = ?""",
            (utc_now(), status, recent_query, foundational_query, candidate_count, error, run_id),
        )


def known_paper_identifiers(connection: sqlite3.Connection) -> set[str]:
    identifiers = set()
    for row in connection.execute("SELECT openalex_id, doi FROM papers"):
        identifiers.add(row["openalex_id"])
        if row["doi"]:
            identifiers.add(row["doi"])
    return identifiers


def save_recommendations(connection: sqlite3.Connection, run_id: int, recommendations: list[dict]) -> int:
    saved_count = 0
    with connection:
        for recommendation in recommendations:
            # OR IGNORE makes saving idempotent: a paper already stored is never duplicated.
            connection.execute(
                """INSERT OR IGNORE INTO papers
                   (openalex_id, doi, title, authors, publication_date, source_url, venue, first_run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    recommendation["openalex_id"],
                    recommendation["doi"],
                    recommendation["title"],
                    json.dumps(recommendation["authors"]),
                    recommendation["publication_date"],
                    recommendation["source_url"],
                    recommendation["venue"],
                    run_id,
                ),
            )
            stored_paper = connection.execute(
                "SELECT 1 FROM papers WHERE openalex_id = ?", (recommendation["openalex_id"],)
            ).fetchone()
            if stored_paper is None:
                # The DOI already belongs to a stored paper under another OpenAlex ID, so this is a duplicate.
                continue
            cursor = connection.execute(
                """INSERT OR IGNORE INTO recommendations
                   (run_id, openalex_id, category, cited_by_count, description, why_worth_reading)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    recommendation["openalex_id"],
                    recommendation["category"],
                    recommendation["cited_by_count"],
                    recommendation["description"],
                    recommendation["why_worth_reading"],
                ),
            )
            saved_count += cursor.rowcount
    return saved_count
