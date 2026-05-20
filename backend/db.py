"""
SQLite-Persistierung für NewsTracker.

Speichert abgerufene Artikel zusammen mit der Fake-News-Bewertung in einer
lokalen SQLite-Datenbank. Für den Produktivbetrieb lässt sich das Schema
1:1 auf PostgreSQL übertragen (siehe README).

Schema:
    news(id, title, url, category, credibility_score, content,
         description, source, label, fetched_at)
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

DEFAULT_DB_PATH = os.environ.get(
    "NEWSTRACKER_DB",
    os.path.join(os.path.dirname(__file__), "..", "data", "newstracker.sqlite3"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    title             TEXT NOT NULL,
    url               TEXT UNIQUE,
    category          TEXT,
    credibility_score REAL,
    content           TEXT,
    description       TEXT,
    source            TEXT,
    label             TEXT,
    fetched_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_news_category ON news(category);
CREATE INDEX IF NOT EXISTS idx_news_label    ON news(label);
"""


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH):
    """Öffnet eine SQLite-Verbindung mit aktivierter Row-Factory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Legt das Schema an, falls es noch nicht existiert."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_articles(
    articles: Iterable[Dict],
    category: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Speichert (oder aktualisiert) Artikel. Gibt die Anzahl Inserts zurück.

    Erwartetes Artikel-Schema (vgl. api_connector.classify_articles):
        title, description, url, source, label, confidence
    """
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    with connect(db_path) as conn:
        for a in articles:
            # INSERT OR IGNORE über die unique URL verhindert Duplikate
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO news
                    (title, url, category, credibility_score, content,
                     description, source, label, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.get("title", ""),
                    a.get("url"),
                    category or a.get("category"),
                    a.get("confidence"),
                    a.get("content"),
                    a.get("description"),
                    a.get("source"),
                    a.get("label"),
                    now,
                ),
            )
            inserted += cur.rowcount
    return inserted


def list_articles(
    category: Optional[str] = None,
    label: Optional[str] = None,
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """Liest Artikel, optional gefiltert nach Kategorie und Label."""
    init_db(db_path)
    sql = "SELECT * FROM news WHERE 1=1"
    params: List = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if label:
        sql += " AND label = ?"
        params.append(label)
    sql += " ORDER BY fetched_at DESC LIMIT ?"
    params.append(limit)

    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]
