"""CPU-only grounded retrieval across durable notes and completed meetings.

Search uses SQLite FTS5, with stable source IDs and optional UTC date filters.
No embeddings, GPU, or external API are needed to retrieve references.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3


STOPWORDS = {
    "از", "به", "در", "را", "با", "که", "این", "آن", "برای", "تا",
    "من", "ما", "تو", "چه", "چی", "چیه", "بود", "بوده", "شد", "شده",
    "کرد", "کرده", "کنه", "کنم", "کنیم", "است", "هست", "هستم",
    "رو", "و", "یا", "یک", "هم", "چطور", "کدام", "آیا", "درباره",
    "هفته", "گذشته", "قبل", "دیروز", "امروز", "این",
    "the", "a", "an", "in", "on", "of", "to", "did", "was",
    "is", "are", "what", "who", "when", "how", "my", "our",
    "last", "week", "yesterday", "today", "this",
}


def tokens(question: str) -> list[str]:
    """Compile user prose into a literal, bounded FTS query, never FTS syntax."""
    words = re.findall(r"\w+", question.casefold(), re.UNICODE)
    kept: list[str] = []
    for word in words:
        if len(word) < 2 or word in STOPWORDS or word in kept:
            continue
        kept.append(word)
        if len(kept) == 12:
            break
    return kept


def fts_query(words: list[str]) -> str:
    return " OR ".join('"' + word.replace('"', '') + '"' for word in words)


def infer_utc_dates(
    question: str, since: date | None, until: date | None,
    today: date | None = None,
) -> tuple[date | None, date | None]:
    """Recognize explicit common relative time phrases; never silently shift
    user-supplied dates. Weeks run Monday-Sunday in UTC.
    """
    if since is not None or until is not None:
        return since, until
    today = today or datetime.now(timezone.utc).date()
    lowered = question.casefold()
    if any(term in lowered for term in ("هفته گذشته", "هفته قبل", "last week")):
        monday = today - timedelta(days=today.weekday())
        return monday - timedelta(days=7), monday - timedelta(days=1)
    if "دیروز" in lowered or "yesterday" in lowered:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    if "امروز" in lowered or "today" in lowered:
        return today, today
    if "این هفته" in lowered or "this week" in lowered:
        monday = today - timedelta(days=today.weekday())
        return monday, today
    return None, None


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with self.connect() as db:
            # The table exists before triggers are installed; the previous
            # meeting store always creates meeting_segments at startup.
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS meeting_memory_fts
                USING fts5(text, speaker, tokenize='unicode61')""")
            db.executescript("""
                CREATE TRIGGER IF NOT EXISTS meeting_memory_add
                AFTER INSERT ON meeting_segments BEGIN
                  INSERT INTO meeting_memory_fts(rowid,text,speaker)
                  VALUES(new.rowid,new.text,new.speaker);
                END;
                CREATE TRIGGER IF NOT EXISTS meeting_memory_change
                AFTER UPDATE OF text,speaker ON meeting_segments BEGIN
                  DELETE FROM meeting_memory_fts WHERE rowid=old.rowid;
                  INSERT INTO meeting_memory_fts(rowid,text,speaker)
                  VALUES(new.rowid,new.text,new.speaker);
                END;
                CREATE TRIGGER IF NOT EXISTS meeting_memory_remove
                AFTER DELETE ON meeting_segments BEGIN
                  DELETE FROM meeting_memory_fts WHERE rowid=old.rowid;
                END;
            """)
            # Backfill pre-existing recordings once. Index is incremental
            # thereafter through insert/update/delete triggers.
            db.execute("""INSERT INTO meeting_memory_fts(rowid,text,speaker)
                SELECT m.rowid,m.text,m.speaker FROM meeting_segments m
                WHERE NOT EXISTS
                  (SELECT 1 FROM meeting_memory_fts f WHERE f.rowid=m.rowid)""")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def search(self, question: str, scope: str = "all",
               since: date | None = None, until: date | None = None,
               limit: int = 12) -> list[dict]:
        words = tokens(question)
        if not words:
            return []
        expr = fts_query(words)
        after = since.isoformat() if since else None
        before = until.isoformat() if until else None
        hits: list[dict] = []
        with self.connect() as db:
            if scope in ("all", "meetings"):
                entries = db.execute("""
                    SELECT s.job_id,s.id AS segment_id,s.speaker,s.start,
                        s.text,m.created_at, rank AS relevance
                    FROM meeting_memory_fts
                    JOIN meeting_segments s ON s.rowid=meeting_memory_fts.rowid
                    JOIN meeting_jobs m ON m.id=s.job_id
                    WHERE meeting_memory_fts MATCH ?
                      AND m.status='completed'
                      AND (? IS NULL OR substr(m.created_at,1,10)>=?)
                      AND (? IS NULL OR substr(m.created_at,1,10)<=?)
                    ORDER BY rank LIMIT ?""",
                    (expr, after, after, before, before, limit * 2)).fetchall()
                for row in entries:
                    hits.append({
                        "id": f"meeting:{row['job_id']}:{row['segment_id']}",
                        "kind": "meeting", "document_id": row["job_id"],
                        "source_id": row["segment_id"],
                        "title": "Meeting " + row["created_at"][:10],
                        "quote": row["text"][:1200], "speaker": row["speaker"],
                        "start": row["start"], "created_at": row["created_at"],
                        "_score": float(row["relevance"]),
                    })
            if scope in ("all", "notes"):
                notes = db.execute("""
                    SELECT n.id,n.title,n.created_at,n.content, rank AS relevance
                    FROM notes_fts
                    JOIN notes n ON notes_fts.note_id=n.id
                    WHERE notes_fts MATCH ?
                      AND (? IS NULL OR substr(n.created_at,1,10)>=?)
                      AND (? IS NULL OR substr(n.created_at,1,10)<=?)
                    ORDER BY rank LIMIT ?""",
                    (expr, after, after, before, before, limit * 2)).fetchall()
                for n in notes:
                    candidates = db.execute("""
                        SELECT source_id,text FROM note_sources
                        WHERE note_id=? ORDER BY source_id LIMIT 1000""",
                        (n["id"],)).fetchall()
                    # Prefer an actual matching citation rather than the
                    # first unrelated line in a long note.
                    ranked = sorted(
                        candidates,
                        key=lambda r: sum(
                            word in r["text"].casefold()
                            for word in words
                        ),
                        reverse=True,
                    )
                    for row in ranked[:2]:
                        hits.append({
                            "id": f"note:{n['id']}:{row['source_id']}",
                            "kind": "note", "document_id": n["id"],
                            "source_id": row["source_id"],
                            "title": n["title"], "quote": row["text"][:1200],
                            "speaker": None, "start": None,
                            "created_at": n["created_at"],
                            "_score": float(n["relevance"]),
                        })
        # FTS scores are negative (lower is more relevant). Prefer documents
        # with matching source lines over title-only matches where possible.
        hits.sort(key=lambda x: (x["_score"], -sum(
            w in x["quote"].casefold() for w in words
        )))
        unique: list[dict] = []
        seen = set()
        for row in hits:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            row.pop("_score")
            unique.append(row)
            if len(unique) >= limit:
                break
        return unique


    def resolve(self, citation_ids: list[str],
                scope: str = "all", since: date | None = None,
                until: date | None = None) -> list[dict]:
        """Revalidate semantic/previous-turn refs against current documents.

        Deleted notes, unfinished meetings and out-of-range dates must never
        be disclosed through stale vectors or stored dialogue citations.
        """
        after = since.isoformat() if since else None
        before = until.isoformat() if until else None
        hits = []
        with self.connect() as db:
            for citation in citation_ids[:48]:
                parts = citation.split(":")
                if len(parts) != 3:
                    continue
                kind, document_id, source_id = parts
                if not re.fullmatch(r"[0-9a-f]{32}", document_id):
                    continue
                if kind == "meeting" and scope in ("all","meetings"):
                    row = db.execute("""SELECT s.text,s.speaker,s.start,j.created_at
                        FROM meeting_segments s JOIN meeting_jobs j ON j.id=s.job_id
                        WHERE s.job_id=? AND s.id=? AND j.status='completed'
                          AND (? IS NULL OR substr(j.created_at,1,10)>=?)
                          AND (? IS NULL OR substr(j.created_at,1,10)<=?)""",
                        (document_id,source_id,after,after,before,before)).fetchone()
                    if row:
                        hits.append({
                            "id":citation,"kind":"meeting","document_id":document_id,
                            "source_id":source_id,"title":"Meeting "+row["created_at"][:10],
                            "quote":row["text"][:1200],"speaker":row["speaker"],
                            "start":row["start"],"created_at":row["created_at"],
                        })
                elif kind == "note" and scope in ("all","notes"):
                    row = db.execute("""SELECT s.text,n.title,n.created_at
                        FROM note_sources s JOIN notes n ON n.id=s.note_id
                        WHERE n.id=? AND s.source_id=?
                          AND (? IS NULL OR substr(n.created_at,1,10)>=?)
                          AND (? IS NULL OR substr(n.created_at,1,10)<=?)""",
                        (document_id,source_id,after,after,before,before)).fetchone()
                    if row:
                        hits.append({
                            "id":citation,"kind":"note","document_id":document_id,
                            "source_id":source_id,"title":row["title"],
                            "quote":row["text"][:1200],"speaker":None,
                            "start":None,"created_at":row["created_at"],
                        })
        return hits
