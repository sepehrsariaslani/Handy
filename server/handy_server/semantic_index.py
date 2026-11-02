"""Optional CPU-friendly multilingual embeddings with a bounded SQLite index.

No embedding model is loaded at import. An administrator explicitly syncs
small batches. Search falls back to lexical retrieval if embeddings are off.
"""
import json
import math
from pathlib import Path
import sqlite3
import threading


class SemanticUnavailable(Exception):
    pass


class SemanticIndex:
    def __init__(self, db_path: Path, enabled: bool = False,
                 model_name: str = "intfloat/multilingual-e5-small"):
        self.db_path = Path(db_path)
        self.enabled = enabled
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        with sqlite3.connect(self.db_path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS semantic_vectors(
                kind TEXT NOT NULL,document_id TEXT NOT NULL,source_id TEXT NOT NULL,
                model TEXT NOT NULL,source_text TEXT NOT NULL,
                vector TEXT NOT NULL,
                PRIMARY KEY(kind,document_id,source_id))""")

    def _encode(self, texts: list[str], query: bool = False) -> list[list[float]]:
        if not self.enabled:
            raise SemanticUnavailable("Semantic search disabled")
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise SemanticUnavailable(
                        "Install the optional semantic dependencies"
                    ) from None
                # Never silently download models when search is disabled.
                self._model = SentenceTransformer(self.model_name, device="cpu")
            prefix = "query: " if query else "passage: "
            try:
                embeddings = self._model.encode(
                    [prefix + text[:800] for text in texts],
                    normalize_embeddings=True, batch_size=8,
                    show_progress_bar=False,
                )
                return [[float(x) for x in row] for row in embeddings]
            except Exception as exc:
                raise SemanticUnavailable(
                    "Embedding model failed (" + type(exc).__name__ + ")"
                ) from None

    def sync(self, limit: int = 100) -> dict:
        if not self.enabled:
            raise SemanticUnavailable("Semantic search is disabled")
        if not 1 <= limit <= 200:
            raise ValueError("Invalid indexing batch size")
        rows = []
        with sqlite3.connect(self.db_path) as db:
            # Only completed meetings become memory, and notes are indexed
            # at their source-line granularity.
            for kind in ("meeting", "note"):
                if len(rows) >= limit:
                    break
                # All missing / changed sources are detected in SQL.
                if kind == "meeting":
                    query = """SELECT 'meeting' AS kind,s.job_id AS document_id,
                        s.id AS source_id,substr(s.text,1,800) AS source_text
                        FROM meeting_segments s
                        JOIN meeting_jobs j ON j.id=s.job_id AND j.status='completed'
                        LEFT JOIN semantic_vectors v ON v.kind='meeting'
                          AND v.document_id=s.job_id AND v.source_id=s.id
                        WHERE v.model IS NULL OR v.model!=?
                          OR v.source_text!=substr(s.text,1,800)
                        ORDER BY s.job_id,s.id LIMIT ?"""
                else:
                    query = """SELECT 'note' AS kind,ns.note_id AS document_id,
                        ns.source_id AS source_id,substr(ns.text,1,800) AS source_text
                        FROM note_sources ns
                        LEFT JOIN semantic_vectors v ON v.kind='note'
                          AND v.document_id=ns.note_id AND v.source_id=ns.source_id
                        WHERE v.model IS NULL OR v.model!=?
                          OR v.source_text!=substr(ns.text,1,800)
                        ORDER BY ns.note_id,ns.source_id LIMIT ?"""
                fetched = db.execute(
                    query,(self.model_name,limit-len(rows))
                ).fetchall()
                rows.extend(fetched)
        if not rows:
            return {"indexed":0, "model":self.model_name}
        vectors = self._encode([row[3] for row in rows])
        with sqlite3.connect(self.db_path,timeout=20) as db:
            db.executemany("""INSERT INTO semantic_vectors
                (kind,document_id,source_id,model,source_text,vector)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(kind,document_id,source_id) DO UPDATE SET
                model=excluded.model,source_text=excluded.source_text,
                vector=excluded.vector""", [
                (*row[:3],self.model_name,row[3],json.dumps(vector))
                for row,vector in zip(rows,vectors)
            ])
        return {"indexed":len(rows),"model":self.model_name}

    def search(self, question: str, limit: int = 12) -> list[tuple[str,float]]:
        if not self.enabled:
            return []
        query = self._encode([question],query=True)[0]
        ranked=[]
        with sqlite3.connect(self.db_path,timeout=20) as db:
            # Bound the current reference implementation: large installations
            # should migrate to vector-database ANN indexing.
            rows=db.execute("""SELECT kind,document_id,source_id,vector
                FROM semantic_vectors WHERE model=? LIMIT 20000""",
                (self.model_name,)).fetchall()
        for kind,document_id,source_id,blob in rows:
            vector=json.loads(blob)
            if len(vector)!=len(query):
                continue
            score=sum(a*b for a,b in zip(query,vector))
            if math.isfinite(score):
                ranked.append((f"{kind}:{document_id}:{source_id}",score))
        ranked.sort(key=lambda x:x[1],reverse=True)
        return ranked[:limit*3]
