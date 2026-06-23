"""
vector_store.py
----------------
Bonus Feature: Vector Search Implementation.

Stores each candidate's resume as a vector embedding in ChromaDB, enabling
semantic search like: "find candidates experienced with cloud deployment"
even if that exact phrase never appears in the resume.

Embedding choice:
  ChromaDB's default embedding function downloads a model from Hugging Face
  the first time it runs, which requires internet access. To keep this
  feature 100% reliable out-of-the-box (no model download, no API key needed),
  we compute our OWN embeddings with scikit-learn's TF-IDF vectorizer and feed
  them to Chroma directly. This is a deliberate engineering trade-off:
  TF-IDF vectors are less "semantic" than a transformer embedding, but they
  are fast, deterministic, and require zero external dependencies -- a good
  fit for a project that should run anywhere, immediately, for any recruiter.

  If you want richer semantic vector search, swap `_vectorize` to use
  sentence-transformers embeddings instead -- the rest of this module is
  embedding-agnostic.
"""

from __future__ import annotations
import os
import sys

# Some cloud hosting platforms (e.g. older Streamlit Community Cloud images)
# ship a system sqlite3 version older than what ChromaDB requires (>=3.35).
# If the optional `pysqlite3-binary` package is installed, swap it in
# transparently before importing chromadb. Safe no-op if it isn't installed.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer

import chromadb

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "storage_data", "vector_store")


class CandidateVectorStore:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or _PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="candidates",
            metadata={"hnsw:space": "cosine"},
        )
        self._vectorizer: TfidfVectorizer | None = None

    def _fit_vectorizer(self, texts: list[str]):
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self._vectorizer.fit(texts)

    def index_candidates(self, candidates: list) -> None:
        """
        (Re)builds the vector index for the given list of Candidate objects.
        Call this whenever a new batch of resumes is processed.

        Note: this always rebuilds the collection from scratch rather than
        incrementally adding to it. Reason: the TF-IDF vectorizer is refit on
        whatever candidate list is passed in, so its output dimensionality
        depends on the size of that list. ChromaDB requires every embedding
        in a collection to have the *same* dimensionality, so mixing vectors
        fit on different-sized batches (e.g. 1 resume one time, 5 resumes
        another) would raise an InvalidArgumentError. Since the app always
        calls this with the full current candidate list (not incrementally),
        a full rebuild is both correct and simple.
        """
        if not candidates:
            return

        self.reset()
        texts = [c.raw_text for c in candidates]

        try:
            self._fit_vectorizer(texts)
            vectors = self._vectorizer.transform(texts).toarray().tolist()
        except ValueError:
            # E.g. "empty vocabulary" if resume text is too short/has only
            # stopwords. Vector search is a bonus feature -- skip indexing
            # rather than crashing the whole app.
            self._vectorizer = None
            return

        ids = [c.file_name for c in candidates]
        metadatas = [
            {
                "name": c.name,
                "email": c.email,
                "skills": ", ".join(c.skills),
                "years_of_experience": c.years_of_experience,
            }
            for c in candidates
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the indexed candidates by free-text query, e.g.
        "candidate with cloud deployment and Docker experience".
        Returns a list of {id, name, skills, distance} dicts, closest first.
        """
        if self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query]).toarray().tolist()
        results = self.collection.query(query_embeddings=query_vec, n_results=top_k)

        output = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i, meta, dist in zip(ids, metadatas, distances):
            output.append({
                "file_name": i,
                "name": meta.get("name", ""),
                "skills": meta.get("skills", ""),
                "years_of_experience": meta.get("years_of_experience", 0),
                "similarity": round(max(0.0, 1 - dist), 3),
            })
        return output

    def reset(self):
        """Clear the collection (useful when starting a fresh batch)."""
        try:
            self.client.delete_collection("candidates")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="candidates", metadata={"hnsw:space": "cosine"}
        )
