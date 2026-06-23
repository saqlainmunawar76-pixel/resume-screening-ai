"""
vector_store.py
----------------
Bonus Feature: Vector Search Implementation.

Stores each candidate's resume as a vector embedding, enabling semantic
search like: "find candidates experienced with cloud deployment" even if
that exact phrase never appears in the resume.

Why a custom store instead of a vector DB library (e.g. ChromaDB)?
  Vector DB libraries pull in heavy dependency chains (gRPC, OpenTelemetry,
  protobuf, etc.) that frequently break on newer Python versions or conflict
  with other packages' pinned versions -- exactly the kind of fragile,
  hard-to-debug deployment failure this project should avoid. Since our
  candidate pool is small (a few hundred resumes at most, realistically),
  a plain NumPy array with cosine similarity is just as fast in practice
  and has zero extra dependencies beyond what the rest of this project
  already needs (numpy + scikit-learn).

Embedding choice:
  Embeddings are computed locally with scikit-learn's TF-IDF vectorizer
  (no model download, no API key, no internet required). If you want
  richer semantic embeddings, swap `_fit_vectorizer`/`_vectorize` for a
  sentence-transformers model -- the rest of this module (storage, cosine
  search) is embedding-agnostic.
"""

from __future__ import annotations
import os
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "storage_data", "vector_store")


class CandidateVectorStore:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or _PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None

        self._load_from_disk()

    # -- persistence -------------------------------------------------------
    @property
    def _vectors_path(self) -> str:
        return os.path.join(self.persist_dir, "vectors.npy")

    @property
    def _metadata_path(self) -> str:
        return os.path.join(self.persist_dir, "metadata.json")

    @property
    def _vectorizer_path(self) -> str:
        return os.path.join(self.persist_dir, "vectorizer.pkl")

    def _load_from_disk(self) -> None:
        try:
            if os.path.exists(self._vectors_path) and os.path.exists(self._metadata_path):
                self._vectors = np.load(self._vectors_path)
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            if os.path.exists(self._vectorizer_path):
                with open(self._vectorizer_path, "rb") as f:
                    self._vectorizer = pickle.load(f)
        except Exception:
            # Corrupted or incompatible cache -- start fresh rather than crash.
            self._vectors, self._metadata, self._vectorizer = None, [], None

    def _save_to_disk(self) -> None:
        try:
            if self._vectors is not None:
                np.save(self._vectors_path, self._vectors)
            with open(self._metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f)
            if self._vectorizer is not None:
                with open(self._vectorizer_path, "wb") as f:
                    pickle.dump(self._vectorizer, f)
        except Exception:
            pass  # persistence is a nice-to-have, not critical to app function

    # -- indexing ------------------------------------------------------------
    def index_candidates(self, candidates: list) -> None:
        """
        (Re)builds the vector index for the given list of Candidate objects.
        Call this whenever a new batch of resumes is processed. Always does
        a full rebuild (simpler and avoids any embedding-dimension drift
        between calls with different-sized candidate batches).
        """
        if not candidates:
            self.reset()
            return

        texts = [c.raw_text for c in candidates]
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
            vectors = vectorizer.fit_transform(texts).toarray()
        except ValueError:
            # E.g. "empty vocabulary" if resume text is too short/has only
            # stopwords. Vector search is a bonus feature -- skip indexing
            # rather than crashing the whole app.
            self.reset()
            return

        self._vectorizer = vectorizer
        self._vectors = vectors
        self._metadata = [
            {
                "file_name": c.file_name,
                "name": c.name,
                "email": c.email,
                "skills": ", ".join(c.skills),
                "years_of_experience": c.years_of_experience,
            }
            for c in candidates
        ]
        self._save_to_disk()

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the indexed candidates by free-text query, e.g.
        "candidate with cloud deployment and Docker experience".
        Returns a list of {file_name, name, skills, similarity} dicts,
        closest match first.
        """
        if self._vectorizer is None or self._vectors is None or len(self._metadata) == 0:
            return []

        query_vec = self._vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_vec, self._vectors).flatten()

        top_k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]

        output = []
        for idx in top_indices:
            meta = self._metadata[idx]
            output.append({
                "file_name": meta.get("file_name", ""),
                "name": meta.get("name", ""),
                "skills": meta.get("skills", ""),
                "years_of_experience": meta.get("years_of_experience", 0),
                "similarity": round(float(max(0.0, sims[idx])), 3),
            })
        return output

    def reset(self) -> None:
        """Clear the index (useful when starting a fresh batch)."""
        self._vectors, self._metadata, self._vectorizer = None, [], None
        for path in (self._vectors_path, self._metadata_path, self._vectorizer_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
