"""
clustering.py
--------------
Bonus Feature: Candidate Clustering.

Groups candidates into clusters based on their skill profiles, so a recruiter
can quickly see natural groupings in a large applicant pool
(e.g. "Backend/Cloud cluster", "Frontend/UI cluster", "AI/ML cluster")
without needing a job description -- useful for general talent-pool analysis.

Method: TF-IDF vectorization of (skills + raw resume text) -> KMeans.
Cluster labels are auto-named using the top distinguishing skill terms per
cluster (explainable, not just "Cluster 0/1/2").
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


@dataclass
class ClusterResult:
    cluster_id: int
    label: str
    candidate_names: list[str] = field(default_factory=list)
    top_terms: list[str] = field(default_factory=list)


def cluster_candidates(candidates: list, n_clusters: int | None = None) -> list[ClusterResult]:
    """
    Clusters candidates by skills/content similarity.
    Auto-picks a sensible n_clusters if not provided (min(3, n_candidates)).
    """
    if len(candidates) < 2:
        return []

    n_clusters = n_clusters or min(3, len(candidates))
    n_clusters = max(1, min(n_clusters, len(candidates)))

    docs = [
        (" ".join(c.skills) + " ") * 3 + c.raw_text  # weight skills higher than free text
        for c in candidates
    ]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = vectorizer.fit_transform(docs)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)

    feature_names = np.array(vectorizer.get_feature_names_out())
    results = []
    for cluster_id in range(n_clusters):
        center = km.cluster_centers_[cluster_id]
        top_idx = center.argsort()[::-1][:5]
        top_terms = [feature_names[i] for i in top_idx if center[i] > 0]

        member_names = [c.name for c, lbl in zip(candidates, labels) if lbl == cluster_id]
        label = " / ".join(t.title() for t in top_terms[:2]) if top_terms else f"Cluster {cluster_id}"

        results.append(ClusterResult(
            cluster_id=cluster_id,
            label=label,
            candidate_names=member_names,
            top_terms=top_terms,
        ))

    return results
