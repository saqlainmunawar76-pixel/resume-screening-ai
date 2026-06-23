"""
matcher.py
----------
Candidate Matching Engine + core AI Features:
  - Skill matching (overlap-based)
  - Semantic similarity analysis (embedding cosine similarity)
  - Candidate ranking
  - Explainable score breakdown (strengths & weaknesses)

Scoring methodology (documented here so it is explainable, not a black box):

    final_score = (skill_weight * skill_score)
                + (semantic_weight * semantic_score)
                + (experience_weight * experience_score)

    Default weights: skill=0.5, semantic=0.3, experience=0.2
    (These are configurable -> see WEIGHTS below.)

  - skill_score:      Jaccard-style overlap between candidate skills and
                       job required+preferred skills (required weighted higher).
  - semantic_score:   Cosine similarity between the embedding of the
                       candidate's full resume text and the job description
                       text. Captures meaning/context beyond exact keywords
                       (e.g. "built REST APIs" matching "API development").
  - experience_score: How candidate's years_of_experience compares to the
                       job's min_experience_years requirement.

Embeddings:
  We try to use sentence-transformers (richer semantic understanding) if
  it's installed AND a model can be loaded. If that's unavailable (e.g. no
  internet to download the model), we gracefully fall back to TF-IDF vectors
  from scikit-learn, which work fully offline. Either way the matching engine
  keeps working -- this fallback is intentional, not a bug.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from extractor import Candidate
from job_parser import JobRequirement

WEIGHTS = {"skill": 0.5, "semantic": 0.3, "experience": 0.2}

_EMBEDDER = None
_EMBEDDER_KIND = None  # "sentence-transformer" | "tfidf" | None


def _try_load_sentence_transformer():
    global _EMBEDDER, _EMBEDDER_KIND
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDER_KIND = "sentence-transformer"
    except Exception:
        _EMBEDDER = None
        _EMBEDDER_KIND = None


def get_embedder_kind() -> str:
    """Lets the UI show which embedding backend is active."""
    if _EMBEDDER_KIND is None:
        _try_load_sentence_transformer()
    return _EMBEDDER_KIND or "tfidf"


def _semantic_scores(resume_texts: list[str], jd_text: str) -> list[float]:
    """
    Returns a cosine-similarity score (0-1) between each resume text and the
    job description text, using whichever embedding backend is available.
    """
    if _EMBEDDER_KIND is None:
        _try_load_sentence_transformer()

    if _EMBEDDER_KIND == "sentence-transformer":
        all_texts = resume_texts + [jd_text]
        vectors = _EMBEDDER.encode(all_texts)
        jd_vec = vectors[-1].reshape(1, -1)
        resume_vecs = vectors[:-1]
        sims = cosine_similarity(resume_vecs, jd_vec).flatten()
        return [float(max(0.0, min(1.0, s))) for s in sims]

    # --- TF-IDF fallback (fully offline, no model download needed) ---
    corpus = resume_texts + [jd_text]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vec = tfidf_matrix[-1]
    resume_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(resume_vecs, jd_vec).flatten()
    return [float(max(0.0, min(1.0, s))) for s in sims]


def _skill_score(candidate_skills: list[str], job: JobRequirement) -> tuple[float, list[str], list[str]]:
    cand_set = set(candidate_skills)
    required = set(job.required_skills)
    preferred = set(job.preferred_skills)

    matched_required = cand_set & required
    matched_preferred = cand_set & preferred
    missing_required = required - cand_set

    if not required and not preferred:
        return 0.5, sorted(matched_required | matched_preferred), sorted(missing_required)

    # Required skills matter more than preferred ones (0.8 / 0.2 split)
    req_ratio = (len(matched_required) / len(required)) if required else 1.0
    pref_ratio = (len(matched_preferred) / len(preferred)) if preferred else 1.0
    score = 0.8 * req_ratio + 0.2 * pref_ratio

    matched = sorted(matched_required | matched_preferred)
    missing = sorted(missing_required)
    return score, matched, missing


def _experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 1.0  # no requirement stated -> don't penalize
    if candidate_years >= required_years:
        return 1.0
    if candidate_years <= 0:
        return 0.0
    return round(candidate_years / required_years, 3)


@dataclass
class MatchResult:
    candidate_name: str
    file_name: str
    final_score: float
    skill_score: float
    semantic_score: float
    experience_score: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _build_strengths_weaknesses(result_partial: dict, candidate: Candidate) -> tuple[list[str], list[str]]:
    strengths, weaknesses = [], []

    if result_partial["skill_score"] >= 0.7:
        strengths.append(f"Strong skill match ({len(result_partial['matched_skills'])} relevant skills found).")
    elif result_partial["skill_score"] < 0.4:
        weaknesses.append("Low overlap with required skill set.")

    if result_partial["semantic_score"] >= 0.5:
        strengths.append("Resume content closely aligns with the job description context.")
    elif result_partial["semantic_score"] < 0.2:
        weaknesses.append("Resume content has limited contextual overlap with the job description.")

    if result_partial["experience_score"] >= 1.0:
        strengths.append("Meets or exceeds required experience level.")
    elif result_partial["experience_score"] < 0.5:
        weaknesses.append("Experience is below the job's requirement.")

    if result_partial["missing_skills"]:
        top_missing = ", ".join(result_partial["missing_skills"][:5])
        weaknesses.append(f"Missing required skill(s): {top_missing}.")

    if candidate.certifications:
        strengths.append("Has relevant certifications listed.")
    if candidate.projects:
        strengths.append("Has demonstrable project experience.")

    if not strengths:
        strengths.append("No standout strengths identified relative to this job.")
    if not weaknesses:
        weaknesses.append("No major gaps identified.")

    return strengths, weaknesses


def rank_candidates(candidates: list[Candidate], job: JobRequirement,
                     weights: dict | None = None) -> list[MatchResult]:
    """
    Core Candidate Matching Engine.
    Returns candidates sorted by final_score, descending, each with a full
    explainable breakdown.
    """
    if not candidates:
        return []

    w = weights or WEIGHTS
    resume_texts = [c.raw_text for c in candidates]
    semantic_scores = _semantic_scores(resume_texts, job.raw_text)

    results = []
    for candidate, sem_score in zip(candidates, semantic_scores):
        skill_score, matched, missing = _skill_score(candidate.skills, job)
        exp_score = _experience_score(candidate.years_of_experience, job.min_experience_years)

        final = (
            w["skill"] * skill_score
            + w["semantic"] * sem_score
            + w["experience"] * exp_score
        )

        partial = {
            "skill_score": round(skill_score, 3),
            "semantic_score": round(sem_score, 3),
            "experience_score": round(exp_score, 3),
            "matched_skills": matched,
            "missing_skills": missing,
        }
        strengths, weaknesses = _build_strengths_weaknesses(partial, candidate)

        results.append(MatchResult(
            candidate_name=candidate.name,
            file_name=candidate.file_name,
            final_score=round(final * 100, 2),  # percentage, easier to read
            skill_score=round(skill_score * 100, 2),
            semantic_score=round(sem_score * 100, 2),
            experience_score=round(exp_score * 100, 2),
            matched_skills=matched,
            missing_skills=missing,
            strengths=strengths,
            weaknesses=weaknesses,
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results
