"""
multi_job.py
------------
Bonus Feature: Multi-Job Candidate Matching.

Often a recruiter has several open positions and one pool of resumes
(e.g. unsolicited applications). This module matches every candidate
against every job in a list and reports, for each candidate, which job
they're the best fit for -- and for each job, who the top candidates are.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from matcher import rank_candidates


@dataclass
class CandidateBestFit:
    candidate_name: str
    file_name: str
    best_job_title: str
    best_score: float
    all_scores: dict = field(default_factory=dict)  # job_title -> score


def match_candidates_to_multiple_jobs(candidates: list, jobs: list) -> dict:
    """
    Returns:
      {
        "by_job": { job_title: [MatchResult, ...] (ranked) },
        "by_candidate": [CandidateBestFit, ...] (one entry per candidate,
            showing their best-fit job across all jobs provided)
      }
    """
    by_job = {}
    candidate_scores: dict[str, dict[str, float]] = {c.file_name: {} for c in candidates}

    for job in jobs:
        results = rank_candidates(candidates, job)
        by_job[job.title] = results
        for r in results:
            candidate_scores[r.file_name][job.title] = r.final_score

    by_candidate = []
    name_lookup = {c.file_name: c.name for c in candidates}
    for file_name, scores in candidate_scores.items():
        if not scores:
            continue
        best_job = max(scores, key=scores.get)
        by_candidate.append(CandidateBestFit(
            candidate_name=name_lookup.get(file_name, file_name),
            file_name=file_name,
            best_job_title=best_job,
            best_score=scores[best_job],
            all_scores=scores,
        ))

    by_candidate.sort(key=lambda x: x.best_score, reverse=True)
    return {"by_job": by_job, "by_candidate": by_candidate}
