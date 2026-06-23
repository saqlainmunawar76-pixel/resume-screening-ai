"""
job_parser.py
-------------
Job Description Analysis (per Task AI-2 requirements):
  - Accept job descriptions as input (plain text)
  - Extract required skills
  - Identify preferred/nice-to-have qualifications
  - Analyze experience requirements
  - Generate a structured job requirements object
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict

from skills_db import extract_skills

EXPERIENCE_REGEX = re.compile(
    r"(\d+)\+?\s*(?:-|to)?\s*(\d+)?\s*(?:years|yrs)\b", re.IGNORECASE
)

# Phrases that usually introduce "nice to have" content vs "must have"
PREFERRED_MARKERS = [
    "preferred", "nice to have", "bonus", "plus", "good to have",
    "is a plus", "added advantage",
]
REQUIRED_MARKERS = [
    "required", "must have", "mandatory", "minimum qualification", "responsibilities",
]


@dataclass
class JobRequirement:
    title: str = ""
    raw_text: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    qualifications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_into_blocks(text: str) -> list[str]:
    """Split JD text into sentence/line blocks for marker-based classification."""
    blocks = re.split(r"[\n.]", text)
    return [b.strip() for b in blocks if b.strip()]


def parse_job_description(text: str, title: str = "") -> JobRequirement:
    job = JobRequirement(title=title or "Untitled Position", raw_text=text)

    blocks = _split_into_blocks(text)
    preferred_text_parts = []
    required_text_parts = []

    for block in blocks:
        lower = block.lower()
        if any(m in lower for m in PREFERRED_MARKERS):
            preferred_text_parts.append(block)
        else:
            required_text_parts.append(block)

    all_skills_in_jd = extract_skills(text)
    preferred_skills = extract_skills(" ".join(preferred_text_parts)) if preferred_text_parts else []

    job.preferred_skills = preferred_skills
    job.required_skills = [s for s in all_skills_in_jd if s not in preferred_skills]

    # Experience requirement: take the largest number mentioned with "years"
    years_found = []
    for match in EXPERIENCE_REGEX.finditer(text):
        years_found.append(float(match.group(1)))
    job.min_experience_years = max(years_found) if years_found else 0.0

    # Qualifications: lines that look like degree/education requirements
    qualification_keywords = [
        "bachelor", "master", "bs ", "ms ", "b.sc", "m.sc", "degree",
        "phd", "diploma",
    ]
    job.qualifications = [
        b for b in blocks
        if any(k in b.lower() for k in qualification_keywords)
    ]

    return job
