"""
storage.py
----------
Simple persistence layer using SQLite (no external DB server required --
keeps the project easy to run anywhere). Stores extracted candidate
information so it survives app restarts, and supports exporting results.

Per task requirement "Store extracted information" + "Proper data handling".
"""

from __future__ import annotations
import sqlite3
import json
import os
from contextlib import contextmanager

from extractor import Candidate

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage_data", "candidates.db")


@contextmanager
def _connection():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                file_name TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT,
                linkedin TEXT,
                github TEXT,
                skills TEXT,
                education TEXT,
                certifications TEXT,
                experience TEXT,
                projects TEXT,
                years_of_experience REAL,
                raw_text TEXT
            )
        """)


def save_candidate(candidate: Candidate):
    init_db()
    with _connection() as conn:
        conn.execute("""
            INSERT INTO candidates
                (file_name, name, email, phone, linkedin, github, skills,
                 education, certifications, experience, projects,
                 years_of_experience, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_name) DO UPDATE SET
                name=excluded.name, email=excluded.email, phone=excluded.phone,
                linkedin=excluded.linkedin, github=excluded.github,
                skills=excluded.skills, education=excluded.education,
                certifications=excluded.certifications, experience=excluded.experience,
                projects=excluded.projects, years_of_experience=excluded.years_of_experience,
                raw_text=excluded.raw_text
        """, (
            candidate.file_name, candidate.name, candidate.email, candidate.phone,
            candidate.linkedin, candidate.github,
            json.dumps(candidate.skills), json.dumps(candidate.education),
            json.dumps(candidate.certifications), json.dumps(candidate.experience),
            json.dumps(candidate.projects), candidate.years_of_experience,
            candidate.raw_text,
        ))


def load_all_candidates() -> list[Candidate]:
    init_db()
    with _connection() as conn:
        rows = conn.execute("SELECT * FROM candidates").fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM candidates").description]

    candidates = []
    for row in rows:
        d = dict(zip(cols, row))
        candidates.append(Candidate(
            file_name=d["file_name"], name=d["name"], email=d["email"], phone=d["phone"],
            linkedin=d["linkedin"], github=d["github"],
            skills=json.loads(d["skills"] or "[]"),
            education=json.loads(d["education"] or "[]"),
            certifications=json.loads(d["certifications"] or "[]"),
            experience=json.loads(d["experience"] or "[]"),
            projects=json.loads(d["projects"] or "[]"),
            years_of_experience=d["years_of_experience"] or 0.0,
            raw_text=d["raw_text"] or "",
        ))
    return candidates


def clear_all():
    init_db()
    with _connection() as conn:
        conn.execute("DELETE FROM candidates")
