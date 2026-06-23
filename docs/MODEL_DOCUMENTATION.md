# Model & Scoring Methodology Documentation

This document explains **how** the system makes its decisions, so the
scoring is explainable rather than a black box — this is one of the
evaluation criteria for the task (Documentation and Explainability — 10%).

## 1. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Resume (PDF)    │────▶│  extractor.py     │────▶│  Candidate object   │
└─────────────────┘     │  (pdfplumber +     │     │  (name, email,      │
                         │   regex + section  │     │   skills, exp...)   │
                         │   heuristics)       │     └─────────┬───────────┘
                                                                │
┌─────────────────┐     ┌──────────────────┐                   │
│ Job Description  │────▶│  job_parser.py    │                   │
│  (plain text)     │     │  (skill + exp      │                   │
└─────────────────┘     │   extraction)       │                   │
                         └─────────┬──────────┘                   │
                                   │                               │
                                   ▼                               ▼
                         ┌─────────────────────────────────────────────┐
                         │            matcher.py (core engine)          │
                         │  skill_score + semantic_score + exp_score    │
                         │       -> weighted final_score (0-100%)       │
                         └─────────────────┬─────────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
     ai_features.py (Gemini)      vector_store.py (NumPy)      clustering.py / multi_job.py
     summaries, interview Qs,     semantic search over           candidate grouping,
     skill-gap, suggestions,      resume pool                    multi-job ranking
     chat assistant
```

All modules live in `src/` and are orchestrated by `app.py` (the Streamlit
dashboard). Extracted candidates are persisted in SQLite (`storage.py`) so
they survive between sessions.

## 2. Resume Information Extraction

Instead of relying on a single rigid template, extraction uses **section-header
detection**: the resume text is scanned line-by-line for known section header
keywords (e.g. "Experience", "Work Experience", "Employment History" all map
to the same canonical "experience" section). Everything between one header
and the next is treated as that section's content.

- **Name**: heuristically chosen from the first few non-empty lines that
  aren't an email/phone/URL and look like a short proper name.
- **Email / Phone / LinkedIn / GitHub**: regex pattern matching.
- **Skills**: matched against a curated taxonomy (`skills_db.py`) covering
  programming languages, frameworks, cloud/devops, AI/ML, databases,
  networking, soft skills, and design tools. Matching is word-boundary aware
  so short tokens (e.g. "r", "c") don't false-positive inside other words.
- **Years of Experience**: extracted via an explicit "X years of experience"
  pattern if present in the resume text.

**Why rule-based instead of a big NLP model?** It's deterministic,
explainable, fast, and works fully offline — a recruiter can see exactly
*why* a field was extracted (or not). It does mean unusually-formatted
resumes (e.g. heavily templated PDFs with text in tables/columns) may extract
less cleanly; this is a known trade-off documented as a limitation below.

## 3. Job Description Analysis

The job description text is split into blocks (lines/sentences). Blocks
containing phrases like "preferred", "nice to have", "bonus" are treated as
**preferred** qualifications; everything else is treated as **required**.
Skills are then extracted from each group using the same skill taxonomy used
for resumes — this ensures candidate skills and job skills are compared on
the same vocabulary. Minimum experience is extracted from "X years"-style
phrases (the largest number mentioned is used).

## 4. Candidate Matching Engine — Scoring Formula

```
final_score = (0.5 × skill_score) + (0.3 × semantic_score) + (0.2 × experience_score)
```

These weights are configurable in `matcher.py` (`WEIGHTS` dict) — they were
chosen so that **explicit skill matches matter most** (recruiters care most
about hard skill fit), semantic similarity adds context (catches paraphrased
skills, e.g. "built APIs" ≈ "REST API development"), and experience acts as a
secondary filter.

### 4.1 Skill Score
```
skill_score = 0.8 × (matched_required / total_required)
            + 0.2 × (matched_preferred / total_preferred)
```
Required skills are weighted 4x more heavily than preferred ones.

### 4.2 Semantic Score
Cosine similarity between an embedding of the full resume text and the job
description text. The system tries to use **sentence-transformers**
(`all-MiniLM-L6-v2`) for richer semantic embeddings; if that model isn't
available (e.g. no internet to download it, or the package isn't installed),
it **automatically falls back to TF-IDF vectors** (scikit-learn) — fully
offline, no model download required. Either backend produces a 0–1 cosine
similarity score. The active backend is shown live in the dashboard header.

### 4.3 Experience Score
```
if candidate_years >= required_years: 1.0
elif candidate_years == 0: 0.0
else: candidate_years / required_years   (partial credit)
```
If the job doesn't state a minimum experience requirement, this score
defaults to 1.0 (no penalty).

### 4.4 Strengths / Weaknesses (Explainability)
For every candidate, the system generates human-readable strengths and
weaknesses directly from the three sub-scores and the matched/missing skill
lists — so a recruiter can see *why* a candidate ranked where they did
without needing to interpret raw numbers.

## 5. AI (Generative) Features

Resume summarization, hiring recommendations, interview question generation,
narrative skill-gap analysis, resume improvement suggestions, and the chat
assistant are powered by **Google Gemini** (`gemini-2.0-flash`) via
`ai_features.py`.

**Graceful degradation**: every one of these functions has a rule-based
fallback that runs automatically if no `GEMINI_API_KEY` is configured (see
`.env.example`). This means the core ranking system is fully usable for free,
with optional richer AI text on top.

## 6. Bonus Features — Design Notes

| Feature | Module | Approach |
|---|---|---|
| Interview Question Generation | `ai_features.py` | LLM prompt grounded in candidate skills + job gaps; template fallback |
| Skill Gap Analysis | `ai_features.py` + `matcher.py` | Structured missing-skills list (matcher) + narrative explanation (LLM) |
| AI Chat Assistant | `ai_features.py` | LLM with candidate-pool context injected into the prompt (RAG-lite) |
| Resume Improvement Suggestions | `ai_features.py` | LLM + rule-based checklist (missing LinkedIn/GitHub/certs/etc.) |
| Candidate Clustering | `clustering.py` | TF-IDF + KMeans, auto-labeled by top distinguishing terms per cluster |
| Vector Search | `vector_store.py` | Self-computed TF-IDF embeddings + NumPy cosine similarity, with on-disk persistence (no heavy vector-DB dependency, avoiding the deployment fragility those bring) |
| Multi-Job Candidate Matching | `multi_job.py` | Runs the core matching engine once per job, then aggregates best-fit-per-candidate and top-candidates-per-job views |

## 7. Known Limitations

- Rule-based extraction works best on resumes with clear section headers;
  heavily designed/multi-column PDFs may need manual review.
- TF-IDF semantic similarity (the offline fallback) is keyword-driven, not
  true semantic understanding — install `sentence-transformers` for richer
  results if internet access is available.
- The skill taxonomy in `skills_db.py` is curated and can be extended, but
  it won't catch every possible skill phrasing out of the box.
- AI-generated text (Gemini) requires an API key and internet access; without
  it, the system uses simpler template-based text instead of failing.
