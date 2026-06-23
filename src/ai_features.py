"""
ai_features.py
--------------
Generative AI Features (uses Google Gemini API):
  - Resume summarization
  - Keyword extraction (LLM-assisted, on top of the rule-based one in skills_db)
  - Recommendation generation (should we shortlist this candidate? why?)

Bonus Features powered by the same LLM:
  - Interview Question Generation
  - Candidate Skill Gap Analysis (narrative version, on top of matcher.py's
    structured missing_skills list)
  - Resume Improvement Suggestions
  - AI Chat Assistant for Recruiters (ask free-form questions about a candidate
    pool / a specific candidate)

Design philosophy:
  - The system must NOT crash or become unusable if no API key is configured.
    Every function below has a clearly-labelled fallback that uses the
    structured data we already extracted (skills_db + matcher.py) to produce
    a reasonable, rule-based answer. This means recruiters can use the core
    ranking system for free, and get richer AI-written text once they add a
    Gemini API key.
  - Model: gemini-2.5-flash (fast + cost-efficient, good enough for resume-length
    text). Google periodically retires older Gemini models (e.g. gemini-2.0-flash
    was shut down June 2026) -- if this model name is ever deprecated, check
    https://ai.google.dev/gemini-api/docs/models for the current recommended
    replacement and update MODEL_NAME below.
"""

from __future__ import annotations
import os
import json

MODEL_NAME = "gemini-2.5-flash"
# Tried in order if the primary model above fails (e.g. retired by Google).
# Keeps the app working through model deprecations without needing a manual fix.
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]

_client = None
_client_checked = False
_working_model = None


def _get_client():
    """
    Lazily create the Gemini client (new unified `google-genai` SDK -- the
    older `google-generativeai` package is deprecated). Returns None if no
    API key is set or the SDK isn't installed -- callers must handle the
    None case by using their rule-based fallback.
    """
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
    except Exception:
        _client = None
    return _client


def is_ai_enabled() -> bool:
    return _get_client() is not None


def _generate(prompt: str, max_tokens: int = 500) -> str | None:
    client = _get_client()
    if client is None:
        return None

    try:
        from google.genai import types
    except Exception:
        return None

    global _working_model
    models_to_try = [_working_model] if _working_model else []
    models_to_try += [m for m in FALLBACK_MODELS if m != _working_model]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.4,
                    # Gemini 2.5+ models think by default, and those thinking
                    # tokens are deducted from max_output_tokens -- causing
                    # responses to truncate mid-sentence. We don't need
                    # chain-of-thought for these short, factual prompts, so
                    # we disable it to keep the full budget for visible text.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (response.text or "").strip()
            if text:
                _working_model = model_name  # remember what worked
                return text
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# 1. Resume Summarization
# ---------------------------------------------------------------------------
def summarize_resume(candidate) -> str:
    prompt = f"""Summarize this candidate's resume in 3 concise sentences for a recruiter.
Focus on their core expertise, years of experience, and standout qualifications.

Name: {candidate.name}
Skills: {', '.join(candidate.skills)}
Experience: {' '.join(candidate.experience[:6])}
Projects: {' '.join(candidate.projects[:4])}
Education: {' '.join(candidate.education)}
"""
    result = _generate(prompt, max_tokens=350)
    if result:
        return result

    # --- Fallback: rule-based summary from structured fields ---
    skills_str = ", ".join(candidate.skills[:6]) if candidate.skills else "no listed skills"
    exp = f"{candidate.years_of_experience:.0f}+ years of experience" if candidate.years_of_experience else "early-career profile"
    return (
        f"{candidate.name} is a candidate with {exp}, skilled in {skills_str}. "
        f"Has {len(candidate.projects)} project(s) and {len(candidate.certifications)} certification(s) on record."
    )


# ---------------------------------------------------------------------------
# 2. Recommendation generation
# ---------------------------------------------------------------------------
def generate_recommendation(match_result, candidate) -> str:
    prompt = f"""You are a recruiter assistant. Based on this candidate match data, write a
2-sentence hiring recommendation (shortlist / consider / not a fit) with a brief reason.

Candidate: {match_result.candidate_name}
Final Match Score: {match_result.final_score}%
Skill Score: {match_result.skill_score}%
Matched Skills: {', '.join(match_result.matched_skills)}
Missing Skills: {', '.join(match_result.missing_skills)}
Strengths: {'; '.join(match_result.strengths)}
Weaknesses: {'; '.join(match_result.weaknesses)}
"""
    result = _generate(prompt, max_tokens=250)
    if result:
        return result

    # --- Fallback ---
    score = match_result.final_score
    if score >= 75:
        verdict = "Strongly recommend shortlisting"
    elif score >= 50:
        verdict = "Consider shortlisting with a screening call"
    else:
        verdict = "Not a strong fit for this role"
    return f"{verdict} (score: {score}%). {match_result.strengths[0]}"


# ---------------------------------------------------------------------------
# 3. Bonus: Interview Question Generation
# ---------------------------------------------------------------------------
def generate_interview_questions(candidate, job, n: int = 5) -> list[str]:
    prompt = f"""Generate {n} targeted interview questions for this candidate applying to
the role "{job.title}". Mix technical questions (based on their listed skills/projects)
with 1-2 questions probing any gaps versus the job requirements.

Candidate skills: {', '.join(candidate.skills)}
Candidate projects: {' '.join(candidate.projects[:3])}
Job required skills: {', '.join(job.required_skills)}

Return ONLY a numbered list of {n} questions, nothing else.
"""
    result = _generate(prompt, max_tokens=550)
    if result:
        questions = [line.strip(" -.\t") for line in result.split("\n") if line.strip()]
        # strip leading numbering like "1." if present
        cleaned = []
        for q in questions:
            parts = q.split(".", 1)
            cleaned.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else q)
        return cleaned[:n]

    # --- Fallback: template questions from matched/missing skills ---
    matched = candidate.skills[:3] if candidate.skills else ["your core skill set"]
    missing = [s for s in job.required_skills if s not in candidate.skills][:2]
    qs = [f"Can you walk us through a project where you used {s}?" for s in matched]
    qs += [f"You haven't listed {s} on your resume — do you have any exposure to it?" for s in missing]
    qs.append("Describe a challenging technical problem you solved recently and how you approached it.")
    return qs[:n]


# ---------------------------------------------------------------------------
# 4. Bonus: Candidate Skill Gap Analysis (narrative)
# ---------------------------------------------------------------------------
def skill_gap_analysis(candidate, job) -> str:
    missing = [s for s in job.required_skills if s not in candidate.skills]
    prompt = f"""Write a short, constructive skill-gap analysis (3-4 sentences) for this
candidate relative to the job "{job.title}". Be specific about which skills are missing
and how critical each one is.

Candidate skills: {', '.join(candidate.skills)}
Required skills for job: {', '.join(job.required_skills)}
Missing skills: {', '.join(missing)}
"""
    result = _generate(prompt, max_tokens=350)
    if result:
        return result

    if not missing:
        return f"{candidate.name} covers all the required skills listed for this role -- no significant gaps identified."
    return (
        f"{candidate.name} is missing {len(missing)} required skill(s) for this role: "
        f"{', '.join(missing)}. Depending on how core these are to the day-to-day work, "
        f"this could be addressed through onboarding/training or a targeted interview screen."
    )


# ---------------------------------------------------------------------------
# 5. Bonus: Resume Improvement Suggestions
# ---------------------------------------------------------------------------
def resume_improvement_suggestions(candidate) -> list[str]:
    prompt = f"""Give 3 specific, actionable suggestions to improve this resume for better
ATS screening and recruiter impact. Be concrete (not generic advice).

Name: {candidate.name}
Skills listed: {', '.join(candidate.skills)}
Has certifications: {bool(candidate.certifications)}
Has projects: {bool(candidate.projects)}
Has LinkedIn: {bool(candidate.linkedin)}
Has GitHub: {bool(candidate.github)}
Years of experience stated explicitly: {candidate.years_of_experience}

Return ONLY a numbered list of 3 suggestions.
"""
    result = _generate(prompt, max_tokens=400)
    if result:
        lines = [l.strip(" -.\t") for l in result.split("\n") if l.strip()]
        return lines[:3]

    # --- Fallback: rule-based checklist ---
    suggestions = []
    if not candidate.linkedin:
        suggestions.append("Add a LinkedIn profile link -- recruiters often cross-check this.")
    if not candidate.github and any(s in candidate.skills for s in ["python", "javascript", "java", "react"]):
        suggestions.append("Add a GitHub link to showcase your code, especially given your technical skill set.")
    if candidate.years_of_experience == 0:
        suggestions.append("State your years of experience explicitly (e.g. 'X years of experience in...') so ATS systems can parse it.")
    if not candidate.certifications:
        suggestions.append("Consider adding relevant certifications to strengthen your profile.")
    if len(candidate.skills) < 5:
        suggestions.append("List more specific technical skills/tools you've used -- this resume has very few detected.")
    if not suggestions:
        suggestions.append("Resume structure looks solid -- consider quantifying achievements with metrics (e.g. 'reduced load time by 30%').")
    return suggestions[:3]


# ---------------------------------------------------------------------------
# 6. Bonus: AI Chat Assistant for Recruiters
# ---------------------------------------------------------------------------
def chat_with_assistant(question: str, candidates_context: str, history: list[dict] | None = None) -> str:
    """
    `candidates_context` is a compact text block (built by app.py) summarizing
    the current candidate pool + rankings, so the assistant can answer
    questions like "who has the most AWS experience?" or "compare candidate A and B".
    """
    prompt = f"""You are a recruiting assistant helping a recruiter evaluate candidates.
Use ONLY the context below to answer. If the answer isn't in the context, say so honestly.

CONTEXT:
{candidates_context}

QUESTION: {question}

Answer concisely and reference candidate names directly.
"""
    result = _generate(prompt, max_tokens=700)
    if result:
        return result

    return (
        "AI chat assistant requires a GEMINI_API_KEY to be set (see README/.env.example). "
        "Without it, please use the Rankings and Compare tabs to review candidates directly."
    )
