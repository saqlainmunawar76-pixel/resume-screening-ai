"""
skills_db.py
------------
A lightweight, extensible skills taxonomy used for:
  - Extracting skills mentioned in a resume
  - Extracting required/preferred skills from a job description
  - Computing skill-overlap scores in the matching engine

Why a curated list instead of a giant NLP model?
  - Fast, deterministic, explainable ("why did this candidate score X?")
  - Works fully offline (no model download required)
  - Easy for a recruiter to extend (just add a word to a list)

The list is intentionally broad (tech + soft skills) and can be extended by
editing SKILL_CATEGORIES or by passing a custom list to any extractor function.
"""

from __future__ import annotations
import re

SKILL_CATEGORIES: dict[str, list[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "go", "golang", "rust", "php", "ruby", "kotlin", "swift", "r",
        "matlab", "scala", "perl", "dart", "sql", "bash", "shell",
    ],
    "web_dev": [
        "html", "css", "react", "react.js", "angular", "vue", "vue.js",
        "next.js", "nuxt.js", "node.js", "express", "express.js", "django",
        "flask", "fastapi", "spring", "spring boot", "asp.net", "laravel",
        "bootstrap", "tailwind", "tailwind css", "jquery", "rest api",
        "graphql", "webpack", "vite",
    ],
    "data_ai_ml": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "opencv",
        "hugging face", "transformers", "langchain", "llamaindex",
        "data analysis", "data science", "data visualization", "statistics",
        "spacy", "nltk", "llm", "generative ai", "gpt", "rag",
        "vector database", "chromadb", "faiss", "pinecone", "feature engineering",
        "model deployment", "mlops", "neural network", "regression",
        "classification", "clustering",
    ],
    "databases": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "firebase",
        "supabase", "redis", "oracle", "sql server", "dynamodb", "cassandra",
        "elasticsearch",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "ci/cd", "jenkins", "terraform", "ansible", "linux", "git", "github",
        "gitlab", "nginx", "devops", "microservices", "serverless", "lambda",
    ],
    "automation_tools": [
        "n8n", "zapier", "make.com", "automation", "workflow automation",
        "rpa", "uipath",
    ],
    "mobile": [
        "android", "ios", "flutter", "react native", "swiftui", "kotlin multiplatform",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "problem-solving", "time management", "critical thinking",
        "project management", "agile", "scrum", "collaboration",
        "adaptability", "creativity", "presentation skills",
    ],
    "design": [
        "ui/ux", "ui", "ux", "figma", "adobe xd", "photoshop", "illustrator",
        "canva", "graphic design", "wireframing", "prototyping",
    ],
    "networking": [
        "networking", "ccna", "tcp/ip", "network security", "routing",
        "switching", "firewall", "vpn", "dns", "dhcp",
    ],
}

# Flat list of all known skills (lower-cased) for quick lookup
ALL_SKILLS: list[str] = sorted(
    {skill.lower() for skills in SKILL_CATEGORIES.values() for skill in skills},
    key=len,
    reverse=True,  # match longer phrases first (e.g. "machine learning" before "learning")
)


def extract_skills(text: str, custom_skills: list[str] | None = None) -> list[str]:
    """
    Find which known skills are mentioned in `text`.
    Uses word-boundary aware matching so 'r' doesn't match inside 'react'.
    Returns skills in their canonical (lower-case) form, de-duplicated.
    """
    if not text:
        return []
    text_lower = text.lower()
    candidates = custom_skills if custom_skills else ALL_SKILLS

    found = []
    for skill in candidates:
        skill_lower = skill.lower()
        # Build a safe regex: escape special chars, allow word boundaries
        # (skills with non-word chars like 'c++', 'node.js' need a relaxed boundary)
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill_lower) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill_lower)

    # de-duplicate while preserving first-seen order
    seen = set()
    unique = []
    for s in found:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def categorize_skill(skill: str) -> str:
    """Return the category name a skill belongs to, or 'other'."""
    skill_lower = skill.lower()
    for category, skills in SKILL_CATEGORIES.items():
        if skill_lower in [s.lower() for s in skills]:
            return category
    return "other"
