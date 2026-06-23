# 🧠 AI-Powered Resume Screening & Candidate Ranking System

**TEYZIX CORE Internship (June Batch 2026) — Task AI-2**
Domain: Artificial Intelligence | Difficulty: Intermediate

An intelligent system that automatically parses resumes, analyzes job
descriptions, and produces an explainable, ranked shortlist of candidates —
built to replace hours of manual resume screening with a few minutes of
automated, transparent scoring.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### Core Requirements
- 📤 **Resume Processing** — upload multiple PDF resumes, automatic text extraction, persistent storage (SQLite)
- 🔍 **Candidate Information Extraction** — Name, Email, Phone, LinkedIn, GitHub, Skills, Education, Certifications, Experience, Projects
- 🎯 **Job Description Analysis** — extracts required vs. preferred skills, experience requirements, qualifications
- 🏆 **Candidate Matching Engine** — explainable weighted scoring (skill match + semantic similarity + experience)
- 📊 **Dashboard** — upload, rank, view candidate profiles, compare candidates, export to CSV

### 🌟 Bonus Features (all 7 implemented)
| # | Feature | Where |
|---|---|---|
| 1 | Interview Question Generation | Rankings tab → "Generate AI Insights" |
| 2 | Candidate Skill Gap Analysis | Rankings tab → "Generate AI Insights" |
| 3 | AI Chat Assistant for Recruiters | "AI Chat Assistant" tab |
| 4 | Resume Improvement Suggestions | Rankings tab → "Generate AI Insights" |
| 5 | Candidate Clustering | "Clustering" tab |
| 6 | Vector Search Implementation | "Vector Search" tab |
| 7 | Multi-Job Candidate Matching | "Multi-Job Matching" tab |

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Dashboard**: Streamlit
- **PDF Parsing**: pdfplumber
- **Matching/ML**: scikit-learn (TF-IDF, cosine similarity, KMeans clustering)
- **Vector Search**: ChromaDB
- **Generative AI**: Google Gemini API (`gemini-2.0-flash`) — optional, with full offline fallback
- **Storage**: SQLite

> The system works fully **without** any API key. AI-generated text
> (summaries, interview questions, chat) is enhanced when a free Gemini API
> key is added, but every feature has a rule-based fallback.

---

## 📁 Project Structure

```
.
├── app.py                          # Streamlit dashboard (entry point)
├── requirements.txt
├── .env.example                    # Copy to .env and add your Gemini API key (optional)
├── src/
│   ├── extractor.py                # PDF parsing + candidate info extraction
│   ├── skills_db.py                 # Curated skill taxonomy
│   ├── job_parser.py                # Job description analysis
│   ├── matcher.py                   # Core matching engine + explainable scoring
│   ├── ai_features.py               # Gemini-powered features (+ offline fallback)
│   ├── vector_store.py              # Vector search (ChromaDB, bonus feature)
│   ├── clustering.py                # Candidate clustering (bonus feature)
│   ├── multi_job.py                 # Multi-job matching (bonus feature)
│   └── storage.py                   # SQLite persistence
├── data/
│   ├── sample_resumes/              # 5 synthetic sample resumes (PDF) for testing
│   ├── sample_job_description.txt   # Sample JD #1 (Backend Developer)
│   └── sample_job_description_2.txt # Sample JD #2 (AI/ML Engineer)
├── docs/
│   └── MODEL_DOCUMENTATION.md       # Scoring methodology & architecture explained
└── screenshots/                     # Add dashboard screenshots here before submission
```

---

## 🚀 Getting Started

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. (Optional) Enable AI-generated text
```bash
cp .env.example .env
# Edit .env and paste your free Gemini API key from https://aistudio.google.com/app/apikey
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. Try it out
1. Go to **📤 Upload Resumes** tab → click **"📁 Load Sample Resumes"** (or upload your own PDFs)
2. Go to **🎯 Job Description** tab → paste a JD (try `data/sample_job_description.txt`) → **Analyze**
3. Go to **🏆 Rankings** tab → **Rank Candidates Against This Job**
4. Expand any candidate → **"✨ Generate AI Insights"** for summary, interview questions, skill gaps, and resume suggestions
5. Explore **Clustering**, **Vector Search**, **Multi-Job Matching**, and the **AI Chat Assistant** tabs

---

## 📐 Scoring Methodology (Explainability)

```
final_score = 0.5 × skill_score + 0.3 × semantic_score + 0.2 × experience_score
```

Every candidate gets a full breakdown: matched/missing skills,
strengths/weaknesses, and sub-scores — never just a single opaque number.
Full details: [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md).

---

## 🧪 Testing Data

`data/sample_resumes/` contains 5 fictional, synthetically generated resumes
covering different profiles (backend dev, frontend dev, AI/ML engineer,
fresh graduate, network engineer) so the matching engine and bonus features
can be demoed immediately without needing real candidate data.

---

## 👤 Author

**Saqlain** — IT Student, Emerson University Multan
Built as part of the TEYZIX CORE Internship Program (June Batch 2026)
Ref ID: TC-INT-18991230-763

## 📄 License

MIT — feel free to use this project as a learning reference.
