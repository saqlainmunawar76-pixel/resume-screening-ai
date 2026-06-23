TEYZIX CORE Internship - Task AI-2
AI-Powered Resume Screening & Candidate Ranking System

Intern: Saqlain (Ref ID: TC-INT-18991230-763)

DESCRIPTION
-----------
An AI-powered system that extracts candidate information from PDF resumes,
analyzes job descriptions, and ranks candidates with an explainable matching
score (skill match + semantic similarity + experience match). Includes all
7 bonus features: interview question generation, skill gap analysis, an AI
chat assistant for recruiters, resume improvement suggestions, candidate
clustering, vector search, and multi-job candidate matching.

HOW TO RUN
----------
1. Install Python 3.10+ and run:
       pip install -r requirements.txt
2. (Optional) Copy .env.example to .env and add a free Gemini API key from
   https://aistudio.google.com/app/apikey to enable richer AI-generated text.
   The app works without this too -- it uses rule-based fallback logic.
3. Run the dashboard:
       streamlit run app.py
4. In the app: go to "Upload Resumes" tab -> click "Load Sample Resumes" to
   try it instantly with included test data, then analyze a job description
   (see data/sample_job_description.txt) and rank candidates.

See README.md for full documentation and docs/MODEL_DOCUMENTATION.md for the
scoring methodology.
