"""
app.py
------
AI-Powered Resume Screening & Candidate Ranking System
TEYZIX CORE Internship - Task AI-2

Main Streamlit dashboard tying together every module in src/:
  extractor.py    -> resume parsing
  job_parser.py   -> job description analysis
  matcher.py      -> candidate matching engine (core AI)
  ai_features.py  -> Gemini-powered generative features (+ offline fallback)
  vector_store.py -> vector search (bonus)
  clustering.py   -> candidate clustering (bonus)
  multi_job.py    -> multi-job matching (bonus)
  storage.py      -> SQLite persistence

Run with:  streamlit run app.py
"""

import os
import sys
import io
import json
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from extractor import extract_candidate_info, Candidate
from job_parser import parse_job_description
from matcher import rank_candidates, get_embedder_kind
import ai_features
from vector_store import CandidateVectorStore
from clustering import cluster_candidates
from multi_job import match_candidates_to_multiple_jobs
import storage

# Load .env if present (for GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(
    page_title="AI Resume Screening System | TEYZIX CORE",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling (TEYZIX brand green, polished card/tab/badge styling)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #134a2e 0%, #1a5d3a 55%, #2c7a4f 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 18px;
        color: white;
        box-shadow: 0 6px 20px rgba(26, 93, 58, 0.25);
    }
    .hero-title {
        font-size: 1.85rem; font-weight: 800; margin: 0; color: white;
        letter-spacing: -0.3px;
    }
    .hero-sub {
        font-size: 0.95rem; margin-top: 6px; color: #d9efe2; font-weight: 400;
    }
    .hero-pill {
        display: inline-block; background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.3); border-radius: 999px;
        padding: 4px 13px; font-size: 0.78rem; margin-top: 14px; margin-right: 8px;
        color: white;
    }

    .score-badge {
        display: inline-block; padding: 5px 16px; border-radius: 999px;
        font-weight: 700; color: white; font-size: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 8px 14px; font-weight: 600;
    }
    div[data-testid="stMetricValue"] { color: #1a5d3a; }

    div[data-testid="stSidebarContent"] { padding-top: 1.2rem; }
    .sidebar-card {
        background: #f4f8f6; border: 1px solid #e0eee6; border-radius: 12px;
        padding: 14px 16px; margin-bottom: 12px;
    }
    .sidebar-card h4 { margin: 0 0 4px 0; font-size: 0.85rem; color: #1a5d3a; }
    .sidebar-card p { margin: 0; font-size: 1.4rem; font-weight: 700; color: #222; }
    .sidebar-section-title {
        font-size: 0.78rem; font-weight: 700; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.5px; margin: 18px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 70:
        return "#1a5d3a"
    if score >= 45:
        return "#c98a1c"
    return "#b03a2e"


def score_badge(score: float) -> str:
    return f'<span class="score-badge" style="background:{score_color(score)}">{score:.1f}%</span>'


def safe_index_candidates(candidates: list) -> None:
    """Reindex the vector store, but never let a vector-search hiccup crash
    the whole dashboard -- it's a bonus feature, not core functionality."""
    try:
        st.session_state.vector_store.index_candidates(candidates)
    except Exception as e:
        st.toast(f"⚠️ Vector search index couldn't be rebuilt ({type(e).__name__}). Other features are unaffected.", icon="⚠️")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "candidates" not in st.session_state:
    st.session_state.candidates = storage.load_all_candidates()
if "job" not in st.session_state:
    st.session_state.job = None
if "jobs_multi" not in st.session_state:
    st.session_state.jobs_multi = []
if "results" not in st.session_state:
    st.session_state.results = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = CandidateVectorStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar — at-a-glance dashboard
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Resume Screening AI")
    st.caption("TEYZIX CORE Internship — Task AI-2")
    st.divider()

    st.markdown(f"""
    <div class="sidebar-card"><h4>📁 Candidates Loaded</h4><p>{len(st.session_state.candidates)}</p></div>
    """, unsafe_allow_html=True)

    active_job = st.session_state.job.title if st.session_state.job else "None analyzed yet"
    st.markdown(f"""
    <div class="sidebar-card"><h4>🎯 Active Job</h4><p style="font-size:1rem;">{active_job}</p></div>
    """, unsafe_allow_html=True)

    ranked_count = len(st.session_state.results)
    st.markdown(f"""
    <div class="sidebar-card"><h4>🏆 Candidates Ranked</h4><p>{ranked_count}</p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)
    if ai_features.is_ai_enabled():
        st.success("Gemini AI: Active", icon="🟢")
    else:
        st.warning("Gemini AI: Offline mode", icon="🟡")
    st.caption(f"Semantic engine: **{get_embedder_kind()}**")

    st.markdown('<div class="sidebar-section-title">Quick Links</div>', unsafe_allow_html=True)
    st.caption("📤 Upload Resumes → 🎯 Job Description → 🏆 Rankings")
    st.caption("Explore Clustering, Vector Search, Multi-Job Matching, and the AI Chat Assistant for deeper insights.")

# ---------------------------------------------------------------------------
# Header (hero banner)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <p class="hero-title">🧠 AI-Powered Resume Screening &amp; Candidate Ranking System</p>
    <p class="hero-sub">Parse resumes, analyze job descriptions, and get an explainable, ranked shortlist — in minutes, not hours.</p>
    <span class="hero-pill">⚡ Explainable Scoring</span>
    <span class="hero-pill">🔎 Semantic Matching</span>
    <span class="hero-pill">🤖 AI-Powered Insights</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "📤 Upload Resumes",
    "🎯 Job Description",
    "🏆 Rankings",
    "🔍 Candidate Profile",
    "⚖️ Compare",
    "🧩 Clustering",
    "🔎 Vector Search",
    "📊 Multi-Job Matching",
    "💬 AI Chat Assistant",
])

# ---------------------------------------------------------------------------
# TAB 1: Upload Resumes
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Upload Resumes (PDF)")
    st.write("Upload one or more candidate resumes. Each resume is parsed and stored automatically.")

    uploaded_files = st.file_uploader("Choose PDF resume(s)", type=["pdf"], accept_multiple_files=True)

    col_a, col_b = st.columns(2)
    with col_a:
        process_clicked = st.button("⚙️ Process Uploaded Resumes", type="primary", width='stretch')
    with col_b:
        load_samples_clicked = st.button("📁 Load Sample Resumes (for testing)", width='stretch')

    if process_clicked and uploaded_files:
        progress = st.progress(0, text="Processing resumes...")
        new_candidates = []
        for i, file in enumerate(uploaded_files):
            tmp_path = os.path.join("storage_data", "_tmp_upload.pdf")
            os.makedirs("storage_data", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(file.getbuffer())
            candidate = extract_candidate_info(tmp_path, file_name=file.name)
            storage.save_candidate(candidate)
            new_candidates.append(candidate)
            progress.progress((i + 1) / len(uploaded_files), text=f"Processed {file.name}")
        st.session_state.candidates = storage.load_all_candidates()
        safe_index_candidates(st.session_state.candidates)
        st.success(f"✅ Processed and stored {len(new_candidates)} resume(s).")

    if load_samples_clicked:
        sample_dir = "data/sample_resumes"
        if os.path.isdir(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.endswith(".pdf")]
            for f in files:
                c = extract_candidate_info(os.path.join(sample_dir, f), file_name=f)
                storage.save_candidate(c)
            st.session_state.candidates = storage.load_all_candidates()
            safe_index_candidates(st.session_state.candidates)
            st.success(f"✅ Loaded {len(files)} sample resumes.")
        else:
            st.error("Sample resumes folder not found.")

    st.divider()
    st.subheader(f"Stored Candidates ({len(st.session_state.candidates)})")
    if st.session_state.candidates:
        df = pd.DataFrame([{
            "Name": c.name, "Email": c.email, "Phone": c.phone,
            "Skills (count)": len(c.skills), "Years Exp": c.years_of_experience,
            "File": c.file_name,
        } for c in st.session_state.candidates])
        st.dataframe(df, width='stretch', hide_index=True)

        if st.button("🗑️ Clear All Candidates"):
            storage.clear_all()
            st.session_state.candidates = []
            st.session_state.vector_store.reset()
            st.rerun()
    else:
        st.info("No candidates yet. Upload resumes above or load sample data to get started.")

# ---------------------------------------------------------------------------
# TAB 2: Job Description
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Job Description Analysis")
    job_title = st.text_input("Job Title", placeholder="e.g. Python Backend Developer")
    jd_text = st.text_area(
        "Paste the Job Description",
        height=220,
        placeholder="Paste required skills, preferred qualifications, experience requirements, etc.",
    )

    if st.button("🔍 Analyze Job Description", type="primary"):
        if not jd_text.strip():
            st.warning("Please paste a job description first.")
        else:
            job = parse_job_description(jd_text, title=job_title)
            st.session_state.job = job
            st.success("Job description analyzed.")

    if st.session_state.job:
        job = st.session_state.job
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Required Skills", len(job.required_skills))
        c2.metric("Preferred Skills", len(job.preferred_skills))
        c3.metric("Min. Experience", f"{job.min_experience_years:.0f} yrs" if job.min_experience_years else "Not specified")

        st.write("**Required Skills:**", ", ".join(job.required_skills) or "None detected")
        st.write("**Preferred Skills:**", ", ".join(job.preferred_skills) or "None detected")
        st.write("**Qualifications:**")
        for q in job.qualifications:
            st.write(f"- {q}")

        st.caption("💡 Add this job to the multi-job list (used in the 'Multi-Job Matching' tab):")
        if st.button("➕ Add to Multi-Job List"):
            st.session_state.jobs_multi.append(job)
            st.success(f"Added '{job.title}' to multi-job list ({len(st.session_state.jobs_multi)} jobs total).")

# ---------------------------------------------------------------------------
# TAB 3: Rankings
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Candidate Rankings")

    if not st.session_state.candidates:
        st.info("Upload resumes first (see 'Upload Resumes' tab).")
    elif not st.session_state.job:
        st.info("Analyze a job description first (see 'Job Description' tab).")
    else:
        if st.button("🏆 Rank Candidates Against This Job", type="primary"):
            st.session_state.results = rank_candidates(st.session_state.candidates, st.session_state.job)

        if st.session_state.results:
            st.caption(f"Ranked against: **{st.session_state.job.title}**")
            cand_lookup = {c.file_name: c for c in st.session_state.candidates}

            for rank, r in enumerate(st.session_state.results, start=1):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
                    with col1:
                        st.markdown(f"**#{rank} — {r.candidate_name}**")
                        st.caption(r.file_name)
                    with col2:
                        st.markdown(score_badge(r.final_score), unsafe_allow_html=True)
                        st.caption("Final Match Score")
                    with col3:
                        st.write(f"Skill: {r.skill_score}%")
                        st.write(f"Semantic: {r.semantic_score}%")
                        st.write(f"Experience: {r.experience_score}%")

                    with st.expander("📋 Full Breakdown & AI Insights"):
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("**✅ Strengths**")
                            for s in r.strengths:
                                st.write(f"- {s}")
                            st.markdown("**⚠️ Weaknesses**")
                            for w in r.weaknesses:
                                st.write(f"- {w}")
                        with cc2:
                            st.markdown("**Matched Skills**")
                            st.write(", ".join(r.matched_skills) or "None")
                            st.markdown("**Missing Skills**")
                            st.write(", ".join(r.missing_skills) or "None")

                        candidate_obj = cand_lookup.get(r.file_name)
                        if candidate_obj and st.button(f"✨ Generate AI Insights for {r.candidate_name}", key=f"ai_{r.file_name}"):
                            with st.spinner("Generating..."):
                                summary = ai_features.summarize_resume(candidate_obj)
                                recommendation = ai_features.generate_recommendation(r, candidate_obj)
                                gap = ai_features.skill_gap_analysis(candidate_obj, st.session_state.job)
                                questions = ai_features.generate_interview_questions(candidate_obj, st.session_state.job)
                                suggestions = ai_features.resume_improvement_suggestions(candidate_obj)

                            st.markdown("**📝 AI Summary**")
                            st.write(summary)
                            st.markdown("**💡 Hiring Recommendation**")
                            st.write(recommendation)
                            st.markdown("**📊 Skill Gap Analysis**")
                            st.write(gap)
                            st.markdown("**🎤 Suggested Interview Questions**")
                            for q in questions:
                                st.write(f"- {q}")
                            st.markdown("**🛠️ Resume Improvement Suggestions**")
                            for s in suggestions:
                                st.write(f"- {s}")

            st.divider()
            export_df = pd.DataFrame([r.to_dict() for r in st.session_state.results])
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Rankings as CSV", data=csv_bytes,
                file_name="candidate_rankings.csv", mime="text/csv",
            )

# ---------------------------------------------------------------------------
# TAB 4: Candidate Profile (extracted info viewer)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("View Extracted Candidate Information")
    if not st.session_state.candidates:
        st.info("No candidates stored yet.")
    else:
        names = [f"{c.name} ({c.file_name})" for c in st.session_state.candidates]
        choice = st.selectbox("Select a candidate", names)
        idx = names.index(choice)
        c = st.session_state.candidates[idx]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {c.name}")
            st.write(f"📧 {c.email or 'Not found'}")
            st.write(f"📱 {c.phone or 'Not found'}")
            st.write(f"🔗 LinkedIn: {c.linkedin or 'Not found'}")
            st.write(f"🐙 GitHub: {c.github or 'Not found'}")
            st.write(f"📅 Years of experience: {c.years_of_experience}")
        with col2:
            st.markdown("**Skills detected:**")
            st.write(", ".join(c.skills) or "None detected")

        st.markdown("**🎓 Education**")
        for e in c.education:
            st.write(f"- {e}")
        st.markdown("**💼 Experience**")
        for e in c.experience:
            st.write(f"- {e}")
        st.markdown("**🚀 Projects**")
        for p in c.projects:
            st.write(f"- {p}")
        st.markdown("**📜 Certifications**")
        for cert in c.certifications:
            st.write(f"- {cert}")

        with st.expander("📄 View Raw Extracted Text"):
            st.text(c.raw_text)

# ---------------------------------------------------------------------------
# TAB 5: Compare Candidates
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Compare Candidates Side-by-Side")
    if len(st.session_state.candidates) < 2:
        st.info("Need at least 2 candidates to compare.")
    else:
        names = [f"{c.name} ({c.file_name})" for c in st.session_state.candidates]
        selected = st.multiselect("Select candidates to compare", names, default=names[:2], max_selections=4)
        if selected:
            chosen = [st.session_state.candidates[names.index(s)] for s in selected]
            compare_df = pd.DataFrame([{
                "Name": c.name,
                "Years Exp": c.years_of_experience,
                "Skills Count": len(c.skills),
                "Top Skills": ", ".join(c.skills[:6]),
                "Certifications": len(c.certifications),
                "Projects": len(c.projects),
            } for c in chosen])
            st.dataframe(compare_df, width='stretch', hide_index=True)

            if st.session_state.results:
                score_lookup = {r.file_name: r.final_score for r in st.session_state.results}
                score_df = pd.DataFrame([{
                    "Name": c.name,
                    "Match Score": score_lookup.get(c.file_name, "Not ranked yet"),
                } for c in chosen])
                st.bar_chart(score_df.set_index("Name")) if all(isinstance(v, (int, float)) for v in score_lookup.values()) else st.dataframe(score_df)

# ---------------------------------------------------------------------------
# TAB 6: Clustering (Bonus)
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("🧩 Candidate Clustering")
    st.caption("Groups candidates by skill-profile similarity — useful for exploring a talent pool without a specific job description.")
    if len(st.session_state.candidates) < 2:
        st.info("Need at least 2 candidates to cluster.")
    else:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=min(6, len(st.session_state.candidates)), value=min(3, len(st.session_state.candidates)))
        if st.button("🧩 Run Clustering"):
            clusters = cluster_candidates(st.session_state.candidates, n_clusters=n_clusters)
            for cl in clusters:
                with st.container(border=True):
                    st.markdown(f"**Cluster: {cl.label}**")
                    st.caption(f"Top distinguishing terms: {', '.join(cl.top_terms)}")
                    st.write("Members:", ", ".join(cl.candidate_names) or "None")

# ---------------------------------------------------------------------------
# TAB 7: Vector Search (Bonus)
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("🔎 Semantic Vector Search")
    st.caption("Search candidates by meaning, not just exact keywords. Example: 'someone experienced with cloud deployment'.")
    if not st.session_state.candidates:
        st.info("No candidates indexed yet. Upload resumes first.")
    else:
        if st.button("🔁 (Re)build Vector Index"):
            safe_index_candidates(st.session_state.candidates)
            st.success("Vector index rebuilt.")

        query = st.text_input("Search query", placeholder="e.g. candidate with cloud and automation experience")
        if st.button("Search", type="primary") and query:
            results = st.session_state.vector_store.semantic_search(query, top_k=5)
            if not results:
                st.warning("Index is empty — click '(Re)build Vector Index' first.")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['name']}** — similarity: {r['similarity']}")
                    st.caption(f"Skills: {r['skills']}")

# ---------------------------------------------------------------------------
# TAB 8: Multi-Job Matching (Bonus)
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("📊 Multi-Job Candidate Matching")
    st.caption("Match one candidate pool against multiple job openings at once. Add jobs from the 'Job Description' tab.")

    if st.session_state.jobs_multi:
        st.write("**Jobs in list:**", ", ".join(j.title for j in st.session_state.jobs_multi))
        if st.button("🗑️ Clear Job List"):
            st.session_state.jobs_multi = []
            st.rerun()

    if not st.session_state.candidates or len(st.session_state.jobs_multi) < 1:
        st.info("Add candidates and at least one job (via the Job Description tab → 'Add to Multi-Job List') to use this feature.")
    else:
        if st.button("📊 Run Multi-Job Matching", type="primary"):
            result = match_candidates_to_multiple_jobs(st.session_state.candidates, st.session_state.jobs_multi)

            st.markdown("### Best-Fit Job per Candidate")
            best_fit_df = pd.DataFrame([{
                "Candidate": cb.candidate_name,
                "Best-Fit Job": cb.best_job_title,
                "Best Score": cb.best_score,
                **{f"Score: {job}": score for job, score in cb.all_scores.items()},
            } for cb in result["by_candidate"]])
            st.dataframe(best_fit_df, width='stretch', hide_index=True)

            st.markdown("### Top Candidates per Job")
            for job_title, results in result["by_job"].items():
                st.markdown(f"**{job_title}**")
                top_df = pd.DataFrame([{"Candidate": r.candidate_name, "Score": r.final_score} for r in results[:5]])
                st.dataframe(top_df, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# TAB 9: AI Chat Assistant (Bonus)
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("💬 AI Chat Assistant for Recruiters")
    st.caption("Ask free-form questions about the current candidate pool, e.g. 'Who has the most AWS experience?' or 'Compare Ahmed and Bilal.'")

    if not st.session_state.candidates:
        st.info("Upload resumes first to chat about candidates.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask about your candidates...")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            # Build compact context from candidates + last ranking (if available)
            context_lines = []
            score_lookup = {r.file_name: r.final_score for r in st.session_state.results} if st.session_state.results else {}
            for c in st.session_state.candidates:
                score_str = f", match score: {score_lookup[c.file_name]}%" if c.file_name in score_lookup else ""
                context_lines.append(
                    f"- {c.name}: {c.years_of_experience} yrs exp, skills: {', '.join(c.skills)}{score_str}"
                )
            context = "\n".join(context_lines)

            with st.spinner("Thinking..."):
                answer = ai_features.chat_with_assistant(user_q, context)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

st.divider()
st.caption("Built with Python, Streamlit, scikit-learn, ChromaDB & Google Gemini · TEYZIX CORE Internship Task AI-2 · by Saqlain")
