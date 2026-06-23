"""
Generates 5 synthetic sample resume PDFs (fictional candidates) for testing
and demoing the resume screening system. No real personal data is used.
"""
import sys, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resumes"
os.makedirs(OUT_DIR, exist_ok=True)

RESUMES = [
    {
        "file": "ahmed_raza.pdf",
        "name": "Ahmed Raza",
        "contact": "ahmed.raza.dev@gmail.com | +92-300-1234567 | linkedin.com/in/ahmedraza | github.com/ahmedraza",
        "summary": "Backend developer with 4 years of experience building scalable REST APIs and data pipelines.",
        "skills": "Python, Django, Flask, REST API, PostgreSQL, MySQL, Docker, Git, AWS, Machine Learning, scikit-learn",
        "education": ["BS Computer Science - Emerson University Multan (2019-2023)"],
        "experience": [
            "Backend Developer, TechNova Pvt Ltd (2022-Present)",
            "- Built and maintained REST APIs serving 50k+ daily requests using Django and PostgreSQL",
            "- Implemented Docker-based deployment pipeline, reducing deployment time by 40%",
            "- Worked with AWS EC2 and S3 for scalable backend infrastructure",
            "Software Engineering Intern, CodeWorks (2021-2022)",
            "- Developed Flask microservices and integrated third-party REST APIs",
        ],
        "projects": [
            "E-Commerce Platform API - Django REST Framework, PostgreSQL, Stripe integration",
            "ML-based Sentiment Analyzer - Python, scikit-learn, NLP for product reviews",
        ],
        "certifications": ["AWS Certified Cloud Practitioner", "Python for Data Science - Coursera"],
        "years": "4+ years experience",
    },
    {
        "file": "sana_malik.pdf",
        "name": "Sana Malik",
        "contact": "sana.malik.ui@gmail.com | +92-301-9876543 | linkedin.com/in/sanamalik",
        "summary": "Frontend developer focused on building accessible, responsive web interfaces.",
        "skills": "HTML, CSS, JavaScript, React, React.js, Tailwind CSS, Figma, Git, UI/UX",
        "education": ["BS Software Engineering - COMSATS University (2018-2022)"],
        "experience": [
            "Frontend Developer, PixelCraft Studio (2022-Present)",
            "- Built responsive React.js dashboards used by 10k+ active users",
            "- Collaborated with designers using Figma to implement pixel-perfect UI",
        ],
        "projects": [
            "Portfolio Builder SaaS - React, Tailwind CSS",
            "Recipe Sharing App - React Native, Firebase",
        ],
        "certifications": ["Meta Front-End Developer Certificate"],
        "years": "2 years experience",
    },
    {
        "file": "bilal_hussain.pdf",
        "name": "Bilal Hussain",
        "contact": "bilal.hussain.ai@gmail.com | +92-302-5551234 | github.com/bilalh",
        "summary": "AI/ML engineer specializing in NLP and generative AI applications.",
        "skills": "Python, Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch, Hugging Face, LangChain, "
                  "Pandas, NumPy, SQL, Vector Database, ChromaDB, Generative AI",
        "education": ["MS Artificial Intelligence - NUST (2021-2023)", "BS Computer Science - UET Lahore (2017-2021)"],
        "experience": [
            "AI Engineer, DeepLogic AI (2023-Present)",
            "- Built RAG-based chatbot using LangChain and ChromaDB for enterprise document search",
            "- Fine-tuned transformer models for sentiment classification with 92% accuracy",
            "- 3+ years experience in applied machine learning and NLP",
        ],
        "projects": [
            "Resume Parser & Matcher - Python, spaCy, Sentence Transformers",
            "AI Document Summarizer - Hugging Face Transformers, Streamlit",
        ],
        "certifications": ["Deep Learning Specialization - Coursera", "TensorFlow Developer Certificate"],
        "years": "3+ years experience",
    },
    {
        "file": "ayesha_tariq.pdf",
        "name": "Ayesha Tariq",
        "contact": "ayesha.tariq@gmail.com | +92-303-7778888",
        "summary": "Recent computer science graduate eager to start a career in software development.",
        "skills": "Python, Java, C++, HTML, CSS, JavaScript, MySQL, Git, Data Structures",
        "education": ["BS Computer Science - University of Punjab (2020-2024)"],
        "experience": [
            "Software Development Intern, ByteForge Solutions (Summer 2023)",
            "- Assisted in developing a Java-based inventory management system",
        ],
        "projects": [
            "Library Management System - Java, MySQL",
            "Personal Budget Tracker - Python, Tkinter",
        ],
        "certifications": ["No certifications listed"],
        "years": "Fresh graduate, internship experience only",
    },
    {
        "file": "usman_farooq.pdf",
        "name": "Usman Farooq",
        "contact": "usman.farooq.net@gmail.com | +92-304-4445566 | linkedin.com/in/usmanfarooq",
        "summary": "Network and systems engineer with growing expertise in automation and cloud infrastructure.",
        "skills": "Networking, CCNA, TCP/IP, Routing, Switching, Linux, Python, AWS, Docker, n8n, Automation",
        "education": ["BS Information Technology - Emerson University Multan (2019-2023)"],
        "experience": [
            "Network Support Engineer, ConnectPro ISP (2023-Present)",
            "- Managed enterprise routing/switching infrastructure for 200+ clients",
            "- Automated network monitoring workflows using Python and n8n, cutting manual checks by 60%",
            "2 years of professional experience in network administration",
        ],
        "projects": [
            "Network Monitoring Automation Tool - Python, n8n",
            "Home Lab Cloud Setup - AWS, Docker",
        ],
        "certifications": ["CCNA Certified", "AWS Cloud Practitioner (in progress)"],
        "years": "2 years experience",
    },
]


def make_resume_pdf(data: dict, out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    y = height - margin
    accent = HexColor("#1a5d3a")

    def line(text, font="Helvetica", size=10, gap=14, color=None):
        nonlocal y
        c.setFont(font, size)
        c.setFillColor(color or HexColor("#222222"))
        c.drawString(margin, y, text)
        y -= gap

    def section_header(title):
        nonlocal y
        y -= 6
        c.setFillColor(accent)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, title)
        c.setStrokeColor(accent)
        c.line(margin, y - 3, width - margin, y - 3)
        y -= 18

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(accent)
    c.drawString(margin, y, data["name"])
    y -= 20
    line(data["contact"], size=9, gap=16, color=HexColor("#444444"))

    section_header("Summary")
    line(data["summary"], gap=14)
    line(data["years"], gap=14)

    section_header("Skills")
    line(data["skills"], gap=14)

    section_header("Experience")
    for item in data["experience"]:
        line(item, gap=13)

    section_header("Projects")
    for item in data["projects"]:
        line(item, gap=13)

    section_header("Education")
    for item in data["education"]:
        line(item, gap=13)

    section_header("Certifications")
    for item in data["certifications"]:
        line(item, gap=13)

    c.save()


if __name__ == "__main__":
    for r in RESUMES:
        out_path = os.path.join(OUT_DIR, r["file"])
        make_resume_pdf(r, out_path)
        print("Created:", out_path)
