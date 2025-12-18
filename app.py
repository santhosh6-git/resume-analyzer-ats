from flask import Flask, render_template, request
import os
import re
import pdfplumber
from datetime import datetime
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads/resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- SKILL MAP WITH WEIGHTS (ATS STYLE) ----------------
SKILL_MAP = {
    "python": {"aliases": ["python"], "weight": 3},
    "java": {"aliases": ["java"], "weight": 3},
    "sql": {"aliases": ["sql", "mysql", "postgresql", "sqlite"], "weight": 2},
    "html": {"aliases": ["html"], "weight": 1},
    "css": {"aliases": ["css"], "weight": 1},
    "javascript": {"aliases": ["javascript", "js"], "weight": 2},
    "node.js": {"aliases": ["node", "node.js", "nodejs"], "weight": 2},
    "react": {"aliases": ["react", "reactjs"], "weight": 2},
    "flask": {"aliases": ["flask"], "weight": 2},
    "django": {"aliases": ["django"], "weight": 2},
    "machine learning": {"aliases": ["machine learning", "ml"], "weight": 3},
    "data analysis": {"aliases": ["data analysis", "data analytics"], "weight": 2},
    "git": {"aliases": ["git", "github"], "weight": 1},
    "api": {"aliases": ["api", "rest api", "restful"], "weight": 1}
}

history = []  # in-memory (resets on restart)

# ---------------- HELPERS ----------------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()

def extract_skills(text):
    """
    Returns dict: {skill_name: weight}
    """
    found = {}
    for skill, data in SKILL_MAP.items():
        for alias in data["aliases"]:
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                found[skill] = data["weight"]
                break
    return found

def nlp_similarity(resume, job):
    if not resume.strip() or not job.strip():
        return 0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume, job])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return int(score * 100)

def skill_score(resume_skills, job_skills):
    """
    Weighted ATS-style skill score
    """
    if not job_skills:
        return 0

    total_weight = sum(job_skills.values())
    matched_weight = 0

    for skill, weight in job_skills.items():
        if skill in resume_skills:
            matched_weight += weight

    return int((matched_weight / total_weight) * 100)

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def dashboard():
    result = None

    if request.method == "POST":
        job_desc = request.form.get("job", "").strip()
        pdf = request.files.get("resume")

        if pdf and pdf.filename.endswith(".pdf"):
            filename = secure_filename(pdf.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
            pdf.save(filepath)

            resume_text = extract_text_from_pdf(filepath)
            job_text = job_desc.lower()

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            nlp = nlp_similarity(resume_text, job_text)
            skills = skill_score(resume_skills, job_skills)

            final_score = int((0.6 * nlp) + (0.4 * skills))

            # -------- PORTFOLIO-QUALITY UX LOGIC --------
            if not job_skills:
                missing = []
                matched = []
                message = "Paste a more detailed job description with required technical skills to get accurate results"
            else:
                matched = list(set(job_skills.keys()) & set(resume_skills.keys()))
                missing = list(set(job_skills.keys()) - set(resume_skills.keys()))
                message = None

            result = {
                "score": final_score,
                "matched": matched,
                "missing": missing,
                "message": message,
                "time": datetime.now().strftime("%d %b %Y %H:%M")
            }

            history.insert(0, result)

    return render_template("dashboard.html", result=result, history=history)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
