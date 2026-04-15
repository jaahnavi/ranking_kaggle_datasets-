"""
resume_gap_analyzer.py
──────────────────────────────────────────────────────────────────
Compares a resume against a job description and outputs a JSON with:
  - missing_skills  : skills in the JD not found in the resume
  - gap_summary     : a concise, human-readable narrative of the gaps

Usage:
    python resume_gap_analyzer.py                    # interactive prompts
    python resume_gap_analyzer.py --resume r.pdf --jd jd.txt
    python resume_gap_analyzer.py --resume r.pdf --jd jd.txt --out gaps.json

Supported file types: .pdf  .docx  .txt
"""

import re
import json
import argparse
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  FILE PARSERS
# ═══════════════════════════════════════════════════════════════

def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(pg.extract_text() for pg in reader.pages if pg.extract_text())
    except Exception as e:
        raise RuntimeError(f"PDF read failed: {e}")


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_text(source: str) -> str:
    """Load text from a file path or return the string as-is."""
    p = Path(source.strip().strip('"').strip("'"))
    if p.is_file():
        ext = p.suffix.lower()
        if ext == ".pdf":
            return _read_pdf(p)
        if ext == ".docx":
            return _read_docx(p)
        if ext == ".txt":
            return p.read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf .docx .txt")
    return source  # treat as raw text


# ═══════════════════════════════════════════════════════════════
#  SKILL VOCABULARY
# ═══════════════════════════════════════════════════════════════

SKILLS = [
    # Languages & scripting
    "python", "sql", "r", "java", "scala", "javascript", "bash",
    "matlab", "vba", "dax", "sas", "spss",
    # BI & visualisation
    "power bi", "tableau", "looker", "excel", "google sheets",
    "jupyter", "databricks", "alteryx", "data visualization",
    # Cloud & platforms
    "aws", "azure", "gcp", "snowflake", "redshift", "bigquery",
    "s3", "lambda", "glue", "synapse", "fabric",
    # Engineering & pipelines
    "spark", "kafka", "airflow", "dbt", "etl", "elt", "fivetran",
    "data pipeline", "data warehouse", "data lake", "mlops",
    # Databases
    "postgresql", "mysql", "sql server", "mongodb", "oracle",
    "redis", "elasticsearch",
    # ML / statistics
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "xgboost", "statistics", "a/b testing",
    "regression", "classification", "clustering", "forecasting",
    "hypothesis testing", "time series", "feature engineering",
    # Data modelling
    "data modeling", "star schema", "dimensional modeling", "olap",
    # Analytics & reporting
    "kpi", "dashboard", "reporting", "business intelligence",
    "data analysis", "data engineering", "product analytics",
    # Tools & practices
    "data governance", "data quality", "stakeholder management",
    "agile", "scrum", "git", "docker", "cross-functional",
    # Python libs
    "pandas", "numpy", "matplotlib",
]

# Skills where substring matching causes false positives → use word boundaries
_BOUNDARY_SKILLS = {"r", "sql", "scala", "dax", "vba", "sas", "java"}


def extract_skills(text: str) -> set:
    t = text.lower()
    found = set()
    for skill in SKILLS:
        if skill in _BOUNDARY_SKILLS:
            if re.search(r"\b" + re.escape(skill) + r"\b", t):
                found.add(skill)
        else:
            if skill in t:
                found.add(skill)
    return found


# ═══════════════════════════════════════════════════════════════
#  SECTION SPLITTER  (used for summary extraction)
# ═══════════════════════════════════════════════════════════════

_SECTION_HEADERS = {
    "summary":    ["professional summary", "summary", "profile", "objective", "about me"],
    "skills":     ["technical skills", "core skills", "skills", "competencies",
                   "tools & technologies", "tools", "expertise"],
    "experience": ["work experience", "professional experience", "experience",
                   "employment", "work history"],
    "education":  ["education", "academic background", "qualifications"],
    "certs":      ["certifications", "certificates", "credentials", "licenses",
                   "training", "courses"],
    "projects":   ["projects", "portfolio", "personal projects", "academic projects"],
}


def _detect_header(line: str) -> str | None:
    clean = line.strip().lower().rstrip(":").strip()
    for section, keys in _SECTION_HEADERS.items():
        for key in keys:
            if clean == key or clean.startswith(key + " ") or clean.endswith(" " + key):
                return section
    return None


def split_sections(text: str) -> dict:
    sections = {s: [] for s in _SECTION_HEADERS}
    current = None
    for line in text.split("\n"):
        detected = _detect_header(line)
        if detected:
            current = detected
            continue
        if current:
            sections[current].append(line)
    return {s: "\n".join(lines).strip() for s, lines in sections.items()}


# ═══════════════════════════════════════════════════════════════
#  CANDIDATE NAME  (best-effort: first non-empty line of resume)
# ═══════════════════════════════════════════════════════════════

def guess_name(resume: str) -> str:
    for line in resume.split("\n"):
        clean = line.strip()
        # Skip lines that look like emails, phones, URLs, or are too long
        if (clean
                and len(clean.split()) <= 5
                and not re.search(r"[@/\\.|:]", clean)
                and not re.search(r"\d{4}", clean)):
            return clean.title()
    return "The candidate"


# ═══════════════════════════════════════════════════════════════
#  SEVERITY HELPER
# ═══════════════════════════════════════════════════════════════

def _severity(skill: str, jd: str) -> str:
    count = jd.lower().count(skill.lower())
    return "high" if count >= 3 else "medium" if count == 2 else "low"


# ═══════════════════════════════════════════════════════════════
#  GAP SUMMARY BUILDER
# ═══════════════════════════════════════════════════════════════

def _matched_skills(resume: str, jd_skills: set) -> list:
    resume_skills = extract_skills(resume)
    return sorted(resume_skills & jd_skills)


def _years_in(text: str) -> int | None:
    m = re.search(r"(\d+)\+?\s*years?\s*(of\s+)?(experience|exp)", text, re.I)
    return int(m.group(1)) if m else None


def _group_skills(skills: list) -> dict:
    """Bucket missing skills into readable categories."""
    buckets = {
        "cloud/infrastructure": [],
        "ML frameworks":        [],
        "data engineering":     [],
        "analytics/BI":         [],
        "other":                [],
    }
    cloud = {"aws", "azure", "gcp", "snowflake", "redshift", "bigquery",
             "synapse", "fabric", "s3", "lambda", "glue"}
    ml    = {"tensorflow", "pytorch", "scikit-learn", "xgboost", "deep learning",
             "machine learning", "nlp", "classification", "regression",
             "forecasting", "time series", "a/b testing", "feature engineering",
             "clustering", "hypothesis testing"}
    de    = {"spark", "kafka", "airflow", "dbt", "etl", "elt", "fivetran",
             "data pipeline", "data warehouse", "data lake", "mlops",
             "data modeling", "star schema", "dimensional modeling"}
    bi    = {"power bi", "tableau", "looker", "data visualization", "dashboard",
             "reporting", "business intelligence", "kpi"}

    for s in skills:
        if s in cloud:
            buckets["cloud/infrastructure"].append(s)
        elif s in ml:
            buckets["ML frameworks"].append(s)
        elif s in de:
            buckets["data engineering"].append(s)
        elif s in bi:
            buckets["analytics/BI"].append(s)
        else:
            buckets["other"].append(s)

    return {k: v for k, v in buckets.items() if v}


def build_summary(name: str, resume: str, jd: str,
                  matched: list, missing: list) -> str:
    sections   = split_sections(resume)
    jd_skills  = extract_skills(jd)
    grouped    = _group_skills(missing)

    # --- what the candidate DOES have ---
    strong = ", ".join(matched[:6]) if matched else "some relevant skills"
    has_str = f"{name} brings solid experience in {strong}."

    # --- what the JD asks for ---
    jd_yrs = _years_in(jd)
    seniority_note = f" The role asks for {jd_yrs}+ years of experience." if jd_yrs else ""

    # --- gap narrative per bucket ---
    gap_parts = []
    if grouped.get("cloud/infrastructure"):
        gap_parts.append(f"cloud/infrastructure skills ({', '.join(grouped['cloud/infrastructure'])})")
    if grouped.get("ML frameworks"):
        gap_parts.append(f"ML depth ({', '.join(grouped['ML frameworks'])})")
    if grouped.get("data engineering"):
        gap_parts.append(f"data engineering tools ({', '.join(grouped['data engineering'])})")
    if grouped.get("analytics/BI"):
        gap_parts.append(f"BI/analytics capabilities ({', '.join(grouped['analytics/BI'])})")
    if grouped.get("other"):
        gap_parts.append(f"additional skills ({', '.join(grouped['other'])})")

    if gap_parts:
        gap_str = "The primary gaps are " + "; ".join(gap_parts) + "."
    else:
        gap_str = "No major skill gaps detected — resume aligns closely with the JD."

    # --- cert / project signals ---
    cert_note = ""
    if not sections.get("certs", "").strip():
        cert_note = " No certifications section was found, which may hurt ATS ranking."

    proj_note = ""
    if not sections.get("projects", "").strip():
        proj_note = " Adding hands-on projects demonstrating the missing skills would strengthen the application significantly."

    # --- leadership signal ---
    leadership_jd     = bool(re.search(r"(lead|manag|mentor|team lead|people manager)", jd, re.I))
    leadership_resume = bool(re.search(r"(led|managed|mentored|team lead|head of)", resume, re.I))
    lead_note = ""
    if leadership_jd and not leadership_resume:
        lead_note = " The JD expects leadership or team-management experience that is not clearly demonstrated in the resume."

    # --- overall verdict ---
    match_pct = round(len(matched) / max(len(jd_skills), 1) * 100)
    if match_pct >= 70:
        verdict = "This is a strong match — targeted polishing of the resume should be sufficient."
    elif match_pct >= 45:
        verdict = "This is a realistic opportunity with focused upskilling on the gaps above."
    else:
        verdict = "This is a stretch role; significant upskilling is needed before applying."

    summary = (
        f"{has_str}{seniority_note} "
        f"{gap_str}"
        f"{lead_note}"
        f"{cert_note}"
        f"{proj_note} "
        f"{verdict}"
    )
    return re.sub(r" {2,}", " ", summary).strip()


# ═══════════════════════════════════════════════════════════════
#  MAIN ANALYSER
# ═══════════════════════════════════════════════════════════════

def analyze(resume_text: str, jd_text: str) -> dict:
    name      = guess_name(resume_text)
    jd_skills = extract_skills(jd_text)
    matched   = _matched_skills(resume_text, jd_skills)
    missing   = sorted(jd_skills - set(matched))

    # Capitalise for readability
    missing_display = [s.title() for s in missing]

    summary = build_summary(name, resume_text, jd_text, matched, missing)

    return {
        "missing_skills": missing_display,
        "gap_summary":    summary,
    }


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE INPUT HELPER
# ═══════════════════════════════════════════════════════════════

def _prompt(label: str) -> str:
    print(f"\n── {label} {'─'*(50-len(label))}")
    print("  Enter a file path (.pdf / .docx / .txt)  OR  paste text.")
    print("  When pasting, type END on its own line when done.\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        stripped = line.strip().strip('"').strip("'")
        if not lines and Path(stripped).is_file():
            print(f"  Reading {Path(stripped).name} …")
            return load_text(stripped)
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare resume vs JD → JSON gap report"
    )
    parser.add_argument("--resume", default=None, help="Resume file path or raw text")
    parser.add_argument("--jd",     default=None, help="Job description file path or raw text")
    parser.add_argument("--out",    default=None, help="Optional: save output to .json file")
    args = parser.parse_args()

    if args.resume:
        resume_text = load_text(args.resume)
    else:
        resume_text = _prompt("STEP 1 — RESUME  (file path or paste)")

    if args.jd:
        jd_text = load_text(args.jd)
    else:
        jd_text = _prompt("STEP 2 — JOB DESCRIPTION  (file path or paste)")

    if not resume_text.strip() or not jd_text.strip():
        print("Error: resume and job description cannot be empty.")
        sys.exit(1)

    print("\nAnalyzing …\n")
    result = analyze(resume_text, jd_text)
    output = json.dumps(result, indent=2)

    print(output)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()