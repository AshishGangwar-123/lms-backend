from __future__ import annotations

import json
import os
import re
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def get_cors_origins() -> List[str]:
    default_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "https://lmsprototype-six.vercel.app",
    ]
    configured_origins = os.getenv("CORS_ORIGINS", "")
    extra_origins = [origin for origin in configured_origins.split(",") if origin.strip()]
    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        extra_origins.append(frontend_url)
    return _dedupe(default_origins + extra_origins)


app = FastAPI(title="NLM Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SYSTEM_PROMPT = """
You are an AI mock interviewer.
Ask interview questions naturally, evaluate the user's answer,
give short useful feedback, and continue the interview.
""".strip()

MOCK_INTERVIEW_SESSIONS: Dict[str, List[Any]] = {}

SKILL_DB = [
    "python", "java", "c", "c++", "javascript", "typescript", "react", "node.js",
    "nodejs", "express", "mongodb", "sql", "mysql", "postgresql", "html", "css",
    "git", "github", "django", "flask", "fastapi", "machine learning",
    "deep learning", "nlp", "rag", "langchain", "docker", "aws", "rest api",
    "restful api", "streamlit", "pandas", "numpy", "scikit-learn", "opencv",
    "firebase", "tailwind", "bootstrap", "linux", "oop", "data structures",
    "algorithms", "api", "jwt", "socket.io", "redis", "kubernetes", "tableau",
    "power bi", "excel", "figma", "canva", "tensorflow", "pytorch", "keras",
    "spring boot", "hibernate", "microservices", "graphql", "postman",
    "azure", "gcp", "problem solving", "communication", "leadership",
]

ACTION_VERBS = [
    "developed", "built", "designed", "created", "implemented", "optimized",
    "improved", "led", "managed", "automated", "deployed", "integrated",
    "analyzed", "engineered", "tested", "delivered", "launched", "solved",
    "reduced", "increased", "boosted", "streamlined", "enhanced",
]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


def get_groq_api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


@lru_cache(maxsize=1)
def get_mock_interview_llm() -> ChatGroq:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "200"))

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text_parts = []

    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        except Exception:
            continue

    return "\n".join(text_parts).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    patterns = [
        r"(\+91[\-\s]?)?[6-9]\d{9}",
        r"\+?\d[\d\-\s]{8,15}\d",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def extract_links(text: str) -> List[str]:
    links = re.findall(
        r"(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)",
        text,
        flags=re.IGNORECASE,
    )
    cleaned_links = [link.rstrip(".,;)") for link in links]
    return sorted(set(cleaned_links))


def extract_skills(text: str, skill_db: List[str]) -> List[str]:
    text_lower = text.lower()
    found_skills = []

    for skill in skill_db:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills))


def extract_keywords_from_jd(jd_text: str, skill_db: List[str]) -> List[str]:
    if not jd_text or not jd_text.strip():
        return []

    jd_lower = jd_text.lower()
    matched = []

    for skill in skill_db:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, jd_lower):
            matched.append(skill)

    return sorted(set(matched))


def compare_resume_with_jd(resume_skills: List[str], jd_keywords: List[str]):
    matched = [skill for skill in jd_keywords if skill in resume_skills]
    missing = [skill for skill in jd_keywords if skill not in resume_skills]

    if not jd_keywords:
        match_score = 0.0
    else:
        match_score = round((len(matched) / len(jd_keywords)) * 100, 2)

    return matched, missing, match_score


def detect_resume_sections(resume_text: str) -> Dict[str, bool]:
    text_lower = resume_text.lower()
    section_aliases = {
        "summary": ["summary", "profile", "objective", "about me", "professional summary"],
        "education": ["education", "academic", "qualification"],
        "experience": ["experience", "work experience", "internship", "employment"],
        "projects": ["project", "projects"],
        "skills": ["skills", "technical skills", "core skills"],
        "certifications": ["certification", "certifications", "licenses"],
        "achievements": ["achievement", "achievements", "awards"],
    }

    detected = {}
    for section, aliases in section_aliases.items():
        detected[section] = any(alias in text_lower for alias in aliases)
    return detected


def extract_action_verbs(resume_text: str) -> List[str]:
    text_lower = resume_text.lower()
    found = [
        verb for verb in ACTION_VERBS
        if re.search(r"\b" + re.escape(verb) + r"\b", text_lower)
    ]
    return sorted(set(found))


def has_quantified_achievements(resume_text: str) -> bool:
    patterns = [
        r"\b\d+%\b",
        r"\b\d+\+?\b",
        r"\b\d+\s*(users|projects|clients|days|months|years|hours|weeks)\b",
        r"\b(increased|reduced|improved|boosted|saved)\b.*\b\d+",
        r"\b\d+\s*(x|times)\b",
    ]
    text_lower = resume_text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def extract_possible_job_titles(text: str) -> List[str]:
    titles = [
        "developer", "software engineer", "engineer", "data analyst",
        "analyst", "frontend developer", "backend developer", "full stack developer",
        "intern", "designer", "manager", "consultant", "tester", "qa engineer",
    ]
    text_lower = text.lower()
    return [title for title in titles if title in text_lower]


def calculate_ats_analysis(
    email: Optional[str],
    phone: Optional[str],
    links: List[str],
    skills: List[str],
    resume_text: str,
    jd_keywords: List[str],
    matched_keywords: List[str],
) -> Dict[str, Any]:
    strengths = []
    improvements = []

    sections = detect_resume_sections(resume_text)
    action_verbs_found = extract_action_verbs(resume_text)
    quantified = has_quantified_achievements(resume_text)
    word_count = len(resume_text.split())
    job_titles_found = extract_possible_job_titles(resume_text)

    jd_score = 0
    if jd_keywords:
        keyword_ratio = len(matched_keywords) / max(len(jd_keywords), 1)

        if keyword_ratio >= 0.85:
            jd_score += 42
            strengths.append("Resume is strongly aligned with the job description keywords.")
        elif keyword_ratio >= 0.65:
            jd_score += 34
            strengths.append("Resume has a good keyword match with the job description.")
        elif keyword_ratio >= 0.45:
            jd_score += 24
            strengths.append("Resume has partial keyword alignment with the job description.")
        elif keyword_ratio >= 0.25:
            jd_score += 14
            improvements.append("Add more important keywords from the job description.")
        else:
            jd_score += 6
            improvements.append("Resume is weakly aligned with the job description keywords.")

        if len(matched_keywords) >= 5:
            jd_score += 5
        elif len(matched_keywords) >= 3:
            jd_score += 3

        if job_titles_found:
            jd_score += 3
    else:
        improvements.append("Add a job description for a more accurate ATS match analysis.")

    jd_score = min(jd_score, 50)

    ats_parsability = 0
    if email:
        ats_parsability += 4
        strengths.append("Email address is present.")
    else:
        improvements.append("Add a professional email address.")

    if phone:
        ats_parsability += 4
        strengths.append("Phone number is present.")
    else:
        improvements.append("Add a phone number.")

    if links:
        ats_parsability += 3
        strengths.append("Professional links are included.")
    else:
        improvements.append("Add LinkedIn or GitHub/profile links if relevant.")

    if sections["education"]:
        ats_parsability += 4
    else:
        improvements.append("Add an Education section.")

    if sections["experience"]:
        ats_parsability += 5
    else:
        improvements.append("Add an Experience section if applicable.")

    if sections["projects"]:
        ats_parsability += 4
    else:
        improvements.append("Add a Projects section.")

    if sections["skills"]:
        ats_parsability += 4
    else:
        improvements.append("Add a dedicated Skills section.")

    if sections["summary"]:
        ats_parsability += 2
    else:
        improvements.append("Add a short professional summary.")

    ats_parsability = min(ats_parsability, 30)

    content_quality = 0
    if len(skills) >= 8:
        content_quality += 5
        strengths.append("Good range of technical skills detected.")
    elif len(skills) >= 5:
        content_quality += 3
    else:
        improvements.append("Expand and clarify your technical skills.")

    if action_verbs_found:
        content_quality += 5
        strengths.append("Strong action verbs are used.")
    else:
        improvements.append("Use stronger action verbs like developed, built, optimized, led.")

    if quantified:
        content_quality += 6
        strengths.append("Resume includes measurable or quantified impact.")
    else:
        improvements.append("Add quantified achievements, metrics, or business impact.")

    if sections["experience"] and sections["projects"] and quantified:
        content_quality += 2

    if word_count < 180:
        improvements.append("Resume content looks too short; add more relevant detail.")
    elif word_count > 900:
        improvements.append("Resume may be too long; tighten weak or repetitive content.")

    content_quality = min(content_quality, 20)
    final_score = jd_score + ats_parsability + content_quality

    if not jd_keywords:
        final_score = min(final_score, 78)
    if not sections["experience"] and not sections["projects"]:
        final_score = min(final_score, 72)
    if not quantified:
        final_score = min(final_score, 84)

    final_score = min(final_score, 96)

    return {
        "ats_score": int(round(final_score)),
        "strengths_rule_based": strengths,
        "improvements_rule_based": improvements,
        "detected_sections": sections,
        "action_verbs_found": action_verbs_found,
        "score_breakdown": {
            "jd_match": int(round(jd_score)),
            "ats_parsability": int(round(ats_parsability)),
            "content_quality": int(round(content_quality)),
        },
    }


def parse_llm_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
        "improved_summary": "",
    }


def generate_llm_feedback(
    llm: ChatGroq,
    resume_text: str,
    analysis_data: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""
You are an ATS resume reviewer for job seekers.

Important:
- Do not invent or change the ATS score.
- The score is already computed separately.
- Use the structured analysis as the source of truth.
- Your job is only to explain the result and suggest improvements.

Resume text:
{resume_text[:5000]}

Structured analysis:
{json.dumps(analysis_data, indent=2)}

Return valid JSON only with exactly these keys:
summary
strengths
weaknesses
suggestions
improved_summary

Rules:
- Keep feedback strict and realistic, not overly positive.
- If keyword match is weak, say it clearly.
- If quantified impact is missing, mention it clearly.
- If sections are missing, mention them clearly.
- improved_summary should be concise, ATS-friendly, and role-focused.
- No markdown.
- No extra text before or after JSON.
"""

    response = llm.invoke(prompt)
    return parse_llm_json(response.content)


def analyze_resume_file(pdf_path: str, job_description: str = "") -> Dict[str, Any]:
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise ValueError(
            "Could not extract text from the uploaded resume. This PDF may be scanned or image-based."
        )

    email = extract_email(cleaned_text)
    phone = extract_phone(cleaned_text)
    links = extract_links(cleaned_text)
    resume_skills = extract_skills(cleaned_text, SKILL_DB)

    jd_keywords = extract_keywords_from_jd(job_description, SKILL_DB)
    matched_keywords, missing_keywords, jd_match_score = compare_resume_with_jd(
        resume_skills,
        jd_keywords,
    )

    rule_analysis = calculate_ats_analysis(
        email=email,
        phone=phone,
        links=links,
        skills=resume_skills,
        resume_text=cleaned_text,
        jd_keywords=jd_keywords,
        matched_keywords=matched_keywords,
    )

    base_analysis = {
        "contact_details": {
            "email": email,
            "phone": phone,
            "links": links,
        },
        "resume_skills": resume_skills,
        "job_description_keywords": jd_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_match_score": jd_match_score,
        "ats_score": rule_analysis["ats_score"],
        "resume_score": rule_analysis["ats_score"],
        "strengths_rule_based": rule_analysis["strengths_rule_based"],
        "improvements_rule_based": rule_analysis["improvements_rule_based"],
        "detected_sections": rule_analysis["detected_sections"],
        "action_verbs_found": rule_analysis["action_verbs_found"],
        "score_breakdown": rule_analysis["score_breakdown"],
    }

    llm_feedback = {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
        "improved_summary": "",
    }

    groq_api_key = get_groq_api_key()
    if groq_api_key:
        try:
            llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), api_key=groq_api_key)
            llm_feedback = generate_llm_feedback(llm, cleaned_text, base_analysis)
        except Exception:
            pass

    return {
        "atsScore": base_analysis["ats_score"],
        "keywordMatch": base_analysis["keyword_match_score"],
        "resumeScore": base_analysis["resume_score"],
        "missingKeywords": base_analysis["missing_keywords"],
        "matchedKeywords": base_analysis["matched_keywords"],
        "strengths": (
            llm_feedback["strengths"]
            if llm_feedback["strengths"]
            else base_analysis["strengths_rule_based"]
        ),
        "improvements": (
            llm_feedback["suggestions"]
            if llm_feedback["suggestions"]
            else base_analysis["improvements_rule_based"]
        ),
        "coverLetterSkills": [
            {
                "skill": "Keyword Alignment",
                "score": round(min(3.0, (base_analysis["keyword_match_score"] / 100) * 3), 1),
                "status": "good" if base_analysis["keyword_match_score"] >= 60 else "warning",
            },
            {
                "skill": "ATS Readability",
                "score": round(min(3.0, (base_analysis["ats_score"] / 100) * 3), 1),
                "status": "good" if base_analysis["ats_score"] >= 70 else "warning",
            },
            {
                "skill": "Content Strength",
                "score": round(min(3.0, (len(base_analysis["resume_skills"]) / 10) * 3), 1),
                "status": "good" if len(base_analysis["resume_skills"]) >= 6 else "warning",
            },
        ],
        "summary": llm_feedback.get("summary", ""),
        "improvedSummary": llm_feedback.get("improved_summary", ""),
        "details": {
            "email": email,
            "phone": phone,
            "links": links,
            "skillsFound": resume_skills,
            "detectedSections": base_analysis["detected_sections"],
            "actionVerbsFound": base_analysis["action_verbs_found"],
            "scoreBreakdown": base_analysis["score_breakdown"],
        },
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "NLM Backend API is running",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "nlm-backend",
        "groq_api_key_configured": bool(get_groq_api_key()),
        "allowed_origins": get_cors_origins(),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        return ChatResponse(reply="Please enter a message.")

    try:
        llm = get_mock_interview_llm()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    if req.session_id not in MOCK_INTERVIEW_SESSIONS:
        MOCK_INTERVIEW_SESSIONS[req.session_id] = [SystemMessage(content=SYSTEM_PROMPT)]

    session = MOCK_INTERVIEW_SESSIONS[req.session_id]
    session.append(HumanMessage(content=req.message))

    try:
        response = llm.invoke(session)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Mock interview failed: {error}") from error

    session.append(AIMessage(content=response.content))
    return ChatResponse(reply=response.content)


@app.post("/api/reset/{session_id}")
def reset_chat(session_id: str) -> Dict[str, str]:
    MOCK_INTERVIEW_SESSIONS[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
    return {"message": "Session reset successful"}


@app.post("/analyze-resume")
@app.post("/api/resume/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(""),
) -> Dict[str, Any]:
    if not resume:
        raise HTTPException(status_code=400, detail="Resume file is required.")

    filename = (resume.filename or "").lower()
    content_type = (resume.content_type or "").lower()
    allowed_pdf_types = {
        "application/pdf",
        "application/x-pdf",
        "binary/octet-stream",
        "application/octet-stream",
    }

    is_pdf = filename.endswith(".pdf") or content_type in allowed_pdf_types
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported right now.")

    temp_file_path = None
    try:
        file_bytes = await resume.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded resume file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        return analyze_resume_file(temp_file_path, job_description)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {error}") from error
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
