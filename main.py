import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mockInterview import router as mock_router
from resumeAnalyser import router as resume_router

load_dotenv()

app = FastAPI(title="LMS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        # yahan apna deployed frontend domain bhi add karo
        "https://lmsprototype-six.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mock_router)
app.include_router(resume_router)


@app.get("/")
def root():
    return {"message": "LMS backend is running"}


@app.get("/api/all-health")
def all_health():
    return {
        "status": "ok",
        "groq_api_key_found": bool(os.getenv("GROQ_API_KEY"))
    }