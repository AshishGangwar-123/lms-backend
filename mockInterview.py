print("APP.PY LOADED")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()

app = FastAPI(title="Mock Interview API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=200,
    api_key=GROQ_API_KEY,
)

SYSTEM_PROMPT = """
You are an AI mock interviewer.
Ask interview questions naturally, evaluate the user's answer,
give short useful feedback, and continue the interview.
"""

sessions = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    return {"message": "Mock Interview API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok", "groq_api_key_found": bool(GROQ_API_KEY)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        if not req.message.strip():
            return ChatResponse(reply="Please enter a message.")

        if req.session_id not in sessions:
            sessions[req.session_id] = [SystemMessage(content=SYSTEM_PROMPT)]

        sessions[req.session_id].append(HumanMessage(content=req.message))
        response = llm.invoke(sessions[req.session_id])
        sessions[req.session_id].append(AIMessage(content=response.content))

        return ChatResponse(reply=response.content)

    except Exception as e:
        return ChatResponse(reply=f"Backend error: {str(e)}")


@app.post("/api/reset/{session_id}")
def reset_chat(session_id: str):
    sessions[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
    return {"message": "Session reset successful"}