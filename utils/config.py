"""Configuration for the browser voice demo."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CALLS_LOGS_DIR = LOGS_DIR / "calls"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CALLS_LOGS_DIR, exist_ok=True)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")  # gemini or groq
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

AGENT_NAME = os.getenv("AGENT_NAME", "Jamie")
DEEPGRAM_TTS_VOICE = os.getenv("DEEPGRAM_TTS_VOICE", "aura-2-luna-en")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")

MISSING_KEYS = []
if not DEEPGRAM_API_KEY:
    MISSING_KEYS.append("DEEPGRAM_API_KEY")
if LLM_BACKEND.lower() == "groq":
    if not GROQ_API_KEY:
        MISSING_KEYS.append("GROQ_API_KEY")
elif not GOOGLE_API_KEY:
    MISSING_KEYS.append("GOOGLE_API_KEY")
