import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms at 16 kHz — the window size Silero VAD was trained on

ASR_MODEL = os.environ.get("ASR_MODEL", "gemini-2.5-flash")
MT_MODEL = os.environ.get("MT_MODEL", "gemini-2.5-flash")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.environ.get("TTS_VOICE", "Kore")
TTS_SAMPLE_RATE = 24000

GROQ_ASR_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
GROQ_MT_MODEL = os.environ.get("GROQ_MT_MODEL", "llama-3.3-70b-versatile")
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "hi-IN-SwaraNeural")


def make_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (put it in .env)")
    return genai.Client(api_key=key)


def make_groq_client():
    from groq import AsyncGroq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set (put it in .env)")
    return AsyncGroq(api_key=key)
