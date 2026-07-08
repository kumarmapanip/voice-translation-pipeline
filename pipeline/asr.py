import io
import wave

from google import genai
from google.genai import types

from .config import ASR_MODEL, SAMPLE_RATE

PROMPT = (
    "Transcribe this English speech verbatim. Output only the transcript text, "
    "with normal punctuation. If there is no intelligible speech, output an empty string."
)


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class GroqASR:
    """Whisper large-v3-turbo on Groq — a purpose-built ASR model, so it's
    both faster and steadier than asking a general LLM to transcribe.
    About a second per clause in practice."""

    def __init__(self, client, model: str = None):
        from .config import GROQ_ASR_MODEL

        self.client = client
        self.model = model or GROQ_ASR_MODEL

    async def transcribe(self, pcm: bytes) -> str:
        resp = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("clause.wav", pcm_to_wav(pcm)),
            language="en",
            temperature=0.0,
            response_format="text",
        )
        text = resp if isinstance(resp, str) else resp.text
        return (text or "").strip()


class GeminiASR:
    def __init__(self, client: genai.Client, model: str = ASR_MODEL):
        self.client = client
        self.model = model

    async def transcribe(self, pcm: bytes) -> str:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pcm_to_wav(pcm), mime_type="audio/wav"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (resp.text or "").strip()
