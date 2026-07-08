import asyncio
import io
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from google import genai
from google.genai import types

from .config import TTS_MODEL, TTS_SAMPLE_RATE, TTS_VOICE


def _to_int16_pcm(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TTS_SAMPLE_RATE:
        # Simple linear resample if the TTS returned a different rate.
        old_len = len(audio)
        new_len = int(old_len * TTS_SAMPLE_RATE / sample_rate)
        indices = np.linspace(0, old_len - 1, new_len)
        audio = np.interp(indices, np.arange(old_len), audio)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


class GeminiTTS:
    """Gemini Hindi TTS."""

    sample_rate = TTS_SAMPLE_RATE

    def __init__(self, client: genai.Client, model: str = TTS_MODEL, voice: str = TTS_VOICE):
        self.client = client
        self.model = model
        self.voice = voice

    async def synthesize(self, hindi: str) -> bytes:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=hindi,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                    )
                ),
            ),
        )
        part = resp.candidates[0].content.parts[0]
        return part.inline_data.data


class EdgeTTS:
    """Microsoft's free Edge read-aloud voices — no API key, and the Hindi
    voices are surprisingly good."""

    sample_rate = TTS_SAMPLE_RATE

    def __init__(self, voice: str = None):
        from .config import EDGE_TTS_VOICE

        self.voice = voice or EDGE_TTS_VOICE

    async def synthesize(self, hindi: str) -> bytes:
        import edge_tts

        com = edge_tts.Communicate(hindi, voice=self.voice)
        mp3 = bytearray()
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])

        def _decode() -> bytes:
            buf = io.BytesIO(bytes(mp3))
            audio, rate = sf.read(buf, dtype="float32")
            return _to_int16_pcm(audio, rate)

        return await asyncio.to_thread(_decode)


class SayTTS:
    """Fully offline fallback — macOS's built-in Hindi voice."""

    sample_rate = TTS_SAMPLE_RATE

    async def synthesize(self, hindi: str) -> bytes:
        def _run() -> bytes:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=True) as aiff:
                subprocess.run(["say", "-v", "Lekha", "-o", aiff.name, hindi], check=True)
                audio, rate = sf.read(aiff.name, dtype="float32")
                return _to_int16_pcm(audio, rate)

        return await asyncio.to_thread(_run)
