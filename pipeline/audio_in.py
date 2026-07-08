import asyncio
import logging
import wave

import numpy as np

from .config import FRAME_SAMPLES, SAMPLE_RATE

logger = logging.getLogger(__name__)

FRAME_BYTES = FRAME_SAMPLES * 2  # int16 mono


class WavSource:
    """Reads fixed-size PCM frames from a 16 kHz mono int16 WAV file.

    realtime=True paces frames at wall-clock speed, like a live mic.
    """

    def __init__(self, path: str, realtime: bool = False):
        self.path = path
        self.realtime = realtime

    async def frames(self):
        with wave.open(self.path, "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise SystemExit(
                    f"{self.path}: expected {SAMPLE_RATE} Hz mono 16-bit WAV, got "
                    f"{w.getframerate()} Hz, {w.getnchannels()} ch, {w.getsampwidth() * 8}-bit"
                )
            frame_s = FRAME_SAMPLES / SAMPLE_RATE
            while True:
                data = w.readframes(FRAME_SAMPLES)
                if not data:
                    break
                if len(data) < FRAME_BYTES:
                    data += b"\x00" * (FRAME_BYTES - len(data))
                yield data
                if self.realtime:
                    await asyncio.sleep(frame_s)
        # Give the segmenter trailing silence so the last clause closes.
        for _ in range(30):
            yield b"\x00" * FRAME_BYTES
            if self.realtime:
                await asyncio.sleep(frame_s)


class MicSource:
    """Yields fixed-size PCM frames from the default input device."""

    async def frames(self):
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue()

        def callback(indata, _frames, _time, status):
            if status:
                logger.warning("[mic] %s", status)
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        with stream:
            while True:
                yield await q.get()


def pcm_to_float(frame: bytes) -> np.ndarray:
    return np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
