from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .audio_in import pcm_to_float
from .config import FRAME_SAMPLES, SAMPLE_RATE

MODEL_PATH = "models/silero_vad.onnx"


class SileroVAD:
    def __init__(self, model_path: str = MODEL_PATH):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.reset()

    # Silero v5 quietly returns ~0 for everything unless each window carries
    # the last 64 samples of the previous frame. Learned the hard way.
    CONTEXT = 64

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)

    def prob(self, frame: bytes) -> float:
        audio = pcm_to_float(frame)
        x = np.concatenate([self._context, audio]).reshape(1, self.CONTEXT + FRAME_SAMPLES)
        self._context = audio[-self.CONTEXT:]
        out, self._state = self.session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        return float(out[0, 0])


@dataclass
class SpeechStart:
    pass


@dataclass
class Clause:
    audio: bytes  # 16 kHz mono int16 PCM
    duration_s: float


class ClauseSegmenter:
    """Chops the frame stream into clause-sized speech segments.

    A clause ends when the speaker pauses for `end_silence_ms`. If they just
    won't stop, `max_clause_s` forces a cut anyway — otherwise one run-on
    sentence stalls everything queued behind it.
    """

    def __init__(
        self,
        vad: SileroVAD,
        threshold: float = 0.5,
        end_silence_ms: int = 450,
        min_clause_ms: int = 250,
        max_clause_s: float = 12.0,
        pad_frames: int = 6,
    ):
        self.vad = vad
        self.threshold = threshold
        self.frame_ms = FRAME_SAMPLES * 1000 / SAMPLE_RATE
        self.end_silence_frames = int(end_silence_ms / self.frame_ms)
        self.min_clause_frames = int(min_clause_ms / self.frame_ms)
        self.max_clause_frames = int(max_clause_s * 1000 / self.frame_ms)
        self.pad_frames = pad_frames
        self._prebuffer: list[bytes] = []
        self._clause: list[bytes] = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0

    def feed(self, frame: bytes):
        """Returns a list of SpeechStart/Clause events for this frame."""
        events = []
        is_speech = self.vad.prob(frame) >= self.threshold

        if not self._in_speech:
            self._prebuffer.append(frame)
            if len(self._prebuffer) > self.pad_frames:
                self._prebuffer.pop(0)
            if is_speech:
                self._in_speech = True
                self._clause = list(self._prebuffer)
                self._prebuffer = []
                self._silence_run = 0
                self._speech_frames = 1
                events.append(SpeechStart())
            return events

        self._clause.append(frame)
        if is_speech:
            self._silence_run = 0
            self._speech_frames += 1
        else:
            self._silence_run += 1

        if self._silence_run >= self.end_silence_frames or len(self._clause) >= self.max_clause_frames:
            if self._speech_frames >= self.min_clause_frames:
                audio = b"".join(self._clause)
                events.append(Clause(audio=audio, duration_s=len(audio) / 2 / SAMPLE_RATE))
            self._in_speech = False
            self._clause = []
            self._speech_frames = 0
            self._silence_run = 0
        return events
