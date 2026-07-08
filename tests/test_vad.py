import unittest

from pipeline.audio_in import FRAME_BYTES
from pipeline.vad import Clause, ClauseSegmenter, SpeechStart


class FakeVAD:
    """Deterministic VAD for testing the segmenter state machine."""

    def __init__(self):
        self.speech = False

    def prob(self, frame: bytes) -> float:
        return 0.8 if self.speech else 0.1


class TestClauseSegmenter(unittest.TestCase):
    def _make_segmenter(self, **kwargs):
        return ClauseSegmenter(FakeVAD(), **kwargs)

    def _frames(self, n):
        return [b"\x00" * FRAME_BYTES for _ in range(n)]

    def test_silence_produces_no_clauses(self):
        segmenter = self._make_segmenter()
        for frame in self._frames(50):
            events = segmenter.feed(frame)
            self.assertEqual(events, [])

    def test_speech_then_silence_emits_clause(self):
        segmenter = self._make_segmenter()
        segmenter.vad.speech = True
        events = []
        # 30 speech frames
        for _ in range(30):
            events.extend(segmenter.feed(b"\x01" * FRAME_BYTES))
        self.assertEqual(len([e for e in events if isinstance(e, SpeechStart)]), 1)

        # Enough silence to end the clause
        segmenter.vad.speech = False
        for _ in range(segmenter.end_silence_frames + 1):
            events.extend(segmenter.feed(b"\x00" * FRAME_BYTES))

        clauses = [e for e in events if isinstance(e, Clause)]
        self.assertEqual(len(clauses), 1)
        self.assertGreater(clauses[0].duration_s, 0)

    def test_short_noise_is_ignored(self):
        segmenter = self._make_segmenter()
        events = []
        segmenter.vad.speech = True
        for _ in range(segmenter.min_clause_frames - 1):
            events.extend(segmenter.feed(b"\x01" * FRAME_BYTES))
        segmenter.vad.speech = False
        for _ in range(segmenter.end_silence_frames + 1):
            events.extend(segmenter.feed(b"\x00" * FRAME_BYTES))

        clauses = [e for e in events if isinstance(e, Clause)]
        self.assertEqual(len(clauses), 0)


if __name__ == "__main__":
    unittest.main()
