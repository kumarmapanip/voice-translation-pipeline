import asyncio
import logging
import time

from .util import with_retries
from .vad import Clause, ClauseSegmenter, SpeechStart

logger = logging.getLogger(__name__)

END = object()

# Ignore audio for this long after barge-in so residual TTS output is not
# re-transcribed.
BARGE_IN_COOLDOWN_S = 0.5


class Pipeline:
    """Runs every stage as its own task, connected by queues.

    That's where the speed comes from: while one clause is playing out loud,
    the next is mid-translation and a third is still in ASR. The queues are
    bounded on purpose — if a stage falls behind, upstream blocks instead of
    piling up stale work.
    """

    def __init__(
        self,
        source,
        segmenter: ClauseSegmenter,
        asr,
        translator,
        tts,
        sink,
        verbose: bool = True,
    ):
        self.source = source
        self.segmenter = segmenter
        self.asr = asr
        self.translator = translator
        self.tts = tts
        self.sink = sink
        self.verbose = verbose
        self.asr_q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.mt_q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.tts_q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.results: list[dict] = []
        self._barge_in_cooldown_until = 0.0

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    @staticmethod
    def _flush(q: asyncio.Queue):
        while not q.empty():
            item = q.get_nowait()
            if item is END:
                q.put_nowait(END)
                return

    def _in_barge_in_cooldown(self) -> bool:
        return time.monotonic() < self._barge_in_cooldown_until

    async def _ingest(self):
        async for frame in self.source.frames():
            for event in self.segmenter.feed(frame):
                if isinstance(event, SpeechStart):
                    if self.sink.playing:
                        self._log("[barge-in] stopping playback")
                        self.sink.stop_playback()
                        self._flush(self.mt_q)
                        self._flush(self.tts_q)
                        self._barge_in_cooldown_until = (
                            time.monotonic() + BARGE_IN_COOLDOWN_S
                        )
                elif isinstance(event, Clause):
                    if self._in_barge_in_cooldown():
                        self._log(
                            f"[barge-in] dropping clause during cooldown "
                            f"({event.duration_s:.2f}s)"
                        )
                        continue
                    await self.asr_q.put((event, time.monotonic()))
        await self.asr_q.put(END)

    async def _asr_stage(self):
        while (item := await self.asr_q.get()) is not END:
            clause, t0 = item
            try:
                text = await with_retries(self.asr.transcribe, clause.audio)
            except Exception as e:
                logger.warning("[asr] dropping clause after retries: %s", e)
                continue
            t1 = time.monotonic()
            if text:
                self._log(f"  EN ({(t1 - t0) * 1000:.0f} ms): {text}")
                await self.mt_q.put((text, t0, t1))
        await self.mt_q.put(END)

    async def _mt_stage(self):
        while (item := await self.mt_q.get()) is not END:
            text, t0, t1 = item
            try:
                hindi = await with_retries(self.translator.translate, text)
            except Exception as e:
                logger.warning("[mt] dropping clause after retries: %s", e)
                continue
            t2 = time.monotonic()
            if hindi:
                self._log(f"  HI ({(t2 - t1) * 1000:.0f} ms): {hindi}")
                self.results.append({"en": text, "hi": hindi})
                await self.tts_q.put((hindi, t0, t2))
        await self.tts_q.put(END)

    async def _tts_stage(self):
        while (item := await self.tts_q.get()) is not END:
            hindi, t0, t2 = item
            if self.tts is None:
                continue
            try:
                audio = await with_retries(self.tts.synthesize, hindi)
            except Exception as e:
                logger.warning("[tts] dropping clause after retries: %s", e)
                continue
            t3 = time.monotonic()
            self._log(
                f"  TTS ({(t3 - t2) * 1000:.0f} ms), total {(t3 - t0) * 1000:.0f} ms"
            )
            self.sink.feed(audio)

    async def run(self):
        self.sink.start()
        try:
            await asyncio.gather(
                self._ingest(), self._asr_stage(), self._mt_stage(), self._tts_stage()
            )
            await self.sink.drain()
        finally:
            self.sink.close()
        return self.results
