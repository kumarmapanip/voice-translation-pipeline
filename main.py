import argparse
import asyncio
import logging
import os

from pipeline.asr import GeminiASR, GroqASR
from pipeline.audio_in import MicSource, WavSource
from pipeline.config import make_client, make_groq_client
from pipeline.orchestrator import Pipeline
from pipeline.playback import SpeakerSink, WavSink
from pipeline.translate import GeminiTranslator, GroqTranslator
from pipeline.tts import EdgeTTS, GeminiTTS, SayTTS
from pipeline.vad import ClauseSegmenter, SileroVAD


class _QuietThirdParty(logging.Filter):
    """Keeps HTTP-client chatter off the console. It still lands in pipeline.log,
    which is handy when debugging a flaky API call after the fact."""

    NOISY = ("httpx", "httpcore", "groq._base_client")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(self.NOISY):
            return record.levelno >= logging.WARNING
        return True


def _setup_logging():
    console = logging.StreamHandler()
    console.addFilter(_QuietThirdParty())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            console,
            logging.FileHandler("pipeline.log", mode="a"),
        ],
    )


def main():
    _setup_logging()
    p = argparse.ArgumentParser(description="English to Hindi voice translation")
    p.add_argument("--wav", help="input WAV file (16 kHz mono int16); default is live mic")
    p.add_argument("--out", help="write Hindi audio to this WAV file instead of the speaker")
    p.add_argument(
        "--engine",
        choices=["auto", "groq", "gemini"],
        default="auto",
        help="ASR + translation backend; auto picks groq if GROQ_API_KEY is set",
    )
    p.add_argument("--tts", choices=["edge", "gemini", "say", "none"], default="edge")
    p.add_argument("--realtime", action="store_true", help="pace WAV input at wall-clock speed")
    args = p.parse_args()

    engine = args.engine
    if engine == "auto":
        engine = "groq" if os.environ.get("GROQ_API_KEY") else "gemini"

    if engine == "groq":
        groq = make_groq_client()
        asr, translator = GroqASR(groq), GroqTranslator(groq)
    else:
        gemini = make_client()
        asr, translator = GeminiASR(gemini), GeminiTranslator(gemini)

    if args.tts == "gemini":
        tts = GeminiTTS(make_client())
    else:
        tts = {"edge": EdgeTTS(), "say": SayTTS(), "none": None}[args.tts]

    source = WavSource(args.wav, realtime=args.realtime) if args.wav else MicSource()
    rate = tts.sample_rate if tts else 24000
    sink = WavSink(args.out, rate) if args.out else SpeakerSink(rate)

    print(f"[engine: {engine} | tts: {args.tts}]", flush=True)
    pipe = Pipeline(
        source=source,
        segmenter=ClauseSegmenter(SileroVAD()),
        asr=asr,
        translator=translator,
        tts=tts,
        sink=sink,
    )
    try:
        asyncio.run(pipe.run())
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
