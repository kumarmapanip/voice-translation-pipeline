# Voice translation pipeline (English → Hindi)

Speak English into your mic, hear Hindi out of your speaker a few seconds
later. Under the hood: mic → VAD segmentation → ASR → translation → Hindi TTS
→ playback.

## How it works

The trick to keeping this fast isn't the models — it's deciding where to cut
the audio. Translating word-by-word doesn't work (English puts the verb where
Hindi wants it last), and waiting for full sentences feels sluggish. So the
unit of work is a *clause*: Silero VAD (the ONNX model in `models/silero_vad.onnx`)
watches for ~450 ms pauses and cuts there, which turns out to be just enough
context to translate well.

Every stage runs as its own asyncio task, connected by small bounded queues.
That means while one clause is playing out loud, the next is mid-translation
and a third is still being transcribed — the stages overlap instead of adding
up.

```
pipeline/
  audio_in.py      mic or WAV file → 32 ms PCM frames
  vad.py           Silero VAD + clause segmenter (also drives barge-in)
  asr.py           Whisper-on-Groq or Gemini transcription, per clause
  translate.py     translation with rolling context + glossary
  tts.py           Edge, Gemini, or macOS say voices
  playback.py      speaker sink with instant barge-in cut, or WAV file sink
  orchestrator.py  wires it all together: queues, retries, latency logs
```

A few details that matter in practice:

- **Messy speech.** A small normalizer expands abbreviations ("ETA" → the full
  phrase) and tidies time tokens ("5pm" → "5 PM") before translation, so the
  model gets consistent input. The prompt handles the rest: fillers get
  dropped, proper nouns stay in Latin script, times get localized ("5 PM" →
  "शाम 5 बजे"). Domain terms you care about go in `glossary.json`, e.g.
  `{"Asterisk": "Asterisk"}`.
- **Flaky APIs.** Rate limits and 503s get a couple of retries with backoff;
  if that fails, the clause is dropped and the stream keeps moving. A live
  pipeline should degrade, not stall.
- **Talking over it.** If you speak while Hindi is playing, playback stops
  immediately, queued audio is flushed, and a short cooldown stops the tail
  end of the speaker output from being re-transcribed as a phantom clause.

## Getting started

```sh
make setup
echo 'GROQ_API_KEY=your-key' > .env
make run        # live mic mode — Ctrl+C to stop
```

Grab a free key at console.groq.com/keys. No mic handy? Test with a file:

```sh
make sample     # synthesizes test_input.wav with the macOS voice
make demo       # runs it through the pipeline and plays the Hindi
```

Or with your own words:

```sh
make sample TEXT="The demo is at 9 AM. Uh, please join on Google Meet."
```

`make test` runs the unit tests, `make clean` clears generated audio and logs.

## Backends

Two interchangeable engines for ASR + translation, picked with `--engine`
(the default `auto` uses Groq whenever `GROQ_API_KEY` is set):

- **groq** — Whisper large-v3-turbo for ASR, Llama 3.3 70B for translation.
  Fast, and the free tier (1000–2000 requests/day) is plenty for real testing.
  This is the one you want.
- **gemini** — `gemini-2.5-flash` for both. Works, but the free tier allows
  only ~20 requests/day per model, which live use burns through in minutes.

For TTS (`--tts`): `edge` is the default — Microsoft's free neural Hindi
voices, no key needed. `say` works fully offline using macOS's built-in Lekha
voice, `gemini` uses Gemini's native TTS (quota-limited), and `none` skips
audio entirely, which is the fastest way to iterate on translation quality.
Audio decoding goes through `soundfile`, so nothing here is macOS-only except
the `say` fallback and the Makefile's sample/demo helpers.

Model overrides, if you want to experiment: `GROQ_ASR_MODEL`, `GROQ_MT_MODEL`,
`EDGE_TTS_VOICE` (try `hi-IN-MadhurNeural` for a male voice); for the Gemini
engine, `ASR_MODEL`, `MT_MODEL`, `TTS_MODEL`, `TTS_VOICE`. Gemini calls run
with thinking disabled — it roughly halves the latency and transcription
doesn't need it.

## Running it directly

Everything the Makefile does, you can also do by hand:

```sh
.venv/bin/python main.py                                      # live mic
.venv/bin/python main.py --wav in.wav --out out.wav           # file mode
.venv/bin/python main.py --tts none                           # text only
.venv/bin/python main.py --wav in.wav --out out.wav --realtime  # pace file input like live speech
```

Input WAVs must be 16 kHz mono 16-bit (`make sample` produces exactly that).
