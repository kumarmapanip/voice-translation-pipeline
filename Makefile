VENV := .venv
PY := $(VENV)/bin/python
VAD_MODEL := models/silero_vad.onnx
VAD_URL := https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# What you'd say into the mic, for `make sample`. Override like:
#   make sample TEXT="The demo is at 9 AM on Google Meet."
TEXT := Can we move the meeting to 5 PM tomorrow? [[slnc 700]] Also, uh, can you send the Google Meet link?

.PHONY: setup run demo sample test lint clean

## setup: create the venv, install deps, fetch the VAD model
setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	@mkdir -p models
	@test -f $(VAD_MODEL) || curl -sL -o $(VAD_MODEL) $(VAD_URL)
	@test -f .env || echo "Don't forget: put GROQ_API_KEY=... in .env"

## run: live mode — speak into the mic, Hindi comes out the speaker
run:
	$(PY) main.py

## demo: run the pipeline over test_input.wav and play the result
demo:
	$(PY) main.py --wav test_input.wav --out out_hindi.wav
	afplay out_hindi.wav

## sample: record a fresh test_input.wav from TEXT using the macOS voice
sample:
	say -v Samantha -o test_input.wav --data-format=LEI16@16000 "$(TEXT)"
	@echo "test_input.wav ready — try: make demo"

## test: run the unit tests (segmenter state machine, normalizer)
test:
	$(PY) -m unittest discover tests

## clean: remove generated audio and logs
clean:
	rm -f out_*.wav pipeline.log
