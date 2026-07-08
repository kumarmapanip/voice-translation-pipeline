import asyncio
import io
import threading
import wave


class SpeakerSink:
    """Feeds PCM straight to the speaker. stop_playback() clears the buffer
    on the spot, which is what makes barge-in feel instant."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = None

    def start(self):
        import sounddevice as sd

        def callback(outdata, frames, _time, _status):
            need = frames * 2
            with self._lock:
                chunk = bytes(self._buf[:need])
                del self._buf[:need]
            chunk += b"\x00" * (need - len(chunk))
            outdata[:] = chunk

        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", callback=callback
        )
        self._stream.start()

    @property
    def playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def feed(self, pcm: bytes):
        with self._lock:
            self._buf.extend(pcm)

    def stop_playback(self):
        with self._lock:
            self._buf.clear()

    async def drain(self):
        while self.playing:
            await asyncio.sleep(0.05)

    def close(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


class WavSink:
    """Collects PCM and writes it out as a WAV on close — for headless and
    file-mode runs where there's no speaker involved."""

    def __init__(self, path: str, sample_rate: int):
        self.path = path
        self.sample_rate = sample_rate
        self._buf = io.BytesIO()

    def start(self):
        pass

    @property
    def playing(self) -> bool:
        return False

    def feed(self, pcm: bytes):
        self._buf.write(pcm)

    def stop_playback(self):
        pass

    async def drain(self):
        pass

    def close(self):
        with wave.open(self.path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self._buf.getvalue())
