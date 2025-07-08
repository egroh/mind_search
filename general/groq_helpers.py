from pathlib import Path
import os, tempfile
# import sounddevice as sd, soundfile as sf

import numpy as np
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# env + client
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
_client = Groq()  # picks up GROQ_API_KEY

CHAT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
STT_MODEL = os.getenv("GROQ_STT_MODEL", "distil-whisper-large-v3-en")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def chat_groq(
    messages: list[dict], temperature: float = 0.7, max_tokens: int = 512
) -> str:
    """
    messages – OpenAI-style list[{"role": "...", "content": "..."}]
    returns assistant reply text
    """
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------
def transcribe_wav(path: str) -> str:
    """
    Send an existing WAV file to Groq Whisper and return plain text.

    Note: with response_format='text' the SDK already returns a bare string,
    *not* an object with .text – that caused your previous AttributeError.
    """
    with open(path, "rb") as f:
        resp = _client.audio.transcriptions.create(
            file=f,
            model=STT_MODEL,
            response_format="text",
            language="en",
        )
    # `resp` IS the text because of response_format="text"
    return resp.strip()


# --------------------------------------------------------------------------
# Audio recorder — fixed for universal PySoundFile
# --------------------------------------------------------------------------
class Recorder:
    """
    Collects mono int16 audio frames.  start() begins capture;
    stop_and_save() stops and returns a temp WAV path.
    """

    def __init__(self, fs: int = 16_000):
        self.fs = fs
        self._frames: list[np.ndarray] = []
        self._stream = None

    def _callback(self, in_data, _frames, _time, _status):
        if _status:
            print(_status)
        # in_data is already a NumPy array; copy to detach from sounddevice
        self._frames.append(in_data.copy())

    def start(self):
        self._frames.clear()
        # self._stream = sd.InputStream(
        #     samplerate=self.fs,
        #     channels=1,
        #     dtype="int16",
        #     callback=self._callback,
        # )
        self._stream.start()

    def stop_and_save(self) -> str:
        """Stops recording and returns path to a temp WAV file."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:  # safety net
            raise RuntimeError("No audio captured")

        audio = np.concatenate(self._frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#        sf.write(tmp.name, audio, self.fs, subtype="PCM_16")  # single call
        return tmp.name
