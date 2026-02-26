"""
Microphone recording utilities.

* ``MicrophoneRecorder``  – simple fixed-duration capture (legacy).
* ``ContinuousMicRecorder`` – energy-based VAD with continuous capture.
  Mic stays open, silence gaps split the audio into utterances, and
  each utterance is yielded as a WAV file path.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple fixed-duration recorder (kept for backward compat / one-shot mode)
# ---------------------------------------------------------------------------

class MicrophoneRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def record_to_wav(self, seconds: int, output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        frames = int(seconds * self.sample_rate)
        audio = sd.rec(frames, samplerate=self.sample_rate, channels=self.channels, dtype="float32")
        sd.wait()

        sf.write(output, audio, self.sample_rate)
        return str(output)


# ---------------------------------------------------------------------------
# Continuous recorder with energy-based Voice Activity Detection
# ---------------------------------------------------------------------------

class ContinuousMicRecorder:
    """
    Keeps the microphone open indefinitely.  Audio is split into
    *utterances* using energy-based silence detection:
      1. Wait until energy exceeds ``energy_threshold`` (speech start).
      2. Keep recording while energy stays above threshold.
      3. When energy drops below threshold for ``silence_duration_ms``,
         treat the chunk as one complete utterance.
      4. If the utterance is shorter than ``min_speech_ms`` it is
         discarded (cough / click noise).  Otherwise it is written as
         a WAV file and yielded to the caller.

    Use ``listen()`` as a generator — each iteration yields the file
    path of one complete utterance WAV.  Calling ``stop()`` (or
    pressing Ctrl-C) terminates the stream.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        energy_threshold: float = 0.02,
        silence_duration_ms: int = 800,
        min_speech_ms: int = 400,
        max_speech_s: float = 30.0,
        output_dir: str = "./outputs",
        prefix: str = "utterance",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.energy_threshold = energy_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_s = max_speech_s
        self.output_dir = Path(output_dir)
        self.prefix = prefix

        self._stop_event = threading.Event()
        self._utterance_counter = 0

    # ── public API ──────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the recorder to stop after the current utterance."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def listen(self) -> Generator[str, None, None]:
        """
        Generator that yields WAV file paths — one per detected
        utterance.  Blocks between utterances.

        Usage::

            recorder = ContinuousMicRecorder()
            for wav_path in recorder.listen():
                text = asr.transcribe(wav_path)
                ...
        """
        self._stop_event.clear()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        block_duration_ms = 30  # process audio in 30 ms blocks
        block_size = int(self.sample_rate * block_duration_ms / 1000)
        silence_blocks_needed = int(self.silence_duration_ms / block_duration_ms)
        min_speech_blocks = int(self.min_speech_ms / block_duration_ms)
        max_speech_blocks = int(self.max_speech_s * 1000 / block_duration_ms)

        # Shared state between callback and main thread
        audio_queue: deque[np.ndarray] = deque()
        queue_lock = threading.Lock()

        def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
            with queue_lock:
                audio_queue.append(indata.copy())

        logger.info(
            "Continuous mic started  sr=%d  threshold=%.4f  "
            "silence=%dms  min_speech=%dms",
            self.sample_rate, self.energy_threshold,
            self.silence_duration_ms, self.min_speech_ms,
        )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=block_size,
            callback=_audio_callback,
        ):
            while not self._stop_event.is_set():
                utterance = self._collect_one_utterance(
                    audio_queue, queue_lock,
                    silence_blocks_needed, min_speech_blocks,
                    max_speech_blocks,
                )
                if utterance is None:
                    # stop was called while waiting
                    break

                wav_path = self._save_utterance(utterance)
                if wav_path:
                    yield wav_path

        logger.info("Continuous mic stopped.")

    # ── internals ───────────────────────────────────────────────────

    def _rms(self, block: np.ndarray) -> float:
        """Root-mean-square energy of an audio block."""
        return float(np.sqrt(np.mean(block ** 2)))

    def _collect_one_utterance(
        self,
        audio_queue: deque,
        lock: threading.Lock,
        silence_blocks_needed: int,
        min_speech_blocks: int,
        max_speech_blocks: int,
    ) -> Optional[np.ndarray]:
        """
        Block until one complete utterance is detected, then return
        it as a numpy array.  Returns ``None`` if ``stop()`` is called.
        """
        speech_chunks: list[np.ndarray] = []
        is_speaking = False
        silent_blocks = 0
        speech_blocks = 0

        while not self._stop_event.is_set():
            # Drain available blocks
            blocks: list[np.ndarray] = []
            with lock:
                while audio_queue:
                    blocks.append(audio_queue.popleft())

            if not blocks:
                time.sleep(0.01)
                continue

            for block in blocks:
                energy = self._rms(block)

                if not is_speaking:
                    if energy >= self.energy_threshold:
                        is_speaking = True
                        silent_blocks = 0
                        speech_blocks = 1
                        speech_chunks.append(block)
                        logger.debug("Speech start detected (energy=%.4f)", energy)
                else:
                    speech_chunks.append(block)
                    speech_blocks += 1

                    if energy < self.energy_threshold:
                        silent_blocks += 1
                    else:
                        silent_blocks = 0

                    # Utterance ended by silence
                    if silent_blocks >= silence_blocks_needed:
                        if speech_blocks >= min_speech_blocks:
                            logger.debug(
                                "Utterance complete: %d blocks (%.1fs)",
                                speech_blocks,
                                speech_blocks * 0.03,
                            )
                            return np.concatenate(speech_chunks, axis=0)
                        else:
                            # Too short — discard (noise / click)
                            logger.debug("Discarded short segment (%d blocks)", speech_blocks)
                            speech_chunks.clear()
                            is_speaking = False
                            silent_blocks = 0
                            speech_blocks = 0

                    # Safety cap — avoid unbounded recording
                    if speech_blocks >= max_speech_blocks:
                        logger.warning("Max speech length reached (%ds)", int(self.max_speech_s))
                        return np.concatenate(speech_chunks, axis=0)

        return None

    def _save_utterance(self, audio: np.ndarray) -> Optional[str]:
        """Write a numpy audio array to WAV and return the path."""
        self._utterance_counter += 1
        filename = f"{self.prefix}_{self._utterance_counter:04d}.wav"
        path = self.output_dir / filename
        sf.write(str(path), audio, self.sample_rate)
        logger.info("Saved utterance: %s (%.1fs)", path, len(audio) / self.sample_rate)
        return str(path)
