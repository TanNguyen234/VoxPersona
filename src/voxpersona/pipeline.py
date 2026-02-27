"""
VoxPersona pipeline — orchestrates STT, LLM and TTS components.

Supports three pipeline modes that determine which components are loaded:

* ``SPEECH_TO_SPEECH``: Audio → Whisper ASR → Qwen LLM → F5-TTS  (full voice loop)
      Audio source can be **microphone** (/mic, /listen) or **any audio file** (/audio).
* ``LLM_TTS``         : Qwen LLM → F5-TTS                (text in, voice out)
* ``LLM_ONLY``        : Qwen LLM                          (text in, text out)
* ``TEST_MIC``        : Whisper ASR only                  (audio → text, for testing STT)
* ``TEST_VOICE``      : F5-TTS only                       (text → audio, for testing TTS)

Components that are not required by the selected mode are **never loaded**,
keeping GPU memory usage minimal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from voxpersona.config import AppConfig, PipelineMode, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    user_text: str
    assistant_text: str
    tts_audio_path: Optional[str] = None


class VoxPersonaPipeline:
    """Central pipeline — loads only the components needed by *mode*."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mode = PipelineMode.from_str(config.pipeline_mode)

        logger.info("Pipeline mode: %s", self.mode.value)

        # ── Components (initialised lazily per mode) ────────────────
        self.recorder = None
        self.continuous_recorder = None
        self.asr = None
        self.chat = None
        self.tts = None

        self._init_components()

    # ── Component initialisation ────────────────────────────────────

    def _init_components(self) -> None:
        """Instantiate only the components required by ``self.mode``."""

        needs_asr = self.mode in (
            PipelineMode.SPEECH_TO_SPEECH,
            PipelineMode.TEST_MIC,
        )
        needs_mic = self.mode in (
            PipelineMode.SPEECH_TO_SPEECH,
            PipelineMode.TEST_MIC,
        )
        needs_llm = self.mode in (
            PipelineMode.SPEECH_TO_SPEECH,
            PipelineMode.LLM_TTS,
            PipelineMode.LLM_ONLY,
        )
        needs_tts = self.mode in (
            PipelineMode.SPEECH_TO_SPEECH,
            PipelineMode.LLM_TTS,
            PipelineMode.TEST_VOICE,
        )

        if needs_asr:
            self._init_asr()
        if needs_mic:
            self._init_mic()
        if needs_llm:
            self._init_llm()
        if needs_tts:
            self._init_tts()

        logger.info(
            "Components loaded — ASR=%s  MIC=%s  LLM=%s  TTS=%s",
            self.asr is not None,
            self.recorder is not None,
            self.chat is not None,
            self.tts is not None,
        )

    def _init_asr(self) -> None:
        """Load Whisper ASR — accepts audio from *any* source (mic or file)."""
        from voxpersona.models.whisper_asr import WhisperASR

        self.asr = WhisperASR(
            model_id=self.config.whisper_model_id,
            device=self.config.device,
            chunk_length_s=self.config.asr_chunk_length_s,
        )

    def _init_mic(self) -> None:
        """Load microphone recorders (fixed-duration + continuous VAD)."""
        from voxpersona.audio.recorder import ContinuousMicRecorder, MicrophoneRecorder

        self.recorder = MicrophoneRecorder(
            sample_rate=self.config.mic_sample_rate,
            channels=self.config.mic_channels,
        )
        self.continuous_recorder = ContinuousMicRecorder(
            sample_rate=self.config.mic_sample_rate,
            channels=self.config.mic_channels,
            energy_threshold=self.config.vad_energy_threshold,
            silence_duration_ms=self.config.vad_silence_duration_ms,
            min_speech_ms=self.config.vad_min_speech_ms,
            max_speech_s=self.config.vad_max_speech_s,
            output_dir="./outputs",
            prefix="utterance",
        )

    def _init_llm(self) -> None:
        from voxpersona.models.qwen_chat import QwenChat

        self.chat = QwenChat(
            model_id=self.config.qwen_model_id,
            system_prompt=SYSTEM_PROMPT,
            max_history=self.config.max_history,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )

    def _init_tts(self) -> None:
        if not self.config.f5_enabled:
            logger.warning(
                "Pipeline mode '%s' expects TTS but F5_ENABLED is false. "
                "TTS output will be skipped.",
                self.mode.value,
            )
            return

        from voxpersona.models.f5_tts import F5TTS

        self.tts = F5TTS(
            model=self.config.f5_model,
            vocoder_name=self.config.f5_vocoder,
            ckpt_file=self.config.f5_ckpt_file,
            vocab_file=self.config.f5_vocab_file,
            speed=self.config.f5_speed,
        )

    # ── Capability checks ───────────────────────────────────────────

    @property
    def has_asr(self) -> bool:
        """True when Whisper ASR is loaded (any audio → text)."""
        return self.asr is not None

    @property
    def has_mic(self) -> bool:
        """True when microphone recorders are available."""
        return self.recorder is not None

    @property
    def has_llm(self) -> bool:
        """True when Qwen LLM is loaded."""
        return self.chat is not None

    @property
    def has_tts(self) -> bool:
        return self.tts is not None

    # Backward compat alias
    has_stt = has_asr

    def _require_asr(self, action: str) -> None:
        if not self.has_asr:
            raise RuntimeError(
                f"Cannot {action}: ASR is not available in '{self.mode.value}' mode. "
                "Use 'speech_to_speech' mode for voice input."
            )

    def _require_mic(self, action: str) -> None:
        if not self.has_mic:
            raise RuntimeError(
                f"Cannot {action}: Microphone is not available in '{self.mode.value}' mode. "
                "Use 'speech_to_speech' mode for mic input."
            )

    def _require_llm(self, action: str) -> None:
        if not self.has_llm:
            raise RuntimeError(
                f"Cannot {action}: LLM is not available in '{self.mode.value}' mode."
            )

    def _require_tts(self, action: str) -> None:
        if not self.has_tts:
            raise RuntimeError(
                f"Cannot {action}: TTS is not available in '{self.mode.value}' mode. "
                "Ensure F5_ENABLED=true and use a pipeline with TTS."
            )

    # ── Single-turn helpers ─────────────────────────────────────────

    def run_text_turn(self, user_text: str) -> TurnResult:
        """Text → LLM → (optional) TTS.  Requires LLM."""
        self._require_llm("run text turn")
        assistant_text = self.chat.call_stream(user_text)
        tts_path = self._maybe_tts(assistant_text)
        return TurnResult(user_text=user_text, assistant_text=assistant_text, tts_audio_path=tts_path)

    def run_audio_turn(self, audio_path: str) -> TurnResult:
        """Audio file → STT → LLM → (optional) TTS.

        Accepts **any** audio file (wav, mp3, flac, …) that Whisper can
        decode.  The audio does not need to come from the microphone —
        it can be a pre-recorded file, an upload, or output from another
        pipeline.
        """
        self._require_asr("transcribe audio file")

        from pathlib import Path
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        user_text = self.asr.transcribe(str(path))
        assistant_text = self.chat.call_stream(user_text)
        tts_path = self._maybe_tts(assistant_text)
        return TurnResult(user_text=user_text, assistant_text=assistant_text, tts_audio_path=tts_path)

    def run_mic_turn(self, record_seconds: Optional[int] = None, output_wav: str = "./outputs/mic_input.wav") -> TurnResult:
        """Mic → STT → LLM → (optional) TTS.  Requires mic hardware."""
        self._require_mic("record from microphone")
        seconds = record_seconds or self.config.mic_record_seconds
        wav_path = self.recorder.record_to_wav(seconds=seconds, output_path=output_wav)
        user_text = self.asr.transcribe(wav_path)
        assistant_text = self.chat.call_stream(user_text)
        tts_path = self._maybe_tts(assistant_text)
        return TurnResult(user_text=user_text, assistant_text=assistant_text, tts_audio_path=tts_path)

    # ── Continuous listening mode ───────────────────────────────────

    def run_continuous_listen(
        self,
        on_utterance: Optional[Callable[[TurnResult], None]] = None,
    ) -> None:
        """
        Keep mic open, segment by VAD, feed each utterance through
        ASR → Chat → (optional) TTS.  Runs until ``stop_continuous()``
        is called or Ctrl-C.

        Requires ``SPEECH_TO_SPEECH`` mode.
        """
        self._require_mic("continuous listen")
        print("[Listen] Mic liên tục — nói bất cứ lúc nào.  Ctrl-C để dừng.")
        try:
            for wav_path in self.continuous_recorder.listen():
                user_text = self.asr.transcribe(wav_path)
                if not user_text.strip():
                    continue

                print(f"\n[ASR] {user_text}")
                assistant_text = self.chat.call_stream(user_text)
                tts_path = self._maybe_tts(assistant_text)

                result = TurnResult(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    tts_audio_path=tts_path,
                )
                if on_utterance:
                    on_utterance(result)

        except KeyboardInterrupt:
            self.stop_continuous()
            print("\n[Listen] Đã dừng.")

    def stop_continuous(self) -> None:
        """Stop the continuous recorder (no-op if recorder is not loaded)."""
        if self.continuous_recorder is not None:
            self.continuous_recorder.stop()

    # ── Test-only turns ─────────────────────────────────────────────

    def run_test_asr(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text.  No LLM, no TTS.

        Use this to quickly verify that Whisper is working or to benchmark
        transcription quality / latency.
        """
        self._require_asr("run test ASR")

        from pathlib import Path
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        return self.asr.transcribe(str(path))

    def run_test_mic_asr(
        self,
        record_seconds: Optional[int] = None,
        output_wav: str = "./outputs/test_mic.wav",
    ) -> str:
        """Record from mic then transcribe.  No LLM, no TTS."""
        self._require_mic("run test mic ASR")
        seconds = record_seconds or self.config.mic_record_seconds
        wav_path = self.recorder.record_to_wav(seconds=seconds, output_path=output_wav)
        return self.asr.transcribe(wav_path)

    def run_test_tts(self, text: str, output_path: Optional[str] = None) -> str:
        """Synthesize *text* with F5-TTS and return the output path.

        No ASR, no LLM.  Use this to test voice cloning quality or
        experiment with ref_audio / ref_text settings.
        """
        self._require_tts("run test TTS")
        if not self.config.f5_ref_text:
            raise ValueError("F5_REF_TEXT is required for TTS synthesis.")

        out = output_path or self.config.f5_output_path
        return self.tts.synthesize(
            ref_audio=self.config.f5_ref_audio,
            ref_text=self.config.f5_ref_text,
            gen_text=text,
            output_path=out,
        )

    # ── TTS helper ──────────────────────────────────────────────────

    def _maybe_tts(self, text: str) -> Optional[str]:
        if not self.tts:
            return None
        if not self.config.f5_ref_text:
            raise ValueError("F5_REF_TEXT is required when F5_ENABLED=true")

        return self.tts.synthesize(
            ref_audio=self.config.f5_ref_audio,
            ref_text=self.config.f5_ref_text,
            gen_text=text,
            output_path=self.config.f5_output_path,
        )
