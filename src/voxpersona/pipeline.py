"""
VoxPersona pipeline — orchestrates STT, LLM and TTS components.

Supports three pipeline modes that determine which components are loaded:

* ``SPEECH_TO_SPEECH``: Whisper ASR → Qwen LLM → F5-TTS  (full voice loop)
* ``LLM_TTS``         : Qwen LLM → F5-TTS                (text in, voice out)
* ``LLM_ONLY``        : Qwen LLM                          (text in, text out)

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

        needs_stt = self.mode is PipelineMode.SPEECH_TO_SPEECH
        needs_tts = self.mode in (PipelineMode.SPEECH_TO_SPEECH, PipelineMode.LLM_TTS)

        # -- STT + mic recorders (only for speech-to-speech) --
        if needs_stt:
            self._init_stt()

        # -- LLM (always needed) --
        self._init_llm()

        # -- TTS (speech-to-speech & llm_tts) --
        if needs_tts:
            self._init_tts()

        logger.info(
            "Components loaded — STT=%s  LLM=True  TTS=%s",
            needs_stt,
            self.tts is not None,
        )

    def _init_stt(self) -> None:
        from voxpersona.audio.recorder import ContinuousMicRecorder, MicrophoneRecorder
        from voxpersona.models.whisper_asr import WhisperASR

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
        self.asr = WhisperASR(
            model_id=self.config.whisper_model_id,
            device=self.config.device,
            chunk_length_s=self.config.asr_chunk_length_s,
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
    def has_stt(self) -> bool:
        return self.asr is not None

    @property
    def has_tts(self) -> bool:
        return self.tts is not None

    def _require_stt(self, action: str) -> None:
        if not self.has_stt:
            raise RuntimeError(
                f"Cannot {action}: STT is not available in '{self.mode.value}' mode. "
                "Use 'speech_to_speech' mode for voice input."
            )

    # ── Single-turn helpers ─────────────────────────────────────────

    def run_text_turn(self, user_text: str) -> TurnResult:
        """Text → LLM → (optional) TTS.  Available in ALL modes."""
        assistant_text = self.chat.call_stream(user_text)
        tts_path = self._maybe_tts(assistant_text)
        return TurnResult(user_text=user_text, assistant_text=assistant_text, tts_audio_path=tts_path)

    def run_mic_turn(self, record_seconds: Optional[int] = None, output_wav: str = "./outputs/mic_input.wav") -> TurnResult:
        """Mic → STT → LLM → (optional) TTS.  Requires SPEECH_TO_SPEECH mode."""
        self._require_stt("record from microphone")
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
        self._require_stt("continuous listen")
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
