from dataclasses import dataclass
from typing import Callable, Optional

from voxpersona.audio.recorder import ContinuousMicRecorder, MicrophoneRecorder
from voxpersona.config import AppConfig, SYSTEM_PROMPT
from voxpersona.models.f5_tts import F5TTS
from voxpersona.models.qwen_chat import QwenChat
from voxpersona.models.whisper_asr import WhisperASR


@dataclass
class TurnResult:
    user_text: str
    assistant_text: str
    tts_audio_path: Optional[str] = None


class VoxPersonaPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # Legacy fixed-duration recorder (for /mic)
        self.recorder = MicrophoneRecorder(
            sample_rate=config.mic_sample_rate,
            channels=config.mic_channels,
        )

        # Continuous VAD recorder (for /listen)
        self.continuous_recorder = ContinuousMicRecorder(
            sample_rate=config.mic_sample_rate,
            channels=config.mic_channels,
            energy_threshold=config.vad_energy_threshold,
            silence_duration_ms=config.vad_silence_duration_ms,
            min_speech_ms=config.vad_min_speech_ms,
            max_speech_s=config.vad_max_speech_s,
            output_dir="./outputs",
            prefix="utterance",
        )

        self.asr = WhisperASR(
            model_id=config.whisper_model_id,
            device=config.device,
            chunk_length_s=config.asr_chunk_length_s,
        )
        self.chat = QwenChat(
            model_id=config.qwen_model_id,
            system_prompt=SYSTEM_PROMPT,
            max_history=config.max_history,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        self.tts: Optional[F5TTS] = None
        if config.f5_enabled:
            self.tts = F5TTS(
                model=config.f5_model,
                vocoder_name=config.f5_vocoder,
                ckpt_file=config.f5_ckpt_file,
                vocab_file=config.f5_vocab_file,
                speed=config.f5_speed,
            )

    # ── Single-turn helpers ─────────────────────────────────────────

    def run_text_turn(self, user_text: str) -> TurnResult:
        assistant_text = self.chat.call_stream(user_text)
        tts_path = self._maybe_tts(assistant_text)
        return TurnResult(user_text=user_text, assistant_text=assistant_text, tts_audio_path=tts_path)

    def run_mic_turn(self, record_seconds: Optional[int] = None, output_wav: str = "./outputs/mic_input.wav") -> TurnResult:
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

        ``on_utterance`` is a callback fired after each turn so the
        caller (CLI) can print / play audio.
        """
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
        """Stop the continuous recorder."""
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
