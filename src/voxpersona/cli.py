"""
VoxPersona CLI entry-point.

Supports three pipeline modes selected via ``--pipeline``:

* ``speech_to_speech`` – full voice loop  (Audio → STT → LLM → TTS)
      Audio input can be **microphone** (/mic, /listen) or **any audio file** (/audio).
* ``llm_tts``          – text in, voice out (LLM → TTS)
* ``llm_only``         – text in, text out  (LLM)
* ``test_mic``         – STT only (mic/audio → Whisper → print text)
* ``test_voice``       – TTS only (text → F5-TTS → audio file)

The ``--mode`` flag controls the *interaction style* (interactive REPL,
single-shot text, single-shot audio file, single-shot mic, continuous listen)
and is orthogonal to the pipeline mode — though audio / mic / listen require
the pipeline to include ASR.
"""

from __future__ import annotations

import argparse

from voxpersona.config import AppConfig, PipelineMode
from voxpersona.pipeline import VoxPersonaPipeline
from voxpersona.utils.logging import setup_logging

_PIPELINE_CHOICES = [m.value for m in PipelineMode]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoxPersona CLI — multi-mode AI voice / text assistant",
    )
    parser.add_argument(
        "--pipeline",
        choices=_PIPELINE_CHOICES,
        default=None,
        help=(
            "Pipeline mode: speech_to_speech (STT+LLM+TTS), "
            "llm_tts (LLM+TTS), llm_only (LLM), "
            "test_mic (STT only), test_voice (TTS only). "
            "Overrides PIPELINE_MODE env var."
        ),
    )
    parser.add_argument(
        "--tts-text",
        type=str,
        default="",
        help="Text to synthesize in test_voice mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "text", "audio", "mic", "listen"],
        default="interactive",
        help="Interaction mode.",
    )
    parser.add_argument("--text", type=str, default="")
    parser.add_argument(
        "--audio",
        type=str,
        default="",
        help="Path to an audio file for --mode audio (wav, mp3, flac, …).",
    )
    parser.add_argument("--record-seconds", type=int, default=None)
    return parser


# ── Callbacks ───────────────────────────────────────────────────────

def _on_utterance(result) -> None:
    """Callback for continuous-listen mode."""
    if result.tts_audio_path:
        print(f"\n[TTS] Saved: {result.tts_audio_path}")
    print()


def _run_test_mic_continuous(pipeline: VoxPersonaPipeline) -> None:
    """Continuous mic → Whisper only (no LLM, no TTS).  Print transcriptions."""
    print("[Test-Mic] Mic liên tục — nói bất cứ lúc nào.  Ctrl-C để dừng.")
    try:
        for wav_path in pipeline.continuous_recorder.listen():
            text = pipeline.asr.transcribe(wav_path)
            if not text.strip():
                continue
            print(f"\n[ASR] {text}\n")
    except KeyboardInterrupt:
        pipeline.stop_continuous()
        print("\n[Test-Mic] Đã dừng.")


# ── Interactive REPL ────────────────────────────────────────────────

def _print_interactive_banner(pipeline: VoxPersonaPipeline) -> None:
    """Show available commands based on active pipeline mode."""
    mode = pipeline.mode
    print(f"\n=== VoxPersona Interactive  [{mode.value}] ===")

    if mode is PipelineMode.TEST_MIC:
        print("  /audio <path> : transcribe audio → print text")
        print("  /mic          : ghi âm → transcribe → print text")
        print("  /listen       : mic liên tục → transcribe mỗi câu")
    elif mode is PipelineMode.TEST_VOICE:
        print("  (text)        : gõ văn bản → F5-TTS → lưu audio")
    else:
        if pipeline.has_asr:
            print("  /audio <path> : transcribe audio file → LLM → (TTS)")
            if pipeline.has_mic:
                print("  /mic          : ghi âm cố định (1 lần)")
                print("  /listen       : bật mic liên tục (VAD), Ctrl-C để dừng")

        if pipeline.has_tts:
            print("  (text)        : gõ văn bản → LLM → TTS")
        else:
            print("  (text)        : gõ văn bản → LLM")

    print("  /mode         : hiển thị pipeline mode hiện tại")
    print("  /exit         : thoát")


def run_interactive(pipeline: VoxPersonaPipeline, default_record_seconds: int) -> None:
    _print_interactive_banner(pipeline)

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/mode":
            asr_label = "ON" if pipeline.has_asr else "OFF"
            mic_label = "ON" if pipeline.has_mic else "OFF"
            llm_label = "ON" if pipeline.has_llm else "OFF"
            tts_label = "ON" if pipeline.has_tts else "OFF"
            print(f"[Mode] {pipeline.mode.value}  (ASR={asr_label}  MIC={mic_label}  LLM={llm_label}  TTS={tts_label})")
            continue

        # /audio <path>  — transcribe arbitrary audio file
        if user_input.startswith("/audio"):
            if not pipeline.has_asr:
                print("[!] /audio không khả dụng trong mode này.")
                continue
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print("[!] Cú pháp: /audio <đường_dẫn_file>")
                continue
            audio_path = parts[1].strip().strip('"').strip("'")
            try:
                if pipeline.mode is PipelineMode.TEST_MIC:
                    text = pipeline.run_test_asr(audio_path)
                    print(f"\n[ASR] {text}")
                else:
                    result = pipeline.run_audio_turn(audio_path)
                    print(f"\n[ASR] {result.user_text}")
                    if result.tts_audio_path:
                        print(f"\n[TTS] Saved: {result.tts_audio_path}")
            except FileNotFoundError as e:
                print(f"[!] {e}")
            print()
            continue

        if user_input == "/mic":
            if not pipeline.has_mic:
                print("[!] /mic không khả dụng trong mode này.")
                continue
            print(f"[Mic] Đang ghi âm {default_record_seconds}s...")
            if pipeline.mode is PipelineMode.TEST_MIC:
                text = pipeline.run_test_mic_asr(record_seconds=default_record_seconds)
                print(f"\n[ASR] {text}")
            else:
                result = pipeline.run_mic_turn(record_seconds=default_record_seconds)
                print(f"\n[ASR] {result.user_text}")
                if result.tts_audio_path:
                    print(f"\n[TTS] Saved: {result.tts_audio_path}")
            print()
            continue

        if user_input == "/listen":
            if not pipeline.has_mic:
                print("[!] /listen không khả dụng trong mode này.")
                continue
            if pipeline.mode is PipelineMode.TEST_MIC:
                _run_test_mic_continuous(pipeline)
            else:
                pipeline.run_continuous_listen(on_utterance=_on_utterance)
            continue

        # Default: text input
        if pipeline.mode is PipelineMode.TEST_VOICE:
            try:
                out_path = pipeline.run_test_tts(user_input)
                print(f"\n[TTS] Saved: {out_path}")
            except ValueError as e:
                print(f"[!] {e}")
            print()
            continue

        if not pipeline.has_llm:
            print("[!] LLM không khả dụng trong mode này.")
            continue

        result = pipeline.run_text_turn(user_input)
        if result.tts_audio_path:
            print(f"\n[TTS] Saved: {result.tts_audio_path}")
        print()


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    # CLI flag --pipeline overrides the env-based config default
    config = AppConfig()
    if args.pipeline:
        config = AppConfig(pipeline_mode=args.pipeline)

    pipeline = VoxPersonaPipeline(config)

    if args.mode == "interactive":
        run_interactive(pipeline, default_record_seconds=args.record_seconds or config.mic_record_seconds)
        return

    if args.mode == "text":
        # test_voice: text → TTS only (no LLM)
        if pipeline.mode is PipelineMode.TEST_VOICE:
            if not args.tts_text.strip() and not args.text.strip():
                raise ValueError("--text or --tts-text is required for test_voice mode")
            synth_text = args.tts_text.strip() or args.text.strip()
            out = pipeline.run_test_tts(synth_text)
            print(f"TTS: {out}")
            return

        if not args.text.strip():
            raise ValueError("--text is required when --mode text")
        result = pipeline.run_text_turn(args.text.strip())
        print(f"\nAssistant: {result.assistant_text}")
        if result.tts_audio_path:
            print(f"TTS: {result.tts_audio_path}")
        return

    if args.mode == "audio":
        if not args.audio.strip():
            raise ValueError("--audio <path> is required when --mode audio")
        if pipeline.mode is PipelineMode.TEST_MIC:
            text = pipeline.run_test_asr(args.audio.strip())
            print(f"ASR: {text}")
        else:
            result = pipeline.run_audio_turn(args.audio.strip())
            print(f"ASR: {result.user_text}")
            print(f"Assistant: {result.assistant_text}")
            if result.tts_audio_path:
                print(f"TTS: {result.tts_audio_path}")
        return

    if args.mode == "listen":
        if pipeline.mode is PipelineMode.TEST_MIC:
            _run_test_mic_continuous(pipeline)
        else:
            pipeline.run_continuous_listen(on_utterance=_on_utterance)
        return

    if args.mode == "mic":
        secs = args.record_seconds or config.mic_record_seconds
        if pipeline.mode is PipelineMode.TEST_MIC:
            text = pipeline.run_test_mic_asr(record_seconds=secs)
            print(f"ASR: {text}")
        else:
            result = pipeline.run_mic_turn(record_seconds=secs)
            print(f"ASR: {result.user_text}")
            print(f"Assistant: {result.assistant_text}")
            if result.tts_audio_path:
                print(f"TTS: {result.tts_audio_path}")
        return


if __name__ == "__main__":
    main()
