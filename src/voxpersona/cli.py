"""
VoxPersona CLI entry-point.

Supports three pipeline modes selected via ``--pipeline``:

* ``speech_to_speech`` – full voice loop  (STT → LLM → TTS)
* ``llm_tts``          – text in, voice out (LLM → TTS)
* ``llm_only``         – text in, text out  (LLM)

The ``--mode`` flag controls the *interaction style* (interactive REPL,
single-shot text, single-shot mic, continuous listen) and is orthogonal
to the pipeline mode — though mic / listen are only valid when the
pipeline includes STT.
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
            "llm_tts (LLM+TTS), llm_only (LLM). "
            "Overrides PIPELINE_MODE env var."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "text", "mic", "listen"],
        default="interactive",
        help="Interaction mode.",
    )
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--record-seconds", type=int, default=None)
    return parser


# ── Callbacks ───────────────────────────────────────────────────────

def _on_utterance(result) -> None:
    """Callback for continuous-listen mode."""
    if result.tts_audio_path:
        print(f"\n[TTS] Saved: {result.tts_audio_path}")
    print()


# ── Interactive REPL ────────────────────────────────────────────────

def _print_interactive_banner(pipeline: VoxPersonaPipeline) -> None:
    """Show available commands based on active pipeline mode."""
    mode = pipeline.mode
    print(f"\n=== VoxPersona Interactive  [{mode.value}] ===")

    if pipeline.has_stt:
        print("  /mic       : ghi âm cố định (1 lần)")
        print("  /listen    : bật mic liên tục (VAD), Ctrl-C để dừng")

    if pipeline.has_tts:
        print("  (text)     : gõ văn bản → LLM → TTS")
    else:
        print("  (text)     : gõ văn bản → LLM")

    print("  /mode      : hiển thị pipeline mode hiện tại")
    print("  /exit      : thoát")


def run_interactive(pipeline: VoxPersonaPipeline, default_record_seconds: int) -> None:
    _print_interactive_banner(pipeline)

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/mode":
            stt_label = "ON" if pipeline.has_stt else "OFF"
            tts_label = "ON" if pipeline.has_tts else "OFF"
            print(f"[Mode] {pipeline.mode.value}  (STT={stt_label}  LLM=ON  TTS={tts_label})")
            continue

        if user_input == "/mic":
            if not pipeline.has_stt:
                print("[!] /mic không khả dụng trong mode này. Cần pipeline 'speech_to_speech'.")
                continue
            print(f"[Mic] Đang ghi âm {default_record_seconds}s...")
            result = pipeline.run_mic_turn(record_seconds=default_record_seconds)
            print(f"\n[ASR] {result.user_text}")
            if result.tts_audio_path:
                print(f"\n[TTS] Saved: {result.tts_audio_path}")
            print()
            continue

        if user_input == "/listen":
            if not pipeline.has_stt:
                print("[!] /listen không khả dụng trong mode này. Cần pipeline 'speech_to_speech'.")
                continue
            pipeline.run_continuous_listen(on_utterance=_on_utterance)
            continue

        # Default: text → LLM → (optional) TTS
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
        # Replace pipeline_mode in the frozen dataclass via reconstruction
        config = AppConfig(pipeline_mode=args.pipeline)

    pipeline = VoxPersonaPipeline(config)

    if args.mode == "interactive":
        run_interactive(pipeline, default_record_seconds=args.record_seconds or config.mic_record_seconds)
        return

    if args.mode == "text":
        if not args.text.strip():
            raise ValueError("--text is required when --mode text")
        result = pipeline.run_text_turn(args.text.strip())
        print(f"\nAssistant: {result.assistant_text}")
        if result.tts_audio_path:
            print(f"TTS: {result.tts_audio_path}")
        return

    if args.mode == "listen":
        pipeline.run_continuous_listen(on_utterance=_on_utterance)
        return

    # mode == "mic" (single shot)
    result = pipeline.run_mic_turn(record_seconds=args.record_seconds or config.mic_record_seconds)
    print(f"ASR: {result.user_text}")
    print(f"Assistant: {result.assistant_text}")
    if result.tts_audio_path:
        print(f"TTS: {result.tts_audio_path}")


if __name__ == "__main__":
    main()
