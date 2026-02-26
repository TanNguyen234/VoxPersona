import argparse

from voxpersona.config import AppConfig
from voxpersona.pipeline import VoxPersonaPipeline
from voxpersona.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VoxPersona CLI (Mic ASR + Qwen chat + optional F5-TTS)")
    parser.add_argument("--mode", choices=["interactive", "text", "mic", "listen"], default="interactive")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--record-seconds", type=int, default=None)
    return parser


def _on_utterance(result) -> None:
    """Callback for continuous-listen mode — prints TTS info if any."""
    if result.tts_audio_path:
        print(f"\n[TTS] Saved: {result.tts_audio_path}")
    print()


def run_interactive(pipeline: VoxPersonaPipeline, default_record_seconds: int) -> None:
    print("=== VoxPersona Interactive ===")
    print("- /mic       : ghi âm cố định (1 lần)")
    print("- /listen    : bật mic liên tục (VAD), Ctrl-C để dừng")
    print("- /exit      : thoát")

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            break

        if user_input == "/mic":
            print(f"[Mic] Đang ghi âm {default_record_seconds}s...")
            result = pipeline.run_mic_turn(record_seconds=default_record_seconds)
            print(f"\n[ASR] {result.user_text}")
            if result.tts_audio_path:
                print(f"\n[TTS] Saved: {result.tts_audio_path}")
            print()
            continue

        if user_input == "/listen":
            pipeline.run_continuous_listen(on_utterance=_on_utterance)
            continue

        result = pipeline.run_text_turn(user_input)
        if result.tts_audio_path:
            print(f"\n[TTS] Saved: {result.tts_audio_path}")
        print()


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = AppConfig()
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
