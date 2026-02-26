import argparse
import json
from pathlib import Path
from typing import Dict, List

KEYWORDS: Dict[str, List[str]] = {
    "whisper": ["whisper", "AutoModelForSpeechSeq2Seq", "automatic-speech-recognition"],
    "qwen": ["qwen", "AutoModelForCausalLM", "TextIteratorStreamer"],
    "f5": ["f5", "f5-tts", "infer_process", "f5-tts_infer-cli"],
}


def collect_sources(ipynb_path: Path) -> Dict[str, List[str]]:
    notebook = json.loads(ipynb_path.read_text(encoding="utf-8"))
    outputs: Dict[str, List[str]] = {"whisper": [], "qwen": [], "f5": []}

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        if not source_lines:
            continue

        cell_text = "\n".join(source_lines)
        lowered = cell_text.lower()

        for section, patterns in KEYWORDS.items():
            if any(pattern.lower() in lowered for pattern in patterns):
                outputs[section].append(cell_text)

    return outputs


def write_outputs(extracted: Dict[str, List[str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for section, blocks in extracted.items():
        destination = out_dir / f"{section}_from_notebook.py"
        content = "\n\n".join(blocks) if blocks else "# No matching code blocks found.\n"
        destination.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Whisper/Qwen/F5 code blocks from a notebook")
    parser.add_argument("--input", required=True, help="Path to .ipynb file")
    parser.add_argument("--output-dir", default="./notebook_extracts", help="Folder to write extracted .py files")
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output_dir)

    extracted = collect_sources(source)
    write_outputs(extracted, out_dir)

    print(f"Extraction completed: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
