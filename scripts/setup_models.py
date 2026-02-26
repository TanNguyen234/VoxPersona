"""
Setup script: clone external model repos and download weights.

Usage:
    python scripts/setup_models.py          # setup all
    python scripts/setup_models.py --f5     # F5-TTS only
    python scripts/setup_models.py --check  # verify installation
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"

# ── Registry of external models that require git clone ──────────────────────
MODELS = {
    "f5-tts": {
        "repo_url": "https://github.com/SWivid/F5-TTS.git",
        "clone_dir": VENDOR_DIR / "F5-TTS",
        "install_editable": True,
        "weights": {
            "hf_repo": "SWivid/F5-TTS",
            "hf_filename": "F5TTS_Base/model_1200000.pt",
            "local_path": VENDOR_DIR / "F5-TTS" / "ckpts" / "model_1200000.pt",
        },
    },
    "f5-tts-viet": {
        "repo_url": "https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git",
        "clone_dir": VENDOR_DIR / "F5-TTS-Vietnamese",
        "install_editable": True,
        "weights": {
            "hf_repo": "hynt/F5-TTS-Vietnamese-ViVoice",
            "hf_files": ["model_last.pt", "vocab.txt"],
            "local_dir": VENDOR_DIR / "F5-TTS-Vietnamese" / "ckpts",
        },
    },
}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] Already cloned: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)])


def install_editable(repo_dir: Path) -> None:
    _run([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)])


def download_hf_file(repo_id: str, filename: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"  [skip] Already downloaded: {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_path.parent),
        local_dir_use_symlinks=False,
    )
    print(f"  Downloaded: {local_path}")


def download_hf_files(repo_id: str, filenames: list[str], local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for fname in filenames:
        dest = local_dir / fname
        if dest.exists():
            print(f"  [skip] Already downloaded: {dest}")
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded: {dest}")


def setup_model(name: str) -> None:
    info = MODELS[name]
    print(f"\n{'='*60}")
    print(f"Setting up: {name}")
    print(f"{'='*60}")

    clone_repo(info["repo_url"], info["clone_dir"])

    if info.get("install_editable"):
        install_editable(info["clone_dir"])

    weights = info.get("weights")
    if weights:
        if "hf_filename" in weights:
            download_hf_file(weights["hf_repo"], weights["hf_filename"], weights["local_path"])
        elif "hf_files" in weights:
            download_hf_files(weights["hf_repo"], weights["hf_files"], weights["local_dir"])


def check_installation() -> None:
    print("\n=== Installation Check ===")
    all_ok = True
    for name, info in MODELS.items():
        clone_ok = info["clone_dir"].exists()
        weights = info.get("weights", {})
        if "local_path" in weights:
            weight_ok = weights["local_path"].exists()
        elif "local_dir" in weights:
            weight_ok = weights["local_dir"].exists() and any(weights["local_dir"].iterdir())
        else:
            weight_ok = True
        status = "OK" if (clone_ok and weight_ok) else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  [{status}] {name}: clone={'OK' if clone_ok else 'NO'}, weights={'OK' if weight_ok else 'NO'}")
    if all_ok:
        print("\nAll models ready.")
    else:
        print("\nSome models are missing. Run: python scripts/setup_models.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup external model repos and weights")
    parser.add_argument("--f5", action="store_true", help="Setup F5-TTS (original)")
    parser.add_argument("--f5-viet", action="store_true", help="Setup F5-TTS Vietnamese")
    parser.add_argument("--all", action="store_true", help="Setup all models (default)")
    parser.add_argument("--check", action="store_true", help="Check installation status only")
    args = parser.parse_args()

    if args.check:
        check_installation()
        return

    selected = []
    if args.f5:
        selected.append("f5-tts")
    if args.f5_viet:
        selected.append("f5-tts-viet")
    if args.all or not selected:
        selected = list(MODELS.keys())

    for name in selected:
        setup_model(name)

    print("\n=== Setup Complete ===")
    check_installation()


if __name__ == "__main__":
    main()
