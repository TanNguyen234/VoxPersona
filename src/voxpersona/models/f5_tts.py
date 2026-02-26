"""
F5-TTS adapter — calls ``f5-tts_infer-cli`` via subprocess.

The CLI is available after ``pip install -e vendor/F5-TTS`` (or the
Vietnamese fork).  The Python API in the notebook was unreliable, so
this module delegates entirely to the CLI which is known to work.

Requires:
    python scripts/setup_models.py   # clone repos + install editable
"""
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CLI_NAME = "f5-tts_infer-cli"


class F5TTS:
    """Wrapper around the f5-tts_infer-cli command."""

    def __init__(
        self,
        model: str = "F5TTS_Base",
        vocoder_name: str = "vocos",
        ckpt_file: str = "",
        vocab_file: str = "",
        speed: float = 1.0,
    ) -> None:
        if not shutil.which(_CLI_NAME):
            raise FileNotFoundError(
                f"'{_CLI_NAME}' not found on PATH.  "
                "Run: python scripts/setup_models.py"
            )

        self.model = model
        self.vocoder_name = vocoder_name
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.speed = speed

        logger.info(
            "F5-TTS CLI ready  model=%s  vocoder=%s  ckpt=%s",
            model, vocoder_name, ckpt_file or "(default)",
        )

    def synthesize(
        self,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        output_path: str,
    ) -> str:
        """Synthesize *gen_text* using the voice from *ref_audio*."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            _CLI_NAME,
            "--model", self.model,
            "--ref_audio", str(ref_audio),
            "--ref_text", ref_text,
            "--gen_text", gen_text,
            "--speed", str(self.speed),
            "--vocoder_name", self.vocoder_name,
        ]

        if self.ckpt_file:
            cmd += ["--ckpt_file", str(self.ckpt_file)]
        if self.vocab_file:
            cmd += ["--vocab_file", str(self.vocab_file)]

        # CLI writes to a default location; we specify output_dir + output_file
        cmd += ["--output_dir", str(output.parent)]

        logger.info("F5-TTS CLI: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("F5-TTS CLI stderr:\n%s", result.stderr)
            raise RuntimeError(f"f5-tts_infer-cli failed (rc={result.returncode})")

        # CLI typically writes infer_cli_out.wav in the output_dir
        default_out = output.parent / "infer_cli_out.wav"
        if default_out.exists() and default_out != output:
            default_out.rename(output)

        logger.info("TTS output saved: %s", output)
        return str(output)
