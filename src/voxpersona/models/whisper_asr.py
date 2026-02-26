from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperASR:
    def __init__(self, model_id: str, device: str = "auto", chunk_length_s: int = 30) -> None:
        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32

        if device == "auto":
            resolved_device = "cuda:0" if use_cuda else "cpu"
        else:
            resolved_device = device

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(resolved_device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=resolved_device,
        )

    def transcribe(self, audio_path: str, language: Optional[str] = "vi") -> str:
        result = self.pipe(audio_path, generate_kwargs={"language": language} if language else None)
        return result["text"].strip()
