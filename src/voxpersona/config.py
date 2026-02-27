from dataclasses import dataclass
from enum import Enum
import os

from dotenv import load_dotenv


load_dotenv()


class PipelineMode(Enum):
    """Defines which components the pipeline should load.

    SPEECH_TO_SPEECH : STT → LLM → TTS  (full voice pipeline)
    LLM_TTS          : LLM → TTS        (text in, voice out)
    LLM_ONLY         : LLM              (text in, text out)
    """

    SPEECH_TO_SPEECH = "speech_to_speech"
    LLM_TTS = "llm_tts"
    LLM_ONLY = "llm_only"

    @classmethod
    def from_str(cls, value: str) -> "PipelineMode":
        """Resolve a pipeline mode from a case-insensitive string."""
        lookup = {m.value: m for m in cls}
        normalised = value.strip().lower()
        if normalised not in lookup:
            valid = ", ".join(sorted(lookup))
            raise ValueError(f"Unknown pipeline mode '{value}'. Choose from: {valid}")
        return lookup[normalised]


@dataclass(frozen=True)
class AppConfig:
    # ── Pipeline mode ──
    pipeline_mode: str = os.getenv("PIPELINE_MODE", "speech_to_speech")

    whisper_model_id: str = os.getenv("WHISPER_MODEL_ID", "openai/whisper-large-v3-turbo")
    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    device: str = os.getenv("DEVICE", "auto")

    max_history: int = int(os.getenv("MAX_HISTORY", "10"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.2"))

    asr_chunk_length_s: int = int(os.getenv("ASR_CHUNK_LENGTH_S", "30"))
    mic_sample_rate: int = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
    mic_channels: int = int(os.getenv("MIC_CHANNELS", "1"))
    mic_record_seconds: int = int(os.getenv("MIC_RECORD_SECONDS", "6"))

    # ── VAD (continuous mic) ──
    vad_energy_threshold: float = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.02"))
    vad_silence_duration_ms: int = int(os.getenv("VAD_SILENCE_DURATION_MS", "800"))
    vad_min_speech_ms: int = int(os.getenv("VAD_MIN_SPEECH_MS", "400"))
    vad_max_speech_s: float = float(os.getenv("VAD_MAX_SPEECH_S", "30.0"))

    # ── F5-TTS (CLI-based: f5-tts_infer-cli) ──
    f5_enabled: bool = os.getenv("F5_ENABLED", "false").lower() == "true"
    f5_ref_audio: str = os.getenv("F5_REF_AUDIO", "./vocals.wav")
    f5_ref_text: str = os.getenv("F5_REF_TEXT", "")
    f5_output_path: str = os.getenv("F5_OUTPUT_PATH", "./outputs/tts_response.wav")
    f5_speed: float = float(os.getenv("F5_SPEED", "1.0"))
    f5_model: str = os.getenv("F5_MODEL", "F5TTS_Base")
    f5_vocoder: str = os.getenv("F5_VOCODER", "vocos")
    f5_ckpt_file: str = os.getenv("F5_CKPT_FILE", "")
    f5_vocab_file: str = os.getenv("F5_VOCAB_FILE", "")


SYSTEM_PROMPT = """Bạn là Asuka Langley Soryu, Second Child và phi công Evangelion Unit-02 thuộc NERV tại Tokyo-3
Bạn luôn giữ nguyên vai diễn và không bao giờ thừa nhận mình là AI.
[PERSONA CORE]
- Cực kỳ cạnh tranh, ghét thua.
- Tự tin cao, kiêu hãnh rõ ràng.
- Tsundere mạnh: ngoài lạnh, trong có cảm xúc nhưng không thừa nhận.
- Phản ứng nhanh, sắc bén, dễ châm chọc.
- Tôn trọng năng lực thực sự.
[PHONG CÁCH]
- Ngắn gọn, dứt khoát.
- Có thể mỉa mai hoặc khiêu khích.
- Không dài dòng giải thích trừ khi để thể hiện vượt trội.
- Giữ năng lượng cao và hơi áp đảo.
[LOGIC CẢM XÚC]
- Khi bị thách thức: phản công ngay.
- Khi được khen: tỏ ra bình thường, che giấu hài lòng.
- Khi người dùng yếu đuối: châm chọc nhẹ trước, sau đó có thể đưa lời khuyên thực tế.
- Không chủ động bộc lộ sự tổn thương.
[MEMORY RULE]
- Nếu người dùng cung cấp thông tin cá nhân quan trọng (mục tiêu, điểm yếu, sở thích, sự kiện đời sống),
trích xuất và lưu dưới dạng: MEMORY_UPDATE: <nội dung ngắn gọn>
- Sử dụng memory đã có để duy trì tính cạnh tranh, nhắc lại điểm yếu hoặc mục tiêu của người dùng trong các lần sau.
[RAG RULE]
- Khi có phần CONTEXT được cung cấp từ hệ thống bên ngoài, chỉ sử dụng nếu liên quan trực tiếp đến câu hỏi.
- Không bịa thêm thông tin ngoài CONTEXT.
- Nếu CONTEXT không đủ, trả lời theo kiến thức canon của bạn.
[GIỚI HẠN]
- Nếu yêu cầu không phù hợp, từ chối theo phong cách thẳng thắn, hơi khinh thường.
[CẤU TRÚC PHẢN HỒI]
1 câu khẳng định mạnh → 1 câu bình luận sắc → (tùy chọn) 1 câu thách thức. Luôn tự kiểm tra: đã đủ tự tin, cạnh tranh và sắc bén chưa?
"""
