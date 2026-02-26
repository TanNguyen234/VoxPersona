# VoxPersona

Voice-first persona chatbot: nói chuyện với nhân vật anime bằng giọng nói thật.

| Component | Model | Source |
|-----------|-------|--------|
| ASR | `openai/whisper-large-v3-turbo` | HuggingFace (auto-download) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace (auto-download) |
| TTS | `F5-TTS` / `F5-TTS-Vietnamese` | Git clone + weights download |

## 1) Kiến trúc

```text
Mic ──► Whisper ASR ──► Qwen Chat (streaming) ──► F5-TTS ──► Speaker
                 ▲                                    │
            text input                          .wav output
```

```text
VoxPersona/
  src/voxpersona/
    audio/recorder.py          # Ghi âm microphone
    models/
      whisper_asr.py           # ASR adapter
      qwen_chat.py             # LLM streaming + conversation history
      f5_tts.py                # F5-TTS Python API wrapper
    pipeline.py                # Orchestration end-to-end
    cli.py                     # CLI entrypoint
    config.py                  # .env config + system prompt
  scripts/
    setup_models.py            # Clone repos + download weights
  vendor/                      # (git-ignored) cloned model repos
  .env.example
  requirements.txt
  pyproject.toml
  run.py
```

## 2) Yêu cầu

- Python 3.10+
- GPU CUDA (khuyến nghị >= 8 GB VRAM)
- Git
- FFmpeg (cho xử lý audio)
- Windows: microphone hoạt động

## 3) Cài đặt

### 3.1 Tạo môi trường ảo

```powershell
cd D:\Projects\VoxPersona
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.2 Cài dependencies cơ bản

```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3.3 Clone repos & tải model weights

F5-TTS là research model, **không có trên PyPI** — cần clone repo gốc rồi cài editable.
Script `setup_models.py` tự động hóa toàn bộ quá trình:

```powershell
# Clone tất cả model repos + tải weights (F5 base + F5 Vietnamese)
python scripts/setup_models.py

# Hoặc chỉ clone F5 base:
python scripts/setup_models.py --f5

# Hoặc chỉ clone F5 Vietnamese:
python scripts/setup_models.py --f5-viet

# Kiểm tra trạng thái cài đặt:
python scripts/setup_models.py --check
```

Script sẽ:
1. `git clone --depth 1` repo vào `vendor/`
2. `pip install -e vendor/F5-TTS` (hoặc F5-TTS-Vietnamese)
3. Tải weights từ HuggingFace vào `vendor/.../ckpts/`

### 3.4 Thiết lập biến môi trường

```powershell
copy .env.example .env
```

Chỉnh `.env` theo tài nguyên máy. Đặc biệt:

```dotenv
F5_ENABLED=true
F5_CKPT_PATH=./vendor/F5-TTS/ckpts/model_1200000.pt

# Hoặc dùng bản Vietnamese:
# F5_CKPT_PATH=./vendor/F5-TTS-Vietnamese/ckpts/model_last.pt
# F5_VOCAB_PATH=./vendor/F5-TTS-Vietnamese/ckpts/vocab.txt
```

## 4) Chạy project

### 4.1 Interactive (khuyến nghị)

```powershell
python run.py --mode interactive
```

Trong interactive:
- Gõ text bình thường để chat
- Gõ `/mic` để ghi âm mic → ASR → chat
- Gõ `/exit` để thoát

### 4.2 Text 1 lượt

```powershell
python run.py --mode text --text "Asuka, đánh giá kế hoạch học AI của tôi"
```

### 4.3 Mic 1 lượt

```powershell
python run.py --mode mic --record-seconds 7
```

### 4.4 Trích code model từ notebook

```powershell
python tools/extract_models_from_notebook.py --input project2.ipynb --output-dir notebook_extracts
```

## 5) Cấu trúc vendor/

Sau khi chạy `setup_models.py`, thư mục `vendor/` sẽ có:

```text
vendor/
  F5-TTS/                    # git clone từ SWivid/F5-TTS
    ckpts/
      model_1200000.pt       # ~1.2 GB, tải từ HuggingFace
    ...
  F5-TTS-Vietnamese/         # git clone từ nguyenthienhy/F5-TTS-Vietnamese
    ckpts/
      model_last.pt          # weights fine-tuned tiếng Việt
      vocab.txt              # Vietnamese vocabulary
    ...
```

Toàn bộ `vendor/` được git-ignore — không push lên repo.

## 6) Điểm kỹ thuật

- **F5-TTS dùng Python API** (`from f5_tts.model import DiT`) thay vì gọi CLI subprocess
- LLM streaming: `TextIteratorStreamer` + `threading.Thread`
- GPU cleanup: `gc.collect()` + `torch.cuda.empty_cache()`
- Conversation history có giới hạn `MAX_HISTORY` tránh tràn VRAM
- Pipeline duy nhất cho cả text input và mic input

## 7) Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| Thiếu VRAM | Giảm `MAX_NEW_TOKENS`, dùng Qwen bản nhỏ hơn |
| Mic không ghi âm | Kiểm tra quyền microphone Windows |
| ASR chậm | Xác nhận `torch.cuda.is_available() == True` |
| F5 import error | Chạy lại `python scripts/setup_models.py --check` |
| `f5_tts.model not found` | Đảm bảo đã `pip install -e vendor/F5-TTS` |

## 8) Roadmap

- [ ] UI realtime (FastAPI + WebSocket)
- [ ] Voice Activity Detection (VAD)
- [ ] Memory extraction → persistent store
- [ ] Latency benchmarks (ASR / LLM / TTS)
