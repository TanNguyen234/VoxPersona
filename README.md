# VoxPersona

Voice-first persona chatbot: nói chuyện với nhân vật anime bằng giọng nói thật.

| Component | Model | Integration |
|-----------|-------|-------------|
| ASR | `openai/whisper-large-v3-turbo` | HuggingFace `pipeline` (auto-download) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | `TextIteratorStreamer` + `threading` (streaming) |
| TTS | `F5-TTS` / `F5-TTS-Vietnamese` | CLI subprocess (`f5-tts_infer-cli`) |
| VAD | Energy-based silence detection | `sounddevice.InputStream` callback |

## 1) Kiến trúc

VoxPersona hỗ trợ **3 pipeline modes**, chọn qua `--pipeline` hoặc `PIPELINE_MODE` env:

| Mode | Pipeline | Mô tả | VRAM |
|------|----------|-------|------|
| `speech_to_speech` | Mic → Whisper → Qwen → F5-TTS | Full voice loop | Cao nhất |
| `llm_tts` | Text → Qwen → F5-TTS | Gõ text, nhận voice | Trung bình |
| `llm_only` | Text → Qwen | Gõ text, nhận text | Thấp nhất |

**Chỉ các model cần thiết mới được load** — nếu chọn `llm_only`, Whisper và F5-TTS không bao giờ được tải, tiết kiệm RAM/VRAM đáng kể.

```text
  ┌─────────────────────────────────────────────────────────┐
  │  speech_to_speech (default)                             │
  │  Mic ──► VAD ──► Whisper ASR ──► Qwen LLM ──► F5-TTS   │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────┐
  │  llm_tts                                │
  │  Text input ──► Qwen LLM ──► F5-TTS     │
  └─────────────────────────────────────────┘
  ┌──────────────────────────┐
  │  llm_only                │
  │  Text input ──► Qwen LLM │
  └──────────────────────────┘
```

```text
VoxPersona/
  src/voxpersona/
    audio/
      recorder.py              # MicrophoneRecorder (fixed) + ContinuousMicRecorder (VAD)
    models/
      whisper_asr.py           # ASR adapter (HuggingFace pipeline)
      qwen_chat.py             # LLM streaming + conversation history
      f5_tts.py                # F5-TTS CLI subprocess wrapper
    pipeline.py                # Orchestration: text / mic / continuous listen
    cli.py                     # CLI entrypoint with /mic, /listen, /exit
    config.py                  # .env config + system prompt (Asuka persona)
  scripts/
    setup_models.py            # Clone F5 repos + download weights
  vendor/                      # (git-ignored) cloned model repos
  .env.example
  requirements.txt
  pyproject.toml
  run.py
```

## 2) Yêu cầu

- Python 3.10+
- GPU CUDA (khuyến nghị >= 8 GB VRAM)
- Git, Git LFS
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

### 3.3 Clone repos & tải model weights (F5-TTS)

F5-TTS là research model, **không có trên PyPI**.  Script `setup_models.py` sẽ:

1. `git clone --depth 1` repo F5 vào `vendor/`
2. `pip install -e vendor/F5-TTS` — cài CLI `f5-tts_infer-cli`
3. Tải weights từ HuggingFace vào `vendor/.../ckpts/`

```powershell
# Clone tất cả (F5 base + F5 Vietnamese)
python scripts/setup_models.py

# Hoặc chỉ 1 trong 2:
python scripts/setup_models.py --f5
python scripts/setup_models.py --f5-viet

# Kiểm tra trạng thái:
python scripts/setup_models.py --check
```

**Quan trọng:** Cả 2 repos cần được clone và cài để TTS hoạt động:

| Repo | Mục đích |
|------|----------|
| `SWivid/F5-TTS` | Codebase gốc + CLI `f5-tts_infer-cli` |
| `nguyenthienhy/F5-TTS-Vietnamese` | Weights fine-tuned tiếng Việt |

### 3.4 Thiết lập biến môi trường

```powershell
copy .env.example .env
```

Chỉnh `.env` theo tài nguyên máy.  Đặc biệt nếu dùng bản Vietnamese:

```dotenv
F5_ENABLED=true
F5_CKPT_FILE=./vendor/F5-TTS-Vietnamese/ckpts/model_last.pt
F5_VOCAB_FILE=./vendor/F5-TTS-Vietnamese/ckpts/vocab.txt
```

## 4) Chạy project

### 4.1 Chọn pipeline mode

Thêm `--pipeline` để chọn mode (mặc định `speech_to_speech`):

```powershell
# Mode 1: Full voice — STT + LLM + TTS (mặc định)
python run.py --pipeline speech_to_speech

# Mode 2: Text in, voice out — LLM + TTS (không cần mic/Whisper)
python run.py --pipeline llm_tts

# Mode 3: Text only — chỉ LLM (nhẹ nhất, không Whisper, không F5)
python run.py --pipeline llm_only
```

Hoặc đặt trong `.env`:
```dotenv
PIPELINE_MODE=llm_only
```

### 4.2 Interactive (khuyến nghị)

```powershell
python run.py --pipeline llm_tts --mode interactive
```

Trong interactive:
- Gõ text bình thường để chat → LLM → (TTS nếu có)
- `/mic` — ghi âm mic (chỉ ở `speech_to_speech`)
- `/listen` — bật mic liên tục VAD (chỉ ở `speech_to_speech`)
- `/mode` — hiển thị pipeline mode hiện tại
- `/exit` — thoát

### 4.3 Continuous listen (chỉ speech_to_speech)

```powershell
python run.py --pipeline speech_to_speech --mode listen
```

Mic bật liên tục.  Khi bạn nói xong (im lặng >= `VAD_SILENCE_DURATION_MS`),
audio tự động cắt → Whisper ASR → Qwen Chat → (optional) F5 TTS.
Nói tiếp là hệ thống tiếp tục nghe.

### 4.4 Text 1 lượt

```powershell
# Với TTS output
python run.py --pipeline llm_tts --mode text --text "Asuka, đánh giá kế hoạch học AI của tôi"

# Chỉ text (nhẹ nhất)
python run.py --pipeline llm_only --mode text --text "Asuka, bạn nghĩ gì về deep learning?"
```

### 4.5 Mic 1 lượt (chỉ speech_to_speech)

```powershell
python run.py --pipeline speech_to_speech --mode mic --record-seconds 7
```

## 5) Continuous Mic — VAD chi tiết

`ContinuousMicRecorder` sử dụng energy-based Voice Activity Detection:

| Parameter | Env var | Default | Ý nghĩa |
|-----------|---------|---------|----------|
| Energy threshold | `VAD_ENERGY_THRESHOLD` | `0.02` | Ngưỡng RMS để phát hiện giọng nói |
| Silence duration | `VAD_SILENCE_DURATION_MS` | `800` | Khoảng im lặng (ms) để cắt 1 câu |
| Min speech | `VAD_MIN_SPEECH_MS` | `400` | Bỏ qua tiếng ngắn < 400ms (click/cough) |
| Max speech | `VAD_MAX_SPEECH_S` | `30.0` | Giới hạn tối đa 1 utterance (giây) |

**Cách hoạt động:**
1. `sounddevice.InputStream` callback liên tục thu audio (block 30ms)
2. Khi RMS energy > threshold → bắt đầu ghi nhận utterance
3. Khi energy < threshold kéo dài >= `silence_duration_ms` → utterance kết thúc
4. Nếu utterance > `min_speech_ms` → lưu WAV → đưa vào Whisper
5. Nếu < `min_speech_ms` → bỏ qua (tiếng ồn)
6. Lặp lại cho đến khi user nhấn Ctrl-C

## 6) Cấu trúc vendor/

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

## 7) Điểm kỹ thuật nổi bật

| Feature | Chi tiết |
|---------|----------|
| **F5-TTS via CLI** | Gọi `f5-tts_infer-cli` bằng `subprocess` — cách duy nhất hoạt động ổn định (Python API bị lỗi) |
| **LLM streaming** | `TextIteratorStreamer` + `threading.Thread` — in từng token realtime |
| **GPU cleanup** | `gc.collect()` + `torch.cuda.empty_cache()` sau mỗi session |
| **Continuous mic** | Energy-based VAD, mic mở liên tục, tự segment utterances |
| **Conversation history** | Giới hạn `MAX_HISTORY` messages để tránh tràn VRAM |
| **Multi-mode pipeline** | 3 chế độ: `speech_to_speech`, `llm_tts`, `llm_only` — lazy-load components |

## 8) Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `f5-tts_infer-cli` not found | `python scripts/setup_models.py` → `pip install -e vendor/F5-TTS` |
| Mic không nhận voice (listen mode) | Giảm `VAD_ENERGY_THRESHOLD` (thử `0.01`) |
| VAD cắt giữa câu | Tăng `VAD_SILENCE_DURATION_MS` (thử `1200`) |
| VAD bắt tiếng ồn | Tăng `VAD_ENERGY_THRESHOLD` (thử `0.04`) |
| Thiếu VRAM | Chuyển sang `--pipeline llm_only`, giảm `MAX_NEW_TOKENS`, hoặc dùng Qwen bản nhỏ hơn |
| ASR chậm | Xác nhận `torch.cuda.is_available() == True` |
| Git LFS lỗi | Cài Git LFS: `git lfs install` rồi chạy lại setup |

## 9) Roadmap

- [x] Continuous mic (VAD-based utterance segmentation)
- [x] F5-TTS CLI integration
- [x] Multi-mode pipeline (speech_to_speech / llm_tts / llm_only)
- [ ] Audio playback tự động sau TTS
- [ ] UI realtime (FastAPI + WebSocket)
- [ ] Memory extraction → persistent store
- [ ] Latency benchmarks (ASR / LLM / TTS)
