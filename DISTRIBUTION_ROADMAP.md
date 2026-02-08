# Anatomix v2 — Self-Contained Distribution Build

## Context

Rebuild Anatomix as a self-contained macOS app that bundles speech-to-text (Whisper) and local LLM parsing, with an optional paid cloud API tier. The goal is a seamless user experience — press record, speak, get flashcards — with no external dependencies (no Ollama, no Super Whisper).

**Work in a fresh clone** to avoid compromising the current working build.

---

## Step 0: Clone & Set Up New Project

- Clone current repo to a new directory (e.g., `~/Code/Anatomix-v2`)
- Initialize fresh `uv` venv
- Keep original at `~/Code/AnatomyReviewPipeline` untouched

---

## Step 1: Integrate Local Whisper (Speech-to-Text)

**Goal:** Replace external Super Whisper dependency with built-in transcription.

### Tech stack:
- **`mlx-whisper`** — 2x faster than whisper.cpp on Apple Silicon, GPU-accelerated via Metal
- **Model:** `whisper-large-v3-turbo` (1.5 GB) — best accuracy-to-size ratio, critical for medical terminology
- **Audio recording:** `sounddevice` (Python-side via pywebview JS→Python bridge)

### Changes:
- **`server.py`**: New endpoint `/api/transcribe` that:
  1. Receives audio data (WAV) from frontend
  2. Runs mlx-whisper transcription
  3. Returns text
- **`server.py`**: New endpoint `/api/record/start` and `/api/record/stop` using `sounddevice`
  - Or: expose recording via pywebview's JS→Python bridge in `app.py`
- **`static/index.html`**: Add record button with start/stop + visual indicator
- **`requirements.txt`**: Add `mlx-whisper`, `sounddevice`
- **First-run:** Download whisper model to `~/.anatomix/models/` on first use (show progress)

### Key detail — microphone permissions:
- `sounddevice` handles macOS mic permission dialogs natively
- Need `NSMicrophoneUsageDescription` in the app's Info.plist when packaging

---

## Step 2: Replace Ollama with Built-in LLM

**Goal:** Bundle a local LLM for parsing — no Ollama required.

### Tech stack:
- **`llama-cpp-python`** with Metal GPU acceleration
- **Model:** Qwen3-4B Q4_K_M (2.5 GB GGUF) — best quality-to-size ratio for structured extraction
  - Fallback option: Qwen2.5-3B (2.1 GB) if size is a concern
- **GBNF grammar constraints** — forces 100% valid JSON output, ~25% better extraction accuracy

### Changes:
- **`server.py`**: Replace Ollama HTTP calls with direct `llama-cpp-python` inference
  - Load model once at startup, keep in memory
  - Use GBNF grammar to enforce JSON schema
  - Reuse existing system prompt
- **`requirements.txt`**: Replace `requests` (Ollama) with `llama-cpp-python`
- **First-run:** Download LLM model to `~/.anatomix/models/` on first use

### Install note:
```bash
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
```

---

## Step 3: Add Cloud API Backend (Paid Tier)

**Goal:** Optional cloud API for users who want better accuracy or don't want local models.

### Changes:
- **`server.py`**: Add cloud parsing backend (Anthropic or OpenAI API)
  - Config-driven: `~/.anatomix/config.json` stores `{"backend": "local" | "cloud", "api_key": "..."}`
  - Both backends use the same system prompt, same response format
  - Cloud Whisper option too (OpenAI Whisper API) for users without Apple Silicon
- **`static/index.html`**: Settings panel:
  - Toggle: Local / Cloud
  - API key input (for cloud)
  - Model selection
- **`requirements.txt`**: Add `anthropic` or `openai` SDK

---

## Step 4: Remove Hardcoded Paths & Config

- Replace all hardcoded `/Users/parker/...` paths with dynamic resolution
- Centralize config in `~/.anatomix/config.json`:
  - Backend choice (local/cloud)
  - API key
  - Model preferences
  - Whisper model choice
- All user data stays in `~/.anatomix/` (entries, models, config, logs)

---

## Step 5: First-Run Experience

- Detect first run (no `~/.anatomix/config.json`)
- Show setup flow in the app:
  1. Welcome screen
  2. "Downloading speech recognition model..." (progress bar, ~1.5 GB)
  3. "Downloading language model..." (progress bar, ~2.5 GB)
  4. Mic permission test
  5. Ready to use
- Option to skip local models and use cloud API instead (enter API key)

---

## Step 6: Package as .app

### Recommended approach (based on research):
**Don't use py2app/PyInstaller** — bundling native ML libraries (Metal shaders, compiled C++) with these tools is fragile and poorly documented.

Instead, use the **self-contained directory approach**:
1. Ship a directory containing:
   - Python venv with all dependencies pre-installed
   - App code (`app.py`, `server.py`, `static/`)
   - `launch.sh` script
2. Wrap in an AppleScript `.app` bundle (existing pattern works)
3. Models download to `~/.anatomix/models/` on first run (not bundled)
4. Distribute as a `.dmg` (drag app to Applications)

**App download size:** ~100-150 MB (without models)
**After first-run model download:** ~4 GB in `~/.anatomix/models/`

---

## Step 7 (Later): Code Signing & Notarization

- Apple Developer account ($99/year) for code signing
- Without it: users right-click → Open on first launch
- Required for any paid distribution

---

## Bundle Size Summary

| Component | Size | When |
|-----------|------|------|
| App (Python + venv + code) | ~100-150 MB | Initial download |
| Whisper large-v3-turbo | 1.5 GB | First-run download |
| Qwen3-4B Q4_K_M | 2.5 GB | First-run download |
| **Total on disk** | **~4.2 GB** | After setup |

---

## Verification

After each step, verify by:
1. Launch the app and confirm existing functionality still works
2. Step 1: Record audio → get transcription text
3. Step 2: Transcription → parsed flashcard JSON (without Ollama running)
4. Step 3: Toggle to cloud → same parsing works via API
5. Step 5: Delete `~/.anatomix/`, relaunch → setup flow appears
6. Step 6: Move `.app` to `/Applications` → launches correctly

---

## Implementation Order

Start with **Step 0** (clone), then **Step 2** (replace Ollama — biggest architectural change), then **Step 1** (add Whisper), then **Steps 3-6** in order. This way you always have a working app at each stage.
