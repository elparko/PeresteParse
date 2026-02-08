# Pereste

Voice transcriptions → LLM-parsed flashcards → Anki export. Works with any subject.

## Features

- **Voice recording** — Record directly in-app via Parakeet MLX speech-to-text
- **LLM parsing** — Transcriptions are parsed into structured question/answer/notes cards
- **Cloud or Local** — Use Anthropic Claude API or run a local Qwen3-4B model on your Mac
- **Anki export** — Export as `.apkg` with section filtering, or CSV
- **Session tracking** — Correct/incorrect stats, section grouping, search and filter
- **Native macOS app** — Runs as a standalone `.app` via pywebview

## Quick Start

### From source

```bash
# Clone and install
git clone https://github.com/youruser/peresteparse.git
cd peresteparse
uv sync

# Install llama-cpp-python with Metal (Apple Silicon GPU)
CMAKE_ARGS="-DGGML_METAL=on" uv pip install 'llama-cpp-python>=0.3.0'

# Run
uv run python app.py
```

The setup wizard will guide you through choosing Cloud (Anthropic API) or Local (downloads ~2.5GB model).

### Build macOS .app

```bash
bash scripts/build_app.sh
open dist/Pereste.app
```

To create a distributable DMG:

```bash
bash scripts/create_dmg.sh
```

## Architecture

| File | Purpose |
|------|---------|
| `server.py` | Flask backend — parsing, export, config, recording APIs |
| `static/index.html` | React 18 SPA (served as static file) |
| `app.py` | pywebview native window wrapper |

**Data directory:** `~/.peresteparse/` — config, entries, downloaded models.

## Configuration

On first launch, a setup wizard lets you choose:

- **Cloud** — Enter your Anthropic API key. Supports Claude Sonnet 4.5 and Haiku 4.5.
- **Local** — Downloads Qwen3-4B (Q4_K_M GGUF, ~2.5GB) and runs via llama-cpp-python with Metal acceleration.

Settings are stored in `~/.peresteparse/config.json` and can be changed anytime via the gear icon.

## Tech Stack

- **Backend**: Flask + Flask-CORS
- **Frontend**: React 18 (CDN) + Babel
- **Cloud LLM**: Anthropic Claude API
- **Local LLM**: llama-cpp-python + Qwen3-4B GGUF with GBNF grammar
- **Speech-to-text**: Parakeet MLX
- **Anki export**: genanki
- **Native wrapper**: pywebview
- **Fonts**: JetBrains Mono + Outfit

## Development

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=server

# Run server directly (browser mode)
uv run python server.py
```

## License

MIT
