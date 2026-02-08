# Pereste Parse

Voice transcriptions → LLM-parsed flashcards → Anki export. Works with any subject.

**By Pereste** — Voice to flashcards.

## Features

- **Voice recording** — Record directly in-app via Parakeet MLX speech-to-text
- **LLM parsing** — Transcriptions are parsed into structured question/answer/notes cards
- **Cloud or Local** — Use Anthropic Claude API or run a local Qwen3-4B model on your Mac
- **Anki export** — Export as `.apkg` with section filtering, or CSV
- **Session tracking** — Correct/incorrect stats, section grouping, search and filter
- **Native macOS app** — Runs as a standalone `.app` via pywebview

## Installation

### Download the App (Recommended for Users)

1. **Download** the latest `Pereste Parse.dmg` from [GitHub Releases](https://github.com/elparko/PeresteParse/releases/latest)
2. **Open** the DMG file
3. **Drag** Pereste Parse to your Applications folder
4. **First Launch**: Right-click (or Control+click) on Pereste Parse → Select **"Open"**
5. Click **"Open"** in the security dialog
6. The app will now run normally on all future launches

#### Security Note

⚠️ This app is **not code-signed**. macOS will show an "unverified developer" warning on first launch. The right-click method above allows you to bypass this for applications you trust.

See [SECURITY.md](SECURITY.md) for more information about security and privacy.

### System Requirements

- macOS 11.0 (Big Sur) or later
- Apple Silicon (M1/M2/M3/M4) or Intel processor
- 4GB+ RAM recommended
- 5GB free disk space (for LLM models)

## Quick Start

### From source

```bash
# Clone and install
git clone https://github.com/youruser/pereste-parse.git
cd pereste-parse
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
open "dist/Pereste Parse.app"
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

**Data directory:** `~/.pereste/` — config, entries, downloaded models.

## Configuration

On first launch, a setup wizard lets you choose:

- **Cloud** — Enter your Anthropic API key. Supports Claude Sonnet 4.5 and Haiku 4.5.
- **Local** — Downloads Qwen3-4B (Q4_K_M GGUF, ~2.5GB) and runs via llama-cpp-python with Metal acceleration.

Settings are stored in `~/.pereste/config.json` and can be changed anytime via the gear icon.

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

## Security & Privacy

Pereste Parse is designed with privacy in mind:

- **Local-first**: All data stored in `~/.pereste/` on your Mac
- **No tracking**: No analytics or telemetry
- **Local mode**: Process everything on-device (no internet required)
- **Cloud mode**: Optional - only sends transcriptions to Anthropic API when configured

Security features include:
- CORS restricted to localhost only
- Input sanitization and validation
- Path traversal protection
- Automatic log rotation
- Secure headers

For detailed security information, see [SECURITY.md](SECURITY.md).

To report security vulnerabilities, please open a [GitHub Issue](https://github.com/elparko/PeresteParse/issues) with the "security" label.

## License

MIT
