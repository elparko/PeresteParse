# Anatomix

A lightweight local tool that takes voice transcriptions (from Super Whisper) of Gray's Anatomy Review question sessions and auto-parses them into structured data via a local Ollama model, displaying results in a spreadsheet view and exporting Anki cards.

## Features

- **Rapid-fire parsing** — Paste transcription, hit Ctrl+Enter, instant structured output
- **Session management** — Set section once, auto-applies to all questions
- **Real-time stats** — Track correct/incorrect answers and score percentage
- **Anki export** — One-click export to tab-separated .txt file for direct Anki import
- **CSV export** — Export all data to CSV for further analysis
- **Local processing** — Everything runs on your machine via Ollama
- **Persistent storage** — All entries saved in browser localStorage

## Quick Start

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com
```

### 2. Pull the model

```bash
ollama pull llama3.2
```

### 3. Install Python dependencies

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 4. Run the server

```bash
python server.py
```

The app will open at **http://localhost:5111**

## Usage Workflow

1. **Set your section** — Enter the anatomical region (e.g. "Upper Limb", "Thorax")
2. **Paste transcription** — Copy your Super Whisper transcription into the text box
3. **Parse** — Hit Ctrl+Enter or click "Parse Question"
4. **Review** — Check the parsed entry in the table below
5. **Repeat** — Text box clears automatically, ready for next question
6. **Export** — When done, click "Export Anki .txt" to generate importable Anki cards

## Data Model

Each parsed question includes:
- **number** — Question number from the book (parsed by LLM)
- **section** — Anatomical region (set by you)
- **result** — "correct" or "incorrect" (parsed by LLM)
- **front** — Anki card front — clean question/clinical scenario
- **back** — Anki card back — answer + key knowledge + reasoning
- **tags** — 1-4 topic tags for Anki organization
- **notes** — Extra context, mnemonics, connections (optional)

## Anki Import

1. Export as .txt from the app
2. In Anki: File → Import
3. Select the exported .txt file
4. Choose your deck
5. Field separator: Tab
6. Done!

## Configuration

Set environment variables to customize Ollama:

```bash
export OLLAMA_URL=http://localhost:11434  # Default
export OLLAMA_MODEL=llama3.2              # Default

python server.py
```

## Recommended Models

- **llama3.2** (3B) — Best balance of speed and accuracy
- **llama3.2:1b** — Faster, slightly less accurate
- **phi3:mini** — Smallest/fastest option

## Troubleshooting

**Ollama connection error?**
- Make sure Ollama is running: `ollama serve`
- Check the model is pulled: `ollama list`

**Port 5111 already in use?**
- Edit `server.py` and change the port number in the last line

**Parsing errors?**
- Try a different model (see Configuration above)
- Check your transcription quality
- Make sure question numbers are mentioned in the transcription

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: React (via CDN)
- **LLM**: Ollama (local)
- **Storage**: Browser localStorage
- **Fonts**: JetBrains Mono + Outfit

## License

MIT
