import os
import json
import re
import time
import tempfile
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

SAMPLE_RATE = 16000  # Parakeet expects 16kHz mono audio

# Data persistence
DATA_DIR = Path.home() / '.peresteparse'
DATA_FILE = DATA_DIR / 'entries.json'
LOG_FILE = DATA_DIR / 'debug.log'
DATA_DIR.mkdir(exist_ok=True)

# Setup logging
import logging
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Model management ---
MODELS_DIR = DATA_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

_llm = None
_llm_lock = threading.Lock()
_stt = None
_stt_lock = threading.Lock()
_download_status = {'llm': 'unknown', 'stt': 'unknown'}

# Recording state
_recording = False
_recorded_frames = []
_recording_lock = threading.Lock()


def get_model_path():
    """Get the path to the LLM GGUF file."""
    return MODELS_DIR / get_setting('llm_filename')


def get_llm():
    """Lazy-load the LLM. Returns the model or None if not available."""
    global _llm
    if _llm is not None:
        return _llm

    model_path = get_model_path()
    if not model_path.exists():
        _download_status['llm'] = 'missing'
        return None

    with _llm_lock:
        if _llm is not None:
            return _llm
        try:
            from llama_cpp import Llama
            logging.info(f"Loading LLM from {model_path}...")
            _llm = Llama(
                model_path=str(model_path),
                n_ctx=get_setting('llm_context_size'),
                n_gpu_layers=-1,  # all layers on Metal GPU
                verbose=False,
            )
            _download_status['llm'] = 'ready'
            logging.info("LLM loaded successfully")
            return _llm
        except Exception as e:
            logging.error(f"Failed to load LLM: {e}")
            _download_status['llm'] = 'error'
            return None


def download_model_background():
    """Download the LLM model in a background thread."""
    global _download_status
    _download_status['llm'] = 'downloading'
    try:
        from huggingface_hub import hf_hub_download
        repo_id = get_setting('llm_repo_id')
        filename = get_setting('llm_filename')
        logging.info(f"Downloading {repo_id}/{filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        _download_status['llm'] = 'ready'
        logging.info("LLM model download complete")
    except Exception as e:
        _download_status['llm'] = 'error'
        _download_status['llm_error'] = str(e)
        logging.error(f"Model download failed: {e}")


def check_model_status():
    """Check and update the current model status."""
    global _download_status
    if _llm is not None:
        _download_status['llm'] = 'ready'
    elif _download_status['llm'] == 'downloading':
        pass  # in progress
    elif get_model_path().exists():
        _download_status['llm'] = 'ready'
    else:
        _download_status['llm'] = 'missing'

    # STT model status — parakeet-mlx manages its own cache via HF hub
    if _stt is not None:
        _download_status['stt'] = 'ready'
    elif _download_status.get('stt') == 'downloading':
        pass
    else:
        _download_status.setdefault('stt', 'unknown')

    return _download_status.copy()


def get_stt():
    """Lazy-load the speech-to-text model. Returns the model or None."""
    global _stt
    if _stt is not None:
        return _stt

    with _stt_lock:
        if _stt is not None:
            return _stt
        try:
            from parakeet_mlx import from_pretrained
            stt_model_id = get_setting('stt_model_id')
            logging.info(f"Loading STT model {stt_model_id}...")
            _download_status['stt'] = 'downloading'
            _stt = from_pretrained(stt_model_id)
            _download_status['stt'] = 'ready'
            logging.info("STT model loaded successfully")
            return _stt
        except Exception as e:
            logging.error(f"Failed to load STT model: {e}")
            _download_status['stt'] = 'error'
            _download_status['stt_error'] = str(e)
            return None


# --- Config management ---
CONFIG_FILE = DATA_DIR / 'config.json'

DEFAULT_CONFIG = {
    'parsing_backend': 'local',       # 'local' or 'cloud'
    'cloud_provider': 'anthropic',
    'api_key': '',
    'cloud_model': 'claude-sonnet-4-5-20250929',
    'llm_repo_id': 'Qwen/Qwen3-4B-GGUF',
    'llm_filename': 'Qwen3-4B-Q4_K_M.gguf',
    'llm_context_size': 8192,
    'stt_model_id': 'mlx-community/parakeet-tdt-0.6b-v3',
    'setup_complete': False,
    'default_result': 'incorrect',    # 'correct' or 'incorrect'
    'debug_mode': False,
}


def get_setting(key):
    """Get a setting value. Env var overrides config.json."""
    env_key = key.upper()
    env_val = os.environ.get(env_key)
    if env_val is not None:
        if key == 'llm_context_size':
            return int(env_val)
        return env_val
    return load_config()[key]


def load_config():
    """Load config from file, filling missing keys with defaults."""
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        for key, val in DEFAULT_CONFIG.items():
            config.setdefault(key, val)
        return config
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save config to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def mask_api_key(key):
    """Mask an API key for display, showing only the last 4 characters."""
    if not key or len(key) < 8:
        return ''
    return key[:3] + '...' + key[-4:]


SYSTEM_PROMPT = """You are a precise JSON parser for study transcriptions. A student is reviewing questions and recording voice notes about each question.

From the transcription, extract:
- "front": A clean, concise version of the question (like an Anki card front). Write it as a proper question.
- "back": ONLY the direct answer. Keep this SHORT (2-3 sentences maximum).
  * State the correct answer clearly
  * Include ONLY the essential reasoning
  * Do NOT include definitions, additional details, or supplementary information here
  * Example: "The answer is X because it does Y."
- "notes": ALL additional information goes here. This should be DETAILED.
  * Definitions and key terminology
  * Related concepts and their relationships
  * Real-world applications or significance
  * Additional context from the transcription
  * Mnemonics, confusion points, or connections to other topics
  * Format with bullet points for organization
  * This field should contain most of the educational content
- "tags": array of 1-4 short topic tags for Anki (e.g. ["topic_name", "subtopic", "concept"])

IMPORTANT: The back should be SHORT (just the answer + brief reasoning). All definitions and detailed information belong in notes.

CRITICAL JSON FORMATTING RULES:
1. Your response must be ONLY a valid JSON object starting with { and ending with }
2. ALL string values MUST have BOTH opening AND closing quotes
3. Do not include any text before or after the JSON
4. Do not use markdown code blocks
5. Do not add explanations
6. Verify all quotes are properly paired before responding

Example format:
{
  "front": "What is the question?",
  "back": "The answer is X because Y.",
  "notes": "Detail 1. Detail 2.",
  "tags": ["tag1", "tag2"]
}"""

# GBNF grammar to enforce valid JSON output matching our schema
PARSE_GRAMMAR = r"""
root ::= "{" ws "\"front\"" ws ":" ws string ws "," ws "\"back\"" ws ":" ws string ws "," ws "\"notes\"" ws ":" ws string ws "," ws "\"tags\"" ws ":" ws tag-arr ws "}"
tag-arr ::= "[]" | "[" ws string ( ws "," ws string )* ws "]"
string ::= "\"" chars "\""
chars ::= char chars | ""
char ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "n" | "t" | "r" | "/" | "b" | "f"
ws ::= [ \t\n]*
"""


def extract_json(text):
    """Extract JSON from LLM response, handling markdown code blocks and extra text."""
    text = text.strip()

    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try to find JSON object anywhere in the text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    # Clean up common issues
    text = text.strip()

    # Remove any text before the first {
    if '{' in text:
        text = text[text.index('{'):]

    # Remove any text after the last }
    if '}' in text:
        text = text[:text.rindex('}') + 1]

    # Try to parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        original_text = text

        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)

        # Fix missing quotes around string values (common LLM error)
        # Pattern: "key": value" where value should be quoted
        text = re.sub(r':\s+([^"\[\{\d\-][^,\}\]]*)"', r': "\1"', text)

        # Fix lines like: "front": What is...?" -> "front": "What is...?"
        # Match: "key": text where text doesn't start with a quote
        text = re.sub(r'("(?:front|back|notes|section)":\s+)([^"\[\{])', r'\1"\2', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logging.error(f"Could not fix JSON. Original: {original_text[:200]}")
            logging.error(f"After fixes: {text[:200]}")
            raise e


def cloud_parse(transcription):
    """Parse transcription using Anthropic API. Returns dict or raises."""
    config = load_config()
    if not config.get('api_key'):
        raise ValueError("No API key configured. Set your Anthropic API key in Settings.")
    import anthropic
    logging.info(f"[CLOUD INPUT] model={config['cloud_model']}, transcription={transcription!r}")
    client = anthropic.Anthropic(api_key=config['api_key'])
    response = client.messages.create(
        model=config['cloud_model'],
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcription}],
        temperature=0.1,
    )
    text = response.content[0].text
    logging.info(f"[CLOUD OUTPUT] raw response: {text}")
    parsed = extract_json(text)
    if config.get('debug_mode', False):
        parsed['_debug'] = {
            'backend': 'cloud',
            'model': config['cloud_model'],
            'input': transcription,
            'raw_output': text,
        }
    return parsed


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/health', methods=['GET'])
def health():
    """Check parsing backend availability."""
    config = load_config()

    if config['parsing_backend'] == 'cloud':
        if config.get('api_key'):
            return jsonify({
                'status': 'ok',
                'backend': 'cloud',
                'provider': config['cloud_provider'],
                'model': config['cloud_model'],
            })
        else:
            return jsonify({
                'status': 'error',
                'backend': 'cloud',
                'message': 'No API key configured. Open Settings to add your Anthropic API key.',
            }), 503

    # Local backend
    status = check_model_status()

    llm_filename = get_setting('llm_filename')
    if status['llm'] == 'ready':
        return jsonify({
            'status': 'ok',
            'model': llm_filename,
            'backend': 'local'
        })
    elif status['llm'] == 'downloading':
        return jsonify({
            'status': 'downloading',
            'message': 'Model is being downloaded...',
            'model': llm_filename
        }), 503
    elif status['llm'] == 'missing':
        return jsonify({
            'status': 'error',
            'message': f'Model not found. Download it via the setup flow.',
            'model': llm_filename
        }), 503
    else:
        return jsonify({
            'status': 'error',
            'message': status.get('llm_error', 'Unknown error loading model'),
            'model': llm_filename
        }), 503


@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Check download/readiness status of all models."""
    status = check_model_status()
    model_path = get_model_path()
    status['llm_model'] = get_setting('llm_filename')
    status['llm_repo'] = get_setting('llm_repo_id')
    status['llm_path'] = str(model_path)
    status['llm_exists'] = model_path.exists()
    if model_path.exists():
        status['llm_size_mb'] = round(model_path.stat().st_size / (1024 * 1024), 1)
    status['stt_model'] = get_setting('stt_model_id')
    return jsonify(status)


@app.route('/api/models/download', methods=['POST'])
def models_download():
    """Trigger model download in background."""
    status = check_model_status()
    if status['llm'] == 'downloading':
        return jsonify({'status': 'already_downloading'})
    if status['llm'] == 'ready' and get_model_path().exists():
        return jsonify({'status': 'already_exists'})

    thread = threading.Thread(target=download_model_background, daemon=True)
    thread.start()
    return jsonify({'status': 'download_started'})


@app.route('/api/models/delete', methods=['POST'])
def models_delete():
    """Delete the downloaded LLM model file."""
    global _llm
    model_path = get_model_path()
    if not model_path.exists():
        return jsonify({'status': 'not_found', 'message': 'No model file to delete.'})
    try:
        model_path.unlink()
        _llm = None
        _download_status['llm'] = 'missing'
        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration (API key masked)."""
    config = load_config()
    config['api_key'] = mask_api_key(config.get('api_key', ''))
    return jsonify(config)


@app.route('/api/config', methods=['POST'])
def set_config():
    """Update configuration."""
    data = request.json
    config = load_config()

    # Only update allowed keys
    allowed_keys = {
        'parsing_backend', 'cloud_provider', 'api_key', 'cloud_model',
        'llm_repo_id', 'llm_filename', 'llm_context_size', 'stt_model_id',
        'setup_complete', 'default_result', 'debug_mode',
    }
    for key in allowed_keys:
        if key in data:
            config[key] = data[key]

    # Validate parsing_backend
    if config['parsing_backend'] not in ('local', 'cloud'):
        return jsonify({'error': 'parsing_backend must be "local" or "cloud"'}), 400

    # Validate debug_mode is boolean
    if 'debug_mode' in data:
        config['debug_mode'] = bool(config['debug_mode'])

    # Validate llm_context_size is integer
    if not isinstance(config['llm_context_size'], int):
        try:
            config['llm_context_size'] = int(config['llm_context_size'])
        except (ValueError, TypeError):
            return jsonify({'error': 'llm_context_size must be an integer'}), 400

    save_config(config)

    # Return config with masked key
    config['api_key'] = mask_api_key(config.get('api_key', ''))
    return jsonify(config)


@app.route('/api/save', methods=['POST'])
def save_entries():
    """Save entries to persistent storage."""
    try:
        data = request.json
        entries = data.get('entries', [])

        logging.info(f"Saving {len(entries)} entries to {DATA_FILE}")

        with open(DATA_FILE, 'w') as f:
            json.dump(entries, f, indent=2)

        logging.info(f"Successfully saved {len(entries)} entries")
        return jsonify({'status': 'ok', 'count': len(entries)})
    except Exception as e:
        logging.error(f"Error saving entries: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """Export entries as CSV file directly to Downloads folder."""
    try:
        import csv

        data = request.json
        entries = data.get('entries', [])
        filter_type = data.get('filter', 'all')

        if not entries:
            return jsonify({'error': 'No entries to export'}), 400

        # Save directly to Downloads folder
        downloads_dir = Path.home() / 'Downloads'
        filename = f'peresteparse-{filter_type}-{int(time.time())}.csv'
        filepath = downloads_dir / filename

        # Write CSV file
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            headers = ['number', 'section', 'result', 'front', 'back', 'tags', 'notes']
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')

            writer.writeheader()
            for entry in entries:
                # Convert arrays to strings
                row = {}
                for key, value in entry.items():
                    if key not in headers:
                        continue  # Skip fields not in headers
                    if isinstance(value, list):
                        # Convert all lists to strings
                        row[key] = '; '.join(str(item) for item in value)
                    elif value is None:
                        row[key] = ''
                    else:
                        row[key] = str(value)
                writer.writerow(row)

        logging.info(f"Exported {len(entries)} entries to {filepath}")
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })

    except Exception as e:
        logging.error(f"Error exporting CSV: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-apkg', methods=['POST'])
def export_apkg():
    """Export entries as native Anki .apkg file to Downloads folder."""
    try:
        import genanki

        data = request.json
        entries = data.get('entries', [])
        filter_type = data.get('filter', 'all')

        if not entries:
            return jsonify({'error': 'No entries to export'}), 400

        # Stable IDs so re-imports update existing cards
        PERESTEPARSE_MODEL_ID = 1607392319
        PERESTEPARSE_DECK_ID = 2059400110

        model = genanki.Model(
            PERESTEPARSE_MODEL_ID,
            'Pereste',
            fields=[{'name': 'Front'}, {'name': 'Back'}, {'name': 'Notes'}],
            templates=[{
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Back}}'
                        '{{#Notes}}<br><br><div class="notes">{{Notes}}</div>{{/Notes}}',
            }],
            css='.card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }'
                ' .notes { font-size: 16px; color: #666; font-style: italic; }'
        )

        deck = genanki.Deck(PERESTEPARSE_DECK_ID, 'Pereste')

        for entry in entries:
            front = entry.get('front', '') or ''
            back = (entry.get('back', '') or '').replace('\n', '<br>')
            notes = (entry.get('notes', '') or '').replace('\n', '<br>')
            tags = entry.get('tags', []) or []
            note = genanki.Note(
                model=model,
                fields=[front, back, notes],
                tags=tags
            )
            deck.add_note(note)

        # Save to Downloads folder
        downloads_dir = Path.home() / 'Downloads'
        filename = f'peresteparse-{filter_type}-{int(time.time())}.apkg'
        filepath = downloads_dir / filename
        genanki.Package(deck).write_to_file(str(filepath))

        logging.info(f"Exported {len(entries)} entries to {filepath}")
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })

    except Exception as e:
        logging.error(f"Error exporting APKG: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/load', methods=['GET'])
def load_entries():
    """Load entries from persistent storage."""
    try:
        if not DATA_FILE.exists():
            logging.info(f"Data file does not exist, returning empty entries")
            return jsonify({'entries': []})

        # Check if file is empty
        if DATA_FILE.stat().st_size == 0:
            logging.info(f"Data file is empty, returning empty entries")
            return jsonify({'entries': []})

        with open(DATA_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                logging.info(f"Data file content is empty, returning empty entries")
                return jsonify({'entries': []})
            entries = json.loads(content)

        logging.info(f"Loaded {len(entries)} entries from {DATA_FILE}")
        return jsonify({'entries': entries})
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error loading entries: {str(e)}")
        return jsonify({'entries': []})  # Return empty instead of error
    except Exception as e:
        logging.error(f"Error loading entries: {str(e)}")
        return jsonify({'entries': []})  # Return empty instead of error


def _apply_section(parsed_data, section):
    """Apply section field and inject section tag into parsed data."""
    parsed_data['section'] = section if section else parsed_data.get('section', 'General')
    section_tag = parsed_data['section'].lower().replace(' ', '_')
    if 'tags' not in parsed_data:
        parsed_data['tags'] = []
    if section_tag not in parsed_data['tags']:
        parsed_data['tags'].insert(0, section_tag)
    return parsed_data


@app.route('/api/parse', methods=['POST'])
def parse():
    """Parse a transcription using local LLM or cloud API based on config."""
    data = request.json
    transcription = data.get('transcription', '').strip()
    section = data.get('section', '').strip()

    if not transcription:
        return jsonify({'error': 'No transcription provided'}), 400

    config = load_config()

    # --- Cloud backend ---
    if config['parsing_backend'] == 'cloud':
        max_retries = 2
        last_error = None
        for attempt in range(max_retries):
            try:
                parsed_data = cloud_parse(transcription)
                _apply_section(parsed_data, section)
                logging.info(f"Cloud parse succeeded on attempt {attempt + 1}")
                return jsonify(parsed_data)
            except ValueError as e:
                # Missing API key — no point retrying
                return jsonify({'error': str(e)}), 400
            except json.JSONDecodeError as e:
                last_error = e
                logging.warning(f"Cloud attempt {attempt + 1} JSON error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    return jsonify({
                        'error': 'Failed to parse JSON from cloud response',
                        'details': str(last_error),
                        'attempts': max_retries
                    }), 500
            except Exception as e:
                return jsonify({
                    'error': 'Cloud API error',
                    'details': str(e)
                }), 500

    # --- Local backend ---
    llm = get_llm()
    if llm is None:
        return jsonify({
            'error': 'LLM model not loaded. Download it first via /api/models/download.',
            'model_status': check_model_status()
        }), 503

    max_retries = 2
    last_error = None
    generated_text = None

    # Load GBNF grammar for JSON enforcement
    try:
        from llama_cpp import LlamaGrammar
        grammar = LlamaGrammar.from_string(PARSE_GRAMMAR)
    except Exception as e:
        logging.warning(f"Could not load GBNF grammar, proceeding without: {e}")
        grammar = None

    logging.info(f"[LOCAL INPUT] transcription={transcription!r}")

    for attempt in range(max_retries):
        try:
            # Call local LLM
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcription}
                ],
                temperature=0.1,
                max_tokens=1024,
                grammar=grammar,
            )

            generated_text = response['choices'][0]['message']['content']

            # Debug: log the raw response
            logging.info(f"[LOCAL OUTPUT] attempt {attempt + 1}: {generated_text}")

            # Extract JSON from response
            parsed_data = extract_json(generated_text)
            _apply_section(parsed_data, section)
            if config.get('debug_mode', False):
                parsed_data['_debug'] = {
                    'backend': 'local',
                    'model': get_setting('llm_filename'),
                    'input': transcription,
                    'raw_output': generated_text,
                    'attempt': attempt + 1,
                }

            logging.info(f"Successfully parsed JSON on attempt {attempt + 1}")
            return jsonify(parsed_data)

        except json.JSONDecodeError as e:
            last_error = e
            logging.warning(f"Attempt {attempt + 1} failed with JSON parse error: {str(e)}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(0.5)
            else:
                logging.error(f"All {max_retries} attempts failed")
                logging.error(f"Raw response that failed: {generated_text}")
                return jsonify({
                    'error': 'Failed to parse JSON from LLM response after multiple attempts',
                    'details': str(last_error),
                    'raw_response': generated_text if generated_text else 'N/A',
                    'attempts': max_retries
                }), 500

        except Exception as e:
            return jsonify({
                'error': 'LLM inference error',
                'details': str(e)
            }), 500


@app.route('/api/record/start', methods=['POST'])
def record_start():
    """Start recording audio from the microphone."""
    global _recording, _recorded_frames
    with _recording_lock:
        if _recording:
            return jsonify({'error': 'Already recording'}), 409
        _recording = True
        _recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        if _recording:
            _recorded_frames.append(indata.copy())

    try:
        import sounddevice as sd
        _stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=audio_callback,
        )
        _stream.start()
        # Store stream reference so we can stop it later
        app.config['_audio_stream'] = _stream
        logging.info("Recording started")
        return jsonify({'status': 'recording'})
    except Exception as e:
        with _recording_lock:
            _recording = False
        logging.error(f"Failed to start recording: {e}")
        return jsonify({'error': f'Failed to start recording: {str(e)}'}), 500


@app.route('/api/record/stop', methods=['POST'])
def record_stop():
    """Stop recording and return the transcription."""
    global _recording, _recorded_frames
    import numpy as np

    with _recording_lock:
        if not _recording:
            return jsonify({'error': 'Not currently recording'}), 409
        _recording = False

    # Stop the audio stream
    stream = app.config.pop('_audio_stream', None)
    if stream is not None:
        stream.stop()
        stream.close()

    if not _recorded_frames:
        return jsonify({'error': 'No audio recorded'}), 400

    # Combine recorded frames into single array
    audio_data = np.concatenate(_recorded_frames, axis=0).flatten()
    _recorded_frames = []

    duration = len(audio_data) / SAMPLE_RATE
    logging.info(f"Recording stopped. Duration: {duration:.1f}s, samples: {len(audio_data)}")

    if duration < 0.5:
        return jsonify({'error': 'Recording too short (< 0.5s)'}), 400

    # Transcribe
    try:
        stt = get_stt()
        if stt is None:
            return jsonify({'error': 'STT model not available', 'model_status': check_model_status()}), 503

        logging.info("Transcribing audio...")
        # parakeet-mlx expects a file path, not a numpy array
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_data, SAMPLE_RATE, format='WAV')
            tmp_path = tmp.name
        try:
            result = stt.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)
        text = result.text.strip()
        logging.info(f"Transcription result: {text[:100]}...")

        return jsonify({
            'text': text,
            'duration': round(duration, 1),
        })
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribe uploaded audio file."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        import numpy as np
        import soundfile as sf
        import io

        # Read the audio file
        audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()), dtype='float32')

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            import soxr
            audio_data = soxr.resample(audio_data, sample_rate, SAMPLE_RATE)

        duration = len(audio_data) / SAMPLE_RATE
        logging.info(f"Received audio: {duration:.1f}s at {sample_rate}Hz")

        stt = get_stt()
        if stt is None:
            return jsonify({'error': 'STT model not available', 'model_status': check_model_status()}), 503

        logging.info("Transcribing uploaded audio...")
        # parakeet-mlx expects a file path, not a numpy array
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_data, SAMPLE_RATE, format='WAV')
            tmp_path = tmp.name
        try:
            result = stt.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)
        text = result.text.strip()
        logging.info(f"Transcription result: {text[:100]}...")

        return jsonify({
            'text': text,
            'duration': round(duration, 1),
        })
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500


if __name__ == '__main__':
    print(f'Starting server on http://localhost:5111')
    print(f'LLM Model: {get_setting("llm_repo_id")}/{get_setting("llm_filename")}')
    print(f'Context Size: {get_setting("llm_context_size")}')
    print(f'Models dir: {MODELS_DIR}')
    status = check_model_status()
    if status['llm'] == 'ready':
        print('LLM model: ready')
    else:
        print(f'LLM model: {status["llm"]} — download via the app or POST /api/models/download')
    app.run(host='0.0.0.0', port=5111, debug=False)
