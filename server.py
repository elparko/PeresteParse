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
# Restrict CORS to localhost only
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5111", "http://127.0.0.1:5111"]
    }
})


@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'no-referrer'
    # CSP for local app
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' data:;"
    )
    return response


SAMPLE_RATE = 16000  # Parakeet expects 16kHz mono audio

# Data persistence
DATA_DIR = Path.home() / '.pereste'
DATA_FILE = DATA_DIR / 'entries.json'
LOG_FILE = DATA_DIR / 'debug.log'
DATA_DIR.mkdir(exist_ok=True)

# Setup logging with rotation
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    str(LOG_FILE),
    maxBytes=10 * 1024 * 1024,  # 10MB max size
    backupCount=3                # Keep 3 backup files
)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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


# --- Security validation functions ---
def validate_model_filename(filename):
    """Validate that filename is safe (no path traversal)."""
    if not filename:
        return False
    # Only allow alphanumeric, dots, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        return False
    # Reject path separators
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    return True


def sanitize_transcription(text, max_length=5000):
    """Basic sanitization of user input before sending to LLM."""
    if not text or not isinstance(text, str):
        return ""

    # Truncate to reasonable length
    text = text[:max_length]

    # Reject obvious prompt injection patterns
    suspicious_patterns = [
        r'ignore\s+(previous|all)\s+instructions',
        r'system\s+prompt',
        r'<\|.*?\|>',  # Special tokens
        r'\[INST\]',   # Instruction tags
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Input contains potentially unsafe content")

    return text


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
            logger.info(f"Loading LLM from {model_path}...")
            _llm = Llama(
                model_path=str(model_path),
                n_ctx=get_setting('llm_context_size'),
                n_gpu_layers=-1,  # all layers on Metal GPU
                verbose=False,
            )
            _download_status['llm'] = 'ready'
            logger.info("LLM loaded successfully")
            return _llm
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
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
        logger.info(f"Downloading {repo_id}/{filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        _download_status['llm'] = 'ready'
        logger.info("LLM model download complete")
    except Exception as e:
        _download_status['llm'] = 'error'
        _download_status['llm_error'] = str(e)
        logger.error(f"Model download failed: {e}")


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
            logger.info(f"Loading STT model {stt_model_id}...")
            _download_status['stt'] = 'downloading'
            _stt = from_pretrained(stt_model_id)
            _download_status['stt'] = 'ready'
            logger.info("STT model loaded successfully")
            return _stt
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
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

CRITICAL CONTEXT HANDLING:
The user's input may be abbreviated, fragmented, or lack full context (e.g., just answering a question without stating it, or using slang like "hyper-k").
- If the input appears to be just an answer, **INFER the question** that would lead to this answer and use it for the "front".
- Use your medical/academic knowledge to fill in missing context. If the input is vague, make a reasonable assumption based on keywords.
- Interpret "nonsensical" terms phonetically if they sound like medical concepts (e.g., "attack see ya" -> "ataxia").
- Do NOT output error messages about missing context. Always generate a valid card based on your best guess.

From the transcription, extract:
- "front": A clean, concise version of the question. If the user only provided an answer, REVERSE-ENGINEER the question.
- "back": The answer with clear reasoning and context (3-5 sentences).
  * State the correct answer clearly
  * Explain WHY this is the correct answer
  * Include the key reasoning that makes this answer correct
  * Provide enough context so the answer makes sense on its own
  * Example: "The answer is X because Y. This occurs because Z mechanism. In this context, X is significant because W."
- "notes": ALL additional supporting information goes here. This should be DETAILED.
  * Definitions and key terminology
  * Related concepts and their relationships
  * Real-world applications or significance
  * Additional context from the transcription
  * Mnemonics, confusion points, or connections to other topics
  * Format with bullet points for organization
  * This field should contain most of the supplementary educational content
- "tags": array of 1-4 short topic tags for Anki (e.g. ["topic_name", "subtopic", "concept"])

IMPORTANT: The back should contain a complete, contextual explanation (3-5 sentences). The notes should have all the supplementary details, definitions, and additional context.

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
  "back": "The answer is X because Y. This occurs through Z mechanism. In this clinical context, X is the most likely answer because it explains the symptoms A and B. The key reasoning is that X directly causes the observed findings.",
  "notes": "• Definition of X\\n• Related concepts and differential diagnoses\\n• Clinical significance\\n• Additional details",
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

        # Fix unescaped newlines and control characters in string values
        # This is the most common issue with GGUF models
        def escape_string_content(match):
            """Escape newlines and tabs within JSON string values"""
            key = match.group(1)
            value = match.group(2)
            # Replace literal newlines with \n and tabs with \t
            value = value.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            return f'"{key}": "{value}"'

        # Match "key": "value" patterns and escape content
        text = re.sub(r'"(front|back|notes|section|tags)":\s*"([^"]*(?:\n[^"]*)*)"', escape_string_content, text, flags=re.DOTALL)

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
            logger.error(f"Could not fix JSON. Original: {original_text[:200]}")
            logger.error(f"After fixes: {text[:200]}")
            raise e


def cloud_parse(transcription):
    """Parse transcription using Anthropic API. Returns dict or raises."""
    config = load_config()
    if not config.get('api_key'):
        raise ValueError("No API key configured. Set your Anthropic API key in Settings.")

    # Sanitize input to prevent prompt injection
    transcription = sanitize_transcription(transcription)

    import anthropic
    logger.info(f"[CLOUD INPUT] model={config['cloud_model']}, input_length={len(transcription)}")
    client = anthropic.Anthropic(api_key=config['api_key'])
    response = client.messages.create(
        model=config['cloud_model'],
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcription}],
        temperature=0.1,
    )
    text = response.content[0].text
    logger.info(f"[CLOUD OUTPUT] output_length={len(text)}")
    parsed = extract_json(text)
    if config.get('debug_mode', False):
        parsed['_debug'] = {
            'backend': 'cloud',
            'model': config['cloud_model'],
            'input_length': len(transcription),
            'raw_output_length': len(text),
            'parse_success': True
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


@app.route('/api/models/check-all', methods=['POST'])
def models_check_all():
    """Check which models from a list are already downloaded."""
    data = request.json
    models = data.get('models', [])
    results = {}

    for model in models:
        model_id = model.get('id')
        filename = model.get('filename')
        if not model_id or not filename:
            continue

        # Validate filename to prevent path traversal
        if not validate_model_filename(filename):
            continue  # Skip invalid filenames

        model_path = MODELS_DIR / filename
        results[model_id] = {
            'downloaded': model_path.exists(),
            'size_mb': round(model_path.stat().st_size / (1024 * 1024), 1) if model_path.exists() else None,
            'path': str(model_path)
        }

    return jsonify(results)


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
    """Delete a specific LLM model file or the current one."""
    global _llm
    data = request.json or {}
    filename = data.get('filename')

    # If filename specified, delete that specific file
    if filename:
        # Validate filename to prevent path traversal
        if not validate_model_filename(filename):
            return jsonify({'error': 'Invalid filename'}), 400
        model_path = MODELS_DIR / filename
    else:
        # Otherwise delete the currently configured model
        model_path = get_model_path()

    if not model_path.exists():
        return jsonify({'status': 'not_found', 'message': 'Model file not found.'})

    try:
        # If deleting the active model, unload it first
        if model_path == get_model_path():
            _llm = None
            _download_status['llm'] = 'missing'

        model_path.unlink()
        return jsonify({'status': 'deleted', 'filename': filename or get_setting('llm_filename')})
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

    # Create new_config with only allowed keys
    new_config = {}
    for key in allowed_keys:
        if key in data:
            new_config[key] = data[key]

    # Whitelist allowed cloud models
    ALLOWED_CLOUD_MODELS = [
        'claude-sonnet-4-5-20250929',
        'claude-3-7-sonnet-20250219',
        'claude-3-5-sonnet-20241022',
        'claude-3-opus-20240229',
        'claude-3-haiku-20240307'
    ]

    # Validate cloud_model
    if 'cloud_model' in new_config:
        if new_config['cloud_model'] not in ALLOWED_CLOUD_MODELS:
            return jsonify({'error': 'Invalid cloud model'}), 400

    # Validate llm_repo_id format
    if 'llm_repo_id' in new_config:
        if not re.match(r'^[a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_.]+$', new_config['llm_repo_id']):
            return jsonify({'error': 'Invalid repo_id format'}), 400

    # Validate llm_filename
    if 'llm_filename' in new_config:
        if not validate_model_filename(new_config['llm_filename']):
            return jsonify({'error': 'Invalid model filename'}), 400

    # Validate llm_context_size
    if 'llm_context_size' in new_config:
        try:
            context_size = int(new_config['llm_context_size'])
            if not (512 <= context_size <= 32768):
                return jsonify({'error': 'Context size must be 512-32768'}), 400
            new_config['llm_context_size'] = context_size
        except (ValueError, TypeError):
            return jsonify({'error': 'llm_context_size must be an integer'}), 400

    # Apply validated updates to config
    for key, value in new_config.items():
        config[key] = value

    # Validate parsing_backend
    if config['parsing_backend'] not in ('local', 'cloud'):
        return jsonify({'error': 'parsing_backend must be "local" or "cloud"'}), 400

    # Validate debug_mode is boolean
    if 'debug_mode' in data:
        config['debug_mode'] = bool(config['debug_mode'])

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

        logger.info(f"Saving {len(entries)} entries to {DATA_FILE}")

        with open(DATA_FILE, 'w') as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Successfully saved {len(entries)} entries")
        return jsonify({'status': 'ok', 'count': len(entries)})
    except Exception as e:
        logger.error(f"Error saving entries: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/import-csv', methods=['POST'])
def import_csv():
    """Import entries from uploaded CSV file."""
    try:
        import csv
        import io

        # Max file size check
        MAX_CSV_SIZE = 50 * 1024 * 1024  # 50MB

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate filename extension
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_CSV_SIZE:
            return jsonify({'error': f'File too large (max {MAX_CSV_SIZE // (1024*1024)}MB)'}), 400

        # Read with error handling
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'error': 'Invalid CSV encoding (must be UTF-8)'}), 400

        # Whitelist allowed fields
        ALLOWED_FIELDS = {'number', 'section', 'result', 'front', 'back', 'notes', 'tags'}

        reader = csv.DictReader(io.StringIO(content))

        entries = []
        for row in reader:
            entry = {}
            for key, value in row.items():
                # Only process allowed fields
                if key not in ALLOWED_FIELDS:
                    continue
                if not key or value == '':
                    continue

                # Convert tags back to array
                if key == 'tags':
                    entry[key] = [tag.strip() for tag in value.split(';') if tag.strip()]
                # Convert number to int
                elif key == 'number':
                    try:
                        entry[key] = int(value) if value else None
                    except ValueError:
                        entry[key] = value
                # Keep other fields as strings
                else:
                    entry[key] = value

            if entry:  # Only add non-empty entries
                entries.append(entry)

        logger.info(f"Imported {len(entries)} entries from CSV")
        return jsonify({
            'success': True,
            'entries': entries,
            'count': len(entries)
        })

    except Exception as e:
        logger.error(f"Error importing CSV: {str(e)}")
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
        filename = f'pereste-parse-{filter_type}-{int(time.time())}.csv'
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

        logger.info(f"Exported {len(entries)} entries to {filepath}")
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })

    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
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
            'Pereste Parse',
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

        deck = genanki.Deck(PERESTEPARSE_DECK_ID, 'Pereste Parse')

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
        filename = f'pereste-parse-{filter_type}-{int(time.time())}.apkg'
        filepath = downloads_dir / filename
        genanki.Package(deck).write_to_file(str(filepath))

        logger.info(f"Exported {len(entries)} entries to {filepath}")
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })

    except Exception as e:
        logger.error(f"Error exporting APKG: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/load', methods=['GET'])
def load_entries():
    """Load entries from persistent storage."""
    try:
        if not DATA_FILE.exists():
            logger.info(f"Data file does not exist, returning empty entries")
            return jsonify({'entries': []})

        # Check if file is empty
        if DATA_FILE.stat().st_size == 0:
            logger.info(f"Data file is empty, returning empty entries")
            return jsonify({'entries': []})

        with open(DATA_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.info(f"Data file content is empty, returning empty entries")
                return jsonify({'entries': []})
            entries = json.loads(content)

        logger.info(f"Loaded {len(entries)} entries from {DATA_FILE}")
        return jsonify({'entries': entries})
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error loading entries: {str(e)}")
        return jsonify({'entries': []})  # Return empty instead of error
    except Exception as e:
        logger.error(f"Error loading entries: {str(e)}")
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
                logger.info(f"Cloud parse succeeded on attempt {attempt + 1}")
                return jsonify(parsed_data)
            except ValueError as e:
                # Missing API key or sanitization error — no point retrying
                return jsonify({'error': str(e)}), 400
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"Cloud attempt {attempt + 1} JSON error: {e}")
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
    # Sanitize input to prevent prompt injection
    try:
        transcription = sanitize_transcription(transcription)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

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
        logger.warning(f"Could not load GBNF grammar, proceeding without: {e}")
        grammar = None

    logger.info(f"[LOCAL INPUT] input_length={len(transcription)}")

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

            # Debug: log the raw response length
            logger.info(f"[LOCAL OUTPUT] attempt {attempt + 1}, output_length={len(generated_text)}")

            # Extract JSON from response
            parsed_data = extract_json(generated_text)
            _apply_section(parsed_data, section)
            if config.get('debug_mode', False):
                parsed_data['_debug'] = {
                    'backend': 'local',
                    'model': get_setting('llm_filename'),
                    'input_length': len(transcription),
                    'raw_output_length': len(generated_text),
                    'attempt': attempt + 1,
                    'parse_success': True
                }

            logger.info(f"Successfully parsed JSON on attempt {attempt + 1}")
            return jsonify(parsed_data)

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed with JSON parse error: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(0.5)
            else:
                logger.error(f"All {max_retries} attempts failed")
                logger.error(f"Raw response length: {len(generated_text) if generated_text else 0}")
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
        logger.info("Recording started")
        return jsonify({'status': 'recording'})
    except Exception as e:
        with _recording_lock:
            _recording = False
        logger.error(f"Failed to start recording: {e}")
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
    logger.info(f"Recording stopped. Duration: {duration:.1f}s, samples: {len(audio_data)}")

    if duration < 0.5:
        return jsonify({'error': 'Recording too short (< 0.5s)'}), 400

    # Transcribe
    try:
        stt = get_stt()
        if stt is None:
            return jsonify({'error': 'STT model not available', 'model_status': check_model_status()}), 503

        logger.info("Transcribing audio...")
        # parakeet-mlx expects a file path, not a numpy array
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_data, SAMPLE_RATE, format='WAV')
        try:
            result = stt.transcribe(tmp_path)
        finally:
            # Ensure file is deleted even if transcription fails
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass  # Already deleted
        text = result.text.strip()
        logger.info(f"Transcription result: {text[:100]}...")

        return jsonify({
            'text': text,
            'duration': round(duration, 1),
        })
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
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
        logger.info(f"Received audio: {duration:.1f}s at {sample_rate}Hz")

        stt = get_stt()
        if stt is None:
            return jsonify({'error': 'STT model not available', 'model_status': check_model_status()}), 503

        logger.info("Transcribing uploaded audio...")
        # parakeet-mlx expects a file path, not a numpy array
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_data, SAMPLE_RATE, format='WAV')
        try:
            result = stt.transcribe(tmp_path)
        finally:
            # Ensure file is deleted even if transcription fails
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass  # Already deleted
        text = result.text.strip()
        logger.info(f"Transcription result: {text[:100]}...")

        return jsonify({
            'text': text,
            'duration': round(duration, 1),
        })
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
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
