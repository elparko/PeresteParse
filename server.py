import os
import json
import re
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder='static')
CORS(app)

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
OLLAMA_CONTEXT_SIZE = int(os.environ.get('OLLAMA_CONTEXT_SIZE', '8192'))

# Data persistence
DATA_DIR = Path.home() / '.anatomix'
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

SYSTEM_PROMPT = """You are a precise JSON parser for medical study transcriptions. A medical student is reviewing Gray's Anatomy questions and recording voice notes about each question.

From the transcription, extract:
- "number": question number (integer or null)
- "result": "correct" or "incorrect"
- "front": A clean, concise version of the question or clinical scenario (like an Anki card front). Write it as a proper question.
- "back": ONLY the direct answer. Keep this SHORT (2-3 sentences maximum).
  * State the correct answer clearly
  * Include ONLY the essential reasoning
  * Do NOT include definitions, additional details, or supplementary information here
  * Example: "The radial nerve is the answer because it innervates the posterior compartment of the arm."
- "notes": ALL additional information goes here. This should be DETAILED.
  * Anatomical definitions and terminology
  * Related structures and their relationships
  * Clinical correlations and significance
  * Additional context from the transcription
  * Mnemonics, confusion points, or connections to other topics
  * Format with bullet points for organization
  * This field should contain most of the educational content
- "tags": array of 1-4 short topic tags for Anki (e.g. ["brachial_plexus", "nerve_injury", "upper_limb"])

IMPORTANT: The back should be SHORT (just the answer + brief reasoning). All definitions and detailed information belong in notes.

CRITICAL: Your response must be ONLY a valid JSON object starting with { and ending with }. Do not include any text before or after the JSON. Do not use markdown code blocks. Do not add explanations."""


def extract_json(text):
    """Extract JSON from Ollama response, handling markdown code blocks and extra text."""
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
    except json.JSONDecodeError:
        # Last resort: try to fix common JSON issues
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return json.loads(text)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/health', methods=['GET'])
def health():
    """Check Ollama connection and model availability."""
    try:
        # Check if Ollama is running
        response = requests.get(f'{OLLAMA_URL}/api/tags', timeout=5)
        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': 'Ollama is not responding'
            }), 503

        # Check if the model is available
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]

        # Check for exact match or model with tag
        model_available = any(
            OLLAMA_MODEL in name or name.startswith(OLLAMA_MODEL + ':')
            for name in model_names
        )

        if not model_available:
            return jsonify({
                'status': 'error',
                'message': f'Model {OLLAMA_MODEL} not found. Available models: {", ".join(model_names)}',
                'available_models': model_names
            }), 404

        return jsonify({
            'status': 'ok',
            'ollama_url': OLLAMA_URL,
            'model': OLLAMA_MODEL
        })

    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'message': f'Could not connect to Ollama: {str(e)}'
        }), 503


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
        filename = f'anatomix-{filter_type}-{int(time.time())}.csv'
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


@app.route('/api/parse', methods=['POST'])
def parse():
    """Parse a transcription using Ollama."""
    data = request.json
    transcription = data.get('transcription', '').strip()
    section = data.get('section', '').strip()

    if not transcription:
        return jsonify({'error': 'No transcription provided'}), 400

    try:
        # Call Ollama
        response = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': OLLAMA_MODEL,
                'prompt': transcription,
                'system': SYSTEM_PROMPT,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'num_predict': 1024,
                    'num_ctx': OLLAMA_CONTEXT_SIZE
                }
            },
            timeout=30
        )

        if response.status_code != 200:
            return jsonify({
                'error': f'Ollama returned status {response.status_code}',
                'details': response.text
            }), 502

        result = response.json()
        generated_text = result.get('response', '')

        # Debug: log the raw response
        logging.info(f"Raw Ollama response: {generated_text}")

        # Extract JSON from response
        parsed_data = extract_json(generated_text)

        # Ensure section field is set
        parsed_data['section'] = section if section else parsed_data.get('section', 'General')

        # Auto-inject section tag (always)
        section_tag = parsed_data['section'].lower().replace(' ', '_')
        if 'tags' not in parsed_data:
            parsed_data['tags'] = []
        if section_tag not in parsed_data['tags']:
            parsed_data['tags'].insert(0, section_tag)

        return jsonify(parsed_data)

    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error: {str(e)}")
        logging.error(f"Raw response that failed: {generated_text if 'generated_text' in locals() else 'N/A'}")
        return jsonify({
            'error': 'Failed to parse JSON from Ollama response',
            'details': str(e),
            'raw_response': generated_text if 'generated_text' in locals() else 'N/A'
        }), 500

    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to connect to Ollama',
            'details': str(e)
        }), 503

    except Exception as e:
        return jsonify({
            'error': 'Unexpected error',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    print(f'Starting server on http://localhost:5111')
    print(f'Ollama URL: {OLLAMA_URL}')
    print(f'Ollama Model: {OLLAMA_MODEL}')
    print(f'Context Size: {OLLAMA_CONTEXT_SIZE}')
    app.run(host='0.0.0.0', port=5111, debug=True)
