"""Shared fixtures for Pereste tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import server


@pytest.fixture
def app(tmp_path):
    """Create Flask test app with temp data directory."""
    data_dir = tmp_path / '.peresteparse'
    data_dir.mkdir()
    data_file = data_dir / 'entries.json'
    models_dir = data_dir / 'models'
    models_dir.mkdir()

    config_file = data_dir / 'config.json'

    with patch.object(server, 'DATA_DIR', data_dir), \
         patch.object(server, 'DATA_FILE', data_file), \
         patch.object(server, 'MODELS_DIR', models_dir), \
         patch.object(server, 'CONFIG_FILE', config_file):
        server.app.config['TESTING'] = True
        yield server.app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def data_file(tmp_path):
    """Path to temp entries.json (matches the patched DATA_FILE)."""
    return tmp_path / '.peresteparse' / 'entries.json'


@pytest.fixture
def models_dir(tmp_path):
    """Path to temp models directory (created on disk)."""
    d = tmp_path / '.peresteparse' / 'models'
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_entries():
    """Sample entries for testing."""
    return [
        {
            "id": "entry-1",
            "number": 1,
            "section": "Upper Limb",
            "result": "correct",
            "front": "What nerve innervates the deltoid muscle?",
            "back": "The axillary nerve innervates the deltoid.",
            "notes": "The axillary nerve (C5, C6) arises from the posterior cord of the brachial plexus.",
            "tags": ["upper_limb", "axillary_nerve", "deltoid"],
            "timestamp": "2025-01-01T00:00:00Z"
        },
        {
            "id": "entry-2",
            "number": 2,
            "section": "Upper Limb",
            "result": "incorrect",
            "front": "Which artery supplies the scapular anastomosis?",
            "back": "The suprascapular artery and dorsal scapular artery.",
            "notes": "Multiple arteries contribute to the scapular anastomosis.",
            "tags": ["upper_limb", "scapular_anastomosis"],
            "timestamp": "2025-01-01T00:01:00Z"
        },
        {
            "id": "entry-3",
            "number": 5,
            "section": "Thorax",
            "result": "correct",
            "front": "What is the innervation of the diaphragm?",
            "back": "The phrenic nerve (C3, C4, C5) provides motor innervation to the diaphragm.",
            "notes": "C3, C4, C5 keeps the diaphragm alive. Sensory innervation also from phrenic nerve (central) and intercostal nerves (peripheral).",
            "tags": ["thorax", "phrenic_nerve", "diaphragm"],
            "timestamp": "2025-01-01T00:02:00Z"
        }
    ]


@pytest.fixture
def mock_llm_json_response():
    """Mock a successful LLM chat completion response with valid JSON."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "front": "What is the origin of the radial nerve?",
                    "back": "The radial nerve originates from the posterior cord of the brachial plexus (C5-T1).",
                    "notes": "The radial nerve is the largest branch of the posterior cord. It innervates the posterior compartment of the arm and forearm.",
                    "tags": ["radial_nerve", "brachial_plexus"]
                })
            }
        }]
    }


@pytest.fixture
def mock_llm_markdown_response():
    """Mock LLM response with JSON wrapped in markdown code blocks."""
    return {
        "choices": [{
            "message": {
                "content": "Here is the parsed result:\n```json\n" + json.dumps({
                    "front": "Which muscle abducts the arm?",
                    "back": "The deltoid muscle abducts the arm.",
                    "notes": "The supraspinatus initiates abduction (first 15 degrees).",
                    "tags": ["deltoid", "abduction"]
                }) + "\n```\n"
            }
        }]
    }
