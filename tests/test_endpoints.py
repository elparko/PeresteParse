"""Integration tests for all API endpoints in server.py."""

import csv
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import server


class TestStaticFileServing:
    """Tests for static file serving endpoints."""

    def test_index_returns_html(self, client):
        response = client.get('/')
        assert response.status_code == 200
        assert b'Pereste' in response.data

    def test_serve_manifest(self, client):
        response = client.get('/manifest.json')
        assert response.status_code == 200
        data = response.get_json()
        assert data['name'] == 'Pereste'

    def test_serve_nonexistent_returns_404(self, client):
        response = client.get('/nonexistent-file.js')
        assert response.status_code == 404


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_ok_when_model_exists(self, client, models_dir):
        # Create a fake model file
        (models_dir / server.DEFAULT_CONFIG['llm_filename']).write_bytes(b'fake model data')

        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['backend'] == 'local'

    def test_health_error_when_model_missing(self, client):
        response = client.get('/api/health')
        assert response.status_code == 503
        data = response.get_json()
        assert data['status'] == 'error'

    def test_health_downloading_state(self, client):
        with patch.object(server, '_download_status', {'llm': 'downloading'}):
            response = client.get('/api/health')
        assert response.status_code == 503
        data = response.get_json()
        assert data['status'] == 'downloading'

    def test_health_ok_when_llm_loaded(self, client):
        mock_llm = MagicMock()
        with patch.object(server, '_llm', mock_llm):
            response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'


class TestModelsStatusEndpoint:
    """Tests for GET /api/models/status."""

    def test_models_status_missing(self, client):
        response = client.get('/api/models/status')
        assert response.status_code == 200
        data = response.get_json()
        assert data['llm'] == 'missing'
        assert data['llm_exists'] is False

    def test_models_status_ready(self, client, models_dir):
        (models_dir / server.DEFAULT_CONFIG['llm_filename']).write_bytes(b'x' * 1024 * 1024)

        response = client.get('/api/models/status')
        assert response.status_code == 200
        data = response.get_json()
        assert data['llm'] == 'ready'
        assert data['llm_exists'] is True
        assert data['llm_size_mb'] == 1.0

    def test_models_status_includes_model_info(self, client):
        response = client.get('/api/models/status')
        data = response.get_json()
        assert 'llm_model' in data
        assert 'llm_repo' in data
        assert 'llm_path' in data


class TestModelsDownloadEndpoint:
    """Tests for POST /api/models/download."""

    def test_download_starts(self, client):
        with patch.object(server, 'download_model_background'):
            response = client.post('/api/models/download')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'download_started'

    def test_download_already_in_progress(self, client):
        with patch.object(server, '_download_status', {'llm': 'downloading'}):
            response = client.post('/api/models/download')
        data = response.get_json()
        assert data['status'] == 'already_downloading'

    def test_download_already_exists(self, client, models_dir):
        (models_dir / server.DEFAULT_CONFIG['llm_filename']).write_bytes(b'fake model')
        response = client.post('/api/models/download')
        data = response.get_json()
        assert data['status'] == 'already_exists'


class TestSaveEndpoint:
    """Tests for POST /api/save."""

    def test_save_entries(self, client, sample_entries, data_file):
        response = client.post('/api/save',
            json={'entries': sample_entries},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['count'] == 3

        assert data_file.exists()
        saved = json.loads(data_file.read_text())
        assert len(saved) == 3
        assert saved[0]['id'] == 'entry-1'

    def test_save_empty_entries(self, client, data_file):
        response = client.post('/api/save',
            json={'entries': []},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['count'] == 0

    def test_save_overwrites_existing(self, client, sample_entries, data_file):
        client.post('/api/save', json={'entries': sample_entries})
        client.post('/api/save', json={'entries': [sample_entries[0]]})

        saved = json.loads(data_file.read_text())
        assert len(saved) == 1


class TestLoadEndpoint:
    """Tests for GET /api/load."""

    def test_load_existing_entries(self, client, sample_entries, data_file):
        data_file.write_text(json.dumps(sample_entries))

        response = client.get('/api/load')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['entries']) == 3
        assert data['entries'][0]['id'] == 'entry-1'

    def test_load_no_file(self, client):
        response = client.get('/api/load')
        assert response.status_code == 200
        data = response.get_json()
        assert data['entries'] == []

    def test_load_empty_file(self, client, data_file):
        data_file.write_text('')

        response = client.get('/api/load')
        assert response.status_code == 200
        data = response.get_json()
        assert data['entries'] == []

    def test_load_corrupt_json(self, client, data_file):
        data_file.write_text('{not valid json!!!}}}')

        response = client.get('/api/load')
        assert response.status_code == 200
        data = response.get_json()
        assert data['entries'] == []

    def test_load_whitespace_only_file(self, client, data_file):
        data_file.write_text('   \n\t  ')

        response = client.get('/api/load')
        assert response.status_code == 200
        data = response.get_json()
        assert data['entries'] == []


class TestExportCsvEndpoint:
    """Tests for POST /api/export-csv."""

    def test_export_csv_all(self, client, sample_entries, tmp_path):
        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-csv',
                json={'entries': sample_entries, 'filter': 'all'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'pereste-parse-all-' in data['filename']
        assert data['filename'].endswith('.csv')

        csv_path = Path(data['path'])
        assert csv_path.exists()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]['section'] == 'Upper Limb'

    def test_export_csv_tags_joined(self, client, sample_entries, tmp_path):
        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-csv',
                json={'entries': sample_entries, 'filter': 'all'},
                content_type='application/json'
            )

        data = response.get_json()
        csv_path = Path(data['path'])
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert '; ' in rows[0]['tags']

    def test_export_csv_no_entries(self, client):
        response = client.post('/api/export-csv',
            json={'entries': [], 'filter': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_export_csv_null_values_handled(self, client, tmp_path):
        entries = [{
            "number": None,
            "section": "Test",
            "result": "correct",
            "front": "Q?",
            "back": "A.",
            "notes": None,
            "tags": []
        }]
        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-csv',
                json={'entries': entries, 'filter': 'all'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        csv_path = Path(data['path'])
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]['number'] == ''

    def test_export_csv_headers_correct(self, client, sample_entries, tmp_path):
        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-csv',
                json={'entries': sample_entries, 'filter': 'all'},
                content_type='application/json'
            )

        data = response.get_json()
        csv_path = Path(data['path'])
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == ['number', 'section', 'result', 'front', 'back', 'tags', 'notes']


class TestExportApkg:
    """Tests for POST /api/export-apkg."""

    def test_export_apkg_success(self, client, sample_entries, tmp_path):
        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-apkg',
                json={'entries': sample_entries, 'filter': 'all'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'pereste-parse-all-' in data['filename']
        assert data['filename'].endswith('.apkg')

        apkg_path = Path(data['path'])
        assert apkg_path.exists()

    def test_export_apkg_empty(self, client):
        response = client.post('/api/export-apkg',
            json={'entries': [], 'filter': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_export_apkg_with_filter(self, client, sample_entries, tmp_path):
        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-apkg',
                json={'entries': sample_entries, 'filter': 'incorrect'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert 'pereste-parse-incorrect-' in data['filename']

    def test_export_apkg_is_valid_zip(self, client, sample_entries, tmp_path):
        import zipfile

        with patch('server.Path.home', return_value=tmp_path):
            downloads = tmp_path / 'Downloads'
            downloads.mkdir()

            response = client.post('/api/export-apkg',
                json={'entries': sample_entries, 'filter': 'all'},
                content_type='application/json'
            )

        data = response.get_json()
        apkg_path = Path(data['path'])
        assert zipfile.is_zipfile(apkg_path)


class TestParseEndpoint:
    """Tests for POST /api/parse."""

    def _mock_llm(self, response_content):
        """Helper: create a mock LLM that returns given content."""
        mock = MagicMock()
        mock.create_chat_completion.return_value = {
            "choices": [{"message": {"content": response_content}}]
        }
        return mock

    def test_parse_success(self, client, mock_llm_json_response):
        content = mock_llm_json_response['choices'][0]['message']['content']
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 42, I got it correct. The radial nerve...', 'section': 'Upper Limb'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data['section'] == 'Upper Limb'
        assert data['front'] == 'What is the origin of the radial nerve?'

    def test_parse_injects_section_tag(self, client, mock_llm_json_response):
        content = mock_llm_json_response['choices'][0]['message']['content']
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 42...', 'section': 'Upper Limb'},
                content_type='application/json'
            )

        data = response.get_json()
        assert 'upper_limb' in data['tags']
        assert data['tags'][0] == 'upper_limb'

    def test_parse_section_tag_not_duplicated(self, client):
        content = json.dumps({
            "front": "Q?", "back": "A.",
            "notes": "", "tags": ["upper_limb", "nerve"]
        })
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 1...', 'section': 'Upper Limb'},
                content_type='application/json'
            )

        data = response.get_json()
        assert data['tags'].count('upper_limb') == 1

    def test_parse_with_markdown_wrapped_response(self, client, mock_llm_markdown_response):
        content = mock_llm_markdown_response['choices'][0]['message']['content']
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 10...', 'section': 'Upper Limb'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data['front'] == 'Which muscle abducts the arm?'

    def test_parse_empty_transcription(self, client):
        response = client.post('/api/parse',
            json={'transcription': '', 'section': 'Upper Limb'},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_parse_whitespace_only_transcription(self, client):
        response = client.post('/api/parse',
            json={'transcription': '   \n  ', 'section': 'Upper Limb'},
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_parse_no_section_defaults(self, client, mock_llm_json_response):
        content = mock_llm_json_response['choices'][0]['message']['content']
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 42...', 'section': ''},
                content_type='application/json'
            )

        data = response.get_json()
        assert data['section'] in ['General', '']

    def test_parse_model_not_loaded(self, client):
        """When no LLM model is available, should return 503."""
        with patch.object(server, '_llm', None):
            response = client.post('/api/parse',
                json={'transcription': 'Question 1...', 'section': 'Test'},
                content_type='application/json'
            )

        assert response.status_code == 503
        data = response.get_json()
        assert 'model_status' in data

    def test_parse_llm_inference_error(self, client):
        """When LLM raises an exception, should return 500."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("Metal device error")

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 1...', 'section': 'Test'},
                content_type='application/json'
            )

        assert response.status_code == 500
        data = response.get_json()
        assert 'LLM inference error' in data['error']

    def test_parse_malformed_llm_response_retries_then_fails(self, client):
        """When LLM returns unparseable text, should retry then return 500."""
        mock_llm = self._mock_llm("I cannot parse this transcription.")

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True), \
             patch('server.time.sleep'):
            response = client.post('/api/parse',
                json={'transcription': 'gibberish', 'section': 'Test'},
                content_type='application/json'
            )

        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data
        assert 'raw_response' in data
        assert data['attempts'] == 2

    def test_parse_retry_succeeds_on_second_attempt(self, client):
        """First attempt returns bad JSON, second attempt succeeds."""
        good_json = json.dumps({
            "front": "Q?", "back": "A.",
            "notes": "", "tags": ["test"]
        })

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": "not valid json"}}]},
            {"choices": [{"message": {"content": good_json}}]},
        ]

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True), \
             patch('server.time.sleep'):
            response = client.post('/api/parse',
                json={'transcription': 'Question 1...', 'section': 'Test'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data['front'] == 'Q?'

    def test_parse_fixes_missing_opening_quote(self, client):
        """extract_json auto-repairs missing opening quotes from LLM."""
        malformed = '{"front": What is the deltoid?", "back": "A muscle.", "notes": "", "tags": ["anatomy"]}'
        mock_llm = self._mock_llm(malformed)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 1...', 'section': 'Upper Limb'},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert "What is the deltoid?" in data['front']

    def test_parse_creates_tags_if_missing(self, client):
        """If LLM omits tags, parse should still inject section tag."""
        content = json.dumps({
            "front": "Q?", "back": "A.", "notes": ""
        })
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            response = client.post('/api/parse',
                json={'transcription': 'Question 1...', 'section': 'Thorax'},
                content_type='application/json'
            )

        data = response.get_json()
        assert 'thorax' in data['tags']

    def test_parse_calls_llm_with_correct_params(self, client, mock_llm_json_response):
        """Verify the LLM is called with system prompt and correct parameters."""
        content = mock_llm_json_response['choices'][0]['message']['content']
        mock_llm = self._mock_llm(content)

        with patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            client.post('/api/parse',
                json={'transcription': 'Test transcription', 'section': 'Test'},
                content_type='application/json'
            )

        call_args = mock_llm.create_chat_completion.call_args
        messages = call_args[1]['messages'] if 'messages' in call_args[1] else call_args[0][0]
        assert messages[0]['role'] == 'system'
        assert 'JSON parser' in messages[0]['content']
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'Test transcription'
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 1024


class TestSaveLoadRoundtrip:
    """Tests that save -> load preserves data exactly."""

    def test_roundtrip(self, client, sample_entries, data_file):
        client.post('/api/save',
            json={'entries': sample_entries},
            content_type='application/json'
        )

        response = client.get('/api/load')
        data = response.get_json()

        assert len(data['entries']) == len(sample_entries)
        for saved, original in zip(data['entries'], sample_entries):
            assert saved['id'] == original['id']
            assert saved['number'] == original['number']
            assert saved['section'] == original['section']
            assert saved['result'] == original['result']
            assert saved['front'] == original['front']
            assert saved['back'] == original['back']
            assert saved['tags'] == original['tags']
