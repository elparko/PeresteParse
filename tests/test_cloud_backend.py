"""Tests for cloud API backend (Anthropic) and config management."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import server


class TestLoadConfig:
    """Tests for load_config()."""

    def test_returns_defaults_when_no_file(self, tmp_path):
        config_file = tmp_path / 'config.json'
        with patch.object(server, 'CONFIG_FILE', config_file):
            config = server.load_config()
        assert config == server.DEFAULT_CONFIG
        assert config['parsing_backend'] == 'local'

    def test_loads_existing_config(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test-key',
        }))
        with patch.object(server, 'CONFIG_FILE', config_file):
            config = server.load_config()
        assert config['parsing_backend'] == 'cloud'
        assert config['api_key'] == 'sk-test-key'

    def test_fills_missing_keys_with_defaults(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({'parsing_backend': 'cloud'}))
        with patch.object(server, 'CONFIG_FILE', config_file):
            config = server.load_config()
        assert config['parsing_backend'] == 'cloud'
        assert config['cloud_model'] == server.DEFAULT_CONFIG['cloud_model']
        assert config['api_key'] == ''

    def test_returns_defaults_on_corrupt_file(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text('not json')
        with patch.object(server, 'CONFIG_FILE', config_file):
            config = server.load_config()
        assert config == server.DEFAULT_CONFIG


class TestSaveConfig:
    """Tests for save_config()."""

    def test_saves_config_to_file(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config = {'parsing_backend': 'cloud', 'api_key': 'sk-123'}
        with patch.object(server, 'CONFIG_FILE', config_file):
            server.save_config(config)
        saved = json.loads(config_file.read_text())
        assert saved['parsing_backend'] == 'cloud'
        assert saved['api_key'] == 'sk-123'


class TestMaskApiKey:
    """Tests for mask_api_key()."""

    def test_masks_long_key(self):
        assert server.mask_api_key('sk-ant-api-1234567890abcdef') == 'sk-...cdef'

    def test_returns_empty_for_short_key(self):
        assert server.mask_api_key('short') == ''

    def test_returns_empty_for_empty_key(self):
        assert server.mask_api_key('') == ''

    def test_returns_empty_for_none(self):
        assert server.mask_api_key(None) == ''


class TestConfigEndpoints:
    """Tests for GET/POST /api/config."""

    def test_get_config_returns_defaults(self, client):
        res = client.get('/api/config')
        assert res.status_code == 200
        data = res.get_json()
        assert data['parsing_backend'] == 'local'
        assert data['api_key'] == ''  # masked (empty)

    def test_get_config_masks_api_key(self, client, tmp_path):
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-ant-api-1234567890abcdef',
        }))
        with patch.object(server, 'CONFIG_FILE', config_file):
            res = client.get('/api/config')
        data = res.get_json()
        assert data['api_key'] == 'sk-...cdef'
        assert 'sk-ant-api' not in data['api_key']

    def test_post_config_saves_settings(self, client):
        res = client.post('/api/config',
                          json={'parsing_backend': 'cloud', 'api_key': 'sk-test-key-12345678'})
        assert res.status_code == 200
        data = res.get_json()
        assert data['parsing_backend'] == 'cloud'
        # Key should be masked in response
        assert data['api_key'] == 'sk-...5678'

    def test_post_config_rejects_invalid_backend(self, client):
        res = client.post('/api/config', json={'parsing_backend': 'invalid'})
        assert res.status_code == 400
        assert 'error' in res.get_json()

    def test_post_config_ignores_unknown_keys(self, client):
        res = client.post('/api/config', json={'unknown_key': 'value'})
        assert res.status_code == 200
        data = res.get_json()
        assert 'unknown_key' not in data


class TestCloudParse:
    """Tests for cloud_parse() function."""

    def test_cloud_parse_returns_parsed_json(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test-key-12345678',
            'cloud_model': 'claude-sonnet-4-5-20250929',
            'cloud_provider': 'anthropic',
        }))

        expected = {
            "front": "What is the longest bone?",
            "back": "The femur.",
            "notes": "The femur is the longest and strongest bone.",
            "tags": ["lower_limb"]
        }

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(expected))]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.object(server, 'CONFIG_FILE', config_file), \
             patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            result = server.cloud_parse("Question 1 correct. The femur is the longest bone.")

        assert result['front'] == "What is the longest bone?"
        assert result['back'] == 'The femur.'

    def test_cloud_parse_raises_without_api_key(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': '',
        }))

        with patch.object(server, 'CONFIG_FILE', config_file):
            with pytest.raises(ValueError, match="No API key"):
                server.cloud_parse("test transcription")


class TestParseEndpointCloud:
    """Tests for /api/parse routing to cloud backend."""

    def test_parse_routes_to_cloud(self, client, tmp_path):
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test-key-12345678',
            'cloud_model': 'claude-sonnet-4-5-20250929',
            'cloud_provider': 'anthropic',
        }))

        expected = {
            "front": "What is the longest bone?",
            "back": "The femur.",
            "notes": "Notes here.",
            "tags": ["lower_limb"]
        }

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(expected))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.object(server, 'CONFIG_FILE', config_file), \
             patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            res = client.post('/api/parse', json={
                'transcription': 'Question 1 correct, the femur.',
                'section': 'Lower Limb'
            })

        assert res.status_code == 200
        data = res.get_json()
        assert data['section'] == 'Lower Limb'
        assert 'lower_limb' in data['tags']

    def test_parse_cloud_missing_api_key(self, client, tmp_path):
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': '',
        }))

        with patch.object(server, 'CONFIG_FILE', config_file):
            res = client.post('/api/parse', json={
                'transcription': 'test',
                'section': 'Test'
            })

        assert res.status_code == 400
        assert 'API key' in res.get_json()['error']

    def test_parse_cloud_api_error(self, client, tmp_path):
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test-key-12345678',
            'cloud_model': 'claude-sonnet-4-5-20250929',
            'cloud_provider': 'anthropic',
        }))

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API rate limited")
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.object(server, 'CONFIG_FILE', config_file), \
             patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            res = client.post('/api/parse', json={
                'transcription': 'test',
                'section': 'Test'
            })

        assert res.status_code == 500
        assert 'Cloud API error' in res.get_json()['error']


class TestParseEndpointLocal:
    """Tests for /api/parse still working with local backend."""

    def test_parse_local_when_configured(self, client, tmp_path):
        """Verify local parsing still works when config says 'local'."""
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({'parsing_backend': 'local'}))

        expected = json.dumps({
            "front": "What is tested?",
            "back": "Local parsing.",
            "notes": "Notes.",
            "tags": ["test"]
        })

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": expected}}]
        }

        with patch.object(server, 'CONFIG_FILE', config_file), \
             patch.object(server, '_llm', mock_llm), \
             patch('server.LlamaGrammar', create=True):
            res = client.post('/api/parse', json={
                'transcription': 'test transcription',
                'section': 'Test'
            })

        assert res.status_code == 200
        data = res.get_json()
        assert data['section'] == 'Test'


class TestHealthCloud:
    """Tests for /api/health with cloud backend."""

    def test_health_cloud_ok_with_api_key(self, client, tmp_path):
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test-key-12345678',
            'cloud_model': 'claude-sonnet-4-5-20250929',
            'cloud_provider': 'anthropic',
        }))

        with patch.object(server, 'CONFIG_FILE', config_file):
            res = client.get('/api/health')

        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == 'ok'
        assert data['backend'] == 'cloud'
        assert data['provider'] == 'anthropic'

    def test_health_cloud_error_no_api_key(self, client, tmp_path):
        config_file = tmp_path / '.pereste' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': '',
        }))

        with patch.object(server, 'CONFIG_FILE', config_file):
            res = client.get('/api/health')

        assert res.status_code == 503
        data = res.get_json()
        assert data['status'] == 'error'
        assert data['backend'] == 'cloud'
        assert 'API key' in data['message']

    def test_health_local_still_works(self, client, models_dir):
        """Verify health endpoint still works for local backend."""
        filename = server.DEFAULT_CONFIG['llm_filename']
        (models_dir / filename).write_bytes(b'fake')
        with patch.object(server, '_llm', None), \
             patch.object(server, '_download_status', {'llm': 'unknown', 'stt': 'unknown'}):
            res = client.get('/api/health')
        assert res.status_code == 200
        data = res.get_json()
        assert data['backend'] == 'local'


class TestApplySection:
    """Tests for _apply_section() helper."""

    def test_applies_section_and_tag(self):
        data = {"tags": ["nerve"]}
        server._apply_section(data, "Upper Limb")
        assert data['section'] == 'Upper Limb'
        assert data['tags'][0] == 'upper_limb'

    def test_uses_default_section_when_empty(self):
        data = {"tags": []}
        server._apply_section(data, "")
        assert data['section'] == 'General'
        assert 'general' in data['tags']

    def test_does_not_duplicate_tag(self):
        data = {"tags": ["upper_limb", "nerve"]}
        server._apply_section(data, "Upper Limb")
        assert data['tags'].count('upper_limb') == 1

    def test_creates_tags_if_missing(self):
        data = {}
        server._apply_section(data, "Thorax")
        assert data['tags'] == ['thorax']


class TestGetSetting:
    """Tests for get_setting() — env var override and config fallback."""

    def test_returns_config_value(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({'llm_filename': 'custom-model.gguf'}))
        with patch.object(server, 'CONFIG_FILE', config_file):
            assert server.get_setting('llm_filename') == 'custom-model.gguf'

    def test_returns_default_when_no_config(self, tmp_path):
        config_file = tmp_path / 'config.json'
        with patch.object(server, 'CONFIG_FILE', config_file):
            assert server.get_setting('llm_filename') == server.DEFAULT_CONFIG['llm_filename']

    def test_env_var_overrides_config(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({'llm_filename': 'config-model.gguf'}))
        with patch.object(server, 'CONFIG_FILE', config_file), \
             patch.dict('os.environ', {'LLM_FILENAME': 'env-model.gguf'}):
            assert server.get_setting('llm_filename') == 'env-model.gguf'

    def test_int_casting_for_context_size(self, tmp_path):
        config_file = tmp_path / 'config.json'
        with patch.object(server, 'CONFIG_FILE', config_file), \
             patch.dict('os.environ', {'LLM_CONTEXT_SIZE': '4096'}):
            result = server.get_setting('llm_context_size')
            assert result == 4096
            assert isinstance(result, int)

    def test_config_context_size_is_int(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({'llm_context_size': 16384}))
        with patch.object(server, 'CONFIG_FILE', config_file):
            result = server.get_setting('llm_context_size')
            assert result == 16384
            assert isinstance(result, int)

    def test_all_new_default_keys(self, tmp_path):
        config_file = tmp_path / 'config.json'
        with patch.object(server, 'CONFIG_FILE', config_file):
            assert server.get_setting('llm_repo_id') == 'Qwen/Qwen3-4B-GGUF'
            assert server.get_setting('llm_filename') == 'Qwen3-4B-Q4_K_M.gguf'
            assert server.get_setting('llm_context_size') == 8192
            assert server.get_setting('stt_model_id') == 'mlx-community/parakeet-tdt-0.6b-v3'


class TestConfigEndpointsNewKeys:
    """Tests for POST /api/config with new local model keys."""

    def test_post_config_saves_llm_settings(self, client):
        res = client.post('/api/config', json={
            'llm_repo_id': 'custom/repo',
            'llm_filename': 'custom.gguf',
            'llm_context_size': 4096,
            'stt_model_id': 'custom/stt-model',
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data['llm_repo_id'] == 'custom/repo'
        assert data['llm_filename'] == 'custom.gguf'
        assert data['llm_context_size'] == 4096
        assert data['stt_model_id'] == 'custom/stt-model'

    def test_post_config_casts_context_size_string(self, client):
        res = client.post('/api/config', json={'llm_context_size': '2048'})
        assert res.status_code == 200
        data = res.get_json()
        assert data['llm_context_size'] == 2048

    def test_post_config_rejects_invalid_context_size(self, client):
        res = client.post('/api/config', json={'llm_context_size': 'not_a_number'})
        assert res.status_code == 400
        assert 'llm_context_size' in res.get_json()['error']

    def test_get_config_includes_new_keys(self, client):
        res = client.get('/api/config')
        assert res.status_code == 200
        data = res.get_json()
        assert 'llm_repo_id' in data
        assert 'llm_filename' in data
        assert 'llm_context_size' in data
        assert 'stt_model_id' in data


class TestLoadConfigNewKeys:
    """Tests for load_config() with new default keys."""

    def test_defaults_include_new_keys(self, tmp_path):
        config_file = tmp_path / 'config.json'
        with patch.object(server, 'CONFIG_FILE', config_file):
            config = server.load_config()
        assert config['llm_repo_id'] == 'Qwen/Qwen3-4B-GGUF'
        assert config['llm_filename'] == 'Qwen3-4B-Q4_K_M.gguf'
        assert config['llm_context_size'] == 8192
        assert config['stt_model_id'] == 'mlx-community/parakeet-tdt-0.6b-v3'

    def test_fills_new_keys_for_old_config(self, tmp_path):
        """Config file from Phase 4 (without new keys) gets defaults filled in."""
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test',
            'cloud_model': 'claude-sonnet-4-5-20250929',
        }))
        with patch.object(server, 'CONFIG_FILE', config_file):
            config = server.load_config()
        assert config['parsing_backend'] == 'cloud'
        assert config['llm_repo_id'] == 'Qwen/Qwen3-4B-GGUF'
        assert config['llm_context_size'] == 8192
