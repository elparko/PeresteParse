"""Tests for Phase 6: First-Run Experience (setup_complete flag)."""

import json
import pytest
import server


class TestSetupCompleteDefault:
    """Verify setup_complete exists in DEFAULT_CONFIG and load_config."""

    def test_default_config_has_setup_complete(self):
        assert 'setup_complete' in server.DEFAULT_CONFIG
        assert server.DEFAULT_CONFIG['setup_complete'] is False

    def test_load_config_returns_setup_complete(self, client):
        config = server.load_config()
        assert 'setup_complete' in config
        assert config['setup_complete'] is False

    def test_old_config_gets_default(self, client, tmp_path):
        """An existing config without setup_complete gets the default value."""
        config_file = tmp_path / '.peresteparse' / 'config.json'
        config_file.write_text(json.dumps({
            'parsing_backend': 'cloud',
            'api_key': 'sk-test-1234',
        }))
        config = server.load_config()
        assert config['setup_complete'] is False
        assert config['parsing_backend'] == 'cloud'


class TestSetupCompleteEndpoints:
    """Verify GET/POST /api/config handles setup_complete."""

    def test_get_config_returns_setup_complete(self, client):
        res = client.get('/api/config')
        data = res.get_json()
        assert 'setup_complete' in data
        assert data['setup_complete'] is False

    def test_post_config_can_set_setup_complete(self, client):
        res = client.post('/api/config',
                          json={'setup_complete': True},
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert data['setup_complete'] is True

    def test_setup_complete_persists(self, client):
        client.post('/api/config',
                     json={'setup_complete': True},
                     content_type='application/json')
        res = client.get('/api/config')
        data = res.get_json()
        assert data['setup_complete'] is True

    def test_setup_complete_alongside_other_settings(self, client):
        res = client.post('/api/config',
                          json={
                              'setup_complete': True,
                              'parsing_backend': 'cloud',
                              'api_key': 'sk-ant-test-key-12345678',
                          },
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert data['setup_complete'] is True
        assert data['parsing_backend'] == 'cloud'
        # API key should be masked in response
        assert 'sk-ant-test-key-12345678' not in data['api_key']

    def test_setup_complete_false_by_default_on_fresh_config(self, client):
        """GET /api/config on a fresh install returns setup_complete=False."""
        res = client.get('/api/config')
        assert res.get_json()['setup_complete'] is False
