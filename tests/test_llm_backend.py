"""Tests for the LLM backend (model management, loading, download)."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import server


class TestGetModelPath:
    """Tests for get_model_path()."""

    def test_returns_path_in_models_dir(self, models_dir):
        with patch.object(server, 'MODELS_DIR', models_dir):
            path = server.get_model_path()
        assert path.parent == models_dir
        assert path.name == server.DEFAULT_CONFIG['llm_filename']


class TestCheckModelStatus:
    """Tests for check_model_status()."""

    def test_status_missing_when_no_model(self, models_dir):
        with patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_llm', None), \
             patch.object(server, '_download_status', {'llm': 'unknown'}):
            status = server.check_model_status()
        assert status['llm'] == 'missing'

    def test_status_ready_when_model_file_exists(self, models_dir):
        (models_dir / server.DEFAULT_CONFIG['llm_filename']).write_bytes(b'fake')
        with patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_llm', None), \
             patch.object(server, '_download_status', {'llm': 'unknown'}):
            status = server.check_model_status()
        assert status['llm'] == 'ready'

    def test_status_ready_when_llm_loaded(self, models_dir):
        mock_llm = MagicMock()
        with patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_llm', mock_llm), \
             patch.object(server, '_download_status', {'llm': 'unknown'}):
            status = server.check_model_status()
        assert status['llm'] == 'ready'

    def test_status_downloading_preserved(self, models_dir):
        with patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_llm', None), \
             patch.object(server, '_download_status', {'llm': 'downloading'}):
            status = server.check_model_status()
        assert status['llm'] == 'downloading'


class TestGetLlm:
    """Tests for get_llm() lazy loading."""

    def test_returns_cached_llm(self):
        mock_llm = MagicMock()
        with patch.object(server, '_llm', mock_llm):
            result = server.get_llm()
        assert result is mock_llm

    def test_returns_none_when_model_missing(self, models_dir):
        with patch.object(server, '_llm', None), \
             patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_download_status', {'llm': 'unknown'}):
            result = server.get_llm()
        assert result is None

    def test_loads_model_when_file_exists(self, models_dir):
        (models_dir / server.DEFAULT_CONFIG['llm_filename']).write_bytes(b'fake model')
        mock_llama_cls = MagicMock()
        mock_llama_instance = MagicMock()
        mock_llama_cls.return_value = mock_llama_instance

        with patch.object(server, '_llm', None), \
             patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_download_status', {'llm': 'unknown'}), \
             patch.dict('sys.modules', {'llama_cpp': MagicMock(Llama=mock_llama_cls)}):
            result = server.get_llm()

        assert result is mock_llama_instance

    def test_returns_none_on_load_error(self, models_dir):
        (models_dir / server.DEFAULT_CONFIG['llm_filename']).write_bytes(b'corrupt')

        with patch.object(server, '_llm', None), \
             patch.object(server, 'MODELS_DIR', models_dir), \
             patch.object(server, '_download_status', {'llm': 'unknown'}), \
             patch.dict('sys.modules', {'llama_cpp': MagicMock(Llama=MagicMock(side_effect=RuntimeError("bad model")))}):
            result = server.get_llm()

        assert result is None


class TestDownloadModelBackground:
    """Tests for download_model_background()."""

    def test_download_sets_status_downloading_then_ready(self):
        mock_download = MagicMock()
        statuses = []

        original_status = server._download_status.copy()

        def track_status(*args, **kwargs):
            statuses.append(server._download_status['llm'])

        mock_download.side_effect = track_status

        with patch.object(server, '_download_status', {'llm': 'missing'}), \
             patch.dict('sys.modules', {'huggingface_hub': MagicMock(hf_hub_download=mock_download)}):
            server.download_model_background()
            assert 'downloading' in statuses
            assert server._download_status['llm'] == 'ready'

    def test_download_sets_error_on_failure(self):
        mock_download = MagicMock(side_effect=Exception("Network error"))

        with patch.object(server, '_download_status', {'llm': 'missing'}), \
             patch.dict('sys.modules', {'huggingface_hub': MagicMock(hf_hub_download=mock_download)}):
            server.download_model_background()
            assert server._download_status['llm'] == 'error'
            assert 'Network error' in server._download_status['llm_error']


class TestGbnfGrammar:
    """Tests for the GBNF grammar constant."""

    def test_grammar_string_is_defined(self):
        assert server.PARSE_GRAMMAR is not None
        assert 'root' in server.PARSE_GRAMMAR
        assert 'string' in server.PARSE_GRAMMAR
        assert 'tag-arr' in server.PARSE_GRAMMAR

    def test_grammar_loads_in_llama_cpp(self):
        """Verify the grammar is valid by loading it with LlamaGrammar."""
        try:
            from llama_cpp import LlamaGrammar
            grammar = LlamaGrammar.from_string(server.PARSE_GRAMMAR)
            assert grammar is not None
        except ImportError:
            pytest.skip("llama-cpp-python not installed")
