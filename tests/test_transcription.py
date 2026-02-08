"""Tests for the speech-to-text (recording + transcription) endpoints."""

import io
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import server


class TestRecordStart:
    """Tests for POST /api/record/start."""

    def test_start_recording(self, client):
        mock_stream = MagicMock()
        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict('sys.modules', {'sounddevice': mock_sd}), \
             patch.object(server, '_recording', False), \
             patch.object(server, '_recorded_frames', []):
            response = client.post('/api/record/start')

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'recording'

    def test_start_recording_already_recording(self, client):
        with patch.object(server, '_recording', True):
            response = client.post('/api/record/start')

        assert response.status_code == 409
        data = response.get_json()
        assert 'Already recording' in data['error']

    def test_start_recording_device_error(self, client):
        mock_sd = MagicMock()
        mock_sd.InputStream.side_effect = RuntimeError("No audio device found")

        with patch.dict('sys.modules', {'sounddevice': mock_sd}), \
             patch.object(server, '_recording', False):
            response = client.post('/api/record/start')

        assert response.status_code == 500
        data = response.get_json()
        assert 'Failed to start recording' in data['error']


class TestRecordStop:
    """Tests for POST /api/record/stop."""

    def test_stop_recording_and_transcribe(self, client):
        # Simulate recorded audio frames (1 second of silence)
        frames = [np.zeros((1600, 1), dtype='float32') for _ in range(10)]
        mock_stream = MagicMock()

        mock_stt = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Question 42, I got it correct."
        mock_stt.transcribe.return_value = mock_result

        with patch.object(server, '_recording', True), \
             patch.object(server, '_recorded_frames', frames), \
             patch('server.get_stt', return_value=mock_stt):
            server.app.config['_audio_stream'] = mock_stream
            response = client.post('/api/record/stop')

        assert response.status_code == 200
        data = response.get_json()
        assert data['text'] == "Question 42, I got it correct."
        assert data['duration'] == 1.0
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        # Verify parakeet-mlx was called with a file path (string)
        call_args = mock_stt.transcribe.call_args[0][0]
        assert isinstance(call_args, str)
        assert call_args.endswith('.wav')

    def test_stop_not_recording(self, client):
        with patch.object(server, '_recording', False):
            response = client.post('/api/record/stop')

        assert response.status_code == 409
        data = response.get_json()
        assert 'Not currently recording' in data['error']

    def test_stop_no_frames(self, client):
        with patch.object(server, '_recording', True), \
             patch.object(server, '_recorded_frames', []):
            server.app.config['_audio_stream'] = MagicMock()
            response = client.post('/api/record/stop')

        assert response.status_code == 400
        data = response.get_json()
        assert 'No audio recorded' in data['error']

    def test_stop_too_short(self, client):
        # Only ~0.1 seconds of audio
        frames = [np.zeros((160, 1), dtype='float32')]

        with patch.object(server, '_recording', True), \
             patch.object(server, '_recorded_frames', frames):
            server.app.config['_audio_stream'] = MagicMock()
            response = client.post('/api/record/stop')

        assert response.status_code == 400
        data = response.get_json()
        assert 'too short' in data['error']

    def test_stop_stt_not_available(self, client):
        frames = [np.zeros((8000, 1), dtype='float32') for _ in range(2)]

        with patch.object(server, '_recording', True), \
             patch.object(server, '_recorded_frames', frames), \
             patch('server.get_stt', return_value=None), \
             patch.object(server, '_download_status', {'llm': 'ready', 'stt': 'missing'}):
            server.app.config['_audio_stream'] = MagicMock()
            response = client.post('/api/record/stop')

        assert response.status_code == 503
        data = response.get_json()
        assert 'STT model not available' in data['error']


class TestTranscribeEndpoint:
    """Tests for POST /api/transcribe (file upload)."""

    def test_transcribe_wav_file(self, client):
        import soundfile as sf

        # Create a synthetic WAV in memory
        audio_data = np.zeros(16000, dtype='float32')  # 1 second of silence
        buf = io.BytesIO()
        sf.write(buf, audio_data, 16000, format='WAV')
        buf.seek(0)

        mock_stt = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Transcribed text here."
        mock_stt.transcribe.return_value = mock_result

        with patch('server.get_stt', return_value=mock_stt):
            response = client.post('/api/transcribe',
                data={'audio': (buf, 'test.wav')},
                content_type='multipart/form-data'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data['text'] == 'Transcribed text here.'
        assert data['duration'] == 1.0
        # Verify parakeet-mlx was called with a file path (string)
        call_args = mock_stt.transcribe.call_args[0][0]
        assert isinstance(call_args, str)
        assert call_args.endswith('.wav')

    def test_transcribe_no_file(self, client):
        response = client.post('/api/transcribe')
        assert response.status_code == 400
        data = response.get_json()
        assert 'No audio file' in data['error']

    def test_transcribe_stt_not_available(self, client):
        import soundfile as sf

        audio_data = np.zeros(16000, dtype='float32')
        buf = io.BytesIO()
        sf.write(buf, audio_data, 16000, format='WAV')
        buf.seek(0)

        with patch('server.get_stt', return_value=None), \
             patch.object(server, '_download_status', {'llm': 'ready', 'stt': 'missing'}):
            response = client.post('/api/transcribe',
                data={'audio': (buf, 'test.wav')},
                content_type='multipart/form-data'
            )

        assert response.status_code == 503


class TestGetStt:
    """Tests for get_stt() lazy loading."""

    def test_returns_cached_stt(self):
        mock_stt = MagicMock()
        with patch.object(server, '_stt', mock_stt):
            result = server.get_stt()
        assert result is mock_stt

    def test_loads_model_on_first_call(self):
        mock_model = MagicMock()
        mock_from_pretrained = MagicMock(return_value=mock_model)

        with patch.object(server, '_stt', None), \
             patch.object(server, '_download_status', {'llm': 'ready', 'stt': 'unknown'}), \
             patch.dict('sys.modules', {'parakeet_mlx': MagicMock(from_pretrained=mock_from_pretrained)}):
            result = server.get_stt()

        assert result is mock_model

    def test_returns_none_on_error(self):
        mock_from_pretrained = MagicMock(side_effect=RuntimeError("Download failed"))
        status = {'llm': 'ready', 'stt': 'unknown'}

        with patch.object(server, '_stt', None), \
             patch.object(server, '_download_status', status), \
             patch.dict('sys.modules', {'parakeet_mlx': MagicMock(from_pretrained=mock_from_pretrained)}):
            result = server.get_stt()
            assert result is None
            assert server._download_status['stt'] == 'error'
