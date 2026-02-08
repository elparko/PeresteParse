#!/usr/bin/env python3
"""
Anatomix - Standalone Native App
Launches Flask server and opens native macOS window
"""
import webview
import threading
import time
import os
import signal
from server import app

def start_server():
    """Start Flask server in background thread"""
    app.run(host='127.0.0.1', port=5111, debug=False, use_reloader=False)

def on_closing():
    """Cleanup when window closes"""
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':
    # Start Flask in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    # Create native window
    window = webview.create_window(
        'Anatomix',
        'http://127.0.0.1:5111',
        width=1400,
        height=900,
        resizable=True,
        fullscreen=False,
        min_size=(800, 600),
        background_color='#1a1a1a'
    )

    # Register cleanup on window close
    window.events.closing += on_closing

    webview.start()
