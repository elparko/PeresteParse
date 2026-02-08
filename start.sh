#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Check if Ollama is running, start if needed
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
fi

# Activate virtual environment and start Flask server
echo "Starting Anatomix server..."
source .venv/bin/activate
python server.py
