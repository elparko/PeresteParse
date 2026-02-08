#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python app.py &
exit 0
