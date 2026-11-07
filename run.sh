#!/usr/bin/env bash
# Start LungLocAI on http://localhost:8000
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt
fi
echo "Open http://localhost:8000 in your browser"
./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
