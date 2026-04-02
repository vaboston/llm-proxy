#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PORT="${PORT:-11435}"

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate and install deps
source "$VENV_DIR/bin/activate"

if [ ! -f "$VENV_DIR/.installed" ] || [ "$SCRIPT_DIR/requirements.txt" -nt "$VENV_DIR/.installed" ]; then
    echo "Installing dependencies..."
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
    touch "$VENV_DIR/.installed"
fi

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "0.0.0.0")
echo "Starting LLM Proxy on port $PORT..."
echo "  UI:     http://${HOST_IP}:$PORT"
echo "  Ollama: http://${HOST_IP}:$PORT/api/chat"
echo "  OpenAI: http://${HOST_IP}:$PORT/v1/chat/completions"
echo ""

cd "$SCRIPT_DIR"
exec "$VENV_DIR/bin/uvicorn" app.main:app --host 0.0.0.0 --port "$PORT" --log-level warning
