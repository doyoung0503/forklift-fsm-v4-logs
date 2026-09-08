#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PORT="${PORT:-8765}"
URL="http://127.0.0.1:${PORT}/"
LOG_FILE="${TMPDIR:-/tmp}/lift_fsm_simulator_${PORT}.log"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 is required. Install it from https://www.python.org/downloads/macos/."
  read -r "?Press Return to close..."
  exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -m http.server "$PORT" --bind 127.0.0.1 >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in {1..50}; do
  if curl --silent --fail "$URL" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Failed to start the simulator server."
    sed -n '1,120p' "$LOG_FILE"
    read -r "?Press Return to close..."
    exit 1
  fi
  sleep 0.1
done

if ! curl --silent --fail "$URL" >/dev/null 2>&1; then
  echo "The simulator server did not become ready: $URL"
  read -r "?Press Return to close..."
  exit 1
fi

echo "Pallet Pickup FSM Simulator"
echo "Open: $URL"
echo "Press Ctrl+C to stop the server."
open "$URL"
wait "$SERVER_PID"

