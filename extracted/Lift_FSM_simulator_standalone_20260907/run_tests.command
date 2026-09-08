#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required to run the tests."
  exit 1
fi

node tests/original_fsm_core.test.js
node tests/drive_error_model.test.js
node tests/estimation_error_model.test.js
node tests/hyperparameter_search.test.js
node tests/browser_bootstrap_smoke.js
