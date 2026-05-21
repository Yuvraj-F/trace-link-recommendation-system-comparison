#!/bin/bash
set -e
cd "$(dirname "$0")"

ENV_DIR=".venv"

PYTHON="$ENV_DIR/bin/python"

MAIN="package/main.py"

if [ ! -f "$ENV_DIR/bin/python" ]; then
    echo "Python environment Not Found"
    echo "Creating new environment..."

    python3 -m venv "$ENV_DIR"

    "$ENV_DIR/bin/python" -m pip install --upgrade pip
fi

"$PYTHON" -m pip install -r requirements.txt

"$PYTHON" "$MAIN"