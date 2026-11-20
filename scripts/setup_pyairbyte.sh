#!/usr/bin/env bash
# Isolated PyAirbyte venv so the demo venv (cognee/pydantic) is never touched.
set -e
cd /home/user
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "python: $PY $($PY --version)"
[ -d .venv-airbyte ] || $PY -m venv .venv-airbyte
.venv-airbyte/bin/pip install -q --upgrade pip
.venv-airbyte/bin/pip install -q airbyte
.venv-airbyte/bin/python -c "import airbyte; print('pyairbyte', airbyte.__version__ if hasattr(airbyte,'__version__') else 'ok')"
echo SETUP_DONE
