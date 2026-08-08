#!/bin/bash
# Build a timestamped PDF of the project book from the Markdown chapters.
#
# The .md files are the SOURCE OF TRUTH. Never edit a generated PDF or HTML —
# edit the Markdown and re-run this script. Each run writes a new file stamped
# with the build date and time into build/ (which is gitignored).
#
#   ./build_pdf.sh
#
# On first run this creates a local .venv/ with the one Python package needed
# (markdown). That takes ~15 seconds; afterwards it is instant.

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"

# macOS/Linux put the interpreter in .venv/bin; Windows (Git Bash) in .venv/Scripts.
if [ -d "$VENV/Scripts" ]; then
  PY="$VENV/Scripts/python.exe"
else
  PY="$VENV/bin/python"
fi

if [ ! -x "$PY" ]; then
  echo "First run — creating $VENV and installing 'markdown'..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$VENV"
  else
    python -m venv "$VENV"          # Windows names it just 'python'
  fi
  if [ -d "$VENV/Scripts" ]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet markdown
  echo "Done."
  echo
fi

exec "$PY" build_pdf.py "$@"
