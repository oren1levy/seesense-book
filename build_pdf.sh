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

if [ ! -x "$VENV/bin/python" ]; then
  echo "First run — creating $VENV and installing 'markdown'..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet markdown
  echo "Done."
  echo
fi

exec "$VENV/bin/python" build_pdf.py "$@"
