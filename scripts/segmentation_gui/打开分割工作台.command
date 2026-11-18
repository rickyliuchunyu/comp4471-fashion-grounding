#!/bin/zsh
set -e
cd "$(dirname "$0")/../.."
if /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:8765/api/gallery | .venv-segmentation/bin/python -c 'import json,sys; sys.exit(0 if "categories" in json.load(sys.stdin) else 1)' 2>/dev/null; then
    open http://127.0.0.1:8765/
else
    exec .venv-segmentation/bin/python scripts/segmentation_gui/serve.py --open
fi
