#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
exec "$PY" train.py "$@"
