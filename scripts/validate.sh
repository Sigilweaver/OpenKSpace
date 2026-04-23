#!/usr/bin/env bash
# Validate the OpenKSpace reconstruction against a numpy reference.
#
# Usage: scripts/validate.sh <file.h5> [--slice N] [--threshold 0.95]
#
# Builds the release binary if missing, then hands off to validate.py.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x target/release/openkspace ]]; then
    echo "Building release binary..."
    cargo build --release
fi

exec python3 scripts/validate.py "$@"
