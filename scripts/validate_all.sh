#!/usr/bin/env bash
# Batch-validation wrapper for CI.
#
# Usage:
#   scripts/validate_all.sh [--slice N] [--threshold F] [--report path] <dir>
#
# Runs scripts/validate.py recursively over every .h5 file under <dir>,
# writes a JSON report, prints a summary table, and exits non-zero if
# any file fails the SSIM threshold.
#
# Environment:
#   PYTHON     - python interpreter (default: python3)
#   BINARY     - openkspace CLI path (default: ./target/release/openkspace)

set -euo pipefail

PYTHON="${PYTHON:-python3}"
BINARY="${BINARY:-./target/release/openkspace}"

if [[ ! -x "$BINARY" ]]; then
    echo "error: openkspace binary not found at $BINARY" >&2
    echo "       run 'cargo build --release' first, or set BINARY=..." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PYTHON" "$SCRIPT_DIR/validate.py" --binary "$BINARY" "$@"
