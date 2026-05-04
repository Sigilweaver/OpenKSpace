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
#   BINARY     - openkspace CLI path (default: ./target/release/openkspace)
# Python dependencies are managed via scripts/pyproject.toml (uv).

set -euo pipefail

BINARY="${BINARY:-./target/release/openkspace}"

if [[ ! -x "$BINARY" ]]; then
    echo "error: openkspace binary not found at $BINARY" >&2
    echo "       run 'cargo build --release' first, or set BINARY=..." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --project "$SCRIPT_DIR" "$SCRIPT_DIR/validate.py" --binary "$BINARY" "$@"
