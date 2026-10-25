#!/usr/bin/env bash
# Numerical |harmorust − original Harmony| table (min / max abs diff).
# Pass --check to fail if a gated max |Δ| exceeds PARITY_MAX_ABS (1e-4).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if [[ -z "${R_LIBS_USER:-}" && -d "${HOME}/R/library/harmony" ]]; then
  export R_LIBS_USER="${HOME}/R/library"
fi
exec cargo run -p harmorust --release -- parity "$@"
