#!/usr/bin/env bash
# Repeatable harmorust vs immunogenomics/harmony timing table.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if [[ -z "${R_LIBS_USER:-}" && -d "${HOME}/R/library/harmony" ]]; then
  export R_LIBS_USER="${HOME}/R/library"
fi
exec cargo run -p harmorust --release -- compare "$@"
