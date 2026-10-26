#!/usr/bin/env bash
# Rewrite golden fixtures from the current faithful kernel.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export HARMORUST_REGEN_FIXTURES=1
cargo test -p harmorust --test fixtures -- --nocapture
echo "fixtures rewritten under crates/harmorust/tests/fixtures/"
