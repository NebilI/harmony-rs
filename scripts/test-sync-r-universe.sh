#!/usr/bin/env bash
# Regression checks for scripts/sync-r-universe.sh.
# Ensures harmorust upserts never drop or rewrite seurust (or other neighbors).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
sync="$root/scripts/sync-r-universe.sh"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

pass=0
fail=0

assert_eq() {
  local label="$1" got="$2" want="$3"
  if [[ "$got" == "$want" ]]; then
    echo "PASS: $label"
    pass=$((pass + 1))
  else
    echo "FAIL: $label" >&2
    echo "  got:  $got" >&2
    echo "  want: $want" >&2
    fail=$((fail + 1))
  fi
}

# Shared registry shaped like production: seurust + harmorust.
cat >"$tmpdir/before.json" <<'JSON'
[
  {
    "package": "seurust",
    "url": "https://github.com/NebilI/seurust",
    "subdir": "seurust",
    "branch": "v0.3.0"
  },
  {
    "package": "harmorust",
    "url": "https://github.com/NebilI/harmony-rs",
    "subdir": "r/harmorust",
    "branch": "v0.1.0"
  }
]
JSON

"$sync" v0.1.1 \
  --from-file "$tmpdir/before.json" \
  --to-file "$tmpdir/after.json"

python3 - "$tmpdir/before.json" "$tmpdir/after.json" <<'PY'
import json, sys
before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))
b = {p["package"]: p for p in before}
a = {p["package"]: p for p in after}
assert list(a) == ["seurust", "harmorust"], list(a)
assert a["harmorust"]["branch"] == "v0.1.1"
assert a["seurust"] == b["seurust"], (a["seurust"], b["seurust"])
print("OK upsert preserves seurust bit-for-bit")
PY
assert_eq "seurust still present after harmorust bump" \
  "$(python3 -c 'import json,sys; print("seurust" in {p["package"] for p in json.load(open(sys.argv[1]))})' "$tmpdir/after.json")" \
  "True"

# Idempotent when already on the tag.
"$sync" v0.1.1 \
  --from-file "$tmpdir/after.json" \
  --to-file "$tmpdir/again.json"
assert_eq "idempotent upsert leaves JSON unchanged" \
  "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))==json.load(open(sys.argv[2])))' "$tmpdir/after.json" "$tmpdir/again.json")" \
  "True"

# Missing packages.json (404 path) creates harmorust-only registry.
"$sync" v0.1.1 \
  --from-file "$tmpdir/does-not-exist.json" \
  --to-file "$tmpdir/fresh.json"
assert_eq "404 path creates harmorust-only array" \
  "$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d)==1 and d[0]["package"]=="harmorust")' "$tmpdir/fresh.json")" \
  "True"

# Empty / non-array must refuse (would otherwise risk wiping neighbors).
printf '%s\n' '{}' >"$tmpdir/object.json"
if "$sync" v0.1.1 \
    --from-file "$tmpdir/object.json" \
    --to-file "$tmpdir/bad-out.json" 2>"$tmpdir/bad.err"; then
  echo "FAIL: object packages.json should have been rejected" >&2
  fail=$((fail + 1))
else
  if grep -q 'must be a JSON array' "$tmpdir/bad.err"; then
    echo "PASS: rejects non-array packages.json"
    pass=$((pass + 1))
  else
    echo "FAIL: unexpected error for object packages.json" >&2
    cat "$tmpdir/bad.err" >&2
    fail=$((fail + 1))
  fi
fi

echo
echo "Passed: $pass  Failed: $fail"
[[ "$fail" -eq 0 ]]
