#!/usr/bin/env bash
# Merge harmorust into NebilI/NebilI.r-universe.dev without dropping seurust.
# Usage:
#   R_UNIVERSE_REGISTRY_TOKEN=... ./scripts/sync-r-universe.sh [tag]
# If tag is omitted, uses v<Version> from r/harmorust/DESCRIPTION.
set -euo pipefail

ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
VERSION=$(awk '/^Version:[[:space:]]*/ { print $2; exit }' "$ROOT/r/harmorust/DESCRIPTION")
if [ -z "${VERSION}" ]; then
  echo "Could not read Version from r/harmorust/DESCRIPTION" >&2
  exit 1
fi

TAG="${1:-v${VERSION}}"
TOKEN="${R_UNIVERSE_REGISTRY_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  echo "R_UNIVERSE_REGISTRY_TOKEN is not set" >&2
  exit 1
fi

OWNER="${GITHUB_OWNER:-NebilI}"
REGISTRY_REPO="${R_UNIVERSE_REGISTRY_REPO:-${OWNER}/${OWNER}.r-universe.dev}"
PKG_URL="${HARMORUST_GIT_URL:-https://github.com/${OWNER}/harmony-rs}"
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

git clone "https://x-access-token:${TOKEN}@github.com/${REGISTRY_REPO}.git" "$WORKDIR/registry"
cd "$WORKDIR/registry"

python3 - "$TAG" "$PKG_URL" <<'PY'
import json
import sys

tag = sys.argv[1]
url = sys.argv[2]
try:
    with open("packages.json", encoding="utf-8") as fh:
        pkgs = json.load(fh)
except FileNotFoundError:
    pkgs = []

if not isinstance(pkgs, list):
    raise SystemExit("packages.json must be a JSON array")

entry = {
    "package": "harmorust",
    "url": url,
    "subdir": "r/harmorust",
    "branch": tag,
}

pkgs = [p for p in pkgs if not (isinstance(p, dict) and p.get("package") == "harmorust")]
pkgs.append(entry)
pkgs.sort(key=lambda p: (0 if p.get("package") == "seurust" else 1, p.get("package", "")))

with open("packages.json", "w", encoding="utf-8") as fh:
    json.dump(pkgs, fh, indent=2)
    fh.write("\n")
PY

git config user.email "github-actions[bot]@users.noreply.github.com"
git config user.name "github-actions[bot]"
git add packages.json
if git diff --cached --quiet; then
  echo "Registry already points harmorust at ${TAG}"
else
  git commit -m "Point harmorust at ${TAG}"
  git push
  echo "Updated ${REGISTRY_REPO} harmorust branch -> ${TAG}"
fi
