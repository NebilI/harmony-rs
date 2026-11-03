#!/usr/bin/env bash
# Upsert harmorust into the shared r-universe registry.
#
# NebilI/NebilI.r-universe.dev packages.json is one JSON array for every
# package in the universe. This script replaces only the object whose
# "package" is "harmorust" and keeps every other entry, in order. It writes a
# one-element array only when packages.json does not exist. If the fetch fails
# for any reason other than HTTP 404, or the file is not a JSON array, it
# exits without committing.
#
# Check a local file (no network, no git). A missing path is the 404 case:
#   RELEASE_TAG=v0.1.0 scripts/sync-r-universe.sh \
#     --from-file current.json --to-file updated.json
#
# Update the registry (same token as publish-harmorust-r.yml):
#   R_UNIVERSE_REGISTRY_TOKEN=... ./scripts/sync-r-universe.sh [tag]
#
# PACKAGES_JSON_URL overrides the fetch URL and disables clone/push unless
# SYNC_R_UNIVERSE_ALLOW_PUSH=1. The workflow does not set either variable.
set -euo pipefail

PACKAGE="harmorust"
PACKAGE_URL="${HARMORUST_GIT_URL:-https://github.com/NebilI/harmony-rs}"
PACKAGE_SUBDIR="r/harmorust"
OWNER="${GITHUB_OWNER:-NebilI}"
REGISTRY_SLUG="${R_UNIVERSE_REGISTRY_REPO:-${OWNER}/${OWNER}.r-universe.dev}"
UNIVERSE_TOKEN="${UNIVERSE_TOKEN:-${R_UNIVERSE_REGISTRY_TOKEN:-}}"

FROM_FILE=""
TO_FILE=""
have_from=0
POSITIONAL=()

usage() {
  echo "Usage: $0 [tag] [--from-file PATH --to-file PATH]" >&2
  echo "  tag may also be set as RELEASE_TAG. Default is v<Version> from DESCRIPTION." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-file)
      [[ $# -ge 2 ]] || { echo "missing value for --from-file" >&2; exit 1; }
      FROM_FILE=$2
      have_from=1
      shift 2
      ;;
    --to-file)
      [[ $# -ge 2 ]] || { echo "missing value for --to-file" >&2; exit 1; }
      TO_FILE=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
if [[ ${#POSITIONAL[@]} -gt 1 ]]; then
  echo "Too many positional arguments" >&2
  usage
  exit 1
fi
if [[ ${#POSITIONAL[@]} -eq 1 ]]; then
  RELEASE_TAG="${POSITIONAL[0]}"
elif [[ -z "${RELEASE_TAG:-}" ]]; then
  VERSION=$(awk '/^Version:[[:space:]]*/ { print $2; exit }' "$ROOT/r/harmorust/DESCRIPTION")
  if [[ -z "${VERSION}" ]]; then
    echo "Could not read Version from r/harmorust/DESCRIPTION" >&2
    exit 1
  fi
  RELEASE_TAG="v${VERSION}"
fi

if [[ "$RELEASE_TAG" == *$'\n'* || "$RELEASE_TAG" == *$'\r'* ]]; then
  echo "RELEASE_TAG must be a single line" >&2
  exit 1
fi
if [[ ! "$REGISTRY_SLUG" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "REGISTRY_REPO must look like owner/repo" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to update packages.json" >&2
  exit 1
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

source_file="$tmpdir/current.json"
result_file="$tmpdir/updated.json"
fetch_status="200"

redact() {
  python3 -c 'import os, sys
token = os.environ.get("UNIVERSE_TOKEN", "")
data = sys.stdin.read()
if token:
    data = data.replace(token, "***")
sys.stdout.write(data)'
}

cat > "$tmpdir/upsert.py" <<'PY'
import json
import sys

def fail(message):
    sys.stderr.write(message + "\n")
    sys.exit(1)

def load_document(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) > 5_000_000:
        fail("packages.json is unexpectedly large; refusing to overwrite the registry")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        fail(f"packages.json is not valid UTF-8 ({exc}); refusing to overwrite the registry")
    if text.strip() == "":
        fail("packages.json is empty; refusing to overwrite the registry")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as extra:
        fail(f"packages.json is not valid JSON ({extra}); refusing to overwrite the registry")
    # Contents API JSON (when the raw media type is not honored) wraps the file.
    if isinstance(data, dict) and data.get("encoding") == "base64" and "content" in data:
        import base64
        try:
            decoded = base64.b64decode(data["content"], validate=False)
            data = json.loads(decoded.decode("utf-8-sig"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as extra:
            fail(
                "packages.json payload from the GitHub API is not a JSON array "
                f"({extra}); refusing to overwrite the registry"
            )
    if not isinstance(data, list):
        fail(
            "packages.json must be a JSON array, got "
            f"{type(data).__name__}; refusing to overwrite the registry"
        )
    return data

def upsert(data, package, url, subdir, branch):
    entry = {
        "package": package,
        "url": url,
        "subdir": subdir,
        "branch": branch,
    }
    updated = []
    seen = False
    changed = False
    for item in data:
        if isinstance(item, dict) and item.get("package") == package:
            if seen:
                changed = True
                continue
            seen = True
            if item != entry:
                changed = True
            updated.append(entry)
        else:
            # Keep every other entry (including seurust) byte-for-byte.
            updated.append(item)
    if not seen:
        updated.append(entry)
        changed = True

    # Hard guard: never drop or rewrite non-harmorust packages in the shared
    # NebilI.r-universe.dev registry (seurust and any future neighbors).
    before_others = [
        item for item in data
        if not (isinstance(item, dict) and item.get("package") == package)
    ]
    after_others = [
        item for item in updated
        if not (isinstance(item, dict) and item.get("package") == package)
    ]
    if before_others != after_others:
        fail(
            "refusing to modify non-"
            f"{package} entries in packages.json "
            f"(before={before_others!r} after={after_others!r})"
        )
    before_names = {
        item.get("package")
        for item in before_others
        if isinstance(item, dict) and isinstance(item.get("package"), str)
    }
    after_names = {
        item.get("package")
        for item in after_others
        if isinstance(item, dict) and isinstance(item.get("package"), str)
    }
    if before_names != after_names:
        fail(
            "refusing to drop shared r-universe packages: "
            f"lost={sorted(before_names - after_names)} "
            f"gained={sorted(after_names - before_names)}"
        )
    if "seurust" in before_names and "seurust" not in after_names:
        fail("refusing to overwrite or drop seurust in packages.json")

    names = []
    for item in updated:
        if isinstance(item, dict) and isinstance(item.get("package"), str):
            names.append(item["package"])
        else:
            names.append("<non-package>")
    return updated, changed, names

def main():
    src, dest, package, url, subdir, branch = sys.argv[1:7]
    data = load_document(src)
    updated, changed, names = upsert(data, package, url, subdir, branch)
    text = json.dumps(updated, indent=2, ensure_ascii=False) + "\n"
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write(text)
    json.dump({"changed": 1 if changed else 0, "packages": names}, sys.stdout)
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
PY

apply_upsert() {
  python3 "$tmpdir/upsert.py" "$1" "$2" \
    "$PACKAGE" "$PACKAGE_URL" "$PACKAGE_SUBDIR" "$RELEASE_TAG"
}

report() {
  local summary="$1"
  changed=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["changed"])' "$summary")
  packages=$(python3 -c 'import json, sys; print(" ".join(json.loads(sys.argv[1])["packages"]))' "$summary")
  echo "r-universe packages: ${packages}"
  if [[ "$changed" == "0" ]]; then
    echo "harmorust entry already matches ${RELEASE_TAG}; not committing"
  else
    echo "harmorust entry will point at ${RELEASE_TAG}"
  fi
}

write_result() {
  local dest_dir tmp_out
  [[ -n "$TO_FILE" ]] || return 0
  dest_dir=$(dirname -- "$TO_FILE")
  mkdir -p "$dest_dir"
  tmp_out=$(mktemp "$dest_dir/.packages.json.XXXXXX")
  cp "$result_file" "$tmp_out"
  mv "$tmp_out" "$TO_FILE"
}

if [[ "$have_from" -eq 1 ]]; then
  if [[ -z "$TO_FILE" ]]; then
    echo "--to-file is required with --from-file" >&2
    exit 1
  fi
  if [[ ! -e "$FROM_FILE" ]]; then
    printf '%s\n' '[]' > "$source_file"
    fetch_status="404"
  elif [[ ! -f "$FROM_FILE" ]]; then
    echo "Not a file: $FROM_FILE" >&2
    exit 1
  else
    cp -- "$FROM_FILE" "$source_file"
    fetch_status="200"
  fi
else
  if [[ -z "${UNIVERSE_TOKEN:-}" ]]; then
    echo "::warning::R_UNIVERSE_REGISTRY_TOKEN not configured; skipping r-universe sync"
    exit 0
  fi
  if [[ -n "${PACKAGES_JSON_URL:-}" ]]; then
    packages_url="$PACKAGES_JSON_URL"
    curl_retry=()
  else
    packages_url="https://api.github.com/repos/${REGISTRY_SLUG}/contents/packages.json"
    curl_retry=(--retry 2 --retry-delay 1 --retry-connrefused)
  fi
  curl_err="$tmpdir/curl.err"
  if ! http_code=$(curl --silent --show-error \
      --output "$source_file" \
      --write-out "%{http_code}" \
      "${curl_retry[@]}" \
      --max-time 60 \
      --header "Authorization: Bearer ${UNIVERSE_TOKEN}" \
      --header "Accept: application/vnd.github.raw" \
      --header "X-GitHub-Api-Version: 2022-11-28" \
      "$packages_url" 2>"$curl_err"); then
    echo "Failed to fetch packages.json; refusing to overwrite the registry" >&2
    redact < "$curl_err" >&2 || true
    exit 1
  fi
  if [[ "$http_code" == "404" ]]; then
    printf '%s\n' '[]' > "$source_file"
    fetch_status="404"
  elif [[ "$http_code" != "200" ]]; then
    echo "Failed to fetch packages.json (HTTP ${http_code}); refusing to overwrite the registry" >&2
    exit 1
  else
    fetch_status="200"
  fi
fi

if ! summary=$(apply_upsert "$source_file" "$result_file"); then
  exit 1
fi
report "$summary"

skip_push=0
if [[ "$have_from" -eq 1 || -n "$TO_FILE" ]]; then
  skip_push=1
elif [[ -n "${PACKAGES_JSON_URL:-}" && "${SYNC_R_UNIVERSE_ALLOW_PUSH:-}" != "1" ]]; then
  skip_push=1
fi

if [[ "$skip_push" -eq 1 ]]; then
  write_result
  if [[ -n "${PACKAGES_JSON_URL:-}" && -z "$TO_FILE" && "$have_from" -eq 0 ]]; then
    echo "PACKAGES_JSON_URL is set and push is disabled; not cloning the registry"
  fi
  exit 0
fi

if [[ "$changed" == "0" ]]; then
  exit 0
fi

clone_dir="$tmpdir/registry"
clone_log="$tmpdir/clone.log"
export GIT_TERMINAL_PROMPT=0
if ! git clone --quiet \
    "https://x-access-token:${UNIVERSE_TOKEN}@github.com/${REGISTRY_SLUG}.git" \
    "$clone_dir" >"$clone_log" 2>&1; then
  echo "Failed to clone ${REGISTRY_SLUG}; refusing to overwrite the registry" >&2
  redact < "$clone_log" >&2 || true
  exit 1
fi

clone_packages="$clone_dir/packages.json"
if [[ -f "$clone_packages" ]]; then
  cp "$clone_packages" "$source_file"
elif [[ "$fetch_status" == "404" ]]; then
  printf '%s\n' '[]' > "$source_file"
else
  echo "Fetched packages.json but the clone has no packages.json; refusing to overwrite the registry" >&2
  exit 1
fi

if ! summary=$(apply_upsert "$source_file" "$result_file"); then
  exit 1
fi
report "$summary"
if [[ "$changed" == "0" ]]; then
  exit 0
fi

cp "$result_file" "$clone_packages"
git -C "$clone_dir" config user.email "github-actions[bot]@users.noreply.github.com"
git -C "$clone_dir" config user.name "github-actions[bot]"
git -C "$clone_dir" add packages.json
if git -C "$clone_dir" diff --cached --quiet; then
  echo "No changes to packages.json; not committing"
  exit 0
fi
git -C "$clone_dir" -c commit.gpgsign=false commit \
  -m "Point harmorust at ${RELEASE_TAG}" \
  -m "Upsert only the harmorust entry in the shared r-universe packages.json."
push_log="$tmpdir/push.log"
if ! git -C "$clone_dir" push origin HEAD >"$push_log" 2>&1; then
  echo "Failed to push packages.json to ${REGISTRY_SLUG}" >&2
  redact < "$push_log" >&2 || true
  exit 1
fi
redact < "$push_log"
echo "Pushed harmorust entry at ${RELEASE_TAG} to ${REGISTRY_SLUG}"
