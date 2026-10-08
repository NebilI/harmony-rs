#!/usr/bin/env bash
# Create a private GitHub repo and make the Origin copy a GitHub mirror.
#
# GitHub stays the source of truth. Origin browse/search stay at
# https://cursor.com/codebase/nebil-ibrahim/harmony-rs
#
# Requires:
#   - gh auth login   (repo scope)
#   - origin auth login
#   - Cursor GitHub app installed on the GitHub account (Settings → GitHub)
#
# Usage:
#   ./scripts/mirror-github.sh
#   GITHUB_VISIBILITY=public ./scripts/mirror-github.sh
#   RECREATE_ORIGIN_MIRROR=1 ./scripts/mirror-github.sh
set -euo pipefail

GH_OWNER="${GITHUB_OWNER:-NebilI}"
GH_NAME="${GITHUB_REPO:-harmony-rs}"
GH_FULL="$GH_OWNER/$GH_NAME"
VISIBILITY="${GITHUB_VISIBILITY:-private}"
ORIGIN_NS="${ORIGIN_NAMESPACE:-nebil-ibrahim}"
ORIGIN_NAME="${ORIGIN_REPO:-harmony-rs}"
ORIGIN_FULL="$ORIGIN_NS/$ORIGIN_NAME"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: https://cli.github.com" >&2
  exit 1
fi
if ! command -v origin >/dev/null 2>&1; then
  echo "Install Origin CLI: https://cursor.com/docs/origin/cli" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  if [[ -n "${GH_TOKEN:-}" ]]; then
    echo "$GH_TOKEN" | gh auth login --with-token --hostname github.com --insecure-storage
  else
    echo "Not logged in to GitHub. Run: gh auth login" >&2
    exit 1
  fi
fi
if ! origin auth status >/dev/null 2>&1; then
  echo "Not logged in to Origin. Run: origin auth login" >&2
  exit 1
fi

if [[ "$VISIBILITY" != "private" && "$VISIBILITY" != "public" && "$VISIBILITY" != "internal" ]]; then
  echo "GITHUB_VISIBILITY must be private, public, or internal" >&2
  exit 1
fi

if gh repo view "$GH_FULL" >/dev/null 2>&1; then
  echo "GitHub repo $GH_FULL already exists"
else
  echo "Creating $VISIBILITY GitHub repo $GH_FULL"
  gh repo create "$GH_FULL" --"$VISIBILITY" --description "Rust port of Harmony single-cell integration"
fi

if git remote get-url github >/dev/null 2>&1; then
  git remote set-url github "https://github.com/$GH_FULL.git"
else
  git remote add github "https://github.com/$GH_FULL.git"
fi

echo "Pushing main to GitHub"
git push -u github HEAD:main

echo "GitHub source of truth: https://github.com/$GH_FULL"

if origin repo view "$ORIGIN_FULL" >/dev/null 2>&1; then
  echo "Origin repo $ORIGIN_FULL already exists (native copy)."
  if [[ "${RECREATE_ORIGIN_MIRROR:-0}" == "1" ]]; then
    echo "Deleting native Origin repo so it can be recreated as a GitHub mirror"
    origin repo delete "$ORIGIN_FULL" -y
  else
    cat <<EOF
To turn that Origin repo into a GitHub mirror, either:

  1. Open https://cursor.com/codebase → Sync from GitHub → $GH_FULL
     (if the name $ORIGIN_FULL is taken, detach/delete the native copy first), or

  2. Recreate the Origin repo as a mirror (GitHub must already have main):

       RECREATE_ORIGIN_MIRROR=1 ./scripts/mirror-github.sh
EOF
    exit 0
  fi
fi

echo "Creating Origin mirror of $GH_FULL under $ORIGIN_NS"
origin repo create-mirrored "$GH_FULL" --namespace "$ORIGIN_NS"
echo "Origin is now a mirror. Browse: https://cursor.com/codebase/$ORIGIN_FULL"
echo "Pushes to the Origin remote go to GitHub; Origin updates after GitHub accepts them."
