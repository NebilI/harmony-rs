#!/usr/bin/env bash
# Create a private GitHub repo and push this tree.
# Requires: gh auth login   OR   GH_TOKEN with repo scope
set -euo pipefail

OWNER="${GITHUB_OWNER:-NebilI}"
NAME="${GITHUB_REPO:-harmorust}"
VISIBILITY="${GITHUB_VISIBILITY:-private}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: https://cli.github.com" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  if [[ -n "${GH_TOKEN:-}" ]]; then
    echo "$GH_TOKEN" | gh auth login --with-token
  else
    echo "Not logged in. Run: gh auth login" >&2
    echo "Or export GH_TOKEN and re-run this script." >&2
    exit 1
  fi
fi

if gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  echo "Repo $OWNER/$NAME already exists"
else
  gh repo create "$OWNER/$NAME" --"$VISIBILITY" --source=. --remote=github --push
  echo "Created $VISIBILITY repo https://github.com/$OWNER/$NAME"
  exit 0
fi

if git remote get-url github >/dev/null 2>&1; then
  git remote set-url github "https://github.com/$OWNER/$NAME.git"
else
  git remote add github "https://github.com/$OWNER/$NAME.git"
fi
git push -u github HEAD:main
echo "Pushed to https://github.com/$OWNER/$NAME"
