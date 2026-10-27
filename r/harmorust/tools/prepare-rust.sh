#!/bin/sh
# Copy crates/harmorust into r/harmorust/tools/rust as a standalone crate so
# R CMD build / R-universe can compile the CLI without the Cargo workspace.
set -eu

PKG_ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
DEST="$PKG_ROOT/tools/rust"
VERSION=$(awk '/^Version:[[:space:]]*/ { print $2; exit }' "$PKG_ROOT/DESCRIPTION")

if [ -z "${VERSION}" ]; then
  echo "prepare-rust: could not read Version from DESCRIPTION" >&2
  exit 1
fi

WORKSPACE=$(CDPATH= cd "$PKG_ROOT/../.." && pwd)
CRATE="$WORKSPACE/crates/harmorust"

if [ ! -d "$CRATE/src" ]; then
  if [ -f "$DEST/Cargo.toml" ] && [ -d "$DEST/src" ]; then
    echo "prepare-rust: using existing $DEST (workspace crate not found)"
    exit 0
  fi
  echo "prepare-rust: missing $CRATE/src and no packaged rust/ tree" >&2
  exit 1
fi

echo "prepare-rust: copying $CRATE -> $DEST (version $VERSION)"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$CRATE/src" "$DEST/src"
if [ -d "$CRATE/scripts" ]; then
  cp -R "$CRATE/scripts" "$DEST/scripts"
fi
if [ -f "$CRATE/README.md" ]; then
  cp "$CRATE/README.md" "$DEST/README.md"
fi

# In-line workspace inheritance so cargo can build this tree alone.
# An empty [workspace] keeps this copy from joining the monorepo workspace
# when configure runs inside a git checkout.
{
  sed \
    -e "s/^version\\.workspace = true$/version = \"${VERSION}\"/" \
    -e 's/^edition\.workspace = true$/edition = "2021"/' \
    -e 's/^license\.workspace = true$/license = "MIT"/' \
    -e 's/^rust-version\.workspace = true$/rust-version = "1.89"/' \
    "$CRATE/Cargo.toml"
  printf '\n[workspace]\n'
} > "$DEST/Cargo.toml"

if grep -q 'workspace = true' "$DEST/Cargo.toml"; then
  echo "prepare-rust: leftover workspace keys in $DEST/Cargo.toml" >&2
  grep 'workspace = true' "$DEST/Cargo.toml" >&2
  exit 1
fi

if [ -f "$WORKSPACE/Cargo.lock" ]; then
  cp "$WORKSPACE/Cargo.lock" "$DEST/Cargo.lock"
fi

echo "/target" > "$DEST/.gitignore"
echo "prepare-rust: wrote standalone crate $DEST"
