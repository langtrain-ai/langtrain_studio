#!/usr/bin/env bash
# Builds the Langtrain engine (engine/) and installs it as the app's sidecar:
#   app/src-tauri/binaries/langtrain-<target-triple>[.exe]
# which is where Tauri's externalBin expects it. Run it before `pnpm tauri dev`
# or `pnpm tauri build`; the release workflow runs it for every platform.
#
#   scripts/build-engine.sh                 # host platform, release, llama.cpp included
#   scripts/build-engine.sh --target aarch64-apple-darwin
#   scripts/build-engine.sh --fast          # native engine only, no llama.cpp (CI checks)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=""
FAST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --fast) FAST=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TARGET" ] || TARGET="$(rustc -vV | sed -n 's/^host: //p')"

if [ "$FAST" = 1 ]; then
  features=()
else
  # Metal on Apple silicon; llama.cpp everywhere for models the native engine
  # does not cover yet. Portable builds: no target-cpu=native.
  case "$TARGET" in
    aarch64-apple-darwin) features=(--features "llama,metal") ;;
    *) features=(--features llama) ;;
  esac
fi

ext=""
[[ "$TARGET" == *windows* ]] && ext=".exe"

echo "[engine] building langtrain for $TARGET ${features[*]:-(native only)}"
cd "$ROOT/engine"
cargo build --release -p langtrain --no-default-features ${features[@]+"${features[@]}"} --target "$TARGET"

dest="$ROOT/app/src-tauri/binaries"
mkdir -p "$dest"
cp "target/$TARGET/release/langtrain$ext" "$dest/langtrain-$TARGET$ext"
echo "[engine] installed $dest/langtrain-$TARGET$ext"
