#!/bin/sh
# Install the latest Langtrain CLI on macOS or Linux:
#   curl -fsSL https://raw.githubusercontent.com/langtrain-ai/langtrain_studio/main/engine/scripts/install.sh | sh
set -eu

REPO="${LANGTRAIN_REPO:-langtrain-ai/langtrain_studio}"
DEST="${LANGTRAIN_INSTALL_DIR:-$HOME/.langtrain}"

os=$(uname -s)
arch=$(uname -m)
case "$os-$arch" in
  Darwin-arm64)  target=aarch64-apple-darwin ;;
  Darwin-x86_64) target=x86_64-apple-darwin ;;
  Linux-x86_64)  target=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64) target=aarch64-unknown-linux-gnu ;;
  *) echo "Unsupported platform: $os $arch" >&2; exit 1 ;;
esac

tag=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$tag" ] || { echo "Could not find the latest release of $REPO" >&2; exit 1; }
name="langtrain-$tag-$target"
url="https://github.com/$REPO/releases/download/$tag/$name.tar.gz"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "Downloading $name"
curl -fsSL "$url" -o "$tmp/$name.tar.gz"
curl -fsSL "$url.sha256" -o "$tmp/$name.tar.gz.sha256"
(cd "$tmp" && shasum -a 256 -c "$name.tar.gz.sha256" >/dev/null) || { echo "Checksum mismatch" >&2; exit 1; }
tar xzf "$tmp/$name.tar.gz" -C "$tmp"

mkdir -p "$DEST/bin"
cp "$tmp/$name/langtrain" "$DEST/bin/langtrain"
rm -rf "$DEST/bin/langtune" && cp -R "$tmp/$name/langtune" "$DEST/bin/langtune"
chmod +x "$DEST/bin/langtrain"

echo "Installed langtrain $tag to $DEST/bin"
case ":$PATH:" in
  *":$DEST/bin:"*) ;;
  *) echo "Add it to your PATH:  export PATH=\"$DEST/bin:\$PATH\"" ;;
esac
echo "Then run:  langtrain serve   (Studio opens at http://127.0.0.1:11435)"
