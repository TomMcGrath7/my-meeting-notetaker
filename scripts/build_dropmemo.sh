#!/usr/bin/env bash
#
# build_dropmemo.sh — compile DropMemo.app WITHOUT Xcode (just the Swift
# toolchain + swiftc). Produces build/DropMemo.app, ad-hoc signs it, and
# registers it with LaunchServices so "Open With ▸ DropMemo" works for
# .m4a/.wav/.mp3.
#
# Usage:  scripts/build_dropmemo.sh
# Then:   open build/DropMemo.app     (or drag it to /Applications)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO/DropMemo/DropMemoApp.swift"
APP="$REPO/build/DropMemo.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"
BUNDLE_ID="local.dropmemo"

command -v swiftc >/dev/null || { echo "error: swiftc not found (install the Swift toolchain or Xcode CLT)" >&2; exit 1; }

echo "▶ building DropMemo.app"
rm -rf "$APP"
mkdir -p "$MACOS" "$RES"

# --- compile ----------------------------------------------------------------
# -swift-version 5 keeps the @MainActor/closure plumbing simple; macOS 13 is the
# floor for the SwiftUI `Window` scene used here.
swiftc -O -swift-version 5 \
  -parse-as-library \
  -target "arm64-apple-macos13.0" \
  -o "$MACOS/DropMemo" \
  "$SRC"

# --- Info.plist (document types + baked repo path) --------------------------
# DMRepoPath tells the app where this checkout lives so it can find
# scripts/process.sh; override at runtime with the DROPMEMO_REPO env var.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>DropMemo</string>
  <key>CFBundleDisplayName</key>     <string>DropMemo</string>
  <key>CFBundleIdentifier</key>      <string>${BUNDLE_ID}</string>
  <key>CFBundleExecutable</key>      <string>DropMemo</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>CFBundleShortVersionString</key> <string>1.0</string>
  <key>CFBundleVersion</key>         <string>1</string>
  <key>LSMinimumSystemVersion</key>  <string>13.0</string>
  <key>NSPrincipalClass</key>        <string>NSApplication</string>
  <key>NSHighResolutionCapable</key> <true/>
  <key>LSApplicationCategoryType</key> <string>public.app-category.productivity</string>
  <key>DMRepoPath</key>              <string>${REPO}</string>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key>    <string>Audio recording</string>
      <key>CFBundleTypeRole</key>    <string>Viewer</string>
      <key>LSHandlerRank</key>       <string>Alternate</string>
      <key>LSItemContentTypes</key>
      <array>
        <string>public.mpeg-4-audio</string>
        <string>com.microsoft.waveform-audio</string>
        <string>public.mp3</string>
        <string>public.audio</string>
      </array>
    </dict>
  </array>
</dict>
</plist>
PLIST

# --- ad-hoc sign + register --------------------------------------------------
# Ad-hoc signature lets a locally built app launch without "damaged" warnings.
codesign --force --deep --sign - "$APP" 2>/dev/null || echo "  (codesign skipped)"

# Make LaunchServices aware of the document-type handlers immediately.
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[[ -x "$LSREGISTER" ]] && "$LSREGISTER" -f "$APP" || true

echo "✓ built $APP"
echo "  open it:        open \"$APP\""
echo "  install it:     cp -R \"$APP\" /Applications/"
