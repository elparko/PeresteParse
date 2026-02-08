#!/bin/bash
#
# create_dmg.sh — Create Pereste Parse.dmg for distribution
#
# Creates a DMG disk image with drag-to-Applications support.
# Requires Pereste Parse.app to be built first (run build_app.sh).
#
# Usage: bash scripts/create_dmg.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_DIR/dist"
APP_DIR="$DIST_DIR/Pereste Parse.app"
DMG_NAME="Pereste Parse.dmg"
DMG_PATH="$DIST_DIR/$DMG_NAME"
VOLUME_NAME="Pereste Parse"
STAGING_DIR="$DIST_DIR/dmg-staging"

echo "=== Pereste Parse DMG Creator ==="

# ── Verify .app exists ──────────────────────────────────────────────────────
if [ ! -d "$APP_DIR" ]; then
    echo "ERROR: $APP_DIR not found."
    echo "Run 'bash scripts/build_app.sh' first."
    exit 1
fi

# ── Clean previous DMG ──────────────────────────────────────────────────────
if [ -f "$DMG_PATH" ]; then
    echo "Removing previous DMG..."
    rm -f "$DMG_PATH"
fi

# ── Create staging directory ────────────────────────────────────────────────
echo "Preparing DMG contents..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Copy .app to staging
cp -r "$APP_DIR" "$STAGING_DIR/"

# Add Applications symlink for drag-to-install
ln -s /Applications "$STAGING_DIR/Applications"

# ── Create DMG ──────────────────────────────────────────────────────────────
echo "Creating DMG (this may take a moment)..."
hdiutil create \
    -volname "$VOLUME_NAME" \
    -srcfolder "$STAGING_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH" \
    > /dev/null 2>&1

# ── Cleanup ─────────────────────────────────────────────────────────────────
rm -rf "$STAGING_DIR"

# ── Report ──────────────────────────────────────────────────────────────────
DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)
echo ""
echo "=== DMG Created ==="
echo "Output: $DMG_PATH"
echo "Size:   $DMG_SIZE"
echo ""
echo "To test: open $DMG_PATH"
