#!/bin/bash
#
# build_app.sh — Build Pereste Parse.app macOS bundle
#
# Creates a self-contained .app directory with embedded Python venv
# and all dependencies pre-installed.
#
# Usage: bash scripts/build_app.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_DIR/dist"
APP_DIR="$DIST_DIR/Pereste Parse.app"
CONTENTS="$APP_DIR/Contents"
RESOURCES="$CONTENTS/Resources"
MACOS="$CONTENTS/MacOS"
FRAMEWORKS="$CONTENTS/Frameworks"

echo "=== Pereste Parse.app Builder ==="
echo "Project: $PROJECT_DIR"
echo ""

# ── Step 1: Clean previous build ────────────────────────────────────────────
if [ -d "$APP_DIR" ]; then
    echo "Removing previous build..."
    rm -rf "$APP_DIR"
fi

# ── Step 2: Create .app directory structure ─────────────────────────────────
echo "Creating .app directory structure..."
mkdir -p "$MACOS"
mkdir -p "$RESOURCES"
mkdir -p "$FRAMEWORKS"

# ── Step 3: Copy app code ───────────────────────────────────────────────────
echo "Copying application code..."
cp "$PROJECT_DIR/app.py" "$RESOURCES/"
cp "$PROJECT_DIR/server.py" "$RESOURCES/"
cp -r "$PROJECT_DIR/static" "$RESOURCES/"

# ── Step 4: Copy Info.plist ─────────────────────────────────────────────────
echo "Copying Info.plist..."
cp "$PROJECT_DIR/Info.plist" "$CONTENTS/"

# ── Step 5: Generate app icon ───────────────────────────────────────────────
ICON_SOURCE="$PROJECT_DIR/static/plogo-icon.png"
if [ -f "$ICON_SOURCE" ]; then
    echo "Generating app icon..."
    ICONSET_DIR="$DIST_DIR/app.iconset"
    mkdir -p "$ICONSET_DIR"

    # Generate required icon sizes using sips
    sips -z 16 16     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16.png"      > /dev/null 2>&1
    sips -z 32 32     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16@2x.png"   > /dev/null 2>&1
    sips -z 32 32     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32.png"      > /dev/null 2>&1
    sips -z 64 64     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32@2x.png"   > /dev/null 2>&1
    sips -z 128 128   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128.png"    > /dev/null 2>&1
    sips -z 256 256   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128@2x.png" > /dev/null 2>&1
    sips -z 256 256   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256.png"    > /dev/null 2>&1
    sips -z 512 512   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256@2x.png" > /dev/null 2>&1
    sips -z 512 512   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512.png"    > /dev/null 2>&1
    cp "$ICON_SOURCE" "$ICONSET_DIR/icon_512x512@2x.png"

    # Convert iconset to icns
    iconutil -c icns "$ICONSET_DIR" -o "$RESOURCES/app.icns"
    rm -rf "$ICONSET_DIR"
    echo "  Icon generated: app.icns"
else
    echo "  WARNING: $ICON_SOURCE not found, skipping icon generation"
fi

# ── Step 6: Create embedded venv ────────────────────────────────────────────
echo "Creating embedded Python venv..."
PYTHON="$(command -v python3)"
echo "  Using Python: $PYTHON ($($PYTHON --version))"

$PYTHON -m venv "$RESOURCES/venv"
echo "  Venv created at: $RESOURCES/venv"

# ── Step 7: Install dependencies ────────────────────────────────────────────
echo "Installing dependencies (this may take a few minutes)..."
VENV_PIP="$RESOURCES/venv/bin/pip"

# Upgrade pip first
"$VENV_PIP" install --upgrade pip > /dev/null 2>&1

# Install llama-cpp-python with Metal support
echo "  Installing llama-cpp-python with Metal..."
CMAKE_ARGS="-DGGML_METAL=on" "$VENV_PIP" install 'llama-cpp-python>=0.3.0' 2>&1 | tail -1

# Install remaining dependencies
echo "  Installing remaining packages..."
"$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt" 2>&1 | tail -5

echo "  Dependencies installed."

# ── Step 8: Write launcher script ───────────────────────────────────────────
echo "Writing launcher script..."
cat > "$MACOS/PeresteParse" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"

# Include common system paths so tools like ffmpeg are found
export PATH="$DIR/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

cd "$DIR"
exec "$DIR/venv/bin/python" app.py
LAUNCHER
chmod +x "$MACOS/PeresteParse"

# ── Step 9: Report ──────────────────────────────────────────────────────────
echo ""
echo "=== Build Complete ==="
APP_SIZE=$(du -sh "$APP_DIR" | cut -f1)
echo "Output: $APP_DIR"
echo "Size:   $APP_SIZE"
echo ""
echo "To launch:"
echo "  open $APP_DIR"
echo ""
echo "To install:"
echo "  cp -r $APP_DIR /Applications/"
echo ""
echo "Note: On first launch, right-click → Open (or run: xattr -cr $APP_DIR)"
