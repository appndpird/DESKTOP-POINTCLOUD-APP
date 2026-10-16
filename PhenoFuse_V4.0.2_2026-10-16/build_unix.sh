#!/usr/bin/env bash
# Build PhenoApp on Linux or macOS
set -e

echo "=== PhenoApp build (Linux/macOS) ==="

if ! command -v pyinstaller &>/dev/null; then
    echo "ERROR: pyinstaller not installed. Run: pip install pyinstaller"
    exit 1
fi

rm -rf build dist
pyinstaller phenoapp.spec --noconfirm

echo
echo "=== Build complete ==="
echo "Run: ./dist/PhenoApp/PhenoApp"
