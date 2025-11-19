#!/bin/bash
# Go 掃描器構建腳本 (Linux/macOS)

set -e

echo "========================================"
echo "Building Go Scanners for AIVA"
echo "========================================"
echo ""

# 檢查 Go 是否安裝
if ! command -v go &> /dev/null; then
    echo "✗ Go is not installed. Please install Go 1.21 or later."
    exit 1
fi

GO_VERSION=$(go version)
echo "✓ Go is installed: $GO_VERSION"

# 獲取腳本目錄
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 掃描器列表
declare -a SCANNERS=(
    "ssrf_scanner:SSRF Scanner:worker"
    "cspm_scanner:CSPM Scanner:worker"
    "sca_scanner:SCA Scanner:worker"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for scanner_info in "${SCANNERS[@]}"; do
    IFS=':' read -r SCANNER_DIR SCANNER_NAME OUTPUT_NAME <<< "$scanner_info"
    
    echo ""
    echo "Building $SCANNER_NAME..."
    echo "  Directory: $SCANNER_DIR"
    
    if [ ! -d "$SCANNER_DIR" ]; then
        echo "  ✗ Directory not found: $SCANNER_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    cd "$SCANNER_DIR"
    
    if [ ! -f "main.go" ]; then
        echo "  ✗ main.go not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # 下載依賴
    echo "  → Downloading dependencies..."
    go mod download 2>&1 > /dev/null || true
    
    # 構建
    echo "  → Compiling..."
    if go build -o "$OUTPUT_NAME" -ldflags="-s -w" . 2>&1; then
        if [ -f "$OUTPUT_NAME" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_NAME" | cut -f1)
            echo "  ✓ Built successfully: $OUTPUT_NAME ($FILE_SIZE)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ Build failed: output file not found"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "  ✗ Build failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    cd "$SCRIPT_DIR"
done

echo ""
echo "========================================"
echo "Build Summary"
echo "========================================"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Some builds failed. Please check the output above for details."
    exit 1
else
    echo "All scanners built successfully! ✓"
    exit 0
fi
