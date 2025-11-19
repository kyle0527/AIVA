# AIVA Go 掃描器 Makefile
# 重構後的統一構建流程

# 常量定義
BUILD_DIR := bin
GO_FLAGS := -ldflags="-s -w" -trimpath

# 自動檢測平台
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    EXT := 
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM := darwin
    EXT := 
endif
ifeq ($(UNAME_S),Windows)
    PLATFORM := windows
    EXT := .exe
endif
ifndef PLATFORM
    PLATFORM := windows
    EXT := .exe
endif

# 預設目標
.PHONY: all
all: build

# 構建所有掃描器
.PHONY: build
build:
	@echo "🚀 Building all Go scanners for platform: $(PLATFORM)"
	@mkdir -p $(BUILD_DIR)
	@cd cmd/ssrf-scanner && go build $(GO_FLAGS) -o ../../$(BUILD_DIR)/ssrf-scanner$(EXT) .
	@cd cmd/cspm-scanner && go build $(GO_FLAGS) -o ../../$(BUILD_DIR)/cspm-scanner$(EXT) .
	@cd cmd/sca-scanner && go build $(GO_FLAGS) -o ../../$(BUILD_DIR)/sca-scanner$(EXT) .
	@echo "✅ All scanners built successfully"

# 構建單一掃描器
.PHONY: ssrf cspm sca
ssrf:
	@echo "🔍 Building SSRF scanner..."
	@mkdir -p $(BUILD_DIR)
	@cd cmd/ssrf-scanner && go build $(GO_FLAGS) -o ../../$(BUILD_DIR)/ssrf-scanner$(EXT) .

cspm:
	@echo "☁️ Building CSPM scanner..."
	@mkdir -p $(BUILD_DIR)
	@cd cmd/cspm-scanner && go build $(GO_FLAGS) -o ../../$(BUILD_DIR)/cspm-scanner$(EXT) .

sca:
	@echo "📦 Building SCA scanner..."
	@mkdir -p $(BUILD_DIR)
	@cd cmd/sca-scanner && go build $(GO_FLAGS) -o ../../$(BUILD_DIR)/sca-scanner$(EXT) .

# 快速重建
.PHONY: rebuild
rebuild: clean build

# 清理構建檔案
.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/*$(EXT) 2>/dev/null || true
	@echo "✅ Clean completed"

# 檢查前置條件
.PHONY: check
check:
	@echo "🔍 Checking prerequisites..."
	@command -v go >/dev/null 2>&1 || { echo "❌ Go is not installed"; exit 1; }
	@echo "✅ Go version: $$(go version)"

# 開發環境設置
.PHONY: setup
setup: check
	@echo "⚡ Setting up development environment..."
	@go work sync
	@echo "✅ Development environment ready"

# 顯示構建狀態
.PHONY: status
status:
	@echo "📊 Build Status Report"
	@echo "======================"
	@echo "Platform: $(PLATFORM)"
	@echo "Extension: $(EXT)"
	@echo ""
	@ls -lh $(BUILD_DIR)/*$(EXT) 2>/dev/null || echo "No binaries found"

# 幫助資訊
.PHONY: help
help:
	@echo "AIVA Go Scanner Build System (Refactored)"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build        - Build all scanners (default)"
	@echo "  ssrf         - Build SSRF scanner only"
	@echo "  cspm         - Build CSPM scanner only"
	@echo "  sca          - Build SCA scanner only"
	@echo "  rebuild      - Clean and rebuild all"
	@echo "  clean        - Remove all build artifacts"
	@echo "  check        - Check prerequisites"
	@echo "  setup        - Set up development environment"
	@echo "  status       - Show build status"
	@echo ""
	@echo "Usage examples:"
	@echo "  make build    # Build all scanners"
	@echo "  make ssrf     # Build only SSRF scanner"
	@echo "  make rebuild  # Clean and rebuild"

# 跨平台構建
.PHONY: build-windows build-linux build-darwin build-arm64
build-windows:
	@echo "🪟 Building for Windows..."
	@$(PYTHON_BUILD) --scanner all --platform windows

build-linux:
	@echo "🐧 Building for Linux..."
	@$(PYTHON_BUILD) --scanner all --platform linux

build-darwin:
	@echo "🍎 Building for macOS..."
	@$(PYTHON_BUILD) --scanner all --platform darwin

build-arm64:
	@echo "💪 Building for ARM64..."
	@$(PYTHON_BUILD) --scanner all --platform arm64

# 清理構建檔案
.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@$(PYTHON_BUILD) --clean --check
	@for dir in $(SCANNER_DIRS); do \
		if [ -d $$dir ]; then \
			echo "Cleaning $$dir..."; \
			find $$dir -name "worker*" -type f -delete 2>/dev/null || true; \
		fi; \
	done
	@if [ -d $(BUILD_DIR) ]; then rm -rf $(BUILD_DIR); fi
	@echo "✅ Clean completed"

# 檢查前置條件
.PHONY: check
check:
	@echo "🔍 Checking prerequisites..."
	@$(PYTHON_BUILD) --check

# 開發環境設置
.PHONY: setup
setup: check
	@echo "⚡ Setting up development environment..."
	@if ! command -v go >/dev/null 2>&1; then \
		echo "❌ Go is not installed. Please install Go 1.21+"; \
		exit 1; \
	fi
	@echo "📦 Downloading Go dependencies..."
	@go mod download
	@go mod tidy
	@echo "✅ Development environment ready"

# 測試所有掃描器
.PHONY: test
test: build
	@echo "🧪 Testing all scanners..."
	@for dir in $(SCANNER_DIRS); do \
		if [ -f $$dir/worker$(EXT) ]; then \
			echo "Testing $$dir..."; \
			./$$dir/worker$(EXT) --version || echo "⚠️  $$dir test failed"; \
		else \
			echo "❌ $$dir/worker$(EXT) not found"; \
		fi; \
	done

# 安裝掃描器到系統
.PHONY: install
install: build
	@echo "📦 Installing scanners..."
	@mkdir -p /usr/local/bin/aiva-scanners 2>/dev/null || mkdir -p ~/bin/aiva-scanners
	@for dir in $(SCANNER_DIRS); do \
		if [ -f $$dir/worker$(EXT) ]; then \
			cp $$dir/worker$(EXT) /usr/local/bin/aiva-scanners/$$dir$(EXT) 2>/dev/null || \
			cp $$dir/worker$(EXT) ~/bin/aiva-scanners/$$dir$(EXT) || \
			echo "⚠️  Could not install $$dir"; \
		fi; \
	done
	@echo "✅ Installation completed"

# 顯示構建狀態
.PHONY: status
status:
	@echo "📊 Build Status Report"
	@echo "======================"
	@echo "Platform: $(PLATFORM)"
	@echo "Extension: $(EXT)"
	@echo ""
	@for dir in $(SCANNER_DIRS); do \
		printf "%-15s: " $$dir; \
		if [ -f $$dir/worker$(EXT) ]; then \
			size=$$(du -h $$dir/worker$(EXT) 2>/dev/null | cut -f1 || echo "?"); \
			echo "✅ Built ($$size)"; \
		else \
			echo "❌ Missing"; \
		fi; \
	done
	@echo ""
	@if [ -d $(BUILD_DIR) ]; then \
		echo "Build directory: $(BUILD_DIR)/ (exists)"; \
	else \
		echo "Build directory: $(BUILD_DIR)/ (missing)"; \
	fi

# 快速重建
.PHONY: rebuild
rebuild: clean build

# 幫助資訊
.PHONY: help
help:
	@echo "AIVA Go Scanner Build System"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  build        - Build all scanners (default)"
	@echo "  ssrf         - Build SSRF scanner only"
	@echo "  cspm         - Build CSPM scanner only"
	@echo "  sca          - Build SCA scanner only"
	@echo ""
	@echo "Cross-platform builds:"
	@echo "  build-windows - Build for Windows"
	@echo "  build-linux   - Build for Linux"
	@echo "  build-darwin  - Build for macOS"
	@echo "  build-arm64   - Build for ARM64"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        - Remove all build artifacts"
	@echo "  check        - Check prerequisites"
	@echo "  setup        - Set up development environment"
	@echo "  test         - Test all built scanners"
	@echo "  install      - Install scanners to system"
	@echo "  status       - Show build status"
	@echo "  rebuild      - Clean and build"
	@echo ""
	@echo "Usage examples:"
	@echo "  make build           # Build all scanners"
	@echo "  make ssrf            # Build only SSRF scanner"
	@echo "  make build-windows   # Cross-compile for Windows"
	@echo "  make clean build     # Clean rebuild"