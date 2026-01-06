#!/bin/bash
# AIVA CLI 工具統一啟動腳本
# 自動設定 PYTHONPATH 並執行 CLI 工具

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 設定 Python 路徑
export PYTHONPATH="${PROJECT_ROOT}/services/core:${PROJECT_ROOT}/services:${PYTHONPATH}"

# 切換到正確目錄
cd "${PROJECT_ROOT}/services/core"

# 執行 CLI 工具
python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation "$@"
