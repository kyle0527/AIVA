#!/bin/bash
# AIVA Capability CLI 工具啟動腳本
# 用於查詢和執行能力

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 設定 Python 路徑
export PYTHONPATH="${PROJECT_ROOT}/services/core:${PROJECT_ROOT}/services:${PYTHONPATH}"

# 切換到正確目錄
cd "${PROJECT_ROOT}/services/core"

# 執行 Capability CLI
python -m aiva_core.internal_exploration.python_tools.aiva_capability_cli "$@"
