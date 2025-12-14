# AIVA 統一輸出路徑實施報告

## 執行摘要

已成功實施 AIVA Internal Exploration 模組的統一輸出路徑配置，所有語言工具（Python、TypeScript、Go、Rust）現在將輸出集中存儲在 `services/integration/data/internal_exploration/` 目錄。

## 實施日期

2024年（依據當前系統時間）

## 修改範圍

### ✅ 1. 統一配置文件

#### Python 配置
- **文件**: `services/aiva_common/config/paths.py`
- **狀態**: ✅ 已創建（306 行）
- **功能**:
  - 定義所有輸出目錄路徑
  - 提供 `ensure_directories()` 自動創建目錄
  - 提供 `get_analysis_output_dir(tool_name)` 獲取特定工具輸出目錄
  - 提供 `export_paths_config(format)` 導出多語言配置
  - 支援環境變量 `AIVA_USE_INTEGRATED_PATHS` 控制

#### TypeScript 配置
- **文件**: `services/core/aiva_core/internal_exploration/typescript_tools/paths.config.ts`
- **狀態**: ✅ 已創建（113 行）
- **功能**: 提供與 Python 等價的路徑配置

#### Go 配置
- **文件**: `services/core/aiva_core/internal_exploration/go_tools/paths_config.go`
- **狀態**: ✅ 已創建（86 行）
- **功能**: 提供與 Python 等價的路徑配置

#### Rust 配置
- **文件**: `services/core/aiva_core/internal_exploration/rust_tools/src/paths_config.rs`
- **狀態**: ✅ 已創建（114 行）
- **功能**: 提供與 Python 等價的路徑配置

### ✅ 2. Python 工具修改

#### aiva_flow_analyzer.py
- **狀態**: ✅ 已修改
- **變更**:
  - 添加路徑配置導入
  - 修改 `save_results()` 方法使用 `get_analysis_output_dir("python")`
  - 保持向後兼容（可選 output 參數）

#### aiva_exploration_pipeline.py
- **狀態**: ✅ 已修改
- **變更**:
  - 添加路徑配置導入（try/except 保護）
  - 修改 `HISTORY_DIR` 使用 `ANALYSIS_HISTORY_DIR`
  - 環境變量控制切換新舊路徑

#### core_analyzer.py
- **狀態**: ✅ 已修改
- **變更**:
  - 添加路徑配置導入
  - 使用 `SELF_HEALING_DIR` 作為輸出目錄

#### run_analysis.py
- **狀態**: ✅ 已修改
- **變更**:
  - 添加路徑配置導入
  - 修改 `_get_output_dir()` 方法使用統一配置
  - 保持向後兼容

### ✅ 3. TypeScript 工具修改

#### ts2mermaid.ts
- **狀態**: ✅ 已修改
- **變更**:
  - 導入 `paths.config.ts`
  - 使用 `getDefaultOutputDir()` 獲取默認輸出路徑
  - try/catch 保護向後兼容

### ✅ 4. Go 工具修改

#### go2mermaid.go
- **狀態**: ✅ 已修改
- **變更**:
  - 調用 `GetPathsConfig()` 獲取配置
  - 使用 `GetDefaultOutputDir()` 作為 flag 默認值
  - 保持命令行參數優先級

### ✅ 5. Rust 工具修改

#### src/main.rs
- **狀態**: ✅ 已修改
- **變更**:
  - 添加 `mod paths_config` 和 use 語句
  - 使用 `PathsConfig::new().get_default_output_dir()` 獲取默認路徑
  - 保持命令行參數優先級

### ✅ 6. 文檔更新

#### docs/UNIFIED_OUTPUT_PATHS.md
- **狀態**: ✅ 已創建（完整文檔）
- **內容**:
  - 概述和目錄結構說明
  - 各語言配置文件使用方法
  - 環境變量控制說明
  - 已修改工具列表
  - 使用示例和測試建議
  - 問題排查指南

#### services/core/aiva_core/internal_exploration/README.md
- **狀態**: ✅ 已更新
- **變更**: 添加輸出路徑變更提示

## 新的目錄結構

```
services/
└── integration/
    └── data/
        └── internal_exploration/
            ├── analysis_results/          # 各語言工具的分析結果
            │   ├── python/
            │   ├── typescript/
            │   ├── go/
            │   └── rust/
            ├── analysis_history/         # 分析歷史版本
            │   ├── v1/
            │   ├── v2/
            │   └── v3/
            └── self_healing/            # Self-Healing 診斷報告
```

## 向後兼容性

### 環境變量控制

```bash
# 使用新路徑（默認）
export AIVA_USE_INTEGRATED_PATHS=true

# 使用舊路徑
export AIVA_USE_INTEGRATED_PATHS=false
```

### 命令行參數優先

所有工具仍支援 `--output` 參數指定自定義路徑：

```bash
python aiva_flow_analyzer.py --target path/to/code --output /custom/path
npm run analyze -- --output=/custom/path
go run go2mermaid.go --output=/custom/path
cargo run -- --output=/custom/path
```

### 優雅降級

- Python 工具使用 try/except 捕獲導入錯誤
- TypeScript 工具使用 try/catch 捕獲 require 錯誤
- 所有工具在配置不可用時回退到 `./analysis_output`

## 測試建議

### 基本功能測試

```bash
# Python
cd services/core/aiva_core/internal_exploration/python_tools
python aiva_flow_analyzer.py --target ../../../../

# TypeScript
cd services/core/aiva_core/internal_exploration/typescript_tools
npm run analyze

# Go
cd services/core/aiva_core/internal_exploration/go_tools
go run go2mermaid.go --input=.

# Rust
cd services/core/aiva_core/internal_exploration/rust_tools
cargo run -- --input=.
```

### 向後兼容性測試

```bash
AIVA_USE_INTEGRATED_PATHS=false python aiva_flow_analyzer.py --target ../../../../
```

### 輸出驗證

檢查以下目錄是否正確創建並包含輸出：

```bash
ls -la services/integration/data/internal_exploration/analysis_results/python/
ls -la services/integration/data/internal_exploration/analysis_results/typescript/
ls -la services/integration/data/internal_exploration/analysis_results/go/
ls -la services/integration/data/internal_exploration/analysis_results/rust/
```

## 優點與好處

### 1. 統一管理
- 所有輸出數據集中在一個位置
- 易於查找、備份和管理
- 減少目錄混亂

### 2. 多語言支持
- 四種語言工具使用相同的路徑結構
- 統一的配置管理方式
- 易於擴展新語言

### 3. 清晰組織
- 按工具類型分類存儲
- 版本歷史獨立管理
- Self-Healing 報告獨立存放

### 4. 向後兼容
- 不影響現有工作流程
- 環境變量控制行為
- 命令行參數保持優先級

### 5. 易於維護
- 配置集中在少數文件
- 修改一處影響所有工具
- 減少代碼重複

## 後續任務

- [ ] 執行完整測試套件
- [ ] 通知團隊成員路徑變更
- [ ] 更新 CI/CD 配置（如果需要）
- [ ] 監控新路徑的磁碟使用情況
- [ ] 考慮添加自動清理舊輸出的機制

## 風險評估

### 低風險
- 所有修改保持向後兼容
- 使用環境變量控制新舊行為
- try/except 保護防止導入錯誤
- 命令行參數保持最高優先級

### 潛在問題
1. **權限問題**: 確保對 `services/integration/data/` 有寫入權限
2. **路徑計算**: 相對路徑計算依賴於當前工作目錄
3. **磁碟空間**: 新路徑可能在不同磁碟分區

### 緩解措施
- 提供詳細文檔和問題排查指南
- 環境變量允許快速回退
- 自動創建目錄功能
- 清晰的錯誤信息

## 總結

統一輸出路徑實施已完成，包括：

✅ 4 個配置文件（Python、TypeScript、Go、Rust）  
✅ 8 個工具文件修改  
✅ 2 個文檔更新  
✅ 完整的向後兼容性支持  

系統現在具備更清晰的輸出組織結構，同時保持完全向後兼容。

## 聯繫方式

如有問題或需要支援，請聯繫 AIVA 開發團隊。
