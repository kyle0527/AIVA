# AIVA Common Tools Directory
# AIVA 通用工具目錄

本目錄包含 AIVA 專案中的通用工具腳本，按功能分類整理到不同子目錄中。

## 目錄結構 | Directory Structure

```
tools/common/
├── README.md                                # 本文件
├── create_enums_structure.py               # 創建枚舉結構
├── create_schemas_files.py                 # 創建Schema文件
├── diagnose_missing_enums.py              # 診斷缺失枚舉
├── generate_official_schemas.py           # 生成官方Schema
├── generate_programming_language_support.py # 程式語言支援生成
├── generate_typescript_interfaces.py      # 生成TypeScript接口
├── import_path_checker.py                 # 導入路徑檢查器
│
├── analysis/                              # 分析工具
│   ├── analyze_aiva_common_status.py      # AIVA Common狀態分析
│   ├── analyze_core_modules.py           # 核心模組分析
│   ├── analyze_cross_language_ai.py      # 跨語言AI分析
│   ├── analyze_enums.py                  # 枚舉分析配置
│   └── analyze_missing_schemas.py        # 缺失Schema分析
│
├── automation/                           # 自動化腳本
│   ├── check_script_functionality.py     # 腳本功能檢查
│   ├── cleanup_deprecated_files.ps1      # 清理廢棄文件 (PowerShell)
│   ├── generate-contracts.ps1            # 生成合約 (PowerShell)
│   ├── generate-official-contracts.ps1   # 生成官方合約 (PowerShell)
│   └── generate_project_report.sh        # 生成專案報告 (Shell)
│
├── development/                          # 開發工具
│   ├── analyze_codebase.py              # 程式碼庫分析
│   ├── generate_complete_architecture.py # 生成完整架構圖
│   ├── generate_mermaid_diagrams.py     # 生成Mermaid圖表
│   └── py2mermaid.py                    # Python轉Mermaid
│
├── monitoring/                          # 監控工具
│   └── system_health_check.ps1         # 系統健康檢查 (PowerShell)
│
├── quality/                            # 代碼品質工具
│   ├── find_non_cp950_filtered.py      # 查找非CP950字符
│   ├── markdown_check.py               # Markdown檢查
│   ├── replace_emoji.py                # 替換表情符號
│   └── replace_non_cp950.py           # 替換非CP950字符
│
└── schema/                            # Schema管理工具
    ├── analyze_schema_impact.ps1      # Schema影響分析 (PowerShell)
    ├── schema_manager.py              # Schema管理器
    ├── schema_validator.py            # Schema驗證器
    └── unified_schema_manager.py      # 統一Schema管理器
```

## 工具分類說明 | Tool Categories

### 📊 分析工具 (analysis/)
- **目的**: 分析代碼結構、模組狀態、跨語言支援等
- **主要功能**: 代碼分析、模組檢查、枚舉診斷
- **使用場景**: 代碼品質評估、架構分析、問題診斷

### 🤖 自動化腳本 (automation/)
- **目的**: 自動化常見任務和工作流程
- **主要功能**: 腳本功能檢查、文件清理、合約生成
- **使用場景**: CI/CD流程、維護任務、批量處理

### 🛠️ 開發工具 (development/)
- **目的**: 輔助開發過程的工具
- **主要功能**: 架構圖生成、代碼庫分析、文檔生成
- **使用場景**: 架構設計、文檔編寫、開發調試

### 📈 監控工具 (monitoring/)
- **目的**: 系統健康狀態監控
- **主要功能**: 系統健康檢查、狀態報告
- **使用場景**: 系統維護、問題預警、性能監控

### ✅ 品質工具 (quality/)
- **目的**: 代碼品質保證和改進
- **主要功能**: 編碼檢查、格式驗證、內容替換
- **使用場景**: 代碼審查、格式統一、品質提升

### 📋 Schema工具 (schema/)
- **目的**: Schema定義管理和驗證
- **主要功能**: Schema創建、驗證、管理、同步
- **使用場景**: 數據模型管理、接口定義、類型檢查

## 使用方式 | Usage

### 基本使用
所有Python腳本都可以直接執行：
```bash
# 在專案根目錄執行
python tools/common/[category]/[script_name].py
```

### 常用工具示例
```bash
# 分析代碼庫
python tools/common/development/analyze_codebase.py

# 檢查導入路徑
python tools/common/import_path_checker.py --check

# 驗證Schema
python tools/common/schema/schema_validator.py

# 生成架構圖
python tools/common/development/generate_complete_architecture.py
```

### PowerShell 腳本
```powershell
# 執行PowerShell腳本
pwsh -File tools/common/automation/cleanup_deprecated_files.ps1
```

## 路徑標準化 | Path Standardization

所有腳本已統一使用相對路徑計算：
```python
# 標準路徑計算模式
project_root = Path(__file__).parent.parent.parent.parent
```

這確保腳本在任何環境下都能正確定位AIVA專案根目錄。

## 維護指南 | Maintenance Guide

### 添加新工具
1. 選擇適當的分類目錄
2. 使用標準的路徑計算方式
3. 添加適當的文檔字符串
4. 更新本README文件

### 路徑規範
- 使用 `Path(__file__).parent.parent.parent.parent` 計算專案根目錄
- 避免硬編碼絕對路徑
- 使用 Path 對象進行路徑操作

### 代碼規範
- 遵循PEP 8代碼風格
- 添加類型提示
- 包含錯誤處理
- 提供詳細的幫助文檔

## 技術棧 | Technology Stack

- **Python**: 主要開發語言
- **PowerShell**: Windows自動化腳本
- **Shell**: Unix/Linux自動化腳本
- **Pydantic**: 數據驗證和序列化
- **Pathlib**: 現代路徑處理
- **AST**: 代碼解析和分析
- **Mermaid**: 圖表生成

## 依賴管理 | Dependencies

大部分工具只依賴Python標準庫，特殊依賴包括：
- `pydantic`: Schema定義和驗證
- `pathlib`: 路徑處理（Python 3.4+內建）

## 故障排除 | Troubleshooting

### 常見問題
1. **路徑錯誤**: 確保在AIVA專案根目錄執行
2. **導入錯誤**: 檢查Python路徑和虛擬環境
3. **權限問題**: 確保腳本有執行權限
4. **編碼問題**: 大多數腳本使用UTF-8編碼

### 調試技巧
```bash
# 啟用詳細輸出
python tools/common/[script] --verbose

# 檢查語法
python -m py_compile tools/common/[script].py

# 獲取幫助
python tools/common/[script].py --help
```

## 更新記錄 | Change Log

### 2024-10-24
- ✅ 完成所有Python腳本的路徑標準化修復
- ✅ 驗證所有腳本語法正確性
- ✅ 按功能重新組織目錄結構
- ✅ 創建綜合性README文檔

### 未來計劃
- [ ] 添加單元測試
- [ ] 創建統一的CLI接口
- [ ] 增強錯誤處理和日誌記錄
- [ ] 添加配置文件支援

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2024-10-24  
**版本**: 2.0