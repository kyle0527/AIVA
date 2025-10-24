# AIVA 工具集 - 五大模組架構 🛠️

本目錄包含 AIVA 專案的各種開發、調試和自動化工具，已完全重組為五大模組架構，提供全方位的開發支援。

## 📋 目錄結構總覽

```
tools/
├── README.md                    # 本文件 - 完整工具集說明
├── common/                      🏗️ 基礎架構和共用工具
├── core/                        🧠 核心分析引擎  
├── scan/                        🔍 掃描檢測引擎
├── integration/                 🔗 整合服務
└── features/                    ⚡ 功能檢測增強
```

---

## 🏗️ Common 模組 - 基礎架構和共用工具

**位置**: `tools/common/`  
**目的**: 提供項目基礎設施、開發工具和通用功能

### 📊 分析工具 (analysis/)
- `analyze_aiva_common_status.py` - AIVA Common 重構狀態分析
- `analyze_core_modules.py` - 核心模組程式碼分析
- `analyze_cross_language_ai.py` - 跨語言功能和 AI 模組完備性分析
- `analyze_enums.py` - 枚舉分析配置
- `analyze_missing_schemas.py` - 缺失 Schema 分析

### 🤖 自動化腳本 (automation/)
- `check_script_functionality.py` - 腳本功能性檢查器
- `cleanup_deprecated_files.ps1` - 清理廢棄文件 (PowerShell)
- `generate-contracts.ps1` - 生成合約 (PowerShell)
- `generate-official-contracts.ps1` - 生成官方合約 (PowerShell)
- `generate_project_report.sh` - 生成專案報告 (Shell)

### 🛠️ 開發工具 (development/)
- `analyze_codebase.py` - 綜合程式碼庫分析工具
- `generate_complete_architecture.py` - 完整架構圖生成器
- `generate_mermaid_diagrams.py` - Mermaid 圖表生成
- `py2mermaid.py` - Python 轉 Mermaid 流程圖

### 📈 監控工具 (monitoring/)
- `system_health_check.ps1` - 系統健康檢查 (PowerShell)

### ✅ 品質工具 (quality/)
- `find_non_cp950_filtered.py` - CP950 編碼兼容性檢查
- `markdown_check.py` - Markdown 語法檢查
- `replace_emoji.py` - 表情符號替換為中文標籤
- `replace_non_cp950.py` - 非 CP950 字符替換

### 📋 Schema 工具 (schema/)
- `schema_manager.py` - Schema 管理器
- `schema_validator.py` - Schema 驗證器
- `unified_schema_manager.py` - 統一 Schema 管理器
- `analyze_schema_impact.ps1` - Schema 影響分析 (PowerShell)

### 🔧 根目錄工具
- `create_enums_structure.py` - 創建枚舉結構
- `create_schemas_files.py` - 創建 Schema 文件
- `diagnose_missing_enums.py` - 診斷缺失枚舉
- `generate_official_schemas.py` - 生成官方 Schema
- `generate_programming_language_support.py` - 程式語言支援生成
- `generate_typescript_interfaces.py` - 生成 TypeScript 介面
- `import_path_checker.py` - 導入路徑檢查器

---

## 🧠 Core 模組 - 核心分析引擎

**位置**: `tools/core/`  
**目的**: 核心架構分析、遷移管理和結構驗證

### 🔍 核心工具
- `comprehensive_migration_analysis.py` - 綜合遷移分析工具
- `verify_migration_completeness.py` - 驗證遷移完整性
- `compare_schemas.py` - 比較新舊 schema 檔案
- `delete_migrated_files.py` - 刪除已遷移的舊檔案

### 主要功能
- 核心架構分析
- 遷移狀態管理  
- 結構完整性驗證
- 檔案清理自動化

### 使用方式
```bash
# 執行綜合分析
python tools/core/comprehensive_migration_analysis.py

# 驗證遷移完整性  
python tools/core/verify_migration_completeness.py

# 比較 schema 檔案
python tools/core/compare_schemas.py

# 清理已遷移檔案
python tools/core/delete_migrated_files.py
```

---

## 🔍 Scan 模組 - 掃描檢測引擎

**位置**: `tools/scan/`  
**目的**: 程式碼掃描、檢測和功能性分析

### 檔案掃描工具
- `mark_nonfunctional_scripts.py` - 在專案樹中標注腳本功能狀態
- `apply_marks_to_tree.py` - 直接在樹狀圖上標註功能狀態  
- `list_no_functionality_files.py` - 列出需要實作的無功能檔案
- `extract_enhanced.py` - 提取 Enhanced 類別

### 功能標記系統
- ❌ **無功能** - 需要完整實作
- 🔶 **基本架構** - 需要補充功能
- ⚠️ **部分功能** - 可以改進  
- ✅ **完整功能** - 正常運作

### 工作流程
```bash
# 1. 檢查腳本功能性
python tools/common/automation/check_script_functionality.py

# 2. 標記樹狀圖
python tools/scan/mark_nonfunctional_scripts.py

# 3. 列出需要改進的檔案
python tools/scan/list_no_functionality_files.py

# 4. 提取增強類別
python tools/scan/extract_enhanced.py
```

---

## 🔗 Integration 模組 - 整合服務

**位置**: `tools/integration/`  
**目的**: 外部系統整合、多語言代碼生成和插件管理

### 🔧 修復工具
- `fix_all_schema_imports.py` - 批量修復 schemas 模組導入問題
- `fix_field_validators.py` - 修正 Pydantic @field_validator 方法簽名
- `fix_metadata_reserved.py` - 修復 SQLAlchemy metadata 保留字問題  
- `update_imports.py` - 批量更新 import 路徑

### 🔌 插件系統

#### aiva-contracts-tooling/
**功能**: JSON Schema 和 TypeScript 類型生成
```bash
# 列出所有模型
aiva-contracts list-models

# 匯出 JSON Schema  
aiva-contracts export-jsonschema --out ./schemas/aiva_schemas.json

# 生成 TypeScript 定義
aiva-contracts gen-ts --json ./schemas/aiva_schemas.json --out ./schemas/aiva_schemas.d.ts
```

#### aiva-enums-plugin/
**功能**: 集中管理和導出枚舉類型
```bash
# 生成 TypeScript 枚舉
python scripts/gen_ts_enums.py --out ./schemas/enums.ts
```

#### aiva-schemas-plugin/
**功能**: 統一的 Schema 插件系統
```bash
# 批量改寫匯入並清理檔案
python scripts/refactor_imports_and_cleanup.py --repo-root ./services

# 複製到自含插件
python scripts/copy_into_plugin.py --repo-root ./services
```

#### aiva-go-plugin/
**功能**: Go 語言結構體生成
- 從 Python schemas 生成 Go 結構體
- 支援類型映射和標記生成
- Go FFI 整合支援

---

## ⚡ Features 模組 - 功能檢測增強

**位置**: `tools/features/`  
**目的**: 功能增強和圖表品質優化

### 🎨 圖表優化工具

#### mermaid_optimizer.py ⭐
**現代化 Mermaid.js v10+ 圖表優化器**
- 符合最新 Mermaid.js 官方語法規範
- 支援現代主題配置和自定義主題變數
- 支援 HTML 標籤和 CSS 類
- 提供多種節點形狀和連線樣式
- 支援無障礙功能和響應式佈局

**節點形狀**:
- `rectangle` - 標準矩形
- `rounded` - 圓角矩形  
- `stadium` - 體育場形 (Pill)
- `circle` - 圓形
- `rhombus` - 菱形 (決策)
- `hexagon` - 六角形

**連線類型**:
- `arrow` - 實線箭頭
- `dotted` - 虛線箭頭
- `thick` - 粗實線箭頭
- `bidirectional` - 雙向箭頭
- `x_arrow` - X型終止
- `circle_arrow` - 圓型終止

**使用範例**:
```python
from mermaid_optimizer import MermaidOptimizer

# 建立優化器
optimizer = MermaidOptimizer()

# 創建現代化節點
node = optimizer.create_node(
    "ai-core", "AI 核心", "AI Core Engine", 
    "Bio Neuron Network", icon="🤖"
)

# 生成完整圖表
header = optimizer.generate_header("flowchart TD")
```

### 內容處理工具
- `remove_init_marks.py` - 移除 `__init__.py` 檔案的功能標記

---

## 🚀 使用指南

### 快速開始
```bash
# 檢查整體專案狀態
python tools/common/automation/check_script_functionality.py

# 分析代碼庫
python tools/common/development/analyze_codebase.py

# 生成架構圖
python tools/common/development/generate_complete_architecture.py

# 驗證 Schema
python tools/common/schema/schema_validator.py
```

### 典型工作流

#### 1. 代碼品質檢查工作流
```bash
# Step 1: 檢查編碼問題
python tools/common/quality/find_non_cp950_filtered.py

# Step 2: 檢查導入路徑
python tools/common/import_path_checker.py --check

# Step 3: 分析代碼結構
python tools/common/development/analyze_codebase.py

# Step 4: 標記功能狀態
python tools/scan/mark_nonfunctional_scripts.py
```

#### 2. Schema 管理工作流
```bash
# Step 1: 驗證 Schema 完整性
python tools/common/schema/schema_validator.py

# Step 2: 生成官方 Schema
python tools/common/generate_official_schemas.py  

# Step 3: 生成 TypeScript 介面
python tools/common/generate_typescript_interfaces.py

# Step 4: 同步到插件
python tools/integration/aiva-contracts-tooling/scripts/export-jsonschema.py
```

#### 3. 遷移和重構工作流  
```bash
# Step 1: 分析遷移狀態
python tools/core/comprehensive_migration_analysis.py

# Step 2: 修復導入問題  
python tools/integration/fix_all_schema_imports.py

# Step 3: 驗證完整性
python tools/core/verify_migration_completeness.py

# Step 4: 清理舊檔案
python tools/core/delete_migrated_files.py
```

### 常用命令組合

#### Windows 環境 (PowerShell)
```powershell
# 完整檢查流程
python tools/common/automation/check_script_functionality.py
python tools/common/development/analyze_codebase.py  
python tools/scan/mark_nonfunctional_scripts.py
python tools/common/quality/find_non_cp950_filtered.py

# Schema 同步流程
python tools/common/schema/schema_validator.py
python tools/common/generate_official_schemas.py
python tools/integration/aiva-contracts-tooling/export-jsonschema.py
```

#### Unix/Linux 環境
```bash
# 批量執行分析
for tool in tools/common/development/*.py; do
    echo "執行: $tool"
    python "$tool"
done

# 檢查所有模組狀態
python tools/core/comprehensive_migration_analysis.py && \
python tools/scan/mark_nonfunctional_scripts.py && \
python tools/integration/fix_all_schema_imports.py
```

---

## 🔧 技術規範

### 路徑標準化
所有工具已統一使用相對路徑計算：
```python
# 標準路徑計算模式
project_root = Path(__file__).parent.parent.parent  # 從 tools/module/ 計算
```

### 編碼標準
- **檔案編碼**: UTF-8
- **Windows 兼容**: 支援 CP950 編碼檢查
- **跨平台**: Windows/Linux/macOS 通用

### Python 版本支援
- **最低要求**: Python 3.8+
- **建議版本**: Python 3.10+
- **類型提示**: 使用現代 typing 語法

### 依賴管理
```bash
# 核心依賴
pip install pydantic pathlib

# 開發依賴  
pip install datamodel-code-generator mermaid-cli

# 可選依賴
pip install grpcio grpcio-tools  # integration 模組
```

---

## 📊 統計數據

### 工具數量統計
- **Common 模組**: 25個工具 (8個根目錄 + 17個子目錄)
- **Core 模組**: 4個工具
- **Scan 模組**: 4個工具  
- **Integration 模組**: 4個工具 + 4個插件
- **Features 模組**: 2個工具
- **總計**: 39個主要工具 + 4個插件系統

### 語言分布
- **Python**: 35個腳本
- **PowerShell**: 4個腳本
- **Shell**: 1個腳本
- **TypeScript/Node.js**: 4個插件

### 功能覆蓋
- ✅ **代碼分析**: 完整覆蓋
- ✅ **Schema 管理**: 完整覆蓋
- ✅ **多語言支援**: TypeScript、Go 支援
- ✅ **自動化流程**: CI/CD 整合
- ✅ **品質保證**: 編碼檢查、語法驗證

---

## 🔗 相關資源

### 文檔資源
- [五模組架構說明](../README.md)
- [路徑管理最佳實踐](../docs/IMPORT_PATH_BEST_PRACTICES.md)
- [多語言整合文檔](../docs/ARCHITECTURE_MULTILANG.md)
- [Schema 管理指南](../docs/SCHEMAS_DIRECTORIES_EXPLANATION.md)

### 輸出目錄
- `_out/analysis/` - 分析報告
- `_out/architecture_diagrams/` - 架構圖表
- `_out/reports/` - 各類報告
- `schemas/` - 生成的 Schema 文件

### 備份目錄  
- `emoji_backups/` - 表情符號替換備份
- `emoji_backups2/` - 非 CP950 字符替換備份

---

## 🚧 維護指南

### 添加新工具
1. **選擇模組**: 根據功能選擇適當的模組目錄
2. **路徑規範**: 使用標準相對路徑計算
3. **文檔更新**: 更新對應模組的 README
4. **主文檔同步**: 更新本 README 文件

### 路徑管理
```python
# ✅ 正確的路徑計算
project_root = Path(__file__).parent.parent.parent

# ❌ 避免硬編碼
project_root = Path("C:/absolute/path")  # 不要這樣做
```

### 代碼規範
- 遵循 PEP 8 代碼風格
- 添加詳細的文檔字符串
- 包含錯誤處理機制
- 提供命令行參數支援

### 測試規範
```bash
# 語法檢查
python -m py_compile tools/module/script.py

# 執行測試
python -m pytest tools/tests/

# 類型檢查  
mypy tools/module/script.py
```

---

## 📝 更新記錄

### 2024-10-24 - v3.0 大版本更新
- ✅ **完整重組**: 按五大模組重新組織所有工具
- ✅ **路徑標準化**: 39個工具全部統一路徑計算方式
- ✅ **語法驗證**: 100% 工具通過語法檢查
- ✅ **文檔完善**: 每個模組都有詳細 README
- ✅ **插件整合**: 4個多語言插件系統完成整合

### 2024-10-19 - v2.0 模組化
- 🔄 重組工具目錄結構
- 📋 創建模組化 README 文件
- 🔧 修復路徑相關問題

### 2024-10-13 - v1.0 初版
- 🎉 建立基礎工具集
- 📊 添加代碼分析功能
- 🔍 實現腳本功能檢查

---

## 🎯 未來計劃

### 短期目標 (1-2週)
- [ ] 添加單元測試覆蓋
- [ ] 創建統一的 CLI 介面
- [ ] 增強錯誤處理和日誌記錄
- [ ] 添加配置文件支援

### 中期目標 (1-2個月)
- [ ] GitHub Actions 整合
- [ ] 自動化 Schema 同步
- [ ] Web UI 儀表板
- [ ] 效能監控工具

### 長期目標 (3-6個月)  
- [ ] AI 驅動的代碼分析
- [ ] 智能重構建議
- [ ] 跨語言類型安全檢查
- [ ] 分散式工具執行

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2024-10-24  
**版本**: v3.0  
**工具總數**: 39+ 工具 + 4個插件

Last Updated: 2025-10-13
