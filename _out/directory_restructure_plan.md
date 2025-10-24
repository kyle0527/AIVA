# 🗂️ AIVA 目錄重組方案

## 📊 現狀分析

### 目前問題
1. **重複功能**: Schema驗證器存在於3個位置
2. **命名混亂**: 相似功能檔案命名不一致  
3. **層級不清**: tools混合了開發工具和業務工具
4. **測試薄弱**: tests目錄缺乏完整的測試體系

## 🎯 重組目標

1. **消除重複**: 合併相同功能的工具
2. **統一命名**: 建立清晰的命名規範
3. **明確分層**: 區分開發工具、業務工具、系統工具
4. **完善測試**: 建立完整的測試框架

---

# 📁 新目錄結構

## 1️⃣ **tests/** → **testing/**
```
testing/
├── unit/                    # 單元測試
│   ├── common/             # 通用模組測試
│   ├── core/               # 核心模組測試  
│   ├── scan/               # 掃描模組測試
│   ├── integration/        # 整合模組測試
│   └── features/           # 功能模組測試
├── integration/            # 整合測試
│   ├── api/               # API測試
│   ├── database/          # 資料庫測試
│   ├── messaging/         # 消息系統測試
│   └── workflow/          # 工作流測試
├── system/                # 系統測試
│   ├── e2e/              # 端到端測試
│   ├── performance/       # 效能測試
│   ├── security/         # 安全測試
│   └── compatibility/    # 相容性測試
├── fixtures/              # 測試數據
├── mocks/                # 模擬對象
├── utilities/            # 測試工具
├── conftest.py           # pytest配置
└── README.md
```

## 2️⃣ **tools/** → **devtools/** + **utilities/**

### **devtools/** (開發工具)
```
devtools/
├── analysis/              # 代碼分析
│   ├── codebase_analyzer.py
│   ├── complexity_checker.py
│   └── dependency_mapper.py
├── schema/               # Schema工具 (合併版)
│   ├── schema_manager.py        # 統一Schema管理器
│   ├── schema_validator.py      # 統一驗證器
│   ├── schema_generator.py      # 代碼生成器
│   └── cross_lang_sync.py       # 跨語言同步
├── codegen/              # 代碼生成
│   ├── contract_generator.py
│   ├── interface_generator.py
│   └── template_engine.py
├── quality/              # 代碼品質
│   ├── encoding_checker.py
│   ├── style_validator.py
│   └── documentation_checker.py
├── migration/            # 遷移工具
│   ├── import_updater.py
│   ├── structure_migrator.py
│   └── legacy_converter.py
└── README.md
```

### **utilities/** (系統工具)
```
utilities/
├── monitoring/           # 監控工具
│   ├── health_checker.py
│   ├── performance_monitor.py
│   └── resource_tracker.py
├── automation/          # 自動化工具
│   ├── backup_manager.py
│   ├── cleanup_scheduler.py
│   └── report_generator.py
├── diagnostics/         # 診斷工具
│   ├── system_checker.py
│   ├── connectivity_tester.py
│   └── error_analyzer.py
└── README.md
```

## 3️⃣ **scripts/** (保持但優化)
```
scripts/
├── launcher/            # 啟動器 ✅
├── deployment/          # 部署腳本 ✅  
├── testing/            # 測試腳本 → 移至 testing/system/
├── validation/         # 驗證腳本 → 移至 devtools/quality/
├── integration/        # 整合腳本 ✅
├── reporting/          # 報告腳本 → 移至 utilities/automation/
├── maintenance/        # 維護腳本 ✅
├── setup/             # 環境設置 ✅
└── conversion/        # 轉換工具 → 移至 utilities/automation/
```

---

# 🔄 遷移行動計劃

## 階段 1: 重複功能合併 (高優先級)

### 1.1 Schema工具統一
- **目標**: 合併3個Schema驗證器為1個
- **檔案**: 
  - `tools/schema/schema_validator.py` (主要)
  - `services/aiva_common/tools/schema_validator.py` (次要)
  - `tools/schema/schema_manager.py` (管理功能)

### 1.2 測試工具整合
- **目標**: 將散落的測試腳本整合到testing目錄
- **來源**: `scripts/testing/` → `testing/system/`

## 階段 2: 目錄重新組織 (中優先級)

### 2.1 創建新目錄結構
### 2.2 移動檔案到新位置
### 2.3 更新導入路徑

## 階段 3: 文檔和規範 (低優先級)

### 3.1 更新README文件
### 3.2 建立命名規範
### 3.3 創建使用指南

---

# 🎯 具體改進建議

## Schema工具合併方案

### 新的統一Schema管理器
```python
# devtools/schema/schema_manager.py
class UnifiedSchemaManager:
    """統一Schema管理器 - 合併所有Schema相關功能"""
    
    def __init__(self):
        self.validator = SchemaValidator()
        self.generator = SchemaGenerator() 
        self.sync_tool = CrossLangSync()
    
    def validate_all(self) -> bool:
        """執行完整驗證"""
        
    def generate_code(self, languages: list[str]) -> bool:
        """生成多語言代碼"""
        
    def sync_across_languages(self) -> bool:
        """跨語言同步"""
```

## 測試框架現代化

### 新的測試結構
```python
# testing/conftest.py - pytest全局配置
# testing/unit/ - 單元測試
# testing/integration/ - 整合測試  
# testing/system/ - 系統測試
```

## 命名規範統一

### 檔案命名規則
- **檢查工具**: `*_checker.py`
- **驗證工具**: `*_validator.py`
- **生成工具**: `*_generator.py`
- **管理工具**: `*_manager.py`
- **測試腳本**: `test_*.py`

---

# 📈 預期效果

## 即時效果
- ✅ 消除重複代碼
- ✅ 提高開發效率
- ✅ 減少維護成本

## 長期效果  
- ✅ 更好的代碼組織
- ✅ 更容易的工具發現
- ✅ 更簡單的測試執行

---

**建立時間**: 2025-10-24  
**預計完成**: 2025-10-26  
**責任人**: DevOps Team