# Tools - Integration Module

整合模組相關的工具和插件

## 目錄說明

此目錄包含 AIVA 整合模組相關的修復工具和外部插件：

### 🔧 修復工具

1. **`fix_all_schema_imports.py`**
   - 功能：批量修復 schemas 模組的導入問題
   - 用途：自動添加缺失的導入語句和修復導入路徑
   - 修復：HttpUrl、field_validator 等常見缺失導入

2. **`fix_field_validators.py`**
   - 功能：修正 Pydantic @field_validator 方法簽名
   - 修復：將 `self` 參數改為 `cls` 並添加 `@classmethod`
   - 涉及：schemas 目錄下的所有相關檔案

3. **`fix_metadata_reserved.py`**
   - 功能：修復 SQLAlchemy metadata 保留字問題
   - 修復：將 `metadata` 欄位改為 `extra_metadata`
   - 目標：core/storage/models.py

4. **`update_imports.py`**
   - 功能：批量更新 import 路徑
   - 修復：將 `aiva_common` 改為 `services.aiva_common`
   - 範圍：scan、core、function、integration 目錄

### 🔌 插件目錄

#### 1. **`aiva-contracts-tooling/`**
**功能**: JSON Schema 和 TypeScript 類型生成工具
- 從 `aiva_schemas_plugin` 自動匯出 JSON Schema
- 生成 TypeScript `.d.ts` 類型定義
- 支援 CLI 操作和 CI/CD 整合

**主要命令**:
```bash
# 列出所有模型
aiva-contracts list-models

# 匯出 JSON Schema
aiva-contracts export-jsonschema --out ./schemas/aiva_schemas.json

# 生成 TypeScript 定義
aiva-contracts gen-ts --json ./schemas/aiva_schemas.json --out ./schemas/aiva_schemas.d.ts
```

#### 2. **`aiva-enums-plugin/`**
**功能**: 集中管理和導出枚舉類型
- Python 端：轉接 `aiva_common.enums`
- TypeScript 端：生成 `enums.ts` 檔案
- 統一的枚舉管理入口

**主要功能**:
```bash
# 生成 TypeScript 枚舉
python scripts/gen_ts_enums.py --out ./schemas/enums.ts
```

#### 3. **`aiva-schemas-plugin/`**
**功能**: 統一的 Schema 插件系統
- 轉接層：re-export `aiva_common.schemas` 
- 批量重構：統一導入路徑
- 清理工具：移除重複的 schemas.py

**重構工具**:
```bash
# 批量改寫匯入並清理檔案
python scripts/refactor_imports_and_cleanup.py --repo-root ./services

# 複製到自含插件
python scripts/copy_into_plugin.py --repo-root ./services
```

#### 4. **`aiva-go-plugin/`**
**功能**: Go 語言結構體生成
- 從 Python schemas 生成 Go 結構體
- 支援類型映射和標記生成
- Go FFI 整合支援

### 🎯 模組分類

這些工具屬於 **integration** 模組，主要處理：
- 外部系統整合
- 多語言代碼生成
- Schema 轉換和同步
- 插件系統管理

### 🔧 使用方式

所有修復工具都使用相對路徑：

```bash
# 修復 schema 導入問題
python tools/integration/fix_all_schema_imports.py

# 修復 field validator 簽名
python tools/integration/fix_field_validators.py

# 修復 metadata 保留字
python tools/integration/fix_metadata_reserved.py

# 更新 import 路徑
python tools/integration/update_imports.py
```

### 📊 修復狀態

✅ **所有修復工具已更新**
- [x] 路徑計算：使用相對路徑從項目根目錄計算
- [x] 硬編碼清理：移除所有絕對路徑硬編碼
- [x] 跨平台兼容：支援 Windows/Linux/macOS
- [x] 錯誤處理：改善異常處理機制

### 🔗 插件整合

**開發流程**:
1. 修改 Python schemas → 運行 contracts-tooling
2. 更新枚舉定義 → 運行 enums-plugin  
3. 重構 schema 結構 → 運行 schemas-plugin
4. 需要 Go 整合 → 運行 go-plugin

**CI/CD 建議**:
- 在 PR 中自動運行 schema 同步
- 檢查 TypeScript 定義是否最新
- 驗證多語言類型一致性

### 🔗 相關資源

- [插件開發指南](../README.md)
- [Schema 管理最佳實踐](../../guides/architecture/SCHEMA_GUIDE.md)
- [跨語言開發標準](../../guides/development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md)