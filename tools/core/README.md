# Tools - Core Module

核心分析和遷移管理工具

## 目錄說明

此目錄包含 AIVA 核心模組相關的分析和遷移管理工具：

### 🔍 檔案列表

1. **`comprehensive_migration_analysis.py`**
   - 功能：綜合遷移分析工具
   - 用途：分析項目結構變化和遷移狀態
   - 輸出：詳細的遷移報告和統計

2. **`verify_migration_completeness.py`**
   - 功能：驗證遷移完整性
   - 用途：確認所有檔案和模組已正確遷移
   - 檢查：依賴關係、匯入路徑、結構完整性

3. **`compare_schemas.py`**
   - 功能：比較新舊 schema 檔案
   - 用途：驗證 schema 重構的正確性
   - 比對：類別定義、結構變化、遺漏項目

4. **`delete_migrated_files.py`**
   - 功能：刪除已遷移的舊檔案
   - 用途：清理重構後的項目結構
   - 安全：驗證遷移完成後再執行刪除

### 🎯 模組分類

這些工具屬於 **core** 模組，主要處理：
- 核心架構分析
- 遷移狀態管理
- 結構完整性驗證
- 檔案清理自動化

### 🔧 使用方式

所有腳本都使用相對路徑，可從任意位置執行：

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

### 📊 修復狀態

✅ **所有檔案已修復**
- [x] 路徑計算：使用 `Path(__file__).parent.parent.parent` 計算項目根目錄
- [x] 工作目錄：自動切換到正確的 `aiva_common` 目錄
- [x] 相對路徑：移除硬編碼的絕對路徑
- [x] 語法檢查：通過所有語法驗證

### 🔗 相關資源

- [五模組架構說明](../../README.md)
- [依賴管理最佳實踐](../../guides/development/DEPENDENCY_MANAGEMENT_GUIDE.md)