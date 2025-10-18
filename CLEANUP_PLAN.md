# AIVA 廢棄檔案清理計劃

## 📋 清理目標

清理 AIVA 專案中的廢棄檔案、備份檔案和臨時檔案，保持程式碼庫整潔。

## 🔍 發現的廢棄檔案

### 第一類：備份檔案 (.backup)
```
services/scan/aiva_scan/dynamic_engine/example_usage.py.backup
services/core/aiva_core/ui_panel/server.py.backup
services/core/aiva_core/ui_panel/dashboard.py.backup
services/core/aiva_core/ai_engine/bio_neuron_core.py.backup
services/core/aiva_core/ai_engine/knowledge_base.py.backup
services/core/aiva_core/ai_engine_backup/knowledge_base.py.backup
services/core/aiva_core/ai_engine_backup/bio_neuron_core.py.backup
services/function/function_sca_go/internal/analyzer/enhanced_analyzer.go.backup
```

### 第二類：整個備份目錄
```
services/core/aiva_core/ai_engine_backup/  (整個目錄)
├── __init__.py
├── bio_neuron_core.py
├── bio_neuron_core.py.backup
├── bio_neuron_core_v2.py
├── knowledge_base.py
├── knowledge_base.py.backup
└── tools.py
```

### 第三類：歸檔目錄已存在但可能仍有清理空間
```
_archive/                   # 已存在的歸檔目錄
_out/                      # 輸出檔案目錄 
```

## 🗂️ 清理策略

### 階段 1：安全備份
1. 創建備份目錄 `_cleanup_backup/20241018/`
2. 將所有待刪除檔案複製到備份目錄
3. 記錄檔案清單和大小

### 階段 2：刪除備份檔案
1. 刪除所有 `.backup` 檔案
2. 清理 `ai_engine_backup` 整個目錄
3. 驗證主要功能檔案完整性

### 階段 3：清理報告
1. 統計清理檔案數量和大小
2. 驗證系統功能正常
3. 更新文檔

## 🔧 執行腳本

```powershell
# 自動化清理腳本
.\tools\cleanup_deprecated_files.ps1
```

執行將會：
- 安全備份所有待刪除檔案
- 刪除廢棄檔案  
- 生成清理報告
- 驗證系統完整性