# AIVA Scan 模組 - 掃描檢測引擎 🔍

## 📋 模組概述

Scan 模組提供程式碼掃描、檢測和分析功能，專注於檔案內容分析和功能性評估。

## 🔧 工具清單

### 檔案掃描工具

#### 1. `mark_nonfunctional_scripts.py`
**功能**: 在專案樹中標注腳本的功能狀態
- 分析腳本功能性報告
- 在樹狀圖中添加功能標記
- 生成功能性統計摘要

**標記說明**:
- ❌ 無功能 - 需要完整實作
- 🔶 基本架構 - 需要補充功能  
- ⚠️ 部分功能 - 可以改進
- ✅ 完整功能 - 正常運作

**使用方式**:
```bash
python tools/scan/mark_nonfunctional_scripts.py
```

#### 2. `apply_marks_to_tree.py`
**功能**: 直接在樹狀圖上標註功能狀態
- 讀取功能分析結果
- 在樹狀圖檔案上直接添加標記
- 保持樹狀結構完整性

**使用方式**:
```bash
python tools/scan/apply_marks_to_tree.py
```

#### 3. `list_no_functionality_files.py`
**功能**: 列出需要實作的無功能檔案
- 過濾出無功能的腳本檔案
- 排除 `__init__.py` 等基礎檔案
- 提供檔案路徑和問題原因

**使用方式**:
```bash
python tools/scan/list_no_functionality_files.py
```

#### 4. `extract_enhanced.py`
**功能**: 提取 Enhanced 類別
- 從主要 schemas 檔案中提取增強版類別
- 自動生成獨立的 enhanced.py 模組
- 支援動態路徑查找

**使用方式**:
```bash
python tools/scan/extract_enhanced.py
```

## 📂 依賴關係

### 輸入檔案
- `_out/script_functionality_report.json` - 腳本功能分析報告
- `_out/tree_ultimate_chinese_*.txt` - 專案樹狀圖
- `services/aiva_common/schemas.py` - 主要 Schema 定義

### 輸出檔案
- `_out/tree_with_functionality_marks.txt` - 標記版樹狀圖
- `schemas/enhanced.py` - 提取的增強類別

## 🚀 使用工作流

### 1. 功能性分析工作流
```bash
# 1. 檢查腳本功能性 (需要先運行 common/automation/check_script_functionality.py)
python tools/common/automation/check_script_functionality.py

# 2. 標記樹狀圖
python tools/scan/mark_nonfunctional_scripts.py

# 3. 列出需要改進的檔案
python tools/scan/list_no_functionality_files.py
```

### 2. 程式碼提取工作流
```bash
# 提取增強類別
python tools/scan/extract_enhanced.py
```

## ⚙️ 配置說明

### 路徑配置
- 所有腳本使用相對路徑，自動從專案根目錄計算
- 支援跨平台路徑處理
- 無需硬編碼絕對路徑

### 功能標記
腳本會根據以下標準進行功能性分類：
- **程式碼行數**: 實際程式碼行數
- **函數定義**: 是否包含有意義的函數
- **導入語句**: 是否有外部依賴
- **邏輯複雜度**: 控制流程的複雜程度

## 🔗 與其他模組的關係

- **Common**: 依賴基礎分析工具和報告生成
- **Core**: 提供掃描結果給核心分析引擎
- **Integration**: 掃描結果可用於外部工具整合
- **Features**: 掃描功能可增強特定功能檢測

## 📝 維護注意事項

1. **報告依賴**: 確保功能分析報告是最新的
2. **路徑一致性**: 保持相對路徑的一致性
3. **標記規範**: 遵循統一的功能標記規範
4. **檔案編碼**: 統一使用 UTF-8 編碼

---

*最後更新: 2024-10-19*
*模組版本: Scan v1.0*