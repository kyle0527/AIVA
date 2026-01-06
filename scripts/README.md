# 📜 AIVA Scripts - 腳本工具集

> **版本**: v8.0  
> **更新日期**: 2026年1月6日  
> **狀態**: 目錄結構整合完成

---

## 📑 目錄索引

| 目錄 | 說明 | 檔案數 |
|------|------|--------|
| [🤖 **core/**](./core/README.md) | AI 核心分析腳本 | 14 個 |
| [🔗 **common/**](./common/README.md) | 通用服務腳本 | 11 個 |
| [🛠️ **utilities/**](./utilities/README.md) | 實用工具腳本 | 14 個 |
| [🗃️ **_archive/**](./_archive/README.md) | 歷史腳本存檔 | 29 個 |

---

## 📋 概述

AIVA Scripts 是 AIVA 系統的腳本工具集合，經過 v8.0 版本的目錄結構整合，將原本分散的 16 個目錄精簡為 4 個核心目錄。

---

## 🗂️ 子模組說明

### 🤖 [Core - AI 核心分析](./core/README.md)

核心分析工具集，包含：
- **能力分析**: 分析系統能力定義與覆蓋率
- **模組連接**: 分析模組間的連接關係
- **流程追蹤**: 追蹤與分析執行流程

**主要腳本**:
- `run_capability_analysis.py` - 執行完整能力分析
- `analyze_module_connectivity.py` - 模組連接性分析
- `find_executable_flows.py` - 搜尋可執行流程

---

### 🔗 [Common - 通用服務](./common/README.md)

通用基礎設施工具，包含：
- **啟動腳本**: 系統與服務啟動器
- **CLI 介面**: 命令列操作工具
- **驗證工具**: 系統驗證檢查

**主要腳本**:
- `start_ai_service.py` - 啟動 AI 服務
- `aiva_cli.py` - AIVA 命令列介面
- `validate_scan_system.py` - 驗證掃描系統

---

### 🛠️ [Utilities - 實用工具](./utilities/README.md)

輔助工具集，包含：
- **驗證檢查**: 資料與系統驗證
- **生成工具**: 文檔與計畫生成
- **資料庫工具**: 能力資料庫管理

**主要腳本**:
- `health_check.py` - 系統健康檢查
- `generate_ai_capability_reference.py` - 生成能力參考
- `create_capability_db.py` - 建立能力資料庫

---

### 🗃️ [_archive - 歷史存檔](./_archive/README.md)

已歸檔的歷史腳本，保留作為參考用途。

---

## 📊 統計資訊

| 項目 | 數量 |
|------|------|
| 總目錄數 | 4 |
| 總檔案數 | 68 |
| Core 腳本 | 14 |
| Common 腳本 | 11 |
| Utilities 腳本 | 14 |
| Archive 腳本 | 29 |

---

## 🚀 快速開始

### 啟動 AI 服務
```bash
cd scripts/common
python start_ai_service.py
```

### 執行能力分析
```bash
cd scripts/core
python run_capability_analysis.py
```

### 系統健康檢查
```bash
cd scripts/utilities
python health_check.py
```

---

## 📝 更新日誌

### v8.0 (2026-01-06)
- 🔄 目錄結構整合：16 → 4 個目錄
- 📁 根目錄檔案歸類至對應子目錄
- 📝 建立各子模組 README 文檔
- 🗑️ 移除 12 個空白/冗餘資料夾

### v7.0 (2025-12-06)
- 🗑️ 移除 83 個過時腳本
- 🧹 清理 19 個空白子資料夾

---

## 📚 相關文檔

- [FIX_MODULE_IMPORT_GUIDE.md](./FIX_MODULE_IMPORT_GUIDE.md) - 模組導入修復指南

---

**📅 最後更新**: 2026年1月6日  
**🎯 版本**: v8.0
