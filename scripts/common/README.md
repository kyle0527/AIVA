# 🔗 Common - 通用服務腳本

> **所屬層級**: `scripts/common/`
> **上一層級**: [Scripts 根目錄](../README.md)
> **檔案數量**: 10+ 個腳本（含子目錄）

---

## 📋 目錄概述

Common 目錄是 AIVA 系統通用基礎設施的集合點。它收納了系統啟動、CLI 介面操作、驗證、維護、及環境設定工具。
這裡的腳本大多具備跨模組的特性，作為日常運維與測試的統一入口。

---

## 🗂️ 子目錄導覽

| 子目錄 | 說明 | 連結 |
|--------|------|------|
| `maintenance/` | 深入的維護與分析工具（路徑修復、報表生成、核心優化等） | [詳情](./maintenance/README.md) |
| `launcher/` | 系統級別啟動腳本（啟動/停止系統元件） | - |
| `setup/` | 環境初始化腳本 | - |
| `validation/` | 系統健康與診斷腳本 | - |
| `tools/` | 開發工具與依賴管理 | - |
| `data/` | 腳本所需的本地資料庫與配置暫存 | - |

---

## 📂 核心腳本說明

### 🚀 啟動與 CLI 介面

| 腳本 | 功能說明 |
|------|----------|
| `start_ai_simple.py` | 簡化版 AI 服務啟動器 |
| `run_aiva_cli.bat` / `.sh` | 啟動 AIVA 主要命令列介面 |
| `run_capability_cli.bat` / `.sh` | 啟動 AIVA 能力查詢與測試 CLI |

---

## 🚀 快速開始

### 啟動 CLI

```bash
# Windows
.\run_aiva_cli.bat

# Linux/Mac
./run_aiva_cli.sh
```

### 系統診斷 (Validation)

```powershell
cd validation
.\diagnose_system.ps1
```

---

## 💡 最佳實踐

- 若要執行維護任務，請進入 `maintenance/` 目錄並參考其 [README.md](./maintenance/README.md)。
- 啟動全系統或特定微服務時，可利用 `launcher/` 中的腳本，或使用本目錄下的捷徑批次檔。
