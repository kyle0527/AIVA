# AIVA CLI 使用指南

> **更新日期**: 2026年1月12日  
> **狀態**: ✅ 生產就緒

---

## 🚀 快速開始

### 入口腳本

| 腳本 | 說明 | 使用方式 |
|------|------|---------|
| `啟動能力選單.bat` | 互動式能力選單 | 雙擊執行 |
| `執行Flow.bat` | 執行指定 Flow | `執行Flow.bat [ID]` |
| `預覽Flow.bat` | 預覽執行計畫 | `預覽Flow.bat [ID]` |
| `啟動AIVA系統.bat` | 啟動完整系統 | 雙擊執行 |

### 命令列用法

```bash
# 進入 CLI 工具目錄
cd services\core\aiva_core\internal_exploration\python_tools

# 列出可用能力
python aiva_cli_implementation.py --list

# 搜尋能力
python aiva_cli_implementation.py --search xss

# 執行指定 Flow
python aiva_cli_implementation.py --flow 11

# 預覽執行（不實際執行）
python aiva_cli_implementation.py --flow 11 --dry-run

# 互動式選單
python aiva_cli_implementation.py --menu
```

---

## 📋 命令參考

### --list：列出可用能力

```bash
python aiva_cli_implementation.py --list
```

顯示所有可用的 Flow，包含 ID、描述、模組等資訊。

### --search：搜尋能力

```bash
python aiva_cli_implementation.py --search <關鍵字>

# 範例
python aiva_cli_implementation.py --search xss
python aiva_cli_implementation.py --search sql
python aiva_cli_implementation.py --search scan
```

使用關鍵字搜尋相關能力。

### --flow：執行 Flow

```bash
python aiva_cli_implementation.py --flow <ID>

# 範例
python aiva_cli_implementation.py --flow 11
python aiva_cli_implementation.py --flow 42
```

執行指定 ID 的 Flow。

### --dry-run：預覽模式

```bash
python aiva_cli_implementation.py --flow <ID> --dry-run

# 範例
python aiva_cli_implementation.py --flow 11 --dry-run
```

顯示執行計畫但不實際執行，用於確認流程。

### --menu：互動式選單

```bash
python aiva_cli_implementation.py --menu
```

啟動互動式選單，可瀏覽和選擇能力。

---

## 🔧 環境設定

### PYTHONPATH

bat 檔案已自動設定，手動執行時需設定：

```bash
set PYTHONPATH=C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services
```

### 工作目錄

CLI 腳本位於：
```
services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py
```

---

## 📊 能力表

CLI 讀取的能力定義來自：
```
services\integration\data\internal_exploration\latest_classification.json
```

能力表會動態更新，數量和內容會隨開發進度變動。

---

## 🎯 使用場景

### 場景 1：探索可用能力

```bash
# 查看所有能力
python aiva_cli_implementation.py --list

# 搜尋 XSS 相關
python aiva_cli_implementation.py --search xss
```

### 場景 2：執行掃描

```bash
# 先預覽
python aiva_cli_implementation.py --flow 11 --dry-run

# 確認後執行
python aiva_cli_implementation.py --flow 11
```

### 場景 3：互動式操作

```bash
# 啟動選單
python aiva_cli_implementation.py --menu
```

---

## 📚 相關文檔

| 文檔 | 說明 |
|------|------|
| [CLI_ARCHITECTURE_OVERVIEW.md](../../../../guides/architecture/CLI_ARCHITECTURE_OVERVIEW.md) | CLI 架構總覽 |
| [雙CLI架構設計指南.md](../../../../guides/architecture/雙CLI架構設計指南.md) | 詳細設計理念 |
| [DUAL_LOOP_DESIGN_GUIDE.md](../../../../guides/DUAL_LOOP_DESIGN_GUIDE.md) | 雙閉環設計 |
