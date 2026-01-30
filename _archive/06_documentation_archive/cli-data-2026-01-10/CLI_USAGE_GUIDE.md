# AIVA CLI 使用手冊

> **版本**: v1.0  
> **建立日期**: 2026-01-10  
> **對應數據版本**: v6 (276 flows)  
> **狀態**: 驗證中

## 📋 概述

AIVA CLI 工具套件提供三種方式執行 AI 能力：

| 執行方式 | 適用場景 | 說明 |
|---------|---------|------|
| **直接執行腳本** | 本地開發、測試 | 從 python_tools 目錄執行 |
| **模組方式執行** | 整合使用 | 需正確設定 PYTHONPATH |
| **互動選單** | 探索、學習 | 圖形化選單瀏覽能力 |

---

## 🚀 快速開始

### 方式 1: 直接執行（推薦）

```powershell
# 切換到工具目錄
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 查看幫助
python aiva_cli_implementation.py --help

# 列出所有流程
python aiva_cli_implementation.py --list

# Dry Run 預覽（不實際執行）
python aiva_cli_implementation.py --flow 1 --dry-run

# 實際執行流程
python aiva_cli_implementation.py --flow 1

# 啟動互動選單
python aiva_cli_implementation.py --menu
```

### 方式 2: 模組方式執行

```powershell
# 必須先設定 PYTHONPATH
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH = "C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services;C:\D\fold7\AIVA-git"

# 然後執行
python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 1 --dry-run
```

---

## 📊 可用工具

### 1. aiva_cli_implementation.py - 流程執行器

**功能**: 執行特定流程、生成 CLI 文檔

```powershell
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 列出流程
python aiva_cli_implementation.py --list

# 預覽流程（Dry Run）
python aiva_cli_implementation.py --flow <ID> --dry-run

# 執行流程
python aiva_cli_implementation.py --flow <ID>

# 生成文檔
python aiva_cli_implementation.py --generate-doc md    # Markdown
python aiva_cli_implementation.py --generate-doc json  # JSON

# 互動選單
python aiva_cli_implementation.py --menu
python aiva_cli_implementation.py  # 無參數也會啟動選單
```

### 2. aiva_capability_cli.py - 能力查詢器

**功能**: 搜尋、篩選、查看能力詳情

```powershell
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 列出所有能力
python aiva_capability_cli.py --list

# 搜尋能力
python aiva_capability_cli.py --search "vector"
python aiva_capability_cli.py --search "xss" --module core_capabilities

# 查看詳情
python aiva_capability_cli.py --info 13

# 執行流程
python aiva_capability_cli.py --flow 13
```

### 3. aiva_exploration_pipeline.py - 管線執行器

**功能**: 完整分析流程（Analyzer → Classifier → Diff → Docs）

```powershell
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 分析完整 core
python aiva_exploration_pipeline.py --target core --module core

# 分析特定模組
python aiva_exploration_pipeline.py --target cognitive_core --module core

# 自定義深度
python aiva_exploration_pipeline.py --target core --module core --depth 15
```

---

## 📁 數據文件位置

```
services/integration/data/internal_exploration/
├── latest_classification.json          # 最新分類數據（唯一數據源）
├── CLI_USAGE_GUIDE.md                   # 本使用手冊
└── analysis_history/
    └── v6/                              # 當前版本
        ├── classification_data.json     # 分類數據 (643 KB)
        ├── CLI_COMMANDS_REFERENCE.md    # CLI 指令手冊 (41 KB)
        ├── cli_commands_db.json         # 命令資料庫 (210 KB)
        └── ...
```

---

## 🎯 常用流程示例

### 按模組分類的代表流程

| 模組 | Flow ID | 路徑 | 說明 |
|------|---------|------|------|
| cognitive_core | 2 | ai_capability_query -> knowledge_base | AI 能力查詢 |
| cognitive_core | 13 | vector_store -> capability_encoder | 向量存儲 |
| service_backbone | 1 | monitoring -> monitoring | 監控服務 |
| service_backbone | 15 | dispatcher -> message_broker | 消息分發 |
| task_planning | 14 | scenario_manager -> unified_executor | 場景執行 |
| core_capabilities | 18 | task_executor -> capability_registry | 能力註冊 |
| internal_exploration | 6 | core_analyzer -> core_analyzer | 核心分析 |
| learning_system | 20 | rl_trainers -> rl_models | 強化學習 |

---

## ⚠️ 已知問題

### 1. Windows 編碼問題

**現象**: 執行時出現 `UnicodeEncodeError: 'cp950' codec can't encode character`

**原因**: Windows 控制台默認編碼無法顯示 emoji

**解決方案**: 
```powershell
# 設定 UTF-8 編碼
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
```

### 2. 模組找不到錯誤

**現象**: `ModuleNotFoundError: No module named 'aiva_core'` 或 `'aiva_common'`

**原因**: PYTHONPATH 未正確設定

**解決方案**:
```powershell
# 方式 A: 直接執行（推薦）
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_cli_implementation.py --flow 1 --dry-run

# 方式 B: 設定 PYTHONPATH
$env:PYTHONPATH = "C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services;C:\D\fold7\AIVA-git"
```

---

## 📈 驗證清單

- [x] `python aiva_cli_implementation.py --list` 可執行 ✅
- [x] `python aiva_cli_implementation.py --flow 1 --dry-run` 可預覽 ✅
- [ ] `python aiva_cli_implementation.py --menu` 互動選單正常 (待測)
- [x] `python aiva_capability_cli.py --list` 可執行 ✅
- [x] `python aiva_exploration_pipeline.py --help` 可執行 ✅

### 執行前置要求

```powershell
# 必須設定環境變數解決編碼問題
$env:PYTHONIOENCODING = "utf-8"

# 或執行 chcp 65001 切換控制台編碼
chcp 65001
```

---

## 📚 相關文件

- [Python 工具完整手冊](../../core/aiva_core/internal_exploration/python_tools/README.md)
- [CLI_COMMANDS_REFERENCE.md](analysis_history/v6/CLI_COMMANDS_REFERENCE.md) - 276 條 CLI 指令
- [cli_commands_db.json](analysis_history/v6/cli_commands_db.json) - AI 可檢索的命令資料庫
