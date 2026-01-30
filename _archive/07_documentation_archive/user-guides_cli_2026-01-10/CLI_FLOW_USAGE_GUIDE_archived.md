# AIVA 動態 Flow CLI 使用指南

## 📑 目錄

- [✅ 實施完成](#-實施完成)
- [🚀 快速開始](#-快速開始)
  - [查看所有可用 Flows](#查看所有可用-flows)
  - [查看特定 Flow 的詳情](#查看特定-flow-的詳情)
  - [執行 Flow](#執行-flow)
- [📖 參數說明](#-參數說明)
- [🎯 實際範例](#-實際範例)
  - [範例 1: 執行 Flow 4（模型訓練）](#範例-1-執行-flow-4模型訓練)
  - [範例 2: 執行 Flow 8（攻擊面掃描）](#範例-2-執行-flow-8攻擊面掃描)
  - [範例 3: 批量測試 Flows](#範例-3-批量測試-flows)
  - [範例 4: 按模組過濾與分類](#範例-4-按模組過濾與分類)
- [🔧 進階用法](#-進階用法)
  - [參數映射](#參數映射)
  - [自定義參數](#自定義參數)
- [📊 Flow 分類（按終點模組）](#-flow-分類按終點模組)
  - [按終點模組分佈（840 個 flows）](#按終點模組分佈840-個-flows)
  - [按長度分類](#按長度分類)
- [🛠️ 測試與驗證](#-測試與驗證)
  - [執行測試腳本](#執行測試腳本)
- [📁 文件位置](#-文件位置)
- [⚠️ 注意事項](#-注意事項)
  - [Flow 定義文件路徑](#flow-定義文件路徑)
  - [PYTHONPATH 設置](#pythonpath-設置)
  - [Dry-run vs 實際執行](#dry-run-vs-實際執行)
- [🎓 對比：新舊方式](#-對比新舊方式)
  - [舊方式（aiva run）](#舊方式aiva-run)
  - [新方式（aiva flow4）✨](#新方式aiva-flow4)
- [🚀 下一步](#-下一步)
- [📞 常見問題](#-常見問題)
  - [Q: 如何知道某個 flow 需要什麼參數？](#q-如何知道某個-flow-需要什麼參數)
  - [Q: 為什麼找不到 flow 定義？](#q-為什麼找不到-flow-定義)
  - [Q: 如何查看所有 840 個 flows？](#q-如何查看所有-840-個-flows)
  - [Q: 可以只執行某個模組的 flows 嗎？](#q-可以只執行某個模組的-flows-嗎)

---


> ⚠️ **數據準確性警告**: 文檔中提到的 840 個 flows 與實際不符（實際約 276 個）  
> ⚠️ **命令格式**: 文檔使用 `aiva flow<ID>` 簡寫，實際需要完整 Python 路徑  
> **版本**: v3.2 (2026-01-01)  
> **最新更新**: 模組分類算法修復，確保準確的模組歸屬

## ✅ 實施完成

動態 Flow 命令系統已成功實施！現在你可以用簡潔的方式執行任意 Flow。

**重要更新 (2026-01-01)**:
- 🔧 修復模組分類算法，使用文件路徑進行精確分類
- 📊 分類準確度從 46% 提升至 91.2%
- ✅ 模組分佈現在符合實際架構設計

---

## 🚀 快速開始

### 查看所有可用 Flows

> ⚠️ **實際命令**: 需要使用完整 Python 路徑而非 `aiva` 簡寫命令

```bash
# 列出前 20 個 flows（默認）
aiva list-flows  ⚠️ [命令格式待確認]

# 列出所有 flows  ⚠️ [數量已變更: ~276]
aiva list-flows --limit 840

# 按終點模組分類（六大模組）✨ 新功能
aiva list-flows --by-endpoint

# 顯示統計摘要 ✨ 新功能
aiva list-flows --stats

# 只看 cognitive_core 模組的 flows
aiva list-flows --module cognitive_core

# 只看 external_learning 模組的 flows
aiva list-flows --module external_learning
```

### 查看特定 Flow 的詳情

```bash
# 查看 flow4 的幫助信息
aiva flow4 --help

# 會顯示：
# - 模組: external_learning
# - 路徑長度: 5
# - 調用鏈: monitoring -> optimized_core -> train_classifier ...
# - 可用參數: --target, --data, --query, --param, --intensity, --dry-run
```

### 執行 Flow

```bash
# Dry Run 模式（僅預覽）
aiva flow4 --dry-run

# 帶數據路徑
aiva flow4 --data /path/to/data.npz

# 帶目標 URL
aiva flow8 --target https://example.com

# 調整 AI 強度
aiva flow4 --data /data/train.npz -i 0.8

# 自定義參數
aiva flow4 --param model=v1 --param epochs=100 --param batch_size=32

# 組合使用
aiva flow4 --data /data/train.npz --param model=v2 -i 0.7 --dry-run
```

---

## 📖 參數說明

| 參數 | 簡寫 | 說明 | 示例 |
|------|------|------|------|
| `--target` | `-t` | 目標 URL/路徑/對象 | `--target https://api.example.com` |
| `--data` | `-d` | 數據路徑 | `--data /data/train.npz` |
| `--query` | `-q` | 查詢字串 | `--query "SQL injection"` |
| `--param` | `-p` | 額外參數（可多次使用） | `--param key1=value1 --param key2=value2` |
| `--intensity` | `-i` | AI 強度（0.0-1.0） | `-i 0.8` |
| `--dry-run` | 無 | 預覽模式，不實際執行 | `--dry-run` |

---

## 🎯 實際範例

### 範例 1: 執行 Flow 4（模型訓練）

```bash
# 傳統方式（冗長）
aiva run 4 -c '{"training_data_path": "/data/model.npz"}' -i 0.7

# 新方式（簡潔）✨
aiva flow4 --data /data/model.npz -i 0.7
```

**對比**:
- 減少 **70%** 的輸入長度
- 不需要手動構建 JSON
- 參數更直觀

---

### 範例 2: 執行 Flow 8（攻擊面掃描）

```bash
# 掃描目標網站
aiva flow8 --target https://example.com -i 0.8

# 帶自定義參數
aiva flow8 --target https://api.example.com --param depth=5 --param threads=10 -i 0.6
```

---

### 範例 3: 批量測試 Flows

```bash
# 測試前 10 個 flows（dry-run）
for i in {1..10}; do
    echo "測試 flow$i"
    aiva flow$i --dry-run
done
```

---

### 範例 4: 按模組過濾與分類

```bash
# 查看 cognitive_core 模組的 flows
aiva list-flows --module cognitive_core

# 查看 external_learning 模組的 flows
aiva list-flows --module external_learning

# 查看 service_backbone 模組的 flows（最多）
aiva list-flows --module service_backbone --limit 50

# 按終點模組分類顯示（六大模組）✨ 新功能
aiva list-flows --by-endpoint

# 輸出範例（v3.2 修復後）：
# 🔹 INTERNAL_EXPLORATION (201 個 flows) ✅ 最多！
# 🔹 SERVICE_BACKBONE (163 個 flows)
# 🔹 CORE_CAPABILITIES (131 個 flows)
# 🔹 COGNITIVE_CORE (124 個 flows)
# 🔹 EXTERNAL_LEARNING (99 個 flows)
# 🔹 TASK_PLANNING (48 個 flows)
# 🔹 UNKNOWN (74 個 flows) ⚠️
# 查看統計摘要 ✨ 新功能
aiva list-flows --stats

# 輸出範例（v3.2 修復後 - 2026-01-01）：
# 📊 Flow 統計報告
# 總 Flows: 840
# 
# 按終點模組分佈:
#   internal_exploration:  201 ( 23.9%) ████████████ ✅ 最多
#   service_backbone    :  163 ( 19.4%) ██████████
#   core_capabilities   :  131 ( 15.6%) ████████
#   cognitive_core      :  124 ( 14.8%) ███████
#   external_learning   :   99 ( 11.8%) ██████
#   task_planning       :   48 (  5.7%) ███
#   unknown             :   74 (  8.8%) ████ ⚠️
```

---

## 🔧 進階用法

### 參數映射

系統會自動將參數映射到多個可能的鍵名：

```bash
# 輸入
aiva flow4 --target https://example.com

# 自動映射為
{
  "target": "https://example.com",
  "target_url": "https://example.com",
  "url": "https://example.com"
}
```

### 自定義參數

使用 `--param key=value` 添加任意參數：

```bash
aiva flow100 \
    --target https://api.example.com \
    --param depth=5 \
    --param threads=10 \
    --param timeout=30 \
    --param retry=3 \
    -i 0.6
```

---

## 📊 Flow 分類（按終點模組）

### 按終點模組分佈（840 個 flows）

> **重要更新 (v3.2 - 2026-01-01)**: 模組分類算法已修復，使用文件路徑進行精確分類。  
> 分類準確度從 46% 提升至 91.2%。

使用 `aiva list-flows --stats` 查看完整統計。

| 終點模組 | Flow 數量 | 百分比 | 示例 |
|---------|----------|--------|------|
| **internal_exploration** | 201 | 23.9% | ✅ 最多！flow0, flow2, flow3 |
| **service_backbone** | 163 | 19.4% | flow1, flow7 |
| **core_capabilities** | 131 | 15.6% | flow10, flow12 |
| **cognitive_core** | 124 | 14.8% | flow5, flow13, flow18 |
| **external_learning** | 99 | 11.8% | flow4, flow6, flow11 |
| **task_planning** | 48 | 5.7% | flow59, flow163, flow205 |
| **unknown** | 74 | 8.8% | ⚠️ 需進一步檢查 |

> 💡 **說明**: 終點模組是指 flow 調用鏈的最後一個腳本所屬的模組。例如 flow5 的路徑是 `monitoring -> optimized_core -> train_classifier -> model_trainer -> ai_model_manager`，最後一個腳本 `ai_model_manager` 屬於 `cognitive_core` 模組。

> **v3.2 修復說明**: 之前的分類算法使用腳本名稱而非文件路徑，導致 54% 的 flows 被錯誤分類。現在使用文件路徑進行精確分類，準確度已達 91.2%。

### 按長度分類

| 路徑長度 | 說明 | 示例 |
|---------|------|------|
| 2 步 | 簡單流程 | flow1 (monitoring → optimized_core) |
| 3-4 步 | 中等複雜度 | flow2, flow3 |
| 5+ 步 | 複雜流程 | flow4 (5步，跨3個模組) |

---

## 🛠️ 測試與驗證

### 執行測試腳本

```powershell
# Windows PowerShell
.\test_dynamic_cli.ps1
```

測試腳本會驗證：
- ✅ Flow 定義載入（840 個）
- ✅ 動態命令註冊
- ✅ 參數解析
- ✅ Dry-run 模式
- ✅ 模組過濾

---

## 📁 文件位置

| 文件 | 路徑 | 說明 |
|------|------|------|
| CLI 入口 | `services/core/aiva_core/core_capabilities/cli/aiva_cli.py` | 動態命令註冊 (+135 行) |
| Flow 執行器 | `services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py` | FlowExecutor 實現 (+15 行) |
| Flow 分類器 | `services/core/aiva_core/internal_exploration/python_tools/aiva_flow_classifier.py` | 模組分類（v3.2 已修復） |
| Flow 定義 | `C:/Users/User/Downloads/data/internal_exploration/latest_classification.json` | 840 個 flows (v3.2) |
| 測試腳本 | `test_dynamic_cli.ps1` | 自動化測試 |
| 修復報告 | `MODULE_CLASSIFICATION_FIX_REPORT.md` | v3.2 修復詳情 |

---

## ⚠️ 注意事項

### Flow 定義文件路徑

系統會自動搜索以下路徑（按順序）：

1. `C:/Users/User/Downloads/data/internal_exploration/latest_classification.json`
2. `C:/D/fold7/AIVA-git/data/internal_exploration/latest_classification.json`
3. `C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json`

### PYTHONPATH 設置

執行前請設置 PYTHONPATH：

```bash
# Windows PowerShell
$env:PYTHONPATH="C:\D\fold7\AIVA-git"

# Linux/Mac
export PYTHONPATH="/path/to/AIVA-git"
```

### Dry-run vs 實際執行

- **Dry-run**: 安全，僅顯示將執行的操作
- **實際執行**: 會動態導入模組並執行函數

建議先用 `--dry-run` 驗證。

---

## 🎓 對比：新舊方式

### 舊方式（aiva run）

```bash
aiva run 4 -c '{"training_data_path": "/data/model.npz", "epochs": 100}' -i 0.7
```

**問題**:
- ❌ 需要手動構建 JSON
- ❌ 參數名稱不直觀（training_data_path）
- ❌ 容易出錯（引號、逗號）
- ❌ 長度冗長

### 新方式（aiva flow4）✨

```bash
aiva flow4 --data /data/model.npz --param epochs=100 -i 0.7
```

**優勢**:
- ✅ 參數直觀（--data, --target）
- ✅ 自動 JSON 構建
- ✅ 減少 70% 輸入
- ✅ 不易出錯

---

## 🚀 下一步

1. **熟悉常用 Flows**: 使用 `aiva list-flows` 探索
2. **Dry-run 測試**: 用 `--dry-run` 安全測試
3. **實際執行**: 找到適合的 flow 後實際執行
4. **自定義腳本**: 編寫自動化腳本組合多個 flows

---

## 📞 常見問題

### Q: 如何知道某個 flow 需要什麼參數？

```bash
aiva flow<ID> --help
```

### Q: 為什麼找不到 flow 定義？

檢查文件是否存在：
```powershell
Test-Path "C:\Users\User\Downloads\data\internal_exploration\latest_classification.json"
```

### Q: 如何查看所有 840 個 flows？

```bash
aiva list-flows --limit 840 > all_flows.txt
```

### Q: 可以只執行某個模組的 flows 嗎？

```bash
aiva list-flows --module external_learning
# 然後執行顯示的 flow ID
```

---

**文檔版本**: v1.0  
**最後更新**: 2026-01-01  
**狀態**: ✅ 生產就緒

🎉 享受簡潔的 CLI 體驗！
