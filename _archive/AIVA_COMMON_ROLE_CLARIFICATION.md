# aiva_common 角色澄清報告

## 用戶的正確洞察

> **核心觀點**: "aiva_common 的功用是在執行前就要搞定，而非等到執行時再進行確認"

這個理解**完全正確**！

---

## 系統分層架構（正確版本）

```
┌────────────────────────────────────────────────────────────┐
│  入口層 (Entry Point)                                       │
│  - rich_cli.py / api/main.py / start_ai_service.py         │
│  職責: 接收用戶輸入，展示結果                                │
└────────────────────────────────────────────────────────────┘
                          ↓ (使用 aiva_common.schemas)
┌────────────────────────────────────────────────────────────┐
│  調度層 (Orchestration)                                     │
│  - command_center.py (在 aiva_common 中)                    │
│  職責: 路由命令到對應模組                                    │
└────────────────────────────────────────────────────────────┘
                          ↓ (使用 AICommand/AICommandResult)
┌────────────────────────────────────────────────────────────┐
│  決策層 (Decision Making)                                   │
│  - ai_commander.py                                          │
│  職責: AI 分析和決策                                         │
└────────────────────────────────────────────────────────────┘
                          ↓ (使用 aiva_common.schemas)
┌────────────────────────────────────────────────────────────┐
│  執行層 (Execution)                                         │
│  - services/scan/* (掃描能力)                               │
│  - services/features/* (攻擊能力)                           │
│  - services/integration/* (整合能力)                        │
│  職責: 實際執行業務邏輯 ← **這才是業務能力！**                │
└────────────────────────────────────────────────────────────┘
                          ↓ (依賴)
┌────────────────────────────────────────────────────────────┐
│  基礎設施層 (Infrastructure) ← **aiva_common 在這裡！**     │
│  - services/aiva_common/*                                   │
│  職責: 提供數據合約、配置、工具、錯誤處理                      │
│  時機: 導入時/初始化時/全程可用                               │
└────────────────────────────────────────────────────────────┘
```

---

## aiva_common 的真正角色

### ✅ aiva_common 是什麼

**基礎設施層** - 就像 Python 標準庫一樣：

| 模組 | 作用時機 | 角色 |
|------|---------|------|
| `command_center.py` | 系統啟動時初始化 | 命令路由器 |
| `schemas/` | 導入時（編譯時） | 數據合約定義 |
| `config/` | 初始化時 | 加載配置 |
| `utils/` | 全程可用 | 工具函數庫 |
| `error_handling.py` | 全程可用 | 異常處理 |
| `monitoring.py` | 後台運行 | 監控收集 |

### ❌ aiva_common 不是什麼

- ❌ 不是業務能力（不執行掃描、攻擊、分析）
- ❌ 不是 CLI 命令（只提供 CLI 數據模型）
- ❌ 不是執行時動態調用的功能
- ❌ 不需要被 internal_exploration 掃描

---

## CLI 能力的正確定義

### 錯誤理解（之前）

```python
# ❌ 錯誤: 認為 CLI 能力 = 文件名包含 "cli" 的函數
has_cli = "cli" in file_path  # 導致誤判
```

這會把以下內容誤認為 CLI 能力：
- `rich_cli.py` 的內部方法（這是 CLI **實現**，不是 CLI **能力**）
- `client.py` 的方法（因為 "client" 包含 "cli"）

### 正確理解（現在）

**CLI 能力 = 用戶可以通過 CLI 調用的業務功能**

例如：
```bash
# 用戶在 rich_cli.py 中選擇 "1. 漏洞掃描"
# 這會調用 scan 模組的掃描能力
# 所以 "掃描能力" 是 CLI 能力，而不是 rich_cli.py 本身
```

判斷標準：
```python
# ✅ 正確的判斷邏輯
是 CLI 能力 = 該功能可以通過 CLI 入口調用
            AND 該功能執行業務邏輯
            AND 該功能不是 CLI 系統本身的實現
```

具體來說：
- ✅ `scan/engines/python_engine.py` 的掃描函數 - 可以通過 CLI 調用
- ✅ `features/function_sqli/scanner.py` 的注入函數 - 可以通過 CLI 調用
- ❌ `rich_cli.py` 的 `show_main_menu()` - 這是 CLI 實現，不是業務能力
- ❌ `command_center.py` 的 `execute()` - 這是調度器，不是業務能力

---

## internal_exploration 掃描範圍的正確性

### 原始掃描範圍（2025-11-25）

```
✅ services/scan          - 286 個能力 (掃描引擎)
✅ services/core/aiva_core - 216 個能力 (認知核心)
✅ services/integration   - 113 個能力 (整合層)
✅ services/features      - 102 個能力 (攻擊功能)
❌ services/aiva_common   - 0 個能力 (基礎設施)
❌ services/core/ui       - 0 個能力 (用戶界面)
```

### 這個範圍是正確的！

**為什麼不掃描 aiva_common？**

1. **它不提供業務能力** - 只提供數據結構和工具
2. **它是依賴項，不是功能** - 就像不掃描 `pydantic` 或 `fastapi`
3. **它在導入時就完成初始化** - 不是動態調用的能力

**為什麼不掃描 services/core/ui？**

1. **UI 層只是展示** - 不執行業務邏輯
2. **CLI 本身不是能力** - 它只是調用其他能力的入口
3. **用戶關心的是功能，不是界面** - "我能做什麼"，不是"界面長什麼樣"

---

## 那麼 CLI 能力應該如何識別？

### 方法 1: 檢查 command_center 的路由配置

```python
# 查看哪些模組註冊到了 command_center
command_center.register_module("scan", ScanCommandHandler())
command_center.register_module("features", FeaturesCommandHandler())

# 這些模組的能力 = CLI 可調用的能力
```

### 方法 2: 檢查 rich_cli.py 的選單選項

```python
# rich_cli.py 中的選單
menu_options = [
    ("1", "漏洞掃描", "啟動 AI 驅動的安全評估"),    # → scan 模組
    ("2", "能力管理", "管理註冊的安全工具和能力"),   # → integration 模組
    ("3", "AI 對話", "與 AIVA AI 引擎互動"),       # → core 模組
    ...
]

# 選單指向的模組能力 = CLI 能力
```

### 方法 3: 檢查業務能力的調用鏈

```python
# 如果一個能力的調用鏈是:
rich_cli.py → command_center → module_handler → capability
                                                    ↑
                                            這才是 CLI 能力
```

---

## 結論

### 用戶的理解完全正確

1. ✅ **aiva_common 是基礎設施，不是業務能力**
2. ✅ **它的功用在執行前（導入時/初始化時）就完成**
3. ✅ **CLI 指令執行時不需要特別掃描它**
4. ✅ **原始掃描不包含 aiva_common 是正確的設計**

### 當前狀態

1. ✅ 原始掃描數據（802 個能力）- **正確**
2. ✅ 不包含 aiva_common - **正確**
3. ✅ CLI 檢測結果為 0 - **正確**（因為我們修復了誤判邏輯）
4. ⚠️ 如果要識別「CLI 可調用的能力」，需要不同的方法

### 下一步建議

**不是** 修改掃描範圍包含 aiva_common  
**而是** 如果需要標記「CLI 可調用」，應該：

```python
# 方案 A: 基於模組標記
if capability.module in ["scan", "features"]:
    capability.available_via = ["CLI", "API"]

# 方案 B: 基於 command_center 路由
registered_modules = command_center.get_registered_modules()
if capability.module in registered_modules:
    capability.available_via = ["CLI", "API"]
```

---

**生成時間**: 2025-12-15 07:30:00  
**狀態**: ✅ 架構理解已修正
