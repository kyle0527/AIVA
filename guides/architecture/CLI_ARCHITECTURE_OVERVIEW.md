# AIVA 雙 CLI 架構總覽

> **文檔狀態**: 🔑 **核心設計規範**  
> **更新日期**: 2026年1月12日  
> **設計理念**: 內部靈活、外部標準、AI 主導

---

## 🎯 架構設計原則

AIVA 採用**雙層 CLI 架構**，清晰分離 AI 內部通訊與外部模組調用：

```
┌─────────────────────────────────────────────────────────────┐
│                    AI 核心系統（內部）                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  cognitive_core │ task_planning │ internal_exploration │  │
│  │                 service_backbone                      │  │
│  └─────────────────────────────────────────────────────┘    │
│                           ↕                                  │
│                      內部 CLI                                │
│              (通訊方式自由：函數/CLI/MQ)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
                       外部 CLI
               (必須 subprocess + JSON)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     外部功能模組                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │     Features     │    │       Scan       │              │
│  │  (XSS/SQLi/...)  │    │   (多語言引擎)    │              │
│  └──────────────────┘    └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 雙層分離設計

### 內部 CLI（AI 核心模組）

**範圍**：`services/core/aiva_core/` 下的四大模組
- `cognitive_core` - 認知核心、決策引擎、RAG
- `task_planning` - 任務規劃、策略生成
- `internal_exploration` - 內部探索、能力發現
- `service_backbone` - 服務骨幹、API 入口

**特點**：
| 項目 | 說明 |
|------|------|
| 通訊方式 | ✅ 自由（函數調用、CLI、消息隊列皆可） |
| 耦合度 | ✅ 可緊密耦合 |
| 決策權 | ✅ AI 自己決定最佳方式 |
| 中間層 | ✅ 可有可無 |

**調用示例**：
```python
# 方式 1：直接函數調用（推薦）
from aiva_core.cognitive_core import CapabilityOrchestrator
orchestrator = CapabilityOrchestrator()
result = await orchestrator.plan(target_url)

# 方式 2：內部 CLI（如需隔離）
cmd = ["python", "-m", "aiva_core.internal_exploration", "--action", "discover"]
result = await subprocess_run(cmd)
```

---

### 外部 CLI（功能與掃描模組）

**範圍**：
- `services/features/` - 漏洞檢測功能
- `services/scan/` - 多語言掃描引擎

**強制要求**（唯一規範）：
| 規則 | 說明 |
|------|------|
| 調用方式 | **必須** subprocess |
| 輸出格式 | **必須** JSON 到 stdout |
| 錯誤輸出 | **必須** 到 stderr |
| 退出碼 | **必須** 0=成功, 非0=失敗 |
| 中間層 | **禁止** Dispatcher/Coordinator |

**調用示例**：
```python
# AI 直接調用外部模組
cmd = ["python", "-m", "function_xss", "--url", target, "--type", "reflected"]
result = await subprocess_run(cmd)
data = json.loads(result.stdout)  # ← 直接解析，無中間層
```

---

## 🔄 執行入口

### 使用者入口（bat 檔案）

| 檔案 | 功能 | 調用目標 |
|------|------|---------|
| `啟動AIVA系統.bat` | 啟動完整系統 | app.py (uvicorn) |
| `啟動能力選單.bat` | 互動式能力選單 | aiva_cli_implementation.py --menu |
| `執行Flow.bat` | 執行指定流程 | aiva_cli_implementation.py --flow [ID] |
| `預覽Flow.bat` | 預覽執行計畫 | aiva_cli_implementation.py --dry-run |

### CLI 實現

**核心腳本**: `services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py`

**功能**：
- 讀取能力表 JSON（動態，數量隨時變動）
- 關鍵字搜尋能力
- 動態模組導入與執行
- Pipeline 數據傳遞

```bash
# 列出可用能力
python aiva_cli_implementation.py --list

# 搜尋能力
python aiva_cli_implementation.py --search xss

# 執行指定 Flow
python aiva_cli_implementation.py --flow 11

# 預覽執行（不實際執行）
python aiva_cli_implementation.py --flow 11 --dry-run
```

---

## 🌐 外部模組結構

### Features 模組（漏洞檢測）

```
services/features/
├── features_ready/          # 已就緒的功能
│   ├── function_xss/        # XSS 檢測
│   ├── function_sqli/       # SQL 注入
│   ├── function_ssrf/       # SSRF 檢測
│   ├── function_idor/       # IDOR 檢測
│   └── function_bizlogic/   # 業務邏輯漏洞
├── features_in_development/ # 開發中功能
└── features_manual_operation/ # 需手動操作
```

**調用規範**：
```bash
python -m function_xss --url https://target.com --type reflected
# stdout: {"target": "...", "vulnerable": true, "findings": [...]}
```

### Scan 模組（多語言掃描引擎）

```
services/scan/
├── rust_engine/       # Rust 快速偵察 (Phase 0)
├── python_engine/     # Python 深度分析 (Phase 1)
├── typescript_engine/ # TypeScript 動態渲染 (Phase 1)
├── go_engine/         # Go 專項測試 (Phase 2)
└── coordinators/      # 多引擎協調
```

**調用規範**：
```bash
# Rust
cargo run --manifest-path services/scan/rust_engine/Cargo.toml -- --target URL

# Python
python -m services.scan.python_engine --target URL

# Go
go run services/scan/go_engine/cmd/scanner/main.go --target URL
```

---

## 🧠 AI 決策流程

```
1. 使用者提供目標 URL
         ↓
2. AI (capability_orchestrator) 分析目標
         ↓
3. AI 查詢能力表 (JSON 關鍵字搜尋)
         ↓
4. AI 選擇最佳工具組合
         ↓
5. AI 透過 subprocess 調用外部模組
         ↓
6. AI 解析 JSON 結果
         ↓
7. AI 決定是否繼續或調整策略
```

**關鍵**：AI 直接調用、直接解析，無需中間層！

---

## 📊 設計對比

| 項目 | 內部 CLI | 外部 CLI |
|------|---------|---------|
| **範圍** | AI 核心模組 | 功能/掃描模組 |
| **調用方式** | 靈活（函數/CLI/MQ） | **必須 subprocess** |
| **輸出格式** | 可任意 | **必須 JSON** |
| **中間層** | 可有可無 | **絕對禁止** |
| **耦合度** | 可緊密 | **必須鬆散** |
| **語言** | 主要 Python | Python/Rust/Go/TS |

---

## ✅ 設計哲學

### KISS 原則
- AI 能做的，不委託給其他組件
- subprocess 能做的，不用複雜消息隊列
- JSON 能解決的，不設計複雜協議

### YAGNI 原則
- Dispatcher？實際驗證：不需要
- Coordinator 中間層？實際驗證：不需要
- 複雜數據模型？實際驗證：不需要

### 成功標準
- ✅ 能執行：`python -m module --args`
- ✅ 能輸出：JSON 到 stdout
- ✅ 能解析：AI 能 `json.loads()`
- ✅ 能工作：完成功能目標

---

## 📚 相關文檔

| 文檔 | 說明 |
|------|------|
| [DUAL_LOOP_DESIGN_GUIDE.md](../DUAL_LOOP_DESIGN_GUIDE.md) | 雙閉環設計指南 |
| [雙CLI架構設計指南.md](./雙CLI架構設計指南.md) | 詳細設計理念 |
| [DUAL_LOOP_OPERATION_GUIDE.md](../DUAL_LOOP_OPERATION_GUIDE.md) | 操作指南 |
