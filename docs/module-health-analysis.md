# AIVA 功能模組健康度分析報告

**分析日期**: 2026-03-16
**專案版本**: v7.1.0 (Core v4.4.1)
**分析範圍**: `services/` 下所有子模組

---

## 總覽

| 分類 | 模組數 | 狀態 |
|------|--------|------|
| COMPLETE (功能完整) | 11 | 可正常匯入與執行 |
| PARTIAL (部分完成) | 2 | 缺外部依賴或 Handler 未實現 |
| STUB (佔位) | 1 | 僅有 Rust CLI，無 Python 實現 |
| ~~BROKEN~~ ✅ 已修復 | 0 | 原 3 個模組已全部修復 (2026-03-17) |

---

## 一、services/features/ — 17 個功能模組

### COMPLETE — 11 個模組

| 模組 | 用途 | 檔案數 | 大小 | 備註 |
|------|------|--------|------|------|
| `function_bizlogic` | 商業邏輯漏洞 | 9 | 72K | CLI + Manager 完整 |
| `function_forensic` | 數位鑑識 | 4 | 26K | Manager pattern |
| `function_idor` | IDOR 偵測 | 8 | 57K | Smart detection 整合 |
| `function_info_leak` | 敏感資訊偵測 | 2 | 62K | API keys/JWT/密碼等 |
| `function_reverse_engineering` | 逆向工程 | 4 | 32K | Binary/APK 分析 |
| `function_sqli` | SQL 注入偵測 | 25 | 124K | 6 種偵測引擎，最大模組 |
| `function_ssrf` | SSRF 偵測 | 10 | 96K | DNS rebinding + OAST |
| `function_steganography` | 隱寫術分析 | 7 | 36K | AI + StegX 雙引擎 |
| `function_wordlist_generator` | 字典產生器 | 5 | 44K | CommandHandler 已實現 |
| `function_xss` | XSS 偵測 | 16 | 112K | 4 種偵測類型 + CLI |

### PARTIAL — 2 個模組

| 模組 | 用途 | 問題 |
|------|------|------|
| `function_authn_go` | 認證測試 (Go wrapper) | Go binary 未編譯，執行時 RuntimeError |
| `function_social_engineering` | 社交工程 | `handler.py` 未實現、RiskGuard 未整合 |

### STUB — 1 個模組

| 模組 | 用途 | 問題 |
|------|------|------|
| `function_crypto` | 密碼學分析 | 純 Rust CLI，Python `__init__.py` 為空殼 |

### ~~BROKEN~~ ✅ 已修復 — 原 3 個模組 (2026-03-17 修復)

| 模組 | 原問題 | 修復方式 |
|------|------|------|
| `function_exploit` | ~~缺少 `__init__.py`~~ | ✅ 新增 `__init__.py`，匯出 `ExploitManager` |
| `function_postex` | ~~引用已廢棄的 `postex_manager.py`~~ | ✅ 改為匯入 `PostExDetector` + 三引擎 (依 README v1.3.0) |
| `function_web_scanner` | ~~引用已廢棄的 `scanner_manager.py`~~ | ✅ 改為匯入 `WebAttackManager` + 5 掃描器 (依 README v1.3.0) |

**修復說明：**

1. **function_exploit** — 新增 `__init__.py`，匯出 `ExploitManager`（來自 `exploit_manager.py`）。

2. **function_postex** — README v1.3.0 明確說明 `postex_manager.py` 已廢棄移除。新入口為 `detector/postex_detector.py` 的 `PostExDetector`，搭配 `engines/` 下三引擎。

3. **function_web_scanner** — README v1.3.0 明確說明 `scanner_manager.py` 已廢棄移除。新入口為 `integration_tools/web_tools.py` 的 `WebAttackManager`，搭配 `scanners/` 下 5 個掃描引擎。

---

## 二、services/core/ — AI 核心服務

**狀態: COMPLETE** — 結構完整，為專案核心入口。

| 子模組 | 用途 | `__init__.py` | 狀態 |
|--------|------|:---:|------|
| `aiva_core/cognitive_core/` | 認知核心（神經網路、RAG、學習） | ✅ | 完整 |
| `aiva_core/core_capabilities/` | 核心能力（分析、攻擊、對話） | ✅ | 完整 |
| `aiva_core/service_backbone/` | 服務骨幹（API、協調、訊息） | ✅ | 完整 |
| `aiva_core/task_planning/` | 任務規劃（指揮官、執行器） | ✅ | 完整 |
| `aiva_core/internal_exploration/` | 內部探索（多語言工具） | ✅ | 完整 |
| `ui/` | Rich 終端 UI | ✅ | 完整 |

**入口點**: `services/core/main.py` → `aiva_core/service_backbone/api/app.py`

---

## 三、services/scan/ — 多語言掃描引擎

| 引擎 | 語言 | 主要功能 | Dockerfile | 狀態 |
|------|------|----------|:---:|------|
| `python_engine/` | Python | XXE、反序列化、被動分析 | ❌ | COMPLETE |
| `go_engine/` | Go | CSPM、SCA、SSRF | ❌ | COMPLETE |
| `rust_engine/` | Rust | API 掃描、密碼爆破、JS 分析 | ✅ | COMPLETE |
| `typescript_engine/` | TypeScript | DOM 安全、WebSocket、SPA 路由 | ❌ | COMPLETE |
| `coordinators/` | Python | 跨語言協調層 | — | COMPLETE |

---

## 四、services/aiva_common/ — 共用套件

**狀態: COMPLETE** — 為專案所有模組的共用依賴。

| 子模組 | 用途 | `__init__.py` |
|--------|------|:---:|
| `ai/` | AI 工具與適配器 | ✅ |
| `async_utils/` | 非同步工具 | ✅ |
| `cli/` | CLI 框架 | ✅ |
| `config/` | 組態管理 | ✅ |
| `core/` | 核心基類 | ✅ |
| `cross_language/` | 跨語言橋接 | ✅ |
| `detection/` | 統一偵測管理 | ✅ |
| `enums/` | 列舉型別 | ✅ |
| `messaging/` | 訊息傳遞 | ✅ |
| `observability/` | 可觀測性 | ✅ |
| `pipeline/` | 管線框架 | ✅ |
| `plugins/` | 插件系統 | ✅ |
| `protocols/` | 通訊協定 | ❌ 缺少 |
| `schemas/` | 資料結構 (11 子模組) | ✅ |
| `security/` | 安全工具 | ✅ |
| `tools/` | 工具集 | ❌ 缺少 |
| `utils/` | 通用工具 | ✅ |

---

## 五、services/integration/ — 整合層

**狀態: COMPLETE** — 功能協調與資料管理。

包含：能力管理（`capability/`）、協調器（`coordinators/`）、資料管理（`simple_data_manager.py`）、搜尋處理（`search_command_handler.py`）。

---

## 六、services/dashboard/ — 儀表板

**狀態: STUB** — 僅有 `pages/` 目錄，缺少 `__init__.py`、README、Dockerfile。

---

## 問題彙總

### 需立即修復 (P0)

| # | 問題 | 模組 | 狀態 |
|---|------|------|------|
| 1 | ~~缺少 `__init__.py`~~ | `function_exploit` | ✅ 已修復 |
| 2 | ~~引用已廢棄的 `postex_manager.py`~~ | `function_postex` | ✅ 已修復 |
| 3 | ~~引用已廢棄的 `scanner_manager.py`~~ | `function_web_scanner` | ✅ 已修復 |
| 4 | ~~`commander/__init__.py:90` 引用不存在的 `rag_handler.py`~~ | `task_planning` | ✅ 已修復 |

### 功能待完成 (P1)

| # | 問題 | 模組 |
|---|------|------|
| 4 | Go binary 未編譯 | `function_authn_go` |
| 5 | `handler.py` 未實現 | `function_social_engineering` |
| 6 | `aiva_common/protocols/` 缺 `__init__.py` | `aiva_common` |
| 7 | `aiva_common/tools/` 缺 `__init__.py` | `aiva_common` |

### 基礎設施 (P2)

| # | 問題 |
|---|------|
| 8 | 14/17 功能模組缺少 Dockerfile |
| 9 | 3/4 掃描引擎缺少 Dockerfile |
| 10 | Dashboard 無任何組態 |
| 11 | 12/17 功能模組缺少 README |
