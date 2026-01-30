# AIVA 技術文檔中心

**創建日期**: 2025年12月31日  
**狀態**: 已恢復並整合

---

## 📑 目錄

- [概述](#概述)
- [雙閉環架構文檔](#雙閉環架構文檔)
- [CLI 設計文檔](#cli-設計文檔)
- [相關代碼文件](#相關代碼文件)
- [快速導航](#快速導航)

---

## 概述

本目錄集中了 AIVA 的核心技術設計文檔，包括：
- 🔄 雙閉環架構（內閉環 + 外閉環）
- 💻 CLI 命令系統設計
- 📋 完整工作流程說明

這些文檔從 git 歷史中恢復，是理解 AIVA 架構的關鍵參考資料。

---

## 雙閉環架構文檔

### 📌 核心文檔

| 文件 | 說明 | 大小 |
|------|------|------|
| [13_STEPS_WORKFLOW_VERIFICATION.md](dual_loop/13_STEPS_WORKFLOW_VERIFICATION.md) | 📘 **完整 13 步驟工作流程** | 新創建 |
| [EXTERNAL_LOOP_ACTIVATION_PLAN.md](dual_loop/EXTERNAL_LOOP_ACTIVATION_PLAN.md) | 📘 **外閉環激活計劃** | 已恢復 |
| [INTEGRATION_DUAL_LOOP_DESIGN.md](dual_loop/INTEGRATION_DUAL_LOOP_DESIGN.md) | 📘 **雙閉環整合設計** | 已恢復 |

### 🎯 雙閉環概念

```
┌─────────────────────────────────────────────────────────┐
│                    AIVA 雙閉環架構                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  內閉環 (Internal Loop)                                 │
│  ├─ 模組探索 → 能力分析 → RAG 注入                      │
│  └─ 目的: AI 了解自己有哪些能力                         │
│                                                         │
│  外閉環 (External Loop)                                 │
│  ├─ 掃描 → 攻擊 → 偏差分析 → 經驗學習                   │
│  └─ 目的: 使用能力實戰，收集經驗並優化                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 📊 13 步驟流程

1. **Phase 0 (內閉環)**: 步驟 1-3 自我認知
2. **Phase 1 (情報收集)**: 步驟 4-6 目標分析
3. **Phase 2 (外閉環執行)**: 步驟 7-9 掃描與驗證
4. **Phase 3 (攻擊)**: 步驟 10-11 漏洞利用
5. **Phase 4 (學習)**: 步驟 12-13 偏差分析與優化

---

## CLI 設計文檔

### 📌 核心文檔

| 文件 | 說明 | 大小 |
|------|------|------|
| [CAPABILITY_CLI_DESIGN.md](cli_design/CAPABILITY_CLI_DESIGN.md) | 📘 **能力 CLI 設計規範** | 已恢復 |
| [CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md](cli_design/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md) | 📘 **CLI 命令架構分析** | 已恢復 |

### 🎯 CLI 設計原則

AIVA 的 CLI 系統基於**能力發現 (Capability Discovery)** 設計：

1. **自動生成**: 從代碼分析自動生成 CLI 命令
2. **統一接口**: 所有能力都有標準的 CLI 入口
3. **可發現性**: 通過 `--help` 和自動完成發現功能
4. **類型安全**: 參數驗證和類型檢查

### 📋 CLI 命令結構

```bash
# 基本格式
aiva <module> <capability> [--param value]

# 範例
aiva scan ssrf --target http://target.com --callback http://attacker.com
aiva attack sqli --target http://target.com --param id
aiva analyze deviation --task-id abc123
```

---

## 相關代碼文件

### 雙閉環實現

| 路徑 | 說明 |
|------|------|
| [services/aiva_common/schemas/dual_loop.py](../../services/aiva_common/schemas/dual_loop.py) | 雙閉環 Schema 定義 (743 行) |
| [services/core/aiva_core/cognitive_core/internal_loop_connector.py](../../services/core/aiva_core/cognitive_core/internal_loop_connector.py) | 內閉環連接器 (2009 行) |
| [services/core/aiva_core/cognitive_core/external_loop_connector.py](../../services/core/aiva_core/cognitive_core/external_loop_connector.py) | 外閉環連接器 |
| [services/integration/coordinators/base_coordinator.py](../../services/integration/coordinators/base_coordinator.py) | 雙閉環協調器基類 (548 行) |

### CLI 系統實現

| 路徑 | 說明 |
|------|------|
| [services/core/aiva_core/internal_exploration/](../../services/core/aiva_core/internal_exploration/) | 內部探索模組（能力發現） |
| [services/core/aiva_core/cognitive_core/ai_capability_query.py](../../services/core/aiva_core/cognitive_core/ai_capability_query.py) | AI 能力查詢接口 |

---

## 快速導航

### 想了解雙閉環架構？
1. 先讀 [13_STEPS_WORKFLOW_VERIFICATION.md](dual_loop/13_STEPS_WORKFLOW_VERIFICATION.md) 了解整體流程
2. 再讀 [EXTERNAL_LOOP_ACTIVATION_PLAN.md](dual_loop/EXTERNAL_LOOP_ACTIVATION_PLAN.md) 了解外閉環實施
3. 最後讀 [INTEGRATION_DUAL_LOOP_DESIGN.md](dual_loop/INTEGRATION_DUAL_LOOP_DESIGN.md) 了解整合設計

### 想了解 CLI 系統？
1. 先讀 [CAPABILITY_CLI_DESIGN.md](cli_design/CAPABILITY_CLI_DESIGN.md) 了解設計理念
2. 再讀 [CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md](cli_design/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md) 了解架構實現

### 想了解掃描模組職責？
- 參考 [13_STEPS_WORKFLOW_VERIFICATION.md](dual_loop/13_STEPS_WORKFLOW_VERIFICATION.md) 中的「掃描模組的功用」章節

---

## 更新記錄

| 日期 | 變更 |
|------|------|
| 2025-12-31 | 從 git 歷史恢復外閉環和 CLI 設計文檔 |
| 2025-12-30 | 創建 13 步驟工作流程驗證文檔 |

---

## 相關資源

- [INTERNAL_LOOP_EXECUTION_GUIDE.md](../../guides/INTERNAL_LOOP_EXECUTION_GUIDE.md) - 內閉環執行手冊
- [Scan README](../../services/scan/README.md) - 掃描模組說明
- [Services README](../../services/README.md) - 服務架構總覽
- [guides/README.md](../../guides/README.md) - 完整指南索引

---

*技術文檔中心 | AIVA v2.1.2+*
