# 🎯 Core Capabilities - 核心能力模組

> **路徑**: `core_capabilities/`  
> **狀態**: ✅ Production Ready | **最後更新**: 2026-04-05  
> **子模組**: 8 個 | **總文件數**: 23 | **Python 文件**: 23 | **Bug Bounty 整合**: ✅ 已完成  
> **父模組**: [AIVA Core](../README.md)

## 概述

**Core Capabilities** 是 AIVA 五大核心模組之一，作為核心能力編排中心。整合了攻擊鏈編排、代碼分析、CLI 接口、對話助理、數據攝取、編排系統、輸出轉換和結果處理能力，提供完整的能力編排架構。

**核心職責**：
- 🎯 **攻擊執行** - 編排和執行多步驟攻擊鏈
- 🔍 **代碼分析** - AI 增強的代碼安全分析、業務邏輯掃描 ⭐ 新增
- 💬 **對話交互** - 自然語言問答、智能選單和一鍵執行 ⭐ 新增
- 📥 **數據處理** - 掃描結果攝取、處理和輸出轉換
- 🔧 **能力註冊** - CapabilityRegistry 代理模式，遵循 SOT 原則
- 🎯 **Bug Bounty 編排** - Phase1/Phase2 決策整合，HackerOne 實戰優化
- 🖥️ **CLI 接口** - 基於動態 Flow 的統一命令行入口

---

## 📂 子模組 (Submodules)

- [analysis](./analysis/README.md)
- [attack](./attack/README.md)
- [cli](./cli/README.md)
- [dialog](./dialog/README.md)
- [ingestion](./ingestion/README.md)
- [orchestration](./orchestration/README.md)
- [output](./output/README.md)
- [processing](./processing/README.md)

## 📄 檔案概覽 (Files Overview)

- `capability_registry.py` - Capability Registry - 能力註冊表代理
- `multilang_coordinator.py` - Multi-Language AI Coordinator
- `risk_policy_manager.py` - 風險策略管理器
- `task_context.py` - 標準任務參數包 - 統一所有模組的通信接口

