# 📚 Learning System - 統一學習系統

> **路徑**: `cognitive_core/learning_system/`  
> **狀態**: ✅ 正常 | **最後更新**: 2026-04-05  
> **子模組**: 4 個 | **總 Python 文件數**: 18  
> **父模組**: [Cognitive Core](../README.md)

## 概述

**Learning System** 是 AIVA 的經驗學習系統，負責從執行結果和用戶回饋中學習，並優化 AI 決策策略。整合分析引擎、學習系統、執行追蹤、訓練編排和知識管理五大子系統，實現持續自我優化。

> **說明**: 此模組整合自原 `external_learning`，現為 cognitive_core 的子模組。

**核心職責**：
- 📊 **結果分析** - 分析執行結果，提取學習信號
- 🧠 **策略優化** - 基於學習結果優化決策模型
- 🎯 **經驗管理** - 管理歷史經驗，支持決策推理
- 📝 **執行追蹤** - 追蹤跨模組執行狀態，收集性能數據
- 📚 **知識管理** - 模組知識庫與三路比對評估

---

## 📂 子模組 (Submodules)

- [analysis](./analysis/README.md)
- [knowledge](./knowledge/README.md)
- [learning](./learning/README.md)
- [tracing](./tracing/README.md)

## 📄 檔案概覽 (Files Overview)

- `cli_decision_engine.py` - CLI Decision Engine - 基於 RAG 的 CLI 命令決策引擎
- `event_listener.py` - Learning System Event Listener - 統一經驗學習事件監聽器
- `experience_manager.py` - Experience Manager - 經驗管理器
- `flow_executor_adapter.py` - Flow Executor Adapter - CLIDecisionEngine 與 FlowExecutor 的橋接層
- `notification_system.py` - User Notification System - 用戶通知系統
- `rag_trigger.py` - RAG Trigger - RAG 觸發器

