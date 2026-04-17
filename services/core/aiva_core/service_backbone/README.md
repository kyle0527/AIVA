# 🏗️ Service Backbone - 服務骨幹

> **路徑**: `service_backbone/`  
> **狀態**: ✅ 正常 | **最後更新**: 2026-04-05  
> **子模組**: 9 個 | **Python 文件數**: 38  
> **父模組**: [AIVA Core](../README.md)

## 概述

**Service Backbone** 是 AIVA 五大核心模組之一，作為基礎設施服務層，提供所有模組共享的核心服務。包括消息代理、狀態管理、存儲管理、服務協調、性能監控、權限控制等基礎能力，確保整個系統的穩定運行。

**核心職責**：
- 📨 **消息通信** - RabbitMQ 消息代理和發布/訂閱
- 📊 **狀態管理** - 會話狀態追蹤和上下文管理
- 💾 **存儲服務** - 統一的數據持久化接口
- 🎛️ **服務協調** - 跨模組協調和命令路由
- 📈 **性能監控** - 系統指標收集和健康檢查
- 🔐 **權限控制** - RBAC 權限矩陣和授權管理
- 🌐 **API 網關** - FastAPI 統一入口
- 🔧 **系統修復** - 診斷和修復工具 ⭐ 新增

---

## 📂 子模組 (Submodules)

- [adapters](./adapters/README.md)
- [api](./api/README.md)
- [authz](./authz/README.md)
- [coordination](./coordination/README.md)
- [messaging](./messaging/README.md)
- [performance](./performance/README.md)
- [state](./state/README.md)
- [storage](./storage/README.md)

## 📄 檔案概覽 (Files Overview)

- `context_manager.py` - AIVA Context Manager - 上下文管理系統

