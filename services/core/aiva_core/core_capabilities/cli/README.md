# CLI 命令行模組

> **路徑**: `services/core/aiva_core/core_capabilities/cli`  
> **狀態**: ✅ 正常 | **Python 文件數**: 1 | **最後更新**: 2026-04-05

## 概述

AIVA 統一 CLI 入口點，基於動態 Flow 的函數調用系統。從 JSON 配置動態生成 CLI 命令。

## 📄 檔案詳細資訊 (Files Details)

### `aiva_cli.py`
**說明**: AIVA 統一 CLI 入口點 - 基於動態 Flow 的函數調用系統

**函式 (Functions)**:
- `load_flow_definitions()` - 從 latest_classification.json 讀取所有 flows
- `create_flow_command()` - 為指定 flow_id 創建命令函數
- `register_all_flow_commands()` - 為所有 flows 注冊動態命令
- `aiva()` - AIVA - AI-powered Vulnerability Analysis System
- `run()` - 執行指定 Flow（基於 flow_id）
- `query()` - 內部查詢 (Flow 0 別名)
- `train()` - 模型訓練 (Flow 4 別名)
- `scan()` - 攻擊面掃描 (Flow 8 別名)
- `status()` - 查詢掃描狀態 (Flow 2 別名)
- `health()` - 系統健康檢查 (Flow 1 別名)
- `list_flows()` - 列出所有可用的 Flows
- `show_flow_statistics()` - 顯示 Flow 統計信息
- `show_flows_by_endpoint_module()` - 按終點模組（六大模組）分類顯示 Flows

