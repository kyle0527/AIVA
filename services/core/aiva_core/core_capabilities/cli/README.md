# CLI 命令行模組

> **路徑**: `services/core/aiva_core/core_capabilities/cli`  
> **狀態**: ✅ 正常 | **文件數**: 1 | **最後更新**: 2026-01-07

## 概述

AIVA 統一 CLI 入口點，基於動態 Flow 的函數調用系統。從 JSON 配置動態生成 CLI 命令。

## 核心組件

### aiva_cli.py
**主要函數：**
- `load_flow_definitions()` - 從 latest_classification.json 讀取所有 flows
- `create_flow_command()` - 為指定 flow_id 創建 Click 命令函數
- `register_all_flow_commands()` - 註冊所有 flow 到 CLI 群組
- `aiva()` - CLI 主入口點
- `run()` - 執行指定 flow
- `query()` - 查詢模式
- `train()` - 訓練模式
- `scan()` - 掃描目標
- `status()` - 查詢掃描狀態
- `health()` - 系統健康檢查
- `list_flows()` - 列出所有可用 flows
- `show_flow_statistics()` - 顯示 flow 統計資訊
- `show_flows_by_endpoint_module()` - 按端點模組分組顯示 flows

## 使用方式

```bash
# 列出所有 flows
aiva list-flows

# 執行特定 flow
aiva run --flow-id 123 --target http://example.com

# 查詢模式
aiva query "查詢內容"
```

## 依賴關係

- `click` - CLI 框架
- `internal_exploration.FlowExecutor` - Flow 執行器
- `latest_classification.json` - Flow 定義配置
