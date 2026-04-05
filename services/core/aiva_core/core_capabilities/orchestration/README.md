# Orchestration 編排模組

> **路徑**: `services/core/aiva_core/core_capabilities/orchestration`  
> **狀態**: ✅ 正常 | **Python 文件數**: 1 | **最後更新**: 2026-04-05

## 概述

兩階段掃描編排器，負責編排 Phase0 快速偵察和 Phase1 深度掃描的完整流程，包括命令發送、結果接收、AI 分析決策、引擎選擇等功能。

## 核心組件

### two_phase_scan_orchestrator.py
**異常類別：**
- `TwoPhaseOrchestratorError` - 兩階段編排器異常基類
- `Phase0TimeoutError` - Phase0 超時異常
- `Phase1TimeoutError` - Phase1 超時異常
- `AIDecisionError` - AI 決策異常

**主要類別：**
- `TwoPhaseScanOrchestrator` - 兩階段掃描編排器
  - 發送/接收 Phase0 命令和結果
  - AI 分析決策是否需要 Phase1
  - 引擎選擇決策樹
  - 發送/接收 Phase1 命令和結果
  - 進入七階段處理流程

## 掃描流程

```
1. 發送 Phase0 命令 (tasks.scan.phase0)
2. 接收 Phase0 結果 (scan.phase0.completed)
3. AI 分析決策是否需要 Phase1
4. 引擎選擇決策樹（根據 Phase0 結果）
5. 發送 Phase1 命令 (tasks.scan.phase1)
6. 接收 Phase1 結果 (scan.completed)
7. 進入七階段處理流程
```

## 依賴關係

- `aiva_common.enums` - Topic 枚舉
- `aiva_common.mq` - AbstractBroker 消息代理
- `aiva_common.schemas` - Phase0/Phase1 相關 Payload
- `cognitive_core.decision.enhanced_decision_agent` - EnhancedDecisionAgent（Bug Bounty AI 決策）
