# Orchestration 編排模組

> **路徑**: `services/core/aiva_core/core_capabilities/orchestration`  
> **狀態**: ✅ 正常 | **Python 文件數**: 1 | **最後更新**: 2026-04-05

## 概述

兩階段掃描編排器，負責編排 Phase0 快速偵察和 Phase1 深度掃描的完整流程，包括命令發送、結果接收、AI 分析決策、引擎選擇等功能。

## 📄 檔案詳細資訊 (Files Details)

### `two_phase_scan_orchestrator.py`
**說明**: 兩階段掃描編排器 - Phase0/Phase1 流程控制

**類別 (Classes)**:
- `TwoPhaseOrchestratorError` - 兩階段編排器異常基類
- `Phase0TimeoutError` - Phase0 超時異常
- `Phase1TimeoutError` - Phase1 超時異常
- `AIDecisionError` - AI 決策異常
- `TwoPhaseScanOrchestrator` - 兩階段掃描編排器

