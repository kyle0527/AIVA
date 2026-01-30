# Analysis 分析模組

> **路徑**: `core_capabilities/analysis/`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-21  
> **父模組**: [Core Capabilities](../README.md)

## 概述

提供 AI 增強的代碼分析、攻擊面分析和業務邏輯掃描功能，基於 Tree-sitter AST 和神經網路實現智能代碼分析系統。

## 核心組件

### bizlogic_scanner.py ⭐ 新增
- `BIZLOGIC_TARGETS` - 業務邏輯掃描目標配置
- 業務邏輯漏洞掃描功能
- 自動化業務流程分析

### analysis_engine.py
- `AnalysisType` - 分析類型枚舉（安全、漏洞、複雜度、模式、語義、架構）
- `IndexingConfig` - 索引配置（批次大小、並行工作線程）
- `CacheManager` - 分析結果緩存管理器
- `AIAnalysisResult` - AI 分析結果數據結構
- `CodeChunk` - 代碼片段結構（用於分塊處理）
- `AIAnalysisEngine` - 核心分析引擎，整合 Tree-sitter 和神經網路

### initial_surface.py
- `InitialAttackSurface` - 初步攻擊面計算器，從掃描結果識別潛在漏洞點
  - SSRF 參數提示識別
  - XSS 參數提示識別
  - SQL 注入候選識別
  - IDOR 候選識別

## 依賴關係

- `cognitive_core.neural.real_neural_core` - RealDecisionEngine（5M 決策引擎）
- `aiva_common.error_handling` - 統一錯誤處理
- `aiva_common.schemas` - Asset, ScanCompletedPayload
- `features.features_ready.function_bizlogic.business_schemas` - 業務邏輯結構
- `tree_sitter` (可選) - 增強 AST 解析
- `torch`, `numpy` - 神經網路推理
