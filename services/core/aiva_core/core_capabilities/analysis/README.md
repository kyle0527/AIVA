# Analysis 分析模組

> **路徑**: `core_capabilities/analysis/`  
> **狀態**: ✅ 正常 | **Python 文件數**: 4 | **最後更新**: 2026-04-05  
> **父模組**: [Core Capabilities](../README.md)

## 概述

提供 AI 增強的代碼分析、攻擊面分析和業務邏輯掃描功能，基於 Tree-sitter AST 和神經網路實現智能代碼分析系統。

## 📄 檔案詳細資訊 (Files Details)

### `analysis_engine.py`
**說明**: AI增強代碼分析引擎

**類別 (Classes)**:
- `AnalysisType` - 分析類型枚舉
- `IndexingConfig` - 索引配置（從RAG 1遷移）
- `CacheManager` - 緩存管理器，避免重複索引（從RAG 1遷移）
- `AIAnalysisResult` - AI分析結果數據類
- `CodeChunk` - 程式碼片段數據類（從RAG 1遷移）
- `AIAnalysisEngine` - AI驅動的代碼分析引擎

### `bizlogic_scanner.py`
**說明**: 業務邏輯掃描器 - 對靶場執行業務邏輯漏洞掃描


### `initial_surface.py`
**說明**: 無特定描述。

**類別 (Classes)**:
- `InitialAttackSurface` - Compute initial attack surface from scan results.
