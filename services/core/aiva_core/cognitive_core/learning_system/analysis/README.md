# Analysis 分析模組

> **路徑**: `cognitive_core/learning_system/analysis`  
> **狀態**: ✅ 正常 | **Python 文件數**: 2 | **最後更新**: 2026-04-05

## 概述

提供 AST（攻擊流程圖）與實際執行 Trace 的對比分析功能，計算差異指標作為強化學習回饋信號。同時支援動態策略調整，實現自適應測試策略優化。

## 📄 檔案詳細資訊 (Files Details)

### `ast_trace_comparator.py`
**說明**: AST 與 Trace 對比分析模組

**類別 (Classes)**:
- `ComparisonMetrics` - 比較指標
- `StepComparison` - 單步比較結果
- `ASTTraceComparator` - AST 與 Trace 對比分析器

### `dynamic_strategy_adjustment.py`
**說明**: 動態策略調整器 (RL 整合版)

**類別 (Classes)**:
- `RewardConfig` - 執行結果的獎勵配置
- `StrategyAdjuster` - 動態策略調整器

