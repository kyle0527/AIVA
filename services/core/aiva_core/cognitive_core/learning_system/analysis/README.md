# Analysis 分析模組

> **路徑**: `cognitive_core/learning_system/analysis`  
> **狀態**: ✅ 正常 | **Python 文件數**: 2 | **最後更新**: 2026-04-05

## 概述

提供 AST（攻擊流程圖）與實際執行 Trace 的對比分析功能，計算差異指標作為強化學習回饋信號。同時支援動態策略調整，實現自適應測試策略優化。

## 核心組件

### ast_trace_comparator.py

- `ComparisonMetrics` - 比較指標數據類
  - 完成率指標：預期步驟數、完成步驟數、完成率
  - 順序指標：順序匹配率、亂序步驟數
  - 步驟差異：缺失步驟、額外步驟
  - 執行質量：成功/失敗步驟數、錯誤數量
  - 時間指標：總執行時間、平均步驟時間
  - 綜合評分 (0.0-1.0)
- `StepComparison` - 單步比較結果
- `ASTTraceComparator` - AST 與 Trace 對比分析器
  - 比較預期攻擊流程與實際執行軌跡
  - 計算綜合評分供強化學習使用

### dynamic_strategy_adjustment.py

- `StrategyAdjuster` - 動態策略調整器
  - WAF 適應調整
  - 基於歷史成功率調整
  - 基於目標技術棧調整
  - 基於已發現漏洞調整優先級
  - 從測試結果中學習

## 依賴關係

- 內部依賴：
  - `tracing.trace_recorder` (ExecutionTrace, TraceType)
  - `task_planning.planner.ast_parser` (AttackFlowGraph, NodeType)
  - `aiva_common.utils`
- 外部依賴：`dataclasses`, `logging`

## 使用範例

```python
from cognitive_core.learning_system.analysis import ASTTraceComparator, StrategyAdjuster

# AST 與 Trace 對比
comparator = ASTTraceComparator()
metrics = comparator.compare(expected_graph, actual_trace)

print(f"完成率: {metrics.completion_rate:.2%}")
print(f"綜合評分: {metrics.overall_score:.2f}")

# 動態策略調整
adjuster = StrategyAdjuster()
adjusted_plan = adjuster.adjust(base_plan, context)

# 從結果學習
adjuster.learn_from_result(feedback_data)
```
