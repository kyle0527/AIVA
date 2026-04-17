# 📋 Planner - 任務規劃器

> **版本**: v2.5.0  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2026-04-05  
> **父模組**: [Task Planning](../README.md)  
> **符合規範**: [aiva_common](../../../../aiva_common/README.md)  
> **檔案數**: 7 個 Python 模組  
> **代碼行數**: 1,869 行

---

## 📄 檔案詳細資訊 (Files Details)

### `ast_parser.py`
**說明**: AST Parser - 攻擊流程圖解析器

**類別 (Classes)**:
- `NodeType` - 攻擊流程節點類型
- `AttackFlowNode` - 攻擊流程節點
- `AttackFlowEdge` - 攻擊流程邊
- `AttackFlowGraph` - 攻擊流程圖
- `ASTParser` - AST 解析器

### `plan_comparator.py`
**說明**: Plan Comparator - 攻擊計畫對比分析器

**類別 (Classes)**:
- `StepMatch` - 步驟匹配結果
- `PlanComparator` - 攻擊計畫對比分析器

### `task_converter.py`
**說明**: Task Converter - 任務轉換器

**類別 (Classes)**:
- `TaskPriority` - 任務優先級 (AI 規劃器專用)
- `ExecutableTask` - 可執行任務
- `TaskSequence` - 任務序列
- `TaskConverter` - 任務轉換器

### `task_execution_planner.py`
**說明**: AIVA Execution Planner - 執行計劃器

**類別 (Classes)**:
- `ExecutionPlanner` - 執行計劃器 - 負責異步執行計劃和步驟編排
**函式 (Functions)**:
- `get_execution_planner()` - 獲取執行計劃器實例

### `task_generator.py`
**說明**: 無特定描述。

**類別 (Classes)**:
- `TaskGenerator` - Translate strategies into concrete Function tasks.

### `tool_selector.py`
**說明**: Tool Selector - 工具選擇器

**類別 (Classes)**:
- `ServiceType` - 服務類型
- `ToolDecision` - 工具選擇決策
- `ToolSelector` - 工具選擇器

