# Decision 決策支援模組

> **路徑**: `cognitive_core/decision`  
> **狀態**: ✅ 正常 | **Python 文件數**: 6 | **最後更新**: 2026-04-05  
> **Bug Bounty 決策引擎**: ✅ v4.4.0 完全整合

## 概述

提供增強的 AI 決策能力和推理支援，整合風險評估、經驗驅動決策和執行計畫生成。**v4.4.0 重大更新**: 新增 Bug Bounty 決策引擎，針對 HackerOne/Bugcrowd 實戰場景專業優化。

## 🎯 Bug Bounty 決策引擎 (NEW)

### enhanced_decision_agent.py ⭐

**核心類**: `EnhancedDecisionAgent` (2200+ 行代碼)

**四大決策方法**:

#### 1. `decide_scan_strategy()` - 智慧掃描工具選擇
- **功能**: 智慧選擇 nmap/masscan，目標分析和策略適配
- **整合位置**: `task_planning/commander/attack_coordinator.py` (Line 508)
- **特色**: WAF 檢測、時間預估、參數優化

#### 2. `decide_phase1_strategy()` - Phase1 深度掃描決策  
- **功能**: ROI 導向決策，$75/hr 閾值判斷
- **整合位置**: `core_capabilities/orchestration/two_phase_scan_orchestrator.py`
- **特色**: Program Scope 檢查、高價值目標識別、時間投資回報分析

#### 3. `decide_phase2_targets()` - 攻擊目標優先級排序
- **功能**: Tier 1-3 優先級系統 (Critical $10k+, High $5k+, Medium $1k+)
- **整合位置**: 兩個編排器中
- **特色**: 漏洞類型風險評估、獎金潛力計算、攻擊複雜度分析

#### 4. `evaluate_phase2_results()` - 結果評估和後續行動
- **功能**: HackerOne 報告指導、攻擊鏈分析、後續行動建議
- **整合位置**: 兩個編排器中
- **特色**: CVSS 評分輔助、行動建議 (SUBMIT_REPORT/CONTINUE_DEEP_DIVE/CHAIN_VULNERABILITIES)

**實戰優化**:
- ✅ **HackerOne 獎金表**: Critical $10k+, High $5k+, Medium $1k+
- ✅ **WAF 繞過策略**: Cloudflare, Imperva, AWS WAF 專門技術
- ✅ **OWASP WSTG 映射**: 4.1-4.12 完整測試類別覆蓋
- ✅ **5M 神經網絡**: 語意向量 (384) + 特徵向量 (32) 增強決策
- ✅ **CVSS 多版本**: 3.0/3.1/4.0 評分系統支援

**核心組件**:
- `DecisionContext` - 決策上下文，包含風險級別、發現的漏洞、目標資訊等
- `Decision` - 決策結果，包含動作、參數、信心度和推理過程
- `EnhancedDecisionAgent` - Bug Bounty 增強決策代理
  - 整合 RealDecisionEngine (5M 神經網路)
  - Bug Bounty 特化配置 (WAF 繞過策略、速率限制)
  - 規則引擎 + 神經網路混合決策

---

## 傳統決策組件

### scan_execution_planner.py

- `ScanStrategy` - 掃描策略枚舉 (INITIAL_DEEP, INFORMED, MULTI_ENGINE, TARGETED)
- `ExecutionStep` - 執行步驟，包含能力、模組、命令類型和參數
- `ExecutionPlan` - 執行計畫，包含目標、策略和步驟列表
- `NextPhaseDecision` - 下一階段決策
- `ScanExecutionPlanner` - 執行計畫生成器，支持首次深入掃描和知情掃描

### skill_graph.py

- `SkillNode` - 技能節點，包含語言、主題、成功率等
- `SkillEdge` - 技能邊關係 (prerequisite, alternative, complement, sequence)
- `SkillPath` - 技能執行路徑
- `SkillGraphBuilder` - 技能圖構建器，基於 NetworkX
- `SkillGraphAnalyzer` - 技能圖分析器
- `AIVASkillGraph` - AIVA 技能圖主類

### __init__.py

- 版本: `3.0.0-alpha`

## 依賴關係

- 內部依賴：
  - `neural.real_neural_core.RealDecisionEngine`
  - `aiva_common.schemas` (HighLevelIntent, CommandContext)
  - `aiva_common.enums.RiskLevel`
- 外部依賴：`networkx`, `torch`, `asyncio`

## 使用範例

```python
from cognitive_core.decision import EnhancedDecisionAgent, ExecutionPlanner

# 決策代理
agent = EnhancedDecisionAgent(knowledge_base=kb)
decision = agent.make_decision(context)

# 執行計畫生成
planner = ExecutionPlanner()
plan = await planner.generate_plan(
    targets=["http://target.com"],
    constraints={"allowed_capabilities": ["scan", "exploit"]},
    is_new_target=True
)
```

## 📄 檔案詳細資訊 (Files Details)

### `bounty_strategy_agent.py`
**說明**: Bug Bounty 特化決策代理

**類別 (Classes)**:
- `BountyStrategyAgent` - Bug Bounty 特化策略決策代理

### `enhanced_decision_agent.py`
**說明**: AIVA 決策代理增強模組

**類別 (Classes)**:
- `DecisionContext` - 決策上下文
- `Decision` - 決策結果
- `EnhancedDecisionAgent` - 增強的決策代理（繼承 KnowledgeDecisionMixin 取得知識決策方法）
**函式 (Functions)**:
- `demo_enhanced_decision_agent()` - 示範增強決策代理功能

### `knowledge_decision_mixin.py`
**說明**: Knowledge Decision Mixin - 知識驅動決策混入

**類別 (Classes)**:
- `_MixinHostProtocol` - Mixin 宿主類別應實現的協議
- `KnowledgeDecisionMixin` - 知識驅動決策混入

### `scan_execution_planner.py`
**說明**: 执行计划生成器

**類別 (Classes)**:
- `ScanStrategy` - 扫描策略
- `ExecutionStep` - 执行步骤
- `ExecutionPlan` - 执行计划
- `NextPhaseDecision` - 下一阶段决策
- `ExecutionPlanner` - 执行计划生成器

### `skill_graph.py`
**說明**: AIVA 技能圖 (Skill Graph) 模組

**類別 (Classes)**:
- `SkillNode` - 技能節點
- `SkillEdge` - 技能邊關係
- `SkillPath` - 技能執行路徑
- `SkillGraphBuilder` - 技能圖構建器
- `SkillGraphAnalyzer` - 技能圖分析器
- `AIVASkillGraph` - AIVA 技能圖主類
