# 🎯 Commander - AI 指揮協調器

> **版本**: v2.5.0  
> **狀態**: ✅ 生產就緒（重構完成）  
> **最後更新**: 2026-04-05  
> **父模組**: [Task Planning](../README.md)  
> **符合規範**: [aiva_common](../../../../aiva_common/README.md)  
> **檔案數**: 9 個 Python 模組  
> **代碼行數**: 2,029 行

---

## 📄 檔案詳細資訊 (Files Details)

### `attack_coordinator.py`
**說明**: 攻擊執行協調器

**類別 (Classes)**:
- `AttackCoordinator` - 攻擊執行協調器

### `capability_manager.py`
**說明**: 能力選單管理器

**類別 (Classes)**:
- `CapabilityManager` - 能力選單管理器

### `capability_matcher.py`
**說明**: 無特定描述。

**類別 (Classes)**:
- `CapabilityMatcher` - 能力匹配器 - 負責將 AI 意圖轉換為可執行的 Flow ID

### `learning_adapter.py`
**說明**: 學習系統適配器

**類別 (Classes)**:
- `LearningAdapter` - 學習系統適配器

### `plan_builder.py`
**說明**: 攻擊計劃建構器 - 負責生成攻擊計畫和提示詞建構

**類別 (Classes)**:
- `PlanBuilder` - 攻擊計劃建構器

### `policy_manager.py`
**說明**: 風險策略管理器

**類別 (Classes)**:
- `RiskRule` - 風險規則
- `RiskLevel` - 風險等級定義
- `PolicyManager` - 風險策略管理器

### `strategy_engine.py`
**說明**: 策略決策引擎

**類別 (Classes)**:
- `StrategyEngine` - 策略決策引擎

### `types.py`
**說明**: AI Commander 類型定義

**類別 (Classes)**:
- `AITaskType` - AI 任務類型
- `AIComponent` - AI 組件類型

