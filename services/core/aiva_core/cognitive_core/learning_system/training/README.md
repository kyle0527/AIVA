# Training 訓練和場景管理模組

> **路徑**: `cognitive_core/learning_system/training`  
> **狀態**: ✅ 正常 | **文件數**: 2 | **最後更新**: 2026-01-07

## 概述

提供 OWASP 靶場場景管理、訓練編排等功能。負責管理標準靶場場景的定義、加載、驗證和執行，用於 AI 模型訓練和測試。

## 核心組件

### scenario_manager.py

- `ScenarioManager` - OWASP 靶場場景管理器
  - 場景定義和元數據管理
  - 場景驗證和健康檢查
  - 場景執行和結果收集
  - 場景難度評估和分級
  - 訓練數據集構建

**支援的靶場來源：**
- OWASP WebGoat
- OWASP Juice Shop
- DVWA (Damn Vulnerable Web Application)
- 自定義靶場

**難度級別：**
| 級別 | 分數 |
|------|------|
| easy | 1.0 |
| medium | 2.0 |
| hard | 3.0 |
| expert | 4.0 |

### __init__.py

- 導出：`Scenario`, `ScenarioManager`, `ScenarioResult`
- 注意：`TrainingOrchestrator` 已移除，訓練功能整合至 `ContinuousLearningEngine`

## 依賴關係

- 內部依賴：
  - `aiva_common.schemas` (AttackPlan, AttackStep, StandardScenario)
  - `aiva_common.enums.VulnerabilityType`
- 外部依賴：`json`, `pathlib`, `uuid`

## 使用範例

```python
from cognitive_core.learning_system.training import ScenarioManager
from aiva_common.enums import VulnerabilityType

# 初始化場景管理器
manager = ScenarioManager(scenarios_dir="./data/scenarios")

# 創建標準場景
scenario = await manager.create_scenario(
    name="SQL Injection Basic",
    description="基礎 SQL 注入測試場景",
    vulnerability_type=VulnerabilityType.SQL_INJECTION,
    difficulty_level="easy",
    target_config={
        "url": "http://dvwa.local/vulnerabilities/sqli/",
        "auth": {"username": "admin", "password": "admin"}
    },
    expected_plan=attack_plan,
    success_criteria={
        "data_extracted": True,
        "no_detection": True
    },
    tags=["owasp", "sqli", "beginner"]
)

# 加載場景
scenario = await manager.load_scenario("scenario_abc123")

# 獲取所有場景
all_scenarios = await manager.list_scenarios(
    vulnerability_type=VulnerabilityType.SQL_INJECTION,
    difficulty_level="easy"
)

# 驗證場景
is_valid = await manager.validate_scenario(scenario)
```

## 場景元數據

每個場景自動生成的元數據包括：
- `steps_count` - 預期步驟數
- `has_dependencies` - 是否有步驟依賴
- `estimated_duration` - 預估執行時間
- `difficulty_score` - 難度分數
