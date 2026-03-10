# 模組連接打通實施方案

## 📑 目錄

- [問題分析](#問題分析)
- [解決方案](#解決方案)
  - [方案 1: CLI 統一調用（推薦，支持跨語言）](#方案-1-cli-統一調用推薦支持跨語言)
  - [方案 2: 直接導入調用（Python 內部）](#方案-2-直接導入調用python-內部)
- [實施步驟](#實施步驟)
  - [步驟 1: 創建模組調用器](#步驟-1-創建模組調用器)
  - [步驟 2: 更新模組功能](#步驟-2-更新模組功能)
  - [步驟 3: 更新 Flows 配置](#步驟-3-更新-flows-配置)
- [跨語言調用](#跨語言調用)
- [實施優先級](#實施優先級)
  - [第一批（核心路徑）](#第一批核心路徑)
  - [第二批（增強路徑）](#第二批增強路徑)
- [檢查清單](#檢查清單)
- [最佳實踐參考](#最佳實踐參考)
- [下一步](#下一步)

---


**方案日期**: 2026-01-01

**核心概念**: 直接在模組間建立調用連接，而非建立「反向通道」

## 問題分析

當前三個模組只能被調用，無法主動調用其他模組：
- `cognitive_core`: 124 入站，0 出站
- `internal_exploration`: 201 入站，0 出站
- `task_planning`: 48 入站，0 出站

## 解決方案

### 方案 1: CLI 統一調用（推薦，支持跨語言）

通過統一的 CLI 接口實現模組間調用：

```python
# 任何模組都可以這樣調用其他模組
from ..core_capabilities.cli import aiva_cli

result = aiva_cli.execute_capability(
    module='task_planning',
    capability='plan_generator',
    objective='scan target'
)
```

### 方案 2: 直接導入調用（Python 內部）

Python 模組間直接導入：

```python
# 動態導入避免循環依賴
def call_task_planning(**kwargs):
    from ...task_planning.planner import plan_generator
    return plan_generator.generate_plan(**kwargs)
```

## 實施步驟

### 步驟 1: 創建模組調用器

為每個單向模組創建 `module_caller.py`：

**cognitive_core/module_caller.py**:
```python
'''cognitive_core 模組間調用接口'''
from ..core_capabilities.cli import aiva_cli

class ModuleCaller:
    @staticmethod
    def call_task_planning(capability, **kwargs):
        return aiva_cli.execute_capability(
            module='task_planning',
            capability=capability,
            **kwargs
        )
    
    @staticmethod
    def call_core_capabilities(capability, **kwargs):
        return aiva_cli.execute_capability(
            module='core_capabilities',
            capability=capability,
            **kwargs
        )
```

### 步驟 2: 更新模組功能

在需要調用其他模組的地方使用 ModuleCaller：

```python
# cognitive_core/decision/enhanced_decision_agent.py
from ..module_caller import ModuleCaller

def make_decision(objective, context):
    # 決策邏輯...
    
    # 主動調用 task_planning
    plan = ModuleCaller.call_task_planning(
        capability='plan_generator',
        objective=objective
    )
    
    return plan
```

### 步驟 3: 更新 Flows 配置

為新的調用路徑添加 flow 定義到 `latest_classification.json`。

## 跨語言調用

CLI 支持多種語言調用：

**Python**:
```python
from subprocess import run
result = run(['aiva-cli', 'call', 'task_planning.plan_generator'])
```

**Rust**:
```rust
use std::process::Command;
Command::new('aiva-cli').args(['call', 'task_planning.plan_generator']).output()?
```

**Shell**:
```bash
aiva-cli call task_planning.plan_generator --args objective=scan
```

## 實施優先級

### 第一批（核心路徑）
1. cognitive_core → task_planning
2. task_planning → core_capabilities
3. internal_exploration → external_learning

### 第二批（增強路徑）
4. cognitive_core → core_capabilities
5. cognitive_core → external_learning
6. internal_exploration → cognitive_core

## 檢查清單

- [ ] 檢查 aiva_cli.py 是否存在及功能
- [ ] 創建 cognitive_core/module_caller.py
- [ ] 創建 internal_exploration/module_caller.py
- [ ] 創建 task_planning/module_caller.py
- [ ] 更新 cognitive_core 使用新調用接口
- [ ] 更新 internal_exploration 使用新調用接口
- [ ] 更新 task_planning 使用新調用接口
- [ ] 添加新的 flows 到 latest_classification.json
- [ ] 測試 Python 內部調用
- [ ] 測試 CLI 命令行調用
- [ ] 添加錯誤處理和日誌
- [ ] 更新文檔

## 最佳實踐參考

1. **微服務通信**: Spring Cloud, gRPC, REST API
2. **消息隊列**: RabbitMQ, Kafka（異步調用）
3. **服務發現**: Consul, Eureka（動態發現）
4. **錯誤處理**: Circuit Breaker, Retry, Fallback

## 下一步

1. 檢查並完善 `aiva_cli.py`
2. 創建第一個 `module_caller.py`
3. 測試調用
4. 更新 flows 配置
5. 逐步推廣
