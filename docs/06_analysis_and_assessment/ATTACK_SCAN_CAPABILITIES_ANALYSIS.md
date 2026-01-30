# 攻擊與掃描能力分析

## 📑 目錄

- [分析目的](#分析目的)
- [掃描相關能力](#掃描相關能力)
  - [service_backbone (3 個)](#service_backbone-3-個)
    - [scan_result_processor](#scan_result_processor)
    - [scan_module_interface](#scan_module_interface)
    - [two_phase_scan_orchestrator](#two_phase_scan_orchestrator)
- [攻擊執行相關能力](#攻擊執行相關能力)
  - [core_capabilities (1 個)](#core_capabilities-1-個)
    - [attack_chain](#attack_chain)
  - [service_backbone (5 個)](#service_backbone-5-個)
    - [exploit_orchestrator](#exploit_orchestrator)
    - [attack_validator](#attack_validator)
    - [payload_generator](#payload_generator)
    - [execution_status_monitor](#execution_status_monitor)
    - [exploit_manager_legacy](#exploit_manager_legacy)
  - [task_planning (5 個)](#task_planning-5-個)
    - [bizlogic_attack_executor](#bizlogic_attack_executor)
    - [plan_executor](#plan_executor)
    - [attack_executor](#attack_executor)
    - [task_executor](#task_executor)
    - [attack_plan_mapper](#attack_plan_mapper)
- [統計摘要](#統計摘要)
- [關鍵發現](#關鍵發現)

---


**分析日期**: 2026-01-01

## 分析目的

識別 AIVA 系統中可以執行掃描和攻擊操作的能力終點，
分析這些能力的分布和訪問路徑。

## 掃描相關能力

共找到 **3** 個掃描相關能力

### service_backbone (3 個)

#### scan_result_processor

- **路徑數**: 2
- **說明**: scan_result_processor - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py`

#### scan_module_interface

- **路徑數**: 3
- **說明**: scan_module_interface - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py`

#### two_phase_scan_orchestrator

- **路徑數**: 2
- **說明**: two_phase_scan_orchestrator - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py`

## 攻擊執行相關能力

共找到 **11** 個攻擊執行相關能力

### core_capabilities (1 個)

#### attack_chain

- **路徑數**: 1
- **說明**: 攻擊鏈 - 預設攻擊序列執行
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_chain.py`

### service_backbone (5 個)

#### exploit_orchestrator

- **路徑數**: 2
- **說明**: exploit_orchestrator - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator.py`

#### attack_validator

- **路徑數**: 1
- **說明**: attack_validator - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_validator.py`

#### payload_generator

- **路徑數**: 2
- **說明**: payload_generator - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\payload_generator.py`

#### execution_status_monitor

- **路徑數**: 1
- **說明**: execution_status_monitor - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\execution_status_monitor.py`

#### exploit_manager_legacy

- **路徑數**: 1
- **說明**: exploit_manager_legacy - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_manager_legacy.py`

### task_planning (5 個)

#### bizlogic_attack_executor

- **路徑數**: 2
- **說明**: bizlogic_attack_executor - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\bizlogic_attack_executor.py`

#### plan_executor

- **路徑數**: 2
- **說明**: 計劃執行器 - 任務執行管理
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py`

#### attack_executor

- **路徑數**: 1
- **說明**: attack_executor - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_executor.py`

#### task_executor

- **路徑數**: 1
- **說明**: task_executor - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\task_executor.py`

#### attack_plan_mapper

- **路徑數**: 1
- **說明**: attack_plan_mapper - 功能組件
- **位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\attack_plan_mapper.py`

## 統計摘要

| 類型 | 獨特終點數 | 總路徑數 | 平均路徑數 |
|------|-----------|---------|----------|
| 掃描能力 | 3 | 7 | 2.33 |
| 攻擊能力 | 11 | 15 | 1.36 |

## 關鍵發現

1. **掃描能力分布**: 1 個模組提供掃描功能
2. **攻擊能力分布**: 3 個模組提供攻擊功能
3. **能力覆蓋率**: 14.3% 的獨特終點與攻擊或掃描相關
4. **路徑彈性**: 攻擊和掃描能力平均有 1.57 條訪問路徑
