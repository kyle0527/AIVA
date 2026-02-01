# 各模組未被 Flows 覆蓋的能力分析報告

## 📑 目錄

- [📊 整體統計](#-整體統計)
- [📈 各模組覆蓋率](#-各模組覆蓋率)
- [🔍 未被覆蓋的能力詳細](#-未被覆蓋的能力詳細)
  - [cognitive_core](#cognitive_core)
  - [internal_exploration](#internal_exploration)
  - [task_planning](#task_planning)
  - [external_learning](#external_learning)
  - [core_capabilities](#core_capabilities)
  - [service_backbone](#service_backbone)
- [💡 分析與建議](#-分析與建議)
  - [1. 覆蓋率評估](#1-覆蓋率評估)
  - [2. 未覆蓋腳本性質](#2-未覆蓋腳本性質)
  - [3. 建議行動](#3-建議行動)

---


> **生成時間**: 2026-01-01  
> **分析對象**: 六大模組中的所有 Python 腳本

## 📊 整體統計

| 指標 | 數值 |
|------|------|
| 總腳本數 | 113 |
| 被 Flows 覆蓋 | 88 (77.9%) |
| 未被覆蓋 | 36 (22.1%) |

## 📈 各模組覆蓋率

| 模組 | 總腳本數 | 已覆蓋 | 未覆蓋 | 覆蓋率 |
|------|----------|--------|--------|--------|
| cognitive_core | 20 | 18 | 3 | 90.0% |
| service_backbone | 25 | 22 | 4 | 88.0% |
| external_learning | 14 | 12 | 5 | 85.7% |
| internal_exploration | 12 | 10 | 3 | 83.3% |
| core_capabilities | 24 | 15 | 14 | 62.5% |
| task_planning | 18 | 11 | 7 | 61.1% |


## 🔍 未被覆蓋的能力詳細


### cognitive_core

**未覆蓋腳本**: 3 個

- `execution_orchestrator.py`
- `execution_planner.py`
- `manifest_loader.py`

### internal_exploration

**未覆蓋腳本**: 3 個

- `aiva_flow_analyzer.py`
- `aiva_flow_classifier.py`
- `analysis_path_validator.py`

### task_planning

**未覆蓋腳本**: 7 個

- `attack_plan_mapper_backup_20251226.py`
- `command_builder.py`
- `command_router.py`
- `mode_manager.py`
- `orchestrator.py`
- `task_queue_manager.py`
- `unified_executor.py`

### external_learning

**未覆蓋腳本**: 5 個

- `continuous_learning.py`
- `execution_tracer.py`
- `online_learner.py`
- `scalable_bio_trainer.py`
- `unified_tracer.py`

### core_capabilities

**未覆蓋腳本**: 14 個

- `aiva_cli.py`
- `analyze_and_fix_mappings.py`
- `analyze_cli_capabilities.py`
- `analyze_flows.py`
- `analyze_manifest_status.py`
- `extract_script_file_mapping.py`
- `flow_executor.py`
- `initial_surface.py`
- `list_manifests.py`
- `select_test_flows.py`
- `task_context.py`
- `update_script_mappings_with_paths.py`
- `verify_102_scripts.py`
- `verify_reproduction_readiness.py`

### service_backbone

**未覆蓋腳本**: 4 個

- `config.py`
- `logging_formatter.py`
- `models.py`
- `monitoring.py`


## 💡 分析與建議

### 1. 覆蓋率評估

- **整體覆蓋率**: 77.9%
- **最高覆蓋率**: cognitive_core (90.0%)
- **最低覆蓋率**: task_planning (61.1%)

### 2. 未覆蓋腳本性質

未被 flows 覆蓋的 36 個腳本可能屬於以下類別：

1. **內部模組**: 被其他腳本導入使用，不作為獨立入口
2. **工具腳本**: 獨立執行的工具，不通過 flow 調用
3. **測試代碼**: 測試文件或示例代碼
4. **基礎設施**: 配置、初始化等基礎組件
5. **待整合**: 新開發尚未整合到流程中的功能
6. **廢棄代碼**: 可能需要清理的舊代碼

### 3. 建議行動

**立即行動**:
- ✅ 審查未覆蓋腳本列表
- ✅ 識別哪些是必要的內部模組
- ✅ 確定哪些是廢棄代碼

**短期計劃**:
- 🔧 為需要的獨立工具創建對應 flows
- 📝 文檔化工具腳本的使用方式
- 🗑️ 清理確認廢棄的代碼

**長期規劃**:
- 🎯 提升整體覆蓋率到 80%+
- 📚 建立代碼審查機制
- 🔄 定期更新能力地圖

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍
