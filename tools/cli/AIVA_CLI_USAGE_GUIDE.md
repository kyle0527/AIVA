# AIVA CLI 使用指南

基於V9流程分析的命令列介面系統，支援282條分析過的流程路徑。

## 🚀 快速開始

### 基本用法

```bash
# 顯示統計資訊
python aiva_cli_implementation.py stats

# 到達特定目標（自動選擇最短路徑）
python aiva_cli_implementation.py reach ai_model_manager

# 從特定起點到目標
python aiva_cli_implementation.py reach ai_model_manager --from=scalable_bio_trainer

# 列出兩點間的所有路徑
python aiva_cli_implementation.py paths scalable_bio_trainer ai_model_manager
```

## 📋 命令參考

### 1. reach - 到達目標

**語法**: `aiva reach <target> [選項]`

**選項**:
- `--from=<service>`: 指定起始服務
- `--path=<number>`: 選擇特定路徑編號
- `--prefer=<criterion>`: 路徑偏好 (shortest/longest)
- `--interactive`: 互動選擇模式
- `--dry-run`: 試運行模式（不實際執行）

**範例**:
```bash
# 基本用法 - 到達AI模型管理器
aiva reach ai_model_manager

# 指定起點
aiva reach ai_model_manager --from=scalable_bio_trainer

# 選擇第2條路徑
aiva reach ai_model_manager --from=scalable_bio_trainer --path=2

# 偏好最短路徑
aiva reach ai_model_manager --from=scalable_bio_trainer --prefer=shortest

# 互動選擇模式
aiva reach ai_model_manager --from=scalable_bio_trainer --interactive

# 試運行模式
aiva reach ai_model_manager --from=scalable_bio_trainer --dry-run
```

### 2. paths - 路徑查詢

**語法**: `aiva paths <source> <target>`

**範例**:
```bash
# 查看所有可用路徑
aiva paths scalable_bio_trainer ai_model_manager
```

### 3. flow - 流程操作

#### 3.1 列出流程
**語法**: `aiva flow list [--module=<module>]`

```bash
# 列出所有流程
aiva flow list

# 按模組過濾
aiva flow list --module=cognitive_core
```

#### 3.2 顯示流程詳情
**語法**: `aiva flow show <flow_id>`

```bash
# 顯示特定流程
aiva flow show flow_001
```

#### 3.3 執行流程
**語法**: `aiva flow run <flow_id> [--dry-run]`

```bash
# 執行流程
aiva flow run flow_001

# 試運行
aiva flow run flow_001 --dry-run
```

### 4. stats - 統計資訊

**語法**: `aiva stats`

```bash
# 顯示完整統計
aiva stats
```

## 🎯 實際使用場景

### 場景1: 快速到達常用服務

```bash
# 最常用的目標服務
aiva reach ai_model_manager            # AI模型管理
aiva reach model_trainer               # 模型訓練
aiva reach app                         # 主應用
aiva reach train_classifier           # 分類器訓練
aiva reach real_neural_core           # 神經網路核心
```

### 場景2: 指定起點的流程

```bash
# 從生物訓練器開始的常見流程
aiva reach ai_model_manager --from=scalable_bio_trainer
aiva reach model_trainer --from=scalable_bio_trainer
aiva reach training_orchestrator --from=scalable_bio_trainer

# 從初始掃描開始
aiva reach exploit_orchestrator --from=initial_surface
```

### 場景3: 多路徑場景處理

```bash
# 查看可用路徑
aiva paths scalable_bio_trainer authz_mapper

# 互動選擇（推薦用於多路徑場景）
aiva reach authz_mapper --from=scalable_bio_trainer --interactive

# 選擇特定路徑
aiva reach authz_mapper --from=scalable_bio_trainer --path=1
```

### 場景4: 探索和分析

```bash
# 查看系統統計
aiva stats

# 列出認知核心模組的流程
aiva flow list --module=cognitive_core

# 檢查特定流程
aiva flow show flow_045
```

## 🔍 高級用法

### 試運行模式
在實際執行前測試流程：

```bash
aiva reach ai_model_manager --from=scalable_bio_trainer --dry-run
```

### 互動模式
當有多條路徑時，讓系統顯示選項供選擇：

```bash
aiva reach command_repository --from=scalable_bio_trainer --interactive
```

輸出範例：
```
🎯 目標: command_repository
🚀 起點: scalable_bio_trainer
📊 找到 6 條可用路徑

可用路徑:
  1. scalable_bio_trainer → command_repository (長度: 2)
  2. scalable_bio_trainer → rl_trainers → capability_orchestrator → postgresql_vector_store → command_repository (長度: 5)
  3. scalable_bio_trainer → rl_trainers → vector_store → command_repository (長度: 4)
  ...

請選擇路徑 (1-6): 1
```

## 📊 V9分析資料概覽

基於282條已分析流程：

- **流程分佈**:
  - 5步流程: 47.9% (135條)
  - 4步流程: 34.8% (98條)
  - 3步流程: 11.0% (31條)
  - 2步流程: 6.4% (18條)

- **熱門起點**:
  1. scalable_bio_trainer: 266條流程
  2. logging_formatter: 7條流程
  3. monitoring: 5條流程

- **熱門終點**:
  1. ai_model_manager: 13條流程
  2. model_trainer: 12條流程
  3. app: 11條流程

- **多路徑案例**: 59組，支援彈性執行策略

## 🛠️ 實現細節

### 路徑選擇邏輯
1. **shortest**: 選擇步驟最少的路徑
2. **longest**: 選擇步驟最多的路徑
3. **interactive**: 顯示所有選項供用戶選擇

### 錯誤處理
- 找不到路徑時顯示清晰的錯誤訊息
- 無效路徑編號的範圍檢查
- 資料載入失敗的友善提示

### 執行模式
- **正常模式**: 實際執行流程
- **試運行模式**: 只顯示執行計劃，不實際執行

## 🚧 待實現功能

1. **實際服務執行**: 目前只是模擬，需要實現真正的服務調用
2. **錯誤恢復**: 流程執行失敗時的恢復機制
3. **並行執行**: 支援多個獨立路徑的並行執行
4. **配置管理**: 支援不同環境的配置
5. **日誌記錄**: 詳細的執行日誌和審計追蹤

## 🔗 相關檔案

- 主實現: `tools/cli/aiva_cli_implementation.py`
- 流程資料: `services/core/aiva_core/internal_exploration/services_classification_v9_new/classification_data.json`
- 操作手冊: `services/core/aiva_core/internal_exploration/services_classification_v9_new/OPERATION_MANUAL.md`