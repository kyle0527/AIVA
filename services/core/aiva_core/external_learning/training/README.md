# Training - 訓練編排

**導航**: [← 返回 External Learning](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [training_orchestrator.py](#training_orchestratorpy-1245-行-)
  - [distributed_trainer.py](#distributed_trainerpy-633-行-)
- [🔄 訓練流程](#-訓練流程)
- [📊 訓練監控](#-訓練監控)
- [🚀 性能優化](#-性能優化)
- [🔧 容錯機制](#-容錯機制)
- [📚 相關模組](#-相關模組)
- [💡 最佳實踐](#-最佳實踐)

---

## 📋 概述

**定位**: 分布式訓練編排和管理  
**狀態**: ✅ 已實現  
**文件數**: 2 個 Python 文件 (1,878 行)

## 📂 文件結構

```
training/
├── training_orchestrator.py (1,245 行) ⭐⭐⭐ - 訓練編排器
├── distributed_trainer.py (633 行) ⭐ - 分布式訓練
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### training_orchestrator.py (1,245 行) ⭐⭐⭐

**職責**: 統一訓練流程編排和調度

**核心功能**:
- 訓練任務調度
- 資源分配
- 並行訓練管理
- 訓練流程自動化

**使用範例**:
```python
from aiva_core.external_learning.training import TrainingOrchestrator

orchestrator = TrainingOrchestrator()

# 提交訓練任務
job = orchestrator.submit_training_job(
    name="capability_classifier_v2",
    config={
        "model": "random_forest",
        "data_path": "training_data.csv",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 20
        }
    },
    resources={
        "cpu": 4,
        "memory": "8GB",
        "gpu": 1
    }
)

# 監控訓練狀態
status = orchestrator.get_job_status(job.id)
# {"status": "running", "progress": 0.45, "eta": "10min"}

# 等待完成
orchestrator.wait_for_completion(job.id)

# 獲取結果
results = orchestrator.get_results(job.id)
```

**批量訓練**:
```python
# 並行訓練多個模型
jobs = orchestrator.submit_batch_training([
    {"model": "random_forest", "params": {...}},
    {"model": "svm", "params": {...}},
    {"model": "neural_network", "params": {...}}
])

# 等待所有任務完成
orchestrator.wait_for_all(jobs)

# 選擇最佳模型
best_model = orchestrator.select_best_model(
    jobs,
    metric="f1_score"
)
```

**超參數搜索**:
```python
# 自動超參數搜索
search_job = orchestrator.hyperparameter_search(
    model="random_forest",
    param_space={
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    search_strategy="grid",  # or "random", "bayesian"
    metric="f1_score",
    cv=5
)
```

**訓練流水線**:
```python
# 定義訓練流水線
pipeline = orchestrator.create_pipeline([
    {"stage": "data_preprocessing", "script": "preprocess.py"},
    {"stage": "feature_engineering", "script": "features.py"},
    {"stage": "model_training", "script": "train.py"},
    {"stage": "model_evaluation", "script": "evaluate.py"},
    {"stage": "model_deployment", "script": "deploy.py"}
])

# 執行流水線
pipeline.run()
```

---

### distributed_trainer.py (633 行) ⭐

**職責**: 分布式訓練實現

**支持的分布式策略**:
- Data Parallelism (數據並行)
- Model Parallelism (模型並行)
- Distributed Data Parallel (DDP)
- Horovod

**使用範例**:
```python
from aiva_core.external_learning.training import DistributedTrainer

# 初始化分布式訓練
trainer = DistributedTrainer(
    backend="nccl",  # or "gloo", "mpi"
    num_gpus=4,
    strategy="ddp"
)

# 分布式訓練
trainer.train(
    model=model,
    train_loader=train_loader,
    epochs=100,
    checkpoint_interval=10
)
```

**多機訓練**:
```python
# 配置多機訓練
trainer = DistributedTrainer(
    backend="nccl",
    nodes=[
        {"host": "worker1", "gpus": [0, 1, 2, 3]},
        {"host": "worker2", "gpus": [0, 1, 2, 3]},
        {"host": "worker3", "gpus": [0, 1, 2, 3]}
    ],
    master="worker1:29500"
)

trainer.train(...)
```

**梯度累積**:
```python
# 使用梯度累積處理大批量數據
trainer = DistributedTrainer(
    accumulation_steps=4  # 累積 4 步後更新
)
```

## 🔄 訓練流程

```
任務提交
  ↓
資源分配 (orchestrator)
  ↓
數據分發 (distributed_trainer)
  ↓
並行訓練
  ↓
梯度同步
  ↓
模型更新
  ↓
Checkpoint 保存
  ↓
訓練完成
```

## 📊 訓練監控

```python
# 實時監控訓練
from aiva_core.external_learning.training import TrainingMonitor

monitor = TrainingMonitor()

# 監控資源使用
monitor.track_resource_usage(job.id)
# {"cpu": 75%, "memory": "6GB/8GB", "gpu": 90%}

# 監控訓練指標
monitor.track_metrics(job.id)
# {"epoch": 50, "loss": 0.3, "accuracy": 0.92}

# 可視化訓練進度
monitor.plot_training_progress(job.id)
```

## 🚀 性能優化

### 1. 自動混合精度 (AMP)

```python
trainer = DistributedTrainer(
    mixed_precision=True  # 啟用 FP16 訓練
)
```

### 2. 梯度檢查點

```python
trainer = DistributedTrainer(
    gradient_checkpointing=True  # 節省 GPU 內存
)
```

### 3. 數據加載優化

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 多進程數據加載
    pin_memory=True,  # 固定內存
    prefetch_factor=2  # 預取數據
)
```

## 🔧 容錯機制

```python
# 自動 Checkpoint 和恢復
orchestrator = TrainingOrchestrator(
    checkpoint_dir="/checkpoints",
    checkpoint_interval=10,  # 每 10 個 epoch
    auto_resume=True  # 自動恢復
)

# 訓練中斷後自動恢復
orchestrator.submit_training_job(..., resume_from_checkpoint=True)
```

## 📚 相關模組

- [learning](../learning/README.md) - 學習引擎
- [tracing](../tracing/README.md) - 訓練追蹤
- [service_backbone/messaging](../../service_backbone/messaging/README.md) - 任務分發

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準類型
from aiva_common import TaskStatus, AivaMessage

# ❌ 禁止：自定義訓練狀態
class TrainingJobStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

## 💡 最佳實踐

### 1. 訓練配置管理

```yaml
# training_config.yaml
model:
  type: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 20

data:
  train_path: data/train.csv
  val_path: data/val.csv
  batch_size: 32

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping:
    patience: 10
    metric: val_loss

resources:
  gpus: 2
  memory: 16GB
```

### 2. 分布式訓練啟動

```python
# 單機多卡
python train.py --distributed --gpus 4

# 多機多卡
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=3 \
    --node_rank=0 \
    --master_addr="worker1" \
    --master_port=29500 \
    train.py
```

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: External Learning 團隊
