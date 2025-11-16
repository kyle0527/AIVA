# Tracing - 訓練追蹤

**導航**: [← 返回 External Learning](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [experiment_tracker.py](#experiment_trackerpy-342-行-)
  - [metrics_logger.py](#metrics_loggerpy-228-行)
  - [model_versioning.py](#model_versioningpy-127-行)
- [📊 實驗追蹤最佳實踐](#-實驗追蹤最佳實踐)
- [🔍 MLflow 集成](#-mlflow-集成)
- [📈 可視化](#-可視化)
- [📚 相關模組](#-相關模組)

---

## 📋 概述

**定位**: 訓練過程追蹤和實驗管理  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (697 行)

## 📂 文件結構

```
tracing/
├── experiment_tracker.py (342 行) ⭐ - 實驗追蹤
├── metrics_logger.py (228 行) - 指標記錄
├── model_versioning.py (127 行) - 模型版本管理
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### experiment_tracker.py (342 行) ⭐

**職責**: 追蹤和管理機器學習實驗

**功能**:
- 記錄超參數
- 追蹤指標
- 比較實驗結果
- 實驗可視化

**使用範例**:
```python
from aiva_core.external_learning.tracing import ExperimentTracker

tracker = ExperimentTracker(experiment_name="capability_classifier")

# 記錄超參數
tracker.log_params({
    "model": "random_forest",
    "n_estimators": 100,
    "max_depth": 20
})

# 記錄指標
tracker.log_metrics({
    "accuracy": 0.95,
    "f1_score": 0.93,
    "training_time": 120
})

# 記錄模型
tracker.log_model(model, "classifier_v1.pkl")

# 比較實驗
comparison = tracker.compare_experiments([
    "experiment_1",
    "experiment_2",
    "experiment_3"
])
```

**實驗管理**:
```python
# 列出所有實驗
experiments = tracker.list_experiments()

# 加載歷史實驗
exp = tracker.load_experiment("experiment_id_123")

# 恢復最佳模型
best_model = tracker.load_best_model(metric="f1_score")
```

---

### metrics_logger.py (228 行)

**職責**: 訓練指標實時記錄

**使用範例**:
```python
from aiva_core.external_learning.tracing import MetricsLogger

logger = MetricsLogger()

# 記錄每個 epoch 的指標
for epoch in range(100):
    loss = train_one_epoch()
    val_accuracy = validate()
    
    logger.log_metric("loss", loss, step=epoch)
    logger.log_metric("val_accuracy", val_accuracy, step=epoch)

# 可視化指標
logger.plot_metrics()

# 導出指標
logger.export_to_csv("training_metrics.csv")
```

**支持的後端**:
- MLflow
- TensorBoard
- Weights & Biases
- 本地文件系統

---

### model_versioning.py (127 行)

**職責**: 模型版本控制

**使用範例**:
```python
from aiva_core.external_learning.tracing import ModelVersioning

versioning = ModelVersioning(model_name="capability_classifier")

# 保存新版本
versioning.save_version(
    model=trained_model,
    version="v1.2.0",
    metadata={
        "training_data": "dataset_2025_01",
        "accuracy": 0.96,
        "notes": "增加了新的能力類別"
    }
)

# 列出所有版本
versions = versioning.list_versions()
# [
#   {"version": "v1.0.0", "date": "2025-01-01", "accuracy": 0.92},
#   {"version": "v1.1.0", "date": "2025-02-15", "accuracy": 0.94},
#   {"version": "v1.2.0", "date": "2025-03-20", "accuracy": 0.96}
# ]

# 加載特定版本
model_v1 = versioning.load_version("v1.0.0")

# 回滾到上一版本
versioning.rollback()
```

## 📊 實驗追蹤最佳實踐

### 1. 完整記錄

```python
tracker = ExperimentTracker("my_experiment")

# 記錄環境信息
tracker.log_system_info()

# 記錄數據集
tracker.log_dataset_info({
    "name": "training_data_2025",
    "size": 10000,
    "features": 50
})

# 記錄超參數
tracker.log_params({
    "model": "random_forest",
    "n_estimators": 100,
    "max_depth": 20,
    "learning_rate": 0.01
})

# 記錄指標
tracker.log_metrics({
    "train_accuracy": 0.98,
    "val_accuracy": 0.95,
    "test_accuracy": 0.94
})
```

### 2. 自動追蹤

```python
# 使用裝飾器自動追蹤
@tracker.track_experiment
def train_model(params):
    model = create_model(params)
    model.train(...)
    return model, metrics
```

### 3. 實驗對比

```python
# 對比多個實驗
comparison = tracker.compare_experiments(
    experiment_ids=["exp1", "exp2", "exp3"],
    metrics=["accuracy", "f1_score", "training_time"]
)

# 生成對比報告
tracker.generate_comparison_report(comparison, output="report.html")
```

## 🔍 MLflow 集成

```python
import mlflow
from aiva_core.external_learning.tracing import ExperimentTracker

# 使用 MLflow 後端
tracker = ExperimentTracker(
    backend="mlflow",
    tracking_uri="http://mlflow-server:5000"
)

with mlflow.start_run():
    tracker.log_params(params)
    tracker.log_metrics(metrics)
    tracker.log_model(model)
```

## 📈 可視化

```python
# 訓練曲線
tracker.plot_training_curve(metric="loss")

# 指標對比
tracker.plot_metric_comparison(
    experiments=["exp1", "exp2"],
    metric="accuracy"
)

# 超參數重要性
tracker.plot_hyperparameter_importance()
```

## 📚 相關模組

- [learning](../learning/README.md) - 學習引擎
- [training](../training/README.md) - 訓練編排

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準類型
from aiva_common import TaskStatus

# ❌ 禁止：自定義追蹤狀態
class TraceStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: External Learning 團隊
