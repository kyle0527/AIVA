# AI Model - AI 模型訓練

**導航**: [← 返回 External Learning](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [train_classifier.py](#train_classifierpy-184-行)
- [🔍 應用場景](#-應用場景)
- [📚 相關模組](#-相關模組)

---

## 📋 概述

**定位**: AI 分類器訓練  
**狀態**: ✅ 已實現  
**文件數**: 1 個 Python 文件 (184 行)

## 📂 文件結構

```
ai_model/
├── train_classifier.py (184 行) - 分類器訓練腳本
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### train_classifier.py (184 行)

**職責**: 訓練 AI 分類模型 (用於能力分類、威脅識別等)

**支持的模型**:
- Scikit-learn 分類器 (Random Forest, SVM, Naive Bayes)
- 神經網絡分類器 (TensorFlow/PyTorch)
- 預訓練模型微調 (BERT, RoBERTa)

**使用範例**:
```python
from aiva_core.external_learning.ai_model import train_classifier

# 訓練分類器
model = train_classifier.train(
    data_path="training_data.csv",
    model_type="random_forest",
    target_column="category",
    features=["feature1", "feature2", "feature3"]
)

# 保存模型
model.save("classifier_model.pkl")

# 推理
predictions = model.predict(["new_sample_1", "new_sample_2"])
```

## 🔍 應用場景

### 1. 能力分類
```python
# 訓練能力分類器
classifier = train_classifier.train(
    data_path="capabilities.csv",
    model_type="bert",
    target_column="category",  # web_scan, network_scan, etc.
    features=["capability_name", "description"]
)
```

### 2. 威脅識別
```python
# 訓練威脅分類器
threat_classifier = train_classifier.train(
    data_path="threats.csv",
    model_type="svm",
    target_column="threat_level",  # low, medium, high, critical
    features=["indicator", "pattern", "context"]
)
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

# ❌ 禁止：自定義模型狀態
class ModelStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: External Learning 團隊
