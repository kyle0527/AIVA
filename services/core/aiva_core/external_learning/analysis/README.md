# Analysis - 分析工具集

**導航**: [← 返回 External Learning](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [feature_analyzer.py](#feature_analyzerpy-310-行-)
  - [pattern_detector.py](#pattern_detectorpy-245-行)
  - [data_preprocessor.py](#data_preprocessorpy-147-行)
- [📚 相關模組](#-相關模組)

---

## 📋 概述

**定位**: 數據分析和模式識別  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (702 行)

## 📂 文件結構

```
analysis/
├── feature_analyzer.py (310 行) ⭐ - 特徵分析
├── pattern_detector.py (245 行) - 模式檢測
├── data_preprocessor.py (147 行) - 數據預處理
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### feature_analyzer.py (310 行) ⭐

**職責**: 特徵提取和重要性分析

**使用範例**:
```python
from aiva_core.external_learning.analysis import feature_analyzer

# 分析特徵重要性
analyzer = feature_analyzer.FeatureAnalyzer()
importance = analyzer.analyze_importance(
    X=training_data,
    y=labels,
    method="random_forest"
)

# 輸出:
# {
#   "feature1": 0.35,
#   "feature2": 0.28,
#   "feature3": 0.15,
#   ...
# }
```

---

### pattern_detector.py (245 行)

**職責**: 檢測數據中的模式和異常

**使用範例**:
```python
from aiva_core.external_learning.analysis import pattern_detector

detector = pattern_detector.PatternDetector()

# 檢測異常
anomalies = detector.detect_anomalies(data, threshold=0.95)

# 檢測模式
patterns = detector.detect_patterns(data, min_support=0.1)
```

---

### data_preprocessor.py (147 行)

**職責**: 數據清洗和預處理

**使用範例**:
```python
from aiva_core.external_learning.analysis import data_preprocessor

preprocessor = data_preprocessor.DataPreprocessor()

# 清洗數據
cleaned_data = preprocessor.clean(
    data,
    remove_duplicates=True,
    handle_missing="interpolate",
    normalize=True
)
```

## 📚 相關模組

- [learning](../learning/README.md) - 學習引擎
- [ai_model](../ai_model/README.md) - 模型訓練

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準類型
from aiva_common import TaskStatus
from typing import Dict, Any

# ❌ 禁止：自定義分析狀態
class AnalysisStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: External Learning 團隊
