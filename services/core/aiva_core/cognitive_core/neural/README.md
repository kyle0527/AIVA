# Neural 神經網路核心模組

> **路徑**: `cognitive_core/neural`  
> **狀態**: ✅ 正常 | **Python 文件數**: 5 | **最後更新**: 2026-04-05

## 概述

提供生物啟發的神經網路推理能力，包含 500 萬參數的 BioNeuron 核心。使用真實的 PyTorch 神經網路進行決策，支援權重持久化和訓練。

## 核心組件

### real_neural_core.py

- `RealAICore(nn.Module)` - 真實的 AI 核心神經網路
  - 5M 特化神經網路架構 (512 → 1600 → 1200 → 1024 → 512 → 100)
  - 支援權重載入和儲存
  - 使用 SentenceTransformer 進行語意編碼
- `RealDecisionEngine` - 真實決策引擎
  - 整合神經網路進行決策
  - 支援結構化輸入處理

### weight_manager.py

- `WeightMetadata` - 權重檔案元數據（版本、創建時間、哈希等）
- `AIWeightManager` - AI 權重管理器
  - 基於 PyTorch 官方最佳實踐
  - 自動備份和版本管理
  - 檔案完整性檢查

### real_bio_net_adapter.py

- `RealScalableBioNet` - 真實的 ScalableBioNet 向後相容適配器
  - 替換假 AI 核心，保持相同 API
  - 使用真實的 PyTorch 神經網路
- `RealBioNeuronRAGAgent` - 真實 AI 的 RAG 代理

### aiva_embedding.py

- 提供處理高維向量與語義 Embedding 的功能組件

### __init__.py

- 版本: `3.0.0-alpha`

## 依賴關係

- 內部依賴：
  - `learning_system.learning.model_trainer`
  - `learning_system.learning.scalable_bio_trainer`
  - `aiva_common.error_handling`
  - `aiva_common.enums`
- 外部依賴：`torch`, `numpy`, `sentence-transformers`

## 模型架構

```
輸入層 (512) → 隱藏層1 (1600) → 隱藏層2 (1200) → 隱藏層3 (1024) → 隱藏層4 (512) → 輸出層 (100)

總參數: ~5,000,000 (5M)
激活函數: ReLU
Dropout: 0.2
```

## 使用範例

```python
from cognitive_core.neural import RealDecisionEngine, AIWeightManager

# 初始化決策引擎
engine = RealDecisionEngine(use_5m_model=True)

# 進行決策
result = engine.decide(task_description="掃描目標網站")

# 權重管理
manager = AIWeightManager(base_dir="./weights")
filepath, metadata = manager.save_model_weights(model, "aiva_core")
```
