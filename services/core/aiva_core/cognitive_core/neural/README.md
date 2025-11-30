# 🧠 Neural - 神經網路核心

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心功能](#核心功能)
- [📂 檔案列表](#-檔案列表)
- [🔧 核心組件](#-核心組件)
  - [1. `real_neural_core.py` - 神經網路核心](#1-real_neural_corepy---神經網路核心)
  - [2. `real_bio_net_adapter.py` - 生物網路適配器](#2-real_bio_net_adapterpy---生物網路適配器)
  - [3. `bio_neuron_master.py` - 主控系統](#3-bio_neuron_masterpy---主控系統)
  - [4. `ai_model_manager.py` - AI 模型管理器](#4-ai_model_managerpy---ai-模型管理器)
  - [5. `neural_network.py` - 神經網路基礎](#5-neural_networkpy---神經網路基礎)
  - [6. `weight_manager.py` - 權重管理系統](#6-weight_managerpy---權重管理系統)
- [🚀 完整使用流程](#-完整使用流程)
  - [初始化和推理](#初始化和推理)
  - [模型訓練和更新](#模型訓練和更新)
- [📊 性能指標](#-性能指標)
- [🔗 相關模組](#-相關模組)

---

**導航**: [← 返回 Cognitive Core](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: v2.1.2  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2025-12-20  
> **角色**: BioNeuron 神經網路推理和模型管理

---

## 🎯 模組概述

Neural 子模組實現了 AIVA 的生物啟發神經網路核心，包含 500萬參數的 BioNeuron 模型、模型管理系統、權重管理、以及三模式主控系統（UI/AI/Chat）。

### 核心功能
- **神經網路推理** - 500萬參數生物啟發架構
- **模型管理** - 統一的 AI 模型載入和訓練協調
- **權重管理** - 安全的權重持久化和版本控制
- **主控系統** - 支援三種操作模式的統一調度
- **適配器層** - 生物網路與 AIVA 系統的橋接

---

## 📂 檔案列表

| 檔案 | 行數 | 功能 | 狀態 |
|------|------|------|------|
| `real_neural_core.py` | ~800 | 500萬參數神經網路核心 | ✅ |
| `real_bio_net_adapter.py` | ~600 | 生物神經網路適配器 | ✅ |
| `bio_neuron_master.py` | 1462 | BioNeuronRAGAgent 主控系統 | ✅ |
| `ai_model_manager.py` | 735 | AI 模型統一管理器 | ✅ |
| `neural_network.py` | ~400 | 神經網路基礎架構 | ✅ |
| `weight_manager.py` | 453 | 權重管理系統 | ✅ |
| `__init__.py` | ~50 | 模組入口 | ✅ |

**總計**: 7 個 Python 檔案，約 4500+ 行代碼

---

## 🔧 核心組件

### 1. `real_neural_core.py` - 神經網路核心

**功能**: 500萬參數 BioNeuron 神經網路的核心實現

**架構特性**:
```python
BioNeuronCore (5M 參數)
├── Input Layer (128 neurons)
├── Hidden Layers (生物啟發架構)
│   ├── Excitatory neurons (興奮性神經元)
│   ├── Inhibitory neurons (抑制性神經元)
│   └── Neuromodulation (神經調節)
└── Output Layer (決策輸出)
```

**使用範例**:
```python
from aiva_core.cognitive_core.neural import RealNeuralCore

# 初始化神經網路
core = RealNeuralCore(model_path="./weights/bioneuron_5m.pth")

# 推理
input_vector = torch.tensor([...])  # 128維輸入
output = await core.forward(input_vector)

# 批次推理
batch_output = await core.batch_forward(batch_inputs)
```

**關鍵方法**:
- `forward()` - 單次推理
- `batch_forward()` - 批次推理
- `load_weights()` - 載入預訓練權重
- `get_activations()` - 獲取中間層激活

---

### 2. `real_bio_net_adapter.py` - 生物網路適配器

**功能**: 將 BioNeuron 神經網路適配到 AIVA 系統

**適配層職責**:
- 輸入預處理和特徵提取
- 輸出後處理和解釋
- 錯誤處理和容錯機制
- 性能監控和日誌記錄

**使用範例**:
```python
from aiva_core.cognitive_core.neural import RealBioNetAdapter

adapter = RealBioNetAdapter(neural_core)

# 適配層推理
result = await adapter.predict(
    task_description="執行SQL注入測試",
    context={"target": "https://example.com"}
)

# 結果包含
# - decision: 決策結果
# - confidence: 置信度
# - reasoning: 推理過程
```

**關鍵方法**:
- `predict()` - 端到端預測
- `preprocess_input()` - 輸入預處理
- `postprocess_output()` - 輸出後處理
- `explain_decision()` - 決策解釋

---

### 3. `bio_neuron_master.py` - 主控系統

**功能**: BioNeuronRAGAgent 的主控制器，支援三種操作模式

**架構**:
```
┌─────────────────────────────────────────┐
│      BioNeuronRAGAgent (主腦)           │
│  - 決策核心 (500萬參數神經網路)          │
│  - RAG 知識檢索                          │
│  - 抗幻覺機制                            │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐   ┌───▼────┐
│UI Mode│   │AI Mode  │   │Chat Mode│
│ 介面  │   │ 自主    │   │ 對話   │
└───────┘   └─────────┘   └────────┘
```

**三種模式**:

#### UI Mode (介面模式)
- 用戶通過圖形介面控制
- 手動選擇測試項目
- 人工審核決策
```python
master = BioNeuronMaster(mode="ui")
result = await master.execute_with_ui_approval(task)
```

#### AI Mode (自主模式)
- 完全自主決策和執行
- 自動選擇最佳策略
- 無需人工干預
```python
master = BioNeuronMaster(mode="ai")
result = await master.autonomous_execute(task)
```

#### Chat Mode (對話模式)
- 自然語言交互
- 問答式引導
- 教學和演示
```python
master = BioNeuronMaster(mode="chat")
response = await master.chat("如何測試SQL注入？")
```

**使用範例**:
```python
from aiva_core.cognitive_core.neural import BioNeuronMaster

# 初始化（自動選擇模式）
master = BioNeuronMaster(
    mode="ai",
    rag_engine=rag,
    knowledge_base=kb
)

# 處理請求
result = await master.process_request({
    "task": "執行全面安全測試",
    "target": "https://example.com",
    "depth": "deep"
})

# 結果
print(f"決策: {result.decision}")
print(f"置信度: {result.confidence}%")
print(f"推理: {result.reasoning}")
```

**關鍵方法**:
- `process_request()` - 統一請求處理
- `switch_mode()` - 動態切換模式
- `get_decision()` - 獲取 AI 決策
- `execute_with_rag()` - RAG 增強執行

---

### 4. `ai_model_manager.py` - AI 模型管理器

**功能**: 統一管理所有 AI 模型和訓練系統

**管理範圍**:
- BioNeuron 神經網路模型
- 訓練系統協調
- 模型版本控制
- 性能監控

**使用範例**:
```python
from aiva_core.cognitive_core.neural import AIModelManager

manager = AIModelManager()

# 載入模型
model = await manager.load_model(
    model_id="bioneuron-v1",
    device="cuda"
)

# 訓練協調
await manager.coordinate_training(
    trainer=model_trainer,
    config=training_config
)

# 模型評估
metrics = await manager.evaluate_model(
    model_id="bioneuron-v1",
    test_data=test_dataset
)

# 版本管理
await manager.save_model_version(
    model=model,
    version="v1.1",
    notes="修復過擬合問題"
)
```

**關鍵方法**:
- `load_model()` - 載入模型
- `save_model()` - 保存模型
- `coordinate_training()` - 協調訓練
- `evaluate_model()` - 模型評估
- `list_versions()` - 列出所有版本
- `rollback_version()` - 回滾版本

**整合**:
```python
# 與 external_learning 整合
from services.core.aiva_core.external_learning.learning import ModelTrainer

model_manager = AIModelManager()
trainer = ModelTrainer()

# 協調訓練流程
await model_manager.coordinate_training(
    trainer=trainer,
    experiences=training_experiences
)
```

---

### 5. `neural_network.py` - 神經網路基礎

**功能**: 提供可復用的神經網路組件和基礎架構

**組件**:
- 基礎層 (Linear, Conv, Attention)
- 激活函數 (ReLU, GELU, Sigmoid)
- 標準化層 (BatchNorm, LayerNorm)
- Dropout 和正則化
- 自定義生物啟發組件

**使用範例**:
```python
from aiva_core.cognitive_core.neural.neural_network import (
    BiologicalNeuron,
    NeuromodulationLayer,
    SynapticPlasticity
)

# 生物啟發神經元
bio_neuron = BiologicalNeuron(
    input_dim=128,
    output_dim=256,
    neuron_type="excitatory"
)

# 神經調節層
neuromod = NeuromodulationLayer(
    modulator="dopamine",
    target_layer=bio_neuron
)

# 突觸可塑性
plasticity = SynapticPlasticity(
    learning_rule="hebbian"
)
```

---

### 6. `weight_manager.py` - 權重管理系統

**功能**: 安全的模型權重持久化和版本管理

**特性**:
- ✅ 自動載入和儲存
- ✅ 檔案完整性檢查 (SHA-256)
- ✅ 權重版本管理
- ✅ 錯誤處理和容錯
- ✅ 安全的序列化/反序列化

**使用範例**:
```python
from aiva_core.cognitive_core.neural import WeightManager

manager = WeightManager(weights_dir="./weights")

# 保存權重
await manager.save_weights(
    model=model,
    name="bioneuron_5m",
    metadata={
        "version": "1.0",
        "accuracy": 0.95,
        "training_date": "2025-11-16"
    }
)

# 載入權重
weights, metadata = await manager.load_weights(
    name="bioneuron_5m",
    verify_integrity=True
)

# 列出所有權重
versions = manager.list_weights()
for v in versions:
    print(f"{v.name} - {v.version} ({v.size_mb:.2f}MB)")

# 驗證完整性
is_valid = await manager.verify_integrity(
    weight_file="bioneuron_5m.pth"
)
```

**關鍵方法**:
- `save_weights()` - 保存權重
- `load_weights()` - 載入權重
- `verify_integrity()` - 驗證完整性
- `list_weights()` - 列出所有版本
- `delete_weights()` - 刪除權重
- `backup_weights()` - 備份權重

**權重元數據**:
```python
@dataclass
class WeightMetadata:
    name: str
    version: str
    sha256: str
    size_bytes: int
    created_at: datetime
    accuracy: float
    loss: float
    epochs: int
    notes: str
```

---

## 🚀 完整使用流程

### 初始化和推理
```python
from aiva_core.cognitive_core.neural import (
    BioNeuronMaster,
    AIModelManager,
    WeightManager
)

# 1. 初始化權重管理器
weight_manager = WeightManager(weights_dir="./weights")

# 2. 初始化模型管理器
model_manager = AIModelManager()

# 3. 載入模型
model = await model_manager.load_model(
    model_id="bioneuron-v1",
    weight_manager=weight_manager
)

# 4. 初始化主控系統
master = BioNeuronMaster(
    mode="ai",
    model=model,
    rag_engine=rag_engine
)

# 5. 執行推理
result = await master.process_request({
    "task": "執行SQL注入測試",
    "target": "https://example.com"
})

print(f"決策: {result.decision}")
print(f"置信度: {result.confidence}%")
```

### 模型訓練和更新
```python
from aiva_core.cognitive_core.neural import AIModelManager
from services.core.aiva_core.external_learning.learning import ModelTrainer

# 初始化
manager = AIModelManager()
trainer = ModelTrainer()

# 收集訓練數據
experiences = collect_training_experiences()

# 訓練新版本
new_model = await manager.coordinate_training(
    trainer=trainer,
    experiences=experiences,
    config={
        "learning_rate": 0.001,
        "epochs": 10
    }
)

# 評估性能
metrics = await manager.evaluate_model(
    model=new_model,
    test_data=test_dataset
)

# 如果性能提升，保存新版本
if metrics["accuracy"] > 0.95:
    await manager.save_model_version(
        model=new_model,
        version="v1.2",
        notes="提升準確率到95%"
    )
```

---

## 📊 性能指標

| 指標 | 數值 | 備註 |
|------|------|------|
| 模型參數 | 5,000,000 | BioNeuron 神經網路 |
| 推理速度 | < 50ms | 單次推理 |
| 批次推理 | 1000+ samples/s | batch_size=32 |
| 記憶體使用 | ~200MB | 模型載入後 |
| 準確率 | 90%+ | 測試集 |
| GPU 加速 | ✅ 支援 | CUDA/MPS |

---

## 🔗 相關模組

- **[rag](../rag/README.md)** - 提供 RAG 知識增強
- **[decision](../decision/README.md)** - 決策結果輸入
- **[external_learning](../../external_learning/README.md)** - 模型訓練系統

---

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team
