# 方案 D：ONNX + TensorRT (產業標準推理優化)

## 📋 目錄

- [📋 執行摘要](#執行摘要)
- [🎯 方案概述](#方案概述)
  - [核心目標](#核心目標)
  - [技術架構](#技術架構)
- [📊 技術規格](#技術規格)
  - [ONNX 模型規格](#onnx-模型規格)
  - [TensorRT 優化選項](#tensorrt-優化選項)
  - [性能對比](#性能對比)
- [🔧 實施計畫](#實施計畫)
  - [階段 1：訓練與導出 (3 天)](#階段-1訓練與導出-3-天)
  - [階段 2：TensorRT 優化 (5 天)](#階段-2tensorrt-優化-5-天)
  - [階段 3：AIVA 整合 (5 天)](#階段-3aiva-整合-5-天)
- [📈 預期成果](#預期成果)
  - [性能提升](#性能提升)
  - [部署大小](#部署大小)
  - [GPU 加速效果](#gpu-加速效果)
- [💰 成本分析](#成本分析)
  - [硬體成本](#硬體成本)
  - [開發成本](#開發成本)
  - [授權成本](#授權成本)
- [⚠️ 風險評估](#風險評估)
  - [技術風險](#技術風險)
  - [部署風險](#部署風險)
- [🎯 成功標準](#成功標準)
  - [必須達成](#必須達成)
  - [期望達成](#期望達成)
  - [最好達成](#最好達成)
- [✅ 結論與建議](#結論與建議)
  - [核心優勢](#核心優勢)
  - [核心劣勢](#核心劣勢)
  - [適用場景](#適用場景)
  - [不適用場景](#不適用場景)
  - [最終建議](#最終建議)

## 📋 執行摘要

**核心策略**：使用 ONNX 作為模型交換格式,在 Python 訓練後導出,通過 TensorRT 進行推理優化,獲得 GPU 加速與產業級性能。

**開發時間**：2-3 週  
**部署時間**：1 週  
**預估成本**：中高（GPU 硬體 + 授權考量）  
**風險等級**：⭐⭐⭐ 中等

---

## 🎯 方案概述

### 核心目標

將訓練與推理分離,使用產業標準工具鏈:

```
訓練階段：Python + PyTorch/TensorFlow
    ↓ 導出
中間格式：ONNX (開放神經網路交換格式)
    ↓ 優化
推理引擎：TensorRT (NVIDIA GPU 加速)
```

### 技術架構

```
┌─────────────────────────────────────────────────────┐
│         訓練階段 (離線,一次性)                        │
│  ┌──────────────────────────────────────────────┐  │
│  │  Python + PyTorch                             │  │
│  │                                                │  │
│  │  class AIVANet(nn.Module):                    │  │
│  │      def __init__(self):                      │  │
│  │          self.fc1 = nn.Linear(512, 2048)      │  │
│  │          self.fc2 = nn.Linear(2048, 1024)     │  │
│  │          self.fc3 = nn.Linear(1024, 20)       │  │
│  │                                                │  │
│  │  # 訓練數據收集與訓練                         │  │
│  │  for x, y in dataloader:                      │  │
│  │      loss = train_step(x, y)                  │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓ torch.onnx.export()              │
│  ┌──────────────────────────────────────────────┐  │
│  │  ONNX 模型 (model.onnx)                       │  │
│  │  - 權重: 24 MB                                │  │
│  │  - 格式: Protocol Buffers                     │  │
│  │  - 平台無關                                   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────┐
│         推理階段 (在線,高性能)                        │
│  ┌──────────────────────────────────────────────┐  │
│  │  TensorRT 優化器                              │  │
│  │  - 層融合 (Layer Fusion)                      │  │
│  │  - 量化 (INT8/FP16)                           │  │
│  │  - 核心自動調優                               │  │
│  │  - 動態 Batch 處理                            │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓                                   │
│  ┌──────────────────────────────────────────────┐  │
│  │  TensorRT Engine (.trt)                       │  │
│  │  - 優化後權重: ~10 MB (INT8)                  │  │
│  │  - GPU 專用二進制                             │  │
│  │  - 推理延遲: 0.1 ms (GPU)                     │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓                                   │
│  ┌──────────────────────────────────────────────┐  │
│  │  Python Bindings (pycuda)                     │  │
│  │  def predict(features):                       │  │
│  │      return trt_engine.infer(features)        │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📊 技術規格

### ONNX 模型規格

| 屬性 | 值 |
|------|-----|
| **格式版本** | ONNX Opset 17 |
| **輸入形狀** | [batch_size, 512] |
| **輸出形狀** | [batch_size, 20] |
| **參數數量** | 3,166,208 |
| **權重大小** | 24 MB (FP32) |
| **算子支持** | Linear, Tanh, Softmax |

### TensorRT 優化選項

| 優化 | FP32 | FP16 | INT8 |
|------|------|------|------|
| **精度** | 100% | ~99.5% | ~98% |
| **大小** | 24 MB | 12 MB | 6 MB |
| **速度** | 1x | 2x | 4x |
| **GPU 要求** | GTX 1050+ | GTX 1060+ | GTX 1080+ |

### 性能對比

| 指標 | Python | ONNX Runtime | TensorRT FP32 | TensorRT INT8 |
|------|--------|--------------|---------------|---------------|
| **推理延遲** | 0.5 ms | 0.3 ms | 0.1 ms | **0.05 ms** |
| **吞吐量** | 2K/s | 3K/s | 10K/s | **20K/s** |
| **GPU 利用率** | 20% | 40% | 70% | **90%** |
| **批次大小** | 1 | 1-32 | 1-128 | 1-256 |

---

## 🔧 實施計畫

### 階段 1：訓練與導出 (3 天)

**任務 1.1：PyTorch 訓練腳本**
```python
# scripts/train_for_onnx.py

import torch
import torch.nn as nn
import torch.onnx

class AIVANet(nn.Module):
    """ONNX 兼容的 AIVA 網路"""
    
    def __init__(self, input_size=512, num_tools=20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_tools)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

def train_model():
    """訓練並保存模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AIVANet().to(device)
    
    # 訓練循環 (省略細節)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 保存 PyTorch 權重
    torch.save(model.state_dict(), 'models/aiva_trained.pth')
    
    return model

if __name__ == '__main__':
    model = train_model()
    print("訓練完成")
```

**任務 1.2：導出到 ONNX**
```python
# scripts/export_to_onnx.py

import torch
import torch.onnx
from train_for_onnx import AIVANet

def export_onnx():
    """將 PyTorch 模型導出為 ONNX"""
    
    # 載入訓練好的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AIVANet().to(device)
    model.load_state_dict(torch.load('models/aiva_trained.pth'))
    model.eval()
    
    # 準備虛擬輸入
    dummy_input = torch.randn(1, 512, device=device)
    
    # 導出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        'models/aiva.onnx',
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['probabilities'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'probabilities': {0: 'batch_size'}
        }
    )
    
    print("ONNX 導出成功: models/aiva.onnx")
    
    # 驗證 ONNX 模型
    import onnx
    onnx_model = onnx.load('models/aiva.onnx')
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型驗證通過")

if __name__ == '__main__':
    export_onnx()
```

**任務 1.3：ONNX Runtime 測試**
```python
# scripts/test_onnx_runtime.py

import numpy as np
import onnxruntime as ort

def test_onnx_inference():
    """測試 ONNX Runtime 推理"""
    
    # 創建推理會話
    session = ort.InferenceSession(
        'models/aiva.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # 準備輸入
    features = np.random.randn(1, 512).astype(np.float32)
    
    # 推理
    outputs = session.run(
        ['probabilities'],
        {'features': features}
    )
    
    probs = outputs[0][0]
    print(f"輸出形狀: {probs.shape}")
    print(f"機率和: {probs.sum():.4f}")
    print(f"最高機率工具: {probs.argmax()}")
    
    # 性能測試
    import time
    n_iter = 10000
    start = time.perf_counter()
    for _ in range(n_iter):
        session.run(['probabilities'], {'features': features})
    elapsed = time.perf_counter() - start
    
    print(f"平均推理時間: {elapsed/n_iter*1000:.3f} ms")
    print(f"吞吐量: {n_iter/elapsed:.0f} 次/秒")

if __name__ == '__main__':
    test_onnx_inference()
```

### 階段 2：TensorRT 優化 (5 天)

**任務 2.1：安裝 TensorRT**
```bash
# 下載 TensorRT (需要 NVIDIA 帳號)
# https://developer.nvidia.com/tensorrt

# Windows 安裝
# 1. 解壓到 C:\TensorRT-8.6.1
# 2. 設置環境變數
$env:Path += ";C:\TensorRT-8.6.1\lib"
$env:TENSORRT_DIR = "C:\TensorRT-8.6.1"

# 安裝 Python 綁定
pip install tensorrt

# 驗證安裝
python -c "import tensorrt as trt; print(trt.__version__)"
```

**任務 2.2：ONNX → TensorRT 轉換**
```python
# scripts/convert_to_tensorrt.py

import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=False, int8=False):
    """將 ONNX 模型轉換為 TensorRT 引擎"""
    
    # 創建 Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析 ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX 解析失敗")
    
    # 配置構建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    
    # 精度設置
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("啟用 FP16 精度")
    
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 需要校準數據 (省略實現)
        print("啟用 INT8 精度")
    
    # 設置動態形狀 (可選)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'features',
        min=(1, 512),
        opt=(16, 512),
        max=(128, 512)
    )
    config.add_optimization_profile(profile)
    
    # 構建引擎
    print("開始構建 TensorRT 引擎...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT 引擎已保存: {engine_path}")

if __name__ == '__main__':
    build_engine(
        'models/aiva.onnx',
        'models/aiva_fp32.trt',
        fp16=False,
        int8=False
    )
    
    build_engine(
        'models/aiva.onnx',
        'models/aiva_fp16.trt',
        fp16=True,
        int8=False
    )
```

**任務 2.3：TensorRT 推理包裝器**
```python
# aiva_bindings/tensorrt_wrapper.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTEngine:
    """TensorRT 推理引擎包裝器"""
    
    def __init__(self, engine_path):
        # 載入引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 準備輸入輸出緩衝區
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配 GPU 內存
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    'host': None,
                    'device': device_mem,
                    'size': size,
                    'dtype': dtype
                })
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'size': size,
                    'dtype': dtype
                })
    
    def infer(self, features: np.ndarray) -> np.ndarray:
        """執行推理"""
        # 將輸入拷貝到 GPU
        input_data = features.astype(self.inputs[0]['dtype']).ravel()
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            input_data,
            self.stream
        )
        
        # 執行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 將輸出拷貝回 CPU
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream
            )
        
        # 同步
        self.stream.synchronize()
        
        # 返回結果
        return self.outputs[0]['host'].copy()
    
    def __del__(self):
        """清理資源"""
        for inp in self.inputs:
            if inp['device']:
                inp['device'].free()
        for out in self.outputs:
            if out['device']:
                out['device'].free()
```

### 階段 3：AIVA 整合 (5 天)

**任務 3.1：核心選擇邏輯**
```python
# services/core/aiva_core/core.py

class AIVACore:
    def __init__(self, engine_type='python'):
        """
        engine_type: 'python', 'onnx', 'tensorrt'
        """
        self.engine_type = engine_type
        
        if engine_type == 'python':
            from .ai_engine.bio_neuron_core import ScalableBioNet
            self.engine = ScalableBioNet(512, 20)
            logger.info("使用 Python BioNeuron 核心")
        
        elif engine_type == 'onnx':
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                'models/aiva.onnx',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            logger.info("使用 ONNX Runtime")
        
        elif engine_type == 'tensorrt':
            from aiva_bindings.tensorrt_wrapper import TensorRTEngine
            self.engine = TensorRTEngine('models/aiva_fp16.trt')
            logger.info("使用 TensorRT FP16 引擎")
        
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
    
    def select_tool(self, scan_result: dict) -> str:
        """選擇工具"""
        features = self.feature_extractor.extract(scan_result)
        
        if self.engine_type == 'python':
            probs = self.engine.forward(features)
        
        elif self.engine_type == 'onnx':
            features_np = features.astype(np.float32).reshape(1, -1)
            outputs = self.session.run(['probabilities'], {'features': features_np})
            probs = outputs[0][0]
        
        elif self.engine_type == 'tensorrt':
            features_np = features.astype(np.float32).reshape(1, -1)
            probs = self.engine.infer(features_np)
        
        tool_index = np.argmax(probs)
        return self.tools[tool_index]
```

**任務 3.2：配置管理**
```yaml
# config/ai_core.yaml

ai_core:
  # 引擎類型: python, onnx, tensorrt
  engine: tensorrt
  
  # ONNX 配置
  onnx:
    model_path: models/aiva.onnx
    providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
  
  # TensorRT 配置
  tensorrt:
    engine_path: models/aiva_fp16.trt
    precision: fp16  # fp32, fp16, int8
    max_batch_size: 32
    workspace_size: 1073741824  # 1 GB
```

---

## 📈 預期成果

### 性能提升

```
推理延遲對比：
Python:     ████████████████████ 0.5 ms
ONNX:       ████████████ 0.3 ms
TRT FP32:   ████ 0.1 ms
TRT FP16:   ██ 0.07 ms
TRT INT8:   █ 0.05 ms  ← 10x 加速
```

### 部署大小

| 格式 | 大小 | GPU 需求 |
|------|------|----------|
| **PyTorch (.pth)** | 24 MB | 可選 |
| **ONNX (.onnx)** | 24 MB | 可選 |
| **TensorRT FP32** | 24 MB | 必須 |
| **TensorRT FP16** | 12 MB | 必須 |
| **TensorRT INT8** | 6 MB | 必須 |

### GPU 加速效果

```
吞吐量 (次/秒):
CPU (Python):   ██ 2,000
CPU (ONNX):     ███ 3,000
GPU (TRT FP32): ██████████ 10,000
GPU (TRT INT8): ████████████████████ 20,000
```

---

## 💰 成本分析

### 硬體成本

| GPU 型號 | 價格 | FP32 | FP16 | INT8 |
|----------|------|------|------|------|
| **GTX 1650** | $150 | ✅ | ⚠️ | ❌ |
| **RTX 3060** | $330 | ✅ | ✅ | ⚠️ |
| **RTX 4070** | $600 | ✅ | ✅ | ✅ |
| **A100 (雲)** | $2/hr | ✅ | ✅ | ✅ |

### 開發成本

| 階段 | 工時 | 技能需求 | 成本 |
|------|------|----------|------|
| 訓練與導出 | 3 天 | PyTorch | 低 |
| TensorRT 轉換 | 5 天 | CUDA/TRT | 中 |
| AIVA 整合 | 5 天 | Python | 低 |
| 優化調試 | 2-5 天 | GPU 調優 | 中 |
| **總計** | **15-18 天** | **多技能** | **中** |

### 授權成本

| 組件 | 授權 | 商用 |
|------|------|------|
| **ONNX** | Apache-2.0 | ✅ 免費 |
| **ONNX Runtime** | MIT | ✅ 免費 |
| **TensorRT** | NVIDIA EULA | ⚠️ 有限制 |

**TensorRT 授權限制**：
- 開發/測試：免費
- 商業部署：需評估具體使用場景
- 雲端部署：通常已包含在 GPU 實例授權中

---

## ⚠️ 風險評估

### 技術風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **GPU 依賴** | 高 | 中 | 提供 CPU 後備 (ONNX) |
| **TensorRT 版本兼容** | 中 | 中 | 鎖定特定版本 |
| **量化精度損失** | 中 | 低 | 充分測試 INT8 |
| **CUDA 環境複雜** | 中 | 中 | Docker 容器化 |
| **授權合規** | 低 | 高 | 法務審查 |

### 部署風險

```
如果客戶環境無 GPU：
- 回退到 ONNX Runtime (CPU)
- 性能下降 3x，但功能正常
```

---

## 🎯 成功標準

### 必須達成

- ✅ ONNX 模型正確導出
- ✅ TensorRT 引擎成功構建
- ✅ 推理延遲 < 0.2 ms (GPU)
- ✅ GPU 利用率 > 70%
- ✅ CPU 後備可用

### 期望達成

- ✅ INT8 量化準確率 > 95%
- ✅ 支持動態 Batch
- ✅ 跨平台 ONNX 部署
- ✅ Docker 一鍵部署

### 最好達成

- ✅ 多 GPU 並行
- ✅ 模型熱更新
- ✅ 推理延遲 < 0.1 ms
- ✅ AMD GPU 支持 (ROCm)

---

## ✅ 結論與建議

### 核心優勢

1. **產業標準**：ONNX 生態成熟
2. **極致性能**：GPU 加速 10x+
3. **靈活部署**：ONNX 跨平台
4. **訓練分離**：Python 訓練，優化推理
5. **成熟工具鏈**：NVIDIA 官方支持

### 核心劣勢

1. **GPU 依賴**：TensorRT 需要 NVIDIA GPU
2. **環境複雜**：CUDA/cuDNN/TensorRT 安裝
3. **授權考量**：TensorRT 商用需評估
4. **調試困難**：GPU 錯誤難追蹤
5. **成本增加**：硬體投資

### 適用場景

✅ **大規模推理部署**  
✅ 已有 NVIDIA GPU 環境  
✅ 追求極致推理性能  
✅ 需要跨平台模型交換  
✅ 訓練與推理分離架構  

### 不適用場景

❌ **無 GPU 環境**  
❌ 小規模部署 (<1000 次/天)  
❌ 開發階段頻繁改動模型  
❌ 預算有限  
❌ 避免 NVIDIA 生態鎖定  

### 最終建議

**推薦作為第二階段優化方案**

建議路線：
```
第 1 階段：Python 訓練驗證 (3 天)
    ↓
第 2 階段：導出 ONNX，部署到生產 (1 週)
    ↓ (可選，如果需要極致性能)
第 3 階段：TensorRT 優化 (GPU 環境)
```

**關鍵決策點**：
- 如果有 GPU：強烈推薦 TensorRT (10x 加速)
- 如果無 GPU：使用 ONNX Runtime (仍有 1.7x 加速)
- 訓練階段：始終使用 Python (靈活性)

---

**報告生成時間**：2025-11-08  
**版本**：1.0  
**狀態**：待評估
