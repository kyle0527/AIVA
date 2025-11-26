# 方案 B：採用 C++ 原生核心 (資料夾 5)

## 📑 目錄

- [📋 執行摘要](#執行摘要)
- [🎯 方案概述](#方案概述)
  - [核心目標](#核心目標)
  - [技術架構](#技術架構)
- [📊 技術規格](#技術規格)
  - [模型架構](#模型架構)
  - [核心文件結構](#核心文件結構)
  - [性能指標](#性能指標)
- [🔧 實施計畫](#實施計畫)
  - [階段 1：C++ 核心編譯與測試 (3 天)](#階段-1c-核心編譯與測試-3-天)
  - [階段 2：Python 綁定開發 (4 天)](#階段-2python-綁定開發-4-天)
  - [階段 3：AIVA 整合 (5 天)](#階段-3aiva-整合-5-天)
  - [階段 4：訓練與優化 (3-5 天)](#階段-4訓練與優化-35-天)
- [📈 預期成果](#預期成果)
  - [性能對比](#性能對比)
  - [架構對比](#架構對比)
  - [部署優勢](#部署優勢)
- [💰 成本分析](#成本分析)
  - [開發成本](#開發成本)
  - [維護成本](#維護成本)
- [⚠️ 風險評估](#風險評估)
  - [技術風險](#技術風險)
  - [實施風險](#實施風險)
  - [降維風險 (關鍵)](#降維風險-關鍵)
- [🎯 成功標準](#成功標準)
  - [必須達成](#必須達成)
  - [期望達成](#期望達成)
  - [最好達成](#最好達成)
- [🚀 部署計畫](#部署計畫)
  - [開發環境](#開發環境)
  - [生產環境](#生產環境)
- [📊 限制與約束](#限制與約束)
  - [架構限制](#架構限制)
  - [功能限制](#功能限制)
- [🔄 遷移策略](#遷移策略)
  - [漸進式遷移](#漸進式遷移)
  - [回滾計畫](#回滾計畫)
- [✅ 結論與建議](#結論與建議)
  - [核心優勢](#核心優勢)
  - [核心劣勢](#核心劣勢)
  - [適用場景](#適用場景)
  - [不適用場景](#不適用場景)
  - [最終建議](#最終建議)

---
---
---
## 📋 執行摘要

**核心策略**：使用輕量級 C++ 原生 AI 核心，替換現有 Python BioNeuron，追求極致性能與最小化部署。

**開發時間**：2-3 週  
**部署時間**：1 週  
**預估成本**：中（需要 C++ 開發與整合）  
**風險等級**：⭐⭐⭐ 中等

---

## 🎯 方案概述

### 核心目標

使用資料夾 (5) 中的 C++ 原生核心替換 Python 實現：
```
當前：Python BioNeuron (24 MB, 0.5 ms)
    ↓ 替換
目標：C++ 原生核心 (70 KB, 0.05 ms)
```

### 技術架構

```
┌─────────────────────────────────────────────────────┐
│              AIVA Python 層                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  Python Feature Extractor                     │  │
│  │  (特徵提取保持 Python)                        │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓ ctypes/pybind11                   │
│  ┌──────────────────────────────────────────────┐  │
│  │         C API Bridge                          │  │
│  │  - aiva_create()                              │  │
│  │  - aiva_predict()                             │  │
│  │  - aiva_destroy()                             │  │
│  └──────────────┬───────────────────────────────┘  │
└─────────────────┼───────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────┐
│         C++ 原生核心 (.dll/.so)                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  AivaHandle (Opaque Handle)                   │  │
│  │                                                │  │
│  │  Input (16維)                                 │  │
│  │      ↓ Dense Layer (W1)                       │  │
│  │  [32 neurons] × ReLU                          │  │
│  │      ↓ Dense Layer (W2)                       │  │
│  │  [6 outputs] × softmax                        │  │
│  │      ↓                                         │  │
│  │  工具選擇機率分布                              │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  weights.json (20 KB)                         │  │
│  │  - W1: [16 × 32]                              │  │
│  │  - b1: [32]                                   │  │
│  │  - W2: [32 × 6]                               │  │
│  │  - b2: [6]                                    │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📊 技術規格

### 模型架構

| 層級 | 輸入維度 | 輸出維度 | 參數數量 | 激活函數 |
|------|----------|----------|----------|----------|
| **W1** | 16 | 32 | 512 | ReLU |
| **b1** | - | 32 | 32 | - |
| **W2** | 32 | 6 | 192 | Linear |
| **b2** | - | 6 | 6 | - |
| **總計** | - | - | **742** | - |

### 核心文件結構

```
aiva_opt_core/
├── include/
│   └── aiva/
│       └── opt_core.h           2 KB   (C API 頭文件)
│
├── src/
│   └── opt_core.cc             15 KB   (核心實現)
│
├── models/
│   └── weights.json            20 KB   (權重檔案)
│
├── build/
│   ├── libaiva_opt_core.so    50 KB   (Linux)
│   └── aiva_opt_core.dll      50 KB   (Windows)
│
└── bindings/
    └── python/
        └── aiva_core.py       10 KB   (Python 綁定)
────────────────────────────────────────
總計：                        ~150 KB
```

### 性能指標

| 指標 | 數值 | vs Python |
|------|------|-----------|
| **推理延遲** | 0.05 ms | **10x 快** |
| **訓練時間/樣本** | N/A | - |
| **內存佔用** | 1 MB | **50x 小** |
| **檔案大小** | 70 KB | **343x 小** |
| **吞吐量** | 20,000 次/秒 | **10x 高** |
| **啟動時間** | 0.5 ms | **20x 快** |

---

## 🔧 實施計畫

### 階段 1：C++ 核心編譯與測試 (3 天)

**任務 1.1：環境設置**
```bash
# Windows
choco install cmake
choco install visualstudio2022-workload-nativecpp

# Linux
sudo apt install cmake g++ build-essential

# 編譯
cd "C:\Users\User\Downloads\新增資料夾 (5)"
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**任務 1.2：功能測試**
```bash
# 運行 C 範例
./c_example ../models/weights.json

預期輸出：
dims: in=16 out=6
probs: 0.167 0.167 0.167 0.167 0.167 0.167
top-3: (0,0.167) (1,0.167) (2,0.167)
```

**任務 1.3：性能基準測試**
```cpp
// benchmark.cpp
#include "aiva/opt_core.h"
#include <chrono>

int main() {
    AivaHandle* h;
    aiva_create("weights.json", &h);
    
    float x[16] = {0};  // 測試輸入
    float p[6];
    
    // 預熱
    for (int i = 0; i < 1000; ++i) {
        aiva_predict(h, x, 16, p, 6);
    }
    
    // 基準測試
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        aiva_predict(h, x, 16, p, 6);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg = duration.count() / 100000.0;
    
    printf("平均推理時間: %.3f μs\n", avg);
    printf("吞吐量: %.0f 次/秒\n", 1e6 / avg);
    
    aiva_destroy(h);
    return 0;
}
```

### 階段 2：Python 綁定開發 (4 天)

**任務 2.1：ctypes 綁定**
```python
# aiva_bindings/cpp_core.py

import ctypes
import numpy as np
from pathlib import Path

class CppAICore:
    """C++ 核心的 Python 包裝"""
    
    def __init__(self, lib_path: str, weights_path: str):
        # 載入 DLL/SO
        self.lib = ctypes.CDLL(lib_path)
        
        # 定義函數簽名
        self.lib.aiva_create.argtypes = [
            ctypes.c_char_p,           # weights_json_path
            ctypes.POINTER(ctypes.c_void_p)  # out_handle
        ]
        self.lib.aiva_create.restype = ctypes.c_int
        
        self.lib.aiva_predict.argtypes = [
            ctypes.c_void_p,           # handle
            ctypes.POINTER(ctypes.c_float),  # feature
            ctypes.c_int,              # dim
            ctypes.POINTER(ctypes.c_float),  # out_probs
            ctypes.c_int               # out_dim
        ]
        self.lib.aiva_predict.restype = ctypes.c_int
        
        self.lib.aiva_destroy.argtypes = [ctypes.c_void_p]
        self.lib.aiva_destroy.restype = None
        
        # 創建核心實例
        self.handle = ctypes.c_void_p()
        status = self.lib.aiva_create(
            weights_path.encode('utf-8'),
            ctypes.byref(self.handle)
        )
        
        if status != 0:  # AIVA_STATUS_OK
            raise RuntimeError(f"Failed to create C++ core: {status}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向傳播"""
        if x.shape[0] != 16:
            raise ValueError(f"Expected 16 features, got {x.shape[0]}")
        
        # 轉換為 float32
        x = x.astype(np.float32)
        
        # 準備輸出
        probs = np.zeros(6, dtype=np.float32)
        
        # 調用 C++ 核心
        status = self.lib.aiva_predict(
            self.handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            16,
            probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            6
        )
        
        if status != 0:
            raise RuntimeError(f"Prediction failed: {status}")
        
        return probs
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'handle'):
            self.lib.aiva_destroy(self.handle)
```

**任務 2.2：特徵維度適配器**
```python
# aiva_bindings/feature_adapter.py

class FeatureAdapter:
    """將 AIVA 512 維特徵壓縮到 C++ 核心的 16 維"""
    
    def __init__(self):
        # 定義重要特徵索引（根據分析選出最重要的 16 維）
        self.important_indices = [
            0,   # 開放端口數量
            10,  # HTTP 服務存在
            11,  # HTTPS 服務存在
            20,  # MySQL 服務存在
            30,  # SSH 服務存在
            50,  # SQL 注入漏洞
            51,  # XSS 漏洞
            52,  # CSRF 漏洞
            100, # 目標操作系統類型
            150, # 目標 Web 框架
            200, # 歷史成功率
            250, # 歷史平均時間
            300, # 漏洞嚴重度
            350, # 端口開放比例
            400, # 服務版本資訊
            450, # 認證強度
        ]
    
    def compress(self, features_512: np.ndarray) -> np.ndarray:
        """512 維 → 16 維"""
        if len(features_512) != 512:
            raise ValueError(f"Expected 512 features, got {len(features_512)}")
        
        # 方法 1：選擇重要特徵
        compressed = features_512[self.important_indices]
        
        # 方法 2：主成分分析（可選，更精確）
        # compressed = self.pca.transform(features_512.reshape(1, -1))[0]
        
        # 正規化到 [0, 1]
        compressed = np.clip(compressed, 0, 1)
        
        return compressed.astype(np.float32)
```

**任務 2.3：工具映射擴展**
```python
# aiva_bindings/tool_mapper.py

class ToolMapper:
    """將 C++ 的 6 個輸出映射到 AIVA 的 20 個工具"""
    
    def __init__(self):
        # C++ 輸出 6 類別 → AIVA 20 工具映射
        self.mapping = {
            0: [0, 1, 2, 3],      # 掃描類工具
            1: [4, 5, 6],         # SQL 注入工具
            2: [7, 8, 9],         # XSS 工具
            3: [10, 11, 12, 13],  # 暴力破解工具
            4: [14, 15, 16],      # 漏洞利用工具
            5: [17, 18, 19],      # 後滲透工具
        }
    
    def expand(self, cpp_probs: np.ndarray) -> np.ndarray:
        """6 維機率 → 20 維機率"""
        aiva_probs = np.zeros(20, dtype=np.float32)
        
        for cpp_idx, tool_indices in self.mapping.items():
            # 將 C++ 類別機率平均分配到對應的 AIVA 工具
            prob_per_tool = cpp_probs[cpp_idx] / len(tool_indices)
            for tool_idx in tool_indices:
                aiva_probs[tool_idx] = prob_per_tool
        
        # 重新正規化
        aiva_probs = aiva_probs / aiva_probs.sum()
        
        return aiva_probs
```

### 階段 3：AIVA 整合 (5 天)

**任務 3.1：核心替換**
```python
# services/core/aiva_core/core.py

class AIVACore:
    def __init__(self, use_cpp_core: bool = False):
        if use_cpp_core:
            # 使用 C++ 核心
            self.ai_core = CppAICore(
                lib_path='lib/aiva_opt_core.dll',
                weights_path='models/weights.json'
            )
            self.feature_adapter = FeatureAdapter()
            self.tool_mapper = ToolMapper()
            logger.info("使用 C++ 原生核心")
        else:
            # 使用 Python 核心
            self.ai_core = ScalableBioNet(512, 20)
            logger.info("使用 Python BioNeuron 核心")
    
    def select_tool(self, scan_result: dict) -> str:
        """選擇工具（支持兩種核心）"""
        if isinstance(self.ai_core, CppAICore):
            # C++ 核心路徑
            features_512 = self.feature_extractor.extract(scan_result)
            features_16 = self.feature_adapter.compress(features_512)
            probs_6 = self.ai_core.forward(features_16)
            probs_20 = self.tool_mapper.expand(probs_6)
        else:
            # Python 核心路徑
            features_512 = self.feature_extractor.extract(scan_result)
            probs_20 = self.ai_core.forward(features_512)
        
        tool_index = np.argmax(probs_20)
        return self.tools[tool_index]
```

**任務 3.2：配置管理**
```yaml
# config/ai_core.yaml

ai_core:
  # 核心類型選擇
  type: cpp  # 或 'python'
  
  # C++ 核心配置
  cpp:
    library_path: lib/aiva_opt_core.dll
    weights_path: models/weights.json
    feature_compression: true
    compression_method: pca  # 或 'selection'
  
  # Python 核心配置
  python:
    weights_path: models/trained_weights/
    use_trained: true
    confidence_threshold: 0.7
```

### 階段 4：訓練與優化 (3-5 天)

**任務 4.1：離線訓練 C++ 權重**

由於 C++ 核心不支持內建訓練，需要：

```python
# scripts/train_cpp_weights.py

import numpy as np
from sklearn.neural_network import MLPClassifier

def train_cpp_compatible_model():
    """在 Python 訓練，導出到 C++ 格式"""
    
    # 1. 收集並壓縮數據
    collector = TrainingDataCollector()
    adapter = FeatureAdapter()
    
    X_512 = np.array([s['features'] for s in collector.samples])
    X_16 = np.array([adapter.compress(x) for x in X_512])
    y = np.array([s['tool_index'] % 6 for s in collector.samples])  # 映射到 6 類
    
    # 2. 訓練模型（使用 sklearn）
    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    
    model.fit(X_16, y)
    
    # 3. 提取權重
    W1 = model.coefs_[0].T  # [16, 32]
    b1 = model.intercepts_[0]  # [32]
    W2 = model.coefs_[1].T  # [32, 6]
    b2 = model.intercepts_[1]  # [6]
    
    # 4. 導出為 JSON
    import json
    weights = {
        'W1': W1.tolist(),
        'b1': b1.tolist(),
        'W2': W2.tolist(),
        'b2': b2.tolist()
    }
    
    with open('models/weights.json', 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"訓練完成，準確率: {model.score(X_16, y):.2%}")

if __name__ == '__main__':
    train_cpp_compatible_model()
```

**任務 4.2：熱更新機制**
```python
# aiva_bindings/hot_reload.py

class HotReloadableCppCore:
    """支持熱更新權重的 C++ 核心"""
    
    def __init__(self, lib_path, weights_path):
        self.lib_path = lib_path
        self.weights_path = weights_path
        self.core = CppAICore(lib_path, weights_path)
        self.last_modified = os.path.getmtime(weights_path)
    
    def forward(self, x):
        # 檢查權重是否更新
        current_modified = os.path.getmtime(self.weights_path)
        if current_modified > self.last_modified:
            logger.info("偵測到權重更新，重新載入核心")
            del self.core
            self.core = CppAICore(self.lib_path, self.weights_path)
            self.last_modified = current_modified
        
        return self.core.forward(x)
```

---

## 📈 預期成果

### 性能對比

| 指標 | Python 核心 | C++ 核心 | 改善 |
|------|-------------|----------|------|
| **檔案大小** | 24 MB | 70 KB | **343x 小** |
| **內存佔用** | 50 MB | 1 MB | **50x 小** |
| **推理延遲** | 0.5 ms | 0.05 ms | **10x 快** |
| **啟動時間** | 100 ms | 5 ms | **20x 快** |
| **吞吐量** | 2,000/s | 20,000/s | **10x 高** |

### 架構對比

| 層級 | Python BioNeuron | C++ 核心 |
|------|------------------|----------|
| **輸入** | 512 維 | 16 維 |
| **隱藏** | 2048 → 1024 | 32 |
| **輸出** | 20 工具 | 6 類別 |
| **參數** | 3.16M | 742 |
| **特殊** | Spiking Layer | 無 |

### 部署優勢

```
嵌入式設備：✅ 可行 (1 MB 內存)
容器化：✅ 極小鏡像 (+70 KB)
邊緣計算：✅ 低延遲 (0.05 ms)
跨語言：✅ C ABI 通用
```

---

## 💰 成本分析

### 開發成本

| 階段 | 工時 | 技能需求 | 成本 |
|------|------|----------|------|
| C++ 編譯測試 | 3 天 | C++/CMake | 中 |
| Python 綁定 | 4 天 | ctypes/C API | 中 |
| AIVA 整合 | 5 天 | Python/架構 | 中 |
| 訓練優化 | 3-5 天 | ML/數據處理 | 中 |
| **總計** | **15-17 天** | **多技能** | **中** |

### 維護成本

| 項目 | Python 核心 | C++ 核心 | 差異 |
|------|-------------|----------|------|
| **代碼調試** | 容易 | 困難 | ⚠️ |
| **功能擴展** | 快速 | 緩慢 | ⚠️ |
| **依賴管理** | NumPy | 無 | ✅ |
| **跨平台編譯** | 自動 | 手動 | ⚠️ |
| **性能優化** | 有限 | 靈活 | ✅ |

---

## ⚠️ 風險評估

### 技術風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **架構不匹配** | 高 | 高 | 特徵適配器、工具映射 |
| **精度損失** | 高 | 中 | 16 維 < 512 維資訊量 |
| **跨平台問題** | 中 | 中 | 充分測試 Linux/Windows |
| **綁定複雜** | 中 | 中 | 使用成熟的 ctypes |
| **訓練困難** | 低 | 中 | 使用 sklearn 離線訓練 |

### 實施風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **開發超時** | 中 | 高 | 3 週緩衝期 |
| **團隊技能** | 中 | 中 | C++ 專家支援 |
| **整合問題** | 中 | 高 | 增量整合、充分測試 |
| **性能未達預期** | 低 | 中 | 事先基準測試 |

### 降維風險 (關鍵)

**512 維 → 16 維資訊損失**

```
潛在影響：
- 決策準確率可能下降
- 無法捕捉細微特徵
- 某些工具類別難以區分

緩解策略：
1. 使用 PCA 保留最大方差
2. 特徵選擇基於重要性分析
3. 6 類別映射到 20 工具的智能策略
4. 持續監控準確率
```

---

## 🎯 成功標準

### 必須達成

- ✅ C++ 核心正常編譯（Windows + Linux）
- ✅ Python 綁定功能正常
- ✅ 推理延遲 < 0.1 ms
- ✅ 內存佔用 < 5 MB
- ✅ 檔案大小 < 500 KB

### 期望達成

- ✅ 工具選擇準確率 > 60%（降維後）
- ✅ 支持熱更新權重
- ✅ 跨平台無縫運行
- ✅ 完整的錯誤處理

### 最好達成

- ✅ 工具選擇準確率 > 70%
- ✅ 支持多核心實例
- ✅ WASM 編譯版本
- ✅ Rust/Go 綁定

---

## 🚀 部署計畫

### 開發環境

```bash
# 1. 編譯 C++ 核心
cd "C:\Users\User\Downloads\新增資料夾 (5)"
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 2. 安裝 Python 綁定
cd ../bindings/python
pip install -e .

# 3. 測試整合
pytest tests/test_cpp_integration.py

# 4. 訓練並導出權重
python scripts/train_cpp_weights.py

# 5. 驗證性能
python scripts/benchmark_cpp_core.py
```

### 生產環境

```bash
# 1. 複製核心檔案
cp build/aiva_opt_core.dll /opt/aiva/lib/
cp models/weights.json /opt/aiva/models/

# 2. 更新配置
vim /opt/aiva/config/ai_core.yaml
# type: cpp

# 3. 重啟服務
systemctl restart aiva

# 4. 驗證切換成功
curl http://localhost:8000/api/core/status
# 預期: {"core_type": "cpp", "status": "active"}
```

---

## 📊 限制與約束

### 架構限制

1. **輸入維度固定**：16 維（vs Python 的 512 維）
   - 需要壓縮特徵
   - 可能損失資訊

2. **輸出類別固定**：6 類（vs Python 的 20 工具）
   - 需要類別映射
   - 粒度較粗

3. **無生物特性**：標準 MLP（vs Python 的 Spiking Layer）
   - 失去獨特性
   - 標準化處理

### 功能限制

1. **訓練能力**：
   - ❌ 無內建訓練
   - ✅ 需要外部訓練後導入

2. **靈活性**：
   - ❌ 改架構需重編譯
   - ❌ 調試需 C++ 工具

3. **擴展性**：
   - ⚠️ 添加新功能困難
   - ⚠️ 需要 C++ 專業知識

---

## 🔄 遷移策略

### 漸進式遷移

**階段 1：並行運行**
```python
# 同時運行兩個核心，對比結果
python_result = python_core.forward(features_512)
cpp_result_expanded = cpp_pipeline.forward(features_512)

# 記錄差異
diff = np.abs(python_result - cpp_result_expanded).mean()
logger.info(f"核心差異: {diff:.4f}")
```

**階段 2：A/B 測試**
```python
# 隨機選擇核心
if random.random() < 0.5:
    result = use_python_core()
    metrics.record('python')
else:
    result = use_cpp_core()
    metrics.record('cpp')
```

**階段 3：完全切換**
```python
# 配置切換
config.ai_core.type = 'cpp'
```

### 回滾計畫

```python
# 如果 C++ 核心出現問題
if cpp_core_error_rate > threshold:
    logger.warning("C++ 核心錯誤率過高，回滾到 Python")
    config.ai_core.type = 'python'
    restart_core()
```

---

## ✅ 結論與建議

### 核心優勢

1. **極致輕量**：70 KB vs 24 MB
2. **超快速度**：0.05 ms vs 0.5 ms
3. **零依賴**：不需 Python 環境
4. **跨語言**：C API 通用
5. **可嵌入**：邊緣設備友好

### 核心劣勢

1. **開發慢**：3 週 vs 3 天
2. **維護難**：需要 C++ 專業知識
3. **降維損失**：16 維 << 512 維
4. **靈活性低**：改動需重編譯
5. **特色喪失**：無 Spiking Layer

### 適用場景

✅ **成熟產品部署階段**  
✅ 需要極致性能優化  
✅ 嵌入式/邊緣計算環境  
✅ 架構已固定不再改動  
✅ 團隊有 C++ 專業能力  

### 不適用場景

❌ **當前開發階段**  
❌ 需要頻繁調整架構  
❌ 團隊不熟悉 C++  
❌ 追求快速迭代  
❌ 需要保留特殊功能  

### 最終建議

**不建議作為第一階段方案**

建議時機：
- 在 Python 核心驗證成功後
- 需要大規模部署時
- 性能真正成為瓶頸時
- 團隊具備 C++ 能力時

建議路線：
```
第 1 階段：Python 核心開發與驗證 (3 天)
    ↓
第 2 階段：收集數據、訓練優化 (1-2 週)
    ↓
第 3 階段：評估是否需要 C++ (性能 vs 成本)
    ↓
第 4 階段：(可選) 遷移到 C++ 核心 (3 週)
```

---

**報告生成時間**：2025-11-08  
**版本**：1.0  
**狀態**：待評估
