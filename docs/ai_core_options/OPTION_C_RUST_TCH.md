# 方案 C：Rust + tch-rs (PyTorch Rust 綁定)

## 📋 執行摘要

**核心策略**：使用 Rust 實現 AI 核心，通過 `tch-rs` (PyTorch Rust 綁定) 獲得自動微分與訓練能力，結合 Rust 的安全性與 PyTorch 的成熟生態。

**開發時間**：4-6 週  
**部署時間**：1-2 週  
**預估成本**：高（學習曲線 + Rust 生態）  
**風險等級**：⭐⭐⭐⭐ 中高

---

## 🎯 方案概述

### 核心目標

用 Rust 重寫 AI 核心，利用 `tch-rs` 獲得完整的深度學習能力：

```
當前：Python BioNeuron (24 MB, numpy)
    ↓ Rust 重寫
目標：Rust AI Core (5 MB, tch-rs + PyTorch)
```

### 技術架構

```
┌─────────────────────────────────────────────────────┐
│              AIVA Python 層                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  Python Services                              │  │
│  │  (保持不變)                                   │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓ PyO3 (Python ↔ Rust 綁定)        │
│  ┌──────────────────────────────────────────────┐  │
│  │         Rust Python Bindings                  │  │
│  │  pub fn aiva_forward(features: Vec<f32>)      │  │
│  │  pub fn aiva_train_step(x, y)                 │  │
│  │  pub fn aiva_save_weights(path)               │  │
│  └──────────────┬───────────────────────────────┘  │
└─────────────────┼───────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────┐
│         Rust AI Core (.so/.dll)                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  use tch::{nn, nn::Module, Device, Tensor};   │  │
│  │                                                │  │
│  │  pub struct AIVANet {                         │  │
│  │      fc1: nn::Linear,    // 512 → 2048        │  │
│  │      spiking: SpikingLayer,                   │  │
│  │      fc2: nn::Linear,    // 1024 → 20         │  │
│  │      optimizer: nn::Optimizer,                │  │
│  │  }                                            │  │
│  │                                                │  │
│  │  impl AIVANet {                               │  │
│  │      fn forward(&self, x: &Tensor) -> Tensor  │  │
│  │      fn train_step(&mut self, x, y) -> f32    │  │
│  │  }                                            │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  libtorch (C++ PyTorch 核心)                  │  │
│  │  - 自動微分                                   │  │
│  │  - CUDA 加速 (可選)                           │  │
│  │  - 優化器 (Adam, SGD...)                      │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📊 技術規格

### 模型架構

| 層級 | 輸入維度 | 輸出維度 | 參數數量 | 激活函數 |
|------|----------|----------|----------|----------|
| **fc1** | 512 | 2048 | 1,048,576 | Tanh |
| **spiking** | 2048 | 1024 | 2,097,152 | 尖峰 |
| **fc2** | 1024 | 20 | 20,480 | Softmax |
| **總計** | - | - | **3,166,208** | - |

*與 Python 版本相同架構*

### Rust 專案結構

```
aiva-rust-core/
├── Cargo.toml                  1 KB   (專案配置)
├── build.rs                    2 KB   (構建腳本)
│
├── src/
│   ├── lib.rs                  5 KB   (庫根)
│   ├── net.rs                 20 KB   (神經網路)
│   ├── spiking.rs             15 KB   (尖峰層)
│   ├── trainer.rs             15 KB   (訓練器)
│   └── bindings.rs            10 KB   (Python 綁定)
│
├── models/
│   └── weights.pt              24 MB  (PyTorch 格式)
│
├── target/release/
│   └── libaiva_core.so         5 MB   (編譯產物)
│
└── tests/
    ├── test_forward.rs         3 KB
    └── test_training.rs        5 KB
─────────────────────────────────────────
編譯後總計：                    ~30 MB
(5 MB 庫 + 24 MB 權重)
```

### 性能指標

| 指標 | Python + NumPy | Rust + tch | 改善 |
|------|----------------|------------|------|
| **推理延遲** | 0.5 ms | 0.3 ms | **1.7x 快** |
| **訓練速度** | N/A | ✅ 原生 | **新功能** |
| **內存安全** | ⚠️ 手動 | ✅ 編譯時 | **質的提升** |
| **並發安全** | ⚠️ GIL | ✅ 無鎖 | **大幅改善** |
| **檔案大小** | 24 MB | 30 MB | **略大** |
| **啟動時間** | 100 ms | 200 ms | **略慢** |

---

## 🔧 實施計畫

### 階段 1：Rust 環境設置 (3 天)

**任務 1.1：安裝 Rust 工具鏈**
```bash
# Windows
# 下載並運行 rustup-init.exe
# https://rustup.rs/

# 安裝 MSVC 工具鏈
rustup default stable-msvc

# 驗證安裝
rustc --version
cargo --version
```

**任務 1.2：安裝 PyTorch C++ 庫**
```bash
# 下載 libtorch (CPU 版本)
# https://pytorch.org/get-started/locally/

# Windows 範例
Invoke-WebRequest -Uri https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.0%2Bcpu.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip -DestinationPath C:\libtorch

# 設置環境變數
$env:LIBTORCH = "C:\libtorch"
$env:Path += ";C:\libtorch\lib"
```

**任務 1.3：創建 Rust 專案**
```bash
cargo new --lib aiva-rust-core
cd aiva-rust-core
```

**Cargo.toml 配置**
```toml
[package]
name = "aiva-rust-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "aiva_core"
crate-type = ["cdylib", "rlib"]

[dependencies]
tch = "0.14"              # PyTorch Rust 綁定
pyo3 = { version = "0.20", features = ["extension-module"] }
ndarray = "0.15"          # 多維陣列
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"            # 錯誤處理

[dev-dependencies]
approx = "0.5"            # 浮點比較
```

### 階段 2：核心實現 (10-14 天)

**任務 2.1：基礎網路結構**
```rust
// src/net.rs

use tch::{nn, nn::Module, Device, Tensor};

/// AIVA 神經網路
pub struct AIVANet {
    fc1: nn::Linear,
    spiking: SpikingLayer,
    fc2: nn::Linear,
    device: Device,
}

impl AIVANet {
    /// 創建新網路
    pub fn new(vs: &nn::Path, input_size: i64, num_tools: i64) -> Self {
        let fc1 = nn::linear(vs / "fc1", input_size, 2048, Default::default());
        let spiking = SpikingLayer::new(vs / "spiking", 2048, 1024);
        let fc2 = nn::linear(vs / "fc2", 1024, num_tools, Default::default());
        
        let device = vs.device();
        
        Self { fc1, spiking, fc2, device }
    }
    
    /// 前向傳播
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.to_device(self.device);
        
        // Layer 1: 512 → 2048 (Tanh)
        let x = x.apply(&self.fc1).tanh();
        
        // Spiking Layer: 2048 → 1024
        let x = self.spiking.forward(&x);
        
        // Layer 2: 1024 → 20 (Softmax)
        let logits = x.apply(&self.fc2);
        logits.softmax(-1, tch::Kind::Float)
    }
    
    /// 載入權重
    pub fn load_weights(&mut self, path: &str) -> anyhow::Result<()> {
        // 從 Python 訓練的權重轉換而來
        let vs = nn::VarStore::new(self.device);
        vs.load(path)?;
        Ok(())
    }
}
```

**任務 2.2：生物尖峰層**
```rust
// src/spiking.rs

use tch::{nn, Tensor};

/// 生物尖峰神經層
pub struct SpikingLayer {
    w: Tensor,
    threshold: f64,
    refractory_period: i64,
    spike_history: Vec<Tensor>,
}

impl SpikingLayer {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64) -> Self {
        let w = vs.var("weight", &[input_size, output_size], nn::Init::Randn {
            mean: 0.0,
            stdev: 0.01,
        });
        
        Self {
            w,
            threshold: 0.5,
            refractory_period: 2,
            spike_history: Vec::new(),
        }
    }
    
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        // 計算膜電位
        let membrane_potential = x.matmul(&self.w);
        
        // 尖峰判斷
        let spikes = membrane_potential.ge(self.threshold);
        
        // 不反應期處理
        let output = if self.spike_history.len() >= self.refractory_period as usize {
            let recent_spikes = &self.spike_history[self.spike_history.len() - self.refractory_period as usize..];
            let refractory_mask = recent_spikes.iter()
                .fold(Tensor::ones_like(&spikes), |acc, s| acc * (1 - s));
            spikes * refractory_mask
        } else {
            spikes
        };
        
        // 記錄歷史
        self.spike_history.push(output.shallow_clone());
        if self.spike_history.len() > 10 {
            self.spike_history.remove(0);
        }
        
        output.to_kind(tch::Kind::Float)
    }
}
```

**任務 2.3：訓練器**
```rust
// src/trainer.rs

use tch::{nn, nn::OptimizerConfig, Tensor};
use crate::net::AIVANet;

pub struct Trainer {
    net: AIVANet,
    optimizer: nn::Optimizer,
    loss_history: Vec<f64>,
}

impl Trainer {
    pub fn new(net: AIVANet, learning_rate: f64) -> Self {
        let vs = nn::VarStore::new(net.device);
        let optimizer = nn::Adam::default()
            .build(&vs, learning_rate)
            .expect("Failed to create optimizer");
        
        Self {
            net,
            optimizer,
            loss_history: Vec::new(),
        }
    }
    
    /// 訓練一步
    pub fn train_step(&mut self, x: &Tensor, y: &Tensor) -> f64 {
        // 前向傳播
        let pred = self.net.forward(x);
        
        // 計算交叉熵損失
        let loss = pred.cross_entropy_for_logits(y);
        
        // 反向傳播
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
        
        // 記錄損失
        let loss_value = f64::from(loss);
        self.loss_history.push(loss_value);
        
        loss_value
    }
    
    /// 批次訓練
    pub fn train_epoch(&mut self, x_batch: &[Tensor], y_batch: &[Tensor]) -> f64 {
        let mut total_loss = 0.0;
        
        for (x, y) in x_batch.iter().zip(y_batch.iter()) {
            total_loss += self.train_step(x, y);
        }
        
        total_loss / x_batch.len() as f64
    }
    
    /// 保存模型
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let vs = nn::VarStore::new(self.net.device);
        vs.save(path)?;
        Ok(())
    }
}
```

### 階段 3：Python 綁定 (5 天)

**任務 3.1：PyO3 綁定**
```rust
// src/bindings.rs

use pyo3::prelude::*;
use pyo3::types::PyList;
use tch::{Device, Tensor};
use crate::net::AIVANet;
use crate::trainer::Trainer;

#[pyclass]
pub struct RustAICore {
    net: AIVANet,
    trainer: Option<Trainer>,
}

#[pymethods]
impl RustAICore {
    #[new]
    pub fn new(input_size: i64, num_tools: i64, use_cuda: bool) -> Self {
        let device = if use_cuda && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        
        let vs = nn::VarStore::new(device);
        let net = AIVANet::new(&vs.root(), input_size, num_tools);
        
        Self {
            net,
            trainer: None,
        }
    }
    
    /// 前向推理
    pub fn forward(&self, features: Vec<f32>) -> PyResult<Vec<f32>> {
        let x = Tensor::of_slice(&features)
            .view([1, features.len() as i64]);
        
        let probs = self.net.forward(&x);
        
        let probs_vec: Vec<f32> = probs
            .view([-1])
            .try_into()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to convert tensor: {:?}", e)
            ))?;
        
        Ok(probs_vec)
    }
    
    /// 初始化訓練器
    pub fn init_trainer(&mut self, learning_rate: f64) {
        self.trainer = Some(Trainer::new(self.net.clone(), learning_rate));
    }
    
    /// 訓練一步
    pub fn train_step(&mut self, features: Vec<f32>, label: i64) -> PyResult<f64> {
        let trainer = self.trainer.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Trainer not initialized"
            ))?;
        
        let x = Tensor::of_slice(&features).view([1, features.len() as i64]);
        let y = Tensor::of_slice(&[label]).view([1]);
        
        Ok(trainer.train_step(&x, &y))
    }
    
    /// 保存權重
    pub fn save_weights(&self, path: &str) -> PyResult<()> {
        self.net.save_weights(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to save: {:?}", e)
            ))
    }
    
    /// 載入權重
    pub fn load_weights(&mut self, path: &str) -> PyResult<()> {
        self.net.load_weights(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to load: {:?}", e)
            ))
    }
}

#[pymodule]
fn aiva_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustAICore>()?;
    Ok(())
}
```

**任務 3.2：Python 包裝層**
```python
# aiva_bindings/rust_wrapper.py

from aiva_core import RustAICore as _RustAICore
import numpy as np

class RustAICore:
    """Rust AI 核心的 Python 友好包裝"""
    
    def __init__(self, input_size=512, num_tools=20, use_cuda=False):
        self.core = _RustAICore(input_size, num_tools, use_cuda)
        self.is_trained = False
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向推理"""
        if x.shape[0] != 512:
            raise ValueError(f"Expected 512 features, got {x.shape[0]}")
        
        # 轉換為 Python list (Rust 期望)
        x_list = x.astype(np.float32).tolist()
        
        # 調用 Rust 核心
        probs_list = self.core.forward(x_list)
        
        # 轉回 numpy
        return np.array(probs_list, dtype=np.float32)
    
    def init_trainer(self, learning_rate=0.001):
        """初始化訓練器"""
        self.core.init_trainer(learning_rate)
        self.is_trained = True
    
    def train_step(self, x: np.ndarray, y: int) -> float:
        """訓練一步"""
        if not self.is_trained:
            raise RuntimeError("Trainer not initialized")
        
        x_list = x.astype(np.float32).tolist()
        loss = self.core.train_step(x_list, y)
        return loss
    
    def save(self, path: str):
        """保存權重"""
        self.core.save_weights(path)
    
    def load(self, path: str):
        """載入權重"""
        self.core.load_weights(path)
```

### 階段 4：整合與測試 (7 天)

**任務 4.1：AIVA 整合**
```python
# services/core/aiva_core/core.py

class AIVACore:
    def __init__(self, use_rust_core: bool = False):
        if use_rust_core:
            from aiva_bindings.rust_wrapper import RustAICore
            self.ai_core = RustAICore(
                input_size=512,
                num_tools=20,
                use_cuda=torch.cuda.is_available()
            )
            logger.info("使用 Rust AI 核心")
        else:
            self.ai_core = ScalableBioNet(512, 20)
            logger.info("使用 Python BioNeuron 核心")
```

**任務 4.2：性能測試**
```rust
// tests/test_performance.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_inference_speed() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let net = AIVANet::new(&vs.root(), 512, 20);
        
        let x = Tensor::randn(&[1, 512], (tch::Kind::Float, device));
        
        // 預熱
        for _ in 0..100 {
            let _ = net.forward(&x);
        }
        
        // 測試
        let start = Instant::now();
        for _ in 0..10000 {
            let _ = net.forward(&x);
        }
        let duration = start.elapsed();
        
        let avg_ms = duration.as_micros() as f64 / 10000.0 / 1000.0;
        println!("平均推理時間: {:.3} ms", avg_ms);
        
        assert!(avg_ms < 1.0, "推理速度應 < 1 ms");
    }
}
```

---

## 📈 預期成果

### Rust 獨特優勢

| 特性 | Python | Rust | 改善 |
|------|--------|------|------|
| **內存安全** | ⚠️ 手動 | ✅ 編譯時保證 | **質的提升** |
| **並發安全** | ⚠️ GIL 限制 | ✅ 無數據競爭 | **大幅改善** |
| **錯誤處理** | 異常 | Result<T, E> | **明確性提升** |
| **零成本抽象** | ❌ | ✅ | **新特性** |
| **生命週期** | ❌ | ✅ 編譯時檢查 | **新特性** |

### 學習曲線

```
Rust 熟練度
    ↑
100%│                                   ┌──────
    │                               ┌───┘
 80%│                          ┌────┘
    │                     ┌────┘
 60%│                ┌────┘
    │           ┌────┘      ← 陡峭學習曲線
 40%│      ┌────┘
    │  ┌───┘
 20%│──┘
    └────────────────────────────────────────→
      1週  2週  4週  8週 12週 16週   時間
```

預估團隊學習時間：
- 有 C++ 經驗：2-4 週
- 無系統語言經驗：8-12 週

---

## 💰 成本分析

### 開發成本

| 階段 | 工時 | 技能需求 | 成本 |
|------|------|----------|------|
| Rust 學習 | 2-4 週 | 系統編程 | 高 |
| 核心實現 | 2 週 | Rust + ML | 高 |
| Python 綁定 | 1 週 | PyO3 | 中 |
| 整合測試 | 1 週 | 多語言 | 中 |
| **總計** | **6-8 週** | **多技能** | **高** |

### 依賴成本

| 依賴 | 大小 | 授權 | 維護 |
|------|------|------|------|
| **libtorch** | 200 MB | BSD | Meta |
| **tch-rs** | 編譯時 | Apache-2.0 | 社群 |
| **Rust 工具鏈** | 500 MB | MIT | Rust 基金會 |

---

## ⚠️ 風險評估

### 技術風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **學習曲線陡峭** | 極高 | 高 | 專家培訓、充足時間 |
| **編譯複雜** | 高 | 中 | CI/CD 自動化 |
| **跨平台問題** | 中 | 高 | Docker 容器化 |
| **tch-rs 成熟度** | 中 | 中 | 備用方案 (C++ 核心) |
| **調試困難** | 中 | 中 | 完善日誌、單元測試 |

### 團隊風險

```
如果團隊沒有 Rust 經驗：
- 開發時間 × 2
- Bug 修復時間 × 3
- 維護成本持續高企
```

---

## ✅ 結論與建議

### 核心優勢

1. **內存安全**：編譯時保證，無數據競爭
2. **性能優秀**：接近 C++，優於 Python
3. **現代生態**：Cargo、測試、文檔一體化
4. **訓練能力**：tch-rs 提供完整自動微分
5. **未來趨勢**：Rust 在系統編程快速崛起

### 核心劣勢

1. **學習曲線**：6-12 週才能熟練
2. **開發慢**：編譯時間長、調試複雜
3. **依賴龐大**：libtorch 200 MB
4. **社群小**：tch-rs 文檔不如 PyTorch
5. **招聘難**：Rust 人才稀缺

### 適用場景

✅ **長期投資項目**  
✅ 需要極致安全性  
✅ 並發場景複雜  
✅ 團隊願意學習新技術  
✅ 追求現代化技術棧  

### 不適用場景

❌ **快速原型開發**  
❌ 團隊無系統語言經驗  
❌ 3 個月內要交付  
❌ 維護人員不固定  
❌ 預算有限  

### 最終建議

**僅適合作為長期技術投資**

建議時機：
- Python 方案已成熟運行 6 個月+
- 團隊有 2+ 名 Rust 開發者
- 公司支持長期技術升級
- 安全性成為核心需求

不建議原因：
- 學習成本過高（6-12 週）
- 開發週期過長（6-8 週）
- 現階段 Python 方案足夠

---

**報告生成時間**：2025-11-08  
**版本**：1.0  
**狀態**：待評估
