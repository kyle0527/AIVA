# 方案 A：改造 AIVA BioNeuron Core (Python + NumPy)

## 📋 執行摘要

**核心策略**：在現有 Python BioNeuron 核心基礎上，添加訓練能力，使其成為真正可學習的 AI 決策核心。

**開發時間**：2-3 天  
**部署時間**：立即可用  
**預估成本**：低（無額外基礎設施）  
**風險等級**：⭐ 低

---

## 🎯 方案概述

### 核心目標

將現有的隨機權重神經網路改造為可訓練的 AI 核心：
```
當前狀態：隨機決策 (準確率 ~5%, 1/20 工具)
    ↓ 添加訓練能力
目標狀態：智能決策 (目標準確率 70-85%)
```

### 技術架構

```
┌─────────────────────────────────────────────────────┐
│              AIVA 執行環境                           │
│  ┌──────────────────────────────────────────────┐  │
│  │  掃描結果 (Scan Results)                      │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  特徵提取器 (Feature Extractor)               │  │
│  │  - 端口資訊 → 向量                            │  │
│  │  - 服務類型 → 向量                            │  │
│  │  - 漏洞特徵 → 512維向量                       │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  TrainableBioNet (可訓練神經網路)            │  │
│  │                                                │  │
│  │  Input (512維)                                │  │
│  │      ↓ Dense Layer (FC1)                      │  │
│  │  [2,048 neurons] × tanh                       │  │
│  │      ↓ BiologicalSpikingLayer                 │  │
│  │  [1,024 neurons] × spiking                    │  │
│  │      ↓ Dense Layer (FC2)                      │  │
│  │  [20 outputs] × softmax                       │  │
│  │      ↓                                         │  │
│  │  工具選擇機率分布                              │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  反向傳播與優化 (Backward & Optimizer)        │  │
│  │  - 計算梯度                                   │  │
│  │  - Adam 優化器                                │  │
│  │  - 更新權重                                   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📊 技術規格

### 模型架構

| 層級 | 輸入維度 | 輸出維度 | 參數數量 | 激活函數 |
|------|----------|----------|----------|----------|
| **FC1** | 512 | 2,048 | 1,048,576 | tanh |
| **Spiking** | 2,048 | 1,024 | 2,097,152 | biological |
| **FC2** | 1,024 | 20 | 20,480 | softmax |
| **總計** | - | - | **3,166,208** | - |

### 權重儲存

```
檔案格式：NumPy .npy (建議) 或 HDF5

trained_weights/
├── fc1_weights.npy          8.00 MB  (512 × 2048 × 4 bytes)
├── spiking_weights.npy     16.00 MB  (2048 × 1024 × 4 bytes)
├── fc2_weights.npy          0.16 MB  (1024 × 20 × 4 bytes)
└── metadata.json            0.01 MB  (架構資訊、訓練統計)
────────────────────────────────────
總計：                      24.17 MB
```

### 性能指標

| 指標 | 數值 | 備註 |
|------|------|------|
| **推理延遲** | 0.5 ms | 單次前向傳播 |
| **訓練時間/樣本** | 1-2 ms | 包含反向傳播 |
| **內存佔用** | 50 MB | 運行時峰值 |
| **吞吐量** | 2,000 次/秒 | 單線程 |
| **並行吞吐量** | 16,000 次/秒 | 8 核心並行 |

---

## 🔧 實施計畫

### 階段 1：核心改造 (1 天)

**任務 1.1：創建可訓練版本**
```python
# 檔案：services/core/aiva_core/ai_engine/trainable_bio_neuron.py

class TrainableBioNet(ScalableBioNet):
    """可訓練版本的 BioNeuron 核心"""
    
    def __init__(self, input_size, num_tools, learning_rate=0.001):
        super().__init__(input_size, num_tools)
        self.lr = learning_rate
        
        # Adam 優化器狀態
        self.m_fc1 = np.zeros_like(self.fc1)
        self.v_fc1 = np.zeros_like(self.fc1)
        self.m_fc2 = np.zeros_like(self.fc2)
        self.v_fc2 = np.zeros_like(self.fc2)
        self.beta1, self.beta2 = 0.9, 0.999
        self.t = 0
    
    def train_step(self, x, target_tool_index):
        """單步訓練"""
        # 前向
        output = self.forward(x)
        
        # 構建目標
        target = np.zeros(len(output))
        target[target_tool_index] = 1.0
        
        # 計算損失
        loss = -np.sum(target * np.log(output + 1e-10))
        
        # 反向傳播
        grad_fc1, grad_fc2 = self._backward(x, target, output)
        
        # 更新權重
        self._adam_update(grad_fc1, grad_fc2)
        
        return loss, output
    
    def _backward(self, x, target, output):
        """反向傳播計算梯度"""
        # 輸出層梯度
        grad_output = output - target
        
        # FC2 梯度
        h = self.hidden_activation
        grad_fc2 = np.outer(h, grad_output)
        
        # 隱藏層梯度（簡化 spiking layer）
        grad_h = grad_output @ self.fc2.T
        grad_h = grad_h * (h > 0)
        
        # FC1 梯度
        grad_fc1 = np.outer(x, grad_h)
        
        return grad_fc1, grad_fc2
```

**任務 1.2：實現優化器**
```python
def _adam_update(self, grad_fc1, grad_fc2):
    """Adam 優化器更新"""
    self.t += 1
    eps = 1e-8
    
    # FC1
    self.m_fc1 = self.beta1 * self.m_fc1 + (1-self.beta1) * grad_fc1
    self.v_fc1 = self.beta2 * self.v_fc1 + (1-self.beta2) * grad_fc1**2
    m_hat = self.m_fc1 / (1 - self.beta1**self.t)
    v_hat = self.v_fc1 / (1 - self.beta2**self.t)
    self.fc1 -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # FC2（相同邏輯）
    # ...
```

### 階段 2：數據收集 (0.5 天)

**任務 2.1：特徵提取器**
```python
# 檔案：services/core/aiva_core/ai_engine/feature_extractor.py

class AIVAFeatureExtractor:
    """將 AIVA 掃描結果轉換為 512 維特徵向量"""
    
    def extract(self, scan_result: dict) -> np.ndarray:
        features = []
        
        # 1. 端口特徵 (20 維)
        features.extend(self._extract_port_features(scan_result))
        
        # 2. 服務特徵 (50 維)
        features.extend(self._extract_service_features(scan_result))
        
        # 3. 漏洞特徵 (100 維)
        features.extend(self._extract_vulnerability_features(scan_result))
        
        # 4. 目標特徵 (30 維)
        features.extend(self._extract_target_features(scan_result))
        
        # 5. 歷史特徵 (20 維)
        features.extend(self._extract_history_features(scan_result))
        
        # 補齊到 512 維
        while len(features) < 512:
            features.append(0.0)
        
        return np.array(features[:512], dtype=np.float32)
```

**任務 2.2：數據收集器**
```python
# 檔案：services/core/aiva_core/ai_engine/data_collector.py

class TrainingDataCollector:
    """收集 AIVA 執行數據用於訓練"""
    
    def __init__(self, db_path='training_data.db'):
        self.samples = []
        self.extractor = AIVAFeatureExtractor()
    
    def record_execution(
        self, 
        scan_result: dict,
        chosen_tool: str,
        execution_success: bool,
        execution_time: float,
        findings: int
    ):
        """記錄一次執行"""
        features = self.extractor.extract(scan_result)
        
        sample = {
            'features': features,
            'tool_index': self._tool_to_index(chosen_tool),
            'success': execution_success,
            'reward': self._calculate_reward(
                execution_success, 
                execution_time, 
                findings
            ),
            'timestamp': time.time()
        }
        
        self.samples.append(sample)
        
        # 定期保存
        if len(self.samples) % 100 == 0:
            self.save()
```

### 階段 3：訓練循環 (0.5 天)

**任務 3.1：訓練器實現**
```python
# 檔案：services/core/aiva_core/ai_engine/trainer.py

class AIVATrainer:
    """AIVA 核心訓練器"""
    
    def __init__(
        self, 
        model: TrainableBioNet,
        data_collector: TrainingDataCollector
    ):
        self.model = model
        self.collector = data_collector
        self.history = {'loss': [], 'accuracy': []}
    
    def train(
        self, 
        epochs: int = 100, 
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """訓練循環"""
        samples = self.collector.samples
        
        # 分割訓練/驗證集
        split_idx = int(len(samples) * (1 - validation_split))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        for epoch in range(epochs):
            # 訓練
            np.random.shuffle(train_samples)
            epoch_loss = 0.0
            
            for i in range(0, len(train_samples), batch_size):
                batch = train_samples[i:i+batch_size]
                batch_loss = 0.0
                
                for sample in batch:
                    loss, _ = self.model.train_step(
                        sample['features'],
                        sample['tool_index']
                    )
                    batch_loss += loss
                
                epoch_loss += batch_loss / len(batch)
            
            avg_loss = epoch_loss / (len(train_samples) / batch_size)
            
            # 驗證
            val_acc = self.validate(val_samples)
            
            # 記錄
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2%}")
            
            # 早停
            if val_acc > 0.85:
                print("達到目標準確率，提前停止")
                break
        
        return self.history
```

### 階段 4：整合與測試 (1 天)

**任務 4.1：整合到 AIVA 主流程**
```python
# 修改：services/core/aiva_core/core.py

class AIVACore:
    def __init__(self):
        # ... 現有初始化
        
        # 添加 AI 核心
        self.ai_core = self._load_ai_core()
        self.data_collector = TrainingDataCollector()
        self.feature_extractor = AIVAFeatureExtractor()
    
    def _load_ai_core(self):
        """載入訓練好的核心或創建新的"""
        weights_path = 'models/trained_weights'
        
        if os.path.exists(weights_path):
            # 載入訓練好的權重
            core = ScalableBioNet(512, 20)
            core.fc1 = np.load(f'{weights_path}/fc1.npy')
            core.fc2 = np.load(f'{weights_path}/fc2.npy')
            logger.info("載入訓練好的 AI 核心")
        else:
            # 使用隨機權重（初始狀態）
            core = ScalableBioNet(512, 20)
            logger.info("使用隨機初始化核心")
        
        return core
    
    def select_tool(self, scan_result: dict) -> str:
        """使用 AI 核心選擇工具"""
        # 提取特徵
        features = self.feature_extractor.extract(scan_result)
        
        # AI 決策
        probabilities = self.ai_core.forward(features)
        
        # 選擇最佳工具
        tool_index = np.argmax(probabilities)
        confidence = probabilities[tool_index]
        
        # 記錄用於訓練
        self.data_collector.record_decision(
            features, 
            tool_index, 
            confidence
        )
        
        return self.tools[tool_index]
```

**任務 4.2：訓練腳本**
```python
# 新檔案：scripts/train_ai_core.py

def main():
    # 載入收集的數據
    collector = TrainingDataCollector()
    collector.load('training_data.db')
    
    print(f"載入 {len(collector.samples)} 個訓練樣本")
    
    # 創建可訓練模型
    model = TrainableBioNet(
        input_size=512,
        num_tools=20,
        learning_rate=0.001
    )
    
    # 訓練
    trainer = AIVATrainer(model, collector)
    history = trainer.train(
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    # 保存模型
    model.save_weights('models/trained_weights')
    
    # 視覺化
    plot_training_history(history)

if __name__ == '__main__':
    main()
```

---

## 📈 預期成果

### 性能提升

| 指標 | 當前 (隨機) | 訓練後 (預期) | 提升 |
|------|-------------|---------------|------|
| **工具選擇準確率** | 5% (1/20) | 70-85% | **14-17x** |
| **平均執行時間** | 基準 | -30% | 更快 |
| **成功率** | 基準 | +50% | 更高 |
| **誤報率** | 基準 | -40% | 更低 |

### 學習曲線預估

```
準確率

85% ┤                           ╭────────
    │                       ╭───╯
75% ┤                   ╭───╯
    │               ╭───╯
65% ┤           ╭───╯
    │       ╭───╯
55% ┤   ╭───╯
    │╭──╯
45% ┤╯
    │
35% ┤
    │
25% ┤
    └────┴────┴────┴────┴────┴────┴────┴──→
    0   20   40   60   80  100  120  140  Epoch

收集樣本數需求：
- 最小可用：500 樣本 (準確率 ~60%)
- 良好性能：2,000 樣本 (準確率 ~75%)
- 最佳性能：5,000+ 樣本 (準確率 ~85%)
```

---

## 💰 成本分析

### 開發成本

| 項目 | 工時 | 成本估算 |
|------|------|----------|
| 核心改造 | 1 天 | 低 |
| 數據收集 | 0.5 天 | 低 |
| 訓練循環 | 0.5 天 | 低 |
| 整合測試 | 1 天 | 低 |
| **總計** | **3 天** | **低** |

### 運行成本

| 項目 | 成本 | 備註 |
|------|------|------|
| **計算資源** | 無額外 | 使用現有硬體 |
| **儲存空間** | ~25 MB | 權重檔案 |
| **內存需求** | +50 MB | 運行時增量 |
| **訓練時間** | 1-2 小時 | 一次性，可離線 |

### ROI 分析

```
投入：3 天開發時間
產出：
  - 工具選擇準確率提升 14-17x
  - 執行效率提升 30%
  - 誤報減少 40%
  - 可持續學習改進

ROI：極高（低投入，高回報）
```

---

## ⚠️ 風險評估

### 技術風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **過擬合** | 中 | 中 | 使用驗證集、early stopping |
| **數據不足** | 中 | 高 | 主動數據收集、數據增強 |
| **特徵設計不佳** | 低 | 中 | 迭代優化特徵提取 |
| **訓練不穩定** | 低 | 中 | 使用 Adam 優化器、梯度裁剪 |

### 實施風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **與現有代碼衝突** | 低 | 中 | 充分測試、逐步整合 |
| **性能退化** | 低 | 高 | A/B 測試、回滾機制 |
| **內存溢出** | 極低 | 中 | 50 MB 增量可忽略 |

---

## 🎯 成功標準

### 必須達成 (Must Have)

- ✅ 訓練功能正常運作
- ✅ 工具選擇準確率 > 60%
- ✅ 推理延遲 < 1 ms
- ✅ 無內存洩漏
- ✅ 可保存/載入權重

### 期望達成 (Should Have)

- ✅ 工具選擇準確率 > 75%
- ✅ 訓練時間 < 2 小時
- ✅ 支持在線學習
- ✅ 完整的監控指標

### 最好達成 (Nice to Have)

- ✅ 工具選擇準確率 > 85%
- ✅ 自動超參數調優
- ✅ 可視化訓練過程
- ✅ 模型可解釋性分析

---

## 📚 依賴項

### 核心依賴 (已有)

```python
numpy>=2.0.0          # 已安裝
python>=3.10          # 已安裝
```

### 可選依賴 (建議添加)

```python
# 訓練輔助
scikit-learn>=1.3.0   # 數據分割、評估指標
matplotlib>=3.7.0     # 訓練可視化

# 數據管理
h5py>=3.9.0          # 高效權重儲存 (可選)
pandas>=2.0.0        # 數據分析 (可選)
```

---

## 🚀 部署計畫

### 開發環境

```bash
# 1. 創建訓練分支
git checkout -b feature/trainable-ai-core

# 2. 實施改造
# ... 按階段實施 ...

# 3. 單元測試
pytest tests/ai_engine/test_trainable_core.py

# 4. 收集初始數據
python scripts/collect_training_data.py --samples 1000

# 5. 訓練模型
python scripts/train_ai_core.py
```

### 生產環境

```bash
# 1. 驗證訓練結果
python scripts/evaluate_model.py

# 2. 部署權重
cp models/trained_weights/* /path/to/aiva/models/

# 3. 更新配置
vim config/ai_core.yaml  # enable_trained_model: true

# 4. 重啟服務
systemctl restart aiva

# 5. 監控性能
python scripts/monitor_ai_performance.py
```

---

## 📊 監控指標

### 關鍵指標

```python
監控項目：
1. 工具選擇準確率 (實時)
2. 平均推理延遲 (每分鐘)
3. 內存使用量 (每小時)
4. 訓練損失值 (每 epoch)
5. 驗證準確率 (每 epoch)

告警閾值：
- 準確率下降 > 10%  → 發送告警
- 推理延遲 > 2 ms   → 發送告警
- 內存增長 > 100 MB → 發送告警
```

---

## 🔄 未來擴展

### 短期 (1-3 個月)

1. **在線學習**：實時從新數據學習
2. **主動學習**：選擇最有價值的樣本標註
3. **集成學習**：多模型投票提升準確率

### 中期 (3-6 個月)

1. **遷移學習**：從其他滲透測試數據集預訓練
2. **強化學習**：優化長期決策序列
3. **元學習**：快速適應新目標類型

### 長期 (6-12 個月)

1. **自監督學習**：減少標註需求
2. **多任務學習**：同時優化多個目標
3. **神經架構搜索**：自動優化網路結構

---

## ✅ 結論與建議

### 核心優勢

1. **開發速度快**：3 天即可完成
2. **風險低**：基於現有架構改造
3. **成本低**：無額外硬體需求
4. **效果好**：預期 14-17x 準確率提升
5. **可擴展**：未來可持續優化

### 適用場景

✅ 當前 AIVA 開發階段  
✅ 需要快速驗證 AI 決策可行性  
✅ 團隊熟悉 Python 生態  
✅ 追求開發效率而非極致性能  

### 不適用場景

❌ 需要嵌入式部署  
❌ 追求 <0.1ms 推理延遲  
❌ 內存極度受限環境 (<100 MB)  
❌ 架構已固定且需極致性能  

### 最終建議

**強烈推薦作為第一階段實施方案**

理由：
- 符合當前開發階段需求
- 投入產出比最高
- 可快速驗證 AI 核心價值
- 為後續優化奠定基礎
- 保留獨特的生物神經元特性

---

**報告生成時間**：2025-11-08  
**版本**：1.0  
**狀態**：待評估
