# BioNeuronCore AI 架構分析報告

**分析時間**: 2025-10-17  
**目的**: 確認神經網路數組結構是否需要改變以支持訓練

---

## 1. 當前架構分析

### 1.1 神經網路結構

```
ScalableBioNet (500萬參數)
├── fc1: numpy.ndarray (1024, 2048)          # 全連接層 1
│   └── 參數量: 2,097,152
├── spiking1: BiologicalSpikingLayer         # 生物尖峰層
│   ├── weights: numpy.ndarray (2048, 1024)
│   └── 參數量: 2,097,152
└── fc2: numpy.ndarray (1024, num_tools)     # 全連接層 2
    └── 參數量: 1024 × 工具數量
```

**總參數**: ~4,196,352 (7個工具時)

### 1.2 數據類型

```python
# 當前實現
self.fc1 = np.random.randn(input_size, self.hidden_size_1)      # ndarray
self.spiking1.weights = np.random.randn(input_size, output_size) # ndarray
self.fc2 = np.random.randn(self.hidden_size_2, num_tools)       # ndarray
```

**關鍵發現**: 
- ✅ 所有權重都是 **numpy.ndarray** (不是 Layer 對象)
- ✅ 可以直接修改和訓練
- ✅ 支持矩陣運算

---

## 2. 訓練需求分析

### 2.1 反向傳播需求

為了訓練神經網路，需要：

1. **前向傳播** ✅ 已實現
   ```python
   def forward(self, x: np.ndarray) -> np.ndarray:
       x = np.tanh(x @ self.fc1)
       x = self.spiking1.forward(x)
       decision_potential = x @ self.fc2
       return self._softmax(decision_potential)
   ```

2. **梯度計算** ❌ 未實現
   - 需要保存中間層輸出
   - 需要計算梯度

3. **權重更新** ❌ 未實現
   - 需要更新 fc1, spiking1.weights, fc2

### 2.2 當前缺失的功能

```python
# 缺失 1: 中間層輸出保存
self.fc1_output = None  # 需要保存用於反向傳播
self.spiking1_output = None

# 缺失 2: 偏置項
# fc1 和 fc2 沒有偏置項 (bias)

# 缺失 3: 反向傳播方法
def backward(self, grad_output):  # 不存在
    pass

# 缺失 4: 優化器
# 沒有 Adam, SGD 等優化器
```

---

## 3. 問題診斷

### 3.1 訓練腳本錯誤

```python
# train_cli_matching.py 第 127 行
agent.decision_core.fc2.weight -= learning_rate * ...
                        ^^^^^^
# 錯誤: fc2 是 ndarray，沒有 .weight 屬性
```

**原因**: 
- fc2 直接是 ndarray，不是有 weight 屬性的 Layer 對象
- 應該直接操作 `agent.decision_core.fc2`

### 3.2 生物尖峰層的訓練問題

```python
class BiologicalSpikingLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 返回的是 0/1 的尖峰信號
        spikes = (potential > self.threshold).astype(int)
```

**問題**:
- ❌ 輸出是離散的 (0 或 1)，不可微分
- ❌ 無法進行標準反向傳播
- ⚠️ 需要特殊的梯度近似方法

---

## 4. 解決方案建議

### 方案 A: 最小改動 - 直接修改數組 ✅ **推薦**

**優點**:
- 不改變現有架構
- 快速實現
- 保持兼容性

**實現**:
```python
# 1. 保存中間輸出
def forward_with_cache(self, x):
    self.fc1_input = x
    self.fc1_output = np.tanh(x @ self.fc1)
    self.spiking_output = self.spiking1.forward(self.fc1_output)
    self.fc2_output = self.spiking_output @ self.fc2
    return self._softmax(self.fc2_output)

# 2. 簡化的反向傳播（忽略尖峰層梯度）
def update_weights(self, grad_output, learning_rate):
    # 更新 fc2
    self.fc2 -= learning_rate * np.outer(self.spiking_output, grad_output)
    
    # 忽略尖峰層的不可微分性
    # 只更新 fc1 和 fc2
```

### 方案 B: 完整重構 - 添加訓練框架 ⚠️ **工程量大**

**優點**:
- 完整的訓練能力
- 支持複雜優化器

**缺點**:
- 需要大幅修改代碼
- 可能破壞現有功能

**實現**:
```python
class TrainableScalableBioNet(ScalableBioNet):
    def __init__(self, ...):
        super().__init__(...)
        # 添加偏置項
        self.fc1_bias = np.zeros(self.hidden_size_1)
        self.fc2_bias = np.zeros(num_tools)
        
    def backward(self, loss_grad):
        # 完整的反向傳播實現
        pass
        
    def update(self, optimizer):
        # 使用優化器更新
        pass
```

### 方案 C: 使用預訓練嵌入 + 微調 🎯 **實用**

**策略**:
- 不訓練整個神經網路
- 只訓練任務到工具的映射
- 使用簡單的查找表或小型分類器

**實現**:
```python
# 任務描述 -> 工具名稱的映射表
task_to_tool_map = {
    "掃描": "ScanTrigger",
    "SQL注入": "SQLiDetector",
    # ...
}

# 或使用簡單的 TF-IDF + 最近鄰
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
```

---

## 5. 建議行動方案

### 立即執行 (方案 A + C 組合)

1. **保持現有神經網路不變** ✅
   - fc1, spiking1, fc2 維持原樣
   - 不修改數組結構

2. **添加輕量級訓練方法** ✅
   ```python
   # 新增文件: services/core/aiva_core/ai_engine/simple_trainer.py
   
   class SimpleTaskMatcher:
       """簡單的任務-工具配對器（不需要訓練神經網路）"""
       
       def __init__(self, tools):
           self.tools = tools
           self.keyword_map = {
               "掃描": "ScanTrigger",
               "scan": "ScanTrigger",
               "SQL注入": "SQLiDetector",
               "sqli": "SQLiDetector",
               "XSS": "XSSDetector",
               "xss": "XSSDetector",
               "分析": "CodeAnalyzer",
               "analyze": "CodeAnalyzer",
               "讀取": "CodeReader",
               "read": "CodeReader",
               "寫入": "CodeWriter",
               "write": "CodeWriter",
               "報告": "ReportGenerator",
               "report": "ReportGenerator",
           }
       
       def match(self, task_description):
           """基於關鍵字匹配工具"""
           task_lower = task_description.lower()
           
           for keyword, tool_name in self.keyword_map.items():
               if keyword.lower() in task_lower:
                   return tool_name
           
           # 默認工具
           return "CodeReader"
   ```

3. **集成到 BioNeuronRAGAgent** ✅
   ```python
   # 在 invoke 方法中添加預處理
   def invoke(self, task_description: str):
       # 使用簡單匹配器
       matched_tool = self.simple_matcher.match(task_description)
       
       # 然後使用神經網路確認
       neural_decision = self.decision_core.forward(...)
       
       # 組合決策
       if confidence < 0.7:
           return matched_tool  # 信心度低時使用關鍵字匹配
       else:
           return neural_decision  # 信心度高時使用神經網路
   ```

---

## 6. 結論

### 是否需要改變數組？

**答案: 不需要 ❌**

**理由**:

1. **當前數組結構適合直接操作**
   - numpy.ndarray 可以直接修改
   - 支持矩陣運算
   - 不需要重構

2. **生物尖峰層不適合標準訓練**
   - 離散輸出 (0/1) 不可微分
   - 標準反向傳播無法工作
   - 需要特殊處理

3. **更好的解決方案**
   - 使用關鍵字匹配 (簡單、快速、準確)
   - 保留神經網路作為驗證
   - 組合決策提高準確率

### 推薦實施步驟

1. ✅ **創建 SimpleTaskMatcher** (關鍵字匹配器)
2. ✅ **集成到 BioNeuronRAGAgent** 
3. ✅ **測試配對準確率**
4. 🔄 **收集實際使用數據**
5. 🔄 **迭代優化關鍵字映射**

### 未來可選

- 如果需要真正的神經網路訓練，考慮：
  - 使用 PyTorch/TensorFlow 重寫決策核心
  - 或使用 SVM/Random Forest 等可訓練的分類器
  - 但當前的關鍵字匹配已足夠使用

---

**最終建議**: 保持數組結構不變，使用關鍵字匹配 + 神經網路驗證的混合方案。
