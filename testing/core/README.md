# 🧠 AIVA 核心功能測試

這個目錄包含 AIVA 項目的 AI 核心功能測試，專注於驗證人工智能模組的正確性和性能。

## 📁 目錄結構

```
testing/
├── core/                   # AI 核心功能測試 (本目錄)
├── common/                 # 通用測試工具
├── features/               # 功能特性測試
├── integration/            # 集成測試
├── performance/            # 性能測試
├── scan/                   # 掃描和滲透測試
└── README.md
```

## 📄 本目錄檔案說明

### AI 自主測試系統
- **`ai_autonomous_testing_loop.py`** - AI 完全自主的測試、學習、優化閉環系統
- **`enhanced_real_ai_attack_system.py`** - 增強版真實 AI 攻擊系統測試

### AI 核心功能測試
- **`test_ai_analysis_system.py`** - AI 分析系統功能測試
- **`test_ai_components.py`** - AI 組件功能測試
- **`test_intelligent_logic.py`** - 智能邏輯測試

### AI 集成和連接測試
- **`ai_integration_test.py`** - AI 模組集成測試
- **`ai_system_connectivity_check.py`** - AI 系統連接性檢查
- **`ai_security_test.py`** - AI 安全性測試

## 🚀 使用方法

### 執行核心 AI 測試
```bash
# 執行所有核心測試
pytest testing/core/ -v

# 執行特定測試檔案
pytest testing/core/test_ai_analysis_system.py -v

# 執行 AI 自主測試循環
python testing/core/ai_autonomous_testing_loop.py

# 執行 AI 集成測試
python testing/core/ai_integration_test.py
```

### AI 系統連接性檢查
```bash
# 檢查 AI 系統連接性
python testing/core/ai_system_connectivity_check.py

# 執行 AI 安全測試
python testing/core/ai_security_test.py
```

### 增強攻擊系統測試
```bash
# 執行增強版 AI 攻擊系統
python testing/core/enhanced_real_ai_attack_system.py
```

## 🛠️ 測試組件功能

### AI 自主測試系統 (`ai_autonomous_testing_loop.py`)
- **自主發現和測試靶場**
- **動態調整測試策略**
- **實時學習和優化**
- **自動化改進建議**
- **持續性能監控**

### AI 分析系統 (`test_ai_analysis_system.py`)
- **AI 分析引擎功能驗證**
- **數據處理流程測試**
- **模型預測準確性測試**

### AI 組件測試 (`test_ai_components.py`)
- **AI 模組獨立功能測試**
- **組件間交互測試**
- **API 接口測試**

### 智能邏輯測試 (`test_intelligent_logic.py`)
- **決策邏輯驗證**
- **策略生成測試**
- **學習算法測試**

### AI 集成測試 (`ai_integration_test.py`)
- **多模組集成測試**
- **端到端工作流測試**
- **系統整體性能測試**

### 系統連接性檢查 (`ai_system_connectivity_check.py`)
- **AI 與系統組件連接驗證**
- **命令執行能力測試**
- **權限和安全檢查**

### AI 安全測試 (`ai_security_test.py`)
- **AI 模型安全性測試**
- **對抗樣本檢測**
- **權限邊界測試**

### 增強攻擊系統 (`enhanced_real_ai_attack_system.py`)
- **真實環境攻擊模擬**
- **AI 驅動的攻擊策略**
- **自適應攻擊技術**

## 📝 編寫新的 AI 測試

### 測試檔案命名規範
- AI 核心測試: `test_ai_*.py`
- AI 系統測試: `ai_*_test.py`
- 特定功能測試: `test_*.py`

### AI 測試結構範例
```python
#!/usr/bin/env python3
"""
AI 功能測試模板
"""
import pytest
from services.core.aiva_core import AIVACore

class TestAIFunction:
    
    @pytest.fixture
    def ai_core(self):
        return AIVACore()
    
    def test_ai_decision_making(self, ai_core):
        """測試 AI 決策制定"""
        # Arrange
        test_scenario = create_test_scenario()
        
        # Act
        decision = ai_core.make_decision(test_scenario)
        
        # Assert
        assert decision.confidence > 0.7
        assert decision.action is not None
    
    def test_ai_learning(self, ai_core):
        """測試 AI 學習能力"""
        # 測試學習前後的性能差異
        pass
```

### 測試環境設置
```python
# 設置 AI 測試環境
import os
os.environ["AIVA_OFFLINE_MODE"] = "true"
os.environ["AIVA_TEST_MODE"] = "true"
os.environ["AIVA_AI_MODEL"] = "test_model"
```

## 🔧 配置和環境

### 測試環境變數
- **`AIVA_OFFLINE_MODE`** - 離線測試模式
- **`AIVA_TEST_MODE`** - 測試模式標誌
- **`AIVA_AI_MODEL`** - 使用的 AI 模型
- **`AIVA_CORE_MONITOR_INTERVAL`** - 監控間隔設置

### AI 模型配置
- 使用輕量級測試模型
- 配置模擬數據源
- 設置測試環境隔離

## 📊 測試策略和覆蓋率

### 測試層級
1. **單元測試** - 個別 AI 組件功能
2. **集成測試** - AI 模組間交互
3. **系統測試** - 完整 AI 工作流程
4. **性能測試** - AI 響應時間和資源使用

### 覆蓋率目標
- **AI 核心邏輯**: > 95%
- **AI 集成功能**: > 85%
- **AI 安全功能**: > 90%
- **自主測試系統**: > 80%

## 🚨 測試注意事項

### 安全考慮
- 所有攻擊測試僅在受控環境執行
- 使用隔離的測試網路
- 避免對生產系統造成影響

### 性能考慮
- AI 測試可能需要較長執行時間
- 某些測試需要 GPU 資源
- 記憶體使用量較高

### 依賴要求
- TensorFlow/PyTorch (AI 模型)
- 測試資料集
- 模擬環境設置

## 🔄 持續集成

### CI/CD 中的 AI 測試流程
1. **快速驗證** - 基本 AI 功能測試
2. **模型驗證** - AI 模型性能測試
3. **集成測試** - 完整系統測試
4. **安全掃描** - AI 安全性驗證

---

**目錄更新**: 2025-11-12  
**維護者**: AIVA AI Team  
**測試重點**: AI 核心功能 + 自主學習 + 安全驗證