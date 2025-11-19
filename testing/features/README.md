# 🛠️ AIVA 功能特性測試

這個目錄包含 AIVA 項目的功能特性測試，專注於驗證各種安全功能和攻擊檢測能力。

## 📁 目錄結構

```
testing/
├── features/               # 功能特性測試 (本目錄)
│   ├── detectors/         # 檢測器測試
│   ├── testers/           # 測試工具
│   └── [測試檔案]
├── common/                 # 通用測試工具
├── core/                   # AI 核心功能測試
├── integration/            # 集成測試
├── performance/            # 性能測試
├── scan/                   # 掃描和滲透測試
└── README.md
```

## 📄 本目錄檔案說明

### 🎯 核心功能測試
- **`aiva_final_test.py`** - AIVA 系統綜合功能測試
- **`test_exploit_functionality.py`** - 漏洞利用功能測試（包含 SQLi、XSS 等）
- **`real_attack_executor.py`** - 真實攻擊執行器測試

### 🔍 檢測器測試 (`detectors/`)
- **`test_detector.py`** - 加密檢測器功能測試

### 🧪 測試工具 (`testers/`)
- **`cross_user_tester.py`** - 跨用戶權限測試工具（IDOR 檢測）
- **`vertical_escalation_tester.py`** - 垂直權限升級測試工具

## 🚀 使用方法

### 執行功能特性測試
```bash
# 執行所有功能測試
pytest testing/features/ -v

# 執行綜合系統測試
python testing/features/aiva_final_test.py

# 執行漏洞利用功能測試
python testing/features/test_exploit_functionality.py

# 執行真實攻擊測試
python testing/features/real_attack_executor.py
```

### 執行檢測器測試
```bash
# 執行檢測器測試
pytest testing/features/detectors/ -v

# 執行加密檢測器測試
pytest testing/features/detectors/test_detector.py -v
```

### 執行權限測試工具
```bash
# 跨用戶權限測試 (IDOR)
python testing/features/testers/cross_user_tester.py

# 垂直權限升級測試
python testing/features/testers/vertical_escalation_tester.py
```

## 🛠️ 功能測試組件

### AIVA 綜合測試 (`aiva_final_test.py`)
- **XSS 掃描器組件測試**
  - XssPayloadGenerator - XSS 有效載荷生成
  - DomXssDetector - DOM XSS 檢測
- **SQLi 掃描器組件測試**
  - DetectionModels - SQL 注入檢測模型
  - SqliTaskQueue - SQLi 任務隊列
- **系統集成驗證**
- **編碼問題避免機制**

### 漏洞利用功能測試 (`test_exploit_functionality.py`)
- **漏洞利用管理器測試**
- **多種攻擊類型測試**
  - SQL 注入攻擊
  - XSS 攻擊
  - CSRF 攻擊
- **Juice Shop 靶場集成測試**
- **異步攻擊執行測試**

### 真實攻擊執行器 (`real_attack_executor.py`)
- **模擬攻擊轉真實工具執行**
- **多種攻擊工具集成**
  - Nmap 網絡掃描
  - SQLmap SQL 注入
  - Dirb 目錄掃描
  - Hydra 密碼爆破
- **靶場環境真實滲透測試**
- **攻擊結果分析和報告**

### 加密檢測器測試 (`detectors/test_detector.py`)
- **CryptoDetector 功能驗證**
- **弱加密算法檢測**
  - MD5 hash 檢測
  - 私鑰洩漏檢測
  - 不安全隨機數檢測
- **代碼安全掃描測試**

### 權限測試工具

#### 跨用戶測試 (`testers/cross_user_tester.py`)
- **IDOR（不安全直接對象引用）檢測**
- **水平權限升級測試**
- **OWASP WSTG-ATHZ-04 標準實現**
- **跨用戶資源訪問測試**

#### 垂直升級測試 (`testers/vertical_escalation_tester.py`)
- **垂直權限升級檢測**
- **管理員權限繞過測試**
- **權限邊界驗證**

## 📝 編寫新的功能測試

### 測試檔案命名規範
- 功能測試: `test_*.py`
- 檢測器測試: `*_detector.py` 或 `test_detector.py`
- 測試工具: `*_tester.py`
- 執行器: `*_executor.py`

### 功能測試結構範例
```python
#!/usr/bin/env python3
"""
功能測試模板
"""
import pytest
import asyncio
from services.features import FeatureModule

class TestFeatureFunction:
    
    @pytest.fixture
    def feature_module(self):
        return FeatureModule()
    
    def test_detection_capability(self, feature_module):
        """測試檢測能力"""
        # Arrange
        test_payload = create_test_payload()
        
        # Act
        result = feature_module.detect(test_payload)
        
        # Assert
        assert result.detected is True
        assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_async_exploit(self, feature_module):
        """測試異步攻擊功能"""
        result = await feature_module.execute_exploit()
        assert result.success is True
```

### 檢測器測試範例
```python
def test_crypto_detector():
    """加密檢測器測試"""
    detector = CryptoDetector()
    code = "hash = MD5('password'); verify=False;"
    
    findings = detector.detect(code, task_id="t1", scan_id="s1")
    
    assert len(findings) >= 2  # 至少檢測到 MD5 和 verify=False
    assert any("MD5" in f.description for f in findings)
```

## 🔧 測試環境和配置

### 靶場環境要求
- **Juice Shop** - http://localhost:3000
- **DVWA** - 用於 Web 漏洞測試
- **測試網路隔離**

### 測試配置

**研發階段**：測試使用預設值，無需環境變數。

預設配置：
```python
TEST_MODE = True
TARGET_URL = "http://localhost:3000"
ATTACK_MODE = "safe"
MAX_THREADS = 5
```

### 工具依賴
```bash
# 安裝測試所需工具
pip install httpx requests pytest-asyncio

# 外部工具（用於真實攻擊測試）
# nmap, sqlmap, dirb, hydra
```

## 📊 測試策略和覆蓋率

### 功能測試層級
1. **單元測試** - 個別檢測器和組件
2. **集成測試** - 功能模組間協作
3. **系統測試** - 完整攻擊鏈測試
4. **真實環境測試** - 靶場環境驗證

### 覆蓋率目標
- **檢測器功能**: > 95%
- **攻擊模組**: > 85%
- **權限測試**: > 90%
- **集成測試**: > 80%

## 🚨 安全和倫理考慮

### 測試安全原則
- **僅在授權環境測試**
- **使用隔離的測試網路**
- **避免對生產系統造成影響**
- **遵循負責任披露原則**

### 攻擊測試限制
- 所有真實攻擊僅針對受控靶場
- 設置攻擊強度和頻率限制
- 測試完成後清理測試數據

### 合規要求
- 遵循 OWASP 測試指南
- 符合當地法律法規
- 獲得適當的測試授權

## 🔄 持續集成和自動化

### CI/CD 中的功能測試
1. **快速檢測器測試** - 基本功能驗證
2. **模擬攻擊測試** - 安全功能測試  
3. **靶場集成測試** - 真實環境驗證
4. **安全回歸測試** - 確保無誤報

### 自動化測試流程
```yaml
# CI/CD 配置範例
- name: Setup Test Environment
  run: docker-compose up -d juiceshop

- name: Run Feature Tests
  run: |
    pytest testing/features/ -v
    python testing/features/aiva_final_test.py

- name: Security Validation
  run: python testing/features/real_attack_executor.py --safe-mode
```

---

**目錄更新**: 2025-11-12  
**維護者**: AIVA Security Team  
**測試重點**: 功能檢測 + 攻擊模擬 + 權限驗證