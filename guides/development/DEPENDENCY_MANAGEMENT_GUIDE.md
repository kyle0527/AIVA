# AIVA 依賴管理操作指南

> **📋 適用對象**: 開發者、DevOps工程師、系統管理員  
> **🎯 使用場景**: 開發環境設置、依賴問題排查、包管理優化  
> **⏱️ 預計閱讀時間**: 20 分鐘  
> **🔧 技術需求**: Python 3.8+、pip、虛擬環境

## 🚨 **重要通知：ML 依賴混合狀態**

> **⚠️ 當前狀態 (更新日期: 2025年10月31日)**  
> 系統中機器學習依賴處於**混合修復狀態**，部分檔案已採用統一可選依賴框架，部分仍使用傳統直接導入。

### 📊 **混合狀態總覽**

| 狀態 | 檔案數量 | 導入方式 | 型別注解 | 範例檔案 |
|------|----------|----------|----------|----------|
| ✅ **已修復** | 2 個 | `from utilities.optional_deps import deps` | `NDArray` | `bio_neuron_core.py`, `neural_network.py` |
| ⚠️ **未修復** | 16 個 | `import numpy as np` | `np.ndarray` | `memory_manager.py`, `learning_engine.py` 等 |

### 🔍 **相容性分析**

**✅ 技術上完全相容**:
- `NDArray` 本質上是 `np.ndarray` 的型別別名
- 混合使用不會造成運行時錯誤
- 型別檢查器 (Pylance/mypy) 認為兩者相同
- 模組間相互調用無問題

**⚠️ 需要注意的事項**:
- 程式碼風格存在不一致性
- 新開發者可能困惑於兩種導入方式
- 程式碼審查時需要留意風格統一

### 🎯 **開發建議**

| 情況 | 建議做法 | 理由 |
|------|----------|------|
| **新開發 ML 功能** | 使用統一可選依賴框架 | 遵循最新最佳實踐 |
| **修改既有程式碼** | 如無必要，保持現狀 | 避免不必要的修改風險 |
| **大型重構** | 考慮統一至框架模式 | 提升程式碼一致性 |
| **Docker 部署** | 兩種方式均可正常部署 | 已驗證相容性 |

---

## 📑 目錄

1. [🚀 快速開始](#-快速開始)
2. [🤖 ML 依賴混合狀態指南](#-ml-依賴混合狀態指南)
3. [📦 依賴安裝指引](#-依賴安裝指引)
4. [🔧 環境管理](#-環境管理)
5. [⚡ 效能優化](#-效能優化)
6. [🔍 問題診斷](#-問題診斷)
7. [📊 依賴分析](#-依賴分析)
8. [🛠️ 高級配置](#️-高級配置)
9. [📚 最佳實踐](#-最佳實踐)

---

## 🤖 **ML 依賴混合狀態指南**

### 📋 **檔案清單**

#### ✅ **已修復檔案** (使用統一可選依賴框架)
```python
# 導入方式
from utilities.optional_deps import deps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    
# 動態取得模組
np = deps.numpy.module
NDArray = np.ndarray

# 型別注解使用 NDArray
def process_data(data: NDArray) -> NDArray:
    return np.array(data)
```

**已修復檔案列表**:
- `services/core/aiva_core/ai_engine/bio_neuron_core.py`
- `services/core/aiva_core/ai_engine/neural_network.py`

#### ⚠️ **未修復檔案** (傳統直接導入)
```python
# 傳統導入方式
import numpy as np

# 型別注解使用 np.ndarray
def process_data(data: np.ndarray) -> np.ndarray:
    return np.array(data)
```

**未修復檔案列表** (按優先級排序):
1. **核心 AI 引擎** (4個檔案):
   - `services/core/aiva_core/ai_engine/ai_model_manager.py`
   - `services/core/aiva_core/ai_engine/learning_engine.py`
   - `services/core/aiva_core/ai_engine/memory_manager.py`
   - `services/core/aiva_core/ai_engine/performance_enhancements.py`

2. **RAG 向量存儲** (3個檔案):
   - `services/core/aiva_core/rag/postgresql_vector_store.py`
   - `services/core/aiva_core/rag/unified_vector_store.py`
   - `services/core/aiva_core/rag/vector_store.py`

3. **ML 訓練模組** (3個檔案):
   - `services/core/aiva_core/ai_model/train_classifier.py`
   - `services/core/aiva_core/learning/model_trainer.py`
   - `services/core/aiva_core/learning/scalable_bio_trainer.py`

4. **權限與工具** (3個檔案):
   - `services/core/aiva_core/authz/matrix_visualizer.py`
   - `services/core/aiva_core/authz/permission_matrix.py`
   - `services/aiva_common/ai/skill_graph_analyzer.py`

5. **測試與歸檔** (3個檔案):
   - `testing/core/ai_system_connectivity_check.py`
   - `testing/p0_fixes_validation_test.py`
   - `_archive/legacy_components/trainer_legacy.py`

### 🔧 **實用工具**

#### **檢查檔案修復狀態**
```bash
# 檢查特定檔案是否使用統一框架
python -c "
import os
filepath = 'services/core/aiva_core/ai_engine/bio_neuron_core.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
    if 'from utilities.optional_deps import deps' in content:
        print(f'✅ {os.path.basename(filepath)} 已使用統一框架')
    else:
        print(f'⚠️ {os.path.basename(filepath)} 尚未修復')
"
```

#### **測試混合狀態相容性**
```bash
# 測試型別相容性
python -c "
import numpy as np
NDArray = np.ndarray

def old_style(data: 'np.ndarray') -> 'np.ndarray':
    return data

def new_style(data: NDArray) -> NDArray:
    return data

test_data = np.array([1, 2, 3])
result1 = old_style(test_data)
result2 = new_style(test_data)
print('✅ 混合型別注解完全相容')
"
```

### 🎯 **選擇修復策略**

| 情況 | 策略 | 適用場景 |
|------|------|----------|
| 🚀 **積極修復** | 統一所有檔案至可選依賴框架 | 追求程式碼一致性、長期維護 |
| 🎯 **選擇性修復** | 只修復有問題的檔案 | 平衡修復成本與收益 |
| 🛡️ **保守策略** | 維持現狀，新程式碼使用框架 | 穩定性優先、風險控制 |

**建議**: 基於 Docker 部署經驗，系統已穩定運行，建議採用**保守策略**。

---

## 🚀 **快速開始**

### **檢查當前環境**
```bash
# 確認虛擬環境
& "C:/D/fold7/AIVA-git/.venv/Scripts/python.exe" --version

# 檢查核心依賴
& "C:/D/fold7/AIVA-git/.venv/Scripts/python.exe" -c "import fastapi, pydantic, redis; print('✅ 核心依賴正常')"
```

### **執行系統檢查**
```bash
# 完整系統測試
& "C:/D/fold7/AIVA-git/.venv/Scripts/python.exe" testing\common\complete_system_check.py
```

---

## 📦 **依賴安裝指引**

### **1. 基礎開發環境** (已完成 ✅)
```bash
# 已安裝，無需重複執行
pip install -e .
```

### **2. AI 功能依賴** (按需安裝)
```bash
# 機器學習基礎
pip install scikit-learn>=1.3.0 numpy>=1.24.0

# 深度學習 (僅 DQN/PPO 需要)
pip install torch>=2.1.0 torchvision>=0.16.0

# 強化學習環境
pip install gymnasium>=0.29.0
```

### **3. 微服務通訊** (可選)
```bash
# gRPC 支援
pip install grpcio>=1.60.0 grpcio-tools>=1.60.0 protobuf>=4.25.0
```

### **4. 監控和文件** (可選)
```bash
# 監控工具
pip install prometheus-client>=0.20

# PDF 報告
pip install reportlab>=3.6

# 型別提示
pip install types-requests>=2.31.0
```

---

## 🛠️ **Optional Dependency 框架** (新增)

### **統一依賴管理器**
AIVA 使用統一的 Optional Dependency 框架來處理可選依賴，避免導入錯誤：

```python
# utilities/optional_deps.py
from utilities.optional_deps import OptionalDependencyManager

# 註冊可選依賴
deps = OptionalDependencyManager()
deps.register('plotly', ['plotly'])
deps.register('pandas', ['pandas'])
deps.register('sklearn', ['scikit-learn'])

# 檢查依賴可用性
if deps.is_available('plotly'):
    import plotly.graph_objects as go
else:
    # 自動使用 Mock 實現
    go = deps.get_or_mock('plotly').graph_objects
```

### **Mock 實現模式**
當可選依賴不可用時，框架提供 Mock 對象：

```python
# 範例：Plotly Mock 實現
class MockFigure:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_trace(self, *args, **kwargs):
        return self
    
    def update_layout(self, *args, **kwargs):
        return self
    
    def show(self, *args, **kwargs):
        print("Mock figure display (plotly not installed)")

# 範例：Pandas Mock 實現  
class MockDataFrame:
    def __init__(self, *args, **kwargs):
        self.data = {}
    
    def to_dict(self, *args, **kwargs):
        return {}
    
    def to_json(self, *args, **kwargs):
        return "{}"
```

### **使用指南**
1. **檢查依賴**: 使用 `deps.is_available()` 檢查
2. **獲取模組**: 使用 `deps.get_or_mock()` 安全導入
3. **Mock 處理**: 自動回退到 Mock 實現，不會中斷系統運行
4. **日誌記錄**: 自動記錄缺失依賴的警告信息

---

## ⚠️ **常見問題處理**

### **問題 1: Optional Dependencies 缺失**
```python
# 問題：導入錯誤 "ModuleNotFoundError: No module named 'plotly'"
# 解決方案：使用統一框架
from utilities.optional_deps import OptionalDependencyManager
deps = OptionalDependencyManager()
plotly = deps.get_or_mock('plotly')  # 自動處理缺失依賴
```

### **問題 2: FastAPI 循環導入**
```bash
# 解決方案
pip uninstall fastapi -y
pip install fastapi==0.115.0
```

### **問題 3: 模組導入失敗**  
```bash
# 重新安裝專案
pip install -e .
```

### **問題 4: Services 導入路徑錯誤**
```python
# 問題：從錯誤位置導入共享模型
from services.core.models import ConfigUpdatePayload  # ❌ 錯誤

# 解決方案：使用正確路徑
from services.aiva_common.schemas import ConfigUpdatePayload  # ✅ 正確
```

### **問題 5: 配置屬性缺失**
- 檢查 `services/aiva_common/config/unified_config.py`
- 確認所有必要屬性已定義

### **問題 6: Docker 服務未啟動**
```bash
# 啟動基礎服務 (需要 Docker)
docker-compose up -d redis rabbitmq postgres neo4j
```

---

## 🔍 **依賴健康檢查**

### **定期檢查項目**
```bash
# 1. 檢查過時的套件
pip list --outdated

# 2. 檢查安全漏洞
pip audit

# 3. 檢查依賴樹
pip show --verbose fastapi
```

### **清理未使用依賴**
```bash
# 安裝清理工具
pip install pip-autoremove

# 清理未使用的套件 (謹慎使用)
pip-autoremove -y
```

---

## 📊 **版本管理策略**

### **核心依賴版本鎖定** (必須安裝)
| 套件 | 鎖定版本 | 原因 |
|------|----------|------|
| `fastapi` | 0.115.0 | 穩定性 |
| `pydantic` | 2.12.3 | 相容性 |
| `sqlalchemy` | 2.0.44 | 功能完整 |

### **Optional Dependencies** (按需安裝)
| 套件類別 | 套件名稱 | 建議版本 | Mock 支援 |
|----------|----------|----------|-----------|
| **視覺化** | plotly | >=5.17.0 | ✅ MockFigure |
| **數據處理** | pandas | >=2.1.0 | ✅ MockDataFrame |
| **機器學習** | scikit-learn | >=1.3.0 | ✅ MockModel |
| **深度學習** | torch | >=2.1.0 | ✅ MockTensor |
| **數值計算** | numpy | >=1.24.0 | ✅ MockArray |

### **允許彈性更新的套件**
- 開發工具 (black, ruff, mypy)
- 監控工具 (psutil)
- 文件工具 (types-*)
- Optional dependencies (由框架自動處理)

---

## 🎯 **開發階段依賴指引**

### **階段 1: 核心功能開發** (目前階段)
- ✅ 基礎 Web 框架
- ✅ 資料庫連接
- ✅ 訊息佇列
- ⏳ Docker 服務設置

### **階段 2: AI 功能整合**
- 🔄 安裝機器學習依賴
- 🔄 配置深度學習框架
- 🔄 整合 RL 環境

### **階段 3: 生產部署**
- 🔄 監控工具整合
- 🔄 安全性強化
- 🔄 性能優化工具

---

## 📋 **檢查清單**

### **新開發者加入**
- [ ] 檢查 Python 版本 (3.13.9)
- [ ] 創建虛擬環境
- [ ] 安裝基礎依賴 (`pip install -e .`)
- [ ] 測試 Optional Dependency 框架 (`python -c "from utilities.optional_deps import OptionalDependencyManager; print('框架運行正常')"`)
- [ ] 執行系統檢查測試
- [ ] 確認配置文件正確

### **功能開發前**
- [ ] 檢查相關依賴是否已安裝
- [ ] 使用 `OptionalDependencyManager.is_available()` 驗證可選依賴
- [ ] 執行依賴健康檢查
- [ ] 更新文件記錄

### **功能完成後**
- [ ] 檢查是否引入新依賴
- [ ] 如果添加 optional dependency，註冊到 `OptionalDependencyManager`
- [ ] 實現相應的 Mock 類別 (如果需要)
- [ ] 更新 `pyproject.toml`
- [ ] 更新依賴分析報告
- [ ] 執行完整系統測試

### **Optional Dependency 開發檢查**
- [ ] 新依賴已註冊到 `OptionalDependencyManager`
- [ ] Mock 實現提供核心功能的無操作版本
- [ ] 測試有無依賴兩種情況下的程式運行
- [ ] 添加適當的日誌記錄和用戶提示

---

## 🔗 **相關資源**

### **核心文件**
- [Optional Dependency 框架](../../utilities/optional_deps.py) - 統一依賴管理器實現
- [專案配置](../../pyproject.toml) - 主要依賴配置
- [需求清單](../../requirements.txt) - 完整依賴列表

### **實現範例**
- [Matrix Visualizer](../../services/core/aiva_core/authz/matrix_visualizer.py) - Plotly 整合範例
- [Permission Matrix](../../services/core/aiva_core/authz/permission_matrix.py) - Pandas 整合範例
- [Services Core Init](../../services/core/aiva_core/__init__.py) - 導入路徑修復範例

### **文件和測試**
- [依賴分析詳細報告](./DEPENDENCY_ANALYSIS_REPORT.md)
- [系統測試腳本](../../testing/common/complete_system_check.py)
- [導入問題修復指南](../troubleshooting/IMPORT_ISSUES_RESOLUTION_GUIDE.md) (待建立)