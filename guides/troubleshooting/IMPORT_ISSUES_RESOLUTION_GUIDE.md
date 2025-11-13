# 📥 **導入問題修復指南** ✅ 11/10驗證 (10/31實測驗證)

> **版本**: 1.0.0  
> **最後更新**: 2025-10-31  
> **狀態**: ✅ 已完成並驗證  

---

## 🎯 **指南目標**

本指南提供系統性的方法來診斷和修復 AIVA 專案中的導入問題，包括：
- Services 架構導入路徑問題
- Optional dependencies 缺失問題  
- 循環導入問題
- 模組找不到問題

---

## 🔍 **問題診斷流程**

### **Step 1: 識別問題類型**
```bash
# 常見錯誤訊息分類
ModuleNotFoundError: No module named 'xxx'           # 缺失依賴
ImportError: cannot import name 'xxx' from 'yyy'    # 導入路徑錯誤
ImportError: attempted relative import              # 相對導入問題
RecursionError: maximum recursion depth exceeded   # 循環導入
```

### **Step 2: 快速診斷命令**
```bash
# 檢查模組是否存在
python -c "import sys; print('Python version:', sys.version_info[:2])"

# 檢查導入路徑 (使用臨時文件避免 PowerShell 引號問題)
echo "import sys; print('sys.path:', sys.path)" > temp_test.py && python temp_test.py && del temp_test.py

# 檢查 OptionalDependencyManager 可用性
python -c "from utilities.optional_deps import OptionalDependencyManager; print('OptionalDependencyManager available')"

# 檢查特定模組可用性
python -c "import importlib.util; spec = importlib.util.find_spec('plotly'); print('plotly available:', spec is not None)"
```

---

## ⚡ **Services 導入路徑修復**

### **問題類型**: 錯誤的服務間導入路徑

#### **症狀**
```python
# ❌ 錯誤：從錯誤位置導入共享模型
from services.core.models import ConfigUpdatePayload
ImportError: cannot import name 'ConfigUpdatePayload' from 'services.core.models'
```

#### **修復方法**
```python
# ✅ 正確：使用統一的共享架構
from services.aiva_common.schemas import ConfigUpdatePayload

# ✅ 正確：從正確的層級導入
from services.aiva_common.schemas import (
    ConfigUpdatePayload,
    FindingPayload, 
    ResponsePayload,
    ErrorPayload
)
```

#### **架構規則**
- `services.aiva_common.schemas` - 共享數據模型
- `services.core` - 核心服務專用邏輯
- `services.integration` - 整合服務專用邏輯
- 避免跨層級直接導入

---

## 🔧 **Optional Dependencies 修復**

### **問題類型**: 可選依賴缺失導致系統崩潰

#### **症狀**
```python
import plotly.graph_objects as go
ModuleNotFoundError: No module named 'plotly'
```

#### **修復方法 - 使用統一框架**
```python
# ❌ 直接導入 (會導致錯誤)
import plotly.graph_objects as go

# ✅ 使用 Optional Dependency 框架
from utilities.optional_deps import OptionalDependencyManager

deps = OptionalDependencyManager()
deps.register('plotly', ['plotly'])

if deps.is_available('plotly'):
    import plotly.graph_objects as go
else:
    # 自動使用 Mock 實現
    go = deps.get_or_mock('plotly').graph_objects
```

#### **Mock 實現範例**
```python
# 為缺失的依賴提供 Mock 類別
class MockFigure:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_trace(self, *args, **kwargs):
        return self
    
    def update_layout(self, *args, **kwargs):
        return self
    
    def show(self, *args, **kwargs):
        print("Mock figure display (plotly not installed)")
        
class MockDataFrame:
    def __init__(self, *args, **kwargs):
        self.data = {}
    
    def to_dict(self, *args, **kwargs):
        return {}
    
    def to_json(self, *args, **kwargs):
        return "{}"
```

---

## 🔄 **循環導入修復**

### **問題類型**: 模組間相互導入造成循環依賴

#### **診斷方法**
```python
# 使用 importlib 檢查導入鏈
import importlib.util
import sys

def find_import_cycle(module_name):
    if module_name in sys.modules:
        print(f"{module_name} already imported")
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"Module {module_name} not found")
```

#### **修復策略**
1. **重構導入順序**: 將共同依賴提取到上層模組
2. **延遲導入**: 在函數內部進行導入
3. **介面抽象**: 使用 Protocol 或 ABC 打破循環

```python
# ✅ 延遲導入解決循環依賴
def get_processor():
    from services.core.processor import DataProcessor
    return DataProcessor()

# ✅ 使用 TYPE_CHECKING 條件導入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from services.core.models import SomeModel
```

---

## 📦 **常見依賴修復配方**

### **Plotly 相關**
```python
from utilities.optional_deps import OptionalDependencyManager
deps = OptionalDependencyManager()
deps.register('plotly', ['plotly'])
go = deps.get_or_mock('plotly').graph_objects
```

### **Pandas 相關**  
```python
deps.register('pandas', ['pandas'])
pd = deps.get_or_mock('pandas')
```

### **Scikit-learn 相關**
```python
deps.register('sklearn', ['scikit-learn'])
sklearn = deps.get_or_mock('sklearn')
```

### **Deep Learning 相關**
```python
deps.register('torch', ['torch'])
torch = deps.get_or_mock('torch')
```

---

## 🛠️ **系統性修復工具**

### **批量掃描腳本**
```python
#!/usr/bin/env python3
"""批量掃描和修復導入問題"""

import os
import re
from pathlib import Path

def scan_import_issues(directory):
    """掃描目錄中的導入問題"""
    issues = []
    
    for py_file in Path(directory).rglob("*.py"):
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 檢查常見問題模式
        patterns = {
            'wrong_service_import': r'from services\.core\.models import',
            'direct_optional_import': r'import (plotly|pandas|sklearn|torch)',
            'missing_optional_check': r'import (plotly|pandas|sklearn|torch)(?!.*optional_deps)'
        }
        
        for issue_type, pattern in patterns.items():
            if re.search(pattern, content):
                issues.append((py_file, issue_type))
    
    return issues

# 使用範例
issues = scan_import_issues("services/")
for file_path, issue in issues:
    print(f"{file_path}: {issue}")
```

### **自動修復腳本**
```python
def fix_service_imports(file_path):
    """自動修復服務導入路徑"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修復常見錯誤路徑
    replacements = {
        'from services.core.models import': 'from services.aiva_common.schemas import',
        'from services.integration.models import': 'from services.aiva_common.schemas import'
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
```

---

## ✅ **修復後驗證**

### **單元測試驗證**
```python
def test_imports():
    """測試所有關鍵導入是否正常"""
    try:
        from services.aiva_common.schemas import ConfigUpdatePayload
        from utilities.optional_deps import OptionalDependencyManager
        print("✅ 核心導入測試通過")
    except ImportError as e:
        print(f"❌ 導入測試失敗: {e}")

def test_optional_deps():
    """測試可選依賴框架"""
    deps = OptionalDependencyManager()
    deps.register('plotly', ['plotly'])
    
    # 測試可用性檢查
    is_available = deps.is_available('plotly')
    print(f"Plotly 可用: {is_available}")
    
    # 測試安全獲取
    plotly_module = deps.get_or_mock('plotly')
    print(f"Plotly 模組獲取: {'✅ 成功' if plotly_module else '❌ 失敗'}")
```

### **系統健康檢查**
```bash
# 執行完整導入測試 (使用臨時文件)
echo "from utilities.optional_deps import OptionalDependencyManager; from services.aiva_common.schemas import ConfigUpdatePayload; print('✅ 所有關鍵導入正常')" > temp_import_test.py && python temp_import_test.py && del temp_import_test.py

# 檢查服務啟動
python -m services.core.main --check-imports
```

---

## 📋 **修復檢查清單**

### **修復前檢查**
- [ ] 備份相關檔案
- [ ] 識別問題類型和範圍
- [ ] 檢查依賴是否真的需要安裝

### **修復過程**
- [ ] 應用適當的修復策略
- [ ] 更新相關的導入語句
- [ ] 實現必要的 Mock 類別

### **修復後驗證**
- [ ] 執行單元測試
- [ ] 測試有無依賴兩種情況
- [ ] 確認系統功能正常
- [ ] 更新相關文件

---

## 🔗 **相關資源**

- [Optional Dependency 框架](../../utilities/optional_deps.py)
- [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)
- [Services 架構文件](../architecture/SERVICES_ARCHITECTURE.md)
- [測試指南](../development/TESTING_GUIDE.md)

---

## 📞 **支援聯絡**

遇到複雜的導入問題？
- 檢查 [GitHub Issues](https://github.com/your-repo/issues)
- 查看 [疑難排解 FAQ](./FAQ.md)
- 參考現有的修復範例在專案中