# 後續修復指引 - Integration 測試工具遷移

**日期**: 2025年11月17日  
**狀態**: 規劃階段  
**優先級**: P0（高優先級）

---

## 📋 概述

本指引詳細說明如何將 Integration 模組中的測試工具遷移到 Features 模組，以符合 AIVA 五大模組架構原則。

---

## 🎯 遷移目標

### 需要遷移的檔案（共 5 個）

| 檔案 | 當前位置 | 目標位置 | 代碼行數 | 優先級 |
|------|---------|---------|----------|--------|
| xss_attack_tools.py | integration/capability/ | features/function_xss/integration_tools.py | 1096 | P0 |
| sql_injection_tools.py | integration/capability/ | features/function_sqli/integration_tools.py | 734 | P0 |
| sql_injection_bounty_hunter.py | integration/capability/ | features/function_sqli/bounty_hunter.py | 777 | P0 |
| web_attack.py | integration/capability/ | features/function_web_scanner/scanner.py | 882 | P0 |
| ddos_attack_tools.py | integration/capability/ | 評估是否需要 | 784 | P2 |

---

## 📝 詳細遷移步驟

### **階段 1: XSS 工具遷移** ⚠️

#### 1.1 創建目標目錄
```powershell
New-Item -Path "C:\D\fold7\AIVA-git\services\features\function_xss\integration_tools" -ItemType Directory -Force
```

#### 1.2 移動檔案
```powershell
Move-Item `
    -Path "C:\D\fold7\AIVA-git\services\integration\capability\xss_attack_tools.py" `
    -Destination "C:\D\fold7\AIVA-git\services\features\function_xss\integration_tools\xss_tools.py"
```

#### 1.3 更新 Import 路徑

**原始路徑**:
```python
from ...core.base_capability import BaseCapability
from ...aiva_common.schemas import APIResponse
```

**新路徑**:
```python
from services.aiva_common.schemas import APIResponse
# 如果需要 BaseCapability，從 features/base/ 導入
from services.features.base.feature_base import FeatureBase
```

#### 1.4 更新依賴檔案

搜尋並更新所有導入 xss_attack_tools 的檔案：
```powershell
# 搜尋導入語句
grep -r "from.*xss_attack_tools import" services/
grep -r "import.*xss_attack_tools" services/
```

可能的依賴檔案：
- `services/integration/capability/examples.py`
- `services/integration/capability/cli.py`
- `services/integration/capability/registry.py`

#### 1.5 創建 Integration 調用接口

在 `services/integration/capability/` 創建新的調用接口：

```python
# services/integration/capability/xss_integration.py
"""XSS Integration Interface - 協調 XSS 測試"""

from services.features.function_xss.integration_tools import xss_tools

class XSSIntegrationCoordinator:
    """XSS 測試協調器 - Integration 模組接口"""
    
    async def coordinate_xss_scan(self, target_url: str, options: dict):
        """協調 XSS 掃描 - 不執行實際測試"""
        # 1. 驗證參數
        # 2. 調用 Features 模組
        results = await xss_tools.ReflectedXSSScanner().scan_target(...)
        # 3. 收集和標準化結果
        # 4. 返回給調用者
        return results
```

#### 1.6 驗證功能

```python
# 測試遷移後的功能
import asyncio
from services.features.function_xss.integration_tools import xss_tools

async def test_xss_migration():
    scanner = xss_tools.ReflectedXSSScanner()
    # 測試基本功能...
    
asyncio.run(test_xss_migration())
```

---

### **階段 2: SQL 注入工具遷移** ⚠️

#### 2.1 創建目標目錄
```powershell
New-Item -Path "C:\D\fold7\AIVA-git\services\features\function_sqli\integration_tools" -ItemType Directory -Force
```

#### 2.2 移動檔案（2個）
```powershell
# 移動主要工具
Move-Item `
    -Path "C:\D\fold7\AIVA-git\services\integration\capability\sql_injection_tools.py" `
    -Destination "C:\D\fold7\AIVA-git\services\features\function_sqli\integration_tools\sql_tools.py"

# 移動 Bounty Hunter
Move-Item `
    -Path "C:\D\fold7\AIVA-git\services\integration\capability\sql_injection_bounty_hunter.py" `
    -Destination "C:\D\fold7\AIVA-git\services\features\function_sqli\integration_tools\bounty_hunter.py"
```

#### 2.3 更新 Import 路徑

**原始路徑**:
```python
from ...core.base_capability import BaseCapability
```

**新路徑**:
```python
from services.features.base.feature_base import FeatureBase
```

#### 2.4 創建 Integration 調用接口

```python
# services/integration/capability/sql_integration.py
"""SQL Injection Integration Interface"""

class SQLIntegrationCoordinator:
    """SQL 注入測試協調器"""
    
    async def coordinate_sql_scan(self, target_url: str, options: dict):
        """協調 SQL 注入掃描"""
        # 協調邏輯...
```

---

### **階段 3: Web 掃描器遷移** ⚠️

#### 3.1 創建新的 Features 子模組
```powershell
New-Item -Path "C:\D\fold7\AIVA-git\services\features\function_web_scanner" -ItemType Directory -Force
```

#### 3.2 移動檔案
```powershell
Move-Item `
    -Path "C:\D\fold7\AIVA-git\services\integration\capability\web_attack.py" `
    -Destination "C:\D\fold7\AIVA-git\services\features\function_web_scanner\scanner.py"
```

#### 3.3 創建模組結構
```
services/features/function_web_scanner/
├── scanner.py          # 主掃描器（從 web_attack.py 移過來）
├── __init__.py
├── __main__.py         # 執行入口
└── README.md           # 模組文檔
```

---

### **階段 4: DDoS 工具評估** 📋

#### 4.1 評估是否需要

**考慮因素**:
- DDoS 通常不適合 Bug Bounty 場景
- 可能涉及法律風險
- AIVA 定位為 Bug Bounty 工具

**建議選項**:
1. **刪除** - 如果確定不需要（推薦）
2. **移動到 Features** - 如果確定需要保留
3. **標記為實驗性** - 僅用於授權測試

#### 4.2 如果決定刪除
```powershell
Remove-Item "C:\D\fold7\AIVA-git\services\integration\capability\ddos_attack_tools.py"
```

---

## 🔧 通用步驟模板

對於每個檔案遷移，遵循以下標準流程：

### **步驟 1: 準備**
```powershell
# 1. 確認目標目錄存在
Test-Path "目標目錄路徑"

# 2. 創建目錄（如果不存在）
New-Item -Path "目標目錄路徑" -ItemType Directory -Force

# 3. 備份原檔案
Copy-Item "源檔案" "源檔案.backup"
```

### **步驟 2: 移動**
```powershell
# 移動檔案
Move-Item -Path "源檔案" -Destination "目標檔案"
```

### **步驟 3: 更新 Import**
```python
# 搜尋所有 import 語句
grep -r "from.*檔案名 import" services/

# 批量替換（使用編輯器或腳本）
# 原: from ...integration.capability.xxx import
# 新: from services.features.function_xxx.xxx import
```

### **步驟 4: 驗證**
```python
# 1. 語法檢查
python -m py_compile 目標檔案

# 2. Import 檢查
python -c "from services.features.xxx import 模組"

# 3. 功能測試
pytest tests/test_xxx.py
```

### **步驟 5: 清理**
```powershell
# 刪除備份（確認無誤後）
Remove-Item "源檔案.backup"

# 更新 Git
git add .
git commit -m "refactor: 遷移 xxx 到 Features 模組"
```

---

## ⚠️ 注意事項

### **Import 路徑變更**

| 舊路徑 | 新路徑 |
|--------|--------|
| `from ...core.base_capability` | `from services.features.base` |
| `from ...aiva_common` | `from services.aiva_common` |
| `from ..registry` | 需要調整為新的註冊機制 |

### **BaseCapability 替換**

如果使用了 `BaseCapability`，需要：
1. 改為繼承 `FeatureBase`
2. 或創建 Features 專用的 Base 類

### **Registry 整合**

Features 模組可能需要自己的註冊機制：
```python
# services/features/registry.py
class FeatureRegistry:
    """Features 模組註冊器"""
    # ...
```

---

## 📊 進度追蹤

| 任務 | 狀態 | 負責人 | 完成日期 |
|------|------|--------|----------|
| XSS 工具遷移 | ⚠️ 待開始 | - | - |
| SQL 工具遷移 | ⚠️ 待開始 | - | - |
| Web 掃描器遷移 | ⚠️ 待開始 | - | - |
| DDoS 工具評估 | ⚠️ 待開始 | - | - |
| Integration 接口重構 | ⚠️ 待開始 | - | - |
| 測試驗證 | ⚠️ 待開始 | - | - |
| 文檔更新 | ⚠️ 待開始 | - | - |

---

## 🧪 測試計劃

### **單元測試**
```python
# tests/features/test_xss_tools.py
def test_xss_scanner_import():
    from services.features.function_xss.integration_tools import xss_tools
    assert xss_tools.ReflectedXSSScanner is not None
```

### **整合測試**
```python
# tests/integration/test_xss_coordination.py
async def test_xss_coordination():
    coordinator = XSSIntegrationCoordinator()
    results = await coordinator.coordinate_xss_scan(...)
    assert results is not None
```

### **功能測試**
```bash
# 端到端測試
python -m services.features.function_xss --test
```

---

## 📚 相關資源

- [架構違反分析報告](./ARCHITECTURE_VIOLATION_ANALYSIS.md)
- [Features 模組 README](./services/features/README.md)
- [Integration 模組 README](./services/integration/README.md)
- [AIVA 架構原則](./docs/ARCHITECTURE_PRINCIPLES.md)

---

## 🔄 回滾計劃

如果遷移出現問題：

```powershell
# 1. 恢復備份
Copy-Item "源檔案.backup" "源檔案"

# 2. Git 回滾
git reset --hard HEAD^

# 3. 重新評估問題
# 4. 調整計劃再試
```

---

**最後更新**: 2025年11月17日  
**維護者**: AIVA Architecture Team
