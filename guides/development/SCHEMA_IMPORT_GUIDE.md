---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Guide
---

# AIVA Schema 導入規範指南

> **📋 適用對象**: 所有AIVA開發者、跨語言模組貢獻者  
> **🎯 使用場景**: Schema導入、跨語言開發、標準化實施  
> **⏱️ 預計閱讀時間**: 15 分鐘  
> **🔧 技術需求**: Python/Go/Rust/TypeScript 開發環境

---

## 📑 目錄

1. [🎯 重要更新 (v3.1)](#-重要更新-v31)
2. [🔥 必須遵循的規範](#-必須遵循的規範)
3. [📦 各語言導入標準](#-各語言導入標準)
4. [⚠️ 禁止事項](#️-禁止事項)
5. [🔧 遷移指南](#-遷移指南)
6. [🧪 驗證方法](#-驗證方法)
7. [🔍 疑難排解](#-疑難排解)
8. [📚 最佳實踐](#-最佳實踐)

---

## 🎯 重要更新 (v3.1)

**⚠️ Schema 標準化完成**: AIVA 已實現 100% 跨語言 Schema 標準化！

### 🔥 必須遵循的新規範

#### Go 模組
```go
// ✅ 必須使用 - 標準 schema 導入
import schemas "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"

// ❌ 嚴格禁止 - 自定義 FindingPayload
type FindingPayload struct {...}  // 會被 CI 拒絕
```

#### Rust 模組  
```rust
// ✅ 必須使用 - 生成的 schema
use crate::schemas::generated::FindingPayload;

// ❌ 嚴格禁止 - 自定義結構
struct FindingPayload {...}  // 會被 CI 拒絕
```

#### TypeScript 模組
```typescript
// ✅ 必須使用 - 標準定義
import { FindingPayload } from '../schemas/aiva_schemas';

// ❌ 嚴格禁止 - 自定義接口
interface FindingPayload {...}  // 會被 CI 拒絕
```

**驗證命令**: `python tools/schema_compliance_validator.py --workspace . --language all`

## 📋 導入規範總覽

### ✅ 推薦做法

#### 1. **模組間相對導入** (首選)
```python
# 在 services/core/ 中導入 aiva_common
from ..aiva_common.enums import Severity, Confidence, TaskStatus
from ..aiva_common.schemas import FindingPayload, ScanStartPayload

# 在 services/features/ 中導入 aiva_common  
from ..aiva_common.enums import AssetType, VulnerabilityStatus
from ..aiva_common.schemas.generated.tasks import FunctionTaskPayload
```

#### 2. **包級導入** (需先安裝)
```bash
# 先安裝為開發包
pip install -e .

# 然後可以使用包級導入
from aiva_common.enums import Severity, Confidence
from aiva_common.schemas import FindingPayload
```

### ❌ 避免做法

#### 1. **絕對路徑導入** (不可移植)
```python
# ❌ 路徑依賴，不可移植
from services.aiva_common.enums import Severity
from C:\D\fold7\AIVA-git\services.aiva_common.enums import Severity
```

#### 2. **硬編碼路徑**
```python
# ❌ 系統依賴，無法跨平台
import sys
sys.path.append('C:\\D\\fold7\\AIVA-git\\services')
from aiva_common.enums import Severity
```

## 🎯 各模組具體規範

### Core 模組 (`services/core/`)
```python
# ✅ 正確方式
from ..aiva_common.enums import (
    AttackPathEdgeType,
    Confidence,
    RiskLevel,
    Severity,
    TaskStatus,
)
from ..aiva_common.schemas import CVSSv3Metrics, CVEReference
```

### Features 模組 (`services/features/`)
```python
# ✅ 正確方式
from ..aiva_common.enums import (
    AssetType,
    Confidence,
    Severity,
    VulnerabilityStatus,
)
from ..aiva_common.schemas.generated.tasks import FunctionTaskPayload
from ..aiva_common.schemas.generated.findings import FindingPayload
```

### Integration 模組 (`services/integration/`)
```python
# ✅ 正確方式
from ..aiva_common.enums.assets import AssetType, AssetStatus
from ..aiva_common.enums.common import Confidence, Severity
from ..aiva_common.enums.security import VulnerabilityStatus
```

### Scan 模組 (`services/scan/`)
```python
# ✅ 正確方式
from ..aiva_common.schemas import ScanStartPayload, CVSSv3Metrics
from ..aiva_common.enums import Severity, Confidence
```

## 🔧 環境設置建議

### 開發環境
```bash
# 1. 克隆項目
git clone https://github.com/your-org/AIVA.git
cd AIVA

# 2. 安裝為開發包 (可選)
pip install -e .

# 3. 設置 PYTHONPATH (替代方案)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 生產環境
```bash
# 1. 使用 Docker (推薦)
docker build -t aiva .
docker run aiva

# 2. 或安裝為正式包
pip install .
```

## 🌍 跨平台兼容性

### Windows
```powershell
# PowerShell
$env:PYTHONPATH += ";$(Get-Location)"
```

### Linux/macOS
```bash
# Bash/Zsh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🧪 導入測試

### 驗證導入是否正確
```python
# 測試腳本：test_imports.py
def test_relative_imports():
    """測試相對路徑導入"""
    try:
        from services.core import models
        from services.features import __init__
        print("✅ 相對路徑導入成功")
    except ImportError as e:
        print(f"❌ 相對路徑導入失敗: {e}")

def test_package_imports():
    """測試包級導入"""
    try:
        from aiva_common.enums import Severity
        from aiva_common.schemas import FindingPayload
        print("✅ 包級導入成功")
    except ImportError as e:
        print(f"❌ 包級導入失敗: {e}")

if __name__ == "__main__":
    test_relative_imports()
    test_package_imports()
```

## 📝 最佳實踐總結

1. **首選相對導入**: 使用 `..aiva_common` 確保可移植性
2. **避免絕對路徑**: 不使用 `services.aiva_common` 避免環境依賴
3. **統一導入風格**: 所有模組使用相同的導入模式
4. **測試多環境**: 在不同系統、路徑下驗證導入正確性
5. **文檔說明**: README中明確說明導入規範

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025-10-26  
**版本**: 1.0