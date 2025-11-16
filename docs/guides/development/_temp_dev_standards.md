# 開發規範模板

## 🛠️ aiva_common 修復規範

> **核心原則**: 本模組作為 AIVA 系統的組成部分，必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範與最佳實踐。

### 📌 必讀規範

**完整規範文檔**: [aiva_common/README.md - 開發指南](../../../aiva_common/README.md#-開發指南)

### 🎯 核心設計原則 (摘要)

#### 1️⃣ 四層優先級原則

```
1. 國際標準/官方規範 (最高優先級)
   ✅ CVSS, CVE, CWE, CAPEC, SARIF, MITRE ATT&CK
   
2. 程式語言標準庫 (次高優先級)
   ✅ Python: enum.Enum, typing 模組
   
3. aiva_common 統一定義 (系統內部標準)
   ✅ Severity, Confidence, TaskStatus, VulnerabilityType
   
4. 模組專屬枚舉 (最低優先級)
   ⚠️ 僅當功能完全限於該模組內部時才允許
```

#### 2️⃣ 禁止重複定義

```python
# ❌ 嚴格禁止 - 重複定義已存在的枚舉
class Severity(str, Enum):  # 錯誤！aiva_common 已定義
    HIGH = "high"

# ✅ 正確做法 - 直接使用 aiva_common
from aiva_common import Severity, FindingPayload
```

#### 3️⃣ 模組專屬枚舉判斷標準

**只有滿足所有條件時才能自定義：**

- ✅ 該枚舉僅用於模組內部，不會跨模組傳遞
- ✅ 該枚舉與業務邏輯強綁定，無法抽象為通用概念
- ✅ 該枚舉在 aiva_common 中不存在類似定義
- ✅ 該枚舉未來不太可能被其他模組使用

### 🚫 常見錯誤與正確做法

#### 錯誤 1: 重複定義枚舉

```python
# ❌ 錯誤：在模組內重複定義
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"

# ✅ 正確：使用 aiva_common
from aiva_common import TaskStatus
```

#### 錯誤 2: 自創非標準概念

```python
# ❌ 錯誤：自創嚴重程度
class MySeverity(str, Enum):
    SUPER_HIGH = "super_high"

# ✅ 正確：使用標準 Severity
from aiva_common import Severity
# 使用 Severity.CRITICAL 表示最高嚴重程度
```

#### 錯誤 3: 破壞性修改 Schema

```python
# ❌ 錯誤：移除預設值導致向後不兼容
class MySchema(BaseModel):
    new_field: str  # 破壞舊代碼！

# ✅ 正確：使用可選或預設值
class MySchema(BaseModel):
    new_field: Optional[str] = None  # 向後兼容
```

### 📋 開發檢查清單

**新增功能前必須確認：**

- [ ] 檢查 aiva_common 是否已有相關枚舉
- [ ] 檢查是否有相關的國際標準（CVSS、SARIF 等）
- [ ] 確認新定義不會與 aiva_common 重複
- [ ] 評估是否需要加入 aiva_common 而非模組內部
- [ ] 檢查是否符合四層優先級原則

**使用 aiva_common 時必須：**

- [ ] 導入標準枚舉：`from aiva_common import Severity, Confidence, TaskStatus`
- [ ] 導入標準 Schema：`from aiva_common import FindingPayload, CVSSv3Metrics`
- [ ] 使用完整的型別標註
- [ ] 添加完整的 docstring

### 🔗 相關文檔連結

- 📖 [aiva_common 完整開發指南](../../../aiva_common/README.md#-開發指南)
- 📖 [修復規範詳細說明](../../../aiva_common/README.md#-開發規範與最佳實踐)
- 📖 [架構修復完成報告](../../../aiva_common/README.md#-架構修復完成報告)
- 📖 [跨語言 Schema 架構](../../../aiva_common/README.md#️-跨語言-schema-架構)

### 📊 修復規範版本

- **規範版本**: v1.0
- **最後更新**: 2025年11月16日
- **維護者**: AIVA 開發團隊
- **狀態**: ✅ 全面實施中

---

> ⚠️ **重要提醒**: 違反 aiva_common 修復規範可能導致：
> - 數據類型不一致問題
> - 跨模組通信失敗
> - 代碼審查不通過
> - 系統穩定性風險
>
> 請務必在開發前詳細閱讀完整的修復規範文檔！
