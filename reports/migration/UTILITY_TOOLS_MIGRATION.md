# 🛠️ 實用工具遷移文檔

## 目錄

- [概述](#概述)
- [遷移日期與目的](#遷移日期與目的)
- [已遷移工具清單](#已遷移工具清單)
- [工具使用指南](#工具使用指南)
- [目錄結構](#目錄結構)
- [相關文檔](#相關文檔)

---

## 概述

本文檔記錄從 `testing/` 目錄遷移到 `services/` 各模組的實用工具。這些工具原本是測試腳本，經過轉換後成為可直接使用的系統檢查和診斷工具。

## 遷移日期與目的

**遷移日期**: 2025-01-22

**遷移目的**:
- ✅ 將實際可用的工具直接放在對應服務模組，方便查找和使用
- ✅ 移除測試相關術語，轉換為實用工具
- ✅ 提高工具的可訪問性和實用性
- ✅ 按照五大模組架構組織工具

---

## 已遷移工具清單

### 1. Core 模組工具 (`services/core/tools/`)

#### `system_connectivity_checker.py`
- **原始位置**: `testing/core/ai_system_connectivity_check.py`
- **功能**: AI 系統連接檢查工具
- **用途**: 驗證 AI 決策能夠實際執行系統命令和操作的完整流程

**主要功能**:
- ✅ 檢查 AI 核心組件載入
- ✅ 檢查 AI 工具系統連接
- ✅ 檢查 AI 決策 → 工具調用
- ✅ 檢查工具 → 系統命令執行
- ✅ 檢查文件系統訪問
- ✅ 檢查網路連接
- ✅ 檢查 AI 訓練系統與存儲連接
- ✅ 檢查命令執行鏈

**使用方法**:
```bash
cd services/core/tools
python system_connectivity_checker.py
```

**輸出**:
- 終端顯示檢查進度和結果
- 生成整體通連性評分
- 提供改善建議

---

### 2. Common 模組工具 (`services/aiva_common/tools/`)

#### `module_connectivity_checker.py`
- **原始位置**: `testing/common/module_connectivity_tester.py`
- **功能**: AIVA 五大模組通連性檢查工具
- **用途**: 基於 Schema 自動化系統，檢查五大模組間的通連性

**主要功能**:
- 🔍 模組間通信檢查
- 📡 Schema 一致性驗證
- 🔄 跨語言數據傳遞檢查
- 📊 通連性健康度報告
- 🚀 端到端工作流檢查

**檢查項目**:
1. Schema 導入連通性
   - 基礎類型 Schema
   - 消息 Schema
   - 任務 Schema
   - 發現 Schema

2. 跨模組消息傳遞
   - 消息創建
   - Schema 驗證
   - 序列化/反序列化

3. 模組整合點
   - AI 引擎整合
   - 攻擊引擎整合
   - 掃描引擎整合
   - 功能檢測整合
   - 整合服務

4. Go Schema 連接性
   - Go Schema 文件存在性
   - 結構體數量
   - JSON 標籤有效性

**使用方法**:
```bash
cd services/aiva_common/tools
python module_connectivity_checker.py
```

**輸出**:
- 通連性得分 (0-100)
- 詳細檢查結果
- 改進建議
- JSON 格式詳細報告 (`aiva_connectivity_report_*.json`)
- 日誌文件 (`aiva_connectivity_check.log`)

**評分標準**:
- 🟢 90+ 分: 優秀
- 🟡 70-89 分: 良好
- 🟠 50-69 分: 尚可
- 🔴 <50 分: 需要改善

---

### 3. Integration 模組工具 (`services/integration/tools/`)

#### `sop_compliance_checker.py`
- **原始位置**: `testing/integration/aiva_system_connectivity_sop_check.py`
- **功能**: AIVA 系統 SOP 合規性檢查工具
- **用途**: 遵循單一真實來源原則和分層責任架構進行全面檢查

**主要功能**:
1. Schema 定義體系檢查
   - 權威定義來源檢查
   - Enum 定義檢查
   - 導入導出完整性檢查

2. AI 核心模組檢查
   - AI 引擎核心檢查
   - 工具系統檢查
   - 學習系統檢查

3. 系統工具連接檢查
   - 文件操作工具
   - 命令執行工具
   - 掃描觸發工具

4. 命令執行鏈檢查
   - 命令構建
   - 系統執行

**使用方法**:
```bash
cd services/integration/tools
python sop_compliance_checker.py
```

**輸出**:
- 各類別檢查結果
- 整體合規率
- JSON 格式報告 (`sop_compliance_report_*.json`)

**合規率評估**:
- 🎉 90%+: 高度符合 SOP 標準
- ⚠️  70-89%: 基本符合，建議改善
- ❌ <70%: 需要加強合規性

---

### 4. Features 模組工具 (`services/features/common/testers/`)

#### `vertical_escalation_tester.py`
- **原始位置**: `testing/features/testers/vertical_escalation_tester.py`
- **功能**: 垂直權限提升測試器
- **用途**: 實現 OWASP WSTG-ATHZ-03 標準的垂直權限提升檢測

**主要功能**:
- 檢測低權限用戶訪問高權限功能
- 根據 URL 推斷所需權限級別
- 執行垂直權限提升測試
- 提供詳細的測試結果和證據

**權限級別**:
- ADMIN (管理員)
- USER (用戶)
- GUEST (訪客)
- ANONYMOUS (匿名)

**使用方法**:
```python
import httpx
from services.features.common.testers.vertical_escalation_tester import (
    VerticalEscalationTester,
    PrivilegeLevel
)

async with httpx.AsyncClient() as client:
    tester = VerticalEscalationTester(client)
    result = await tester.test_vertical_idor(
        url="https://example.com/admin/dashboard",
        user_auth={"Authorization": "Bearer user_token"}
    )
    
    if result.vulnerable:
        print(f"發現漏洞! 證據: {result.evidence}")
```

#### `cross_user_tester.py`
- **原始位置**: `testing/features/testers/cross_user_tester.py`
- **功能**: 跨用戶測試器 (水平權限提升)
- **用途**: 實現 OWASP WSTG-ATHZ-04 標準的 IDOR 檢測

**主要功能**:
- 檢測攻擊者通過修改參數訪問其他用戶資源
- 執行水平 IDOR 測試
- 計算響應相似度分數
- 提供詳細的測試結果和證據

**使用方法**:
```python
import httpx
from services.features.common.testers.cross_user_tester import CrossUserTester

async with httpx.AsyncClient() as client:
    tester = CrossUserTester(client)
    result = await tester.test_horizontal_idor(
        url="https://example.com/api/user/profile",
        resource_id="user_123",
        user_a_auth={"Authorization": "Bearer token_a"},
        user_b_auth={"Authorization": "Bearer token_b"}
    )
    
    if result.vulnerable:
        print(f"發現 IDOR 漏洞! 相似度: {result.similarity_score}")
```

---

## 工具使用指南

### 通用執行步驟

1. **切換到工具目錄**:
   ```bash
   cd services/{module}/tools/
   ```

2. **執行工具**:
   ```bash
   python {tool_name}.py
   ```

3. **查看結果**:
   - 終端實時輸出
   - 生成的報告文件
   - 日誌文件

### 定期檢查建議

**每日**:
- `system_connectivity_checker.py` - 確保 AI 系統正常運作

**每週**:
- `module_connectivity_checker.py` - 檢查模組間通連性
- `sop_compliance_checker.py` - 確保 SOP 合規

**按需**:
- `vertical_escalation_tester.py` - 安全測試時使用
- `cross_user_tester.py` - 安全測試時使用

---

## 目錄結構

```
services/
├── core/
│   └── tools/
│       └── system_connectivity_checker.py    # AI 系統連接檢查
│
├── aiva_common/
│   └── tools/
│       └── module_connectivity_checker.py    # 模組通連性檢查
│
├── integration/
│   └── tools/
│       └── sop_compliance_checker.py         # SOP 合規性檢查
│
└── features/
    └── common/
        └── testers/
            ├── vertical_escalation_tester.py  # 垂直權限提升測試器
            └── cross_user_tester.py          # 跨用戶測試器
```

---

## 相關文檔

- [TESTING_SCRIPTS_REORGANIZATION.md](../TESTING_SCRIPTS_REORGANIZATION.md) - Testing 和 Scripts 重組計劃
- [testing/README.md](../testing/README.md) - Testing 目錄說明
- [testing/_archive/README.md](../testing/_archive/README.md) - 歸檔測試說明
- [scripts/_archive/README.md](../scripts/_archive/README.md) - 歸檔腳本說明

---

## 總結

透過此次遷移，我們將 **5 個**實用工具從 `testing/` 目錄移至對應的 `services/` 模組：

| 模組 | 工具數量 | 主要用途 |
|------|---------|---------|
| Core | 1 | AI 系統連接檢查 |
| Common | 1 | 模組通連性檢查 |
| Integration | 1 | SOP 合規性檢查 |
| Features | 2 | 安全測試器 (垂直/水平權限提升) |

這些工具現在可以直接在對應模組中使用，無需在 `testing/` 目錄中尋找，大幅提高了工具的可用性和可維護性。
