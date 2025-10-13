# 四大模組全面修正計畫

**執行時間:** 2025-10-13  
**目標:** 修正 ThreatIntel, AuthZ, PostEx, Remediation 四大模組的所有問題

## 🎯 修正項目

### 1. ✅ 移除重複的 Enum 定義

#### intel_aggregator.py

- ✅ 移除 `IntelSource` enum (使用 `aiva_common.enums.IntelSource`)
- ✅ 移除 `ThreatLevel` enum (使用 `aiva_common.enums.ThreatLevel`)

#### permission_matrix.py

- 🔄 移除 `Permission` enum (使用 `aiva_common.enums.Permission`)
- 🔄 移除 `AccessDecision` enum (使用 `aiva_common.enums.AccessDecision`)

### 2. 🔄 修復 Typing 問題

所有檔案需要修正:

- `Dict` → `dict`
- `List` → `list`  
- `Optional[X]` → `X | None`
- `Set` → `set`

影響檔案:

- `services/threat_intel/intel_aggregator.py` (28 處)
- `services/threat_intel/ioc_enricher.py`
- `services/threat_intel/mitre_mapper.py`
- 其他所有模組

### 3. 🔄 修復 Import 排序

所有檔案需要按照順序:

1. 標準庫 (如 `import os`)
2. 第三方庫 (如 `import pandas`)
3. 本地導入 (如 `from services.aiva_common`)

### 4. 🔄 添加 aiva_common 整合

為每個模組創建 worker 類:

- `ThreatIntelWorker` - 整合 `ThreatIntelLookupPayload`
- `AuthZWorker` - 整合 `AuthZCheckPayload`
- `PostExWorker` - 整合 `PostExTestPayload`
- `RemediationWorker` - 整合 `RemediationGeneratePayload`

---

## 📋 執行順序

### Phase 1: Enum 清理 ✅

1. ✅ `intel_aggregator.py` - 完成
2. 🔄 `permission_matrix.py` - 進行中

### Phase 2: Typing 修復

使用批量替換修正所有檔案的 typing 問題

### Phase 3: Import 排序

使用 `isort` 或手動修正

### Phase 4: 格式化

運行 `black` 統一格式

### Phase 5: Lint 檢查

運行 `ruff check --fix` 自動修復

---

## 🔧 自動修復命令

```bash
# Phase 2-5: 自動修復
cd c:\D\E\AIVA\AIVA-main

# 排序導入
isort services/threat_intel services/authz services/postex services/remediation

# 格式化代碼
black services/threat_intel services/authz services/postex services/remediation

# 修復 lint 錯誤
ruff check --fix services/threat_intel services/authz services/postex services/remediation
```

---

## ✅ 已完成

1. ✅ 添加所有必要的 Enum 到 `aiva_common/enums.py`:
   - `Permission`, `AccessDecision`
   - `PostExTestType`, `PersistenceType`

2. ✅ 添加所有必要的 Schema 到 `aiva_common/schemas.py`:
   - `PostExTestPayload`, `PostExResultPayload`

3. ✅ 添加所有必要的 Topic 到 `aiva_common/enums.py`:
   - PostEx 相關 topics

4. ✅ 修正 `intel_aggregator.py` 的重複 Enum

---

## 📊 進度追蹤

| 任務 | 狀態 | 檔案數 |
|------|------|--------|
| Enum 清理 | 50% | 2/4 |
| Typing 修復 | 0% | 0/14 |
| Import 排序 | 0% | 0/14 |
| 格式化 | 0% | 0/14 |
| Worker 整合 | 0% | 0/4 |

**總進度:** 10% (1.5/15 項)
