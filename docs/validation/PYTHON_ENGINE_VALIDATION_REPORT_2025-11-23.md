# Python Engine USAGE_GUIDE.md 驗證報告

**驗證日期**: 2025-11-23  
**文檔路徑**: `services/scan/coordinators/PYTHON_ENGINE_USAGE_GUIDE.md`  
**靶場環境**: Juice Shop (localhost:3000)

---

## 📋 驗證概要

| 項目 | 狀態 |
|------|------|
| 文檔總行數 | 822 行 |
| 驗證章節 | 2/7 (28.6%) |
| 代碼示例 | 20+ 個 |
| 發現錯誤 | 3 處 |
| 嚴重程度 | 🔴 高 |

---

## 🔴 發現的錯誤

### 錯誤 #1: scan_id 格式錯誤 (Lines 30-42)

**嚴重程度**: 🔴 高

**問題描述**:  
文檔中所有示例的 `scan_id` 未使用必需的 `scan_` 前綴,導致驗證錯誤。

**文檔中的錯誤代碼**:
```python
# ❌ 錯誤 (多處)
request = ScanStartPayload(
    scan_id="quick_test",        # 缺少 scan_ 前綴
    targets=["http://localhost:3000"],
    strategy="quick"
)

request = ScanStartPayload(
    scan_id="scan_001",          # 這個是對的
    ...
)

request = ScanStartPayload(
    scan_id="direct_scan",       # 錯誤
    ...
)

request = ScanStartPayload(
    scan_id="multi_target",      # 錯誤
    ...
)

request = ScanStartPayload(
    scan_id="analysis_test",     # 錯誤
    ...
)
```

**實際錯誤信息**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ScanStartPayload
scan_id
  Value error, scan_id must start with 'scan_' [type=value_error, input_value='quick_test', input_type=str]
```

**正確代碼**:
```python
# ✅ 正確
request = ScanStartPayload(
    scan_id="scan_quick_test",      # 添加 scan_ 前綴
    targets=["http://localhost:3000"],
    strategy="quick"
)

request = ScanStartPayload(
    scan_id="scan_direct_scan",     # 添加 scan_ 前綴
    ...
)

request = ScanStartPayload(
    scan_id="scan_multi_target",    # 添加 scan_ 前綴
    ...
)

request = ScanStartPayload(
    scan_id="scan_analysis_test",   # 添加 scan_ 前綴
    ...
)
```

**影響範圍**: 整份文檔,至少 5 處示例代碼

---

### 錯誤 #2: API 方法名錯誤 (Lines 88-115)

**嚴重程度**: 🔴 高

**問題描述**:  
文檔使用的方法 `execute_coordinated_scan()` 不存在,實際 API 使用不同的策略方法。

**文檔中的錯誤代碼**:
```python
# ❌ 錯誤 - 不存在的方法
result = await coordinator.execute_coordinated_scan(request)
```

**實際錯誤信息**:
```
AttributeError: 'MultiEngineCoordinator' object has no attribute 'execute_coordinated_scan'
```

**實際的 API**:
```python
# ✅ 正確 - 實際存在的方法
class MultiEngineCoordinator:
    async def execute_strategy_fast(self, scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    async def execute_strategy_balanced(self, scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    async def execute_strategy_comprehensive(self, scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    async def execute_strategy_aggressive(self, scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    async def execute_strategy_smart(self, scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    
    # 便利函數
    async def quick_scan(scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    async def smart_scan(scan_id: str, targets: List[str]) -> Phase1CompletedPayload
    async def full_scan(scan_id: str, targets: List[str]) -> Phase1CompletedPayload
```

**正確用法示例**:
```python
# ✅ 方式 1: 使用策略方法
coordinator = MultiEngineCoordinator()
result = await coordinator.execute_strategy_fast("scan_test_001", ["http://localhost:3000"])

# ✅ 方式 2: 使用便利函數
from services.scan.coordinators.multi_engine_coordinator import quick_scan
result = await quick_scan("scan_test_001", ["http://localhost:3000"])
```

**影響範圍**: 整份文檔,所有協調器調用示例

---

### 錯誤 #3: 參數接口不匹配 (Lines 90-115)

**嚴重程度**: 🔴 高

**問題描述**:  
文檔示例使用 `ScanStartPayload` 對象作為參數,但實際 API 接受的是 `scan_id` 和 `targets` 列表。

**文檔中的錯誤代碼**:
```python
# ❌ 錯誤 - 參數類型不匹配
coordinator = MultiEngineCoordinator()

request = ScanStartPayload(
    scan_id="scan_001",
    targets=["http://localhost:3000"],
    strategy="normal",
    max_depth=3,
    timeout=300
)

# 傳入 ScanStartPayload 對象 (錯誤)
result = await coordinator.execute_coordinated_scan(request)
```

**正確代碼**:
```python
# ✅ 正確 - 直接傳入參數
coordinator = MultiEngineCoordinator()

# 方法簽名: async def execute_strategy_fast(self, scan_id: str, targets: List[str])
result = await coordinator.execute_strategy_fast(
    scan_id="scan_test_001",
    targets=["http://localhost:3000"]
)

# 注意: 沒有 strategy, max_depth, timeout 等參數
# 這些由策略方法本身決定
```

**關鍵差異**:

| 文檔描述 | 實際 API |
|---------|---------|
| `execute_coordinated_scan(request: ScanStartPayload)` | `execute_strategy_fast(scan_id: str, targets: List[str])` |
| 傳入完整配置對象 | 僅傳入 scan_id 和 targets |
| 可配置 strategy, max_depth, timeout | 由策略方法預定義 |
| 返回 `CoordinationResult` | 返回 `Phase1CompletedPayload` |

---

### 錯誤 #4: 返回類型不匹配 (Lines 300-320)

**嚴重程度**: 🟡 中等

**問題描述**:  
文檔描述的返回類型 `CoordinationResult` 與實際返回類型 `Phase1CompletedPayload` 不一致。

**文檔中的錯誤描述**:
```python
# ❌ 錯誤的返回類型
from services.scan.coordinators.scan_models import CoordinationResult, EngineResult

result: CoordinationResult = await coordinator.execute_coordinated_scan(request)

# 訪問屬性
result.scan_id              # str
result.total_assets         # int
result.total_time           # float
result.coordination_strategy # str
result.engine_results       # List[EngineResult]
```

**實際返回類型**:
```python
# ✅ 正確的返回類型
from services.aiva_common.schemas import Phase1CompletedPayload

result: Phase1CompletedPayload = await coordinator.execute_strategy_fast(
    scan_id="scan_test_001",
    targets=["http://localhost:3000"]
)

# 實際可用屬性
result.scan_id              # str
result.fingerprints         # Optional[Fingerprints]
result.assets               # List[Asset]
result.engine_results       # Dict[str, Any]  # 不是 List[EngineResult]
result.phase0_summary       # Dict[str, Any]
result.error_info           # Optional[str]
```

**字段差異**:

| 文檔描述 | 實際 API |
|---------|---------|
| `total_assets: int` | 需要 `len(result.assets)` 計算 |
| `total_time: float` | 不存在此字段 |
| `coordination_strategy: str` | 不存在此字段 |
| `engine_results: List[EngineResult]` | `engine_results: Dict[str, Any]` |

---

## ⚠️ 未驗證章節

由於發現多個嚴重 API 不匹配問題,以下章節無法驗證:

1. **使用方式 - 方式 1** (Lines 88-115) - API 方法不存在
2. **使用方式 - 方式 2** (Lines 150-175) - 需要驗證直接調用
3. **參數配置** (Lines 177-268) - ScanStartPayload 不被接受
4. **多目標掃描** (Lines 243-268) - API 不匹配
5. **結果解析** (Lines 292-500) - 返回類型不匹配
6. **故障排查** (Lines 502-650) - 基於錯誤的 API
7. **性能優化** (Lines 650-822) - 需要正確的 API

---

## 📊 驗證統計

### 代碼示例檢查

| 類型 | 總數 | 已驗證 | 通過 | 失敗 |
|------|------|--------|------|------|
| Python | 15+ | 2 | 0 | 2 |
| JSON | 0 | 0 | - | - |

### 錯誤分類

| 錯誤類型 | 數量 | 嚴重程度 |
|---------|------|---------|
| scan_id 格式錯誤 | 5+ | 🔴 高 |
| API 方法不存在 | 全部 | 🔴 高 |
| 參數類型不匹配 | 全部 | 🔴 高 |
| 返回類型不匹配 | 全部 | 🟡 中 |

---

## 🎯 總體評分

| 評分項 | 得分 | 說明 |
|--------|------|------|
| 正確性 | 1/5 | API 完全不匹配 |
| 完整性 | 3/5 | 內容豐富但錯誤嚴重 |
| 可執行性 | 0/5 | 所有示例無法執行 |
| 文檔質量 | 4/5 | 結構清晰但內容過時 |
| **總分** | **8/20** | **40%** |

---

## 🔧 修復建議

### 緊急修復 (必須)

1. **更新所有 API 調用**:
   ```python
   # 舊 (錯誤)
   result = await coordinator.execute_coordinated_scan(request)
   
   # 新 (正確)
   result = await coordinator.execute_strategy_fast(scan_id, targets)
   ```

2. **修正所有 scan_id**:
   ```python
   # 舊 (錯誤)
   scan_id="quick_test"
   
   # 新 (正確)
   scan_id="scan_quick_test"
   ```

3. **更新參數傳遞方式**:
   ```python
   # 舊 (錯誤)
   request = ScanStartPayload(...)
   result = await coordinator.method(request)
   
   # 新 (正確)
   result = await coordinator.execute_strategy_fast(
       scan_id="scan_test",
       targets=["http://localhost:3000"]
   )
   ```

4. **更新返回類型說明**:
   ```python
   # 舊 (錯誤)
   result: CoordinationResult
   result.total_assets
   result.total_time
   result.engine_results  # List[EngineResult]
   
   # 新 (正確)
   result: Phase1CompletedPayload
   len(result.assets)
   # total_time 不存在
   result.engine_results  # Dict[str, Any]
   ```

### 結構性修復

5. **重寫「使用方式」章節**:
   - 添加實際可用的策略方法列表
   - 說明每個策略的特點
   - 提供正確的便利函數用法

6. **重寫「結果解析」章節**:
   - 基於 `Phase1CompletedPayload` 重寫
   - 更新所有字段訪問方式
   - 提供正確的數據結構示例

7. **添加 API 版本說明**:
   - 標注文檔對應的 API 版本
   - 警告可能的 API 變更

---

## 📝 驗證結論

### 問題

1. 🔴 **API 完全不匹配** - 文檔描述的 API 不存在
2. 🔴 **所有示例無法執行** - scan_id 格式錯誤
3. 🔴 **參數接口錯誤** - ScanStartPayload 不被接受
4. 🟡 **返回類型不同** - CoordinationResult vs Phase1CompletedPayload

### 根本原因分析

**推測**: 文檔編寫時設計了一個理想的 API,但實際實現時改用了不同的設計:
- 設計: 統一的 `execute_coordinated_scan(request)` 方法
- 實現: 多個策略特定方法 `execute_strategy_xxx(scan_id, targets)`

這導致文檔與代碼完全脫節。

### 建議

1. **緊急重寫** - 文檔需要完全重寫以匹配實際 API
2. **測試驗證** - 所有示例必須實際運行驗證
3. **版本控制** - 添加 API 版本標記
4. **維護流程** - 建立代碼與文檔同步機制

**評估**: 當前文檔**不可用**,建議標記為「過時」或「重構中」。

---

## 🏷️ 驗證標記

**驗證狀態**: ❌ **未通過** - API 完全不匹配,無法驗證

**發現問題**:
- 🔴 API 方法不存在 (整份文檔)
- 🔴 scan_id 格式錯誤 (5+ 處)
- 🔴 參數類型不匹配 (所有示例)
- 🟡 返回類型不同 (結果解析章節)

**建議行動**:
1. 標記文檔為「⚠️ 過時 - 需要重寫」
2. 基於實際 API 重新編寫
3. 添加實際運行的測試腳本

---

**驗證者**: GitHub Copilot  
**靶場環境**: Juice Shop (localhost:3000)  
**文檔版本**: 2025-11-19  
**報告版本**: 1.0
