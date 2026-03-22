# AIVA 已驗證的致命問題清單

> 建立日期：2026-03-21
> 驗證方式：直接讀取原始碼 + Python AST 靜態分析 + 實際執行 JSON 解析
> 標記說明：🔴 致命 / 🟠 高 / 🟡 中

---

## BUG-001 🔴 欄位名稱錯誤導致 525 個 flows 全部失效

**位置：** `services/core/aiva_core/internal_exploration/aiva_external_executor.py`

**已驗證：**
```python
# 執行器使用：
f.get("operable")        # ← 永遠回傳 None（欄位不存在）

# JSON 中實際欄位名稱：
f.get("is_operable")     # ← 287 個為 True
```

**實測數據：**
```
總 flows：525
operable=True：0（全部失效）
is_operable=True：287（實際應可用）
```

**影響：** AI 啟動後可執行能力數量 = 0，整個功能執行層完全失效。

**修復（1 行）：**
```python
# 找到所有 f.get("operable") 改成：
f.get("is_operable", False)
```

**搜尋位置：**
```bash
grep -n '"operable"' services/core/aiva_core/internal_exploration/aiva_external_executor.py
```

---

## BUG-002 🔴 external_classification.json 全為 Windows 絕對路徑

**位置：** `services/integration/data/internal_exploration/external_classification.json`

**已驗證：**
```
總 flows：525
Windows 路徑（C:\D\fold7\AIVA-git\...）：521
無路徑（Go/Rust binary）：4
Linux 可用路徑：0
```

**影響：** 執行器嘗試用 `C:\D\fold7\...` import Python 模組，在 Linux 環境全部失敗。

**修復：**
```bash
# 在 Linux 環境重新執行分類器
python services/core/aiva_core/internal_exploration/aiva_external_classifier.py
```
重新產生 JSON，路徑會自動變成當前環境的 Linux 格式。

---

## BUG-003 🔴 CAPABILITY_CONFIGS 全部 14 筆的 module/class/entry 為 None

**位置：** `services/aiva_common/enums/capabilities.py`

**已驗證：**
```python
from services.aiva_common.enums.capabilities import CAPABILITY_CONFIGS
# 輸出：
# Total CAPABILITY_CONFIGS: 14
# [sql_injection] module=None entry=None class=None
# [xss_reflected] module=None entry=None class=None
# ... 全部 14 筆都是 None
```

**影響：** 任何透過枚舉系統查詢「要呼叫哪個模組的哪個類別」的流程，結果都是 `None`。`register_standardized_capabilities.py` 注冊的全部能力無法被找到。

**修復：** 補全對應關係，參見 `docs/capability_mapping_todo.md` 中的 CAPABILITY_CONFIGS 範本。

---

## BUG-004 🔴 core capability_registry 本地快取永遠空，sync 從未自動呼叫

**位置：** `services/core/aiva_core/core_capabilities/capability_registry.py:305-320`

**已驗證程式碼：**
```python
def get_capability(self, name: str) -> CapabilityInfo | None:
    if name in self._capabilities:        # 只查本地快取
        return self._capabilities.get(name)
    # 注意：如需查詢 integration registry，請使用 sync_from_integration_registry()
    logger.debug(f"Capability {name} not found in local cache")
    return None                           # 永遠 None
```

**問題：** `sync_from_integration_registry()` 存在但從未被自動呼叫。啟動流程中找不到任何呼叫點。

**影響：** 所有 `get_capability(name)` 呼叫永遠回傳 `None`，即使 integration registry 有該能力。

**修復：**
```python
# 在系統啟動時（app.py 或 core 初始化）加入：
capability_registry = get_capability_registry()
await capability_registry.sync_from_integration_registry()
```

---

## BUG-005 🟠 attack_coordinator fallback 缺少主要功能模組

**位置：** `services/core/aiva_core/task_planning/commander/attack_coordinator.py`

**已驗證程式碼：**
```python
if self.unified_executor:
    # 使用完整的並行執行器 ✅
else:
    handlers = {
        "httpx": self._execute_httpx_tool,
        "port_scanner": self._execute_port_scanner_tool,
        "waf_detector": self._execute_waf_detector_tool,
    }
    # function_sqli / xss / ssrf / idor 等... 全部不在 handlers 裡
```

**影響：** 當 `unified_executor` 初始化失敗（None）時，SQLi/XSS/SSRF/IDOR 等核心測試全部無法執行，只剩 3 個 fallback 工具。

**修復：**
1. 確保 `unified_executor` 在所有環境下都能正常初始化
2. 或將核心功能模組加入 fallback handlers

---

## BUG-006 🟠 RAG 為 None 時靜默回傳空結果，決策繼續執行

**位置：** `services/core/aiva_core/cognitive_core/internal_loop_connector.py:501`

**已驗證程式碼：**
```python
if self.rag_kb is None:
    logger.error(CapabilityScopeClassifier.ERROR_RAG_NOT_INITIALIZED)
    return RAGQueryResult(
        results=[],
        total_found=0,
        ...
    )
# 上層收到空結果，不知道是「沒有相關知識」還是「RAG 根本沒連線」
```

**影響：** AI 在 RAG 未連線的情況下會繼續運作，但所有決策都是基於空的知識庫。錯誤不會傳播到上層。

**修復：**
```python
if self.rag_kb is None:
    raise RuntimeError("RAG knowledge base not initialized")
    # 或在 RAGQueryResult 加入 is_fallback: bool 欄位，讓上層知道
```

---

## BUG-007 🟠 external_loop_connector 訓練結果不持久化

**位置：** `services/core/aiva_core/cognitive_core/external_loop_connector.py`

**問題：**
```python
async def _register_new_weights(self, training_result):
    # 只記錄 log，沒有寫回磁碟
    logger.info(f"New weights registered: {training_result.model_id}")
    # 缺少：await self.weight_manager.save(training_result.weights)
```

**觸發條件魔術數字（無調整機制）：**
```python
def _is_significant_deviation(self, deviations):
    if len(deviations) >= 3: return True    # 3 個偏差就觸發？
    if total_score >= 5.0: return True      # 5.0 怎麼來的？
```

**影響：** 訓練產生的新權重在重啟後消失，學習迴路實際上沒有累積效果。

---

## BUG-008 🟡 全系統「靜默成功」模式隱藏錯誤

**分佈：** 遍佈全部核心模組

**模式：**
```python
try:
    result = await something_critical()
except Exception as e:
    logger.error(f"Failed: {e}")
    return {}    # 返回空結果，不拋出異常
```

**影響：** 上層拿到空 `{}` 繼續往下走，最終做出基於空資料的決策。問題不是崩潰，而是靜默的錯誤決策，比崩潰更難除錯。

**建議：** 對關鍵路徑（RAG 查詢、能力執行、攻擊結果處理）使用 Result 型別或明確的 error flag，而非空 dict。

---

## 其他已發現的設計問題（非 Bug，但需注意）

### D-001 ability registry 有兩套（core vs integration）

- `services/core/aiva_core/core_capabilities/capability_registry.py` — 本地快取，Proxy 模式
- `services/integration/capability/registry.py` — 實際儲存（SQLite + ChromaDB）

兩套之間的同步必須手動觸發 `sync_from_integration_registry()`，沒有自動同步機制。

### D-002 classification 資料兩套來源

- `external_classification.json` — 外部模組（by `aiva_external_classifier.py`）
- `internal_classification.json` — 內部模組（by `aiva_internal_classifier.py`？）

兩者產生時機不同，可能出現不一致。

### D-003 Web Scanner integration_tools 有 4 個 NotImplementedError

**位置：** `services/features/function_web_scanner/integration_tools/web_tools.py`

```python
class SubdomainEnumerator:
    def enumerate_subdomains(self, domain): raise NotImplementedError
class DirectoryScanner:
    def scan_directories(self, ...): raise NotImplementedError
class VulnerabilityScanner:
    def scan_vulnerabilities(self, ...): raise NotImplementedError
class TechnologyDetector:
    def detect_technologies(self, ...): raise NotImplementedError
```

應改用 `scanners/` 目錄下的具體實作，而非這些包裝類別。

---

## 修復優先順序

| 優先度 | Bug ID | 說明 | 預估工時 |
|:------:|--------|------|:--------:|
| 🔴 P0 | BUG-001 | `operable` → `is_operable` 欄位名稱 | 10 分鐘 |
| 🔴 P0 | BUG-002 | 重跑 classifier 產生 Linux 路徑 | 30 分鐘 |
| 🔴 P1 | BUG-003 | 補全 CAPABILITY_CONFIGS | 2-4 小時 |
| 🔴 P1 | BUG-004 | 啟動時加入 sync_from_integration_registry() | 1 小時 |
| 🟠 P2 | BUG-005 | attack_coordinator fallback 補完 | 2 小時 |
| 🟠 P2 | BUG-006 | RAG None 時明確傳播錯誤 | 1 小時 |
| 🟠 P2 | BUG-007 | 訓練結果持久化 | 2 小時 |
| 🟡 P3 | BUG-008 | 靜默成功模式重構 | 1-2 天 |
| 🟡 P3 | D-003 | Web Scanner NotImplementedError 修復 | 2 小時 |
