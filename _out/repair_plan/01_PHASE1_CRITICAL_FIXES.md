# Phase 1: 關鍵阻塞修復 — 讓掃描流程能跑通

> 優先級: P0（最高）  
> 目標: 修復所有阻止 `POST /scan` 端到端執行的問題  
> 前置條件: 無  
> 驗證方式: 啟動服務 → 送出 POST /scan → 13 步驟流程走完

---

## 問題鏈分析

```
POST /scan
  → app.py 檢查 commander 是否存在
    → commander 在 startup() 中初始化 CommanderCoordinator
      → CommanderCoordinator.__init__() 匯入 AttackCoordinator (延遲)
        → attack_coordinator.py 第 53 行:
          from services.features.function_sqli.detector.sqli_detector import SqliDetector
            → ❌ 該檔案已被刪除！
              → ImportError
                → commander = None
                  → /scan 走 fallback 舊模式（功能受限）
```

**關鍵結論**: 一個被刪除的檔案，導致整條指揮鏈斷裂。

---

## 修復 1.1: SQLi 匯入路徑修正

### 問題
`services/features/function_sqli/detector/sqli_detector.py` 已被刪除，但有 3 個檔案仍在引用它。

### 受影響檔案

| 檔案 | 行號 | 引用方式 |
|------|------|----------|
| `services/core/aiva_core/task_planning/commander/attack_coordinator.py` | 53 | `from services.features.function_sqli.detector.sqli_detector import SqliDetector` |
| `services/features/function_sqli/__init__.py` | 24 | `from .detector.sqli_detector import SqliDetector` |
| `services/core/aiva_core/service_backbone/api/unified_function_caller.py` | 277 | `from services.function.function_sqli.aiva_func_sqli.smart_sqli_detector import ...` |

### 現有替代方案

重構後的 SQLi 模組使用 `SmartDetectionManager` 作為統一入口：

```
services/features/function_sqli/
  ├── smart_detection_manager.py    ← SmartDetectionManager (新的統一入口)
  ├── engines/
  │   ├── base_detector.py          ← BaseDetector (抽象基類)
  │   ├── boolean_detection_engine.py
  │   ├── error_detection_engine.py
  │   ├── time_detection_engine.py
  │   ├── union_detection_engine.py
  │   └── oob_detection_engine.py
  └── (已刪除) detector/sqli_detector.py
```

### 修復動作

**選項 A（推薦）: 建立相容層**

建立 `services/features/function_sqli/detector/__init__.py` 和 `sqli_detector.py`，內容為轉接 SmartDetectionManager：

```python
# services/features/function_sqli/detector/__init__.py
"""Compatibility shim — 轉接到 SmartDetectionManager"""

# services/features/function_sqli/detector/sqli_detector.py
"""SqliDetector 相容層

原始 SqliDetector 已重構為 SmartDetectionManager。
此檔案保留向後相容的匯入路徑。
"""
from services.features.function_sqli.smart_detection_manager import SmartDetectionManager


class SqliDetector:
    """向後相容的 SQL 注入檢測器
    
    內部委託給 SmartDetectionManager。
    attack_coordinator.py 和其他模組可繼續用 SqliDetector 這個名稱。
    """
    
    def __init__(self, *args, **kwargs):
        self._manager = SmartDetectionManager(*args, **kwargs)
    
    async def detect(self, target_url, params, method="GET"):
        """轉接到 SmartDetectionManager"""
        return await self._manager.run_detection(target_url, params, method)
    
    @property
    def engines(self):
        return self._manager.detector_classes
```

**選項 B: 直接修改所有引用**

將 3 個引用點改為直接使用 `SmartDetectionManager`。更乾淨但改動範圍大。

```python
# attack_coordinator.py 第 53 行
# 改前:
from services.features.function_sqli.detector.sqli_detector import SqliDetector
# 改後:
from services.features.function_sqli.smart_detection_manager import SmartDetectionManager as SqliDetector
```

```python
# function_sqli/__init__.py 第 24 行
# 改前:
from .detector.sqli_detector import SqliDetector
# 改後:
from .smart_detection_manager import SmartDetectionManager as SqliDetector
```

### 建議

採用**選項 A**，因為：
1. 不需修改已有程式碼，減少連鎖風險
2. `SqliDetector` 這個名稱已被多處引用，保持一致
3. 未來可逐步遷移到 SmartDetectionManager

### 驗證指令

```powershell
python -c "import sys; sys.path.insert(0,'.'); from services.features.function_sqli.detector.sqli_detector import SqliDetector; print('OK:', SqliDetector)"
```

---

## 修復 1.2: AttackCoordinator 匯入修復

### 問題
`attack_coordinator.py` 匯入 `SqliDetector` 失敗後，整個模組無法載入。
由 `CommanderCoordinator.__init__.py` 中的 `from .attack_coordinator import AttackCoordinator` 觸發。

### 修復動作

完成 1.1 後，此問題自動解決。但需要額外檢查另一個可能斷裂的匯入：

```python
# attack_coordinator.py 第 37-45 行
try:
    from services.features.function_exploit.executor.attack_executor import AttackExecutor
except ImportError as e:
    raise ImportError(
        "❌ 缺少必要依賴 AttackExecutor\n"
        ...
    ) from e
```

需確認 `services/features/function_exploit/executor/attack_executor.py` 存在。

### 驗證指令

```powershell
python -c "
import sys; sys.path.insert(0,'.')
from services.core.aiva_core.task_planning.commander.attack_coordinator import AttackCoordinator
print('OK: AttackCoordinator imported')
ac = AttackCoordinator()
print('OK: AttackCoordinator instantiated')
"
```

---

## 修復 1.3: CommanderCoordinator 初始化驗證

### 問題
即使 1.1 和 1.2 修好，`CommanderCoordinator` 在 app.py startup 中仍可能因其他原因失敗。
目前整個初始化被 try/except 包裹，失敗時 `commander = None`。

### 需要確認的項目

1. **AttackCoordinator 是否能被 CommanderCoordinator 延遲載入？**
   - `CommanderCoordinator.__init__()` 不會立即建立 AttackCoordinator（使用 `@property` 延遲載入）
   - 只有當 `/scan` 觸發 `execute_command(TWO_PHASE_SCAN, ...)` 時才初始化
   - 但 `from .attack_coordinator import AttackCoordinator` 在 `__init__.py` 頂層匯入 → **模組載入時就會失敗**

2. **MultiEngineCoordinator 路徑是否正確？**
   ```python
   from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
   ```
   需確認 `services/scan/` 目錄是否存在。

### 修復動作

```python
# app.py 第 325 行 - 現有程式碼已設計為失敗降級:
except Exception as e:
    logger.error(f"❌ [啟動] CommanderCoordinator initialization failed: {e}")
    commander = None
```

修復 1.1 後，此處應能正常初始化。若仍有問題，需追蹤 exception 訊息。

### 驗證指令

```powershell
python -c "
import sys; sys.path.insert(0,'.')
from services.core.aiva_core.task_planning.commander import CommanderCoordinator
c = CommanderCoordinator()
print('OK: CommanderCoordinator created')
print('  Has execute_command:', hasattr(c, 'execute_command'))
"
```

---

## 修復 1.4: RealDecisionEngine.decide() 介面修正

### 問題

`decide()` 方法簽名為 `decide(self, input_data: torch.Tensor) -> torch.Tensor`，
但呼叫端（如測試或 attack_coordinator）可能傳入 `dict`。

### 分析

```python
# 目前的 decide() 實作 (real_neural_core.py 第 912-938 行):
def decide(self, input_data: torch.Tensor) -> torch.Tensor:
    """直接前向傳播 — 要求 Tensor 輸入"""
    self.ai_core.eval()
    with torch.no_grad():
        return self.ai_core(input_data)

# 已有的 encode_input() 方法 (第 427-470 行):
def encode_input(self, text: str) -> torch.Tensor:
    """將文字轉為 512 維 Tensor"""
    ...

# 已有的 generate_decision() 方法:
def generate_decision(self, input_text: str) -> dict:
    """完整的 Bug Bounty 決策流程"""
    ...
```

### 修復動作

在 `RealDecisionEngine` 中加入型別分派，讓 `decide()` 也能接受 dict/str：

```python
def decide(self, input_data):
    """決策方法 — 支援多種輸入格式
    
    Args:
        input_data: torch.Tensor、str 或 dict
    """
    if isinstance(input_data, str):
        input_data = self.encode_input(input_data)
    elif isinstance(input_data, dict):
        # 從 dict 中提取可用文字
        text = input_data.get('payload', '') or input_data.get('url', '') or str(input_data)
        input_data = self.encode_input(text)
    
    self.ai_core.eval()
    with torch.no_grad():
        return self.ai_core(input_data)
```

### 權重缺失 keys 的處理

`aux_output_layer.weight` 和 `aux_output_layer.bias` 在載入權重時缺失，但這是**已知且已處理的情況**：

```python
# real_neural_core.py 第 487-491 行:
_AUX_LAYER_KEYS = {"aux_output_layer.weight", "aux_output_layer.bias"}
# 載入時使用 Xavier 初始化代替
```

此處**不需要修復**，但建議重新訓練模型以包含 aux_output_layer 的權重。

### 驗證指令

```powershell
python -c "
import sys; sys.path.insert(0,'.')
from services.core.aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine
engine = RealDecisionEngine(use_5m_model=True)

# 測試 dict 輸入
result = engine.decide({'url': 'http://example.com', 'type': 'xss'})
print('dict 輸入 OK, 輸出形狀:', result.shape)

# 測試 str 輸入
result = engine.decide('scan http://example.com for XSS')
print('str 輸入 OK, 輸出形狀:', result.shape)

# 測試 Tensor 輸入
import torch
result = engine.decide(torch.randn(1, 512))
print('Tensor 輸入 OK, 輸出形狀:', result.shape)
"
```

---

## 修復 1.5: 端到端 POST /scan 測試

### 前置條件

修復 1.1 ~ 1.4 全部完成。

### 測試步驟

```powershell
# 步驟 1: 驗證 app 能載入
python -c "
import sys; sys.path.insert(0,'.')
from services.core.aiva_core.service_backbone.api.app import app
print('Routes:', [r.path for r in app.routes if hasattr(r, 'path')])
"

# 步驟 2: 啟動服務
cd C:\D\fold7\AIVA-git
python -m uvicorn services.core.aiva_core.service_backbone.api.app:app --host 0.0.0.0 --port 8000

# 步驟 3: 在另一個終端測試 (PowerShell)
$body = @{
    target = "http://testphp.vulnweb.com"
    scan_type = "quick"
    max_depth = 2
    timeout = 60
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/scan" -Method POST -Body $body -ContentType "application/json"
```

### 預期結果

1. 服務啟動時：`✅ [啟動] CommanderCoordinator ready`（不是 commander = None）
2. POST /scan 返回 `scan_id` + `status: started`
3. 日誌顯示 13 步驟執行過程
4. 不需要 RabbitMQ（自動走本地直接執行模式）

### 可能的額外問題

1. **`services/scan/coordinators/multi_engine_coordinator.py` 可能不存在**
   - 影響: `multilang_coordinator = None`（不致命，掃描仍可執行）
   
2. **`AttackExecutor` 匯入可能失敗**
   - `attack_coordinator.py` 第 37-45 行是 hard import（會 raise）
   - 需確認 `services/features/function_exploit/executor/attack_executor.py` 存在

3. **Phase0 的實際工具（httpx, nuclei 等）可能未安裝**
   - `_plan_phase0_tasks()` 規劃了 http_probe、port_scan、directory_scan、tech_detect 等
   - `_execute_parallel_tasks()` 中使用 subprocess 呼叫這些工具
   - 如果外部工具不在 PATH 中，個別 task 會失敗但不影響整體流程

---

## 修復順序與驗證流程

```
1.1 建立 SqliDetector 相容層
  ↓ 驗證: import SqliDetector 成功
1.2 確認 AttackCoordinator import
  ↓ 驗證: import AttackCoordinator 成功
1.3 確認 CommanderCoordinator init
  ↓ 驗證: CommanderCoordinator() 成功
1.4 修復 decide() 介面
  ↓ 驗證: engine.decide({...}) 成功
1.5 完整端到端測試
  ↓ 驗證: POST /scan → 13 步驟完成
```

每一步驟都有明確的驗證指令，修復一個驗證一個，不要跳步。
