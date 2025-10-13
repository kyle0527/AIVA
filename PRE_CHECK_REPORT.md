# AIVA 前置檢查報告

**檢查日期**: 2025-10-13  
**目的**: 確認需新增功能的前置條件

---

## ✅ 檢查結果

### 1. ConfigControlCenter - ✅ 存在
**檔案**: `services/scan/aiva_scan/config_control_center.py`  
**狀態**: 已確認存在

### 2. StrategyController.apply_to_config - ✅ 存在
**方法位置**: `services/scan/aiva_scan/strategy_controller.py:287`  
**範例使用**: Line 94 有使用範例

**程式碼片段**:
```python
# Line 94 - 使用範例
controller.apply_to_config(config_center)

# Line 287 - 方法定義
def apply_to_config(self, config_center) -> None:
    ...
```

**結論**: ✅ 方法已存在,**但在 ScanOrchestrator 中未被呼叫**

---

### 3. RiskAssessmentEngine - ❌ 不存在
**預期路徑**: `services/core/aiva_core/analysis/risk_assessment_engine.py`  
**狀態**: ❌ 檔案不存在

**可用檔案** (在 `services/core/aiva_core/analysis/` 目錄):
- `strategy_generator.py` ✅
- `initial_surface.py` ✅
- `dynamic_strategy_adjustment.py` ✅

**結論**: 需要**新建** `risk_assessment_engine.py` 檔案

---

### 4. SensitiveMatch / JavaScriptAnalysisResult Schema - ❓ 待確認
**檢查命令**未返回結果,需進一步確認

**建議**: 檢查以下檔案:
```powershell
# 檢查完整的 Schema 定義
Get-Content "c:\D\E\AIVA\AIVA-main\services\aiva_common\schemas.py" | Select-String "Sensitive|JavaScript"

# 或檢查 info_gatherer 相關 Schema
Get-Content "c:\D\E\AIVA\AIVA-main\services\scan\aiva_scan\info_gatherer\*.py" | Select-String "class.*Match|class.*Result"
```

---

## 📊 任務可行性分析

| 任務 | 前置條件 | 狀態 | 可行性 |
|------|---------|------|--------|
| **P0-1: 重構 worker.py** | ScanOrchestrator 存在 | ✅ 已確認 | ✅ 立即可行 |
| **P0-2: StrategyController 整合** | ConfigControlCenter + apply_to_config | ✅ 都存在 | ✅ 立即可行 |
| **P1-3: 動態引擎擴充** | ScanContext 方法 | ⚠️ 需先新增 | ⚠️ 依賴 P1-4 |
| **P1-4: ScanContext 新增欄位** | Schema 定義 | ❓ 待確認 | ⚠️ 可能需先建 Schema |
| **P2-5: ThreatIntel 整合** | RiskAssessmentEngine | ❌ 不存在 | ❌ 需先建立檔案 |
| **P2-6: 撰寫文檔** | 無 | ✅ | ✅ 立即可行 |

---

## 🎯 修正後的執行建議

### 階段 1: 立即可行 (本週)

#### ✅ Task 1: 重構 worker.py (4 小時)
**前置條件**: 無 (ScanOrchestrator 已存在)  
**優先級**: P0

#### ✅ Task 2: StrategyController 整合 (2 小時)
**前置條件**: 無 (ConfigControlCenter 和 apply_to_config 都存在)  
**優先級**: P0

**實作**:
```python
# services/scan/aiva_scan/scan_orchestrator.py

async def execute_scan(self, request: ScanStartPayload):
    # ... 現有代碼 ...
    
    strategy_controller = StrategyController(request.strategy)
    
    # ✨ 新增: 應用策略到配置中心
    from .config_control_center import ConfigControlCenter
    config_center = ConfigControlCenter.get_instance()
    strategy_controller.apply_to_config(config_center)
    
    strategy_params = strategy_controller.get_parameters()
    # ...
```

#### ✅ Task 3: 撰寫文檔 (4 小時)
**前置條件**: 無  
**優先級**: P2 (但可提前完成)

---

### 階段 2: 需先準備 Schema (下週)

#### ⚠️ Task 4.1: 確認/建立 Schema (2 小時)
**優先級**: P1

**需確認**:
1. `SensitiveMatch` Schema 是否存在
2. `JavaScriptAnalysisResult` Schema 是否存在

**如不存在,需建立**:
```python
# services/aiva_common/schemas.py

class SensitiveMatch(BaseModel):
    """敏感資訊匹配結果"""
    match_id: str
    pattern_name: str  # e.g., "password", "api_key", "credit_card"
    matched_text: str
    context: str  # 前後文
    confidence: float  # 0.0 - 1.0
    line_number: int | None = None
    file_path: str | None = None

class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""
    analysis_id: str
    url: str
    findings: list[str]  # e.g., ["uses_eval", "dom_manipulation"]
    apis_called: list[str]  # 發現的 API 端點
    suspicious_patterns: list[str]
    risk_score: float  # 0.0 - 10.0
```

#### ⚠️ Task 4.2: ScanContext 新增方法 (2 小時)
**優先級**: P1  
**依賴**: Task 4.1

#### ⚠️ Task 5: 動態引擎擴充 (6 小時)
**優先級**: P1  
**依賴**: Task 4.2

---

### 階段 3: 建立新檔案 (第三週)

#### ❌ Task 6.1: 建立 RiskAssessmentEngine (4 小時)
**優先級**: P2

**新建檔案**: `services/core/aiva_core/analysis/risk_assessment_engine.py`

**基礎結構**:
```python
"""
風險評估引擎

根據漏洞類型、CVSS 分數、威脅情報等多維度評估風險。
"""

from services.aiva_common.schemas import FindingPayload
from services.threat_intel.intel_aggregator import IntelAggregator

class RiskAssessmentEngine:
    """風險評估引擎"""
    
    def __init__(self):
        self.intel_aggregator = IntelAggregator()
    
    async def assess_risk(self, finding: FindingPayload) -> float:
        """
        評估漏洞風險分數
        
        Returns:
            float: 0.0 - 10.0 的風險分數
        """
        # 1. 基礎 CVSS 分數
        base_score = self._calculate_cvss_score(finding)
        
        # 2. 威脅情報調整
        if finding.cve_id:
            intel = await self.intel_aggregator.query_cve(finding.cve_id)
            if intel and intel.is_actively_exploited:
                base_score *= 1.5
        
        return min(base_score, 10.0)
    
    def _calculate_cvss_score(self, finding: FindingPayload) -> float:
        """計算基礎 CVSS 分數"""
        # 根據 severity 映射到分數
        severity_scores = {
            "critical": 9.0,
            "high": 7.5,
            "medium": 5.0,
            "low": 3.0,
            "info": 1.0,
        }
        return severity_scores.get(finding.severity.lower(), 5.0)
```

#### ❌ Task 6.2: 整合 ThreatIntel (2 小時)
**優先級**: P2  
**依賴**: Task 6.1

---

## 🚀 最終執行順序 (修正版)

### Week 1: 快速勝利 (10 小時)
1. ✅ Task 2: StrategyController 整合 (2h) - **優先執行**
2. ✅ Task 1: 重構 worker.py (4h)
3. ✅ Task 3: 撰寫文檔 (4h)

### Week 2: Schema 準備 (10 小時)
4. ⚠️ Task 4.1: 確認/建立 Schema (2h)
5. ⚠️ Task 4.2: ScanContext 新增方法 (2h)
6. ⚠️ Task 5: 動態引擎擴充 (6h)

### Week 3: 新檔案建立 (6 小時)
7. ❌ Task 6.1: 建立 RiskAssessmentEngine (4h)
8. ❌ Task 6.2: 整合 ThreatIntel (2h)

---

## 📝 下一步立即行動

### Action 1: 確認 Schema (優先)
```powershell
# 搜尋 SensitiveMatch
Select-String -Path "c:\D\E\AIVA\AIVA-main\services" -Pattern "class SensitiveMatch" -Recurse

# 搜尋 JavaScriptAnalysisResult  
Select-String -Path "c:\D\E\AIVA\AIVA-main\services" -Pattern "class JavaScript.*Result" -Recurse

# 檢查 info_gatherer 的返回類型
Get-Content "c:\D\E\AIVA\AIVA-main\services\scan\aiva_scan\info_gatherer\sensitive_info_detector.py" | Select-String "def detect" -Context 5
```

### Action 2: 開始 Task 2 (立即可行)
```python
# 修改 scan_orchestrator.py
# 位置: execute_scan 方法,第 ~85 行

# 在 strategy_controller = StrategyController(request.strategy) 後新增:
from .config_control_center import ConfigControlCenter

config_center = ConfigControlCenter.get_instance()
strategy_controller.apply_to_config(config_center)
logger.info("Strategy parameters applied to ConfigControlCenter")
```

---

**下次更新**: 完成 Schema 確認後
