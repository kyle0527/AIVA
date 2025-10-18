p AIVA 系統統一修正計劃

## 掃描日期
2025-10-19

## 一、五大模組架構現況

### ✅ 已存在的模組
1. **core** (aiva_core) - 95個Python檔案
   - AI引擎、學習系統、RAG、訓練編排器
   
2. **scan** (aiva_scan) - 31個Python檔案
   - 漏洞掃描、目標探測、環境檢測
   
3. **integration** (aiva_integration) - 53個Python檔案
   - 系統整合、性能監控、跨語言協調
   
4. **common** (aiva_common) - 33個Python檔案
   - 共用schemas、枚舉、工具類別

### ❌ 缺失的模組
5. **attack** (aiva_attack) - **不存在**
   - 需要創建完整的攻擊執行模組

## 二、Schemas 內容分析

### 當前 Schemas 檔案結構 (13個檔案)
```
services/aiva_common/schemas/
├── __init__.py          (導出155個類別)
├── ai.py               (22,673 bytes, 26個class) - AI/訓練/攻擊計劃
├── tasks.py            (19,443 bytes, 33個class) - 任務/場景定義
├── telemetry.py        (14,477 bytes, 14個class) - 遙測/監控
├── enhanced.py         (13,947 bytes, 10個class) - 增強型schemas
├── findings.py         ( 8,469 bytes, 13個class) - 漏洞發現
├── system.py           ( 8,265 bytes,  6個class) - 系統狀態
├── languages.py        ( 7,501 bytes,  9個class) - 多語言支援
├── assets.py           ( 6,373 bytes,  6個class) - 資產管理
├── references.py       ( 5,033 bytes,  4個class) - CVE/CWE引用
├── risk.py             ( 3,176 bytes,  7個class) - 風險評估
├── base.py             ( 2,947 bytes, 10個class) - 基礎類別
├── messaging.py        ( 2,028 bytes,  5個class) - 訊息佇列
└── api_testing.py      (   253 bytes,  0個class) - API測試 (空)
```

### 導出類別統計 (155個)
- **AI相關**: 31個 (AttackPlan, AttackTarget, ModelTraining等)
- **Scan/Attack相關**: 19個 (Vulnerability, Exploit, EASM等)
- **Message/Task相關**: 36個 (各種Payload和Event)
- **Finding/Risk相關**: 10個 (CVSS, Risk評估等)
- **System/Telemetry相關**: 3個 (系統監控)
- **Base/Common相關**: 1個 (ModuleStatus)
- **Other**: 55個 (包含子模組導出)

## 三、識別的問題

### 問題1: Attack 模組完全缺失 ❌
**現況**: 
- `services/attack/aiva_attack` 目錄不存在
- 攻擊相關功能散落在各處

**影響**:
- 違反五大模組架構設計
- 攻擊邏輯無統一管理
- 與 scan/core 模組耦合過緊

**解決方案**:
```
創建 services/attack/aiva_attack/ 模組
├── __init__.py
├── attack_executor.py      # 攻擊執行器
├── exploit_manager.py      # 漏洞利用管理
├── payload_generator.py    # Payload生成器
├── attack_chain.py         # 攻擊鏈編排
└── attack_validator.py     # 攻擊結果驗證
```

### 問題2: Schemas 缺失核心配置類別 ❌
**缺失的 Schemas** (4個):
1. `TrainingOrchestratorConfig` - 訓練編排器配置
2. `ExperienceManagerConfig` - 經驗管理器配置
3. `ModelTrainerConfig` - 模型訓練器配置 (已有但不完整)
4. `PlanExecutorConfig` - 計劃執行器配置

**解決方案**:
在 `ai.py` 中新增這些配置類別

### 問題3: 命名不一致 ⚠️
**問題實例**:
1. `AIVACommand` vs `AivaMessage` - 大小寫不一致
2. `ScanStartPayload` vs `ScanScope` - 後綴不統一
3. `EnhancedVulnerability` vs `Vulnerability` - Enhanced用途不明確
4. `FunctionTask` vs `Task` - 可能有功能重複

**統一規範建議**:
- 類別名稱: PascalCase
- 組織名稱: 統一使用 `AIVA` (全大寫)
- Payload後綴: 用於跨服務通訊的資料
- Request後綴: 用於API請求
- Result後綴: 用於回應結果
- Config後綴: 用於配置
- Enhanced前綴: 僅用於擴展現有類別的增強版本

### 問題4: __init__.py 部分為空 ⚠️
**現況**:
- `services/scan/aiva_scan/__init__.py` - 0 bytes
- `services/integration/aiva_integration/__init__.py` - 0 bytes

**影響**: 
- 無法直接導入子模組
- 模組結構不清晰

**解決方案**:
為每個模組添加完整的 `__init__.py` 並導出主要類別

### 問題5: 導入路徑混亂 ❌
**問題實例**:
```python
# 有些使用相對導入
from .ai_engine import BioNeuronRAGAgent

# 有些使用絕對導入
from services.core.aiva_core.ai_engine import BioNeuronRAGAgent

# 有些使用錯誤的導入
from aiva_core.ai_engine import BioNeuronRAGAgent  # ❌ 找不到
```

**統一規範**:
- 模組內部: 使用相對導入 (`.`)
- 跨模組: 使用絕對導入 (`services.xxx.yyy`)
- 添加 try/except 容錯機制

## 四、修正優先級

### 階段1: 緊急修正 (必須)
1. ✅ 創建 `services/attack/aiva_attack` 模組
2. ✅ 補充缺失的 Schemas 配置類別
3. ✅ 統一所有導入路徑
4. ✅ 完善空白的 `__init__.py` 檔案

### 階段2: 命名統一 (重要)
1. ⚠️ 統一 AIVA 相關類別命名 (AIVAMessage, AIVACommand等)
2. ⚠️ 統一後綴命名 (Payload, Request, Result, Config)
3. ⚠️ 明確 Enhanced 類別的使用場景
4. ⚠️ 移除或合併重複的類別

### 階段3: 優化重構 (建議)
1. 📝 將過大的 schemas 檔案拆分 (ai.py 22KB, tasks.py 19KB)
2. 📝 整理 "Other" 分類中的55個雜項類別
3. 📝 移除空的 api_testing.py 或補充內容
4. 📝 建立 schemas 的完整文檔

## 五、修正執行計劃

### Step 1: 創建 Attack 模組
```bash
# 創建目錄結構
mkdir -p services/attack/aiva_attack
```

**需要創建的檔案**:
- `__init__.py` - 模組初始化和導出
- `attack_executor.py` - 核心攻擊執行器
- `exploit_manager.py` - 漏洞利用管理
- `payload_generator.py` - Payload 生成
- `attack_chain.py` - 攻擊鏈編排
- `attack_validator.py` - 結果驗證

### Step 2: 補充 Schemas
在 `services/aiva_common/schemas/ai.py` 中添加:

```python
class TrainingOrchestratorConfig(BaseModel):
    """訓練編排器配置"""
    orchestrator_id: str
    enabled_trainers: list[str] = Field(default_factory=list)
    training_interval: int = 3600  # 秒
    auto_deploy: bool = False
    max_parallel_trainings: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)

class ExperienceManagerConfig(BaseModel):
    """經驗管理器配置"""
    manager_id: str
    storage_backend: str = "sqlite"
    max_experiences: int = 10000
    retention_days: int = 90
    auto_cleanup: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

class PlanExecutorConfig(BaseModel):
    """計劃執行器配置"""
    executor_id: str
    max_concurrent_plans: int = 5
    timeout_seconds: int = 300
    retry_policy: dict[str, Any] = Field(default_factory=dict)
    safety_checks_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Step 3: 統一命名
創建命名映射表並逐步重命名:
- `AivaMessage` → `AIVAMessage`
- `AIVACommand` → 保持不變
- 所有 Enhanced 類別添加使用說明

### Step 4: 修正導入路徑
為所有模組檔案添加標準導入模式:
```python
try:
    # 相對導入 (模組內部)
    from .submodule import Class
except ImportError:
    # 絕對導入 (跨模組容錯)
    from services.module.aiva_module.submodule import Class
```

### Step 5: 完善 __init__.py
為空白的 `__init__.py` 添加標準模版:
```python
"""
AIVA Module Name
Description...
"""

from .main_class import MainClass
from .helper import Helper

__all__ = [
    "MainClass",
    "Helper",
]

__version__ = "1.0.0"
```

## 六、驗證檢查清單

修正完成後需要驗證:

- [ ] 五大模組全部存在且結構完整
- [ ] 所有 Schemas 配置類別已補充
- [ ] 導入路徑統一且無錯誤
- [ ] __init__.py 檔案完整
- [ ] 命名規範一致
- [ ] 現有測試全部通過
- [ ] 生成完整的架構文檔

## 七、風險評估

### 高風險操作
- 重命名現有類別 (會影響現有代碼)
- 修改導入路徑 (需要全面測試)

### 中風險操作
- 創建新模組 (需確保不衝突)
- 新增 Schemas (需確保向後兼容)

### 低風險操作
- 完善 __init__.py (純新增內容)
- 添加註釋和文檔 (不影響功能)

## 八、時間估算

- **階段1 (緊急修正)**: 2-3小時
- **階段2 (命名統一)**: 3-4小時
- **階段3 (優化重構)**: 4-6小時
- **測試驗證**: 2小時
- **文檔更新**: 1小時

**總計**: 12-16小時

## 九、建議執行順序

1. **立即執行**: 創建 Attack 模組 (解決架構完整性)
2. **立即執行**: 補充缺失的 Schemas (解決導入錯誤)
3. **立即執行**: 修正導入路徑 (解決當前運行錯誤)
4. **本週完成**: 完善 __init__.py
5. **本週完成**: 統一命名規範
6. **下週進行**: 優化重構和文檔

---

**報告生成時間**: 2025-10-19
**執行狀態**: 待執行
**負責人**: AI System Architect
