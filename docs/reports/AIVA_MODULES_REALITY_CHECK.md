# AIVA 六大模組實際運作狀況檢查報告

**檢查日期**: 2025-11-29  
**方法**: 直接執行核心組件，記錄實際可用功能

---

## 📊 執行摘要

系統共有 **782 個能力**，分布於 4 種語言 (Python 63.3%, Rust 15.7%, TypeScript 10.7%, Go 10.2%)

**實際測試結果**: 
- ✅ **可用**: 內部探索、CLI 統計
- ⚠️ **部分可用**: AICommander (使用降級版本)
- ❌ **不可用**: BioNeuron 決策控制器、掃描協調器、Web 掃描器、整合協調器

---

## 1️⃣ Core 核心模組 (services/core/aiva_core)

### 📁 結構
```
cognitive_core/          # 認知核心 (AI 大腦)
├── neural/              # 神經網路 (5M 參數 BioNeuron)
├── decision/            # 決策代理
└── ai_capability_query.py
task_planning/           # 任務規劃
├── ai_commander.py      # AI 指揮官
core_capabilities/       # 核心能力
├── attack/              # 攻擊執行器
├── dialog/              # 對話助理
├── processing/          # 處理引擎
external_learning/       # 外部學習
internal_exploration/    # 內部探索 ✅
service_backbone/        # 服務骨幹
ui_panel/                # UI 面板
```

### 🧪 測試結果

#### ❌ BioNeuronDecisionController
```python
from services.core.aiva_core.cognitive_core.neural.bio_neuron_master import BioNeuronDecisionController
controller = BioNeuronDecisionController()
```
**錯誤**: `NameError: name 'SeverityLevel' is not defined`
- 缺少 SeverityLevel 枚舉的導入
- 權重儲存失敗: `'RealScalableBioNet' object has no attribute 'total_params'`

#### ⚠️ AICommander (部分可用)
```python
from services.core.aiva_core.task_planning.ai_commander import AICommander
commander = AICommander()
# ✅ 可實例化
```
**警告**:
- `Failed to import AI components: cannot import name 'AIVAExperienceManager'`
- `RAG Engine not available: name 'VectorStore' is not defined`
- `Using simplified ExperienceManager` (使用降級版本)
- `MultiLanguageAICoordinator not available`

#### ✅ AttackExecutor (可用)
```python
from services.core.aiva_core.core_capabilities.attack.attack_executor import AttackExecutor
ae = AttackExecutor()
# ✅ 成功實例化
```

#### ✅ Internal Exploration (完全可用)
```python
from services.core.aiva_core.internal_exploration.module_explorer import ModuleExplorer
explorer = ModuleExplorer()
result = await explorer.explore_all_modules()
# ✅ 成功掃描 4 個模組，459 個文件
```
**成功掃描**:
- core/aiva_core: 129 個文件 (Python only)
- scan: 103 個文件 (Python 55, Go 16, Rust 9, TypeScript 16, JS 7)
- features: 140 個文件 (Python 130, Go 9, Rust 1)
- integration: 87 個文件 (Python only)

---

## 2️⃣ Scan 掃描模組 (services/scan)

### 📁 結構
```
coordinators/
├── multi_engine_coordinator.py
engines/
├── nmap_engine/
├── nuclei_engine/
├── ffuf_engine/
└── custom_engine/
```

### 🧪 測試結果

#### ❌ MultiEngineCoordinator
```python
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
coord = MultiEngineCoordinator()
```
**錯誤**:
- `cannot import name 'FunctionTaskSchema' from 'services.aiva_common.schemas.tasks'`
- `AttributeError: 'MultiEngineCoordinator' object has no attribute 'engines'`

**分析**: FunctionTaskSchema 不存在於 tasks.py 中

---

## 3️⃣ Features 功能模組 (services/features)

### 📁 結構
```
function_web_scanner/
function_sqli/
function_xss/
function_ssrf/
function_idor/
function_exploit_framework/
function_postex/
function_crypto/
function_reverse_engineering/
... (共 19 個功能模組)
```

### 🧪 測試結果

#### ❌ WebScanner
```python
from services.features.web.web_scanner import WebScanner
```
**錯誤**: `ModuleNotFoundError: No module named 'services.features.web'`

**分析**: features 模組結構為 `function_*` 而非 `web/`，文件結構與預期不符

---

## 4️⃣ Integration 整合模組 (services/integration)

### 📁 結構
```
aiva_integration/
├── reception/
├── attack_path_analyzer/
├── analysis/
├── remediation/
├── reporting/
├── threat_intel/
└── observability/
```

### 🧪 測試結果

#### ❌ ReceptionCoordinator
```python
from services.integration.aiva_integration.reception.reception_coordinator import ReceptionCoordinator
```
**錯誤**: `ModuleNotFoundError: No module named '...reception_coordinator'`

**分析**: reception 目錄中可能沒有 reception_coordinator.py 文件

---

## 5️⃣ aiva_common 共用模組

### 🧪 測試結果

#### ❌ TaskSchema
```python
from services.aiva_common.schemas.tasks import TaskSchema
```
**錯誤**: `ImportError: cannot import name 'TaskSchema'`

**實際存在的 Schema**:
- ✅ ScanStartPayload
- ✅ ScanStartPayload
- ✅ ScanStopPayload
- ✅ ScanReportPayload
- ⚠️ FunctionTaskSchema (不存在，被 scan coordinator 引用)

---

## 6️⃣ CLI 命令列工具

### 🧪 測試結果

#### ✅ aiva_cli.py (部分可用)
```bash
python aiva_cli.py --help
python aiva_cli.py --stats
```

**可用功能**:
- `--query`: 查詢能力
- `--attack`: AI 執行攻擊
- `--stats`: ✅ 顯示統計 (成功執行)
- `--sync`: 同步能力到 RAG
- `--test`: 運行測試
- `--workflow`: 獲取工作流推薦

**統計輸出**:
```
總計: 782 個能力
模組數: 16
語言數: 4

Top 3 模組:
- scan: 286 (36.6%)
- core/aiva_core: 207 (26.5%)
- integration: 111 (14.2%)
```

---

## 🔍 關鍵問題總結

### ❌ 缺失的依賴項
1. **AIVAExperienceManager** - AICommander 需要
2. **FunctionTaskSchema** - MultiEngineCoordinator 需要
3. **VectorStore** - RAG Engine 需要
4. **MultiLanguageAICoordinator** - AICommander 需要
5. **SeverityLevel** - BioNeuronDecisionController 需要

### ⚠️ 架構不一致
1. **features 模組結構**: 文檔假設 `features.web.web_scanner`，實際為 `features.function_web_scanner.*`
2. **integration 模組**: reception_coordinator.py 文件不存在
3. **Schema 命名**: tasks.py 中沒有通用的 TaskSchema，只有特定的 Payload

### ✅ 實際可用功能
1. **內部探索系統** - 完全可用，可掃描 4 個模組 459 個文件
2. **CLI 統計功能** - 可顯示 782 個能力的分布
3. **AttackExecutor** - 可實例化
4. **對話助理** - 自動初始化 (每次導入都會輸出 log)

---

## 📋 建議修復優先級

### P0 - 立即修復 (阻塞核心功能)
1. ❌ 修復 BioNeuronDecisionController 的 SeverityLevel 導入
2. ❌ 實作缺失的 FunctionTaskSchema
3. ❌ 實作 AIVAExperienceManager

### P1 - 高優先級 (影響多個模組)
4. ⚠️ 修復 VectorStore/RAG Engine 依賴
5. ⚠️ 確認並修正 features 模組結構
6. ⚠️ 檢查 integration/reception 實際文件

### P2 - 中優先級 (降級功能可用)
7. 實作 MultiLanguageAICoordinator
8. 修復 MultiEngineCoordinator.engines 屬性

---

## 結論

**用戶評估準確**: "只有探索我覺得還行，其他都不行"

實際測試證實:
- ✅ **內部探索系統** 完全正常運作
- ⚠️ **大部分組件** 可導入但使用降級/模擬實現
- ❌ **核心 AI 組件** (BioNeuron, 掃描協調器) 無法正常啟動

系統文檔聲稱的功能與實際可用功能存在顯著差距，需要優先修復 P0 級別的依賴項缺失問題。
