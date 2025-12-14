# Internal Exploration 輸出路徑遷移計劃

## 📋 執行摘要

**目標**: 將 internal_exploration 所有輸出數據路徑統一遷移到 `services/integration` 整合模組中

**狀態**: ✅ 分析完成，準備執行  
**優先級**: HIGH  
**預估工時**: 2-3 小時  
**影響範圍**: 5 個 Python 文件 + 配置更新

---

## 🎯 遷移目標

### 當前輸出路徑結構

```
services/core/aiva_core/
├── analysis_results/              # ⚠️ 需遷移 (360+ 個文件)
│   ├── flow_*.md                  # 360 個流程圖
│   ├── analysis_results.json      # 分析結果
│   ├── data_flow_summary.md       # 摘要報告
│   └── ...
│
└── internal_exploration/
    ├── analysis_history/          # ⚠️ 需遷移 (版本化數據)
    │   ├── v1/
    │   ├── v2/
    │   └── latest_classification.json
    │
    └── self_healing/
        └── [輸出到 target/analysis_results/]  # ⚠️ 動態路徑，需重定向
```

### 目標路徑結構

```
services/integration/
└── data/
    └── internal_exploration/      # 🆕 新建目錄
        ├── analysis_results/      # 遷移：當前分析結果
        │   ├── flow_*.md
        │   ├── analysis_results.json
        │   └── data_flow_summary.md
        │
        ├── analysis_history/      # 遷移：版本化歷史
        │   ├── v1/
        │   ├── v2/
        │   └── latest -> v2/      # 符號連結
        │
        └── self_healing/          # 新建：自我診斷報告
            ├── core_analysis_full.json
            ├── core_analysis_quick.md
            └── dataflow_breakpoint_analysis.md
```

---

## 📊 受影響文件分析

### 需要修改的 Python 文件 (5 個)

| 文件 | 代碼行數 | 修改點 | 優先級 |
|------|---------|-------|--------|
| `aiva_flow_analyzer.py` | 1,439 | 1. `save_results()` 默認路徑 | HIGH |
| `aiva_exploration_pipeline.py` | 426 | 2. `HISTORY_DIR` 常量定義 | HIGH |
| `core_analyzer.py` | 514 | 3. `__init__()` output_dir | HIGH |
| `run_analysis.py` | 350+ | 4. `_get_output_dir()` 邏輯 | MEDIUM |
| `aiva_flow_classifier.py` | 700+ | 5. 輸出路徑配置 | LOW |

### 涉及的路徑變量

```python
# 1. aiva_flow_analyzer.py (Line 1194)
def save_results(self, output_dir: str = "aiva_flow_analysis") -> None:
    # ❌ 舊: 相對路徑，輸出到當前目錄
    # ✅ 新: 統一路徑到 integration

# 2. aiva_exploration_pipeline.py (Line 69)
HISTORY_DIR = CURRENT_DIR / "analysis_history"
# ❌ 舊: 輸出到 internal_exploration 下
# ✅ 新: 輸出到 integration/data/internal_exploration/

# 3. core_analyzer.py (Constructor)
def __init__(self, source_path: str, output_dir: Optional[str] = None):
    # ❌ 舊: output_dir 默認為 source_path/analysis_results
    # ✅ 新: 統一到 integration 目錄

# 4. run_analysis.py (Line 64-69)
def _get_output_dir(self, target: Path) -> Path:
    default_dir = target / "analysis_results"
    # ❌ 舊: 輸出到分析目標目錄下
    # ✅ 新: 統一輸出到 integration
```

---

## 🛠️ 實施方案

### 階段 1: 創建整合目錄結構 (5 分鐘)

```bash
# 在 services/integration 下創建新目錄
cd services/integration
mkdir -p data/internal_exploration/analysis_results
mkdir -p data/internal_exploration/analysis_history
mkdir -p data/internal_exploration/self_healing
```

### 階段 2: 添加配置文件 (10 分鐘)

創建統一的配置文件，集中管理所有輸出路徑：

**文件**: `services/aiva_common/config/paths.py`

```python
"""統一路徑配置 - AIVA 系統輸出路徑管理"""
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Integration 模組根目錄
INTEGRATION_ROOT = PROJECT_ROOT / "services" / "integration"

# Internal Exploration 輸出根目錄
INTERNAL_EXPLORATION_DATA = INTEGRATION_ROOT / "data" / "internal_exploration"

# 細分路徑
ANALYSIS_RESULTS_DIR = INTERNAL_EXPLORATION_DATA / "analysis_results"
ANALYSIS_HISTORY_DIR = INTERNAL_EXPLORATION_DATA / "analysis_history"
SELF_HEALING_DIR = INTERNAL_EXPLORATION_DATA / "self_healing"

# 確保目錄存在
def ensure_directories():
    """確保所有輸出目錄存在"""
    for directory in [
        ANALYSIS_RESULTS_DIR,
        ANALYSIS_HISTORY_DIR,
        SELF_HEALING_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
```

### 階段 3: 修改 Python 文件 (60-90 分鐘)

#### 3.1 修改 `aiva_flow_analyzer.py`

```python
# 在文件頂部添加導入
from aiva_common.config.paths import ANALYSIS_RESULTS_DIR, ensure_directories

class AIVAFlowAnalyzer:
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """保存分析結果
        
        Args:
            output_dir: 自定義輸出目錄，默認使用統一路徑
        """
        if output_dir is None:
            ensure_directories()
            output_path = ANALYSIS_RESULTS_DIR
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        # ... 其餘代碼不變
```

#### 3.2 修改 `aiva_exploration_pipeline.py`

```python
# 在文件頂部添加導入
from aiva_common.config.paths import ANALYSIS_HISTORY_DIR, ensure_directories

# 修改常量定義 (Line 69)
# ❌ 舊代碼
# HISTORY_DIR = CURRENT_DIR / "analysis_history"

# ✅ 新代碼
ensure_directories()
HISTORY_DIR = ANALYSIS_HISTORY_DIR
```

#### 3.3 修改 `core_analyzer.py`

```python
# 在文件頂部添加導入
from aiva_common.config.paths import SELF_HEALING_DIR, ensure_directories

class CoreAnalyzer:
    def __init__(self, source_path: str, output_dir: Optional[str] = None):
        """初始化核心分析器
        
        Args:
            source_path: 要分析的源代碼路徑
            output_dir: 輸出目錄，默認使用統一路徑
        """
        self.source_path = Path(source_path)
        
        if output_dir is None:
            ensure_directories()
            self.output_dir = SELF_HEALING_DIR
        else:
            self.output_dir = Path(output_dir)
        
        # ... 其餘代碼不變
```

#### 3.4 修改 `run_analysis.py`

```python
# 在文件頂部添加導入
from aiva_common.config.paths import SELF_HEALING_DIR, ensure_directories

class AnalysisRunner:
    def _get_output_dir(self, target: Path) -> Path:
        """獲取輸出目錄，默認使用統一路徑"""
        if self.output_dir:
            return self.output_dir
        
        # ❌ 舊代碼
        # default_dir = target / "analysis_results"
        
        # ✅ 新代碼：統一輸出到 integration
        ensure_directories()
        return SELF_HEALING_DIR
```

#### 3.5 修改 `aiva_flow_classifier.py`

```python
# 在文件頂部添加導入
from aiva_common.config.paths import ANALYSIS_HISTORY_DIR

# 修改輸出路徑配置 (如果有)
# 根據實際代碼調整
```

### 階段 4: 遷移現有數據 (30 分鐘)

```bash
# 備份現有數據
cd services/core/aiva_core
tar -czf analysis_backup_$(date +%Y%m%d).tar.gz analysis_results/ internal_exploration/analysis_history/

# 遷移數據到新位置
cp -r analysis_results/* ../../integration/data/internal_exploration/analysis_results/
cp -r internal_exploration/analysis_history/* ../../integration/data/internal_exploration/analysis_history/

# 驗證數據完整性
cd ../../integration/data/internal_exploration
find analysis_results -type f | wc -l  # 應該有 360+ 個文件
find analysis_history -type d | wc -l  # 應該有版本目錄
```

### 階段 5: 更新文檔和配置 (15 分鐘)

#### 5.1 更新 `internal_exploration/README.md`

添加新的輸出路徑說明：

```markdown
## 📂 輸出數據位置

所有分析結果統一輸出到 `services/integration/data/internal_exploration/`:

- `analysis_results/` - 當前分析結果（流程圖、JSON報告）
- `analysis_history/` - 版本化歷史記錄
- `self_healing/` - 自我診斷報告

### 訪問輸出數據

\```python
from aiva_common.config.paths import (
    ANALYSIS_RESULTS_DIR,
    ANALYSIS_HISTORY_DIR,
    SELF_HEALING_DIR
)

# 讀取最新分析結果
results_file = ANALYSIS_RESULTS_DIR / "analysis_results.json"
\```
```

#### 5.2 更新 `integration/README.md`

添加 internal_exploration 數據說明：

```markdown
## 💾 資料儲存結構

```
data/integration/
├── internal_exploration/      # 🆕 Internal Exploration 分析數據
│   ├── analysis_results/      # 當前分析結果
│   ├── analysis_history/      # 版本化歷史
│   └── self_healing/          # 自我診斷報告
├── attack_paths/              # 攻擊路徑圖
└── experiences/               # 經驗記錄
```
```

### 階段 6: 測試驗證 (20 分鐘)

```bash
# 1. 測試 Pipeline
cd services/core/aiva_core/internal_exploration
python aiva_exploration_pipeline.py --target core

# 驗證輸出到新路徑
ls ../../../integration/data/internal_exploration/analysis_history/

# 2. 測試 Self-Healing
cd self_healing
python run_analysis.py --target ../cognitive_core --mode quick

# 驗證輸出到新路徑
ls ../../../../integration/data/internal_exploration/self_healing/

# 3. 測試 Flow Analyzer
cd ../python_tools
python aiva_flow_analyzer.py --target ../cognitive_core

# 驗證輸出到新路徑
ls ../../../../integration/data/internal_exploration/analysis_results/
```

---

## 🔍 兼容性考慮

### 向後兼容性

為保持向後兼容，提供配置選項：

```python
# services/aiva_common/config/paths.py

# 環境變量控制是否使用新路徑
USE_INTEGRATED_PATHS = os.getenv("AIVA_USE_INTEGRATED_PATHS", "true").lower() == "true"

if USE_INTEGRATED_PATHS:
    ANALYSIS_RESULTS_DIR = INTEGRATION_ROOT / "data" / "internal_exploration" / "analysis_results"
else:
    # 使用舊路徑（向後兼容）
    ANALYSIS_RESULTS_DIR = PROJECT_ROOT / "services" / "core" / "aiva_core" / "analysis_results"
```

### 遷移期間處理

在遷移期間，同時檢查兩個位置：

```python
def get_analysis_results_path() -> Path:
    """獲取分析結果路徑，優先使用新路徑"""
    new_path = INTEGRATION_ROOT / "data" / "internal_exploration" / "analysis_results"
    old_path = PROJECT_ROOT / "services" / "core" / "aiva_core" / "analysis_results"
    
    if new_path.exists() and (new_path / "analysis_results.json").exists():
        return new_path
    elif old_path.exists():
        return old_path
    else:
        # 默認創建新路徑
        new_path.mkdir(parents=True, exist_ok=True)
        return new_path
```

---

## 📋 檢查清單

### 執行前

- [ ] 備份現有數據
- [ ] 創建新目錄結構
- [ ] 創建 `paths.py` 配置文件
- [ ] 確認團隊成員知悉變更

### 執行中

- [ ] 修改 `aiva_flow_analyzer.py`
- [ ] 修改 `aiva_exploration_pipeline.py`
- [ ] 修改 `core_analyzer.py`
- [ ] 修改 `run_analysis.py`
- [ ] 修改 `aiva_flow_classifier.py`
- [ ] 遷移現有數據

### 執行後

- [ ] 測試 Pipeline 運行
- [ ] 測試 Self-Healing 運行
- [ ] 測試 Flow Analyzer 運行
- [ ] 驗證數據完整性
- [ ] 更新文檔
- [ ] 提交 Git commit
- [ ] 通知團隊

---

## 🎯 預期收益

### 1. 統一管理
- ✅ 所有輸出數據集中在 integration 模組
- ✅ 便於備份和遷移

### 2. 清晰結構
- ✅ 數據分類清晰（analysis_results, history, self_healing）
- ✅ 符合整合模組的設計理念

### 3. 易於整合
- ✅ Integration 模組可直接訪問分析數據
- ✅ 支援未來的數據分析和可視化功能

### 4. 避免污染
- ✅ core 模組保持清潔，不混入輸出數據
- ✅ 符合源代碼和數據分離原則

---

## ⚠️ 風險評估

| 風險 | 級別 | 緩解措施 |
|------|------|----------|
| 數據丟失 | HIGH | 執行前完整備份 |
| 路徑錯誤 | MEDIUM | 充分測試，保留舊路徑檢查 |
| 兼容性問題 | MEDIUM | 提供環境變量開關 |
| 性能影響 | LOW | 路徑變更不影響性能 |

---

## 📝 Rollback 計劃

如果遷移出現問題，回滾步驟：

```bash
# 1. 恢復備份
cd services/core/aiva_core
tar -xzf analysis_backup_YYYYMMDD.tar.gz

# 2. 設置環境變量使用舊路徑
export AIVA_USE_INTEGRATED_PATHS=false

# 3. 重新運行測試
python -m pytest tests/
```

---

## 📅 實施時間表

| 階段 | 任務 | 預估時間 | 負責人 |
|------|------|---------|--------|
| Day 1 | 階段 1-2: 創建結構和配置 | 15 分鐘 | Dev |
| Day 1 | 階段 3: 修改代碼 | 90 分鐘 | Dev |
| Day 1 | 階段 4: 遷移數據 | 30 分鐘 | Dev |
| Day 1 | 階段 5: 更新文檔 | 15 分鐘 | Dev |
| Day 2 | 階段 6: 測試驗證 | 20 分鐘 | QA |
| Day 2 | Code Review | 30 分鐘 | Team |
| Day 2 | 部署上線 | 10 分鐘 | DevOps |

**總計**: 約 3.5 小時（分兩天完成）

---

## ✅ 成功標準

1. 所有測試通過，無錯誤輸出
2. 數據完整性驗證通過（文件數量一致）
3. 文檔更新完成
4. 團隊成員確認理解變更
5. 舊數據備份安全保存

---

**準備好開始遷移了嗎？**

如果確認無誤，可以執行：
```bash
# 一鍵執行遷移腳本（待創建）
python services/core/aiva_core/internal_exploration/migrate_output_paths.py
```
