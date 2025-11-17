# Integration Module Data Storage

整合模組專用資料儲存目錄

## 📂 目錄結構

```
integration/
├── attack_paths/          # 攻擊路徑圖資料
│   ├── attack_graph.pkl   # NetworkX 圖持久化檔案
│   ├── attack_graph_*.pkl # 歷史備份檔案
│   └── exports/           # 匯出的可視化檔案 (HTML, Mermaid)
│
├── experiences/           # 經驗記錄資料庫
│   ├── experience.db      # SQLite 經驗庫 (主資料庫)
│   ├── experience_*.db    # 備份檔案
│   └── exports/           # 匯出的訓練資料集 (JSONL, CSV)
│
├── training_datasets/     # 訓練資料集
│   ├── dataset_*.jsonl    # 訓練資料集 (JSONL 格式)
│   ├── dataset_*.csv      # 訓練資料集 (CSV 格式)
│   └── metadata/          # 資料集元資料
│
└── models/                # 訓練模型檢查點
    ├── attack_*.pth       # PyTorch 模型檔案
    ├── attack_*.onnx      # ONNX 匯出檔案
    └── checkpoints/       # 訓練檢查點
```

## 🗄️ 資料庫說明

### attack_paths/attack_graph.pkl
- **格式**: NetworkX DiGraph (pickle 序列化)
- **用途**: 儲存資產與漏洞的攻擊路徑圖
- **大小**: ~1-10MB (取決於資產數量)
- **更新頻率**: 每日重建 + 即時增量更新
- **備份策略**: 每日備份,保留 7 天

### experiences/experience.db
- **格式**: SQLite 資料庫
- **用途**: 經驗重放記憶體 (Experience Replay Memory)
- **表結構**:
  - `experience_records`: 攻擊執行經驗
  - `training_datasets`: 訓練資料集定義
  - `dataset_samples`: 資料集樣本關聯
  - `model_training_history`: 模型訓練歷史
- **大小**: ~100MB-1GB (取決於經驗數量)
- **更新頻率**: 每次攻擊執行後即時更新
- **備份策略**: 每日備份,保留 30 天

## 🔧 配置方式

### 環境變數 (.env)

```bash
# 整合模組資料儲存根目錄
AIVA_INTEGRATION_DATA_DIR=C:/D/fold7/AIVA-git/data/integration

# 攻擊路徑圖檔案
AIVA_ATTACK_GRAPH_FILE=${AIVA_INTEGRATION_DATA_DIR}/attack_paths/attack_graph.pkl

# 經驗資料庫
AIVA_EXPERIENCE_DB_URL=sqlite:///${AIVA_INTEGRATION_DATA_DIR}/experiences/experience.db

# 訓練資料集輸出目錄
AIVA_TRAINING_DATASET_DIR=${AIVA_INTEGRATION_DATA_DIR}/training_datasets

# 模型檢查點目錄
AIVA_MODEL_CHECKPOINT_DIR=${AIVA_INTEGRATION_DATA_DIR}/models
```

### Python 配置 (config.py)

```python
from pathlib import Path
import os

# 基礎路徑
INTEGRATION_DATA_DIR = Path(os.getenv(
    "AIVA_INTEGRATION_DATA_DIR",
    "C:/D/fold7/AIVA-git/data/integration"
))

# 攻擊路徑
ATTACK_GRAPH_FILE = Path(os.getenv(
    "AIVA_ATTACK_GRAPH_FILE",
    INTEGRATION_DATA_DIR / "attack_paths" / "attack_graph.pkl"
))

# 經驗資料庫
EXPERIENCE_DB_URL = os.getenv(
    "AIVA_EXPERIENCE_DB_URL",
    f"sqlite:///{INTEGRATION_DATA_DIR}/experiences/experience.db"
)

# 訓練資料集
TRAINING_DATASET_DIR = Path(os.getenv(
    "AIVA_TRAINING_DATASET_DIR",
    INTEGRATION_DATA_DIR / "training_datasets"
))

# 模型檢查點
MODEL_CHECKPOINT_DIR = Path(os.getenv(
    "AIVA_MODEL_CHECKPOINT_DIR",
    INTEGRATION_DATA_DIR / "models"
))
```

## 📊 使用範例

### 1. 攻擊路徑引擎

```python
from services.integration.aiva_integration.attack_path_analyzer import AttackPathEngine

# 使用標準化路徑
engine = AttackPathEngine(
    graph_file="data/integration/attack_paths/attack_graph.pkl"
)

# 自動載入既有圖或建立新圖
paths = engine.find_attack_paths()

# 關閉時自動儲存
engine.close()
```

### 2. 經驗資料庫

```python
from services.integration.aiva_integration.reception import ExperienceRepository

# 使用標準化路徑
repo = ExperienceRepository(
    database_url="sqlite:///data/integration/experiences/experience.db"
)

# 儲存經驗
repo.save_experience(
    plan_id="plan_001",
    attack_type="sqli",
    ast_graph={...},
    execution_trace={...},
    metrics={...},
    feedback={...}
)
```

## 🔄 備份與維護

### 自動備份腳本 (services/integration/scripts/backup.py)

```bash
# 手動備份
python services/integration/scripts/backup.py

# 自動備份 (排程任務)
# Windows: Task Scheduler
# Linux: crontab -e
0 2 * * * cd /path/to/AIVA && python services/integration/scripts/backup.py
```

### 清理舊資料

```bash
# 清理 30 天前的備份
python services/integration/scripts/cleanup.py --days 30
```

## 📝 注意事項

1. **路徑一致性**: 所有腳本應使用統一的環境變數配置
2. **備份策略**: 重要資料 (experience.db) 需定期備份
3. **權限管理**: 確保資料目錄有適當的讀寫權限
4. **磁碟空間**: 監控資料目錄大小,適時清理舊資料
5. **並發安全**: SQLite 僅支援有限並發,生產環境建議使用 PostgreSQL

## 🔗 相關文件

### 核心文檔
- 📖 **[整合模組總覽](../../services/integration/README.md)** - 整合模組主文檔
- 📖 **[Integration Core](../../services/integration/aiva_integration/README.md)** - 核心模組實現
- 📖 **[Services 總覽](../../services/README.md)** - 五大核心服務

### 子模組文檔
- 📖 **[Attack Path Analyzer](../../services/integration/aiva_integration/attack_path_analyzer/README.md)** - 攻擊路徑分析引擎
- 📖 **[Experience Repository](../../services/integration/aiva_integration/reception/experience_repository.py)** - 經驗資料庫

### 維護與開發
- 📖 **[維護腳本文檔](../../services/integration/scripts/README.md)** - 備份與清理工具
- 📖 **[建立報告](../../reports/INTEGRATION_DATA_STORAGE_SETUP_REPORT.md)** - 完整建立過程
- 📖 **[Data Storage Guide](../../guides/development/DATA_STORAGE_GUIDE.md)** - 資料儲存總指南
