# AIVA 訓練數據存儲方案

## 📁 數據存儲位置

### 1. 默認數據目錄結構

```
/workspaces/AIVA/data/
├── training/              # 訓練數據
│   ├── experiences/       # 經驗樣本
│   │   ├── experiences.jsonl           # JSONL 格式（快速讀寫）
│   │   ├── high_quality.jsonl          # 高質量樣本
│   │   └── by_type/                    # 按類型分類
│   │       ├── sqli.jsonl
│   │       ├── xss.jsonl
│   │       └── ssrf.jsonl
│   ├── sessions/          # 訓練會話
│   │   ├── training_20250114_120000.json
│   │   └── training_20250114_130000.json
│   ├── traces/            # 執行追蹤
│   │   ├── session_xyz/
│   │   │   ├── trace_001.json
│   │   │   └── trace_002.json
│   │   └── traces.db      # SQLite 追蹤數據庫
│   └── metrics/           # 訓練指標
│       ├── episode_metrics.csv
│       └── model_performance.csv
│
├── models/                # 訓練完成的模型
│   ├── checkpoints/       # 檢查點
│   │   ├── model_epoch_10.pt
│   │   ├── model_epoch_20.pt
│   │   └── best_model.pt
│   ├── production/        # 生產模型
│   │   ├── model_v1.0.pt
│   │   ├── model_v1.1.pt
│   │   └── current.pt -> model_v1.1.pt
│   └── metadata/          # 模型元數據
│       ├── model_v1.0.json
│       └── model_v1.1.json
│
├── knowledge/             # RAG 知識庫
│   ├── vectors/           # 向量數據
│   │   ├── vectors.npy    # NumPy 向量
│   │   ├── data.json      # 文檔和元數據
│   │   └── chroma/        # ChromaDB（可選）
│   ├── entries.json       # 知識條目
│   └── payloads/          # 有效載荷庫
│       ├── sqli_payloads.json
│       ├── xss_payloads.json
│       └── ssrf_payloads.json
│
├── scenarios/             # 靶場場景
│   ├── owasp/
│   │   ├── sqli_basic.json
│   │   ├── xss_stored.json
│   │   └── ssrf_blind.json
│   └── custom/
│       └── my_scenario.json
│
└── logs/                  # 系統日誌
    ├── training_2025-01-14.log
    ├── execution_2025-01-14.log
    └── rag_2025-01-14.log
```

---

## 💾 數據庫方案

### 方案 1: SQLite（推薦用於開發和小規模）

```
/workspaces/AIVA/data/database/
└── aiva.db                # 主數據庫
    ├── experiences        # 經驗樣本表
    ├── traces             # 執行追蹤表
    ├── sessions           # 訓練會話表
    ├── models             # 模型元數據表
    ├── knowledge_entries  # 知識條目表
    └── scenarios          # 場景表
```

### 方案 2: PostgreSQL（推薦用於生產）

```
Docker Compose 配置（已有 docker/docker-compose.yml）
服務: postgres
端口: 5432
數據卷: ./data/postgres
```

### 方案 3: MongoDB（可選，用於非結構化數據）

```
Docker Compose 配置
服務: mongodb
端口: 27017
數據卷: ./data/mongodb
```

---

## 🗄️ 數據類型和大小

### 訓練數據估算

```
單個經驗樣本:        ~5-10 KB (JSON)
1000 個樣本:          ~5-10 MB
10000 個樣本:         ~50-100 MB
100000 個樣本:        ~500 MB - 1 GB

向量嵌入:
- 每個文檔 384 維:   ~1.5 KB (float32)
- 10000 個文檔:      ~15 MB

訓練模型:
- BioNeuron 500萬參數: ~20-50 MB
- 檢查點 (含優化器):  ~100-200 MB
```

---

## 📝 配置示例

### 1. 環境變量配置

```bash
# .env 文件
AIVA_DATA_DIR=/workspaces/AIVA/data
AIVA_DB_TYPE=sqlite  # sqlite / postgres / mongodb
AIVA_DB_URL=sqlite:///./data/database/aiva.db

# PostgreSQL (生產環境)
# AIVA_DB_URL=postgresql://user:pass@localhost:5432/aiva

# 存儲配置
AIVA_EXPERIENCE_STORAGE=database  # database / jsonl / both
AIVA_MODEL_DIR=/workspaces/AIVA/data/models
AIVA_KNOWLEDGE_DIR=/workspaces/AIVA/data/knowledge
```

### 2. Python 配置

```python
from pathlib import Path

# 數據目錄配置
DATA_ROOT = Path("/workspaces/AIVA/data")

STORAGE_CONFIG = {
    # 訓練數據
    "training": {
        "experiences_dir": DATA_ROOT / "training/experiences",
        "sessions_dir": DATA_ROOT / "training/sessions",
        "traces_dir": DATA_ROOT / "training/traces",
        "metrics_dir": DATA_ROOT / "training/metrics",
    },

    # 模型
    "models": {
        "checkpoints_dir": DATA_ROOT / "models/checkpoints",
        "production_dir": DATA_ROOT / "models/production",
        "metadata_dir": DATA_ROOT / "models/metadata",
    },

    # RAG 知識庫
    "knowledge": {
        "vectors_dir": DATA_ROOT / "knowledge/vectors",
        "entries_file": DATA_ROOT / "knowledge/entries.json",
        "payloads_dir": DATA_ROOT / "knowledge/payloads",
    },

    # 場景
    "scenarios": {
        "owasp_dir": DATA_ROOT / "scenarios/owasp",
        "custom_dir": DATA_ROOT / "scenarios/custom",
    },

    # 數據庫
    "database": {
        "sqlite_path": DATA_ROOT / "database/aiva.db",
        "postgres_url": "postgresql://aiva:aiva@localhost:5432/aiva",
    },
}
```

---

## 🔄 數據流程

### 訓練時

```
1. 執行攻擊計畫
   ↓
2. TraceLogger → data/training/traces/ (即時)
   ↓
3. PlanComparator → 生成 ExperienceSample
   ↓
4. ExperienceManager → data/training/experiences/ (JSONL)
                    → database/experiences (可選)
   ↓
5. RAGEngine → data/knowledge/ (知識更新)
   ↓
6. ModelTrainer → data/models/checkpoints/ (訓練中)
   ↓
7. 訓練完成 → data/models/production/ (部署)
```

### 查詢時

```
1. 用戶請求
   ↓
2. RAGEngine 查詢 → data/knowledge/vectors/
   ↓
3. 檢索相關經驗 → database/experiences
   ↓
4. 加載模型 → data/models/production/current.pt
   ↓
5. 生成決策
```

---

## 🛡️ 數據備份策略

### 自動備份

```bash
# 每日備份腳本
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR=/workspaces/AIVA/backups/$DATE

# 備份數據庫
sqlite3 /workspaces/AIVA/data/database/aiva.db ".backup $BACKUP_DIR/aiva.db"

# 備份訓練數據
tar -czf $BACKUP_DIR/training_data.tar.gz /workspaces/AIVA/data/training/

# 備份模型
tar -czf $BACKUP_DIR/models.tar.gz /workspaces/AIVA/data/models/production/

# 備份知識庫
tar -czf $BACKUP_DIR/knowledge.tar.gz /workspaces/AIVA/data/knowledge/

# 刪除 30 天前的備份
find /workspaces/AIVA/backups/ -type d -mtime +30 -exec rm -rf {} \;
```

### 雲端同步（可選）

- Google Drive
- AWS S3
- Azure Blob Storage
- 自建 Git LFS

---

## 📊 數據清理策略

### 經驗樣本

- **保留**: 高質量樣本（quality_score > 0.7）永久保留
- **歸檔**: 中等質量樣本（0.5-0.7）保留 6 個月
- **刪除**: 低質量樣本（< 0.5）保留 1 個月

### 訓練會話

- **保留**: 成功會話保留 3 個月
- **刪除**: 失敗會話保留 1 個月

### 模型檢查點

- **保留**: 最佳模型永久保留
- **保留**: 最近 10 個檢查點
- **刪除**: 其他舊檢查點

---

## 🚀 使用示例

### 初始化存儲

```python
from aiva_core.storage import StorageManager

# 初始化存儲管理器
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="sqlite",
    auto_create_dirs=True
)

# 創建所有必要目錄
storage.initialize()
```

### 保存訓練數據

```python
# 保存經驗樣本
experience_manager = ExperienceManager(storage_backend=storage)
await experience_manager.add_sample(sample)

# 導出為 JSONL
experience_manager.export_to_jsonl(
    "/workspaces/AIVA/data/training/experiences/batch_001.jsonl"
)
```

### 保存模型

```python
# 保存檢查點
model_trainer = ModelTrainer()
model_trainer.save_checkpoint(
    model=bio_neuron_agent.decision_core,
    path="/workspaces/AIVA/data/models/checkpoints/model_epoch_50.pt",
    metadata={
        "epoch": 50,
        "loss": 0.123,
        "accuracy": 0.89
    }
)

# 部署到生產
model_trainer.deploy_model(
    checkpoint_path="checkpoints/model_epoch_50.pt",
    production_path="production/model_v2.0.pt"
)
```

---

## 📈 監控和統計

### 存儲使用情況

```python
storage.get_statistics()
# {
#   "total_experiences": 15234,
#   "total_models": 25,
#   "disk_usage": "2.3 GB",
#   "database_size": "450 MB",
#   "knowledge_entries": 3421
# }
```

### 數據質量報告

```python
experience_manager.get_quality_report()
# {
#   "high_quality": 4521,  # > 0.7
#   "medium_quality": 8234,  # 0.5-0.7
#   "low_quality": 2479,  # < 0.5
#   "avg_quality": 0.63
# }
```

---

## ✅ 推薦配置

### 開發環境

- **數據庫**: SQLite
- **經驗存儲**: JSONL + Database
- **模型存儲**: 本地文件系統
- **備份**: Git + 手動備份

### 生產環境

- **數據庫**: PostgreSQL
- **經驗存儲**: Database
- **模型存儲**: 文件系統 + S3
- **備份**: 自動每日備份 + 雲端同步

### 數據目錄總大小預估

- 初始: < 100 MB
- 訓練 1 個月: 1-5 GB
- 訓練 1 年: 10-50 GB
