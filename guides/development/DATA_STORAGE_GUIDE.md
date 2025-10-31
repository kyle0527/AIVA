# AIVA 數據存儲使用指南

## � 目錄

- [📌 快速開始](#-快速開始)
- [🏗️ 架構設計](#-架構設計)
- [📊 存儲類型](#-存儲類型)
- [🔧 API 使用](#-api-使用)
- [📈 性能優化](#-性能優化)
- [🔒 安全設定](#-安全設定)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

## �📌 快速開始

### 1. 初始化存儲

```bash
# 創建所有必要的目錄和數據庫
python init_storage.py
```

### 2. 在代碼中使用存儲

```python
from aiva_core.storage import StorageManager

# 創建存儲管理器
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="hybrid",  # SQLite + JSONL
    auto_create_dirs=True
)

# 在組件中使用
from aiva_core.learning import ExperienceManager

experience_mgr = ExperienceManager(storage_backend=storage)
```

---

## 🗄️ 存儲後端選擇

### SQLite（默認，推薦開發環境）

```python
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="sqlite"
)
```

**優點**:

- 無需額外配置
- 快速、輕量
- 單文件，易於備份

**適用場景**:

- 開發和測試
- 單機部署
- 小規模訓練（< 100K 樣本）

### PostgreSQL（推薦生產環境）

```python
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="postgres",
    db_config={
        "host": "localhost",
        "port": 5432,
        "database": "aiva",
        "user": "aiva",
        "password": "your-password"
    }
)
```

**優點**:

- 高性能
- 支持並發
- ACID 保證
- 高級查詢功能

**適用場景**:

- 生產部署
- 大規模訓練（> 100K 樣本）
- 多用戶/多進程

### JSONL（文件格式）

```python
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="jsonl"
)
```

**優點**:

- 人類可讀
- 易於導出/分析
- 兼容性好

**適用場景**:

- 數據導出
- 離線分析
- 與其他工具集成

### Hybrid（推薦！）

```python
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="hybrid"  # SQLite + JSONL
)
```

**優點**:

- 結合數據庫和文件優點
- 數據庫用於快速查詢
- JSONL 用於備份和分析

**適用場景**:

- 大多數場景
- 需要高性能 + 易用性

---

## 💾 數據持久化示例

### 1. 保存經驗樣本

```python
from aiva_common.schemas import ExperienceSample, AttackPlan, AttackResult
from datetime import datetime

# 創建經驗樣本
sample = ExperienceSample(
    sample_id="sample-001",
    timestamp=datetime.utcnow(),
    plan=AttackPlan(...),
    trace_id="trace-001",
    actual_result=AttackResult(...),
    quality_score=0.85,
    metadata={"session": "training-001"}
)

# 保存
await storage.save_experience_sample(sample)
```

### 2. 查詢經驗樣本

```python
# 查詢所有高質量樣本
high_quality = await storage.get_experience_samples(
    limit=100,
    min_quality=0.7
)

# 查詢特定類型
sqli_samples = await storage.get_experience_samples(
    vulnerability_type="sqli",
    min_quality=0.5,
    limit=50
)
```

### 3. 保存追蹤記錄

```python
from aiva_common.schemas import TraceRecord

trace = TraceRecord(
    trace_id="trace-001",
    session_id="session-001",
    timestamp=datetime.utcnow(),
    plan=plan,
    steps=[...],
    total_steps=5,
    successful_steps=4,
    failed_steps=1
)

await storage.save_trace(trace)
```

### 4. 保存訓練會話

```python
session_data = {
    "session_id": "training-001",
    "created_at": datetime.utcnow(),
    "session_type": "batch",
    "scenario_id": "SQLI-1",
    "config": {
        "episodes": 100,
        "learning_rate": 0.001
    },
    "total_episodes": 100,
    "successful_episodes": 85,
    "avg_reward": 15.2,
    "status": "completed"
}

await storage.save_training_session(session_data)
```

---

## 🔍 查詢和統計

### 獲取統計信息

```python
stats = await storage.get_statistics()

print(f"總經驗樣本: {stats['total_experiences']}")
print(f"高質量樣本: {stats['high_quality_experiences']}")
print(f"追蹤記錄: {stats['total_traces']}")
print(f"訓練會話: {stats['total_sessions']}")

# 按類型統計
for vuln_type, count in stats['experiences_by_type'].items():
    print(f"{vuln_type}: {count}")
```

### 查詢會話追蹤

```python
# 獲取某個會話的所有追蹤
traces = await storage.get_traces_by_session("session-001")

for trace in traces:
    print(f"Trace {trace.trace_id}: {trace.total_steps} steps")
```

---

## 🛠️ 在 AI 組件中集成

### ExperienceManager

```python
from aiva_core.storage import StorageManager
from aiva_core.learning import ExperienceManager

# 初始化存儲
storage = StorageManager(
    data_root="/workspaces/AIVA/data",
    db_type="hybrid"
)

# 創建經驗管理器
exp_manager = ExperienceManager(storage_backend=storage)

# 執行攻擊並保存經驗
result = await plan_executor.execute(plan)
sample = await exp_manager.create_experience_sample(result)
# 樣本自動保存到存儲後端
```

### TraceLogger

```python
from aiva_core.execution import TraceLogger

# 創建追蹤記錄器
trace_logger = TraceLogger(storage_backend=storage)

# 開始追蹤
await trace_logger.start_trace(
    plan=plan,
    session_id="session-001"
)

# 記錄步驟
await trace_logger.log_step(...)

# 結束追蹤（自動保存）
trace = await trace_logger.end_trace()
```

### BioNeuronMasterController

```python
from aiva_core.bio_neuron_master import BioNeuronMasterController
from aiva_core.rag import RAGEngine

# 初始化 RAG（使用存儲路徑）
rag_engine = RAGEngine(
    vector_store_type="memory",
    data_directory=storage.get_path("knowledge", "vectors")
)

# 創建主控制器
master = BioNeuronMasterController(
    storage_backend=storage,  # 傳入存儲後端
    rag_engine=rag_engine
)

# 處理請求（自動保存訓練數據）
response = await master.process_request(
    user_input="掃描 SQL 注入",
    mode="ai"
)
```

---

## 📂 數據目錄結構

初始化後，數據目錄結構如下：

```
/workspaces/AIVA/data/
├── database/
│   └── aiva.db                      # SQLite 數據庫
│
├── training/
│   ├── experiences/
│   │   ├── experiences.jsonl        # 所有經驗樣本
│   │   ├── high_quality.jsonl       # 高質量樣本
│   │   └── by_type/                 # 按類型分類（可選）
│   ├── sessions/
│   │   └── training_*.json          # 訓練會話數據
│   ├── traces/
│   │   └── session_*/               # 追蹤記錄
│   └── metrics/
│       └── *.csv                    # 訓練指標
│
├── models/
│   ├── checkpoints/
│   │   ├── model_epoch_10.pt
│   │   └── model_epoch_20.pt
│   ├── production/
│   │   ├── current.pt               # 當前生產模型
│   │   └── model_v1.0.pt
│   └── metadata/
│       └── model_*.json             # 模型元數據
│
├── knowledge/
│   ├── vectors/
│   │   ├── vectors.npy              # NumPy 向量
│   │   └── data.json                # 文檔數據
│   ├── entries.json                 # 知識條目
│   └── payloads/
│       ├── sqli_payloads.json
│       └── xss_payloads.json
│
├── scenarios/
│   ├── owasp/
│   │   └── *.json                   # OWASP 場景
│   └── custom/
│       └── *.json                   # 自定義場景
│
└── logs/
    └── *.log                        # 系統日誌
```

---

## 🔧 配置選項

### 環境變量

```bash
# 數據目錄
export AIVA_DATA_DIR=/workspaces/AIVA/data

# 數據庫類型
export AIVA_DB_TYPE=hybrid  # sqlite / postgres / jsonl / hybrid

# PostgreSQL（生產環境）
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=aiva
export POSTGRES_USER=aiva
export POSTGRES_PASSWORD=your-password

# 存儲策略
export AIVA_EXPERIENCE_STORAGE=both  # database / jsonl / both

# 自動備份
export AIVA_AUTO_BACKUP=true
export AIVA_BACKUP_INTERVAL=24  # 小時
export AIVA_BACKUP_DIR=/workspaces/AIVA/backups
export AIVA_BACKUP_KEEP_DAYS=30
```

### Python 配置

```python
from aiva_core.storage.config import get_storage_config

# 獲取默認配置
config = get_storage_config()

# 創建存儲
storage = StorageManager(**config)
```

---

## 🧹 數據管理

### 清理舊數據

```python
# 根據質量閾值清理
from aiva_core.storage.config import QUALITY_THRESHOLDS, DATA_RETENTION

# 刪除低質量樣本（自動根據保留策略）
# 在數據庫層面實現定期清理
```

### 導出數據

```python
# 導出高質量樣本
samples = await storage.get_experience_samples(
    min_quality=0.7,
    limit=10000
)

# 寫入 JSONL
import json
with open("export.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample.model_dump()) + "\n")
```

### 備份

```bash
# 手動備份數據庫
cp /workspaces/AIVA/data/database/aiva.db /workspaces/AIVA/backups/aiva_$(date +%Y%m%d).db

# 備份訓練數據
tar -czf backups/training_$(date +%Y%m%d).tar.gz data/training/

# 備份模型
tar -czf backups/models_$(date +%Y%m%d).tar.gz data/models/production/
```

---

## 🚨 故障排除

### 數據庫鎖定

```python
# SQLite 在高並發時可能鎖定
# 解決方案：使用 PostgreSQL 或增加超時
storage = StorageManager(
    db_type="sqlite",
    db_config={"timeout": 30}  # 增加超時
)
```

### 磁盤空間不足

```bash
# 檢查磁盤使用
du -h /workspaces/AIVA/data/

# 清理舊數據
find data/training/sessions -type f -mtime +90 -delete
```

### 數據恢復

```python
# 從 JSONL 恢復到數據庫
import json
from aiva_common.schemas import ExperienceSample

with open("data/training/experiences/experiences.jsonl") as f:
    for line in f:
        data = json.loads(line)
        sample = ExperienceSample(**data)
        await storage.save_experience_sample(sample)
```

---

## 📊 性能優化

### 批量操作

```python
# 批量保存
samples = [...]  # 大量樣本
for sample in samples:
    await storage.save_experience_sample(sample)

# 使用事務（SQLite/PostgreSQL）
# 在後端實現批量插入優化
```

### 索引優化

數據庫模型已經包含必要的索引：

- `quality_score` + `vulnerability_type`
- `created_at` + `success`
- `session_id` + `created_at`

### 查詢優化

```python
# 使用限制避免加載全部數據
samples = await storage.get_experience_samples(limit=100)

# 使用過濾減少結果集
samples = await storage.get_experience_samples(
    min_quality=0.7,
    vulnerability_type="sqli",
    limit=50
)
```

---

## ✅ 最佳實踐

1. **使用 Hybrid 模式**: 結合數據庫性能和文件易用性
2. **定期備份**: 設置自動備份避免數據丟失
3. **質量過濾**: 只保存高質量樣本用於訓練
4. **定期清理**: 根據保留策略刪除舊數據
5. **監控存儲**: 定期檢查磁盤使用和數據庫大小
6. **事務處理**: 確保數據一致性
7. **索引優化**: 為常用查詢創建索引
8. **分批處理**: 大量數據使用批量操作

---

## 📚 相關文檔

- [數據存儲方案](./DATA_STORAGE_PLAN.md)
- [AI 系統架構](./AI_SYSTEM_OVERVIEW.md)
- [訓練工作流程](./AI_ARCHITECTURE.md)
