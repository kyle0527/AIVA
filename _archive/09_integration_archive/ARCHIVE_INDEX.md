# 整合層封存索引 (Integration Archive Index)

> 封存日期: 2026-02-11
> 封存原因: 清理未使用及已廢棄的模組

---

## 目錄結構

```
09_integration_archive/
├── alembic/                      # PostgreSQL 資料庫遷移 (未使用)
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py
└── deprecated_managers/          # 已廢棄的 Manager 類別
    ├── minimal_manifest.py
    ├── scanner_manager.py
    ├── postex_manager.py
    └── authn_manager.py
```

---

## 封存項目說明

### 1. alembic/ - PostgreSQL 遷移檔案

**封存原因:**
- 專案實際使用 SQLite 作為資料庫後端
- PostgreSQL 遷移檔案從未被執行
- `env.py` 引用不存在的模組路徑
- `001_initial_schema.py` 缺少必要的 import 語句

**原始位置:** `services/integration/alembic/`

**替代方案:**
- 資料庫: SQLite (`data/database/aiva.db`, `experience.db`)
- 儲存管理: `services/core/aiva_core/service_backbone/storage/backends.py`

---

### 2. deprecated_managers/ - 已廢棄的 Manager 類別

#### 2.1 minimal_manifest.py

**封存原因:**
- 2026-01-04 正式標記為棄用
- 已被 `latest_classification.json` (自動產出) 取代
- 手動維護的格式與自動產出不一致

**原始位置:** `services/integration/capability/minimal_manifest.py`

**替代方案:**
- 能力定義: 使用 `aiva_flow_classifier.py` 自動產出
- 數據源: `data/internal_exploration/latest_classification.json`

---

#### 2.2 scanner_manager.py

**封存原因:**
- 不需要額外的同步包裝層
- 直接使用 `WebAttackManager` 即可

**原始位置:** `services/features/function_web_scanner/scanner_manager.py`

**替代方案:**
```python
from services.features.function_web_scanner.integration_tools.web_tools import WebAttackManager

manager = WebAttackManager()
result = await manager.comprehensive_scan(target, options)
```

---

#### 2.3 postex_manager.py

**封存原因:**
- 不需要額外的 Manager 包裝層
- 直接使用 `PostExDetector` 即可

**原始位置:** `services/features/function_postex/postex_manager.py`

**替代方案:**
```python
from services.features.function_postex.detector.postex_detector import PostExDetector

detector = PostExDetector()
findings = detector.analyze(test_type="privilege_escalation", target="localhost")
```

---

#### 2.4 authn_manager.py

**封存原因:**
- 應直接調用 Go 二進制文件
- 不應有 Python 回退機制

**原始位置:** `services/features/function_authn_go/authn_manager.py`

**替代方案:**
```python
import subprocess
import json

result = subprocess.run(['./bin/authn-worker', 'scan', target], capture_output=True)
findings = json.loads(result.stdout)
```

---

## 恢復說明

如需恢復任何封存項目，請：
1. 將檔案從封存目錄複製回原始位置
2. 更新相關的 `__init__.py` 以重新導出
3. 修復任何遺漏的 import 或依賴

**注意:** 封存前已確認這些檔案沒有被其他模組直接引用。
