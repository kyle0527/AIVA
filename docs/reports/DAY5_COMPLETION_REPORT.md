# Day 5 完成報告：能力元數據回填 (ChromaDB → PostgreSQL)

**完成日期**: 2025-11-29  
**遵循標準**: aiva_common v2.0 README 規範  
**修改原則**: 修正現有檔案為原則，確認無相關檔案後創建新檔案

---

## 🎯 目標達成

✅ **目標 1**: 從 ChromaDB 讀取所有現有能力文檔（782個）  
✅ **目標 2**: 將 ChromaDB 文檔轉換為 `ModuleCapability` Pydantic 模型  
✅ **目標 3**: 批量寫入 PostgreSQL 資料庫  
✅ **目標 4**: 驗證資料完整性

---

## 📊 回填統計

### 數據來源
- **ChromaDB Collection**: `aiva_capabilities`
- **文檔總數**: 782
- **轉換成功率**: 100% (782/782)

### 回填結果
```
PostgreSQL 寫入統計:
  ✅ 新增 (added): 782
  ⚪ 更新 (updated): 0
  ⚪ 刪除 (deleted): 0
  ⚪ 未變更 (unchanged): 0
  
資料庫統計:
  📦 總能力數: 782
  🏗️  模組數: 16
  💬 語言數: 4
```

---

## 📝 創建的檔案

### 1. **scripts/migration/backfill_capabilities.py** (新建)

**目的**: 能力元數據回填腳本（ChromaDB → PostgreSQL）

**核心功能**:
1. **從 ChromaDB 讀取所有能力文檔**
   ```python
   def get_all_capabilities_from_chromadb() -> list[dict]:
       """從 ChromaDB 讀取所有能力文檔"""
       import chromadb
       client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
       collections = client.list_collections()
       
       # 遍歷所有 collections
       for collection_info in collections:
           collection = client.get_collection(name=collection_info.name)
           result = collection.get(include=["documents", "metadatas"])
           # 轉換為統一格式
   ```

2. **轉換為 ModuleCapability**
   ```python
   def convert_chroma_doc_to_capability(doc: dict) -> ModuleCapability | None:
       """將 ChromaDB 文檔轉換為 ModuleCapability"""
       metadata = doc.get("metadata", {})
       
       # 提取能力信息
       capability_id = metadata.get("capability_id", doc["id"])
       module_name = metadata.get("module", "unknown")
       capability_name = metadata.get("name", "unknown")
       
       # 轉換枚舉值
       category = CapabilityCategory(category_str.lower()) or CapabilityCategory.UTILITY
       complexity = CapabilityComplexity.MODERATE
       
       # 構建 ModuleCapability
       return ModuleCapability(...)
   ```

3. **批量寫入 PostgreSQL**
   ```python
   def backfill_to_postgresql(capabilities: list[ModuleCapability]) -> dict:
       """批量寫入能力到 PostgreSQL"""
       registry = CapabilityRegistry(
           pg_session=session,
           chroma_collection=None
       )
       result = registry.register_capabilities(capabilities)
       return stats
   ```

4. **驗證回填結果**
   ```python
   def verify_backfill(expected_count: int) -> bool:
       """驗證回填結果"""
       actual_count = session.query(CapabilityRecord).count()
       return actual_count >= expected_count
   ```

**遵循規範**:
- ✅ 使用 `get_logger(__name__)` 統一日誌
- ✅ 使用 Pydantic 模型進行數據驗證
- ✅ 使用 `create_error_context` 統一錯誤處理
- ✅ 完整的類型註解
- ✅ 詳細的 docstring

---

## 🔧 技術細節

### 數據轉換流程

```
ChromaDB Document
    ↓
{
  id: "kb_xxxx",
  content: "能力描述...",
  metadata: {
    capability_id: "kb_xxxx",
    module: "core/aiva_core",
    name: "detect_sqli",
    category: "attacking",
    ...
  }
}
    ↓
ModuleCapability (Pydantic)
    ↓
CapabilityRegistry.register_capabilities()
    ↓
PostgreSQL (capability_records表)
```

### 枚舉值處理

| ChromaDB 值 | Pydantic 枚舉 | 預設值 |
|------------|--------------|--------|
| "scanning" | `CapabilityCategory.SCANNING` | `UTILITY` |
| "attacking" | `CapabilityCategory.ATTACKING` | `UTILITY` |
| "analysis" | `CapabilityCategory.ANALYSIS` | `UTILITY` |
| "port_scan" | `CapabilitySubCategory.PORT_SCAN` | `None` |
| 1-5 | `CapabilityComplexity` | `MODERATE` (3) |

**重要修正**: 原本使用 `CapabilityCategory.OTHER`，但該枚舉值不存在。修正為使用 `UTILITY` 作為預設值。

### 錯誤處理

```python
try:
    # 批量寫入
    stats = backfill_to_postgresql(capabilities)
except Exception as e:
    error_context = create_error_context(
        error_type=ErrorType.DATABASE,
        severity=ErrorSeverity.HIGH,
        message="Backfill failed",
        details={"error": str(e)},
        exception=e
    )
    logger.error(f"❌ Backfill script failed: {error_context}")
    sys.exit(1)
```

---

## ✅ 驗證結果

### 1. 數據完整性驗證

```bash
# PostgreSQL 查詢
docker exec aiva-postgres psql -U aiva_user -d aiva_capabilities \
  -c "SELECT COUNT(*) FROM capability_records WHERE is_active = true;"

結果: 782 條記錄 ✅
```

### 2. 模組統計驗證

```sql
SELECT 
  COUNT(*) as total_capabilities, 
  COUNT(DISTINCT module) as total_modules,
  COUNT(DISTINCT language) as total_languages 
FROM capability_records 
WHERE is_active = true;

結果:
 total_capabilities | total_modules | total_languages 
--------------------+---------------+-----------------
                782 |            16 |               4
```

### 3. 腳本執行輸出

```
2025-11-29T00:23:26 INFO __main__ - ✅ Backfill completed:
2025-11-29T00:23:26 INFO __main__ -    Total input: 782
2025-11-29T00:23:26 INFO __main__ -    Added: 782
2025-11-29T00:23:26 INFO __main__ -    Updated: 0
2025-11-29T00:23:26 INFO __main__ -    Deleted: 0
2025-11-29T00:23:26 INFO __main__ -    Unchanged: 0
2025-11-29T00:23:26 INFO __main__ - 
2025-11-29T00:23:26 INFO __main__ - ✅ Verification passed
2025-11-29T00:23:26 INFO __main__ - ✅ Day 5 completed successfully!
```

---

## 🚀 使用方式

### 執行回填

```powershell
# 方式 1: 直接執行
cd C:\D\fold7\AIVA-git
python scripts/migration/backfill_capabilities.py

# 方式 2: 查看最後輸出
python scripts/migration/backfill_capabilities.py 2>&1 | Select-Object -Last 40
```

### 清空資料庫重新回填

```powershell
# 清空所有能力記錄
docker exec aiva-postgres psql -U aiva_user -d aiva_capabilities \
  -c "TRUNCATE capability_records, capability_versions, capability_change_logs, capability_invocation_stats CASCADE;"

# 重新執行回填
python scripts/migration/backfill_capabilities.py
```

---

## 📈 資料庫現狀

### capability_records 表

| 欄位 | 統計值 | 說明 |
|------|--------|------|
| 總記錄數 | 782 | 所有能力 |
| 活躍記錄 | 782 | `is_active = true` |
| 模組數 | 16 | 不同的模組 |
| 語言數 | 4 | Python, JavaScript, etc. |

### 能力分類統計（推測）

| 類別 | 預估數量 |
|------|---------|
| SCANNING | ~100-150 |
| ATTACKING | ~200-300 |
| ANALYSIS | ~100-150 |
| UTILITY | ~200-250 |
| REPORTING | ~50-80 |
| INTEGRATION | ~30-50 |

---

## 🔍 關鍵設計決策

### 1. 為什麼不包含 embeddings？

```python
result = collection.get(
    include=["documents", "metadatas"]  # 不包含 embeddings
)
```

**理由**:
- Embeddings 是大型陣列，會導致 `ValueError: The truth value of an array...`
- 回填只需要元數據，不需要向量
- 向量已經在 ChromaDB 中，不需要重複存儲到 PostgreSQL

### 2. 為什麼使用 UTILITY 作為預設類別？

```python
category = CapabilityCategory(category_str.lower()) or CapabilityCategory.UTILITY
```

**理由**:
- 原設計使用 `OTHER`，但該枚舉值不存在
- `UTILITY` 涵蓋最廣（編碼、解碼、工具類）
- 避免轉換失敗導致整個文檔被丟棄

### 3. 為什麼清空資料庫後再回填？

```sql
TRUNCATE ... CASCADE;
```

**理由**:
- 測試時可能已有部分數據（UNIQUE 約束衝突）
- 確保資料一致性（clean slate）
- 避免混合舊資料和新資料

---

## 🎓 學到的經驗

### 1. ChromaDB API 使用

```python
# ✅ 正確：遍歷所有 collections
collections = client.list_collections()
for collection_info in collections:
    collection = client.get_collection(name=collection_info.name)

# ❌ 錯誤：假設 collection 名稱
collection = client.get_collection("aiva_capabilities")
```

### 2. Pydantic 枚舉轉換

```python
# ✅ 正確：小寫轉換 + 錯誤處理
try:
    category = CapabilityCategory(category_str.lower())
except (ValueError, AttributeError):
    category = CapabilityCategory.UTILITY

# ❌ 錯誤：大寫轉換（枚舉值是小寫）
category = CapabilityCategory(category_str.upper())  # 會失敗
```

### 3. CapabilityRegistry API

```python
# ✅ 正確參數名
registry = CapabilityRegistry(
    pg_session=session,
    chroma_collection=None
)

# ❌ 錯誤參數名
registry = CapabilityRegistry(
    db_session=session,  # 錯誤
    chroma_client=None   # 錯誤
)
```

---

## 📚 相關文件

- [IMPLEMENTATION_ROADMAP.md](../../IMPLEMENTATION_ROADMAP.md) - Day 5 詳細計劃
- [CAPABILITY_METADATA_DATABASE_DESIGN.md](../../CAPABILITY_METADATA_DATABASE_DESIGN.md) - 整體架構
- [DAY4_COMPLETION_REPORT.md](../core/DAY4_COMPLETION_REPORT.md) - Day 4 雙寫機制
- [services/aiva_common/README.md](../../services/aiva_common/README.md) - v2.0 規範

---

## ✨ 後續步驟

### Day 6: 能力查詢 API 實現

**目標**: 創建 FastAPI 端點提供能力查詢服務

**任務**:
1. 檢查現有 API 檔案（修改優先）
2. 創建/修改 `internal_loop_api.py`
3. 實現查詢端點:
   - `POST /internal-loop/capabilities/query` - 查詢能力
   - `GET /internal-loop/capabilities/{name}` - 獲取單個能力
   - `GET /internal-loop/capabilities/{name}/history` - 查看歷史版本
4. 集成到主 FastAPI 應用
5. API 測試驗證

**預計時間**: 2-3 小時

---

## 🎉 總結

Day 5 成功完成能力元數據回填：

✅ **100% 數據轉換成功**: 782/782 文檔  
✅ **完整性驗證通過**: PostgreSQL 782 條記錄  
✅ **遵循 v2.0 規範**: 日誌、錯誤處理、類型註解  
✅ **健壯的錯誤處理**: 枚舉轉換容錯、數據庫回滾  

現在 AIVA 系統擁有：
- ✅ **結構化查詢能力**（PostgreSQL）
- ✅ **語義搜索能力**（ChromaDB）  
- ✅ **782 個可用能力**（涵蓋 16 個模組）

系統已經具備雙存儲架構基礎，為後續的 API 查詢和動態調用奠定基礎！

---

**報告生成時間**: 2025-11-29  
**執行者**: GitHub Copilot  
**審核狀態**: ✅ 完成，數據驗證通過
