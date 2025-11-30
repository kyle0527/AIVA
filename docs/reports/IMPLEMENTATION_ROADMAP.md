# AIVA 能力元數據數據庫實施路線圖

**目標**: 從文件存儲遷移到數據庫,實現增量更新和數據合約通信

**預計總時間**: 8-10 天  
**優先級**: P0 (內循環優化的基礎)

---

## 📋 目錄

- [實施計劃](#實施計劃)
  - [Week 1: 數據庫基礎 (第 1-3 天)](#week-1-數據庫基礎-第-1-3-天)
    - [Day 1: 環境準備](#day-1-環境準備)
    - [Day 2: 數據合約定義](#day-2-數據合約定義)
    - [Day 3: CapabilityRegistry 核心實現](#day-3-capabilityregistry-核心實現)
  - [Week 2: 雙寫機制與 API (第 4-6 天)](#week-2-雙寫機制與-api-第-4-6-天)
    - [Day 4: 雙寫機制集成](#day-4-雙寫機制集成)
    - [Day 5: 數據回填](#day-5-數據回填)
    - [Day 6: API 實現](#day-6-api-實現)
  - [Week 3: 能力調用器與切換 (第 7-10 天)](#week-3-能力調用器與切換-第-7-10-天)
    - [Day 7-8: CapabilityInvoker 實現](#day-7-8-capabilityinvoker-實現)
    - [Day 9-10: 切換讀路徑與清理](#day-9-10-切換讀路徑與清理)
- [檢查點](#檢查點)
- [成功指標](#成功指標)
- [風險緩解](#風險緩解)
- [後續優化](#後續優化)

---

## 📅 實施計劃

### Week 1: 數據庫基礎 (第 1-3 天)

#### Day 1: 環境準備 ✅

**任務清單**:
- [ ] 安裝 PostgreSQL 數據庫
  ```powershell
  # Windows: 下載安裝 PostgreSQL 14+
  # https://www.postgresql.org/download/windows/
  
  # 或使用 Docker
  docker run -d `
    --name aiva-postgres `
    -e POSTGRES_USER=aiva_user `
    -e POSTGRES_PASSWORD=aiva_password `
    -e POSTGRES_DB=aiva_capabilities `
    -p 5432:5432 `
    postgres:14
  ```

- [ ] 安裝 Python 依賴
  ```powershell
  cd C:\D\fold7\AIVA-git
  pip install sqlalchemy psycopg2-binary alembic pydantic
  ```

- [ ] 配置環境變量
  ```powershell
  # 設置數據庫 URL
  $env:AIVA_CAPABILITY_DB_URL = "postgresql://aiva_user:aiva_password@localhost:5432/aiva_capabilities"
  
  # 永久設置 (可選)
  [System.Environment]::SetEnvironmentVariable("AIVA_CAPABILITY_DB_URL", "postgresql://aiva_user:aiva_password@localhost:5432/aiva_capabilities", "User")
  ```

- [ ] 創建數據庫 schema
  ```powershell
  python scripts/migrations/create_capability_db.py
  ```

**驗證**:
```powershell
# 連接數據庫檢查
psql -h localhost -U aiva_user -d aiva_capabilities -c "\dt"
# 應該看到 4 個表: capability_records, capability_versions, capability_change_logs, capability_invocation_stats
```

#### Day 2: Data Contract 定義 ✅

**任務清單**:
- [ ] 創建 Pydantic schema
  - 位置: `services/aiva_common/schemas/capability_contract.py`
  - 參考: `CAPABILITY_METADATA_DATABASE_DESIGN.md` 中的定義

- [ ] 創建 SQLAlchemy models
  - 位置: `services/core/aiva_core/internal_exploration/models.py`
  - 定義 `CapabilityRecord`, `CapabilityVersion`, `CapabilityChangeLog`, `CapabilityInvocationStats`

- [ ] 單元測試
  ```python
  # tests/test_capability_contract.py
  def test_capability_metadata_serialization():
      cap = CapabilityMetadata(
          name="test_func",
          module="test_module",
          language="Python",
          ...
      )
      json_str = cap.json()
      cap_restored = CapabilityMetadata.parse_raw(json_str)
      assert cap == cap_restored
  ```

**驗證**:
```powershell
pytest tests/test_capability_contract.py -v
```

#### Day 3: CapabilityRegistry 核心實現 ✅

**任務清單**:
- [ ] 實現 `CapabilityRegistry` 類
  - 位置: `services/core/aiva_core/internal_exploration/capability_registry.py`
  - 核心方法:
    - `register_capabilities()` - 批量註冊/更新
    - `_compute_hash()` - 計算內容哈希
    - `_add_capability()` - 新增能力
    - `_update_capability()` - 更新能力
    - `_mark_deleted()` - 標記刪除

- [ ] 實現變更檢測
  ```python
  def detect_changes(old_caps, new_caps):
      # 返回 added, modified, deleted, unchanged 列表
  ```

- [ ] 單元測試
  ```python
  def test_change_detection():
      old_caps = [cap1, cap2]
      new_caps = [cap1_modified, cap3]  # cap1 修改, cap2 刪除, cap3 新增
      
      registry = CapabilityRegistry(...)
      stats = registry.register_capabilities(new_caps)
      
      assert stats["added"] == 1
      assert stats["modified"] == 1
      assert stats["deleted"] == 1
  ```

**驗證**:
```powershell
pytest tests/test_capability_registry.py -v
```

---

### Week 2: 集成與遷移 (第 4-6 天)

#### Day 4: Dual Writing 實現 ✅

**任務清單**:
- [ ] 修改 `internal_loop_connector.py`
  ```python
  class InternalLoopConnector:
      def __init__(self, rag_kb, pg_session):
          self.rag_kb = rag_kb  # ChromaDB (舊)
          self.registry = CapabilityRegistry(pg_session, rag_kb.chroma_collection)  # 新
      
      async def inject_to_rag(self, capabilities):
          # 舊方式: 只寫入 ChromaDB
          # for cap in capabilities:
          #     self.rag_kb.add_knowledge(...)
          
          # ✅ 新方式: 同時寫入 PostgreSQL 和 ChromaDB
          stats = self.registry.register_capabilities(capabilities)
          logger.info(f"Capability sync stats: {stats}")
  ```

- [ ] 測試雙寫
  ```powershell
  # 運行內循環掃描
  python -m aiva_cli internal-loop scan --update-db
  
  # 檢查數據
  psql -U aiva_user -d aiva_capabilities -c "SELECT COUNT(*) FROM capability_records;"
  ```

**驗證**: 數據同時存在於 PostgreSQL 和 ChromaDB

#### Day 5: 數據回填 ✅

**任務清單**:
- [ ] 創建回填腳本
  ```python
  # scripts/backfill_capabilities.py
  from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
  from services.core.aiva_core.internal_exploration.capability_registry import CapabilityRegistry
  
  def backfill():
      # 1. 從 ChromaDB 讀取所有現有能力
      vector_store = VectorStore(backend="chroma", persist_directory="data/vector_db/chroma")
      all_docs = vector_store.get_all_documents()
      
      # 2. 轉換為 CapabilityMetadata
      capabilities = []
      for doc in all_docs:
          cap = convert_doc_to_capability(doc)
          capabilities.append(cap)
      
      # 3. 批量插入 PostgreSQL
      registry = CapabilityRegistry(...)
      stats = registry.register_capabilities(capabilities)
      
      print(f"Backfill completed: {stats}")
  ```

- [ ] 執行回填
  ```powershell
  python scripts/backfill_capabilities.py
  ```

**驗證**:
```powershell
# 檢查記錄數是否一致
python -c "
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
vs = VectorStore(backend='chroma', persist_directory='data/vector_db/chroma')
print(f'ChromaDB count: {len(vs.get_all_documents())}')
"

psql -U aiva_user -d aiva_capabilities -c "SELECT COUNT(*) FROM capability_records;"
```

#### Day 6: API 接口實現 ✅

**任務清單**:
- [ ] 創建 FastAPI 路由
  - 位置: `services/core/aiva_core/internal_exploration/internal_loop_api.py`
  - 端點:
    - `POST /internal-loop/capabilities/query` - 查詢能力
    - `GET /internal-loop/capabilities/{name}` - 獲取單個能力
    - `GET /internal-loop/capabilities/{name}/history` - 查看歷史
    - `POST /internal-loop/scan/trigger` - 觸發掃描

- [ ] 集成到主 FastAPI app
  ```python
  # services/core/aiva_core/api/main.py
  from services.core.aiva_core.internal_exploration.internal_loop_api import router as internal_loop_router
  
  app.include_router(internal_loop_router)
  ```

- [ ] API 測試
  ```powershell
  # 啟動服務
  python -m services.core.aiva_core.api.main
  
  # 測試查詢
  curl http://localhost:8000/internal-loop/capabilities/query `
    -H "Content-Type: application/json" `
    -d '{"query": "SQL injection testing", "top_k": 5}'
  ```

**驗證**: API 返回正確的能力列表

---

### Week 2: AI 決策集成 (第 7-8 天)

#### Day 7: CapabilityInvoker 實現 ✅

**任務清單**:
- [ ] 創建 `CapabilityInvoker` 類
  - 位置: `services/core/aiva_core/task_planning/capability_invoker.py`
  - 核心方法:
    - `invoke_capability(name, params)` - 根據名稱調用能力
    - `_get_invocation_info(name)` - 從 PostgreSQL 查詢調用信息

  ```python
  class CapabilityInvoker:
      async def invoke_capability(self, capability_name: str, parameters: dict):
          # 1. 從數據庫查詢能力元數據
          cap = registry.get_capability_by_name(capability_name)
          
          # 2. 獲取調用信息
          invocation = cap.invocation
          
          # 3. 根據協議執行調用
          if invocation.protocol == "unified_caller":
              result = await unified_caller.call_function(
                  invocation.module_arg,
                  invocation.function_arg,
                  parameters
              )
          elif invocation.protocol == "http":
              result = await self._call_http(invocation.endpoint, parameters)
          
          # 4. 記錄調用統計
          self._update_invocation_stats(capability_name, result)
          
          return result
  ```

- [ ] 單元測試
  ```python
  @pytest.mark.asyncio
  async def test_invoke_capability():
      invoker = CapabilityInvoker(...)
      result = await invoker.invoke_capability(
          "detect_sqli",
          {"target_url": "http://test.com"}
      )
      assert result.success == True
  ```

**驗證**:
```powershell
pytest tests/test_capability_invoker.py -v
```

#### Day 8: Execution Planner 集成 ✅

**任務清單**:
- [ ] 修改 `execution_planner.py`
  ```python
  class ExecutionPlanner:
      def __init__(self):
          self.capability_invoker = CapabilityInvoker(...)
      
      async def execute_plan(self, plan):
          for step in plan.steps:
              if step.type == "execute_capability":
                  # ✅ 使用新的調用方式
                  result = await self.capability_invoker.invoke_capability(
                      capability_name=step.capability_name,
                      parameters=step.parameters
                  )
              # ... 其他步驟類型
  ```

- [ ] 端到端測試
  ```python
  @pytest.mark.asyncio
  async def test_full_pipeline():
      # 1. AI 查詢 RAG
      query_result = await ai_query.query("SQL injection testing")
      capability_name = query_result.results[0].metadata["capability_name"]
      
      # 2. 構建執行計劃
      planner = ExecutionPlanner()
      plan = planner.create_plan(capability_name, {"target_url": "http://test.com"})
      
      # 3. 執行計劃
      result = await planner.execute_plan(plan)
      
      assert result.success == True
  ```

**驗證**:
```powershell
pytest tests/test_full_pipeline.py -v
```

---

### Week 2-3: 切換與清理 (第 9-10 天)

#### Day 9: 切換讀取路徑 ✅

**任務清單**:
- [ ] 修改所有查詢邏輯
  ```python
  # 舊方式
  # results = chroma_client.search(query)
  
  # ✅ 新方式: 優先從 PostgreSQL 讀取
  results = registry.search_capabilities(query)
  ```

- [ ] 停止寫入舊格式
  ```python
  # 移除 internal_loop_connector.py 中的 ChromaDB 直接寫入
  # 只保留通過 registry 的統一寫入
  ```

- [ ] 性能測試
  ```python
  import time
  
  # 測試查詢性能
  start = time.time()
  results = registry.search_capabilities("SQL injection", top_k=10)
  duration = time.time() - start
  
  print(f"Query time: {duration:.3f}s")
  # 應該 < 100ms
  ```

**驗證**: 所有功能正常工作,性能無明顯下降

#### Day 10: 清理與文檔 ✅

**任務清單**:
- [ ] 移除舊代碼
  - 刪除不再使用的文件寫入邏輯
  - 清理臨時測試文件

- [ ] 更新文檔
  - 更新 `README.md` - 添加數據庫配置說明
  - 更新 `SYSTEM_READINESS_ANALYSIS.md` - 標記已解決的問題
  - 創建 `DATABASE_OPERATIONS.md` - 數據庫操作指南

- [ ] 創建運維腳本
  ```powershell
  # scripts/db_backup.ps1 - 數據庫備份
  # scripts/db_stats.ps1 - 查看統計信息
  # scripts/db_cleanup.ps1 - 清理舊版本
  ```

**驗證**: 
```powershell
# 運行完整測試套件
pytest tests/ -v --cov=services

# 檢查測試覆蓋率 > 80%
```

---

## 🔍 關鍵檢查點

### Checkpoint 1 (Day 3 結束)
- [ ] PostgreSQL 數據庫正常運行
- [ ] 4 個表創建成功
- [ ] Data Contract (Pydantic) 定義完成
- [ ] CapabilityRegistry 核心邏輯實現

### Checkpoint 2 (Day 6 結束)
- [ ] Dual Writing 正常工作
- [ ] ChromaDB 數據回填到 PostgreSQL
- [ ] API 接口可正常調用
- [ ] 增量更新邏輯測試通過

### Checkpoint 3 (Day 8 結束)
- [ ] CapabilityInvoker 實現並測試
- [ ] Execution Planner 集成完成
- [ ] 端到端測試通過: RAG 查詢 → AI 決策 → 實際調用

### Checkpoint 4 (Day 10 結束)
- [ ] 切換到新讀取路徑
- [ ] 舊代碼清理完成
- [ ] 文檔更新完成
- [ ] 運維腳本就緒

---

## 📊 成功指標

### 功能指標
- ✅ 內循環掃描後能正確識別新增/修改/刪除
- ✅ AI 查詢 RAG 能獲取完整調用元數據
- ✅ AI 能成功調用任意已註冊的能力
- ✅ 能查看任意能力的變更歷史

### 性能指標
- ✅ 能力查詢響應時間 < 100ms
- ✅ 增量更新處理 782 個能力 < 10 秒
- ✅ 數據庫存儲空間 < 100MB (初期)

### 可維護性指標
- ✅ 代碼測試覆蓋率 > 80%
- ✅ API 文檔完整 (Swagger/OpenAPI)
- ✅ 運維腳本完備

---

## 🚨 風險與緩解

### 風險 1: 數據庫遷移失敗
**緩解**: 
- Dual Writing 策略,保留 ChromaDB 作為備份
- 分階段回滾計劃
- 完整數據備份

### 風險 2: 性能下降
**緩解**:
- PostgreSQL 索引優化
- ChromaDB 向量搜索依然保留 (快速檢索)
- 增加查詢緩存

### 風險 3: Schema 演化
**緩解**:
- 使用 Alembic 進行數據庫 migration
- Versioned Value pattern 保留歷史版本
- Pydantic schema validation

---

## 📞 支持資源

### 技術參考
- `CAPABILITY_METADATA_DATABASE_DESIGN.md` - 詳細設計文檔
- `SYSTEM_READINESS_ANALYSIS.md` - 系統分析報告
- Martin Fowler Patterns of Distributed Systems

### 代碼範例
- `scripts/migrations/create_capability_db.py` - 數據庫創建
- `services/aiva_common/schemas/capability_contract.py` - Data Contract
- `services/core/aiva_core/internal_exploration/capability_registry.py` - Registry 實現

### 測試數據
- `tests/fixtures/sample_capabilities.json` - 測試用能力數據
- `tests/test_capability_registry.py` - 單元測試

---

## ✅ 完成確認

當所有任務完成後,系統應該:

1. **✅ 無文件膨脹**: 內循環掃描不再生成大量 JSON 文件
2. **✅ 增量更新**: 只更新變化的能力,自動識別新增/修改/刪除
3. **✅ 調用清晰**: AI 查到能力後知道如何調用 (protocol, endpoint, parameters)
4. **✅ 歷史追溯**: 可查看任意能力的完整變更歷史
5. **✅ 數據合約**: 內外循環通過標準化 Pydantic schema 通信
6. **✅ 零停機**: 整個遷移過程系統持續可用

🎉 恭喜!您已完成 AIVA 能力元數據數據庫的實施!
