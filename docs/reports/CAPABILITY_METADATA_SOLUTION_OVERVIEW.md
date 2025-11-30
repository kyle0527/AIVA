# 能力元數據管理方案總覽

**問題**: 每次內循環分析後產生大量文件,無法識別變化,AI 不知如何調用能力  
**解決**: 數據庫 + 增量更新 + 數據合約通信

---

## 🎯 核心方案

### 1. **數據庫存儲** (替代文件)

```
文件方式 (舊)                    數據庫方式 (新)
├── capability_1.json            PostgreSQL (關係型)
├── capability_2.json            ├── capability_records (主表)
├── capability_3.json            ├── capability_versions (版本歷史)
├── ...                          └── capability_change_logs (變更記錄)
└── capability_782.json          
                                 ChromaDB (向量搜索)
❌ 782 個文件,難以管理            └── 語義檢索 (RAG 查詢)

                                 ✅ 統一數據庫,易於查詢和管理
```

### 2. **增量更新** (智能變更檢測)

```python
# 使用內容哈希自動識別變化
def process_scan_result(new_capabilities):
    for cap in new_capabilities:
        content_hash = compute_hash(cap)
        old_hash = db.get_hash(cap.name)
        
        if not old_hash:
            db.add(cap)              # ✅ 新增
        elif old_hash != content_hash:
            db.update(cap)           # ✅ 修改
        else:
            pass                     # ✅ 無變化,跳過
    
    # ✅ 自動檢測刪除
    for old_cap in db.get_all():
        if old_cap.name not in new_capabilities:
            db.mark_deleted(old_cap)  # ✅ 刪除
```

**結果**: 
- 首次掃描: 782 個新增
- 第二次掃描: 0 新增, 5 修改, 1 刪除, 776 無變化 ✅

### 3. **調用元數據** (AI 知道如何調用)

```python
# 能力元數據包含完整調用信息
capability = {
    "name": "detect_sqli",
    "module": "function_sqli",
    
    # ✅ 調用信息 (核心!)
    "invocation": {
        "protocol": "unified_caller",  # 使用統一調用器
        "module_arg": "function_sqli",  # 模組名
        "function_arg": "detect_sqli",  # 函數名
        "endpoint": "http://localhost:8001/execute"  # 端點 (如果是 HTTP)
    },
    
    # ✅ Python 代碼範例
    "call_example": "caller.call_function('function_sqli', 'detect_sqli', {'target_url': '...'})"
}
```

**AI 調用流程**:
```
1. AI 查詢 RAG: "SQL injection testing"
   ↓
2. RAG 返回: capability_name="detect_sqli", invocation={...}
   ↓
3. AI 讀取 invocation.protocol="unified_caller"
   ↓
4. AI 調用: UnifiedCaller.call_function(
       module_name="function_sqli",
       function_name="detect_sqli",
       parameters={"target_url": "http://test.com"}
   )
   ↓
5. ✅ 成功執行!
```

### 4. **數據合約** (標準化通信)

```python
# Pydantic Data Contract
from pydantic import BaseModel

class InvocationInfo(BaseModel):
    protocol: str          # "http", "grpc", "direct"
    endpoint: str | None
    module_arg: str
    function_arg: str

class CapabilityMetadata(BaseModel):
    name: str
    module: str
    language: str
    version: int
    content_hash: str
    
    invocation: InvocationInfo  # ✅ 調用元數據
    parameters: List[Parameter]
    return_info: ReturnInfo
    
    created_at: datetime
    updated_at: datetime
```

**好處**:
- ✅ 類型安全 (Pydantic 自動驗證)
- ✅ JSON 序列化/反序列化
- ✅ API 文檔自動生成 (FastAPI)
- ✅ 內外循環通信標準統一

---

## 📊 架構對比

### 舊架構 (文件存儲)

```
內循環掃描
    ↓
生成 782 個 JSON 文件  ❌ 文件膨脹
    ↓
手動加載到 ChromaDB
    ↓
AI 查詢 RAG
    ↓
返回能力名稱
    ↓
❌ AI 不知如何調用 (缺少 invocation 信息)
```

### 新架構 (數據庫 + 數據合約)

```
內循環掃描
    ↓
CapabilityRegistry (註冊中心)
    ├─ 計算內容哈希
    ├─ 檢測變更 (新增/修改/刪除)  ✅ 增量更新
    ├─ 自動添加調用元數據
    └─ 同步到數據庫
         ├─ PostgreSQL (主存儲 + 版本歷史)
         └─ ChromaDB (向量搜索)
    ↓
AI 查詢 RAG
    ↓
返回完整 CapabilityMetadata (包含 invocation)  ✅ 調用信息完整
    ↓
CapabilityInvoker 讀取 invocation
    ↓
根據 protocol 執行調用
    ├─ unified_caller → UnifiedFunctionCaller
    ├─ http → HTTP POST 請求
    └─ grpc → gRPC 調用
    ↓
✅ 成功執行! 並記錄調用統計
```

---

## 🔍 業界最佳實踐參考

### 1. Versioned Value Pattern (Martin Fowler)

每次更新保存新版本,可追溯歷史:

```sql
SELECT * FROM capability_versions 
WHERE capability_key = 'function_sqli::detect_sqli::...'
ORDER BY version DESC;

-- version 3: Added parameter 'timeout'
-- version 2: Changed return type
-- version 1: Initial creation
```

### 2. Hash-Based Change Detection

使用 SHA256 哈希檢測內容變化:

```python
# 只對穩定欄位計算哈希 (排除 timestamp 等易變欄位)
stable_content = {
    "name": cap.name,
    "parameters": cap.parameters,
    "return_type": cap.return_type
}
hash = sha256(json.dumps(stable_content, sort_keys=True))
```

### 3. Schema Migration (Dual Writing)

零停機遷移策略:

```
階段 1: 同時寫入新舊格式 (PostgreSQL + ChromaDB)
階段 2: 數據回填 (將 ChromaDB 歷史數據導入 PostgreSQL)
階段 3: 切換讀取 (從 PostgreSQL 讀取)
階段 4: 停止寫入舊格式,移除舊代碼
```

### 4. Data Contract (Pydantic)

類型安全的數據合約:

```python
# 自動驗證
cap = CapabilityMetadata(
    name="test",
    version="abc"  # ❌ ValidationError: version must be int
)

# JSON 互轉
json_str = cap.json()
cap_restored = CapabilityMetadata.parse_raw(json_str)
```

---

## 📁 關鍵文件

### 設計文檔
- `CAPABILITY_METADATA_DATABASE_DESIGN.md` - 詳細技術設計
- `IMPLEMENTATION_ROADMAP.md` - 實施路線圖
- `SYSTEM_READINESS_ANALYSIS.md` - 系統分析 (已更新解決方案)

### 代碼實現
- `services/aiva_common/schemas/capability_contract.py` - Data Contract (Pydantic)
- `services/core/aiva_core/internal_exploration/models.py` - SQLAlchemy Models
- `services/core/aiva_core/internal_exploration/capability_registry.py` - 註冊中心
- `services/core/aiva_core/task_planning/capability_invoker.py` - 調用器

### 數據庫
- `scripts/migrations/create_capability_db.py` - 數據庫初始化
- `scripts/backfill_capabilities.py` - 數據回填

### API
- `services/core/aiva_core/internal_exploration/internal_loop_api.py` - FastAPI 路由

---

## 🚀 快速開始

### 1. 安裝數據庫

```powershell
# Docker 方式
docker run -d `
  --name aiva-postgres `
  -e POSTGRES_USER=aiva_user `
  -e POSTGRES_PASSWORD=aiva_password `
  -e POSTGRES_DB=aiva_capabilities `
  -p 5432:5432 `
  postgres:14
```

### 2. 創建 Schema

```powershell
cd C:\D\fold7\AIVA-git
python scripts/migrations/create_capability_db.py
```

### 3. 回填數據

```powershell
python scripts/backfill_capabilities.py
```

### 4. 測試查詢

```powershell
# 啟動 API 服務
python -m services.core.aiva_core.api.main

# 測試查詢
curl http://localhost:8000/internal-loop/capabilities/query `
  -H "Content-Type: application/json" `
  -d '{"query": "SQL injection testing", "top_k": 5}'
```

### 5. 查看結果

```json
{
  "query": "SQL injection testing",
  "results": [
    {
      "name": "detect_sqli",
      "module": "function_sqli",
      "invocation": {
        "protocol": "unified_caller",
        "module_arg": "function_sqli",
        "function_arg": "detect_sqli"
      },
      "call_example_python": "caller.call_function('function_sqli', 'detect_sqli', {'target_url': '...'})"
    }
  ],
  "total_found": 1
}
```

---

## ✅ 解決的核心問題

| 問題 | 舊方案 | 新方案 | 狀態 |
|------|--------|--------|------|
| 文件膨脹 | 782 個 JSON 文件 | 數據庫統一存儲 | ✅ 解決 |
| 無法識別變化 | 每次全量更新 | 哈希檢測增量更新 | ✅ 解決 |
| 歷史追溯困難 | 無版本記錄 | 完整版本歷史表 | ✅ 解決 |
| AI 不知如何調用 | 缺少 invocation 信息 | 完整調用元數據 | ✅ 解決 |
| 通信無標準 | 數據格式混亂 | Pydantic Data Contract | ✅ 解決 |

---

## 📞 更多信息

- **詳細設計**: 見 `CAPABILITY_METADATA_DATABASE_DESIGN.md`
- **實施計劃**: 見 `IMPLEMENTATION_ROADMAP.md` (8-10 天完成)
- **業界參考**: Martin Fowler Patterns of Distributed Systems, Schema Migration Best Practices

---

**總結**: 通過數據庫 + 增量更新 + 數據合約,完美解決了內循環文件膨脹、變更檢測、AI 調用三大核心問題! 🎉
