# CLI指令存儲整合使用指南

## 概述

本模組提供完整的CLI指令執行歷史存儲和分析功能,支持追蹤370條複雜流程(4+步)的執行情況。

## 架構組件

### 核心模組

1. **CommandExecutionModel** (`models.py`)
   - 數據模型定義
   - 16個字段支持完整的指令追蹤
   - 9個索引優化查詢性能

2. **CommandRepository** (`command_repository.py`)
   - 指令存儲和查詢接口
   - 5個主要方法:
     - `save_command_execution()` - 保存執行記錄
     - `get_command_history()` - 查詢歷史
     - `get_command_statistics()` - 統計分析
     - `get_popular_capabilities()` - 熱門能力
     - `get_slow_executions()` - 性能瓶頸

3. **StorageManager整合**
   - 統一存儲接口
   - 自動創建CommandRepository實例
   - 代理所有指令相關方法

## 數據結構

### CommandExecutionModel 字段

```python
{
    # 基本信息
    "command_id": str,           # UUID
    "created_at": datetime,      # 創建時間
    "completed_at": datetime,    # 完成時間
    "command_name": str,         # 指令名稱(exec/query/flows/analyze)
    "command_type": str,         # 指令類型
    
    # 能力信息
    "capability_endpoint": str,  # 能力終點(如integration_module_sync)
    
    # 流程追蹤
    "flow_id": str,              # 數據流ID
    "flow_path": List[str],      # 執行路徑(JSON)
    "flow_length": int,          # 路徑長度(2-5)
    "flow_preference": str,      # 偏好(fastest/balanced/complete)
    
    # 模組追蹤
    "primary_module": str,       # 主要模組
    "modules_involved": List[str], # 涉及模組(JSON)
    
    # 執行參數和結果
    "parameters": Dict,          # 執行參數(JSON)
    "status": str,               # 狀態
    "success": bool,             # 是否成功
    "result_data": Dict,         # 結果數據(JSON)
    "error_message": str,        # 錯誤訊息
    
    # 性能指標
    "execution_time_ms": float,  # 執行時間(毫秒)
    
    # 用戶上下文
    "user_id": str,              # 用戶ID
    "session_id": str,           # 會話ID
    
    # AI支持
    "is_ai_generated": bool,     # 是否AI生成
    "natural_language_input": str, # 自然語言輸入
    
    # 元數據
    "extra_metadata": Dict       # 額外元數據(JSON)
}
```

## 使用方式

### 1. 初始化存儲管理器

```python
from services.core.aiva_core.service_backbone.storage import StorageManager

# 開發環境(SQLite)
storage = StorageManager(
    data_root="./data",
    db_type="sqlite"
)

# 生產環境(PostgreSQL)
storage = StorageManager(
    data_root="./data",
    db_type="postgres",
    db_config={
        "host": "localhost",
        "port": 5432,
        "database": "aiva_db",
        "user": "postgres",
        "password": "your_password"
    }
)
```

### 2. 記錄指令執行

#### 基本用法

```python
await storage.save_command_execution(
    command_id="uuid-string",
    command_name="exec",
    command_type="capability_execution",
    capability_endpoint="integration_module_sync",
    flow_path=["capability_cli", "ai_capability_query", "backends", 
               "message_broker", "integration_module_sync"],
    flow_length=5,
    primary_module="internal_exploration",
    modules_involved=["internal_exploration", "cognitive_core", 
                      "service_backbone", "service_backbone", "service_backbone"],
    status="completed",
    success=True,
    execution_time_ms=234.5
)
```

#### 完整參數範例

```python
await storage.save_command_execution(
    # 必填
    command_id="550e8400-e29b-41d4-a716-446655440000",
    command_name="exec",
    command_type="capability_execution",
    capability_endpoint="integration_module_sync",
    flow_path=["capability_cli", "ai_capability_query", "backends", 
               "message_broker", "integration_module_sync"],
    flow_length=5,
    primary_module="internal_exploration",
    modules_involved=["internal_exploration", "cognitive_core", 
                      "service_backbone", "service_backbone", "service_backbone"],
    status="completed",
    success=True,
    execution_time_ms=234.5,
    
    # 選填
    flow_id="flow_14",
    flow_preference="complete",
    parameters={"target": "external_system", "mode": "full_sync"},
    result_data={"synced_records": 1500, "duration": "2.5s"},
    error_message=None,
    user_id="user_001",
    session_id="session_20240115_001",
    is_ai_generated=True,
    natural_language_input="請同步所有外部系統的數據",
    metadata={
        "complexity": "very_high",
        "is_multi_path": True,
        "selected_by": "ai_optimizer"
    }
)
```

### 3. 查詢指令歷史

#### 查詢最近10條記錄

```python
history = await storage.get_command_history(limit=10)
for record in history:
    print(f"{record['command_id']}: {record['capability_endpoint']} - "
          f"{record['status']} ({record['execution_time_ms']}ms)")
```

#### 按能力終點篩選

```python
history = await storage.get_command_history(
    capability_endpoint="integration_module_sync",
    limit=20
)
```

#### 多條件篩選

```python
history = await storage.get_command_history(
    capability_endpoint="integration_module_sync",
    flow_preference="complete",
    success=True,
    user_id="user_001",
    limit=50,
    offset=0
)
```

#### 按時間範圍查詢

```python
from datetime import datetime, timedelta, UTC

start = datetime.now(UTC) - timedelta(days=7)
end = datetime.now(UTC)

history = await storage.get_command_history(
    start_date=start,
    end_date=end,
    limit=100
)
```

### 4. 統計分析

#### 獲取基本統計

```python
stats = await storage.get_command_statistics(days=7)
print(f"總執行次數: {stats['total_executions']}")
print(f"成功率: {stats['success_rate']}%")
print(f"平均執行時間: {stats['avg_execution_time_ms']}ms")
```

#### 按能力統計

```python
stats = await storage.get_command_statistics(
    capability_endpoint="integration_module_sync",
    days=30
)
print(f"流程偏好分佈: {stats['preference_distribution']}")
print(f"模組分佈: {stats['module_distribution']}")
```

#### 按模組統計

```python
stats = await storage.get_command_statistics(
    primary_module="cognitive_core",
    days=7
)
```

### 5. 性能分析

#### 查找熱門能力

```python
popular = await storage.get_popular_capabilities(
    limit=10,
    days=7
)

for cap in popular:
    print(f"{cap['capability_endpoint']}: "
          f"{cap['usage_count']}次使用, "
          f"{cap['success_rate']}%成功率, "
          f"{cap['avg_execution_time_ms']}ms平均時間")
```

#### 識別性能瓶頸

```python
slow = await storage.get_slow_executions(
    threshold_ms=1000.0,  # 超過1秒的執行
    limit=20,
    days=7
)

for cmd in slow:
    print(f"{cmd['capability_endpoint']}: "
          f"{cmd['execution_time_ms']}ms, "
          f"{cmd['flow_length']}步流程, "
          f"路徑: {' → '.join(cmd['flow_path'])}")
```

## 複雜流程支持

### 4+步流程追蹤

本模組專門支持追蹤複雜的多步驟流程(370條4+步流程):

```python
# 5步流程範例
await storage.save_command_execution(
    command_id="cmd_001",
    command_name="exec",
    command_type="capability_execution",
    capability_endpoint="integration_module_sync",
    flow_path=[
        "capability_cli",           # 步驟1
        "ai_capability_query",      # 步驟2
        "backends",                 # 步驟3
        "message_broker",           # 步驟4
        "integration_module_sync"   # 步驟5
    ],
    flow_length=5,
    primary_module="internal_exploration",
    modules_involved=[
        "internal_exploration",
        "cognitive_core",
        "service_backbone",
        "service_backbone",
        "service_backbone"
    ],
    status="completed",
    success=True,
    execution_time_ms=456.7
)
```

### 多路徑選擇

支持記錄不同的流程偏好選擇:

```python
# fastest - 最快路徑(最短)
await storage.save_command_execution(
    ...
    flow_preference="fastest",
    flow_length=2,
    ...
)

# balanced - 平衡路徑(中等)
await storage.save_command_execution(
    ...
    flow_preference="balanced",
    flow_length=3,
    ...
)

# complete - 完整路徑(最長)
await storage.save_command_execution(
    ...
    flow_preference="complete",
    flow_length=5,
    ...
)
```

## 與CLI整合

### 在CLI執行器中使用

```python
class CLICommandExecutor:
    def __init__(self, storage_manager):
        self.storage = storage_manager
    
    async def execute_capability(self, capability_endpoint, **kwargs):
        command_id = str(uuid.uuid4())
        start_time = datetime.now(UTC)
        
        try:
            # 執行能力
            result = await self._do_execute(capability_endpoint, kwargs)
            success = True
            status = "completed"
            error = None
        except Exception as e:
            result = None
            success = False
            status = "failed"
            error = str(e)
        
        # 記錄執行
        execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
        
        await self.storage.save_command_execution(
            command_id=command_id,
            command_name="exec",
            command_type="capability_execution",
            capability_endpoint=capability_endpoint,
            flow_path=kwargs.get("flow_path", []),
            flow_length=len(kwargs.get("flow_path", [])),
            primary_module=kwargs.get("primary_module", "unknown"),
            modules_involved=kwargs.get("modules_involved", []),
            status=status,
            success=success,
            result_data=result,
            error_message=error,
            execution_time_ms=execution_time,
            **kwargs  # 其他參數
        )
        
        return result
```

## 查詢範例

### 1. 查找失敗的執行

```python
failed = await storage.get_command_history(
    success=False,
    limit=50
)
```

### 2. 查找AI生成的指令

```python
ai_commands = await storage.get_command_history(
    is_ai_generated=True,
    limit=100
)
```

### 3. 按用戶查詢

```python
user_history = await storage.get_command_history(
    user_id="user_001",
    session_id="session_20240115_001",
    limit=50
)
```

### 4. 分析特定模組的使用

```python
module_stats = await storage.get_command_statistics(
    primary_module="cognitive_core",
    days=30
)
```

## 性能優化

### 索引使用

CommandExecutionModel 已創建9個索引:

1. `idx_command_status` - 按狀態查詢
2. `idx_success_created` - 成功記錄時間序列
3. `idx_module_capability` - 模組+能力聯合查詢
4. `idx_flow_preference` - 流程偏好查詢
5. `idx_execution_time` - 性能分析
6. `idx_user_session` - 用戶會話查詢
7. `idx_ai_generated` - AI生成指令篩選
8. `idx_created_at` - 時間範圍查詢
9. `idx_capability` - 能力終點查詢

### 查詢建議

1. **使用 limit 參數** - 避免返回過多數據
2. **添加時間範圍** - 縮小查詢範圍
3. **使用索引字段篩選** - 優先使用已索引的字段
4. **避免複雜的JSON查詢** - JSON字段查詢較慢

## 錯誤處理

所有方法都包含錯誤處理,失敗時返回空結果或默認值:

```python
# 保存失敗
result = await storage.save_command_execution(...)
if not result:
    logger.error("Failed to save command execution")

# 查詢失敗
history = await storage.get_command_history(...)
# 返回空列表 []

# 統計失敗
stats = await storage.get_command_statistics(...)
# 返回 {"total_executions": 0, "error": "..."}
```

## 完整範例

參見 `examples/cli_integration_example.py`:

```bash
cd services/core/aiva_core/service_backbone/storage
python -m examples.cli_integration_example
```

## 數據庫遷移

### SQLite (開發環境)

數據庫自動創建在 `data/database/aiva.db`

### PostgreSQL (生產環境)

需要先創建數據庫:

```sql
CREATE DATABASE aiva_db;
```

表結構自動創建。

## 監控和維護

### 定期清理舊數據

```python
# 刪除30天前的記錄(需要自定義實現)
cutoff_date = datetime.now(UTC) - timedelta(days=30)
# TODO: 實現清理方法
```

### 定期統計報告

```python
# 週報
weekly_stats = await storage.get_command_statistics(days=7)

# 熱門能力
top_capabilities = await storage.get_popular_capabilities(limit=20, days=7)

# 性能問題
slow_commands = await storage.get_slow_executions(threshold_ms=500, days=7)
```

## 相關文件

- `models.py` - 數據模型定義
- `command_repository.py` - 儲存庫實現
- `storage_manager.py` - 統一存儲接口
- `backends.py` - 存儲後端實現
- `CAPABILITY_CLI_DESIGN.md` - CLI設計文檔
- `classification_results_final/` - 數據流分類結果

## 問題排查

### 1. 數據庫連接失敗

檢查數據庫配置:
```python
# PostgreSQL
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "aiva_db",
    "user": "postgres",
    "password": "your_password"
}
```

### 2. 表不存在

確保 StorageManager 正確初始化:
```python
storage = StorageManager(...)
# 表應自動創建
```

### 3. 查詢性能慢

- 添加時間範圍限制
- 使用 limit 參數
- 檢查索引是否正確創建

## 未來擴展

- [ ] 添加數據清理方法
- [ ] 實現數據導出功能
- [ ] 添加更多統計維度
- [ ] 支持實時監控dashboard
- [ ] 集成告警機制
