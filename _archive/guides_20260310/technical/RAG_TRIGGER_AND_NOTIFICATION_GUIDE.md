"""RAG 觸發和用戶通知系統使用指南

本文檔說明如何使用 RAG 觸發器和用戶通知系統。

## 架構概覽

```
學習流程:
1. 讀取三個數據源
   ├─ 當前執行記錄 (JSONL)
   ├─ 歷史數據 (JSONL)
   └─ 能力知識庫 (Markdown)

2. 三路比對
   └─ 計算相似度

3. 判斷是否為已知情況
   ├─ 已知 (相似度 >= 0.6) → 使用知識庫
   └─ 未知 (相似度 < 0.6) → 觸發 RAG

4. RAG 搜索 (如果觸發)
   ├─ 搜索 CVE 資料庫
   ├─ 搜索技術文檔
   ├─ 搜索安全研究報告
   └─ 搜索已有知識庫 (向量搜索)

5. 用戶通知
   ├─ 控制台日誌
   ├─ 文件記錄 (notifications.jsonl)
   └─ 回調函數 (可擴展 WebSocket)

6. 生成優化建議

7. 驗證效果

8. 保存 (如果驗證通過)
```

## 1. 基本使用

### 1.1 初始化學習系統

```python
from services.core.aiva_core.cognitive_core.learning_system.experience_manager import ExperienceManager
from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
from services.integration.simple_data_manager import get_data_manager

# 初始化知識庫和 RAG 引擎
knowledge_base = KnowledgeBase()
rag_engine = RAGEngine(knowledge_base)

# 初始化 ExperienceManager（帶 RAG 和通知）
experience_manager = ExperienceManager(
    capacity=10000,
    data_manager=get_data_manager(),  # 讀取整合模組數據
    rag_engine=rag_engine,  # RAG 引擎
    similarity_threshold=0.6,  # 相似度閾值
    enable_notifications=True,  # 啟用用戶通知
)
```

### 1.2 觸發學習流程

```python
import asyncio

# 準備當前數據
current_data = {
    "capability": "xss",
    "target": {"url": "http://example.com"},
    "request": {
        "method": "GET",
        "parameters": {"q": "<script>alert(1)</script>"}
    },
    "response": {
        "status_code": 200,
        "error_message": "WAF blocked",
        "response_type": "blocked"
    },
    "result": {
        "success": False,
        "findings": []
    }
}

# 觸發學習（異步）
result = await experience_manager.trigger_learning_with_rag(
    capability="xss",
    current_data=current_data,
    session_id="learning_001",
)

print(f"學習完成: {result['session_id']}")
print(f"未知情況: {result['unknown_situation_detected']}")
print(f"RAG 觸發: {result['rag_triggered']}")
print(f"驗證通過: {result['validation_passed']}")
```

## 2. 用戶通知系統

### 2.1 通知類型

系統會自動發送以下通知：

1. **未知情況檢測** (WARNING)
   - 當相似度低於閾值時觸發
   - 說明觸發原因和當前數據快照

2. **RAG 搜索啟動** (INFO)
   - 顯示搜索查詢和搜索範圍

3. **RAG 搜索完成** (INFO/WARNING)
   - 顯示找到的資源數量和摘要
   - 如果未找到資源，顯示警告

4. **RAG 搜索失敗** (CRITICAL)
   - 顯示錯誤訊息

5. **學習開始** (INFO)
   - 顯示能力類型和數據源

6. **學習完成** (INFO/WARNING)
   - 顯示改進建議和驗證結果

### 2.2 通知輸出示例

#### 控制台輸出
```
[2026-01-20 10:30:15] ⚠️ [WARNING] 檢測到未知情況 - 系統遇到未知的安全情況，相似度低於閾值。原因: 相似度過低: 0.45 < 0.6 (最相似數據源: historical_data)
[2026-01-20 10:30:15] ℹ️ [INFO] RAG 搜索已啟動 - 正在搜索外部知識資源: xss WAF blocked
[2026-01-20 10:30:17] ℹ️ [INFO] RAG 搜索完成 - 找到 5 個相關資源
```

#### 文件記錄 (notifications.jsonl)
```json
{"notification_id": "notif_alert_20260120_103015", "type": "unknown_situation", "level": "warning", "title": "檢測到未知情況", "message": "系統遇到未知的安全情況，相似度低於閾值。原因: 相似度過低: 0.45 < 0.6 (最相似數據源: historical_data)", "details": {...}, "timestamp": "2026-01-20T10:30:15.123456"}
{"notification_id": "notif_alert_20260120_103015_rag_triggered", "type": "rag_triggered", "level": "info", "title": "RAG 搜索已啟動", "message": "正在搜索外部知識資源: xss WAF blocked", "details": {...}, "timestamp": "2026-01-20T10:30:15.234567"}
{"notification_id": "notif_alert_20260120_103015_rag_completed", "type": "rag_completed", "level": "info", "title": "RAG 搜索完成", "message": "找到 5 個相關資源", "details": {...}, "timestamp": "2026-01-20T10:30:17.345678"}
```

### 2.3 自定義通知回調

可以註冊自定義回調函數來處理通知：

```python
from services.core.aiva_core.cognitive_core.learning_system.notification_system import get_notification_system

notification_system = get_notification_system()

# 自定義回調：發送 WebSocket 消息
def websocket_callback(notification):
    # 發送到 WebSocket
    websocket.send(json.dumps(notification.to_dict()))

# 註冊回調
notification_system.register_callback(websocket_callback)
```

## 3. RAG 觸發機制

### 3.1 觸發條件

RAG 在以下情況自動觸發：

1. **相似度低於閾值**
   - 當前數據與所有數據源的相似度 < 0.6

2. **遇到未見過的錯誤**
   - 錯誤訊息在歷史記錄中不存在

3. **檢測到新型模式**
   - 請求/響應模式與已知模式不匹配

### 3.2 相似度計算

系統通過以下特徵計算相似度：

- 響應狀態碼
- 錯誤訊息
- 響應類型
- 請求方法
- 請求參數
- 執行結果 (success/failure)
- 發現數量

示例：
```python
# 提取特徵
features_current = "status:200 error:WAF blocked type:blocked method:GET success:False findings:0"
features_historical = "status:200 error:WAF denied type:blocked method:GET success:False findings:0"

# 計算相似度 (使用 SequenceMatcher)
similarity = 0.85  # 很相似，不觸發 RAG
```

### 3.3 RAG 搜索範圍（外部搜索）

**重要：RAG 是對外搜索，不是內部向量庫！**

因為內部三個數據源（當前記錄、歷史數據、知識庫）已經找不到資料，所以才需要去外部搜索新知識。

RAG 會搜索以下**外部資源**：

1. **CVE 數據庫** (NVD - National Vulnerability Database)
   - API: `https://services.nvd.nist.gov/rest/json/cves/2.0`
   - 搜索相關漏洞和安全公告
   - 返回: CVE ID、描述、影響、修復建議

2. **Exploit-DB**
   - 網站: `https://www.exploit-db.com`
   - 搜索公開的漏洞利用代碼
   - 返回: 利用技術、繞過方法

3. **Google 搜索** (技術文章)
   - 搜索範圍: Stack Overflow, GitHub, 技術博客
   - 查詢: `{capability} bypass {error_message} site:stackoverflow.com`
   - 返回: 相關技術討論和解決方案

4. **GitHub Security Advisory**
   - API: GitHub GraphQL
   - 網站: `https://github.com/advisories`
   - 搜索: 開源項目的安全公告
   - 返回: 漏洞詳情、修復 PR

5. **安全研究論文**（可擴展）
   - arXiv、IEEE、ACM 等學術資源
   - 安全會議論文（Black Hat、DEF CON）

**搜索方式：**
- HTTP/HTTPS 請求（API 調用）
- 網頁爬取（使用 aiohttp、BeautifulSoup）
- 搜索引擎 API（Google Custom Search API）
- 不是瀏覽器自動化（除非必要）

### 3.4 搜索查詢生成

系統自動從當前數據生成搜索查詢：

```python
# 輸入數據
current_data = {
    "capability": "xss",
    "response": {"error_message": "WAF blocked"},
    "target": {"url": "http://example.com"},
    "result": {"success": False}
}

# 生成查詢
query = "xss WAF blocked http://example.com failure analysis"
```

## 4. 學習流程詳解

### 4.1 數據源讀取

```python
# 1. 當前執行記錄（最近 50 條）
current_records = data_manager.load_capability_data(
    capability="xss",
    limit=50
)

# 2. 歷史數據（最近 7 天，最多 100 條）
historical_data = data_manager.load_capability_data(
    capability="xss",
    start_time=datetime.now() - timedelta(days=7),
    limit=100
)

# 3. 能力知識庫（Markdown 分析報告）
knowledge_base_data = knowledge_manager.get_module_info("xss")
```

### 4.2 三路比對

系統會計算當前數據與三個數據源的相似度：

```python
# 比對結果示例
{
    "current_records": 0.45,  # 最高相似度
    "historical_data": 0.52,
    "knowledge_base": 0.38,
    "max_similarity": 0.52,
    "source": "historical_data"
}
```

### 4.3 優化建議生成

基於比對結果和 RAG 搜索生成優化建議：

```python
optimization_plan = {
    "timestamp": "2026-01-20T10:30:17",
    "data_sources_used": {
        "current_records": 50,
        "historical_data": 100,
        "knowledge_base": 1,
        "rag_results": 5
    },
    "improvements": {
        "success_rate_change": -0.15,  # 成功率下降
        "rag_suggestions": [
            {
                "type": "attack_technique",
                "relevance": 0.89,
                "content_preview": "WAF bypass using HTML encoding..."
            }
        ]
    },
    "new_weights": {
        "learning_rate": 0.01,
        "exploration_rate": 0.1
    }
}
```

### 4.4 驗證和保存

```python
# 驗證效果
validation_passed = _validate_optimization(optimization_plan)

if validation_passed:
    # 保存到 data/optimizations/{capability}_optimization_{timestamp}.json
    _save_optimization("xss", optimization_plan)
    print("✅ 新權重已保存")
else:
    print("❌ 驗證未通過，已丟棄")
```

## 5. 警報歷史查詢

### 5.1 查詢 RAG 觸發歷史

```python
# 獲取最近的 RAG 警報
alerts = experience_manager.rag_trigger.get_alert_history(limit=10)

for alert in alerts:
    print(f"Alert ID: {alert['alert_id']}")
    print(f"Trigger Reason: {alert['trigger_reason']}")
    print(f"RAG Results: {alert['rag_results_count']}")
    print(f"Status: {alert['status']}")
    print("---")
```

### 5.2 查詢通知歷史

```python
from services.core.aiva_core.cognitive_core.learning_system.notification_system import get_notification_system

notification_system = get_notification_system()

# 獲取所有通知
all_notifications = notification_system.get_notification_history(limit=50)

# 只獲取 RAG 相關通知
rag_notifications = notification_system.get_notification_history(
    limit=20,
    notification_type=NotificationType.RAG_COMPLETED
)
```

## 6. 配置和調整

### 6.1 調整相似度閾值

```python
# 降低閾值：更容易觸發 RAG（更敏感）
experience_manager = ExperienceManager(
    similarity_threshold=0.5,  # 默認 0.6
    ...
)

# 提高閾值：減少 RAG 觸發（更保守）
experience_manager = ExperienceManager(
    similarity_threshold=0.7,
    ...
)
```

### 6.2 禁用用戶通知

```python
experience_manager = ExperienceManager(
    enable_notifications=False,  # 禁用通知
    ...
)
```

### 6.3 只輸出到文件（不輸出到控制台）

```python
from services.core.aiva_core.cognitive_core.learning_system.notification_system import NotificationSystem

notification_system = NotificationSystem(
    log_to_console=False,  # 禁用控制台輸出
    save_to_file=True,     # 只保存到文件
)
```

## 7. 完整示例

```python
import asyncio
from services.core.aiva_core.cognitive_core.learning_system.experience_manager import ExperienceManager
from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
from services.integration.simple_data_manager import get_data_manager

async def main():
    # 初始化
    knowledge_base = KnowledgeBase()
    rag_engine = RAGEngine(knowledge_base)
    
    experience_manager = ExperienceManager(
        data_manager=get_data_manager(),
        rag_engine=rag_engine,
        similarity_threshold=0.6,
        enable_notifications=True,
    )
    
    # 模擬當前數據
    current_data = {
        "capability": "xss",
        "target": {"url": "http://example.com"},
        "request": {
            "method": "GET",
            "parameters": {"q": "<script>alert(1)</script>"}
        },
        "response": {
            "status_code": 403,
            "error_message": "Access Denied by WAF",
            "response_type": "blocked"
        },
        "result": {
            "success": False,
            "findings": []
        }
    }
    
    # 觸發學習
    print("🎓 開始學習流程...")
    result = await experience_manager.trigger_learning_with_rag(
        capability="xss",
        current_data=current_data,
    )
    
    # 顯示結果
    print(f"\n✅ 學習完成")
    print(f"  - 會話 ID: {result['session_id']}")
    print(f"  - 未知情況: {result['unknown_situation_detected']}")
    print(f"  - RAG 觸發: {result['rag_triggered']}")
    print(f"  - RAG 結果數: {result['rag_results_count']}")
    print(f"  - 驗證通過: {result['validation_passed']}")
    
    if result['alert']:
        print(f"\n⚠️ RAG 警報:")
        print(f"  - 警報 ID: {result['alert']['alert_id']}")
        print(f"  - 觸發原因: {result['alert']['trigger_reason']}")
        print(f"  - 搜索查詢: {result['alert']['search_query']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 8. 故障排除

### 8.1 RAG 未觸發

**可能原因：**
- 相似度高於閾值（已知情況）
- RAG 引擎未正確初始化
- 數據源為空

**解決方案：**
```python
# 檢查相似度
is_known, max_similarity, source = rag_trigger.check_if_known_situation(...)
print(f"Max similarity: {max_similarity}, threshold: {similarity_threshold}")

# 降低閾值
experience_manager = ExperienceManager(similarity_threshold=0.5)
```

### 8.2 通知未顯示

**可能原因：**
- 通知系統未啟用
- 日誌級別設置過高

**解決方案：**
```python
import logging
logging.basicConfig(level=logging.INFO)  # 確保日誌級別

# 確保通知已啟用
experience_manager = ExperienceManager(enable_notifications=True)
```

### 8.3 RAG 搜索失敗

**可能原因：**
- 知識庫未初始化
- 向量存儲連接失敗

**解決方案：**
```python
# 檢查 RAG 引擎狀態
try:
    results = await rag_engine._search_with_cache("test", "test", 1)
    print(f"RAG 正常: {len(results)} results")
except Exception as e:
    print(f"RAG 錯誤: {e}")
```

## 9. 文件位置

- **RAG 觸發器**: `services/core/aiva_core/cognitive_core/learning_system/rag_trigger.py`
- **通知系統**: `services/core/aiva_core/cognitive_core/learning_system/notification_system.py`
- **經驗管理器**: `services/core/aiva_core/cognitive_core/learning_system/experience_manager.py`
- **通知記錄**: `services/core/aiva_core/cognitive_core/learning_system/data/notifications.jsonl`
- **優化記錄**: `services/core/aiva_core/cognitive_core/learning_system/data/optimizations/*.json`

## 10. 總結

- ✅ **自動檢測未知情況**：相似度 < 0.6 時自動觸發 RAG
- ✅ **實時用戶通知**：通過日誌、文件、回調通知用戶
- ✅ **外部知識搜索**：搜索 CVE、技術文檔、安全研究
- ✅ **效果驗證**：只保存驗證通過的優化
- ✅ **可擴展架構**：支持自定義回調和通知渠道
