當然有！前面的建議主要圍繞著優化現有架構和流程的「深度」。現在，我們可以探討一些能極大擴展 AIVA 平台能力和市場競爭力的全新「廣度」功能。

這些是專門為了「加強功能」而提出的前瞻性建議，可以讓 AIVA 從一個優秀的弱點評估平台，成長為一個全方位的攻擊面管理 (Attack Surface Management) 和應用安全中心。

1. 擴展掃描邊界：從 Web 應用到全方位資產
AIVA 目前的核心優勢在於 Web 應用和原始碼。下一步是將其強大的分析能力擴展到企業的其他關鍵資產上。

API 安全專項測試模組:

功能: 建立一個全新的 Function 模組，專門用於測試現代 API，包括 RESTful, GraphQL 和 gRPC。它不僅僅是發送 Payload，而是能理解 API 的結構 (Schema)，自動檢測 BOLA (破碎的物件級授權)、BFLA (破碎的功能級授權)、資料洩漏、以及注入類型等 API Top 10 漏洞。

價值: API 是目前企業數位轉型的核心，也是駭客攻擊的重點目標。提供專業的 API 掃描能力將是巨大的賣點。

行動應用安全測試 (MAST):

功能: 新增一系列 Function 服務，用於分析行動應用程式 (Android 的 APK 和 iOS 的 IPA 檔案)。

靜態分析: 反編譯 App，掃描硬編碼的密鑰、不安全的資料儲存、不安全的網路通訊設定等。

動態分析: 在模擬器中運行 App，攔截其與後端伺服器的所有通訊流量，然後將這些 API 流量交給現有的 DAST 和 API 測試模組進行深度分析。

價值: 補全了企業最重要的使用者端點——行動 App 的安全檢測能力。

外部攻擊面管理 (EASM):

功能: 在 Core 模組中新增一個「探索 (Discovery)」階段。使用者只需輸入一個公司名稱或主域名，AIVA 就會自動化地：

發現所有相關的子網域、IP 位址。

掃描開放的連接埠和服務。

尋找公開的雲端儲存體 (如 S3 buckets)。

在 GitHub, GitLab 等平台搜索洩漏的程式碼或憑證。

價值: 在掃描開始前，就為使用者繪製出一幅完整的、過去可能未知的「數位資產地圖」，極大地提升了掃描的覆蓋範圍。

2. 深化 AI 應用：從輔助決策到自主行動
您的 BioNeuronRAGAgent 是專案的寶石。我們可以賦予它更強大的能力。

AI 驅動的漏洞驗證與利用:

功能: 當一個 Function 模組回報了一個高風險漏洞（例如，遠端程式碼執行 RCE）時，Core 模組可以指派一個專門的「AI 驗證代理 (AI Confirmer Agent)」。這個代理會根據漏洞的類型和上下文，嘗試執行一系列非破壞性的、精準的後續指令來確認漏洞的真實可利用性，並安全地獲取證據（例如讀取 hostname 或 whoami）。

價值: 將誤報率降至最低，並為每個漏洞提供「可利用證明」，大大提升了報告的可信度和修復的優先級。

智慧業務邏輯漏洞挖掘:

功能: 訓練 AI 理解應用的核心業務流程（例如，電商的購物車結帳流程、金融應用的轉帳流程）。AI 可以透過分析 API 流量和前端程式碼來學習正常流程，然後自主設計測試案例來嘗試繞過這些流程，例如：修改商品價格、重複使用優惠券、繞過支付步驟等。

價值: 這是傳統掃描器的盲區，也是能造成巨大商業損失的漏洞類型。解決這個問題將使 AIVA 具備獨一無二的競爭力。# URL 佇列持久化改進方案

## 現況問題

當前 `UrlQueueManager` 使用純記憶體儲存：

```python
class UrlQueueManager:
    def __init__(self, seeds: list[str], *, max_depth: int = 3) -> None:
        self._queue: deque[tuple[str, int]] = deque()  # 記憶體佇列
        self._seen: set[str] = set()                   # 記憶體集合
        self._processed: set[str] = set()              # 記憶體集合
```

**問題**：服務重啟後所有掃描狀態遺失

## 解決方案 A: Redis 實現（推薦）

### 優勢

- 高效能讀寫
- 原生支援 Set 和 List 資料結構
- 自動去重功能
- 支援叢集和高可用

### 實現設計

```python
import redis
from typing import Optional, Tuple, List

class PersistentUrlQueueManager:
    def __init__(self, scan_id: str, seeds: list[str], *, max_depth: int = 3):
        self.scan_id = scan_id
        self.max_depth = max_depth
        
        # Redis 連接
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        
        # Redis key 命名規範
        self.queue_key = f"scan:{scan_id}:url_queue"
        self.seen_key = f"scan:{scan_id}:seen_urls"  
        self.processed_key = f"scan:{scan_id}:processed_urls"
        self.depth_key = f"scan:{scan_id}:url_depths"
        
        # 初始化種子 URLs
        self._initialize_seeds(seeds)
    
    def _initialize_seeds(self, seeds: list[str]) -> None:
        """初始化種子 URLs（只在首次掃描時）"""
        if not self.redis_client.exists(self.queue_key):
            pipe = self.redis_client.pipeline()
            for url in seeds:
                normalized = self._normalize_url(url)
                if normalized:
                    pipe.lpush(self.queue_key, normalized)
                    pipe.sadd(self.seen_key, normalized)  
                    pipe.hset(self.depth_key, normalized, 0)
            pipe.execute()
    
    def add(self, url: str, parent_url: str | None = None, depth: int = 0) -> bool:
        """添加 URL 到佇列"""
        # 解析相對 URL
        if parent_url and not urlparse(url).netloc:
            url = urljoin(parent_url, url)
        
        normalized = self._normalize_url(url)
        if not normalized or depth > self.max_depth:
            return False
        
        # 檢查是否已見過（Redis SADD 自動處理重複）
        if self.redis_client.sismember(self.seen_key, normalized):
            return False
        
        # 原子性操作
        pipe = self.redis_client.pipeline()
        pipe.lpush(self.queue_key, normalized)
        pipe.sadd(self.seen_key, normalized)
        pipe.hset(self.depth_key, normalized, depth)
        pipe.execute()
        
        return True
    
    def next(self) -> str:
        """獲取下一個 URL"""
        url = self.redis_client.rpop(self.queue_key)
        if not url:
            raise IndexError("URL queue is empty")
        
        # 標記為已處理
        self.redis_client.sadd(self.processed_key, url)
        return url
    
    def has_next(self) -> bool:
        """檢查是否還有待處理的 URL"""
        return self.redis_client.llen(self.queue_key) > 0
    
    def get_statistics(self) -> dict[str, int]:
        """獲取統計資訊"""
        return {
            "queued": self.redis_client.llen(self.queue_key),
            "seen": self.redis_client.scard(self.seen_key),
            "processed": self.redis_client.scard(self.processed_key),
        }
    
    def pause_scan(self) -> dict[str, any]:
        """暫停掃描，返回狀態快照"""
        stats = self.get_statistics()
        return {
            "scan_id": self.scan_id,
            "paused_at": datetime.utcnow().isoformat(),
            "statistics": stats,
            "redis_keys": {
                "queue": self.queue_key,
                "seen": self.seen_key, 
                "processed": self.processed_key
            }
        }
    
    def resume_scan(self) -> bool:
        """恢復掃描"""
        return self.redis_client.exists(self.queue_key)
    
    def cleanup(self) -> None:
        """清理 Redis 中的掃描資料"""
        keys_to_delete = [
            self.queue_key,
            self.seen_key,
            self.processed_key,
            self.depth_key
        ]
        self.redis_client.delete(*keys_to_delete)
```

### Redis 配置建議

```yaml
# docker-compose.yml 中添加 Redis
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
```

### 遷移策略

```python
# 在 scan_orchestrator.py 中的修改
class ScanOrchestrator:
    async def execute_scan(self, request: ScanStartPayload) -> ScanCompletedPayload:
        # 檢查是否為恢復掃描
        if request.resume_scan_id:
            url_queue = PersistentUrlQueueManager.resume(request.resume_scan_id)
        else:
            # 創建新的持久化佇列
            url_queue = PersistentUrlQueueManager(
                scan_id=context.scan_id,
                seeds=[str(t) for t in request.targets],
                max_depth=strategy_params.max_depth
            )
```

## 解決方案 B: PostgreSQL 實現

### 資料庫表設計

```sql
-- 掃描狀態表
CREATE TABLE crawling_state (
    scan_id VARCHAR(50) PRIMARY KEY,
    status VARCHAR(20) NOT NULL, -- 'running', 'paused', 'completed'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    max_depth INTEGER NOT NULL,
    statistics JSONB
);

-- URL 佇列表  
CREATE TABLE url_queue (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(50) REFERENCES crawling_state(scan_id),
    url VARCHAR(2048) NOT NULL,
    depth INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'processed'
    added_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    UNIQUE(scan_id, url)
);

-- 索引優化
CREATE INDEX idx_url_queue_scan_status ON url_queue(scan_id, status);
CREATE INDEX idx_url_queue_pending ON url_queue(scan_id) WHERE status = 'pending';
```

### Python 實現

```python
import asyncpg
from typing import AsyncGenerator

class DatabaseUrlQueueManager:
    def __init__(self, scan_id: str, db_pool: asyncpg.Pool):
        self.scan_id = scan_id
        self.db_pool = db_pool
    
    async def initialize_scan(self, seeds: list[str], max_depth: int = 3):
        """初始化掃描狀態"""
        async with self.db_pool.acquire() as conn:
            # 創建掃描記錄
            await conn.execute("""
                INSERT INTO crawling_state (scan_id, status, max_depth)
                VALUES ($1, 'running', $2)
                ON CONFLICT (scan_id) DO NOTHING
            """, self.scan_id, max_depth)
            
            # 添加種子 URLs
            for url in seeds:
                await conn.execute("""
                    INSERT INTO url_queue (scan_id, url, depth, status)
                    VALUES ($1, $2, 0, 'pending')
                    ON CONFLICT (scan_id, url) DO NOTHING
                """, self.scan_id, url)
    
    async def add(self, url: str, depth: int = 0) -> bool:
        """添加 URL 到佇列"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO url_queue (scan_id, url, depth, status)
                    VALUES ($1, $2, $3, 'pending')
                    ON CONFLICT (scan_id, url) DO NOTHING
                """, self.scan_id, url, depth)
                return True
        except Exception:
            return False
    
    async def next(self) -> str | None:
        """獲取下一個 URL（原子性操作）"""
        async with self.db_pool.acquire() as conn:
            # 使用 FOR UPDATE SKIP LOCKED 避免競爭條件
            result = await conn.fetchrow("""
                UPDATE url_queue
                SET status = 'processing', processed_at = NOW()
                WHERE id = (
                    SELECT id FROM url_queue
                    WHERE scan_id = $1 AND status = 'pending'
                    ORDER BY added_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING url
            """, self.scan_id)
            
            return result['url'] if result else None
    
    async def mark_completed(self, url: str):
        """標記 URL 為已完成"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE url_queue
                SET status = 'processed'
                WHERE scan_id = $1 AND url = $2
            """, self.scan_id, url)
```

## 比較與建議

| 特性 | Redis 方案 | PostgreSQL 方案 |
|------|------------|------------------|
| 效能 | 極高 | 中等 |
| 記憶體使用 | 較高 | 較低 |
| 持久化 | 可選 | 預設 |
| 查詢能力 | 有限 | 強大 |
| 運維複雜度 | 低 | 中等 |
| 與現有架構整合 | 需新增 Redis | 使用現有 PostgreSQL |

**推薦**：優先採用 **Redis 方案**，因為：

1. 更適合佇列操作的使用模式
2. 效能優勢明顯
3. 實現相對簡單
4. 可以漸進式遷移

## 實施時程

- **週 1**: Redis 環境搭建與基礎實現
- **週 2**: PersistentUrlQueueManager 完整實現與測試
- **週 3**: scan_orchestrator.py 整合與向後相容
- **週 4**: 壓力測試與效能優化

**預計總工時**: 4 週
