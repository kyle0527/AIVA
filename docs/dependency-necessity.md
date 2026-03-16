# AIVA 保留依賴必要性分析

**分析日期**: 2026-03-16
**範圍**: 清理後剩餘的 26 個核心依賴 + 7 個 AI optional 依賴

---

## 評估標準

每個依賴從三個面向評估：
1. **是否可自行實作？** — 從零寫的工程成本與維護代價
2. **是否有更輕量替代？** — 標準庫或更小的套件能否取代
3. **移除後的影響範圍** — 影響多少檔案、是否為核心路徑

---

## 核心依賴（26 個）

### 1. fastapi (3 檔)
- **用途**: API 路由、請求驗證、SSE endpoint
- **為何不能替代**: 自行實作 ASGI 框架 = 重造輪子。`starlette` 是替代選項但 fastapi 提供自動 OpenAPI 文件、Pydantic 整合、依賴注入，這些自己寫需數千行程式碼
- **自行實作成本**: 極高（6+ 人月）
- **結論**: **必要，無合理替代**

### 2. uvicorn (2 檔)
- **用途**: ASGI 伺服器，fastapi 的運行容器
- **為何不能替代**: 這是 fastapi 的標配伺服器。替代品 `hypercorn`、`daphne` 效能較差且社群較小。自行寫 ASGI 伺服器完全不現實
- **結論**: **必要，fastapi 的必要搭配**

### 3. pydantic (74 檔)
- **用途**: 資料驗證、schema 定義、序列化/反序列化
- **為何不能替代**: **全專案引用量最高的依賴 (74 檔)**。用 dataclasses + 手動驗證取代需重寫上萬行程式碼，且喪失 runtime 型別驗證、JSON schema 生成等功能
- **自行實作成本**: 不可行（等同重寫半個專案）
- **結論**: **核心基礎設施，完全不可替代**

### 4. pydantic-settings (1 檔)
- **用途**: `settings.py` 中的 `BaseSettings`，從環境變數載入設定
- **為何不能替代**: 可以用 `os.environ` + 手動解析，但會喪失型別驗證、`.env` 自動載入、巢狀設定等功能。只有 1 檔使用但它是全域設定的單一入口
- **替代方案**: 理論上可用 `pydantic.BaseModel` + `os.environ` 自行包裝，但代碼會更冗長
- **結論**: **保留 — 成本低、已是 pydantic 生態的一部分**

### 5. aio-pika (3 檔)
- **用途**: RabbitMQ 非同步客戶端，Worker 間訊息傳遞
- **為何不能替代**: 替代品是 `pika`（sync）但我們需要 async。自行用 AMQP 協定寫客戶端 = 實作整個 AMQP 0-9-1 協定棧
- **結論**: **必要，RabbitMQ async 的唯一成熟選項**

### 6. aiohttp (17 檔)
- **用途**: Async HTTP client + WebSocket，XSS/SQLi/Web 掃描引擎
- **為何不能用 httpx 替代**: `aiohttp` 提供原生 WebSocket 支援（`ws_connect`），`httpx` 不支援 WebSocket。17 個檔案中有部分依賴 `aiohttp.ClientSession` 的特定 API（如 cookie jar 行為、connector 自訂）
- **可否整合**: 中期可將純 HTTP 呼叫遷移到 httpx，但 WebSocket 場景仍需 aiohttp
- **結論**: **必要，WebSocket 場景無替代**

### 7. aiofiles (4 檔)
- **用途**: 非同步檔案 I/O（forensic 模組、AI manager、SSE）
- **為何不能替代**: 可以用 `asyncio.to_thread(open(...))` 替代，但 `aiofiles` 只有 ~200 行，提供更簡潔的 API。已在 4 個檔案中使用
- **替代方案**: `asyncio.to_thread()` + 標準 `open()`，但代碼會更冗長
- **結論**: **保留 — 極輕量，API 更簡潔**

### 8. httpx (22 檔)
- **用途**: 主力 async/sync HTTP client，安全測試引擎核心
- **為何不能替代**: **22 檔引用，第二大依賴**。相比 `requests` 多了 async 支援、HTTP/2、更好的 timeout 控制。相比 `aiohttp` 同時支援 sync/async。安全掃描器深度依賴其 API
- **結論**: **核心基礎設施，不可替代**

### 9. requests (15 檔)
- **用途**: Sync HTTP client，dashboard/偵察/爬蟲
- **為何現在不移除**: 15 個檔案使用中，遷移到 httpx 需逐一修改並測試。`httpx` API 與 `requests` 類似但不完全相容（如 `requests.Session` vs `httpx.Client`）
- **中期計畫**: 逐步遷移至 httpx
- **結論**: **暫時保留，中期遷移目標**

### 10. sqlalchemy (3 檔)
- **用途**: ORM、資料庫 schema 定義、查詢建構
- **為何不能替代**: 用原生 SQL 字串可以取代，但會喪失：schema migration 追蹤、連線池管理、多資料庫相容性。自行實作 ORM = 重造一個 SQLAlchemy
- **結論**: **必要，資料庫抽象層的業界標準**

### 11. asyncpg (2 檔)
- **用途**: PostgreSQL async driver（DB helper + vector store）
- **為何不能替代**: `psycopg3` 是替代品但 asyncpg 是 PostgreSQL async 效能最好的 driver（純 C 實作，比 psycopg3 快 2-3x）。且已與 SQLAlchemy async 整合
- **結論**: **必要，效能最優的 PostgreSQL async driver**

### 12. redis (0 檔 import)
- **用途**: config 中有設定，可能透過 docker-compose 使用
- **狀態**: 零 import，但可能是基礎設施依賴（cache layer、session store）
- **結論**: **⚠️ 待確認 — 若確認無使用可移除**

### 13. grpcio (4 檔)
- **用途**: gRPC runtime，Python ↔ Rust/Go/TypeScript 跨語言通訊
- **為何不能替代**: gRPC 是 AIVA 多語言架構的通訊骨幹。替代方案（REST/WebSocket）會喪失：強型別契約、雙向串流、Protocol Buffers 的高效序列化
- **結論**: **必要，跨語言通訊核心**

### 14. protobuf (4 檔)
- **用途**: Protocol Buffers 序列化/反序列化
- **為何不能替代**: gRPC 的必要搭配。`.proto` 編譯出的 `_pb2.py` 直接依賴此套件
- **結論**: **必要，grpcio 的必要搭配**

### 15. PyJWT (1 檔)
- **用途**: `security.py` 中的 JWT token 編解碼
- **為何不能替代**: 可以用 `cryptography` 自行實作 JWT，但需處理 header/payload base64、簽名驗證、claims 過期等邏輯。PyJWT 極輕量（~30KB），沒有理由自行實作
- **結論**: **保留 — 極輕量，自行實作得不償失**

### 16. cryptography (2 檔)
- **用途**: 加密操作（security.py + config_manager.py）
- **為何不能替代**: **絕對不應自行實作密碼學**。這是 Python 密碼學的事實標準，底層用 OpenSSL/BoringSSL。任何替代方案（如 `pycryptodome`）功能更少且社群更小
- **結論**: **必要，安全性關鍵依賴**

### 17. beautifulsoup4 (2 檔)
- **用途**: HTML 解析（XSS 工具 + Web 爬蟲）
- **為何不能替代**: 可以用 `lxml` 單獨做 HTML 解析，但 bs4 的容錯解析器對殘缺 HTML 更強健（安全掃描器經常遇到畸形 HTML）。且 API 比 lxml 直觀
- **結論**: **保留 — HTML 容錯解析的最佳選擇**

### 18. lxml (1 檔)
- **用途**: XXE（XML External Entity）偵測
- **為何不能替代**: `xml.etree` 標準庫無法模擬 XXE 攻擊向量。`lxml` 提供對 libxml2 的精確控制（entity expansion、DTD 載入），這是 XXE 偵測的核心需求
- **結論**: **必要，XXE 偵測的技術需求**

### 19. dnspython (3 檔)
- **用途**: DNS 解析、子網域掃描、偵察
- **為何不能替代**: `socket.getaddrinfo()` 只做正向查詢。`dnspython` 支援 ANY/MX/TXT/NS 等記錄類型查詢、zone transfer 嘗試、自訂 DNS server，這些是偵察掃描的核心功能
- **結論**: **必要，DNS 偵察無替代**

### 20. click (3 檔)
- **用途**: CLI 框架（aiva_cli、core_analyzer、common cli）
- **為何不能替代**: `argparse` 是標準庫替代品，但 click 提供裝飾器語法、自動 help 生成、命令分群，代碼量更少。3 個檔案使用量不大
- **替代方案**: `argparse`（標準庫），但遷移成本 > 保留成本
- **結論**: **保留 — 遷移到 argparse 不值得**

### 21. rich (21 檔)
- **用途**: Terminal UI、進度條、表格、語法高亮
- **為何不能替代**: **21 檔引用**。用 `print()` + ANSI escape codes 自行實作需上千行程式碼，且喪失 Table、Progress、Panel、Syntax 等高級元件
- **結論**: **必要，Terminal UX 核心**

### 22. PyYAML (6 檔)
- **用途**: YAML 設定檔解析（config, policy, exploit orchestrator）
- **為何不能替代**: 標準庫沒有 YAML 解析器。唯一替代是 `ruamel.yaml`（更重量）。設定檔已是 YAML 格式，不可能為了移除一個依賴而把所有設定改成 JSON
- **結論**: **必要，標準庫無 YAML 支援**

### 23. jinja2 (1 檔)
- **用途**: 報告模板引擎（matrix_visualizer.py）
- **為何不能替代**: 可以用 `str.format()` 或 f-string，但會喪失：迴圈、條件判斷、繼承等模板功能。Jinja2 極輕量且是 Python 模板的事實標準
- **結論**: **保留 — 極輕量，自行實作模板引擎不值得**

### 24. structlog (6 檔)
- **用途**: 結構化日誌（post-exploitation、authz 模組）
- **為何不能替代**: 標準 `logging` 模組輸出純文字。`structlog` 提供 JSON 結構化輸出、context binding、processor pipeline，這些是生產環境日誌分析（ELK/Datadog）的基礎
- **結論**: **必要，生產環境可觀測性需求**

### 25. psutil (5 檔)
- **用途**: 系統監控（記憶體、CPU、進程資訊）
- **為何不能替代**: 可以讀 `/proc/` 但只在 Linux 有效。`psutil` 跨平台且提供統一 API（memory_info、cpu_percent、process iteration）。自行讀 `/proc/` 需處理大量邊界情況
- **結論**: **必要，跨平台系統監控的唯一選擇**

### 26. tenacity (1 檔)
- **用途**: 重試機制（`app.py` 中的 API 啟動重試）
- **為何不能替代**: 可以自行寫 retry loop（~20 行），但 tenacity 提供 exponential backoff、retry condition、stop condition 的宣告式 API
- **替代方案**: 自行實作一個 `retry()` decorator（約 20-30 行）
- **結論**: **保留 — 代碼量極少，但移除也是可行的（最低優先級）**

---

## AI Optional 依賴（7 個）

### torch (8 檔)
- **為何不能替代**: 5M 參數的生物神經網路 + RL 模型用 PyTorch 實作。TensorFlow 是替代品但遷移成本巨大。自行寫神經網路框架完全不現實
- **結論**: **必要**

### numpy (14 檔)
- **為何不能替代**: 14 檔使用，Python 數值計算的事實標準。標準庫的 `array` 模組功能差距巨大
- **結論**: **必要**

### scikit-learn (1 檔)
- **為何不能替代**: model_trainer.py 使用 sklearn 的 ML pipeline。可以用 torch 替代但需大量重寫
- **結論**: **保留**

### transformers (2 檔)
- **為何不能替代**: Hugging Face 模型載入、tokenizer。自行實作 = 重寫 Hugging Face Hub 客戶端
- **結論**: **必要**

### networkx (1 檔)
- **為何不能替代**: skill graph 路徑分析。可以自行實作圖演算法但 networkx 提供了完整的圖論 API
- **結論**: **保留 — 自行實作 Dijkstra/BFS 可行但不值得**

### plotly (1 檔)
- **為何不能替代**: 互動式權限矩陣視覺化。matplotlib 可替代但不支援互動
- **結論**: **保留**

### pandas (4 檔)
- **為何不能替代**: dashboard 資料處理 + 表格操作。可用 dict/list 替代但代碼量會暴增
- **結論**: **保留**

---

## 總結

| 分類 | 數量 | 說明 |
|------|------|------|
| 完全不可替代 | 16 | fastapi, pydantic, httpx, grpcio, cryptography, torch 等 |
| 保留（替代成本 > 收益）| 9 | click, aiofiles, tenacity, jinja2, PyJWT 等 |
| 待確認 | 1 | redis（零 import，需確認基礎設施層面是否使用）|
| 中期遷移目標 | 1 | requests → httpx（15 檔需逐步遷移）|

**結論**: 當前 26+7 個依賴均有明確理由保留，無法以合理的工程成本進一步精簡。
