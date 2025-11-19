# AIVA Scan 模組使用者手冊

> **版本**: v1.0  
> **最後更新**: 2025年11月18日  
> **適用對象**: AIVA 系統管理員、安全測試人員

---

## 📋 目錄

1. [快速開始](#快速開始)
2. [架構概覽](#架構概覽)
3. [兩階段掃描流程](#兩階段掃描流程)
4. [啟動掃描服務](#啟動掃描服務)
5. [發送掃描任務](#發送掃描任務)
6. [監控掃描進度](#監控掃描進度)
7. [查看掃描結果](#查看掃描結果)
8. [故障排除](#故障排除)
9. [進階配置](#進階配置)

---

## 🚀 快速開始

### 前置要求

```bash
# 1. 確認環境
✅ Python 3.11+
✅ Docker 和 Docker Compose
✅ RabbitMQ (通過 Docker)
✅ 虛擬環境已激活

# 2. 檢查服務狀態
docker ps | grep -E "rabbitmq|juice-shop|webgoat"

# 3. 確認 RabbitMQ 可訪問
curl http://localhost:15672  # 管理界面
# 預設帳號: aiva / aiva_mq_password
```

### 30 秒快速測試

```bash
# 1. 進入專案目錄
cd C:\D\fold7\AIVA-git

# 2. 激活虛擬環境
.venv\Scripts\Activate.ps1

# 3. 啟動 Rust Worker (Phase0)
python -m services.scan.engines.rust_engine.worker

# 4. 另開終端，啟動 Python Worker (Phase1)
python -m services.scan.engines.python_engine.worker

# 5. 第三個終端，發送測試任務
python services/scan/engines/python_engine/worker.py --test-phase0
```

---

## 🏗️ 架構概覽

### 核心組件

```
┌─────────────────────────────────────────────────────────────┐
│                        AIVA 系統                             │
│                                                              │
│  ┌────────────┐         ┌──────────────┐                   │
│  │ Core 模組  │────────▶│  RabbitMQ    │                   │
│  │ (指揮中心)  │◀────────│  (消息隊列)   │                   │
│  └────────────┘         └──────────────┘                   │
│                               │                              │
│                               ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Scan 模組 (執行單元)                      │  │
│  │                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐                  │  │
│  │  │ Rust Engine  │  │Python Engine │                  │  │
│  │  │  (Phase0)    │  │  (Phase1)    │                  │  │
│  │  └──────────────┘  └──────────────┘                  │  │
│  │                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐                  │  │
│  │  │TypeScript    │  │  Go Engine   │                  │  │
│  │  │  Engine      │  │  (選用)      │                  │  │
│  │  └──────────────┘  └──────────────┘                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 數據流向

```
用戶 → Core → MQ → Scan → MQ → Core → 後續處理
         ↓                    ↑
    tasks.scan.phase0   scan.phase0.completed
         ↓                    ↑
    tasks.scan.phase1   scan.completed
```

---

## 🎯 兩階段掃描流程

### Phase 0: 快速偵察 (5-10 分鐘)

**目標**: 快速獲取目標的基本資訊

**執行內容**:
- ✅ 目標可達性檢測
- ✅ 技術棧指紋識別 (Web Server, Framework, CMS)
- ✅ 敏感資訊掃描 (API Keys, Passwords, Tokens)
- ✅ 基礎端點發現 (深度 1，最多 50 個 URL)
- ✅ 初步攻擊面評估

**使用引擎**: Rust (高性能)

**輸出數據**:
```json
{
  "scan_id": "scan_abc123",
  "status": "success",
  "execution_time": 450.5,
  "assets": [
    {
      "asset_id": "asset_001",
      "type": "url",
      "value": "https://example.com/api/users",
      "has_form": false
    }
  ],
  "fingerprints": {
    "web_server": {"nginx": "1.21.0"},
    "frameworks": {"react": "18.2.0"},
    "technologies": ["JavaScript", "REST API"]
  },
  "summary": {
    "urls_found": 45,
    "forms_found": 3,
    "apis_found": 8
  }
}
```

### Phase 1: 深度掃描 (10-30 分鐘，按需)

**觸發條件** (Core 模組 AI 決策):
- 發現大量 JavaScript (使用 TypeScript 引擎)
- 發現 HTML 表單 (使用 Python 引擎)
- 發現 REST API (使用 Python 引擎)
- 需要高並發掃描 (使用 Go 引擎)

**執行內容**:
- ✅ 深度爬取 (深度 3-5)
- ✅ 動態內容渲染 (SPA, React, Vue)
- ✅ 表單參數提取
- ✅ API 端點深度分析
- ✅ 入口點完整發現

**使用引擎**: Python, TypeScript, Go, Rust (組合使用)

**輸出數據**:
```json
{
  "scan_id": "scan_abc123",
  "status": "success",
  "execution_time": 1250.8,
  "assets": [
    {
      "asset_id": "asset_100",
      "type": "form",
      "value": "https://example.com/login",
      "parameters": ["username", "password", "csrf_token"],
      "has_form": true
    }
  ],
  "engine_results": {
    "python": {"status": "completed", "findings": 120},
    "typescript": {"status": "completed", "findings": 85}
  },
  "phase0_summary": {
    "urls": 45,
    "execution_time": 450.5
  }
}
```

---

## 🚀 啟動掃描服務

### 方法 1: 手動啟動 Workers (開發/測試)

#### 啟動 Rust Worker (Phase0)

```bash
# 終端 1
cd C:\D\fold7\AIVA-git
.venv\Scripts\Activate.ps1

# 啟動 Rust Worker
python -m services.scan.engines.rust_engine.worker

# 預期輸出:
# [INFO] Rust Worker started
# [INFO] Subscribing to: tasks.scan.phase0
# [INFO] Worker ready, waiting for tasks...
```

#### 啟動 Python Worker (Phase1)

```bash
# 終端 2
cd C:\D\fold7\AIVA-git
.venv\Scripts\Activate.ps1

# 啟動 Python Worker
python -m services.scan.engines.python_engine.worker

# 預期輸出:
# [INFO] Python Worker started
# [INFO] Subscribing to: tasks.scan.phase0, tasks.scan.phase1, tasks.scan.start
# [INFO] Worker ready, waiting for tasks...
```

#### 啟動 TypeScript Worker (選用)

```bash
# 終端 3
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine
npm install
npm start

# 預期輸出:
# TypeScript Worker started
# Subscribing to: tasks.scan.phase1
```

#### 啟動 Go Worker (選用)

```bash
# 終端 4
cd C:\D\fold7\AIVA-git\services\scan\engines\go_engine
go run worker.go

# 預期輸出:
# Go Worker started
# Subscribing to: tasks.scan.phase1
```

### 方法 2: Docker Compose 啟動 (生產環境)

```bash
# 啟動所有 Scan Workers
cd C:\D\fold7\AIVA-git
docker-compose up -d aiva-rust-worker aiva-python-worker

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f aiva-rust-worker
docker-compose logs -f aiva-python-worker
```

### 驗證服務啟動

```bash
# 1. 檢查 RabbitMQ 連接
# 訪問 http://localhost:15672
# 登入: aiva / aiva_mq_password
# 查看 Queues → 應該看到:
#   - tasks.scan.phase0 (1 consumer)
#   - tasks.scan.phase1 (1+ consumer)

# 2. 檢查 Worker 日誌
# 應該看到 "Worker ready" 訊息

# 3. 測試健康狀態
curl http://localhost:8000/health  # 如果有健康檢查端點
```

---

## 📤 發送掃描任務

### 使用 Core 模組接口 (推薦)

Core 模組會自動處理兩階段掃描流程。

```python
# 方法 1: 通過 Core 的兩階段掃描器
from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import TwoPhaseScanOrchestrator
from services.aiva_common.mq import RabbitBroker
import asyncio

async def scan_targets():
    # 初始化
    broker = RabbitBroker("amqp://aiva:aiva_mq_password@localhost:5672/aiva")
    await broker.connect()
    
    orchestrator = TwoPhaseScanOrchestrator(broker)
    
    # 執行兩階段掃描
    result = await orchestrator.execute_two_phase_scan(
        targets=["http://localhost:3000"],  # Juice Shop
        trace_id="test-001"
    )
    
    print(f"掃描完成: {result.scan_id}")
    print(f"狀態: {result.status}")
    print(f"總資產: {len(result.phase1_result.assets)}")
    
    await broker.close()

# 執行
asyncio.run(scan_targets())
```

### 方法 2: 直接發送 MQ 消息 (進階)

```python
import pika
import json
import uuid

# 連接 RabbitMQ
connection = pika.BlockingConnection(
    pika.URLParameters("amqp://aiva:aiva_mq_password@localhost:5672/aiva")
)
channel = connection.channel()

# 發送 Phase0 命令
phase0_message = {
    "trace_id": "manual-test-001",
    "correlation_id": str(uuid.uuid4()),
    "payload": {
        "scan_id": f"scan_{uuid.uuid4().hex[:8]}",
        "targets": ["http://localhost:3000"],
        "timeout_seconds": 600
    }
}

channel.basic_publish(
    exchange='',
    routing_key='tasks.scan.phase0',
    body=json.dumps(phase0_message).encode('utf-8'),
    properties=pika.BasicProperties(
        delivery_mode=2,  # 持久化
        content_type='application/json'
    )
)

print(f"✅ Phase0 任務已發送: {phase0_message['payload']['scan_id']}")
connection.close()
```

### 方法 3: 使用測試腳本

```bash
# 使用內建測試功能
cd C:\D\fold7\AIVA-git

# 測試 Phase0
python -c "
from services.scan.engines.rust_engine.worker import test_phase0_scan
import asyncio
asyncio.run(test_phase0_scan('http://localhost:3000'))
"

# 測試 Phase1
python -c "
from services.scan.engines.python_engine.worker import test_phase1_scan
import asyncio
asyncio.run(test_phase1_scan('http://localhost:3000'))
"
```

---

## 📊 監控掃描進度

### 1. RabbitMQ 管理界面

```bash
# 訪問 http://localhost:15672
# 登入: aiva / aiva_mq_password

# 監控重點:
# - Queues → 查看隊列長度
# - Connections → 查看 Worker 連接狀態
# - Channels → 查看消息流動
```

**關鍵指標**:
- `tasks.scan.phase0`: 待執行的 Phase0 任務
- `tasks.scan.phase1`: 待執行的 Phase1 任務
- `scan.phase0.completed`: Phase0 結果隊列
- `scan.completed`: 最終結果隊列

### 2. Worker 日誌

```bash
# 實時查看 Rust Worker 日誌
docker logs -f aiva-rust-worker

# 實時查看 Python Worker 日誌
docker logs -f aiva-python-worker

# 查看最近 100 行
docker logs --tail 100 aiva-rust-worker
```

**日誌關鍵字**:
- `[Phase0] Starting scan`: Phase0 開始
- `[Phase0] Completed`: Phase0 完成
- `[Phase1] Starting scan`: Phase1 開始
- `[Phase1] Completed`: Phase1 完成
- `[ERROR]`: 錯誤信息

### 3. 查詢掃描狀態 (API)

```python
# 查詢特定掃描的狀態
import requests

scan_id = "scan_abc123"
response = requests.get(f"http://localhost:8000/api/scans/{scan_id}")

if response.status_code == 200:
    data = response.json()
    print(f"狀態: {data['status']}")
    print(f"進度: {data['progress']}%")
    print(f"當前階段: {data['current_phase']}")
```

### 4. 終端監控腳本

```bash
# 創建監控腳本
cat > monitor_scan.sh << 'EOF'
#!/bin/bash
SCAN_ID=$1
while true; do
    clear
    echo "=== Scan Monitor: $SCAN_ID ==="
    echo ""
    
    # RabbitMQ 隊列狀態
    echo "📊 Queue Status:"
    curl -s -u aiva:aiva_mq_password http://localhost:15672/api/queues/%2Faiva/tasks.scan.phase0 | jq '.messages'
    
    # Worker 狀態
    echo ""
    echo "👷 Workers:"
    docker ps --filter "name=aiva-.*-worker" --format "table {{.Names}}\t{{.Status}}"
    
    sleep 5
done
EOF

chmod +x monitor_scan.sh
./monitor_scan.sh scan_abc123
```

---

## 📋 查看掃描結果

### 結果數據結構

```python
# Phase0 結果
{
    "scan_id": "scan_abc123",
    "status": "success",
    "execution_time": 450.5,
    "summary": {
        "urls_found": 45,
        "forms_found": 3,
        "apis_found": 8,
        "scan_duration_seconds": 450
    },
    "fingerprints": {
        "web_server": {"nginx": "1.21.0"},
        "frameworks": {"express": "4.18.2"},
        "cms": {},
        "technologies": ["JavaScript", "Node.js"]
    },
    "assets": [
        {
            "asset_id": "asset_001",
            "type": "url",
            "value": "https://example.com/api/users",
            "parameters": null,
            "has_form": false
        }
    ],
    "recommendations": {
        "needs_phase1": true,
        "suggested_engines": ["python", "typescript"],
        "reason": "檢測到 JavaScript 框架和 API 端點"
    }
}

# Phase1 結果
{
    "scan_id": "scan_abc123",
    "status": "success",
    "execution_time": 1250.8,
    "summary": {
        "urls_found": 234,
        "forms_found": 12,
        "apis_found": 45,
        "scan_duration_seconds": 1250
    },
    "assets": [
        {
            "asset_id": "asset_100",
            "type": "form",
            "value": "https://example.com/login",
            "parameters": ["username", "password", "csrf_token"],
            "has_form": true
        }
    ],
    "engine_results": {
        "python": {
            "status": "completed",
            "findings": 120,
            "execution_time": 800.5
        },
        "typescript": {
            "status": "completed",
            "findings": 85,
            "execution_time": 900.2
        }
    },
    "phase0_summary": {
        "urls": 45,
        "execution_time": 450.5
    }
}
```

### 從 RabbitMQ 消費結果

```python
import pika
import json

# 連接
connection = pika.BlockingConnection(
    pika.URLParameters("amqp://aiva:aiva_mq_password@localhost:5672/aiva")
)
channel = connection.channel()

# 消費 Phase0 結果
def on_phase0_result(ch, method, properties, body):
    result = json.loads(body)
    print(f"📥 Phase0 結果: {result['payload']['scan_id']}")
    print(f"   狀態: {result['payload']['status']}")
    print(f"   資產數: {len(result['payload']['assets'])}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(
    queue='scan.phase0.completed',
    on_message_callback=on_phase0_result
)

print('⏳ 等待 Phase0 結果...')
channel.start_consuming()
```

### 結果匯出

```python
# 匯出為 JSON
import json

with open(f"scan_result_{scan_id}.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# 匯出為 CSV (資產清單)
import csv

with open(f"assets_{scan_id}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["asset_id", "type", "value", "has_form"])
    writer.writeheader()
    for asset in result["assets"]:
        writer.writerow({
            "asset_id": asset["asset_id"],
            "type": asset["type"],
            "value": asset["value"],
            "has_form": asset["has_form"]
        })

# 匯出為 Markdown 報告
report = f"""
# 掃描報告

**掃描 ID**: {result['scan_id']}
**狀態**: {result['status']}
**執行時間**: {result['execution_time']:.2f} 秒

## 摘要

- URLs: {result['summary']['urls_found']}
- 表單: {result['summary']['forms_found']}
- APIs: {result['summary']['apis_found']}

## 資產清單

| ID | 類型 | 值 |
|----|------|-----|
"""

for asset in result["assets"][:10]:  # 前 10 個
    report += f"| {asset['asset_id']} | {asset['type']} | {asset['value']} |\n"

with open(f"report_{scan_id}.md", "w", encoding="utf-8") as f:
    f.write(report)
```

---

## 🔧 故障排除

### 常見問題

#### 1. Worker 無法連接 RabbitMQ

**症狀**:
```
[ERROR] Failed to connect to RabbitMQ
[ERROR] Connection refused: localhost:5672
```

**解決方法**:
```bash
# 1. 檢查 RabbitMQ 是否運行
docker ps | grep rabbitmq

# 2. 檢查端口
netstat -an | findstr 5672
netstat -an | findstr 15672

# 3. 重啟 RabbitMQ
docker restart rabbitmq

# 4. 檢查帳號密碼
# 確認環境變數: RABBITMQ_URL
echo $env:RABBITMQ_URL
```

#### 2. Rust Worker 無法啟動

**症狀**:
```
[ERROR] Rust binary not found
[ERROR] rust_info_gatherer module not available
```

**解決方法**:
```bash
# 1. 編譯 Rust 引擎
cd C:\D\fold7\AIVA-git\services\scan\engines\rust_engine
cargo build --release

# 2. 檢查 Python Bridge
python -c "from services.scan.engines.rust_engine.python_bridge import rust_info_gatherer; print(rust_info_gatherer.is_available())"

# 3. 安裝依賴
pip install -r requirements.txt
```

#### 3. Phase0 超時

**症狀**:
```
[ERROR] Phase0 timeout after 600 seconds
```

**解決方法**:
```python
# 增加超時時間
orchestrator = TwoPhaseScanOrchestrator(broker)
orchestrator.phase0_timeout = 1200  # 20 分鐘

# 或修改配置
phase0_payload = Phase0StartPayload(
    scan_id=scan_id,
    targets=targets,
    timeout_seconds=1200  # 20 分鐘
)
```

#### 4. 掃描卡住不動

**檢查步驟**:
```bash
# 1. 查看 Worker 日誌
docker logs --tail 50 aiva-rust-worker
docker logs --tail 50 aiva-python-worker

# 2. 查看 RabbitMQ 隊列
# 訪問 http://localhost:15672/#/queues/%2Faiva

# 3. 檢查消息是否堆積
# 如果 Ready 數量持續增加 → Worker 處理太慢
# 如果 Unacked 數量持續增加 → Worker 處理中

# 4. 重啟 Worker
docker restart aiva-rust-worker
docker restart aiva-python-worker
```

#### 5. 記憶體不足

**症狀**:
```
[ERROR] MemoryError: Unable to allocate array
```

**解決方法**:
```bash
# 1. 限制掃描範圍
max_depth=2  # 降低深度
max_urls=500  # 限制 URL 數量

# 2. 增加 Docker 記憶體限制
docker update --memory 4g aiva-python-worker

# 3. 使用串行掃描而非並行
# 在 Phase1 中只啟用一個引擎
```

### 調試模式

```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 或設置環境變數
export LOG_LEVEL=DEBUG
python -m services.scan.engines.rust_engine.worker
```

### 健康檢查

```python
# 檢查 Worker 健康狀態
async def health_check():
    broker = RabbitBroker("amqp://aiva:aiva_mq_password@localhost:5672/aiva")
    
    try:
        await broker.connect()
        print("✅ RabbitMQ 連接正常")
        
        # 檢查隊列
        # ... (實現檢查邏輯)
        
        await broker.close()
        return True
    except Exception as e:
        print(f"❌ 健康檢查失敗: {e}")
        return False
```

---

## ⚙️ 進階配置

### 自定義掃描策略

```python
# 快速掃描策略 (適合大範圍偵察)
quick_scan = {
    "phase0": {
        "timeout": 300,  # 5 分鐘
        "max_depth": 1,
        "concurrent_requests": 50
    },
    "phase1": {
        "enabled": False  # 跳過 Phase1
    }
}

# 深度掃描策略 (適合單一目標)
deep_scan = {
    "phase0": {
        "timeout": 600,  # 10 分鐘
        "max_depth": 3,
        "concurrent_requests": 100
    },
    "phase1": {
        "enabled": True,
        "timeout": 3600,  # 60 分鐘
        "max_depth": 5,
        "max_urls": 5000,
        "engines": ["python", "typescript", "rust"]
    }
}

# 平衡掃描策略 (預設)
balanced_scan = {
    "phase0": {
        "timeout": 600,
        "max_depth": 2,
        "concurrent_requests": 100
    },
    "phase1": {
        "enabled": True,
        "timeout": 1800,
        "max_depth": 3,
        "max_urls": 1000,
        "engines": ["python", "typescript"]
    }
}
```

### 引擎優先級配置

```python
# 根據目標特徵選擇引擎
engine_selection_rules = {
    "has_javascript": ["typescript"],
    "has_forms": ["python"],
    "has_api": ["python"],
    "large_site": ["go"],
    "sensitive_scan": ["rust"]
}

# 引擎並發限制
engine_concurrency = {
    "python": 1,      # 串行執行
    "typescript": 1,  # 串行執行
    "go": 3,          # 最多 3 個並發
    "rust": 5         # 最多 5 個並發
}
```

### 效能調優

```python
# Rust Worker 效能配置
rust_config = {
    "max_concurrent_scans": 10,
    "request_timeout": 30,
    "max_retries": 3,
    "user_agent": "AIVA-Scanner/1.0"
}

# Python Worker 效能配置
python_config = {
    "max_workers": 4,
    "chunk_size": 100,
    "cache_enabled": True,
    "cache_ttl": 3600
}
```

### 多目標批次掃描

```python
async def batch_scan(targets: list[str]):
    """批次掃描多個目標"""
    results = []
    
    for i, target in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] 掃描: {target}")
        
        result = await orchestrator.execute_two_phase_scan(
            targets=[target],
            trace_id=f"batch-{i}"
        )
        
        results.append(result)
        
        # 避免過載，間隔 5 秒
        await asyncio.sleep(5)
    
    return results

# 使用範例
targets = [
    "http://localhost:3000",  # Juice Shop
    "http://localhost:8080",  # WebGoat
    "http://localhost:3001",  # Juice Shop 2
]

results = await batch_scan(targets)
```

---

## 📚 API 參考

### Phase0StartPayload

```python
from services.aiva_common.schemas import Phase0StartPayload

payload = Phase0StartPayload(
    scan_id="scan_abc123",           # 必填
    targets=["http://example.com"],  # 必填
    timeout_seconds=600,             # 選填，預設 600
    max_depth=2,                     # 選填，預設 2
    max_urls=50                      # 選填，預設 50
)
```

### Phase1StartPayload

```python
from services.aiva_common.schemas import Phase1StartPayload

payload = Phase1StartPayload(
    scan_id="scan_abc123",
    targets=["http://example.com"],
    phase0_result=phase0_result,     # Phase0 的結果
    selected_engines=["python"],     # 選用的引擎
    max_depth=3,
    max_urls=1000,
    timeout_seconds=1800
)
```

### Asset Schema

```python
from services.aiva_common.schemas import Asset

asset = Asset(
    asset_id="asset_001",
    type="url",  # url, form, api, endpoint
    value="https://example.com/api/users",
    parameters=["id", "name"],  # 選填
    has_form=False
)
```

---

## 📖 範例: 完整掃描流程

```python
#!/usr/bin/env python3
"""
完整的兩階段掃描範例
"""
import asyncio
from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import TwoPhaseScanOrchestrator
from services.aiva_common.mq import RabbitBroker

async def complete_scan_example():
    """完整掃描流程示範"""
    
    # 1. 初始化
    print("🔧 初始化...")
    broker = RabbitBroker("amqp://aiva:aiva_mq_password@localhost:5672/aiva")
    await broker.connect()
    
    orchestrator = TwoPhaseScanOrchestrator(broker)
    
    # 2. 設定目標
    targets = [
        "http://localhost:3000",  # Juice Shop
    ]
    
    print(f"🎯 目標: {targets}")
    
    # 3. 執行兩階段掃描
    print("\n🚀 開始掃描...")
    result = await orchestrator.execute_two_phase_scan(
        targets=targets,
        trace_id="example-001",
        max_depth=3,
        max_urls=1000
    )
    
    # 4. 顯示結果
    print("\n" + "="*80)
    print("✅ 掃描完成")
    print("="*80)
    
    print(f"\n📊 基本資訊:")
    print(f"  掃描 ID: {result.scan_id}")
    print(f"  狀態: {result.status}")
    print(f"  總耗時: {result.total_execution_time:.2f} 秒")
    
    if result.phase0_result:
        print(f"\n📋 Phase0 結果:")
        print(f"  執行時間: {result.phase0_result.execution_time:.2f} 秒")
        print(f"  URLs: {result.phase0_result.summary.urls_found}")
        print(f"  表單: {result.phase0_result.summary.forms_found}")
        print(f"  APIs: {result.phase0_result.summary.apis_found}")
        print(f"  資產數: {len(result.phase0_result.assets)}")
    
    if result.phase1_result:
        print(f"\n📋 Phase1 結果:")
        print(f"  執行時間: {result.phase1_result.execution_time:.2f} 秒")
        print(f"  URLs: {result.phase1_result.summary.urls_found}")
        print(f"  資產數: {len(result.phase1_result.assets)}")
        print(f"  使用引擎: {list(result.phase1_result.engine_results.keys())}")
    
    # 5. 顯示前 10 個資產
    if result.phase1_result and result.phase1_result.assets:
        print(f"\n📦 資產清單 (前 10 個):")
        for i, asset in enumerate(result.phase1_result.assets[:10], 1):
            print(f"  [{i}] {asset.type}: {asset.value}")
    
    # 6. 清理
    await broker.close()
    print("\n✅ 完成")

if __name__ == "__main__":
    asyncio.run(complete_scan_example())
```

---

## 🔗 相關文檔

- [SCAN_FLOW_DIAGRAMS.md](./SCAN_FLOW_DIAGRAMS.md) - 流程圖和架構說明
- [README.md](./README.md) - Scan 模組概覽
- [aiva_common Schema 定義](../aiva_common/schemas/) - 數據模型

---

## 📞 支援

如有問題，請檢查:
1. [故障排除](#故障排除) 章節
2. Worker 日誌輸出
3. RabbitMQ 管理界面

---

**版本歷史**:
- v1.0 (2025-11-18): 初始版本
