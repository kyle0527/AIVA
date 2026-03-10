# AIVA 外部模組執行說明

## 📑 目錄

- [✅ 正確理解](#-正確理解)
- [📚 詳細說明](#-詳細說明)
  - [方式 1: CommandHandler（主要）](#方式-1-commandhandler主要)
  - [方式 2: Direct Import（CLI實施方式）](#方式-2-direct-importcli實施方式)
  - [方式 3: Worker模式（背景服務）](#方式-3-worker模式背景服務)
- [📊 當前可用的功能](#-當前可用的功能)
  - [內部模組執行（✅ 可用）](#內部模組執行-可用)
  - [外部模組執行（❌ 需要修正）](#外部模組執行-需要修正)
  - [外部模組執行（✅ 3種方式可用）](#外部模組執行-3種方式可用)
- [🔧 執行範例](#-執行範例)
  - [範例 1: 使用 CommandHandler](#範例-1-使用-commandhandler)
  - [範例 2: 使用 Direct Import（CLI實施方式）](#範例-2-使用-direct-importcli實施方式)
  - [範例 3: 命令行快速測試](#範例-3-命令行快速測試)
  - [選項 3: 使用現有的測試腳本](#選項-3-使用現有的測試腳本)
- [🎯 針對 Juice Shop 靶場測試](#-針對-juice-shop-靶場測試)
  - [當前可測試的方式](#當前可測試的方式)
- [📝 總結](#-總結)

---


## ✅ 正確理解

外部功能模組有**3種執行方式**：

1. **CommandHandler（主要執行路徑）**
   - async 執行
   - 通過 RabbitMQ 或直接調用
   - 這是設計的主要方式

2. **Worker模式（背景服務）**
   - 持續運行的背景服務
   - 監聽 RabbitMQ 消息隊列
   - 適合長時間運行的任務

3. **Direct Import（CLI實施方式）**
   - 直接導入 detector 類別
   - 不需要 __main__.py
   - 適合快速測試和腳本集成

## 📚 詳細說明

### 方式 1: CommandHandler（主要）

```python
# 直接調用 CommandHandler
from services.features.function_xss.command_handler import XSSCommandHandler
import asyncio

handler = XSSCommandHandler()
result = asyncio.run(handler.handle_command({
    'target': 'http://localhost:3000',
    'scan_type': 'full'
}))
```

**特點**:
- 異步執行
- 主要執行路徑
- 可以集成到系統中

### 方式 2: Direct Import（CLI實施方式）

```python
# 直接導入使用 - 不需要 __main__.py
from services.features.function_xss.detector import XSSDetector

detector = XSSDetector()
result = detector.detect('http://localhost:3000')
print(result)
```

**特點**:
- 同步執行
- 快速測試
- 命令行友好
- 不需要額外的 CLI 接口層

### 方式 3: Worker模式（背景服務）

```bash
# 啟動 Worker
python -m services.features.function_xss.worker.xss_worker
```

然後通過 RabbitMQ 發送任務：

```python
import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

task = {
    "task_id": "test-001",
    "target_url": "http://localhost:3000",
    "scan_type": "xss"
}

channel.basic_publish(
    exchange='',
    routing_key='xss_tasks',
    body=json.dumps(task)
)
```

## 📊 當前可用的功能

### 內部模組執行（✅ 可用）

這些**可以**直接通過 CLI 執行：

```bash
# 執行內部 AI Flow
.\執行Flow.bat 11

# 啟動能力選單
.\啟動能力選單.bat

# 預覽 Flow
.\預覽Flow.bat 11
```

### 外部模組執行（❌ 需要修正）

這些**不能**直接通過 CLI 執行，需要 Docker：

- function_sqli - SQL 注入檢測
- function_xss - XSS 漏洞檢測
- function_ssrf - SSRF 漏洞檢測
- function_idor - IDOR 漏洞檢測

### 外部模組執行（✅ 3種方式可用）

這些模組可以通過3種方式執行：

- function_sqli - SQL 注入檢測
  - ✅ CommandHandler: SQLiCommandHandler
  - ✅ Direct Import: from function_sqli.detector import SQLiDetector
  - ⚠️ Worker模式: 需要 RabbitMQ

- function_xss - XSS 漏洞檢測
  - ✅ CommandHandler: XSSCommandHandler
  - ✅ Direct Import: from function_xss.detector import XSSDetector
  - ⚠️ Worker模式: 需要 RabbitMQ

- function_ssrf - SSRF 漏洞檢測
  - ✅ CommandHandler: SSRFCommandHandler
  - ✅ Direct Import: from function_ssrf.detector import SSRFDetector
  - ⚠️ Worker模式: 需要 RabbitMQ

- function_idor - IDOR 漏洞檢測
  - ✅ CommandHandler: IDORCommandHandler
  - ✅ Direct Import: from function_idor.detector import IDORDetector
  - ⚠️ Worker模式: 需要 RabbitMQ

## 🔧 執行範例

### 範例 1: 使用 CommandHandler

```python
from services.features.function_xss.command_handler import XSSCommandHandler
import asyncio

async def main():
    handler = XSSCommandHandler()
    result = await handler.handle_command({
        'target': 'http://localhost:3000',
        'scan_type': 'full'
    })
    print(result)

asyncio.run(main())
```

### 範例 2: 使用 Direct Import（CLI實施方式）

```python
from services.features.function_xss.detector import XSSDetector

detector = XSSDetector()
result = detector.detect('http://localhost:3000')
print(result)
```

### 範例 3: 命令行快速測試

```bash
python -c "from services.features.function_xss.detector import XSSDetector; \
detector = XSSDetector(); \
print(detector.detect('http://localhost:3000'))"
```
    
    task = {
        "target_url": target_url,
        "module": module
    }
    
    channel.basic_publish(
        exchange='',
        routing_key=f'{module}_tasks',
        body=json.dumps(task)
    )
    
    print(f"✅ 任務已提交: {module} -> {target_url}")

if __name__ == "__main__":
    submit_task(sys.argv[1], sys.argv[2])
```

### 選項 3: 使用現有的測試腳本

檢查模組是否有自己的測試腳本：

```bash
# 檢查 function_crypto
cd services/features/features_ready/function_crypto/rust_core
cargo run -- --help

# 檢查 function_authn_go  
cd services/features/features_in_development/function_authn_go
ls *.py  # 查找測試腳本
```

## 🎯 針對 Juice Shop 靶場測試

### 當前可測試的方式

1. **使用 TypeScript 掃描引擎**（最接近直接 CLI）：

```bash
cd C:\D\fold7\AIVA-git\services\scan\typescript_engine
npm install
npm run start:dev
# 然後通過 API 提交任務
```

2. **使用內部模組分析**（不是攻擊測試）：

```bash
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_external_module_batch_classifier.py -w ../../../features/features_ready -o ./juice_shop_analysis -v
```

## 📝 總結

**核心問題**：
- 外部功能模組（function_*）是微服務，不能直接 CLI 執行
- 需要通過 Docker + RabbitMQ 架構運行
- 我創建的 .bat 檔案引用了錯誤的腳本

**解決方案**：
1. 使用 Docker Compose 啟動外部模組
2. 或者為每個模組創建專門的測試腳本
3. 或者修改架構支持直接 CLI 調用（需要大量修改）

**當前可用**：
- ✅ 內部模組 CLI（執行Flow.bat 等）正常工作
- ✅ 分類器工具正常工作
- ❌ 外部模組 CLI 需要修正架構
