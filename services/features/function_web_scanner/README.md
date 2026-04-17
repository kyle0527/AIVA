# function_web_scanner - Web 應用綜合掃描器

> **版本**: v3.0.0 | **狀態**: 🔧 開發中 (85%) | **語言**: Python

## 🎯 模組概述

綜合性 Web 應用掃描器，整合子域名發現、目錄掃描、漏洞偵測、技術棧識別、端口掃描與網站爬蟲，為滲透測試的偵察階段提供完整的攻擊面映射。

### 功能清單

| 功能 | 說明 |
|------|------|
| 子域名枚舉 | crt.sh、DNS 暴力、Bing/DDG/RapidDNS 搜尋引擎 |
| 目錄爆破 | 並行掃描（20 連線），30 個常用路徑字典 |
| 漏洞偵測 | XSS/SQLi/LFI/CORS/Open Redirect/敏感檔案 |
| 技術棧識別 | HTTP 標頭、HTML 模式、Cookie、Meta tag、JS 函式庫 |
| 端口掃描 | Socket 掃描 19 個常用埠、Banner 擷取 |
| 網站爬蟲 | 廣度優先爬蟲、表單/連結/參數萃取 |

## 📐 架構設計

```
function_web_scanner/
├── __init__.py                      # 模組入口匯出
├── command_handler.py               # 核心命令處理器 (WebScannerCommandHandler)
├── integration_tools/
│   ├── __init__.py
│   └── web_tools.py                 # 全部核心實作 (WebAttackManager 等)
└── scanners/
    ├── __init__.py
    ├── subdomain_scanner.py         # SubdomainScanner
    ├── directory_bruteforcer.py     # DirectoryBruteforcer
    ├── tech_detector.py             # TechDetector
    ├── port_scanner.py              # PortScanner
    └── web_crawler.py               # WebCrawler
```

## 🚀 執行方式

### 綜合掃描（Python 匯入）

```python
import asyncio
from services.features.function_web_scanner import WebAttackManager

async def main():
    manager = WebAttackManager()

    # 綜合掃描（唯一完整入口）
    result = await manager.comprehensive_scan(
        target_url="https://example.com",
        options={
            "subdomain_scan": True,
            "directory_scan": True,
            "vulnerability_scan": True,
            "technology_scan": True,
        }
    )
    print(result)

asyncio.run(main())
```

### 單獨使用子掃描器

```python
import asyncio
from services.features.function_web_scanner import SubdomainScanner, DirectoryBruteforcer

async def main():
    # 子域名枚舉
    scanner = SubdomainScanner()
    subdomains = await scanner.enumerate("example.com")

    # 目錄掃描
    brute = DirectoryBruteforcer()
    dirs = await brute.scan("https://example.com")

asyncio.run(main())
```

## 🔧 內部實作細節

### 1. 子域名枚舉
並行執行：crt.sh JSON API、20 個常見前綴 DNS 暴力破解、搜尋引擎 (Bing, DuckDuckGo HTML, RapidDNS)。

### 2. 漏洞偵測
包含 8 類掃描，並採用基線比對減少誤報：
- Reflected XSS (驗證未編碼反射)
- SQL Injection (Error-based / Boolean-based)
- LFI / Directory Traversal
- CORS 配置錯誤 (萬用字元 + credentials 等)
- Open Redirect
- Missing Security Headers
- Sensitive File Exposure

### 3. 技術棧識別
分析 Server 標頭、WordPress/Drupal HTML 模式、React/Vue 腳本特徵、PHPSESSID Cookie 等。

## 注意事項
- 目前無速率限制 (Rate Limiting)，並行請求可能觸發 WAF/IDS 封鎖。
- 不支援 JavaScript 渲染 (Playwright 尚未實作)。
- 無外部工具整合 (ffuf, nmap, amass 尚未串接)。
