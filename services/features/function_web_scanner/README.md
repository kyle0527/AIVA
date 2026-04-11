# function_web_scanner - Web 應用綜合掃描器

> **版本**: v2.0.0 | **狀態**: 🔧 開發中 (85%) | **語言**: Python | **更新**: 2026-03-17

## 🎯 模組概述

綜合性 Web 應用掃描器，整合子域名發現、目錄掃描、漏洞偵測、技術棧識別、端口掃描與網站爬蟲，為滲透測試的偵察階段提供完整的攻擊面映射。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 子域名枚舉 | ✅ 完成 | crt.sh、DNS 暴力、Bing/DDG/RapidDNS 搜尋引擎 |
| 目錄爆破 | ✅ 完成 | 並行掃描（20 連線），30 個常用路徑字典 |
| 漏洞偵測 | ✅ 完成 | XSS/SQLi/LFI/CORS/Open Redirect/敏感檔案 |
| 技術棧識別 | ✅ 完成 | HTTP 標頭、HTML 模式、Cookie、Meta tag、JS 函式庫 |
| 端口掃描 | ✅ 完成 | Socket 掃描 19 個常用埠、Banner 擷取 |
| 網站爬蟲 | ✅ 完成 | 廣度優先爬蟲、表單/連結/參數萃取 |
| 外部工具整合 | ⏳ 待實作 | Amass/FFuf/Nmap/Wappalyzer（尚未串接） |
| 速率限制 | ⏳ 待實作 | 目前無請求間隔，可能觸發 WAF/IDS |

## 📐 實際架構

```
function_web_scanner/
├── command_handler.py               # 核心命令處理器 (WebScannerCommandHandler)
├── integration_tools/
│   ├── __init__.py                  # 匯出 WebAttackManager 等核心類別
│   └── web_tools.py                 # 全部核心實作（1,100+ 行）
├── scanners/
│   ├── __init__.py
│   ├── subdomain_scanner.py         # SubdomainScanner（基於 crt.sh + DNS）
│   ├── directory_bruteforcer.py     # DirectoryBruteforcer（5000 路徑字典）
│   ├── tech_detector.py             # TechDetector（HTTP 標頭 + HTML 模式）
│   ├── port_scanner.py              # PortScanner（Socket，19 埠）
│   └── web_crawler.py               # WebCrawler（廣度優先）
├── __init__.py                      # 模組入口
└── README.md
```

> **注意**：`wordlists/`、`engines/` 目錄**不存在**。字典硬編碼於 `web_tools.py` 與各 scanner 檔案中。

## 🚀 快速開始

### 基本使用（推薦入口）

```python
import asyncio
from services.features.function_web_scanner import WebAttackManager

async def main():
    manager = WebAttackManager()

    # 綜合掃描（唯一完整入口）
    result = await manager.comprehensive_scan(
        target_url="https://example.com",
        options={
            "subdomain_scan": True,      # 子域名枚舉
            "directory_scan": True,      # 目錄爆破
            "vulnerability_scan": True,  # 漏洞偵測
            "technology_scan": True,     # 技術棧識別
        }
    )

    print(f"子域名: {len(result['subdomains'])}")
    print(f"目錄:   {len(result['directories'])}")
    print(f"漏洞:   {len(result['vulnerabilities'])}")
    print(f"技術棧: {len(result['technologies'])}")

asyncio.run(main())
```

### 單獨使用子掃描器

```python
import asyncio
from services.features.function_web_scanner import (
    SubdomainScanner,
    DirectoryBruteforcer,
    TechDetector,
    PortScanner,
    WebCrawler,
)

async def main():
    # 子域名枚舉
    scanner = SubdomainScanner()
    subdomains = await scanner.enumerate("example.com")

    # 目錄掃描
    brute = DirectoryBruteforcer()
    dirs = await brute.scan("https://example.com")

    # 技術棧識別
    tech = TechDetector()
    techs = await tech.detect("https://example.com")

    # 端口掃描
    port = PortScanner()
    ports = await port.scan("example.com")

    # 網站爬蟲
    crawler = WebCrawler()
    links = await crawler.crawl("https://example.com", max_depth=2)

asyncio.run(main())
```

## 🔧 功能詳解

### 1. 子域名枚舉（SubdomainEnumerator）

並行執行 4 種枚舉方法（30 秒總超時）：

| 方法 | 資料來源 |
|------|---------|
| Certificate Transparency | crt.sh JSON API |
| DNS 暴力破解 | 20 個常見前綴（www/mail/api/dev/test/shop 等） |
| 搜尋引擎 | Bing、DuckDuckGo HTML、RapidDNS |
| 常見子域名 HTTP 探測 | 16 個常見子域名直接 HTTP 連線驗證 |

### 2. 目錄爆破（DirectoryScanner）

- 並行 20 連線，逾時 5 秒/請求
- 預設 30 個路徑：`admin/`、`wp-admin/`、`phpmyadmin/`、`robots.txt`、`.htaccess`、`phpinfo.php` 等
- 回報狀態碼 200 / 301 / 302 / 403 的路徑

### 3. 漏洞偵測（VulnerabilityScanner / WebVulnerabilityScanner）

並行執行 8 類漏洞掃描，含基線比對減少誤報：

| 掃描類型 | 方法 | 信心度 |
|----------|------|--------|
| Reflected XSS | 4 個 payload，驗證未編碼反射，排除 HTML 轉義 | High |
| SQL Injection (Error-based) | 10+ SQL 錯誤模式比對 | High |
| SQL Injection (Boolean-based) | 回應長度與基線差異 > 15% 且狀態碼不同 | Medium |
| Directory Traversal / LFI | 4 個路徑 + 2 個以上內容指標驗證 | High |
| CORS 配置錯誤 | 任意來源反射 / 萬用字元 + credentials / null origin | High |
| Open Redirect | 9 個常見重定向參數測試 | High |
| Missing Security Headers | HSTS/CSP/X-Frame-Options/X-Content-Type 等 6 項 | High |
| Sensitive File Exposure | `.env`/`.git/config`/`phpinfo.php`/`.htpasswd` 等 7 個路徑 | High |

> `WebVulnerabilityScanner` 為 `VulnerabilityScanner` 的別名，兩者完全相同。

### 4. 技術棧識別（TechDetector）

| 類別 | 偵測內容 |
|------|---------|
| Web 服務器 | Server 標頭（Apache/Nginx/IIS/Caddy 等） |
| 框架/CMS | WordPress/Drupal/Joomla/Laravel/Django（HTML 模式） |
| JavaScript 框架 | React/Vue/Angular（腳本名稱/標籤） |
| JavaScript 函式庫 | jQuery/Bootstrap/Lodash/Moment.js |
| CSS 框架 | Bootstrap/Foundation/Bulma |
| Cookie 特徵 | PHPSESSID → PHP；JSESSIONID → Java Servlet |
| Meta 標籤 | generator 標籤 |

### 5. 端口掃描（PortScanner）

Socket 連線掃描 19 個常用埠：`21 22 23 25 53 80 110 143 443 465 587 993 995 3306 3389 5432 6379 8080 8443`

**輸出**：每個開放埠的埠號、服務名稱、Banner 字串（若有）

### 6. 網站爬蟲（WebCrawler）

廣度優先爬蟲，萃取：
- 所有超連結（`<a href>`）
- 表單（`<form action>`、表單欄位）
- URL 中的查詢參數
- 腳本與資源路徑

## 📊 輸出格式

`comprehensive_scan()` 回傳 dict：

```json
{
  "target": "https://example.com",
  "timestamp": "2026-03-17T12:00:00",
  "subdomains": ["api.example.com", "mail.example.com"],
  "directories": [
    {
      "path": "admin/",
      "url": "https://example.com/admin/",
      "status": 403,
      "size": 0
    }
  ],
  "vulnerabilities": [
    {
      "type": "Missing Security Header",
      "severity": "Medium",
      "confidence": "High",
      "location": "https://example.com",
      "payload": "",
      "description": "Missing Strict-Transport-Security header - Enforces HTTPS",
      "recommendation": "Add Strict-Transport-Security: max-age=..."
    }
  ],
  "technologies": ["Server: nginx/1.21.0", "Framework: WordPress"],
  "scan_summary": {
    "total_subdomains": 2,
    "total_directories": 5,
    "total_vulnerabilities": 3,
    "total_technologies": 4,
    "high_severity_vulns": 1,
    "medium_severity_vulns": 2,
    "low_severity_vulns": 0
  }
}
```

## ⏳ 待實作功能

以下功能在現有 README 或代碼中曾被提及，但**尚未實作**：

- **外部工具整合**：Amass、Subfinder、FFuf、Wappalyzer、Nmap、Playwright（README 曾錯誤宣稱已整合）
- **速率限制**：目前掃描不含請求間隔，可能觸發 WAF/IDS
- **隱蔽掃描模式**（`stealth_scan()`）：不存在，切勿在程式碼中呼叫
- **快速/深度掃描捷徑**（`quick_scan()`、`deep_scan()`）：不存在，請使用 `comprehensive_scan()` 並調整 `options`
- **搜尋引擎枚舉進階**（Google dork、VirusTotal API）：需 API 金鑰
- **DNS 區域傳輸測試**
- **遞歸目錄掃描**
- **JavaScript 渲染**（Playwright）

## 🎯 適用場景

✅ **Bug Bounty** — 偵察階段攻擊面映射
✅ **滲透測試** — 資訊收集與漏洞初步探測
✅ **安全審計** — 暴露面分析
✅ **Red Team** — 外部偵察

## 🔗 相關標準

- [OWASP Testing Guide - Information Gathering](https://owasp.org/www-project-web-security-testing-guide/stable/4-Web_Application_Security_Testing/01-Information_Gathering/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [WSTG-INPV-01 (XSS)](https://owasp.org/www-project-web-security-testing-guide/stable/4-Web_Application_Security_Testing/07-Input_Validation_Testing/01-Testing_for_Reflected_Cross_Site_Scripting)
- [WSTG-INPV-05 (SQLi)](https://owasp.org/www-project-web-security-testing-guide/stable/4-Web_Application_Security_Testing/07-Input_Validation_Testing/05-Testing_for_SQL_Injection)

## 📝 更新日誌

### v2.0.0 (2026-03-17)
- ✅ 移除舊版 `BaseCapability` 和 `CapabilityRegistry` 依賴
- ✅ 新增 `WebScannerCommandHandler` 完全支援 `aiva_common` `v2.0` 命令系統
- ✅ 新增 `WebVulnerabilityScanner` 別名（修正 import 錯誤）
- ✅ 新增 `SubdomainResult`、`DirectoryScanResult` dataclass（修正 import 缺失）
- ✅ 實作 `_enumerate_search_engines()`（Bing/DuckDuckGo/RapidDNS）
- ✅ 全面重寫 `VulnerabilityScanner`：基線比對、結果去重、信心度
- ✅ 新增 CORS 配置錯誤偵測
- ✅ 新增開放重定向偵測（9 個常見參數）
- ✅ 新增敏感檔案暴露偵測（.env/.git/phpinfo 等）
- ✅ 改進 XSS 偵測（反射驗證 + HTML 轉義排除）
- ✅ 改進 SQLi 偵測（錯誤訊息模式 + Boolean 差異分析）
- ✅ 更正 README（移除不存在的功能宣稱、修正架構圖）

### v1.3.0 (2026-01-20)
- ✅ 移除 `scanner_manager.py`（廢棄）
- ✅ 完善子域名、目錄、技術棧、端口、爬蟲引擎

### v1.2.0 (2025-12-17)
- ✅ WebAttackManager 架構完成

---

**維護者**: AIVA Team | **授權**: MIT License
