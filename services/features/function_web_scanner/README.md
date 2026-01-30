# function_web_scanner - Web 應用綜合掃描器

> **版本**: v1.3.0 | **狀態**: ✅ 完成 | **語言**: Python | **更新**: 2026-01-20

## 🎯 模組概述

綜合性 Web 應用掃描器，整合子域名發現、目錄爆破、技術棧識別等多種偵察技術，為後續漏洞檢測提供完整的攻擊面。

### 核心功能

✅ **子域名掃描** - 多種方法發現子域名（DNS查詢、證書透明度、搜索引擎）  
✅ **目錄爆破** - 智能字典生成、自適應路徑發現  
✅ **技術棧識別** - Web 框架、CMS、服務器、WAF 檢測  
✅ **端口掃描** - 快速服務識別、版本檢測  
✅ **爬蟲引擎** - 深度爬取、JavaScript 渲染支持  
✅ **API 端點發現** - REST/GraphQL/WebSocket 端點識別  

## 📐 架構設計

```
function_web_scanner/
├── integration_tools/
│   └── web_tools.py             # 核心掃描引擎
├── engines/
│   ├── subdomain_scanner.py     # 子域名掃描引擎
│   ├── directory_bruteforcer.py # 目錄爆破引擎
│   ├── tech_detector.py         # 技術棧識別引擎
│   ├── port_scanner.py          # 端口掃描引擎
│   └── crawler_engine.py        # 爬蟲引擎
├── wordlists/
│   ├── common_directories.txt   # 常見目錄字典
│   ├── api_endpoints.txt        # API 端點字典
│   └── tech_fingerprints.json   # 技術指紋庫
├── scanner_manager.py           # 管理接口（已廢棄）
└── README.md
```

## 🚀 快速開始

### 基本使用

```python
from services.features.function_web_scanner.integration_tools.web_tools import WebAttackManager

# 創建掃描器
manager = WebAttackManager()

# 綜合掃描
result = await manager.comprehensive_scan(
    target="https://example.com",
    options={
        "subdomain_scan": True,
        "directory_scan": True,
        "tech_detect": True,
        "port_scan": True
    }
)

print(f"發現 {len(result['subdomains'])} 個子域名")
print(f"發現 {len(result['directories'])} 個目錄")
print(f"識別 {len(result['technologies'])} 個技術")
```

### CLI 調用

```bash
# external_executor 調用
python aiva_external_executor.py \
    --lang python \
    --module web_scanner \
    --target https://example.com \
    --comprehensive
```

## 🔧 功能詳解

### 1. 子域名掃描

多種方法發現目標子域名：

**被動掃描**:
- 證書透明度日誌 (crt.sh)
- DNS 數據庫查詢 (DNSDumpster)
- 搜索引擎爬取 (Google, Bing)
- 第三方 API (VirusTotal, SecurityTrails)

**主動掃描**:
- DNS 暴力破解
- DNS 區域傳輸測試
- DNSSEC 枚舉

```python
# 子域名掃描
result = await manager.scan_subdomains(
    domain="example.com",
    options={
        "passive": True,      # 被動掃描
        "active": True,       # 主動掃描
        "wordlist": "subdomains-top1million.txt",
        "recursive": True,    # 遞歸掃描發現的子域名
        "dns_server": "8.8.8.8"
    }
)

for subdomain in result['subdomains']:
    print(f"{subdomain['name']} -> {subdomain['ip']}")
```

### 2. 目錄爆破

智能目錄和文件發現：

**特性**:
- 自適應字典（根據已發現內容調整）
- 模糊匹配（404 頁面檢測）
- 遞歸掃描（發現目錄後深入掃描）
- 狀態碼分析（200/301/302/403/500）
- 響應大小過濾（識別真實內容）

```python
# 目錄爆破
result = await manager.scan_directories(
    target="https://example.com",
    options={
        "wordlist": "common.txt",
        "extensions": [".php", ".jsp", ".asp"],
        "recursive": True,
        "recursive_depth": 2,
        "threads": 20,
        "timeout": 10
    }
)

for directory in result['found']:
    print(f"{directory['path']} [{directory['status_code']}]")
```

### 3. 技術棧識別

識別目標使用的技術和服務：

**檢測內容**:
- **Web 服務器**: Apache, Nginx, IIS, Tomcat
- **Web 框架**: Django, Flask, Laravel, Spring Boot
- **CMS**: WordPress, Drupal, Joomla
- **編程語言**: PHP, Python, Ruby, Java, .NET
- **JavaScript 框架**: React, Vue, Angular
- **WAF**: Cloudflare, AWS WAF, Akamai
- **CDN**: Cloudflare, Fastly, Akamai

```python
# 技術棧識別
result = await manager.detect_technologies(
    target="https://example.com",
    options={
        "detailed": True,      # 詳細檢測
        "version_detect": True # 版本識別
    }
)

for tech in result['technologies']:
    print(f"{tech['name']} {tech['version']}")
    print(f"  置信度: {tech['confidence']}%")
    print(f"  已知漏洞: {len(tech['known_cves'])}")
```

### 4. 端口掃描

快速服務識別和版本檢測：

**掃描模式**:
- **快速掃描**: Top 100 常用端口
- **標準掃描**: Top 1000 端口
- **全面掃描**: 所有 65535 端口
- **服務掃描**: 識別服務類型和版本

```python
# 端口掃描
result = await manager.scan_ports(
    target="example.com",
    options={
        "mode": "standard",    # fast/standard/full
        "service_detection": True,
        "version_detection": True,
        "script_scan": False   # Nmap 腳本掃描
    }
)

for port in result['open_ports']:
    print(f"{port['number']}/tcp {port['service']} {port['version']}")
```

### 5. 爬蟲引擎

深度爬取目標網站：

**特性**:
- JavaScript 渲染支持（Playwright）
- 表單參數提取
- API 端點發現
- 鏈接關係圖
- 敏感路徑識別

```python
# 網站爬蟲
result = await manager.crawl_website(
    target="https://example.com",
    options={
        "max_depth": 3,
        "max_pages": 500,
        "js_rendering": True,
        "extract_forms": True,
        "extract_apis": True,
        "follow_external": False
    }
)

print(f"爬取 {result['total_pages']} 個頁面")
print(f"發現 {len(result['forms'])} 個表單")
print(f"發現 {len(result['api_endpoints'])} 個 API 端點")
```

## 📊 輸出格式

```json
{
  "success": true,
  "target": "https://example.com",
  "scan_time": "45.2s",
  "results": {
    "subdomains": [
      {
        "name": "api.example.com",
        "ip": "192.0.2.1",
        "source": "crt.sh",
        "status": "active"
      }
    ],
    "directories": [
      {
        "path": "/admin",
        "status_code": 403,
        "size": 1024,
        "redirect": null
      }
    ],
    "technologies": [
      {
        "name": "Nginx",
        "version": "1.21.0",
        "category": "web_server",
        "confidence": 100,
        "known_cves": ["CVE-2021-23017"]
      }
    ],
    "open_ports": [
      {
        "number": 443,
        "service": "https",
        "version": "OpenSSL 1.1.1"
      }
    ]
  },
  "summary": {
    "total_subdomains": 12,
    "total_directories": 45,
    "total_technologies": 8,
    "total_open_ports": 3
  }
}
```

## 🎯 掃描策略

### 快速掃描（Bug Bounty 初步偵察）

```python
result = await manager.quick_scan(
    target="https://example.com",
    options={
        "timeout": 300  # 5分鐘快速掃描
    }
)
# 包含: 子域名、常見目錄、技術棧
```

### 深度掃描（完整攻擊面映射）

```python
result = await manager.deep_scan(
    target="https://example.com",
    options={
        "timeout": 3600  # 1小時深度掃描
    }
)
# 包含: 所有功能 + 遞歸掃描 + JS 渲染
```

### 隱蔽掃描（規避 WAF/IDS）

```python
result = await manager.stealth_scan(
    target="https://example.com",
    options={
        "rate_limit": 5,      # 每秒5個請求
        "user_agent_rotate": True,
        "proxy": "socks5://127.0.0.1:1080"
    }
)
```

## 🛡️ 防護規避

- **速率限制**: 智能延遲避免觸發 WAF
- **UA 輪換**: 模擬不同瀏覽器
- **代理支持**: HTTP/SOCKS5 代理鏈
- **Headers 偽裝**: 隨機化請求頭
- **錯誤重試**: 自動處理 429/503

## 🎯 適用場景

✅ **Bug Bounty** - 初步偵察和攻擊面映射  
✅ **滲透測試** - 信息收集階段  
✅ **安全審計** - 暴露面分析  
✅ **Red Team** - 外部偵察  

## 📈 性能優化

- ⚡ **異步並發**: 同時處理多個目標
- 🎯 **智能去重**: 避免重複掃描
- 💾 **結果緩存**: 加速重複查詢
- 🔄 **斷點續掃**: 支持暫停/恢復

## 🔗 整合工具

內建整合：
- **Amass** - 子域名發現
- **Subfinder** - 被動子域名掃描
- **FFuf** - 目錄爆破
- **Wappalyzer** - 技術棧識別
- **Nmap** - 端口掃描
- **Playwright** - JavaScript 渲染

## 📝 更新日誌

### v1.3.0 (2026-01-20)
- ✅ 創建完整 README 文檔
- ✅ 移除 scanner_manager.py 統一包裝層
- ✅ 完善子域名掃描引擎
- ✅ 完善目錄爆破引擎
- ✅ 完善技術棧識別引擎
- ✅ 添加端口掃描功能
- ✅ 添加爬蟲引擎

### v1.2.0 (2025-12-17)
- ✅ WebScannerManager 架構完成
- ⚠️ 缺少 README

---

**維護者**: AIVA Team | **授權**: MIT License
