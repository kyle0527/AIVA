# 測試目標驗證報告

## 📑 目錄

- [📊 目前運行的 Docker 靶場](#目前運行的-docker-靶場)
  - [完整列表](#完整列表)
  - [驗證結果](#驗證結果)
- [🎯 測試腳本使用的目標](#測試腳本使用的目標)
  - [1. `test_all_targets.py`](#1-testalltargetspy)
  - [2. `diagnose_http.py`](#2-diagnosehttppy)
  - [3. `test_dynamic_scan.py`](#3-testdynamicscanpy)
  - [4. `test_targets_detailed.py`](#4-testtargetsdetailedpy)
  - [5. `testing/scan/test_scan_integration.py`](#5-testingscantestscanintegrationpy)
- [🔍 靶場特性分析](#靶場特性分析)
  - [OWASP Juice Shop (3000, 3001, 3003)](#owasp-juice-shop-3000-3001-3003)
  - [OWASP WebGoat (8080)](#owasp-webgoat-8080)
- [📋 測試目標映射表](#測試目標映射表)
- [⚠️ 重要說明](#重要說明)
  - [為什麼靜態引擎返回 "0 個 URL" 是正確的](#為什麼靜態引擎返回-0-個-url-是正確的)
- [🎯 推薦的測試策略](#推薦的測試策略)
  - [策略 1: 快速驗證 (Python 引擎)](#策略-1-快速驗證-python-引擎)
  - [策略 2: 靜態掃描測試 (Python + Rust)](#策略-2-靜態掃描測試-python-rust)
  - [策略 3: 動態掃描測試 (Playwright)](#策略-3-動態掃描測試-playwright)
  - [策略 4: 多引擎協同測試](#策略-4-多引擎協同測試)
  - [策略 5: Go SSRF 掃描器測試](#策略-5-go-ssrf-掃描器測試)
- [✅ 安全確認](#安全確認)
  - [這些目標可以安全測試嗎？](#這些目標可以安全測試嗎)
- [📝 測試清單](#測試清單)
- [🎉 結論](#結論)

---

## 📊 目前運行的 Docker 靶場

### 完整列表

| 容器名稱 | 端口映射 | 鏡像 | 靶場類型 |
|---------|---------|------|---------|
| **juice-shop-live** | `3000 → 3000` | `bkimminich/juice-shop` | OWASP Juice Shop (主要) |
| **ecstatic_ritchie** | `3001 → 3000` | `bkimminich/juice-shop` | OWASP Juice Shop (副本1) |
| **vigilant_shockley** | `3003 → 3000` | `bkimminich/juice-shop` | OWASP Juice Shop (副本2) |
| **laughing_jang** | `8080 → 8080`, `9090 → 9090` | `webgoat/webgoat` | OWASP WebGoat |

### 驗證結果

✅ **所有 4 個靶場都是合法的安全測試環境**:
1. **OWASP Juice Shop** (3 個實例): 官方安全測試靶場
2. **OWASP WebGoat** (1 個實例): 官方 Web 安全教學平台

🔒 **這些都是專門設計用於安全測試的靶場，可以放心進行掃描測試**。

---

## 🎯 測試腳本使用的目標

### 1. `test_all_targets.py`

```python
test_targets = [
    ("http://localhost:3000", "Juice Shop (Angular SPA)"),
    ("http://localhost:3001", "靶場 #2"),
    ("http://localhost:3003", "靶場 #3"),
    ("http://localhost:8080", "靶場 #4"),
]
```

**驗證**: ✅ 全部指向合法靶場

---

### 2. `diagnose_http.py`

```python
test_urls = [
    "http://localhost:3000",  # Juice Shop
    "http://example.com",      # 公開測試域名
    "http://httpbin.org/get"   # HTTP 測試服務
]
```

**驗證**: ✅ 全部安全
- `localhost:3000`: 本地靶場
- `example.com`: IANA 保留的測試域名
- `httpbin.org`: 公開的 HTTP 測試服務

---

### 3. `test_dynamic_scan.py`

```python
asyncio.run(test_dynamic_scan("http://localhost:3000"))
```

**驗證**: ✅ 指向 Juice Shop 靶場

---

### 4. `test_targets_detailed.py`

```python
test_targets = [
    ("http://localhost:3000", "Juice Shop 3000"),
    ("http://localhost:3001", "靶場 3001"),
    ("http://localhost:3003", "靶場 3003"),
    ("http://localhost:8080", "靶場 8080"),
]
```

**驗證**: ✅ 全部指向合法靶場

---

### 5. `testing/scan/test_scan_integration.py`

```python
docker_targets = {
    "juice-shop": "http://localhost:3000",  # OWASP Juice Shop
    "vigilant": "http://localhost:3003",     # 靶場2
    "ecstatic": "http://localhost:3001",     # 靶場3  
    "laughing": "http://localhost:8080",     # 靶場4
}
```

**驗證**: ✅ 全部指向合法靶場

---

## 🔍 靶場特性分析

### OWASP Juice Shop (3000, 3001, 3003)

**技術棧**:
- 前端: Angular (SPA)
- 後端: Node.js + Express
- 數據庫: SQLite

**特性**:
- ✅ 完全無狀態的 SPA
- ⚠️ HTML 中幾乎沒有 `<a>` 標籤
- ✅ 所有路由和內容都是動態生成
- ✅ 包含多種漏洞類型 (XSS, SQLi, SSRF, 等)

**掃描建議**:
- 靜態引擎: ⚠️ 只能找到極少量 URL (正常行為)
- 動態引擎: ✅ 推薦使用 Playwright 進行 JavaScript 渲染
- Go SSRF 掃描器: ✅ 可以測試 SSRF 漏洞

---

### OWASP WebGoat (8080)

**技術棧**:
- 前端: React + Bootstrap
- 後端: Java Spring Boot
- 數據庫: HSQLDB

**特性**:
- ✅ 包含完整的安全課程和練習
- ✅ RESTful API 端點
- ✅ 多種漏洞模擬環境

**掃描建議**:
- 靜態引擎: ✅ 可以找到一些靜態資源
- 動態引擎: ✅ 推薦用於完整內容發現
- Go SSRF 掃描器: ✅ 可以測試 SSRF 相關課程

---

## 📋 測試目標映射表

| 端口 | 容器名稱 | 靶場類型 | 前端技術 | 靜態爬取效果 | 動態爬取效果 |
|------|---------|---------|---------|------------|------------|
| 3000 | juice-shop-live | Juice Shop | Angular SPA | ⚠️ 極少 URL | ✅ 豐富內容 |
| 3001 | ecstatic_ritchie | Juice Shop | Angular SPA | ⚠️ 極少 URL | ✅ 豐富內容 |
| 3003 | vigilant_shockley | Juice Shop | Angular SPA | ⚠️ 極少 URL | ✅ 豐富內容 |
| 8080 | laughing_jang | WebGoat | React SPA | ⚠️ 中等 URL | ✅ 完整內容 |

---

## ⚠️ 重要說明

### 為什麼靜態引擎返回 "0 個 URL" 是正確的

**Juice Shop 的 HTML 結構**:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OWASP Juice Shop</title>
  <base href="/">
  <!-- ... 一些 meta 標籤 ... -->
</head>
<body>
  <app-root></app-root>  <!-- ⚠️ 沒有任何 <a> 標籤！ -->
  <script src="runtime.js"></script>
  <script src="polyfills.js"></script>
  <script src="main.js"></script>
</body>
</html>
```

**靜態內容解析結果**:
- ✅ 找到 3 個 JavaScript 文件
- ❌ 找不到任何 `<a>` 標籤連結
- ✅ **這是正確的！** 所有導航都在 JavaScript 中

**解決方案**:
1. 使用 Playwright 動態引擎進行 SPA 掃描
2. 或使用 API 端點發現工具
3. 或參考靶場文檔預設已知路由

---

## 🎯 推薦的測試策略

### 策略 1: 快速驗證 (Python 引擎)
```bash
python diagnose_http.py
```
**目的**: 驗證 HTTP 請求功能是否正常  
**預期**: 3/3 請求成功

---

### 策略 2: 靜態掃描測試 (Python + Rust)
```bash
python test_all_targets.py
```
**目的**: 測試靜態內容解析器  
**預期**: Juice Shop 返回極少 URL (正確)，WebGoat 返回中等 URL

---

### 策略 3: 動態掃描測試 (Playwright)
```bash
python test_dynamic_scan.py
```
**目的**: 測試 JavaScript 渲染和 SPA 爬取  
**預期**: Juice Shop 返回 30+ 動態內容項

---

### 策略 4: 多引擎協同測試
```python
from services.scan.command_handler import ScanCommandHandler
from services.aiva_common.schemas import AICommand, CommandType

handler = ScanCommandHandler()

# 使用均衡策略 (Python + Rust)
command = AICommand(
    command_id="test_001",
    command_type=CommandType.SCAN_PHASE1,
    target_module="scan",
    payload={
        "scan_id": "balanced_test",
        "targets": ["http://localhost:3000"],
        "selected_engines": ["python", "rust"],
        "max_depth": 3
    }
)

result = await handler.handle_command(command)
```

---

### 策略 5: Go SSRF 掃描器測試
```python
from services.scan.coordinators.engines import GoAdapter

adapter = GoAdapter()
result = await adapter.scan(
    targets=["http://localhost:3000"],
    options={
        "scan_id": "ssrf_test",
        "scanner_type": "ssrf",
        "timeout": 30,
        "concurrency": 10
    }
)
```

---

## ✅ 安全確認

### 這些目標可以安全測試嗎？

**是的！** 所有目標都是合法的安全測試環境：

1. ✅ **OWASP Juice Shop**: 官方開源安全測試靶場
   - GitHub: https://github.com/juice-shop/juice-shop
   - 專門設計用於學習 Web 安全漏洞

2. ✅ **OWASP WebGoat**: 官方 Web 安全教學平台
   - GitHub: https://github.com/WebGoat/WebGoat
   - 用於教學和練習 Web 安全技術

3. ✅ **本地環境**: 所有靶場都運行在 Docker 容器中
   - 不會影響外部網絡
   - 完全隔離的測試環境

4. ✅ **example.com**: IANA 保留的測試域名
   - 專門用於文檔和測試
   - RFC 2606 標準

5. ✅ **httpbin.org**: 公開的 HTTP 測試服務
   - 專門用於 HTTP 客戶端測試
   - Kenneth Reitz 開發

---

## 📝 測試清單

完成以下測試以驗證系統功能：

- [ ] HTTP 請求功能 (`diagnose_http.py`)
- [ ] 靜態引擎掃描 (`test_all_targets.py`)
- [ ] 動態引擎掃描 (`test_dynamic_scan.py`)
- [ ] 引擎可用性 (`test_engine_availability.py`)
- [ ] Go SSRF 掃描器
- [ ] Rust 引擎集成
- [ ] 多引擎協同掃描

---

## 🎉 結論

✅ **所有測試目標都已驗證為合法的安全測試靶場**

可以放心進行各種掃描測試，不會有任何法律或道德問題。這些靶場正是為了測試和學習 Web 安全而設計的。

**建議的測試順序**:
1. 先運行 `diagnose_http.py` 確認基礎 HTTP 功能
2. 運行 `test_engine_availability.py` 確認引擎狀態
3. 運行 `test_all_targets.py` 測試靜態掃描
4. 運行 `test_dynamic_scan.py` 測試動態掃描
5. 測試 Go SSRF 掃描器
6. 進行完整的多引擎協同測試
