# AIVA Services/Scan 模組深度分析報告 - 真實實現 vs 模擬實現

**分析時間**: 2025-11-30  
**分析目標**: services/scan/ 模組所有掃描引擎  
**重點**: 確認哪些是真實實現（有網路請求），哪些是模擬實現（僅 sleep）

---

## 🔴 **核心發現: vulnerability_scanner.py 是 100% 模擬實現**

### **1. vulnerability_scanner.py - 完全模擬（⚠️ 嚴重問題）**

**檔案路徑**: `services/scan/engines/python_engine/vulnerability_scanner.py`  
**總行數**: 237 行  
**實現類型**: ❌ **完全模擬 - 無任何真實 HTTP 請求**

#### **關鍵證據**:

```python
# 第 115 行：模擬 SQL 注入檢測
async def _check_sql_injection(self, target_url: str) -> List[Dict[str, Any]]:
    """檢查SQL注入漏洞"""
    sql_payloads = ["'", "' OR '1'='1", "'; DROP TABLE users; --"]
    
    for payload in sql_payloads:
        # ⚠️ 模擬網路延遲，沒有真實請求
        await asyncio.sleep(0.1)  # 👈 這裡只是 sleep
        
        # ⚠️ 沒有 aiohttp/httpx/requests 調用
        if "'" in payload:
            vulnerability = {
                "type": "SQL Injection",
                "severity": "HIGH",
                # ... 假數據
            }
```

```python
# 第 139 行：模擬 XSS 檢測
async def _check_xss(self, target_url: str) -> List[Dict[str, Any]]:
    """檢查XSS漏洞"""
    xss_payloads = ["<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>"]
    
    for payload in xss_payloads:
        await asyncio.sleep(0.1)  # 👈 又是 sleep
        
        # ⚠️ 直接創建假漏洞，沒有測試
        vulnerability = {
            "type": "Cross-Site Scripting (XSS)",
            "severity": "MEDIUM",
            # ... 假數據
        }
```

```python
# 第 159 行：模擬目錄遍歷檢測
async def _check_directory_traversal(self, target_url: str):
    for payload in traversal_payloads:
        await asyncio.sleep(0.1)  # 👈 還是 sleep
        
        # ⚠️ 只檢查 ".." 是否在 payload 中就回報漏洞
        if ".." in payload:
            vulnerability = { ... }
```

#### **問題總結**:
- ✅ **無任何 HTTP 庫導入**: 沒有 `aiohttp`, `httpx`, `requests`
- ✅ **所有檢測都用 `asyncio.sleep()` 模擬延遲**
- ✅ **沒有真實發送 payload 到目標**
- ✅ **直接生成假漏洞報告**

**結論**: ❌ **這是一個完全無用的模擬器，不會發現任何真實漏洞**

---

## ✅ **真實實現的檔案**

### **2. optimized_security_scanner.py - 真實實現**

**檔案路徑**: `services/scan/engines/python_engine/optimized_security_scanner.py`  
**總行數**: 448 行  
**實現類型**: ✅ **真實 HTTP 請求**

#### **證據**:

```python
# 第 13 行：真實 HTTP 庫導入
import aiohttp  # 👈 真實 HTTP 客戶端

# 第 57 行：創建真實連接器
self.connector = aiohttp.TCPConnector(
    limit=self.max_concurrent,
    limit_per_host=5,
    keepalive_timeout=30,
)

# 第 65 行：創建真實會話
self.session = aiohttp.ClientSession(
    connector=self.connector,
    timeout=timeout,
    headers={
        'User-Agent': 'AIVA-SecurityScanner/2.0',
    }
)

# 第 222 行：真實路徑掃描
async def _scan_paths_async(self, target: str) -> List[str]:
    async def check_path(path: str) -> Optional[str]:
        url = f"{target.rstrip('/')}{path}"
        async with self.session.get(url, allow_redirects=False) as response:
            # 👈 真實 HTTP GET 請求
            if response.status in [200, 301, 302, 403]:
                return path
```

**結論**: ✅ **這是真實掃描器，會發送 HTTP 請求**

---

### **3. http_client_hi.py - 真實實現**

**檔案路徑**: `services/scan/engines/python_engine/core_crawling_engine/http_client_hi.py`  
**總行數**: 194 行  
**實現類型**: ✅ **真實 HTTP 請求**

#### **證據**:

```python
# 第 6 行：導入 httpx
import httpx  # 👈 真實 HTTP 客戶端

# 第 66 行：初始化重試客戶端
self._client = RetryingAsyncClient(
    retries=retries,
    timeout=timeout,
    follow_redirects=True,
    headers=self._headers.user_headers,
    limits=httpx.Limits(
        max_connections=pool_size,
        max_keepalive_connections=pool_size // 2,
    ),
)

# 第 96 行：真實 GET 請求
async def get(self, url: str, **kwargs: Any) -> httpx.Response | None:
    host = urlparse(url).netloc
    
    await self._rate_limiter.acquire(host)  # 速率限制
    
    # 應用認證
    if self._auth.credentials:
        kwargs = self._auth.apply_auth_to_request(url, kwargs)
    
    # 發送請求
    response = await self._client.get(url, **kwargs)  # 👈 真實請求
    return response
```

**結論**: ✅ **真實 HTTP 客戶端，會發送請求**

---

### **4. network_scanner.py - 半模擬實現**

**檔案路徑**: `services/scan/engines/python_engine/network_scanner.py`  
**總行數**: 456 行  
**實現類型**: ⚠️ **部分真實（TCP 連接），但服務檢測模擬**

#### **證據**:

```python
# 第 197 行：真實 TCP 連接檢測
async def _check_port(self, host: str, port: int, timeout: float = 1.0) -> str:
    try:
        # 👈 真實 TCP 連接嘗試
        future = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(future, timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return "open"  # 真實檢測到端口開放
    except (ConnectionRefusedError, OSError):
        return "closed"
```

**但是服務橫幅獲取是模擬的**:

```python
# 第 231 行：橫幅獲取有嘗試真實請求
async def _get_banner(self, host: str, port: int) -> str:
    try:
        reader, writer = await asyncio.open_connection(host, port)
        # ... 但超時處理不完善，容易失敗後返回空字串
        return banner
    except Exception:
        return ""  # 失敗時返回空字串，無法區分真實失敗還是模擬
```

**結論**: ⚠️ **端口掃描是真實的，但服務識別品質低**

---

### **5. service_detector.py - 半模擬實現**

**檔案路徑**: `services/scan/engines/python_engine/service_detector.py`  
**總行數**: 619 行  
**實現類型**: ⚠️ **真實 TCP 連接 + 橫幅抓取，但分析邏輯簡陋**

#### **證據**:

```python
# 第 130 行：真實端口檢測
async def _is_port_open(self, host: str, port: int) -> bool:
    try:
        async with asyncio.timeout(2.0):
            _, writer = await asyncio.open_connection(host, port)  # 👈 真實連接
            writer.close()
            await writer.wait_closed()
            return True
    except Exception:
        return False

# 第 138 行：真實橫幅抓取
async def _get_service_banner(self, host: str, port: int) -> str:
    try:
        reader, writer = await asyncio.open_connection(host, port)
        
        # 對 HTTP 服務發送真實 HTTP 請求
        if port in [80, 8080, 443, 8443]:
            request = b"GET / HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n" % host.encode()
            writer.write(request)  # 👈 真實 HTTP 請求
            await writer.drain()
        
        banner_data = await asyncio.wait_for(reader.read(2048), timeout=2.0)
        banner = banner_data.decode('utf-8', errors='ignore').strip()
        
        return banner
    except Exception:
        return ""
```

**結論**: ✅ **真實實現，但錯誤處理可以改進**

---

### **6. fingerprint_manager.py - 真實實現（被動指紋）**

**檔案路徑**: `services/scan/engines/python_engine/fingerprint_manager.py`  
**總行數**: 39 行  
**實現類型**: ✅ **真實實現（分析 HTTP 響應）**

#### **證據**:

```python
# 第 7 行：導入 httpx
import httpx

# 第 34 行：處理真實 HTTP 響應
async def process_response(self, response: httpx.Response) -> None:
    """處理HTTP回應並收集指紋信息"""
    current_fp = self.passive_fp.from_headers(dict(response.headers))
    if current_fp:
        self.collected_fingerprints = self.merger.merge(
            self.collected_fingerprints, current_fp
        )
```

**結論**: ✅ **被動指紋識別，分析真實 HTTP 響應頭**

---

### **7. sensitive_data_scanner.py - 真實實現（內容分析）**

**檔案路徑**: `services/scan/engines/python_engine/sensitive_data_scanner.py`  
**總行數**: 253 行  
**實現類型**: ✅ **真實實現（正則匹配真實內容）**

#### **證據**:

```python
# 第 21 行：定義真實的敏感數據正則
PATTERNS = {
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "github_token": r"ghp_[0-9a-zA-Z]{36}",
    "jwt_token": r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    # ... 更多真實模式
}

# 第 52 行：掃描真實內容
def scan_content(self, content: str, source_url: str) -> list[SensitiveMatch]:
    """掃描內容中的敏感資料"""
    for pattern_name, pattern_regex in self.compiled_patterns.items():
        found_matches = pattern_regex.finditer(content)  # 👈 真實正則匹配
        for match in found_matches:
            # ... 記錄真實發現
```

**結論**: ✅ **真實內容分析器，會識別敏感數據**

---

## 🦀 **Rust Engine 分析**

### **8. rust_engine/src/scanner.rs - 真實實現（內容分析）**

**檔案路徑**: `services/scan/engines/rust_engine/src/scanner.rs`  
**總行數**: 304 行  
**實現類型**: ✅ **真實實現（正則匹配）**

#### **證據**:

```rust
// 第 53 行：編譯真實正則
Pattern {
    name: "AWS Access Key",
    regex: Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
    confidence: 0.95,
}

// 第 168 行：真實並行掃描
fn scan_deep(&self, content: &str, source_url: &str) -> Vec<SensitiveInfo> {
    self.patterns
        .par_iter()  // 👈 Rayon 並行迭代
        .flat_map(|pattern| {
            pattern.regex.find_iter(content)  // 👈 真實正則匹配
                .map(|m| { ... })
                .collect::<Vec<_>>()
        })
        .collect()
}
```

**結論**: ✅ **真實內容掃描器，高效並行**

---

### **9. rust_engine/src/verifier.rs - 真實實現（HTTP 驗證）**

**檔案路徑**: `services/scan/engines/rust_engine/src/verifier.rs`  
**總行數**: 未完整讀取  
**實現類型**: ✅ **真實實現（HTTP 請求驗證）**

#### **證據**:

```rust
// 第 5 行：導入 reqwest（HTTP 客戶端）
use reqwest::Client;

// 第 68 行：定義 HTTP 客戶端字段
client: reqwest::Client,

// 第 77 行：初始化真實客戶端
client: reqwest::Client::builder()
    .timeout(Duration::from_secs(10))
    .build()
    .unwrap(),
```

**結論**: ✅ **真實 HTTP 驗證器，會發送請求驗證密鑰**

---

## 🐹 **Go Engine 分析**

### **10. go_engine/internal/ssrf/detector/ssrf.go - 真實實現**

**檔案路徑**: `services/scan/engines/go_engine/internal/ssrf/detector/ssrf.go`  
**總行數**: 656 行  
**實現類型**: ✅ **真實實現（HTTP 請求測試）**

#### **證據**:

```go
// 第 11 行：導入 HTTP 庫
import (
    "net/http"  // 👈 真實 HTTP 客戶端
    "net/url"
)

// 第 52 行：創建真實 HTTP 客戶端
client := &http.Client{
    Timeout: 10 * time.Second,
    CheckRedirect: func(req *http.Request, via []*http.Request) error {
        if len(via) >= 3 {
            return fmt.Errorf("too many redirects")
        }
        return nil
    },
}

// 第 124 行：定義真實 SSRF 測試 payload
testPayloads := []struct {
    name        string
    url         string
    description string
}{
    {
        name:        "AWS IMDS v1",
        url:         "http://169.254.169.254/latest/meta-data/",
        description: "Attempt to access AWS Instance Metadata Service",
    },
    // ... 更多真實 payload
}
```

**結論**: ✅ **真實 SSRF 檢測器，會嘗試訪問內網服務**

---

## 📘 **TypeScript Engine 分析**

### **11. typescript_engine/src/services/scan-service.ts - 真實實現**

**檔案路徑**: `services/scan/engines/typescript_engine/src/services/scan-service.ts`  
**總行數**: 483 行  
**實現類型**: ✅ **真實實現（Playwright 瀏覽器）**

#### **證據**:

```typescript
// 第 6 行：導入 Playwright
import { Browser, Page, BrowserContext } from 'playwright-core';

// 第 83 行：創建真實瀏覽器上下文
context = await this.browser.newContext({
    viewport: { width: 1920, height: 1080 },
    userAgent: 'Mozilla/5.0 ... AIVA-Scanner/1.0',
});

page = await context.newPage();

// 第 90 行：啟動網路攔截
await this.networkInterceptor.startInterception(page);

// 第 93 行：監聽 WebSocket
this.setupWebSocketMonitoring(page, webSocketEndpoints);
```

**結論**: ✅ **真實動態掃描器，會執行 JavaScript 和攔截網路請求**

---

## 📊 **統計總結**

| 檔案名稱 | 行數 | 真實實現? | HTTP 庫 | 主要功能 |
|---------|------|----------|---------|----------|
| vulnerability_scanner.py | 237 | ❌ **模擬** | 無 | 假漏洞報告 |
| optimized_security_scanner.py | 448 | ✅ 真實 | aiohttp | 路徑掃描 |
| http_client_hi.py | 194 | ✅ 真實 | httpx | HTTP 請求 |
| network_scanner.py | 456 | ⚠️ 半真實 | asyncio TCP | 端口掃描 |
| service_detector.py | 619 | ✅ 真實 | asyncio TCP | 服務識別 |
| fingerprint_manager.py | 39 | ✅ 真實 | httpx | 被動指紋 |
| sensitive_data_scanner.py | 253 | ✅ 真實 | 正則 | 敏感數據 |
| rust_engine/scanner.rs | 304 | ✅ 真實 | 正則 | 敏感數據 |
| rust_engine/verifier.rs | ? | ✅ 真實 | reqwest | 密鑰驗證 |
| go_engine/ssrf.go | 656 | ✅ 真實 | net/http | SSRF 測試 |
| typescript_engine/scan-service.ts | 483 | ✅ 真實 | Playwright | 動態渲染 |

---

## 🚨 **嚴重問題**

### **vulnerability_scanner.py 必須重寫**

**當前狀況**:
- ❌ 完全沒有 HTTP 請求
- ❌ 所有檢測都是假的
- ❌ 使用 `asyncio.sleep()` 模擬延遲
- ❌ 直接生成假漏洞報告

**必須改進**:
1. 導入真實 HTTP 庫（httpx 或 aiohttp）
2. 實際發送 payload 到目標
3. 分析響應判斷是否存在漏洞
4. 實現真實的 SQL 注入、XSS、目錄遍歷檢測

**建議參考**:
- `optimized_security_scanner.py` 的 HTTP 實現
- `http_client_hi.py` 的請求管理
- Go Engine 的 SSRF 檢測邏輯

---

## ✅ **良好實現的檔案**

1. **http_client_hi.py** - 完整的 HTTP 客戶端，含重試和速率限制
2. **optimized_security_scanner.py** - 真實路徑掃描和標頭分析
3. **rust_engine/verifier.rs** - 真實密鑰驗證
4. **go_engine/ssrf.go** - 真實 SSRF 檢測
5. **typescript_engine/scan-service.ts** - 真實動態渲染

---

## 📝 **建議修復順序**

1. **立即修復**: vulnerability_scanner.py（完全重寫）
2. **改進**: network_scanner.py 的錯誤處理
3. **優化**: service_detector.py 的橫幅分析邏輯
4. **整合**: 確保所有引擎通過 multi_engine_coordinator.py 協調

---

## 🔍 **驗證方法**

### 如何驗證 vulnerability_scanner.py 是模擬的:

```bash
# 在目標檔案中搜尋 HTTP 庫
grep -E "aiohttp|httpx|requests" vulnerability_scanner.py
# 結果: 無匹配

# 搜尋 asyncio.sleep
grep "asyncio.sleep" vulnerability_scanner.py
# 結果: 4 次匹配（所有檢測函數都有）
```

### 如何驗證其他檔案是真實的:

```bash
# 檢查 optimized_security_scanner.py
grep "aiohttp" optimized_security_scanner.py
# 結果: 發現 import aiohttp 和 aiohttp.ClientSession

# 檢查 http_client_hi.py
grep "httpx" http_client_hi.py
# 結果: 發現 import httpx 和 httpx.AsyncClient
```

---

**報告結論**: 
- ✅ 大多數引擎是真實實現
- ❌ **vulnerability_scanner.py 是唯一的完全模擬實現**
- ⚠️ 需要立即重寫才能真正檢測漏洞
