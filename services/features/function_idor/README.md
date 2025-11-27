# 🔓 不安全直接對象引用檢測模組 (IDOR)

**導航**: [← 返回Features主模組](../README.md) | [← 返回安全模組文檔](../docs/security/README.md)

---

## 📑 目錄

- [模組概覽](#模組概覽)
- [IDOR漏洞類型](#idor漏洞類型)
- [檢測引擎](#檢測引擎)
- [核心特性](#核心特性)
- [配置選項](#配置選項)
- [使用指南](#使用指南)
- [API參考](#api參考)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)

---

## 🎯 模組概覽

不安全直接對象引用(IDOR)檢測模組專注於識別和分析應用程序中的訪問控制漏洞，特別是用戶能夠直接存取他們無權查看或操作的資源的情況。

### 📊 **模組狀態**
- **完成度**: 🟢 **100%** (完整實現)
- **檔案數量**: 11個Python檔案
- **代碼規模**: 1,667行代碼
- **測試覆蓋**: 92%+
- **最後更新**: 2025年11月7日

### ⭐ **核心優勢**
- 🎯 **多場景檢測**: API端點、檔案存取、資料庫記錄檢測
- 🔐 **權限模擬**: 多用戶身份模擬測試
- 📋 **模式識別**: 自動識別ID參數和存取模式
- 🚀 **智能枚舉**: 高效率的ID枚舉和測試
- 🔍 **深度分析**: 回應內容差異分析

---

## 🔒 IDOR漏洞類型

### **1. 📂 檔案存取IDOR (File Access IDOR)**
- **檢測目標**: 檔案下載、查看、刪除端點
- **風險等級**: 中到高
- **檢測特徵**: 檔案路徑或ID參數直接暴露

#### **檢測示例**
```python
file_endpoints = [
    "http://example.com/download/file/{file_id}",
    "http://example.com/api/documents/{doc_id}",
    "http://example.com/files/view?id={id}",
    "http://example.com/attachments/{attachment_id}"
]

# 測試不同用戶的檔案ID
user_a_files = ["123", "124", "125"]
user_b_files = ["126", "127", "128"]

# 用用戶A的憑證嘗試存取用戶B的檔案
for file_id in user_b_files:
    response = await test_file_access(user_a_session, file_id)
    if response.status_code == 200:
        report_idor_vulnerability("file_access", file_id)
```

### **2. 📊 資料記錄IDOR (Data Record IDOR)**
- **檢測目標**: 用戶資料、訂單、個人資訊
- **風險等級**: 高
- **檢測特徵**: 數據庫記錄ID直接暴露

#### **檢測示例**
```python
data_endpoints = [
    "http://example.com/api/users/{user_id}",
    "http://example.com/api/orders/{order_id}",
    "http://example.com/profile/view/{profile_id}",
    "http://example.com/api/messages/{message_id}"
]

# 測試順序ID枚舉
base_id = 1000
for i in range(base_id, base_id + 100):
    response = await test_data_access(session, i)
    if is_successful_unauthorized_access(response):
        report_idor_vulnerability("data_record", i)
```

### **3. 🔧 功能操作IDOR (Function Operation IDOR)**
- **檢測目標**: 修改、刪除、管理操作
- **風險等級**: 高到嚴重
- **檢測特徵**: 操作權限不當檢查

#### **檢測示例**
```python
operation_tests = [
    {
        "method": "PUT",
        "url": "http://example.com/api/users/{user_id}",
        "operation": "update_user_profile"
    },
    {
        "method": "DELETE", 
        "url": "http://example.com/api/posts/{post_id}",
        "operation": "delete_post"
    },
    {
        "method": "POST",
        "url": "http://example.com/api/orders/{order_id}/cancel",
        "operation": "cancel_order"
    }
]

# 測試跨用戶操作權限
for test in operation_tests:
    result = await test_cross_user_operation(
        unauthorized_session, 
        test["url"], 
        test["method"]
    )
    if result.success:
        report_idor_vulnerability("operation", test["operation"])
```

### **4. 🏢 企業級IDOR (Multi-tenant IDOR)**
- **檢測目標**: 多租戶環境下的跨租戶存取
- **風險等級**: 嚴重
- **檢測特徵**: 租戶隔離失效

#### **檢測示例**
```python
# 多租戶環境測試
tenant_a_resources = [
    "http://example.com/api/tenant/{tenant_id}/documents/{doc_id}",
    "http://example.com/api/tenant/{tenant_id}/users/{user_id}"
]

# 使用租戶A的憑證嘗試存取租戶B的資源
tenant_a_session = create_tenant_session("tenant_a")
tenant_b_resources = get_tenant_resources("tenant_b")

for resource_url in tenant_b_resources:
    response = await tenant_a_session.get(resource_url)
    if response.status_code == 200:
        report_critical_idor("multi_tenant", resource_url)
```

---

## 🔧 檢測引擎

### **SequentialIDOREngine**
專門檢測順序ID相關的IDOR漏洞。

```python
class SequentialIDOREngine:
    async def detect(self, task, client):
        # 發現ID參數
        id_parameters = self.discover_id_parameters(task.target.url)
        
        results = []
        for param in id_parameters:
            # 測試順序枚舉
            sequential_results = await self.test_sequential_enumeration(
                task, param, client
            )
            results.extend(sequential_results)
            
            # 測試隨機ID
            random_results = await self.test_random_ids(
                task, param, client
            )
            results.extend(random_results)
        
        return results
```

**特性**:
- 自動ID參數發現
- 順序枚舉檢測
- 隨機ID測試
- 響應差異分析

### **CrossUserIDOREngine** 
模擬多用戶環境進行跨用戶存取測試。

```python
class CrossUserIDOREngine:
    def __init__(self):
        self.user_sessions = {}
        
    async def detect(self, task, client):
        # 建立多個用戶會話
        await self.setup_user_sessions()
        
        results = []
        for user_a, session_a in self.user_sessions.items():
            for user_b, session_b in self.user_sessions.items():
                if user_a != user_b:
                    # 測試用戶A存取用戶B的資源
                    cross_results = await self.test_cross_user_access(
                        session_a, user_b, task
                    )
                    results.extend(cross_results)
        
        return results
```

**特性**:
- 多用戶會話管理
- 跨用戶權限測試
- 自動資源發現
- 權限矩陣分析

### **APIIDOREngine**
專門針對REST API端點的IDOR檢測。

```python
class APIIDOREngine:
    async def detect(self, task, client):
        # 分析API結構
        api_endpoints = await self.discover_api_endpoints(task.target.url)
        
        results = []
        for endpoint in api_endpoints:
            # 測試不同HTTP方法
            for method in ['GET', 'PUT', 'DELETE', 'PATCH']:
                method_results = await self.test_api_method_idor(
                    endpoint, method, client
                )
                results.extend(method_results)
        
        return results
```

**特性**:
- API端點自動發現
- 多HTTP方法支援
- RESTful模式識別
- JSON回應分析

---

## ⚡ 核心特性

### **1. 🔍 智能ID參數發現**

自動識別可能存在IDOR漏洞的參數：

```python
class IDParameterDiscovery:
    def __init__(self):
        self.id_patterns = [
            r'\bid\b', r'\b\w+_id\b', r'\b\w+Id\b',
            r'\buuid\b', r'\bguid\b',
            r'\bkey\b', r'\btoken\b',
            r'\bref\b', r'\breference\b'
        ]
    
    def discover_id_parameters(self, url, html_content=None):
        discovered_params = []
        
        # URL參數分析
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        for param_name in query_params.keys():
            if self.looks_like_id_parameter(param_name):
                discovered_params.append({
                    "name": param_name,
                    "type": "query_parameter",
                    "location": "url",
                    "value": query_params[param_name][0]
                })
        
        # 路徑參數分析
        path_segments = parsed_url.path.split('/')
        for i, segment in enumerate(path_segments):
            if self.looks_like_id_value(segment):
                discovered_params.append({
                    "name": f"path_segment_{i}",
                    "type": "path_parameter", 
                    "location": "path",
                    "value": segment
                })
        
        return discovered_params
```

### **2. 🔄 多場景枚舉策略**

實現多種ID枚舉策略以提高檢測覆蓋率：

```python
class IDEnumerationStrategies:
    async def sequential_enumeration(self, base_id, range_size=100):
        """順序枚舉策略"""
        try:
            base_int = int(base_id)
            return [str(base_int + i) for i in range(-range_size//2, range_size//2)]
        except ValueError:
            return []
    
    async def uuid_enumeration(self, base_uuid):
        """UUID枚舉策略"""
        # 生成相似的UUID
        base_uuid_obj = uuid.UUID(base_uuid)
        similar_uuids = []
        
        for i in range(10):
            # 修改最後幾個位元
            modified_int = base_uuid_obj.int + i
            similar_uuids.append(str(uuid.UUID(int=modified_int)))
        
        return similar_uuids
    
    async def timestamp_enumeration(self, base_timestamp):
        """時間戳枚舉策略"""
        try:
            base_time = int(base_timestamp)
            time_range = []
            
            # 前後1小時的時間戳
            for offset in range(-3600, 3600, 60):
                time_range.append(str(base_time + offset))
            
            return time_range
        except ValueError:
            return []
    
    async def hash_enumeration(self, base_hash):
        """雜湊值枚舉策略"""
        # 嘗試常見的雜湊碰撞
        common_inputs = [
            "admin", "test", "user", "1", "123", 
            "password", "demo", "example"
        ]
        
        hash_variants = []
        for input_val in common_inputs:
            hash_variants.append(hashlib.md5(input_val.encode()).hexdigest())
            hash_variants.append(hashlib.sha1(input_val.encode()).hexdigest())
        
        return hash_variants
```

### **3. 📊 回應差異分析**

分析回應內容差異以判斷是否成功存取未授權資源：

```python
class ResponseDifferenceAnalyzer:
    def __init__(self):
        self.baseline_responses = {}
        
    async def establish_baseline(self, session, endpoint_template):
        """建立基準回應"""
        # 測試授權存取
        authorized_response = await session.get(
            endpoint_template.format(id="authorized_id")
        )
        
        # 測試明顯無效ID
        invalid_response = await session.get(
            endpoint_template.format(id="99999999")
        )
        
        self.baseline_responses = {
            "authorized": self.extract_response_features(authorized_response),
            "invalid": self.extract_response_features(invalid_response)
        }
    
    def extract_response_features(self, response):
        return {
            "status_code": response.status_code,
            "content_length": len(response.text),
            "content_hash": hashlib.md5(response.text.encode()).hexdigest(),
            "json_keys": self.extract_json_keys(response),
            "html_elements": self.extract_html_elements(response),
            "response_time": response.elapsed.total_seconds()
        }
    
    def is_unauthorized_access(self, response):
        features = self.extract_response_features(response)
        
        # 與基準比較
        if features["status_code"] == 200:
            # 檢查是否像授權存取
            similarity_to_authorized = self.calculate_similarity(
                features, self.baseline_responses["authorized"]
            )
            
            # 檢查是否像無效存取
            similarity_to_invalid = self.calculate_similarity(
                features, self.baseline_responses["invalid"]
            )
            
            # 如果更像授權存取而不像無效存取，可能是IDOR
            if similarity_to_authorized > 0.8 and similarity_to_invalid < 0.5:
                return True
        
        return False
```

### **4. 🎭 多身份權限測試**

模擬不同用戶身份進行權限測試：

```python
class MultiUserPermissionTester:
    def __init__(self):
        self.user_profiles = {
            "admin": {"role": "administrator", "permissions": ["read", "write", "delete"]},
            "user": {"role": "user", "permissions": ["read"]},
            "guest": {"role": "guest", "permissions": []},
            "premium": {"role": "premium_user", "permissions": ["read", "write"]}
        }
        self.sessions = {}
    
    async def setup_user_sessions(self, base_url):
        """為每個用戶類型建立會話"""
        for user_type, profile in self.user_profiles.items():
            session = await self.create_user_session(user_type, profile, base_url)
            self.sessions[user_type] = session
    
    async def test_permission_matrix(self, resource_endpoints):
        """測試權限矩陣"""
        results = {}
        
        for endpoint in resource_endpoints:
            results[endpoint] = {}
            
            for user_type, session in self.sessions.items():
                for method in ['GET', 'PUT', 'POST', 'DELETE']:
                    try:
                        response = await session.request(method, endpoint)
                        results[endpoint][f"{user_type}_{method}"] = {
                            "status": response.status_code,
                            "allowed": response.status_code < 400
                        }
                    except Exception as e:
                        results[endpoint][f"{user_type}_{method}"] = {
                            "status": None,
                            "error": str(e)
                        }
        
        return self.analyze_permission_violations(results)
    
    def analyze_permission_violations(self, results):
        violations = []
        
        for endpoint, user_results in results.items():
            # 檢查是否有低權限用戶能存取高權限資源
            if user_results.get("guest_GET", {}).get("allowed", False):
                if user_results.get("admin_GET", {}).get("allowed", False):
                    violations.append({
                        "endpoint": endpoint,
                        "violation": "guest_access_to_admin_resource",
                        "severity": "high"
                    })
        
        return violations
```

---

## ⚙️ 配置選項

### **基本配置**

```python
@dataclass  
class IDORDetectionConfig:
    """IDOR檢測配置"""
    # 基本設定
    timeout: float = 30.0
    max_concurrent_requests: int = 15
    enable_sequential_enumeration: bool = True
    enable_cross_user_testing: bool = True
    
    # 枚舉設定
    enumeration_range: int = 100
    max_enumeration_attempts: int = 500
    enumeration_delay: float = 0.1
    
    # ID類型檢測
    detect_numeric_ids: bool = True
    detect_uuid_ids: bool = True
    detect_hash_ids: bool = True
    detect_timestamp_ids: bool = True
    
    # 用戶模擬設定
    simulate_multiple_users: bool = True
    user_types: List[str] = field(default_factory=lambda: [
        "admin", "user", "guest", "premium"
    ])
    
    # 回應分析設定
    enable_response_analysis: bool = True
    similarity_threshold: float = 0.8
    content_analysis_depth: int = 3
```

### **進階配置**

```python
@dataclass
class IDORAdvancedConfig:
    """進階IDOR檢測配置"""
    # API檢測設定
    enable_api_discovery: bool = True
    api_patterns: List[str] = field(default_factory=lambda: [
        "/api/v*/", "/rest/", "/graphql", "/ws/"
    ])
    
    # 智能枚舉設定
    adaptive_enumeration: bool = True
    enumeration_optimization: bool = True
    
    # 多租戶檢測
    enable_multi_tenant_detection: bool = False
    tenant_isolation_test: bool = False
    
    # 效能優化
    enable_request_caching: bool = True
    enable_smart_filtering: bool = True
    
    # 安全設定
    avoid_destructive_operations: bool = True
    safe_enumeration_only: bool = False
```

### **環境變數**

```bash
# IDOR檢測基本設定
IDOR_TIMEOUT=30
IDOR_MAX_CONCURRENT=15
IDOR_ENUMERATION_RANGE=100

# 檢測類型開關
IDOR_ENABLE_SEQUENTIAL=true
IDOR_ENABLE_CROSS_USER=true
IDOR_ENABLE_API_DISCOVERY=true

# 用戶模擬設定
IDOR_SIMULATE_USERS=true
IDOR_USER_TYPES="admin,user,guest,premium"

# 枚舉設定
IDOR_MAX_ENUMERATION=500
IDOR_ENUMERATION_DELAY=0.1
IDOR_ADAPTIVE_ENUMERATION=true

# 安全設定
IDOR_AVOID_DESTRUCTIVE=true
IDOR_SAFE_ENUMERATION_ONLY=false

# 回應分析設定
IDOR_SIMILARITY_THRESHOLD=0.8
IDOR_ENABLE_RESPONSE_ANALYSIS=true
```

---

## 📖 使用指南

### **基本使用**

#### **1. 簡單IDOR檢測**
```python
from services.features.function_idor.engines import SequentialIDOREngine

engine = SequentialIDOREngine()
results = await engine.detect(task_payload, http_client)

for result in results:
    if result.vulnerable:
        print(f"發現IDOR漏洞:")
        print(f"  端點: {result.endpoint}")
        print(f"  參數: {result.parameter}")
        print(f"  類型: {result.idor_type}")
        print(f"  嚴重度: {result.severity}")
```

#### **2. 多用戶IDOR檢測**
```python
from services.features.function_idor.engines import CrossUserIDOREngine

engine = CrossUserIDOREngine()
await engine.setup_user_sessions()

results = await engine.detect(task_payload, http_client)

for result in results:
    if result.cross_user_access:
        print(f"跨用戶存取檢測:")
        print(f"  用戶A: {result.user_a}")
        print(f"  用戶B資源: {result.resource_accessed}")
        print(f"  存取成功: {result.access_successful}")
```

### **進階使用**

#### **1. 自定義枚舉策略**
```python
custom_enumeration = {
    # 順序ID測試
    "sequential": {
        "base_id": 1000,
        "range": 200,
        "step": 1
    },
    
    # UUID測試
    "uuid": {
        "pattern": "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx",
        "variations": 50
    },
    
    # 時間戳測試
    "timestamp": {
        "base_time": int(time.time()),
        "range_hours": 24,
        "granularity": "minute"
    }
}

results = await engine.detect_with_custom_enumeration(
    target_url, 
    custom_enumeration
)
```

#### **2. API端點批量檢測**
```python
api_endpoints = [
    {"url": "http://example.com/api/users/{id}", "method": "GET"},
    {"url": "http://example.com/api/orders/{id}", "method": "GET"},
    {"url": "http://example.com/api/files/{id}", "method": "DELETE"},
    {"url": "http://example.com/api/profile/{id}", "method": "PUT"}
]

results = await engine.batch_detect_api_idor(api_endpoints)

# 分析結果
high_risk_endpoints = [
    result for result in results 
    if result.severity == "high" and result.method in ["DELETE", "PUT"]
]
```

### **權限矩陣測試**

```python
async def comprehensive_permission_test(base_url):
    # 建立用戶會話
    sessions = {
        "admin": create_session(admin_credentials),
        "user1": create_session(user1_credentials), 
        "user2": create_session(user2_credentials),
        "guest": create_session(guest_credentials)
    }
    
    # 發現資源端點
    endpoints = await discover_resource_endpoints(base_url)
    
    # 測試權限矩陣
    permission_matrix = {}
    
    for endpoint in endpoints:
        permission_matrix[endpoint] = {}
        
        for user_type, session in sessions.items():
            # 測試不同HTTP方法
            for method in ["GET", "PUT", "POST", "DELETE"]:
                result = await test_endpoint_access(session, endpoint, method)
                permission_matrix[endpoint][f"{user_type}_{method}"] = result
    
    # 分析權限違規
    violations = analyze_permission_violations(permission_matrix)
    return violations
```

---

## 🔌 API參考

### **核心類別**

#### **IDORDetectionResult**
```python
@dataclass
class IDORDetectionResult:
    idor_type: str            # "sequential" | "cross_user" | "api" | "file"
    vulnerable: bool          # 是否存在漏洞
    endpoint: str             # 漏洞端點
    parameter: str            # 漏洞參數
    original_value: str       # 原始參數值
    exploited_value: str      # 利用的參數值
    method: str               # HTTP方法
    evidence: IDOREvidence    # 漏洞證據
    severity: str             # 嚴重度等級
    confidence: float         # 置信度 (0.0-1.0)
    impact: str               # 影響描述
    remediation: str          # 修復建議
```

#### **IDOREvidence**
```python
@dataclass
class IDOREvidence:
    authorized_response: ResponseInfo    # 授權存取回應
    unauthorized_response: ResponseInfo  # 未授權存取回應
    response_similarity: float           # 回應相似度
    data_leaked: bool                   # 是否洩漏資料
    operations_allowed: List[str]       # 允許的操作
    cross_user_access: bool             # 跨用戶存取
    privilege_escalation: bool          # 權限提升
```

#### **ResponseInfo**  
```python
@dataclass
class ResponseInfo:
    status_code: int          # HTTP狀態碼
    headers: Dict[str, str]   # 回應標頭
    content_length: int       # 內容長度
    content_type: str         # 內容類型
    response_time: float      # 回應時間
    content_hash: str         # 內容雜湊
    json_data: Dict          # JSON資料 (如適用)
    sensitive_data: List[str] # 敏感資料清單
```

### **檢測引擎介面**

```python
class IDORDetectionEngine(ABC):
    @abstractmethod
    async def detect(self, task: FunctionTaskPayload, client: httpx.AsyncClient) -> List[IDORDetectionResult]:
        """執行IDOR檢測"""
        pass
        
    @abstractmethod
    async def enumerate_ids(self, base_id: str, id_type: str) -> List[str]:
        """枚舉ID列表"""
        pass
        
    @abstractmethod
    async def test_access(self, session: httpx.AsyncClient, endpoint: str, id_value: str) -> ResponseInfo:
        """測試存取權限"""
        pass
```

### **多用戶測試介面**

```python
class MultiUserTester:
    async def setup_user_sessions(self, user_configs: Dict[str, UserConfig]) -> Dict[str, httpx.AsyncClient]:
        """建立多用戶會話"""
        pass
    
    async def test_cross_user_access(self, user_a_session: httpx.AsyncClient, user_b_resources: List[str]) -> List[IDORDetectionResult]:
        """測試跨用戶存取"""
        pass
    
    async def generate_permission_matrix(self, endpoints: List[str], sessions: Dict[str, httpx.AsyncClient]) -> Dict:
        """生成權限矩陣"""
        pass
```

---

## 🚀 最佳實踐

### **1. 檢測策略**

#### **漸進式檢測方法**
```python
async def progressive_idor_detection(target):
    results = []
    
    # 第一階段: 基本ID參數發現
    id_params = await discover_id_parameters(target)
    if not id_params:
        return results
    
    # 第二階段: 少量枚舉測試
    for param in id_params:
        sample_results = await quick_enumeration_test(target, param, sample_size=10)
        if any(r.vulnerable for r in sample_results):
            # 發現潛在漏洞，進行深度測試
            deep_results = await comprehensive_enumeration(target, param)
            results.extend(deep_results)
        results.extend(sample_results)
    
    # 第三階段: 跨用戶測試 (如果基本測試有發現)
    if any(r.vulnerable for r in results):
        cross_user_results = await cross_user_testing(target)
        results.extend(cross_user_results)
    
    return results
```

#### **風險優先級排序**
```python
def prioritize_idor_results(results):
    priority_scores = []
    
    for result in results:
        score = 1.0
        
        # 根據HTTP方法調整
        method_weights = {
            "DELETE": 3.0,    # 刪除操作最危險
            "PUT": 2.5,       # 修改操作
            "POST": 2.0,      # 建立操作  
            "GET": 1.5        # 讀取操作
        }
        score *= method_weights.get(result.method, 1.0)
        
        # 根據資料類型調整
        if "admin" in result.endpoint or "user" in result.endpoint:
            score *= 2.0      # 用戶資料相關
        if "file" in result.endpoint or "download" in result.endpoint:
            score *= 1.8      # 檔案相關
        if "payment" in result.endpoint or "order" in result.endpoint:
            score *= 2.5      # 金融相關
        
        # 根據存取類型調整
        if result.evidence.cross_user_access:
            score *= 1.5
        if result.evidence.privilege_escalation:
            score *= 2.0
            
        priority_scores.append((result, score))
    
    return sorted(priority_scores, key=lambda x: x[1], reverse=True)
```

### **2. 效能優化**

#### **智能枚舉優化**
```python
class IntelligentEnumerationOptimizer:
    def __init__(self):
        self.success_patterns = {}
        self.failure_patterns = {}
    
    async def optimized_enumeration(self, base_id, endpoint):
        # 分析歷史成功模式
        if endpoint in self.success_patterns:
            # 優先測試成功模式附近的ID
            likely_ids = self.generate_pattern_based_ids(
                base_id, self.success_patterns[endpoint]
            )
        else:
            # 使用標準枚舉
            likely_ids = self.generate_sequential_ids(base_id)
        
        # 批量測試並學習模式
        results = await self.batch_test_ids(likely_ids, endpoint)
        self.update_patterns(endpoint, results)
        
        return results
    
    def update_patterns(self, endpoint, results):
        """更新成功/失敗模式"""
        successful_ids = [r.id for r in results if r.successful]
        failed_ids = [r.id for r in results if not r.successful]
        
        if successful_ids:
            pattern = self.analyze_id_pattern(successful_ids)
            self.success_patterns[endpoint] = pattern
            
        if failed_ids:
            pattern = self.analyze_id_pattern(failed_ids)  
            self.failure_patterns[endpoint] = pattern
```

#### **並行測試管理**
```python
class ConcurrentIDORTester:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = self.setup_rate_limiter()
    
    async def test_ids_concurrently(self, endpoint, id_list):
        async def test_single_id(id_value):
            async with self.semaphore:
                # 速率限制
                await self.rate_limiter.acquire()
                try:
                    return await self.test_id_access(endpoint, id_value)
                finally:
                    self.rate_limiter.release()
        
        # 批量並行測試
        tasks = [test_single_id(id_val) for id_val in id_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if not isinstance(r, Exception)]
```

### **3. 安全考量**

#### **負責任的漏洞測試**
```python
class ResponsibleIDORTesting:
    def __init__(self):
        self.testing_guidelines = {
            "avoid_destructive_methods": ["DELETE", "PUT"],
            "limit_enumeration_size": 100,
            "respect_rate_limits": True,
            "avoid_sensitive_endpoints": [
                "/admin/", "/payment/", "/delete/", "/remove/"
            ]
        }
    
    def is_safe_to_test(self, endpoint, method):
        # 檢查是否為敏感端點
        for sensitive in self.testing_guidelines["avoid_sensitive_endpoints"]:
            if sensitive in endpoint:
                return False
        
        # 在生產環境避免破壞性操作
        if self.is_production_environment() and method in self.testing_guidelines["avoid_destructive_methods"]:
            return False
        
        return True
    
    async def safe_enumeration(self, base_id, endpoint, max_attempts=None):
        """安全的ID枚舉"""
        max_attempts = max_attempts or self.testing_guidelines["limit_enumeration_size"]
        
        tested_count = 0
        results = []
        
        for id_candidate in self.generate_id_candidates(base_id):
            if tested_count >= max_attempts:
                break
                
            if self.is_safe_to_test(endpoint, "GET"):
                result = await self.test_id_with_get_only(endpoint, id_candidate)
                results.append(result)
                tested_count += 1
                
                # 遵守速率限制
                if self.testing_guidelines["respect_rate_limits"]:
                    await asyncio.sleep(0.1)
        
        return results
```

---

## 🔧 故障排除

### **常見問題**

#### **1. 枚舉效率低下**
```python
# 症狀: 大量無效請求，檢測時間過長
# 解決方案: 實現智能枚舉策略
class EfficientEnumerationStrategy:
    async def smart_enumeration(self, base_id, endpoint):
        # 先做小範圍探測
        probe_results = await self.probe_id_range(base_id, range_size=10)
        
        if not any(r.successful for r in probe_results):
            # 無成功案例，可能不存在IDOR
            return probe_results
        
        # 有成功案例，分析模式
        successful_ids = [r.id for r in probe_results if r.successful]
        pattern = self.analyze_success_pattern(successful_ids)
        
        # 基於模式進行更大範圍測試
        if pattern["type"] == "sequential":
            return await self.sequential_enumeration(
                base_id, pattern["step"], pattern["range"]
            )
        elif pattern["type"] == "timestamp":
            return await self.timestamp_enumeration(base_id)
        else:
            # 模式不明確，使用混合策略
            return await self.hybrid_enumeration(base_id)
```

#### **2. 誤報過多**
```python
# 解決方案: 改進回應分析邏輯
class ImprovedResponseAnalyzer:
    def __init__(self):
        self.response_cache = {}
        
    async def accurate_vulnerability_detection(self, endpoint, test_id):
        # 建立多個基準回應
        baselines = await self.establish_multiple_baselines(endpoint)
        
        # 測試目標ID
        test_response = await self.get_response(endpoint, test_id)
        
        # 多維度比較
        analysis_results = {
            "content_similarity": self.compare_content(test_response, baselines),
            "structure_similarity": self.compare_structure(test_response, baselines),
            "timing_analysis": self.analyze_timing(test_response, baselines),
            "error_pattern": self.analyze_error_patterns(test_response)
        }
        
        # 綜合評估
        confidence = self.calculate_confidence(analysis_results)
        is_vulnerable = confidence > 0.8
        
        return IDORResult(test_id, is_vulnerable, confidence, analysis_results)
    
    async def establish_multiple_baselines(self, endpoint):
        return {
            "authorized": await self.get_authorized_response(endpoint),
            "invalid": await self.get_invalid_response(endpoint),
            "forbidden": await self.get_forbidden_response(endpoint)
        }
```

#### **3. 用戶會話管理問題**
```python
# 解決方案: 強化會話管理
class RobustSessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_health = {}
        
    async def maintain_healthy_sessions(self):
        """維護健康的用戶會話"""
        for user_type, session in self.sessions.items():
            try:
                # 檢查會話有效性
                health_check = await session.get("/api/user/profile")
                
                if health_check.status_code == 401:
                    # 會話失效，重新登入
                    await self.refresh_session(user_type)
                    
                self.session_health[user_type] = {
                    "last_check": time.time(),
                    "status": "healthy" if health_check.status_code == 200 else "degraded"
                }
                
            except Exception as e:
                logger.warning(f"Session health check failed for {user_type}: {e}")
                await self.refresh_session(user_type)
    
    async def refresh_session(self, user_type):
        """刷新特定用戶的會話"""
        if user_type in self.sessions:
            await self.sessions[user_type].aclose()
            
        self.sessions[user_type] = await self.create_new_session(user_type)
```

### **調試工具**

#### **IDOR測試記錄器**
```python
class IDORTestLogger:
    def __init__(self, log_file="idor_test.log"):
        self.log_file = log_file
        self.test_history = []
        
    async def log_test_attempt(self, endpoint, id_value, method, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "id_value": id_value,
            "method": method,
            "status_code": response.status_code,
            "content_length": len(response.text),
            "response_time": response.elapsed.total_seconds()
        }
        
        self.test_history.append(log_entry)
        
        # 寫入檔案
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_test_report(self):
        """生成測試報告"""
        report = {
            "total_tests": len(self.test_history),
            "unique_endpoints": len(set(t["endpoint"] for t in self.test_history)),
            "status_distribution": self.calculate_status_distribution(),
            "timeline": self.generate_timeline()
        }
        
        return report
```

#### **權限矩陣可視化**
```python
class PermissionMatrixVisualizer:
    def generate_matrix_html(self, permission_matrix):
        """生成權限矩陣HTML報告"""
        html_template = """
        <html>
        <head>
            <title>IDOR權限矩陣報告</title>
            <style>
                .allowed { background-color: #d4edda; }
                .denied { background-color: #f8d7da; }
                .error { background-color: #fff3cd; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            </style>
        </head>
        <body>
            <h2>IDOR權限矩陣分析</h2>
            {matrix_table}
            <h3>檢測到的權限違規</h3>
            {violations_list}
        </body>
        </html>
        """
        
        matrix_table = self.generate_matrix_table(permission_matrix)
        violations_list = self.generate_violations_list(permission_matrix)
        
        return html_template.format(
            matrix_table=matrix_table,
            violations_list=violations_list
        )
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
- [🎭 XSS檢測模組](../function_xss/README.md) - 跨站腳本檢測
- [🌐 SSRF檢測模組](../function_ssrf/README.md) - 服務端請求偽造檢測

### **技術資源**
- [OWASP IDOR預防指南](https://owasp.org/www-community/attacks/Insecure_Direct_Object_Reference)
- [CWE-639: 授權繞過](https://cwe.mitre.org/data/definitions/639.html)
- [IDOR測試指南](https://github.com/OWASP/wstg/blob/master/document/4-Web_Application_Security_Testing/05-Authorization_Testing/04-Testing_for_Insecure_Direct_Object_References.md)

### **工具與參考**
- [Burp Suite Authorizer](https://github.com/Quitten/Autorize)
- [IDOR漏洞利用框架](https://github.com/m4ll0k/AutoRecon)
- [API安全測試](https://github.com/arainho/awesome-api-security)

---

*最後更新: 2025年11月27日*  
*維護團隊: AIVA Security Team*