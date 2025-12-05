# 🎯 AIVA 十大核心能力實戰指南

**基於**: 2025-11-25 內部分析結果 (16,723 個能力)  
**目的**: 說明最實用的10個能力的使用時機和方法

---

## 📑 目錄

1. [📊 能力選擇標準](#-能力選擇標準)
2. [🚀 十大核心能力詳解](#-十大核心能力詳解)
   - [1️⃣ execute_phase0 - 快速偵察掃描](#1️⃣-execute_phase0---快速偵察掃描)
   - [2️⃣ adjust_from_phase0 - 動態策略調整](#2️⃣-adjust_from_phase0---動態策略調整)
   - [3️⃣ run_ssrf_oob_test - SSRF OOB 檢測](#3️⃣-run_ssrf_oob_test---ssrf-oob-檢測)
   - [4️⃣ search_capabilities - 能力搜索](#4️⃣-search_capabilities---能力搜索)
   - [5️⃣ query_knowledge_base - 知識庫查詢](#5️⃣-query_knowledge_base---知識庫查詢)
   - [其他5個核心能力](#其他5個核心能力)
3. [📊 能力使用統計](#-能力使用統計)
4. [❓ 常見問題](#-常見問題)

---

## 📊 能力選擇標準

從16,723個分析的能力中,根據以下標準篩選:
1. **實戰價值** - 直接用於安全測試
2. **完整度** - 有完整的參數和返回值定義
3. **清晰度** - 文檔描述清楚,易於理解
4. **協調性** - 與其他能力配合良好

---

## 🚀 十大核心能力詳解

### 1️⃣ execute_phase0 - 快速偵察掃描

**位置**: `services/scan/engines/python_engine/scan_orchestrator.py`  
**語言**: Python  
**類型**: 掃描能力

#### 📋 能力描述
執行 Phase0 快速偵察掃描 (5-10 分鐘),包含:
- 敏感資訊掃描 (調用 Rust 引擎)
- 技術棧指紋識別
- 基礎端點發現

#### 🎯 何時使用
```
✅ 使用時機:
- 首次接觸目標網站
- 需要快速了解目標規模
- 決策後續深度掃描策略前

❌ 不適用:
- 已知目標詳細信息
- 時間非常緊急 (< 5分鐘)
- 目標需要深度分析
```

#### 💻 如何使用

**參數**:
```python
request: Phase0StartPayload = {
    "scan_id": "scan_001",
    "targets": ["https://example.com"],
    "max_depth": 3  # 爬取深度
}
```

**調用方式**:
```python
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator

orchestrator = ScanOrchestrator()
result = orchestrator.execute_phase0(request)

# 返回: Phase0CompletedPayload
print(f"URLs 發現: {result.urls_found}")
print(f"表單發現: {result.forms_found}")
print(f"技術棧: {result.tech_stack}")
print(f"是否SPA: {result.is_spa}")
```

**通過 AICommand 調用** (推薦):
```python
from aiva_common.schemas import AICommand, CommandType

command = AICommand(
    command_id="phase0_001",
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={
        "scan_id": "scan_001",
        "targets": ["https://example.com"],
        "max_depth": 3
    },
    timeout=600  # 10分鐘
)

# 由 AICommandCenter 路由到 ScanCommandHandler
result = await command_center.execute(command)
```

#### 📊 返回值解析
```python
Phase0CompletedPayload {
    "scan_id": "scan_001",
    "urls_found": 150,        # 發現的URL數量
    "forms_found": 25,        # 發現的表單數量
    "endpoints_found": 80,    # 發現的API端點數量
    "tech_stack": {           # 技術棧信息
        "framework": "React",
        "server": "Nginx",
        "language": "JavaScript"
    },
    "is_spa": true,           # 是否為單頁應用
    "sensitive_data": [       # 發現的敏感資訊
        {"type": "api_key", "location": "/config.js"},
        {"type": "email", "location": "/contact.html"}
    ]
}
```

#### 🔄 後續動作
```python
# 根據 Phase0 結果決策
if result.is_spa:
    # SPA應用 → 啟動動態掃描
    next_command = CommandType.SCAN_PHASE1_DYNAMIC
elif result.urls_found > 100:
    # 大型網站 → 調整深度策略
    next_command = CommandType.SCAN_COMPREHENSIVE
else:
    # 小型網站 → 標準掃描
    next_command = CommandType.SCAN_PHASE1
```

---

### 2️⃣ adjust_from_phase0 - 動態策略調整

**位置**: `services/scan/engines/python_engine/strategy_controller.py`  
**語言**: Python  
**類型**: 策略優化

#### 📋 能力描述
根據 Phase0 掃描結果動態調整後續掃描策略參數

#### 🎯 何時使用
```
✅ 使用時機:
- Phase0 掃描完成後
- 需要優化後續掃描效率
- 目標特徵與預期不符

❌ 不適用:
- Phase0 未執行
- 已有固定掃描策略
```

#### 💻 如何使用
```python
from services.scan.engines.python_engine.strategy_controller import StrategyController

controller = StrategyController(strategy="balanced")

# 基於 Phase0 結果調整
controller.adjust_from_phase0(phase0_summary={
    "urls_found": 150,
    "forms_found": 25,
    "endpoints_found": 80,
    "tech_stack": {"framework": "React"},
    "is_spa": True
})

# 調整後的參數
params = controller.get_parameters()
print(f"Max pages: {params.max_pages}")  # 自動增加
print(f"Dynamic scan: {params.enable_dynamic_scan}")  # SPA啟用
```

#### 📊 調整邏輯

| Phase0 發現 | 調整動作 | 原因 |
|------------|---------|------|
| urls > 100 | max_pages ↑ | 大型網站需要更多頁面 |
| is_spa = true | enable_dynamic_scan = true | SPA需要動態渲染 |
| forms > 20 | max_forms ↑ | 複雜表單需要更多處理 |
| endpoints > 50 | requests_per_second ↑ | API密集型需要更高並發 |

---

### 3️⃣ run_ssrf_oob_test - SSRF OOB 檢測

**位置**: `services/features/function_ssrf/engines/ssrf_oob_detector.py`  
**語言**: Python  
**類型**: 攻擊能力

#### 📋 能力描述
執行 SSRF (Server-Side Request Forgery) OOB (Out-of-Band) 檢測,測試服務器是否會發起外部請求

#### 🎯 何時使用
```
✅ 使用時機:
- 目標有URL參數輸入點
- 目標可能處理外部資源
- 需要驗證SSRF漏洞

❌ 不適用:
- 無OOB回調服務
- 目標完全離線
- 純靜態網站
```

#### 💻 如何使用
```python
from services.features.function_ssrf.engines.ssrf_oob_detector import SSRFOOBDetector

detector = SSRFOOBDetector()

result = await detector.run_ssrf_oob_test(
    target="https://example.com",
    oob_callback="https://oob.yourserver.com",  # 你的OOB服務器
    test_endpoints=[
        "/api/fetch?url=",
        "/proxy?target=",
        "/download?file="
    ]
)

# 檢查結果
if result.vulnerability_found:
    print(f"發現SSRF: {result.vulnerable_endpoint}")
    print(f"OOB證據: {result.oob_evidence}")
```

#### 🔧 通過 AICommand 調用
```python
command = AICommand(
    command_id="ssrf_001",
    command_type=CommandType.FEATURE_SSRF_TEST,
    target_module="features",
    payload={
        "target": "https://example.com",
        "oob_callback": "https://oob.yourserver.com",
        "test_endpoints": ["/api/fetch?url="]
    }
)
```

#### 📊 OOB檢測流程
```
1. 生成唯一標識符
   └─ oob_id = "ssrf_20251205_abc123"

2. 構造測試URL
   └─ test_url = f"http://{oob_id}.oob.yourserver.com"

3. 注入到目標
   └─ https://example.com/api/fetch?url=test_url

4. 監聽OOB服務器
   └─ 等待DNS查詢或HTTP請求

5. 確認漏洞
   └─ 收到請求 → SSRF確認
```

---

### 4️⃣ search_capabilities - 能力搜索

**位置**: `services/core/aiva_core/core_capabilities/capability_registry.py`  
**語言**: Python  
**類型**: 自我認知

#### 📋 能力描述
搜索系統中已註冊的能力,支持關鍵字匹配

#### 🎯 何時使用
```
✅ 使用時機:
- AI需要查找特定功能的能力
- 用戶詢問"有哪些XSS測試方法"
- 動態選擇能力時

❌ 不適用:
- 已知確切能力名稱
- 不需要搜索功能
```

#### 💻 如何使用
```python
from services.core.aiva_core.core_capabilities.capability_registry import get_capability_registry

registry = get_capability_registry()

# 搜索XSS相關能力
xss_capabilities = registry.search_capabilities("xss")

for cap in xss_capabilities:
    print(f"名稱: {cap.name}")
    print(f"模組: {cap.module}")
    print(f"描述: {cap.description}")
    print("---")

# 輸出範例:
# 名稱: xss_comprehensive_scan
# 模組: features/function_xss
# 描述: 執行全面的XSS檢測...
```

#### 🤖 在 AI 決策中使用
```python
class CapabilityOrchestrator:
    async def find_capabilities_for_task(self, task: str):
        # AI接收任務: "測試XSS漏洞"
        
        # 搜索相關能力
        capabilities = self.registry.search_capabilities("xss")
        
        # 過濾並排序
        available = [
            cap for cap in capabilities 
            if cap.health_score > 0.8
        ]
        
        # 返回最佳能力
        return sorted(available, 
                     key=lambda c: c.success_rate, 
                     reverse=True)
```

---

### 5️⃣ scan_content - 敏感資料掃描

**位置**: `services/scan/engines/python_engine/sensitive_data_scanner.py`  
**語言**: Python  
**類型**: 信息收集

#### 📋 能力描述
掃描內容中的敏感資料 (API密鑰、密碼、電子郵件等)

#### 🎯 何時使用
```
✅ 使用時機:
- 掃描到HTML/JSON/JavaScript內容
- 需要檢查資訊洩露
- Phase0偵察階段

❌ 不適用:
- 二進制內容
- 已加密內容
```

#### 💻 如何使用
```python
from services.scan.engines.python_engine.sensitive_data_scanner import SensitiveDataScanner

scanner = SensitiveDataScanner()

# 掃描HTML內容
matches = scanner.scan_content(
    content=html_content,
    source_url="https://example.com/config.html",
    content_type="html"
)

# 檢查結果
for match in matches:
    print(f"類型: {match.type}")  # api_key, password, email
    print(f"值: {match.value}")
    print(f"位置: {match.location}")
    print(f"嚴重性: {match.severity}")  # high, medium, low
```

#### 📊 支持的敏感資料類型

| 類型 | 正則模式 | 嚴重性 | 範例 |
|-----|---------|--------|------|
| AWS Key | `AKIA[0-9A-Z]{16}` | High | `AKIAIOSFODNN7EXAMPLE` |
| Private Key | `-----BEGIN.*PRIVATE KEY-----` | Critical | RSA私鑰 |
| API Token | `Bearer [a-zA-Z0-9._-]+` | High | `Bearer eyJhbGc...` |
| Password | `password.*=.*['"]\w+['"]` | Medium | `password="secret123"` |
| Email | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | Low | `admin@example.com` |

---

### 6️⃣ is_in_scope - 範圍檢查

**位置**: `services/scan/engines/python_engine/scope_manager.py`  
**語言**: Python  
**類型**: 輔助能力

#### 📋 能力描述
檢查URL是否在掃描範圍內,防止越界

#### 🎯 何時使用
```
✅ 使用時機:
- 每次發現新URL
- 爬蟲添加URL到隊列前
- 防止掃描到外部網站

❌ 不適用:
- 單目標掃描
- 無範圍限制
```

#### 💻 如何使用
```python
from services.scan.engines.python_engine.scope_manager import ScopeManager

manager = ScopeManager(
    allowed_domains=["example.com"],
    include_subdomains=True
)

# 檢查URL
url1 = "https://example.com/page"
url2 = "https://sub.example.com/api"
url3 = "https://external.com/page"

print(manager.is_in_scope(url1))  # True
print(manager.is_in_scope(url2))  # True (子域名)
print(manager.is_in_scope(url3))  # False (外部域名)
```

#### 🔧 批量過濾
```python
discovered_urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://external.com/ads",
    "https://cdn.example.com/static/js"
]

# 過濾出範圍內的URL
in_scope_urls = manager.filter_urls(discovered_urls)
# 結果: ["https://example.com/page1", "https://example.com/page2"]
```

---

### 7️⃣ detect_in_javascript - JS 敏感信息檢測

**位置**: `services/scan/engines/python_engine/info_gatherer/sensitive_info_detector.py`  
**語言**: Python  
**類型**: 信息收集

#### 📋 能力描述
檢測 JavaScript 代碼中的敏感信息 (API端點、密鑰、內部URL等)

#### 🎯 何時使用
```
✅ 使用時機:
- 發現.js文件
- 分析前端JavaScript代碼
- 尋找隱藏的API端點

❌ 不適用:
- 混淆過的JS (需要先解混淆)
- 無JavaScript的靜態網站
```

#### 💻 如何使用
```python
from services.scan.engines.python_engine.info_gatherer.sensitive_info_detector import SensitiveInfoDetector

detector = SensitiveInfoDetector()

# 檢測JS代碼
js_code = """
const API_KEY = "sk-1234567890abcdef";
const baseURL = "https://internal-api.company.com";

function fetchData() {
    fetch(baseURL + "/api/users", {
        headers: {"Authorization": "Bearer " + API_KEY}
    });
}
"""

result = detector.detect_in_javascript(js_code, "https://example.com/app.js")

# 查看結果
for finding in result.findings:
    print(f"類型: {finding.type}")
    print(f"值: {finding.value}")
    print(f"嚴重性: {finding.severity}")
```

#### 📊 檢測內容

| 類型 | 示例 | 用途 |
|-----|------|------|
| API 端點 | `/api/users`, `/admin/delete` | 發現隱藏功能 |
| 內部URL | `https://internal.corp` | 了解內部架構 |
| 密鑰 | `API_KEY = "sk-..."` | 敏感資訊洩露 |
| 認證Token | `Bearer eyJ...` | 可能的越權漏洞 |
| 註釋信息 | `// TODO: 修復安全問題` | 開發者備註 |

---

### 8️⃣ add / next - URL 隊列管理

**位置**: `services/scan/engines/python_engine/core_crawling_engine/url_queue_manager.py`  
**語言**: Python  
**類型**: 爬蟲協調

#### 📋 能力描述
管理待掃描的URL隊列,支持深度控制和去重

#### 🎯 何時使用
```
✅ 使用時機:
- 爬蟲引擎中
- 需要廣度/深度優先搜索
- 大規模URL管理

❌ 不適用:
- 單URL掃描
- 無需爬蟲
```

#### 💻 如何使用

**添加URL**:
```python
from services.scan.engines.python_engine.core_crawling_engine.url_queue_manager import URLQueueManager

queue = URLQueueManager(max_depth=3)

# 添加起始URL
queue.add(
    url="https://example.com",
    parent_url=None,
    depth=0
)

# 爬蟲循環
while queue.has_next():
    url, depth = queue.next()
    print(f"處理: {url} (深度: {depth})")
    
    # 爬取並發現新URL
    discovered_urls = crawl_page(url)
    
    # 批量添加新URL
    added_count = queue.add_batch(
        urls=discovered_urls,
        parent_url=url,
        depth=depth + 1
    )
    print(f"添加了 {added_count} 個新URL")
```

**統計信息**:
```python
stats = queue.get_statistics()
print(f"待處理: {stats['queued_urls']}")
print(f"已處理: {stats['processed_urls']}")
print(f"已發現: {stats['seen_urls']}")
```

#### 🔄 工作流程
```
1. 添加起始URL
   └─ example.com (depth=0)

2. 爬取並發現新URL
   ├─ /page1 (depth=1)
   ├─ /page2 (depth=1)
   └─ /about (depth=1)

3. 繼續爬取
   ├─ /page1 → 發現 /page1/detail (depth=2)
   └─ /page2 → 發現 /page2/item (depth=2)

4. 深度限制
   └─ depth=3 時停止
```

---

### 9️⃣ execute_detection - 漏洞檢測

**位置**: `services/core/aiva_core/core_capabilities/multilang_coordinator.py`  
**語言**: Python  
**類型**: 漏洞測試

#### 📋 能力描述
執行漏洞檢測,支持多種漏洞類型 (XSS, SQLi, SSRF, IDOR)

#### 🎯 何時使用
```
✅ 使用時機:
- Phase1 深度測試階段
- 針對特定漏洞類型測試
- 需要統一的檢測接口

❌ 不適用:
- Phase0 偵察階段
- 無法確定測試類型
```

#### 💻 如何使用
```python
from services.core.aiva_core.core_capabilities.multilang_coordinator import MultilangCoordinator

coordinator = MultilangCoordinator()

# 執行XSS檢測
result = await coordinator.execute_detection(
    vuln_type="xss",
    target="https://example.com/search?q=",
    use_ai=True  # 使用AI增強檢測
)

# 檢查結果
if result.vulnerability_found:
    print(f"發現XSS漏洞!")
    print(f"位置: {result.location}")
    print(f"Payload: {result.successful_payload}")
    print(f"嚴重性: {result.severity}")
```

#### 🤖 AI 增強檢測

當 `use_ai=True` 時:
1. **智能Payload生成**: 根據目標特徵生成定制Payload
2. **上下文感知**: 分析表單結構和過濾規則
3. **繞過技術**: 自動嘗試WAF繞過
4. **結果驗證**: 更準確的漏洞確認

```python
# 標準檢測 vs AI檢測
standard_result = await coordinator.execute_detection(
    vuln_type="xss",
    target=target,
    use_ai=False  # 使用預定義Payload庫
)

ai_result = await coordinator.execute_detection(
    vuln_type="xss",
    target=target,
    use_ai=True  # AI生成定制Payload
)

# AI檢測通常有更高的成功率和更低的誤報率
```

---

### 🔟 push / sample - 經驗管理

**位置**: `services/core/aiva_core/external_learning/experience_manager.py`  
**語言**: Python  
**類型**: 學習優化

#### 📋 能力描述
保存和採樣攻擊經驗,用於AI學習和優化

#### 🎯 何時使用
```
✅ 使用時機:
- 每次攻擊執行後
- 需要訓練AI模型時
- 分析歷史成功案例

❌ 不適用:
- 不需要學習功能
- 一次性測試
```

#### 💻 如何使用

**保存經驗**:
```python
from services.core.aiva_core.external_learning.experience_manager import ExperienceManager

manager = ExperienceManager()

# 執行攻擊並記錄經驗
exp_id = manager.push(
    state={
        "target": "https://example.com/login",
        "form_fields": ["username", "password"],
        "tech_stack": "PHP"
    },
    action={
        "type": "sqli",
        "payload": "' OR '1'='1",
        "injection_point": "username"
    },
    next_state={
        "success": True,
        "response_code": 200,
        "bypassed_auth": True
    },
    reward=0.95,  # 高獎勵 (成功繞過認證)
    metadata={
        "execution_time": 2.5,
        "detection_method": "ai_enhanced"
    }
)
```

**採樣學習**:
```python
# 隨機採樣用於訓練
batch = manager.sample(batch_size=32)

# 優先採樣高質量經驗
high_quality = manager.prioritized_sample(
    batch_size=32,
    min_reward=0.8
)

# 創建訓練數據集
dataset = manager.create_dataset(
    name="sqli_training_v1",
    min_reward=0.7,
    max_samples=1000
)
```

#### 📊 經驗質量評估

| 獎勵值 | 質量 | 說明 | 用途 |
|-------|------|------|------|
| 0.9-1.0 | 優秀 | 成功且高效 | 優先學習 |
| 0.7-0.9 | 良好 | 成功但可優化 | 正常學習 |
| 0.5-0.7 | 一般 | 部分成功 | 參考學習 |
| < 0.5 | 失敗 | 無效或失敗 | 負面學習 |

---

## 🔗 能力組合使用範例

### 場景 1: 完整的網站滲透測試流程

```python
from aiva_common.command_center import AICommandCenter
from aiva_common.schemas import AICommand, CommandType

async def full_penetration_test(target: str):
    """完整的滲透測試流程"""
    
    command_center = AICommandCenter()
    experience_manager = ExperienceManager()
    
    # Step 1: Phase0 快速偵察
    print("🔍 Phase 0: 快速偵察...")
    phase0_cmd = AICommand(
        command_id="phase0_001",
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={"targets": [target], "max_depth": 3}
    )
    phase0_result = await command_center.execute(phase0_cmd)
    
    # Step 2: 根據結果調整策略
    print("⚙️ 調整掃描策略...")
    strategy_controller.adjust_from_phase0(phase0_result.data)
    
    # Step 3: 敏感資訊檢測
    print("🔎 檢測敏感資訊...")
    sensitive_matches = sensitive_scanner.scan_content(
        content=phase0_result.data["html_content"],
        source_url=target,
        content_type="html"
    )
    
    # Step 4: 範圍內URL過濾
    discovered_urls = phase0_result.data["urls_found"]
    in_scope_urls = scope_manager.filter_urls(discovered_urls)
    print(f"📋 發現 {len(in_scope_urls)} 個範圍內URL")
    
    # Step 5: 漏洞檢測 (XSS, SQLi, SSRF)
    for vuln_type in ["xss", "sqli", "ssrf"]:
        print(f"🎯 測試 {vuln_type.upper()}...")
        
        result = await coordinator.execute_detection(
            vuln_type=vuln_type,
            target=target,
            use_ai=True
        )
        
        # Step 6: 記錄經驗
        if result.vulnerability_found:
            experience_manager.push(
                state={"target": target, "vuln_type": vuln_type},
                action={"payload": result.successful_payload},
                next_state={"success": True},
                reward=0.9
            )
    
    print("✅ 滲透測試完成!")
```

### 場景 2: AI 自主決策掃描

```python
from services.core.aiva_core.cognitive_core.capability_orchestrator import CapabilityOrchestrator

async def ai_autonomous_scan(target: str):
    """AI 自主決策的掃描流程"""
    
    orchestrator = CapabilityOrchestrator()
    
    # AI 接收任務需求
    requirement = TaskRequirement(
        task_id="auto_scan_001",
        task_type="comprehensive_scan",
        target=target,
        objectives=[
            "find_vulnerabilities",
            "test_xss",
            "test_sqli",
            "check_sensitive_data"
        ]
    )
    
    # AI 生成執行計劃
    print("🤖 AI 正在分析並生成計劃...")
    plan = await orchestrator.plan(requirement)
    
    print(f"📋 AI 選擇了 {len(plan.selected_capabilities)} 個能力:")
    for cap in plan.selected_capabilities:
        print(f"  - {cap['metadata']['capability_name']}")
    
    print(f"\n🎯 決策理由:\n{plan.reasoning}")
    
    # 執行計劃
    print("\n🚀 開始執行...")
    result = await orchestrator.execute(plan)
    
    # AI 學習優化
    print("\n📚 AI 正在學習...")
    await orchestrator.learn_from_execution(plan, result)
    
    print(f"\n✅ 完成! 發現 {len(result.issues_found)} 個問題")
```

---

## 📊 能力使用決策樹

```
用戶請求
    │
    ├─ "掃描網站" 
    │   ├─ 首次掃描? → execute_phase0
    │   ├─ 需要深度? → adjust_from_phase0
    │   └─ 範圍控制? → is_in_scope
    │
    ├─ "測試漏洞"
    │   ├─ XSS? → execute_detection(vuln_type="xss")
    │   ├─ SQLi? → execute_detection(vuln_type="sqli")
    │   └─ SSRF? → run_ssrf_oob_test
    │
    ├─ "查找能力"
    │   └─ search_capabilities(keyword)
    │
    ├─ "檢測敏感資訊"
    │   ├─ HTML? → scan_content(content_type="html")
    │   ├─ JS? → detect_in_javascript
    │   └─ Headers? → detect_in_headers
    │
    └─ "管理URL"
        ├─ 添加? → queue.add()
        └─ 獲取? → queue.next()
```

---

## 🎯 最佳實踐建議

### 1. 先偵察後攻擊
```python
# ✅ 好的做法
phase0_result = execute_phase0(target)
adjust_from_phase0(phase0_result)
execute_detection("xss", target)  # 基於分析結果

# ❌ 不好的做法
execute_detection("xss", target)  # 直接攻擊,無準備
```

### 2. 使用範圍管理
```python
# ✅ 好的做法
if scope_manager.is_in_scope(url):
    queue.add(url)

# ❌ 不好的做法
queue.add(url)  # 可能掃描到外部網站
```

### 3. 記錄所有經驗
```python
# ✅ 好的做法
result = execute_detection(...)
experience_manager.push(state, action, result, reward)

# ❌ 不好的做法
result = execute_detection(...)  # 不記錄,無法學習
```

### 4. 利用 AI 增強
```python
# ✅ 好的做法
execute_detection("xss", target, use_ai=True)  # AI定制Payload

# ❌ 不好的做法  
execute_detection("xss", target, use_ai=False)  # 僅用通用Payload
```

---

## 🚀 快速開始範例

最簡單的使用方式:

```python
from services.core.aiva_core.cognitive_core.capability_orchestrator import quick_plan_and_execute

# 一行搞定!
plan, result = await quick_plan_and_execute(
    task_type="scan",
    target="https://example.com",
    objectives=["find_vulnerabilities"]
)

print(f"完成! 發現 {len(result.issues_found)} 個問題")
```

---

**總結**: 這10個能力涵蓋了從偵察、檢測到學習的完整流程。通過合理組合使用,可以構建出強大的自動化安全測試系統。AI 會根據任務需求自動選擇和編排這些能力,無需手動管理。
