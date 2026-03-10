# AI 如何操作 features_ready 模組

> **文檔目的**: 詳細說明 AI 如何調用和操作 `services/features/features_ready` 目錄下的功能模組  
> **適用範圍**: XSS、SQLi、SSRF、IDOR、BizLogic、Crypto、InfoLeak 等功能模組  
> **更新日期**: 2026-01-07

---

## 📁 features_ready 目錄結構

```
services/features/features_ready/
├── function_xss/           # XSS 檢測功能
│   ├── command_handler.py  # ✅ AI 命令處理器
│   ├── integration_tools/
│   │   └── xss_tools.py    # XSSManager
│   └── ...
├── function_sqli/          # SQL 注入檢測
│   ├── command_handler.py  # ✅ AI 命令處理器
│   ├── integration_tools/
│   │   └── sql_tools.py    # SQLInjectionManager
│   └── ...
├── function_ssrf/          # SSRF 檢測
│   ├── command_handler.py  # ✅ AI 命令處理器
│   └── ...
├── function_idor/          # IDOR 檢測
│   ├── command_handler.py  # ✅ AI 命令處理器
│   ├── smart_idor_detector.py
│   └── ...
├── function_bizlogic/      # 業務邏輯檢測
│   ├── command_handler.py  # ✅ AI 命令處理器
│   └── ...
├── function_crypto/        # 加密問題檢測
│   └── ...
└── function_info_leak/     # 資訊洩露檢測
    └── ...
```

**關鍵特徵**:
- ✅ 每個功能模組都有獨立的 `command_handler.py`
- ✅ 所有 CommandHandler 實現標準化接口
- ✅ 符合 `aiva_common` 命令系統規範
- ✅ 可直接被 AI 調用，無需額外註冊

---

## 🎯 AI 操作流程

### 完整執行鏈路

```
┌────────────────┐
│   AI 決策      │  EnhancedDecisionAgent.decide()
│  (cognitive)   │  → 決定要執行什麼操作
└───────┬────────┘
        │
        ↓ HighLevelIntent / Decision
┌────────────────┐
│  能力編排器     │  CapabilityOrchestrator.plan()
│ (cognitive)    │  → 生成 AICommand 序列
└───────┬────────┘
        │
        ↓ CapabilityPlan (包含多個 AICommand)
┌────────────────┐
│   命令中心      │  AICommandCenter.execute()
│ (aiva_common)  │  → 路由到對應處理器
└───────┬────────┘
        │
        ↓ 根據 target_module 路由
┌────────────────┐
│ CommandHandler │  XSSCommandHandler.handle_command()
│ (features)     │  → 執行實際功能
└───────┬────────┘
        │
        ↓ 調用內部工具
┌────────────────┐
│  功能管理器     │  XSSManager.comprehensive_scan()
│ (integration)  │  → 執行具體檢測邏輯
└───────┬────────┘
        │
        ↓ 返回結果
┌────────────────┐
│ AICommandResult│  返回給 AI
│                │  → AI 評估結果並決定下一步
└────────────────┘
```

---

## 🔧 方式一：直接調用（推薦用於簡單場景）

### 完整程式碼範例

```python
# ========== 1. 導入必要模組 ==========
from services.features.function_xss.command_handler import XSSCommandHandler
from services.aiva_common.schemas.commands import (
    AICommand,
    CommandType,
    AICommandResult
)
import asyncio

# ========== 2. 創建命令處理器 ==========
xss_handler = XSSCommandHandler()
# ✅ 初始化時會自動創建 XSSManager 實例

# ========== 3. 構建 AI 命令 ==========
command = AICommand(
    command_id="xss_test_001",           # 唯一命令 ID
    command_type=CommandType.FEATURE_XSS_TEST,  # 命令類型
    target_module="features.xss",         # 目標模組
    payload={
        # 必填參數
        "target_url": "https://example.com/search",
        
        # 可選參數：掃描類型
        "scan_type": "comprehensive",  # comprehensive/dalfox/dom/stored/blind/custom
        
        # 可選參數：掃描選項
        "options": {
            "use_dalfox": True,          # 使用 Dalfox 工具
            "scan_dom": True,            # 掃描 DOM XSS
            "scan_stored": True,         # 掃描存儲型 XSS
            "scan_blind": False,         # 掃描盲 XSS（需要 callback server）
            "custom_scan": True,         # 使用自定義 payload
            "callback_server": "http://your-server.com"  # 盲 XSS 回調服務器
        }
    },
    timeout=300  # 超時時間（秒）
)

# ========== 4. 執行命令 ==========
async def execute_xss_test():
    result: AICommandResult = await xss_handler.handle_command(command)
    
    # ========== 5. 處理結果 ==========
    if result.success:
        print(f"✅ XSS 測試成功")
        print(f"發現漏洞數: {result.result['summary']['total_vulnerabilities']}")
        print(f"執行時間: {result.execution_time:.2f}s")
        
        # 提取漏洞詳情
        for vuln in result.result['vulnerabilities']:
            print(f"  - {vuln['type']}: {vuln['url']}")
            print(f"    Payload: {vuln['payload']}")
            print(f"    嚴重性: {vuln['severity']}")
    else:
        print(f"❌ XSS 測試失敗: {result.error}")
        print(f"錯誤代碼: {result.error_code}")

# 執行
asyncio.run(execute_xss_test())
```

### 其他功能模組範例

#### SQLi 測試

```python
from services.features.function_sqli.command_handler import SQLiCommandHandler

sqli_handler = SQLiCommandHandler()

command = AICommand(
    command_type=CommandType.FEATURE_SQLI_TEST,
    target_module="features.sqli",
    payload={
        "target_url": "https://example.com/login",
        "method": "POST",  # GET/POST
        "parameters": {
            "username": "admin",
            "password": "test123"
        },
        "detection_engines": ["boolean", "time", "union", "error"],
        "deep_scan": False  # 是否深度掃描
    }
)

result = await sqli_handler.handle_command(command)
```

#### SSRF 測試

```python
from services.features.function_ssrf.command_handler import SSRFCommandHandler

ssrf_handler = SSRFCommandHandler()

command = AICommand(
    command_type=CommandType.FEATURE_SSRF_TEST,
    target_module="features.ssrf",
    payload={
        "target_url": "https://example.com/api/fetch",
        "parameters": {"url": "http://internal-service"},
        "detection_methods": ["callback", "blind", "internal_scan"],
        "callback_server": "http://your-callback.com"
    }
)

result = await ssrf_handler.handle_command(command)
```

#### IDOR 測試

```python
from services.features.function_idor.command_handler import IDORCommandHandler

idor_handler = IDORCommandHandler()

command = AICommand(
    command_type=CommandType.FEATURE_IDOR_TEST,
    target_module="features.idor",
    payload={
        "target_url": "https://example.com/api/user/123",
        "resource_patterns": ["user_id", "post_id", "document_id"],
        "test_methods": ["sequential", "random", "privilege_escalation"],
        "credentials": {
            "user": "token123",
            "admin": "token456"
        }
    }
)

result = await idor_handler.handle_command(command)
```

---

## 🏢 方式二：透過命令中心（推薦用於複雜場景）

### 完整程式碼範例

```python
# ========== 1. 初始化命令中心 ==========
from services.aiva_common.command_center import AICommandCenter

command_center = AICommandCenter()

# ========== 2. 註冊所有功能模組 ==========
from services.features.function_xss.command_handler import XSSCommandHandler
from services.features.function_sqli.command_handler import SQLiCommandHandler
from services.features.function_ssrf.command_handler import SSRFCommandHandler
from services.features.function_idor.command_handler import IDORCommandHandler
from services.features.function_bizlogic.command_handler import BizLogicCommandHandler

# 註冊處理器
command_center.register_module("features.xss", XSSCommandHandler())
command_center.register_module("features.sqli", SQLiCommandHandler())
command_center.register_module("features.ssrf", SSRFCommandHandler())
command_center.register_module("features.idor", IDORCommandHandler())
command_center.register_module("features.bizlogic", BizLogicCommandHandler())

print("✅ 所有功能模組已註冊到命令中心")

# ========== 3. 透過命令中心執行 ==========
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features.xss",  # 命令中心會自動路由
    payload={"target_url": "https://example.com"}
)

result = await command_center.execute(command)

# ========== 4. 批次執行多個測試 ==========
async def batch_security_scan(target_url: str):
    """批次執行多個安全測試"""
    
    commands = [
        AICommand(
            command_type=CommandType.FEATURE_XSS_TEST,
            target_module="features.xss",
            payload={"target_url": target_url}
        ),
        AICommand(
            command_type=CommandType.FEATURE_SQLI_TEST,
            target_module="features.sqli",
            payload={"target_url": target_url}
        ),
        AICommand(
            command_type=CommandType.FEATURE_SSRF_TEST,
            target_module="features.ssrf",
            payload={"target_url": target_url}
        )
    ]
    
    results = []
    for cmd in commands:
        result = await command_center.execute(cmd)
        results.append(result)
        
        if result.success:
            print(f"✅ {cmd.command_type.value}: 發現 {len(result.result.get('vulnerabilities', []))} 個漏洞")
        else:
            print(f"❌ {cmd.command_type.value}: {result.error}")
    
    return results

# 執行批次掃描
results = await batch_security_scan("https://example.com")
```

---

## 🧠 方式三：透過 AI 決策引擎（自動化場景）

### AI 決策驅動執行

```python
from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
from services.core.aiva_core.cognitive_core.capability_orchestrator import CapabilityOrchestrator
from services.aiva_common.schemas.decisions import DecisionContext, RiskLevel

# ========== 1. 初始化 AI 組件 ==========
decision_agent = EnhancedDecisionAgent(neural_engine=real_decision_engine)
orchestrator = CapabilityOrchestrator(rag_store=vector_store)

# ========== 2. 構建決策上下文 ==========
context = DecisionContext(
    target_info={
        "url": "https://example.com",
        "type": "web_application",
        "tech_stack": ["PHP", "MySQL", "Apache"]
    },
    discovered_vulns=[],  # 已發現的漏洞
    risk_level=RiskLevel.MEDIUM,
    available_tools=["xss_scanner", "sqli_scanner", "ssrf_scanner"],
    attempts_without_success=0
)

# ========== 3. AI 做出決策 ==========
high_level_intent = await decision_agent.decide(context)

print(f"🧠 AI 決策: {high_level_intent.intent_type}")
print(f"   目標: {high_level_intent.target_module}")
print(f"   推理: {high_level_intent.reasoning}")

# ========== 4. 執行 AI 決策 ==========
# AI 決策會轉換為具體的 AICommand
decision = Decision(
    action="RUN_TOOL",
    params={
        "tool": "xss_scanner",
        "target_url": "https://example.com"
    },
    confidence=0.85
)

# 執行決策（自動調用對應模組）
result = await decision_agent.execute_decision(decision, context)

# ========== 5. AI 評估結果 ==========
if result["success"]:
    # AI 分析發現的漏洞
    vulnerabilities = result.get("vulnerabilities", [])
    
    # 更新上下文
    context.discovered_vulns.extend([v['type'] for v in vulnerabilities])
    
    # AI 決定下一步行動
    next_decision = await decision_agent.decide(context)
    print(f"🧠 下一步決策: {next_decision.intent_type}")
```

### AI 自動編排多步驟執行

```python
from services.core.aiva_core.cognitive_core.capability_orchestrator import (
    CapabilityOrchestrator,
    TaskRequirement
)

# ========== 1. 定義任務需求 ==========
requirement = TaskRequirement(
    task_type="security_scan",
    description="對 example.com 進行全面的安全掃描",
    target_info={
        "url": "https://example.com",
        "priority": "high"
    },
    constraints={
        "max_duration": 1800,  # 30 分鐘
        "stealth_mode": False
    }
)

# ========== 2. AI 生成執行計劃 ==========
orchestrator = CapabilityOrchestrator(rag_store=vector_store)
plan = await orchestrator.plan(requirement)

print(f"📋 AI 生成的執行計劃:")
print(f"   計劃 ID: {plan.plan_id}")
print(f"   命令數: {len(plan.commands)}")
print(f"   預估時間: {plan.estimated_duration}s")

# ========== 3. 執行計劃 ==========
execution_result = await orchestrator.execute(plan)

print(f"✅ 執行完成:")
print(f"   成功命令: {len(execution_result.completed_commands)}")
print(f"   失敗命令: {len(execution_result.failed_commands)}")
print(f"   發現問題: {len(execution_result.issues_found)}")
print(f"   總耗時: {execution_result.total_duration:.2f}s")

# ========== 4. AI 學習優化 ==========
orchestrator.learn_from_execution(plan, execution_result)
```

---

## 📊 AICommandResult 結構

### 成功結果

```python
AICommandResult(
    command_id="xss_test_001",
    status=CommandStatus.COMPLETED,  # COMPLETED / FAILED / TIMEOUT
    success=True,
    execution_time=12.5,  # 秒
    started_at=datetime(2026, 1, 7, 10, 0, 0),
    completed_at=datetime(2026, 1, 7, 10, 0, 12),
    
    # 結果數據
    result={
        "summary": {
            "total_vulnerabilities": 3,
            "severity_breakdown": {
                "high": 1,
                "medium": 2,
                "low": 0
            }
        },
        "vulnerabilities": [
            {
                "type": "reflected_xss",
                "url": "https://example.com/search?q=test",
                "parameter": "q",
                "payload": "<script>alert(1)</script>",
                "severity": "high",
                "confidence": 0.95,
                "evidence": "Response contained unescaped payload"
            }
        ],
        "scan_details": {
            "scan_type": "comprehensive",
            "payloads_tested": 150,
            "requests_sent": 300
        }
    },
    
    # 性能指標
    metrics={
        "scan_type": "comprehensive",
        "target_url": "https://example.com",
        "vulnerabilities_found": 3,
        "timestamp": "2026-01-07T10:00:12"
    },
    
    # 錯誤資訊（成功時為 None）
    error=None,
    error_code=None,
    error_details=None
)
```

### 失敗結果

```python
AICommandResult(
    command_id="xss_test_002",
    status=CommandStatus.FAILED,
    success=False,
    execution_time=5.2,
    started_at=datetime(2026, 1, 7, 10, 5, 0),
    completed_at=datetime(2026, 1, 7, 10, 5, 5),
    
    # 錯誤資訊
    error="參數錯誤: 缺少必要參數 target_url",
    error_code="INVALID_PARAMETER",
    error_details={
        "exception_type": "ValueError",
        "parameter_error": "缺少必要參數: target_url",
        "received_payload": {}
    },
    
    result=None,
    metrics=None
)
```

---

## 🎨 進階場景

### 場景 1: AI 根據技術棧選擇測試模組

```python
async def ai_smart_scan(target_url: str, tech_stack: list):
    """AI 根據技術棧智能選擇測試模組"""
    
    # 構建上下文
    context = DecisionContext(
        target_info={
            "url": target_url,
            "tech_stack": tech_stack  # ["PHP", "MySQL", "Apache"]
        },
        risk_level=RiskLevel.MEDIUM,
        available_tools=["xss", "sqli", "ssrf", "idor"]
    )
    
    # AI 分析並選擇工具
    decision = await decision_agent.decide(context)
    
    # 執行 AI 選擇的測試
    if "sqli" in decision.params.get("recommended_tools", []):
        # PHP + MySQL → 高機率存在 SQLi
        sqli_command = AICommand(
            command_type=CommandType.FEATURE_SQLI_TEST,
            target_module="features.sqli",
            payload={"target_url": target_url, "deep_scan": True}
        )
        await command_center.execute(sqli_command)
    
    if "xss" in decision.params.get("recommended_tools", []):
        # Web 應用 → 測試 XSS
        xss_command = AICommand(
            command_type=CommandType.FEATURE_XSS_TEST,
            target_module="features.xss",
            payload={"target_url": target_url}
        )
        await command_center.execute(xss_command)
```

### 場景 2: AI 根據掃描結果動態調整策略

```python
async def ai_adaptive_scan(target_url: str):
    """AI 根據掃描結果動態調整策略"""
    
    context = DecisionContext(
        target_info={"url": target_url},
        discovered_vulns=[],
        attempts_without_success=0
    )
    
    max_iterations = 5
    for i in range(max_iterations):
        # AI 決定下一步
        decision = await decision_agent.decide(context)
        
        print(f"🧠 第 {i+1} 輪決策: {decision.action}")
        
        # 執行決策
        result = await decision_agent.execute_decision(decision, context)
        
        if result["success"]:
            # 發現漏洞，更新上下文
            vulns = result.get("vulnerabilities", [])
            context.discovered_vulns.extend([v['type'] for v in vulns])
            context.attempts_without_success = 0
            
            print(f"✅ 發現 {len(vulns)} 個漏洞")
        else:
            # 未發現漏洞，累計失敗次數
            context.attempts_without_success += 1
            
            if context.attempts_without_success >= 3:
                print("⚠️ 連續 3 次未發現漏洞，AI 建議切換策略")
                # AI 可能會建議更深入的測試或切換工具
        
        # AI 評估是否繼續
        if len(context.discovered_vulns) >= 10:
            print("✅ 已發現足夠漏洞，停止掃描")
            break
```

### 場景 3: 多目標並行掃描

```python
async def ai_parallel_scan(target_urls: list):
    """AI 並行掃描多個目標"""
    
    tasks = []
    for url in target_urls:
        # 為每個目標創建掃描任務
        task = asyncio.create_task(ai_smart_scan(url, ["PHP", "MySQL"]))
        tasks.append(task)
    
    # 並行執行
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 彙總結果
    total_vulns = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ {target_urls[i]} 掃描失敗: {result}")
        else:
            vulns = len(result.get("vulnerabilities", []))
            total_vulns += vulns
            print(f"✅ {target_urls[i]}: 發現 {vulns} 個漏洞")
    
    print(f"\n📊 總計發現 {total_vulns} 個漏洞")
```

---

## 🔐 安全注意事項

### 1. 目標授權

```python
# ❌ 危險：未經授權掃描
command = AICommand(
    target_module="features.xss",
    payload={"target_url": "https://random-website.com"}  # 未經授權
)

# ✅ 安全：檢查授權
def is_authorized_target(url: str) -> bool:
    """檢查目標是否在授權白名單中"""
    authorized_domains = [
        "example.com",
        "testsite.local",
        "bugbounty.target.com"
    ]
    return any(domain in url for domain in authorized_domains)

if is_authorized_target(target_url):
    result = await command_center.execute(command)
else:
    print("❌ 未經授權的掃描目標")
```

### 2. 速率限制

```python
# ✅ 添加速率限制
import asyncio

async def rate_limited_scan(urls: list, delay: float = 2.0):
    """帶速率限制的掃描"""
    for url in urls:
        command = AICommand(...)
        result = await command_center.execute(command)
        
        # 每次掃描後延遲
        await asyncio.sleep(delay)
```

### 3. 日誌記錄

```python
# ✅ 記錄所有操作
import logging

logger = logging.getLogger(__name__)

async def logged_execution(command: AICommand):
    """記錄執行過程的包裝器"""
    logger.info(f"開始執行命令: {command.command_id}")
    logger.info(f"  類型: {command.command_type}")
    logger.info(f"  目標: {command.payload.get('target_url')}")
    
    try:
        result = await command_center.execute(command)
        
        if result.success:
            logger.info(f"命令執行成功: {command.command_id}")
            logger.info(f"  發現漏洞: {len(result.result.get('vulnerabilities', []))}")
        else:
            logger.error(f"命令執行失敗: {command.command_id}")
            logger.error(f"  錯誤: {result.error}")
        
        return result
    except Exception as e:
        logger.exception(f"命令執行異常: {command.command_id}")
        raise
```

---

## 📈 性能優化

### 1. 命令池化

```python
from concurrent.futures import ThreadPoolExecutor

class CommandPool:
    """命令執行池"""
    
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_commands = []
    
    async def submit(self, command: AICommand):
        """提交命令到池中"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(command_center.execute(command))
        )
        return await future

# 使用命令池
pool = CommandPool(max_workers=3)
results = await asyncio.gather(
    pool.submit(command1),
    pool.submit(command2),
    pool.submit(command3)
)
```

### 2. 結果快取

```python
from functools import lru_cache
import hashlib

class ResultCache:
    """結果快取器"""
    
    def __init__(self):
        self.cache = {}
    
    def _get_cache_key(self, command: AICommand) -> str:
        """生成快取鍵"""
        payload_str = str(command.payload)
        return hashlib.md5(
            f"{command.command_type}:{payload_str}".encode()
        ).hexdigest()
    
    async def execute_with_cache(self, command: AICommand):
        """帶快取的執行"""
        cache_key = self._get_cache_key(command)
        
        if cache_key in self.cache:
            print(f"💾 使用快取結果: {cache_key}")
            return self.cache[cache_key]
        
        result = await command_center.execute(command)
        self.cache[cache_key] = result
        return result
```

---

## 🎓 總結

### ✅ AI 可以操作 features_ready 模組的三種方式

| 方式 | 適用場景 | 複雜度 | 靈活性 |
|------|---------|--------|--------|
| **直接調用** | 簡單單一測試 | ⭐ | ⭐⭐ |
| **命令中心** | 批次測試、複雜場景 | ⭐⭐ | ⭐⭐⭐ |
| **AI 決策引擎** | 自動化、智能決策 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 🎯 關鍵要點

1. **標準化接口**: 所有功能模組都實現 `CommandHandler` 協議
2. **統一命令格式**: 使用 `AICommand` 和 `AICommandResult`
3. **AI 友好**: 支援 AI 決策驅動和自動化編排
4. **安全可控**: 提供授權檢查、速率限制、日誌記錄
5. **高性能**: 支援並行執行、結果快取

### 📝 快速參考

```python
# 最簡單的使用方式
from services.features.function_xss.command_handler import XSSCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

handler = XSSCommandHandler()
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features.xss",
    payload={"target_url": "https://example.com"}
)
result = await handler.handle_command(command)
```

---

**文檔完成**: 2026-01-07  
**作者**: GitHub Copilot  
**版本**: 1.0
