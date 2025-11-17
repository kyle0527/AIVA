# AIVA 內閉環到外閉環能力轉換方案

**日期**: 2025-11-16  
**版本**: v1.0  
**狀態**: 📋 規劃中

---

## 📊 執行摘要

### 當前狀態
- **總能力數**: 692 個 (已通過實際分析驗證)
- **語言分布**: Python 411 (59.4%), Rust 115 (16.6%), Go 88 (12.7%), TypeScript 78 (11.3%)
- **模組分布**: scan (268), core/aiva_core (206), integration (76), features (54), 其他 (88)
- **內部能力**: 100% 完成自我探索和分析
- **對外能力**: ~11 個 UI 接口, ~24 個執行接口 (約 5%)

### 轉換目標
將內部的 692 個能力，通過 **CapabilityRegistry** 和 **AI 編排系統**，對外提供企業級的智能安全測試服務。

---

## 🎯 核心概念：內閉環 vs 外閉環

### 內閉環 (Internal Loop) - 已完成 ✅
```
目的: 自我認知、能力發現、性能優化
流程: internal_exploration → capability_analyzer → knowledge_base
結果: 692 個能力被識別、分類、索引
狀態: ✅ 完成 (通過 run_capability_analysis.py 驗證)
```

### 外閉環 (External Loop) - 規劃中 📋
```
目的: 對外服務、用戶交互、價值交付
流程: user_request → capability_registry → ai_orchestrator → tool_execution → result_feedback
結果: 將內部能力轉化為可調用的外部服務
狀態: 📋 需要完成
```

---

## 🏗️ 架構設計：從內到外的轉換

### 轉換架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                     用戶層 (External)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Web UI   │  │ REST API │  │  CLI     │  │  SDK     │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│              AI 編排層 (Orchestration Layer)                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         EnhancedDecisionAgent (認知決策)             │    │
│  │  - 理解用戶意圖 (HighLevelIntent)                    │    │
│  │  - 決策「做什麼」(What) 和「為什麼」(Why)              │    │
│  └────────────────────┬─────────────────────────────────┘    │
│                       │                                        │
│  ┌────────────────────▼───────────────────────────────┐      │
│  │      AttackOrchestrator (任務編排)                  │      │
│  │  - 將意圖轉換為執行計劃 (ExecutionPlan)              │      │
│  │  - 決策「怎麼做」(How) - 生成 AST                    │      │
│  └────────────────────┬───────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│           能力註冊與查詢層 (Capability Registry)             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          CapabilityRegistry (能力註冊表)              │  │
│  │  - 從 internal_exploration 載入 692 個能力            │  │
│  │  - 提供能力查詢、搜索、過濾接口                        │  │
│  │  - 管理能力元數據和依賴關係                           │  │
│  └────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              能力執行層 (Execution Layer)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Python能力   │  │ Rust能力     │  │ Go能力       │     │
│  │ (411個)      │  │ (115個)      │  │ (88個)       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ TS能力       │  │ 工具調用     │  │ 腳本執行     │     │
│  │ (78個)       │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              反饋學習層 (Feedback Loop)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      ExternalLoopConnector (外部學習連接器)           │  │
│  │  - 收集執行結果和用戶反饋                              │  │
│  │  - 觸發偏差分析和模型訓練                              │  │
│  │  - 更新 CapabilityRegistry 的能力評分                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 能力分類與對外映射

### 1. 掃描類能力 (Scan Capabilities) - 268 個

#### 1.1 基礎掃描能力
```python
# 能力來源: scan 模組 (268 個能力)
scan_capabilities = {
    "網路掃描": {
        "能力數": 45,
        "語言": "Go + Python",
        "代表能力": [
            "port_scan",           # 端口掃描
            "service_detection",   # 服務識別
            "os_fingerprint",      # 操作系統指紋
            "network_topology"     # 網路拓撲
        ],
        "對外接口": "POST /api/v1/scan/network",
        "AI編排": "NetworkScanOrchestrator"
    },
    
    "Web掃描": {
        "能力數": 62,
        "語言": "Python + TypeScript",
        "代表能力": [
            "url_crawler",         # URL爬蟲
            "js_analysis",         # JS分析
            "api_discovery",       # API發現
            "waf_detection"        # WAF檢測
        ],
        "對外接口": "POST /api/v1/scan/web",
        "AI編排": "WebScanOrchestrator"
    },
    
    "漏洞掃描": {
        "能力數": 89,
        "語言": "Rust + Python",
        "代表能力": [
            "sql_injection_scan",  # SQL注入掃描
            "xss_detection",       # XSS檢測
            "ssrf_check",          # SSRF檢查
            "deserialization"      # 反序列化漏洞
        ],
        "對外接口": "POST /api/v1/scan/vulnerability",
        "AI編排": "VulnerabilityScanOrchestrator"
    },
    
    "敏感信息掃描": {
        "能力數": 72,
        "語言": "Rust (115個方法中的子集)",
        "代表能力": [
            "SensitiveInfoScanner::scan",      # 敏感信息掃描
            "SecretDetector::scan_content",    # 密鑰檢測
            "EntropyDetector::detect_line",    # 熵值檢測
            "Verifier::verify"                 # 驗證器
        ],
        "對外接口": "POST /api/v1/scan/sensitive",
        "AI編排": "SensitiveInfoOrchestrator"
    }
}
```

#### 對外轉換示例
```python
# services/core/aiva_core/core_capabilities/scan_api.py (新增)
from fastapi import APIRouter, Depends
from aiva_core.core_capabilities.capability_registry import get_capability_registry
from aiva_core.task_planning.planner.orchestrator import AttackOrchestrator

router = APIRouter(prefix="/api/v1/scan", tags=["Scan"])

@router.post("/vulnerability")
async def scan_vulnerability(
    target: str,
    scan_type: str = "comprehensive",
    registry = Depends(get_capability_registry)
):
    """對外漏洞掃描接口"""
    
    # 1. 從 CapabilityRegistry 查詢可用的掃描能力
    vuln_capabilities = registry.list_capabilities(
        module="scan",
        filter_func=lambda c: "injection" in c.name or "xss" in c.name
    )
    
    # 2. AI 決策選擇最佳掃描策略
    orchestrator = AttackOrchestrator()
    plan = orchestrator.create_execution_plan({
        "intent": "vulnerability_scan",
        "target": target,
        "available_capabilities": [c.name for c in vuln_capabilities]
    })
    
    # 3. 執行掃描
    results = await execute_scan_plan(plan)
    
    # 4. 返回結果
    return {
        "scan_id": plan.plan_id,
        "target": target,
        "vulnerabilities_found": results["vulnerabilities"],
        "recommendations": results["recommendations"]
    }
```

### 2. 分析類能力 (Analysis Capabilities) - 206 個

#### 2.1 代碼分析能力
```python
# 能力來源: core/aiva_core 模組 (206 個能力)
analysis_capabilities = {
    "靜態代碼分析": {
        "能力數": 58,
        "語言": "Python + TypeScript",
        "代表能力": [
            "ast_parse",              # AST解析
            "control_flow_analysis",  # 控制流分析
            "data_flow_analysis",     # 數據流分析
            "dependency_analysis"     # 依賴分析
        ],
        "對外接口": "POST /api/v1/analysis/static",
        "AI編排": "StaticAnalysisOrchestrator"
    },
    
    "動態分析": {
        "能力數": 42,
        "語言": "Python + Go",
        "代表能力": [
            "trace_execution",        # 執行追蹤
            "memory_analysis",        # 記憶體分析
            "behavior_monitor",       # 行為監控
            "runtime_profiling"       # 運行時分析
        ],
        "對外接口": "POST /api/v1/analysis/dynamic",
        "AI編排": "DynamicAnalysisOrchestrator"
    },
    
    "AI增強分析": {
        "能力數": 106,
        "語言": "Python (cognitive_core)",
        "代表能力": [
            "semantic_search",        # 語義搜索
            "rag_enhanced_query",     # RAG增強查詢
            "neural_reasoning",       # 神經推理
            "pattern_recognition"     # 模式識別
        ],
        "對外接口": "POST /api/v1/analysis/ai",
        "AI編排": "AIAnalysisOrchestrator"
    }
}
```

### 3. 集成類能力 (Integration Capabilities) - 76 個

#### 3.1 工具集成能力
```python
# 能力來源: integration 模組 (76 個能力)
integration_capabilities = {
    "安全工具集成": {
        "能力數": 32,
        "工具類型": ["Metasploit", "Burp Suite", "Nmap", "SQLMap"],
        "代表能力": [
            "metasploit_exec",        # Metasploit執行
            "burp_passive_scan",      # Burp被動掃描
            "nmap_service_scan",      # Nmap服務掃描
            "sqlmap_injection_test"   # SQLMap注入測試
        ],
        "對外接口": "POST /api/v1/tools/{tool_name}/execute",
        "AI編排": "ToolIntegrationOrchestrator"
    },
    
    "數據源集成": {
        "能力數": 24,
        "數據源": ["CVE", "NVD", "GitHub", "Exploit-DB"],
        "代表能力": [
            "cve_lookup",             # CVE查詢
            "exploit_search",         # Exploit搜索
            "github_vuln_check",      # GitHub漏洞檢查
            "threat_intelligence"     # 威脅情報
        ],
        "對外接口": "GET /api/v1/intelligence/{source}/{query}",
        "AI編排": "IntelligenceOrchestrator"
    },
    
    "平台集成": {
        "能力數": 20,
        "平台": ["Jenkins", "GitLab CI", "GitHub Actions"],
        "代表能力": [
            "ci_integration",         # CI集成
            "pipeline_trigger",       # 流水線觸發
            "result_reporting",       # 結果上報
            "ticket_creation"         # 工單創建
        ],
        "對外接口": "POST /api/v1/integration/ci",
        "AI編排": "CIPlatformOrchestrator"
    }
}
```

### 4. 特徵類能力 (Feature Capabilities) - 54 個

#### 4.1 高級特徵能力
```python
# 能力來源: features 模組 (54 個能力)
feature_capabilities = {
    "業務邏輯測試": {
        "能力數": 18,
        "語言": "Python",
        "代表能力": [
            "price_manipulation_test",  # 價格操縱測試
            "workflow_bypass_test",     # 工作流繞過測試
            "privilege_escalation",     # 權限提升
            "race_condition_test"       # 競態條件測試
        ],
        "對外接口": "POST /api/v1/test/business-logic",
        "AI編排": "BusinessLogicOrchestrator"
    },
    
    "AI對話能力": {
        "能力數": 16,
        "語言": "Python",
        "代表能力": [
            "natural_language_query",   # 自然語言查詢
            "intent_understanding",     # 意圖理解
            "dialogue_management",      # 對話管理
            "report_generation"         # 報告生成
        ],
        "對外接口": "POST /api/v1/chat",
        "AI編排": "ConversationalOrchestrator"
    },
    
    "自動化編排": {
        "能力數": 20,
        "語言": "Python + TypeScript",
        "代表能力": [
            "task_scheduling",          # 任務調度
            "resource_allocation",      # 資源分配
            "error_recovery",           # 錯誤恢復
            "performance_optimization"  # 性能優化
        ],
        "對外接口": "POST /api/v1/automation/orchestrate",
        "AI編排": "AutomationOrchestrator"
    }
}
```

---

## 🤖 AI 編排系統設計

### 核心編排組件

#### 1. 高階意圖決策器 (EnhancedDecisionAgent)

```python
# services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py
# 已存在，需要擴展

class EnhancedDecisionAgent:
    """高階決策代理 - 決定「做什麼」和「為什麼」"""
    
    def decide(self, context: DecisionContext) -> HighLevelIntent:
        """
        從用戶請求到高階意圖的轉換
        
        輸入: DecisionContext
            - user_request: "掃描 example.com 的 SQL 注入漏洞"
            - risk_level: RiskLevel.MEDIUM
            - available_tools: ["sqlmap", "manual_test"]
        
        輸出: HighLevelIntent
            - intent_type: IntentType.TEST_VULNERABILITY
            - target: TargetInfo(target_value="example.com", target_type="url")
            - parameters: {"vuln_type": "sql_injection"}
            - constraints: DecisionConstraints(risk_level="medium")
            - confidence: 0.92
            - reasoning: "檢測到SQL注入測試意圖，選擇綜合掃描策略"
        """
        # 已實現，見 enhanced_decision_agent.py line 183-244
        pass

    def _map_user_request_to_intent(self, user_request: str) -> IntentType:
        """將用戶請求映射到意圖類型"""
        intent_patterns = {
            "掃描|scan|檢測|detect": IntentType.SCAN_SURFACE,
            "測試|test|檢查|check.*漏洞|vulnerability": IntentType.TEST_VULNERABILITY,
            "攻擊|exploit|利用": IntentType.EXPLOIT_TARGET,
            "分析|analyze|評估|assess": IntentType.ANALYZE_RESULTS
        }
        
        for pattern, intent_type in intent_patterns.items():
            if re.search(pattern, user_request, re.IGNORECASE):
                return intent_type
        
        return IntentType.SCAN_SURFACE  # 默認
```

#### 2. 任務編排器 (AttackOrchestrator)

```python
# services/core/aiva_core/task_planning/planner/orchestrator.py
# 已存在，需要增強能力查詢功能

class AttackOrchestrator:
    """任務編排器 - 決定「怎麼做」"""
    
    def __init__(self):
        self.ast_parser = ASTParser()
        self.task_converter = TaskConverter()
        self.tool_selector = ToolSelector()
        self.capability_registry = get_capability_registry()  # 新增
    
    def create_execution_plan_from_intent(
        self, 
        intent: HighLevelIntent
    ) -> ExecutionPlan:
        """
        從高階意圖創建執行計劃
        
        輸入: HighLevelIntent
            - intent_type: TEST_VULNERABILITY
            - target: example.com
            - parameters: {"vuln_type": "sql_injection"}
        
        輸出: ExecutionPlan
            - plan_id: "plan_a3f5b2c1"
            - task_sequence: [
                Task1: reconnaissance (nmap, whatweb),
                Task2: vulnerability_scan (sqlmap),
                Task3: verification (manual_check)
              ]
            - tool_decisions: {
                "task1": ToolDecision(tool="nmap", confidence=0.95),
                "task2": ToolDecision(tool="sqlmap", confidence=0.88)
              }
        """
        
        # 1. 從 CapabilityRegistry 查詢相關能力
        relevant_capabilities = self._query_capabilities_for_intent(intent)
        
        # 2. 生成 AST 計劃
        ast_plan = self._generate_ast_from_intent(intent, relevant_capabilities)
        
        # 3. 創建執行計劃
        plan = self.create_execution_plan(ast_plan)
        
        return plan
    
    def _query_capabilities_for_intent(
        self, 
        intent: HighLevelIntent
    ) -> list[CapabilityInfo]:
        """從註冊表查詢相關能力"""
        
        # 基於意圖類型查詢
        if intent.intent_type == IntentType.TEST_VULNERABILITY:
            vuln_type = intent.parameters.get("vuln_type", "")
            
            # 搜索相關掃描能力
            capabilities = self.capability_registry.search_capabilities(vuln_type)
            
            # 過濾掃描模組
            capabilities = [
                c for c in capabilities 
                if c.module == "scan" or "scan" in c.name.lower()
            ]
            
            return capabilities
        
        # 其他意圖類型的查詢邏輯...
        return []
```

#### 3. 能力執行器 (CapabilityExecutor)

```python
# services/core/aiva_core/core_capabilities/capability_executor.py (新增)

class CapabilityExecutor:
    """能力執行器 - 統一的能力調用接口"""
    
    def __init__(self):
        self.registry = get_capability_registry()
        self.python_executor = PythonCapabilityExecutor()
        self.rust_executor = RustCapabilityExecutor()
        self.go_executor = GoCapabilityExecutor()
        self.ts_executor = TypeScriptCapabilityExecutor()
    
    async def execute_capability(
        self,
        capability_name: str,
        parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """執行指定的能力"""
        
        # 1. 從註冊表獲取能力信息
        capability = self.registry.get_capability(capability_name)
        
        if not capability:
            raise ValueError(f"Capability {capability_name} not found")
        
        # 2. 根據語言選擇執行器
        executor = self._select_executor(capability.language)
        
        # 3. 執行能力
        try:
            result = await executor.execute(
                capability=capability,
                parameters=parameters
            )
            
            # 4. 記錄執行結果用於學習
            await self._record_execution_result(capability, result)
            
            return {
                "success": True,
                "result": result,
                "capability": capability.name,
                "execution_time": result.get("execution_time", 0)
            }
            
        except Exception as e:
            # 5. 錯誤處理和記錄
            await self._record_execution_error(capability, e)
            
            return {
                "success": False,
                "error": str(e),
                "capability": capability.name
            }
    
    def _select_executor(self, language: str):
        """選擇對應語言的執行器"""
        executors = {
            "python": self.python_executor,
            "rust": self.rust_executor,
            "go": self.go_executor,
            "typescript": self.ts_executor
        }
        return executors.get(language, self.python_executor)
```

---

## 🔄 完整的數據流示例

### 場景：用戶請求 SQL 注入掃描

#### Step 1: 用戶輸入
```python
# 用戶通過 Web UI 提交請求
user_request = {
    "action": "掃描目標網站的 SQL 注入漏洞",
    "target": "https://example.com",
    "depth": "comprehensive",
    "risk_tolerance": "medium"
}
```

#### Step 2: 意圖理解 (EnhancedDecisionAgent)
```python
# services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py

decision_agent = EnhancedDecisionAgent()

context = DecisionContext()
context.target_info = {"value": "https://example.com", "type": "url"}
context.risk_level = RiskLevel.MEDIUM
context.available_tools = ["sqlmap", "havij", "manual_test"]

# 生成高階意圖
intent = decision_agent.decide(context)

# 輸出:
# HighLevelIntent(
#     intent_type=IntentType.TEST_VULNERABILITY,
#     target=TargetInfo(target_value="https://example.com", target_type="url"),
#     parameters={"vuln_type": "sql_injection", "depth": "comprehensive"},
#     constraints=DecisionConstraints(risk_level="medium", time_limit=3600),
#     confidence=0.92,
#     reasoning="檢測到 SQL 注入測試意圖，目標為 Web 應用..."
# )
```

#### Step 3: 能力查詢 (CapabilityRegistry)
```python
# services/core/aiva_core/core_capabilities/capability_registry.py

registry = get_capability_registry()

# 查詢 SQL 注入相關能力
sql_capabilities = registry.search_capabilities("sql_injection")

# 結果: 找到 23 個相關能力
# [
#     CapabilityInfo(name="sql_injection_scan", module="scan", language="python"),
#     CapabilityInfo(name="SecretDetector::scan_content", module="scan", language="rust"),
#     CapabilityInfo(name="sqlmap_execute", module="integration", language="python"),
#     ...
# ]

# 按模組過濾掃描能力
scan_capabilities = [c for c in sql_capabilities if c.module == "scan"]
# 結果: 15 個掃描類能力
```

#### Step 4: 任務編排 (AttackOrchestrator)
```python
# services/core/aiva_core/task_planning/planner/orchestrator.py

orchestrator = AttackOrchestrator()

# 從意圖創建執行計劃
plan = orchestrator.create_execution_plan_from_intent(intent)

# 輸出:
# ExecutionPlan(
#     plan_id="plan_f8e2d1a3",
#     task_sequence=TaskSequence(tasks=[
#         ExecutableTask(
#             task_id="task_001",
#             name="reconnaissance",
#             capabilities=["port_scan", "service_detection"],
#             status=TaskStatus.PENDING
#         ),
#         ExecutableTask(
#             task_id="task_002",
#             name="sql_injection_scan",
#             capabilities=["sql_injection_scan", "sqlmap_execute"],
#             status=TaskStatus.PENDING,
#             dependencies=["task_001"]
#         ),
#         ExecutableTask(
#             task_id="task_003",
#             name="result_verification",
#             capabilities=["Verifier::verify"],
#             status=TaskStatus.PENDING,
#             dependencies=["task_002"]
#         )
#     ]),
#     tool_decisions={
#         "task_001": ToolDecision(tool="nmap", confidence=0.95),
#         "task_002": ToolDecision(tool="sqlmap", confidence=0.88)
#     }
# )
```

#### Step 5: 能力執行 (CapabilityExecutor)
```python
# services/core/aiva_core/core_capabilities/capability_executor.py

executor = CapabilityExecutor()

# 執行任務序列
for task in plan.task_sequence.get_runnable_tasks():
    for capability_name in task.capabilities:
        result = await executor.execute_capability(
            capability_name=capability_name,
            parameters={
                "target": intent.target.target_value,
                "options": task.parameters
            }
        )
        
        # 結果:
        # {
        #     "success": True,
        #     "result": {
        #         "vulnerabilities_found": 3,
        #         "vulnerability_details": [
        #             {
        #                 "type": "SQL Injection",
        #                 "severity": "High",
        #                 "location": "/api/users?id=1",
        #                 "payload": "1' OR '1'='1"
        #             },
        #             ...
        #         ]
        #     },
        #     "capability": "sql_injection_scan",
        #     "execution_time": 12.5
        # }
        
        # 更新任務狀態
        orchestrator.update_task_status(
            plan, task.task_id, 
            TaskStatus.COMPLETED, 
            result=result
        )
```

#### Step 6: 結果反饋 (ExternalLoopConnector)
```python
# services/core/aiva_core/cognitive_core/external_loop_connector.py

external_loop = ExternalLoopConnector()

# 處理執行結果並觸發學習
learning_result = await external_loop.process_execution_result(
    plan=plan.to_dict(),
    trace=[
        {"task": "reconnaissance", "duration": 5.2, "success": True},
        {"task": "sql_injection_scan", "duration": 12.5, "success": True},
        {"task": "result_verification", "duration": 3.1, "success": True}
    ]
)

# 輸出:
# {
#     "deviations_found": 0,
#     "deviations_significant": False,
#     "training_triggered": False,
#     "weights_updated": False,
#     "performance_improvement": {
#         "sql_injection_scan": {"confidence": 0.88 -> 0.91}
#     }
# }
```

#### Step 7: 用戶響應
```json
{
    "scan_id": "scan_f8e2d1a3_20251116",
    "status": "completed",
    "target": "https://example.com",
    "scan_type": "SQL Injection",
    "results": {
        "vulnerabilities_found": 3,
        "high_severity": 2,
        "medium_severity": 1,
        "details": [
            {
                "type": "SQL Injection",
                "severity": "High",
                "location": "/api/users?id=1",
                "payload": "1' OR '1'='1",
                "recommendation": "使用參數化查詢，避免字符串拼接"
            }
        ]
    },
    "execution_stats": {
        "total_time": "20.8s",
        "tasks_completed": 3,
        "capabilities_used": ["port_scan", "sql_injection_scan", "Verifier::verify"]
    },
    "ai_confidence": 0.91,
    "next_recommended_actions": [
        "深度測試發現的注入點",
        "檢查其他 API 端點",
        "生成詳細的安全報告"
    ]
}
```

---

## 📝 實施步驟

### Phase 1: 基礎設施準備 (1-2 週)

#### 1.1 CapabilityRegistry 初始化
```bash
# 步驟1: 確保能力分析最新
cd C:\D\fold7\AIVA-git
python run_capability_analysis.py

# 步驟2: 測試 CapabilityRegistry 載入
python -c "
import asyncio
from services.core.aiva_core.core_capabilities.capability_registry import initialize_capability_registry

async def test():
    result = await initialize_capability_registry()
    print(f'載入能力數: {result[\"capabilities_loaded\"]}')

asyncio.run(test())
"

# 預期輸出: 載入能力數: 692
```

#### 1.2 創建 CapabilityExecutor
```python
# services/core/aiva_core/core_capabilities/capability_executor.py (新文件)
# 實現多語言能力的統一執行接口
```

#### 1.3 增強 AttackOrchestrator
```python
# 在 services/core/aiva_core/task_planning/planner/orchestrator.py 中新增:
# - create_execution_plan_from_intent() 方法
# - _query_capabilities_for_intent() 方法
# - 與 CapabilityRegistry 的集成
```

### Phase 2: API 層構建 (2-3 週)

#### 2.1 創建統一 API 網關
```python
# services/core/aiva_core/api/gateway.py (新文件)
from fastapi import FastAPI, APIRouter
from aiva_core.api.scan_api import router as scan_router
from aiva_core.api.analysis_api import router as analysis_router
from aiva_core.api.integration_api import router as integration_router

app = FastAPI(title="AIVA API Gateway", version="1.0.0")

# 註冊所有子路由
app.include_router(scan_router)
app.include_router(analysis_router)
app.include_router(integration_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "capabilities": 692}
```

#### 2.2 實現掃描 API
```python
# services/core/aiva_core/api/scan_api.py (新文件)
# 實現:
# - POST /api/v1/scan/network
# - POST /api/v1/scan/web
# - POST /api/v1/scan/vulnerability
# - POST /api/v1/scan/sensitive
```

#### 2.3 實現分析 API
```python
# services/core/aiva_core/api/analysis_api.py (新文件)
# 實現:
# - POST /api/v1/analysis/static
# - POST /api/v1/analysis/dynamic
# - POST /api/v1/analysis/ai
```

### Phase 3: AI 編排集成 (2-3 週)

#### 3.1 擴展 EnhancedDecisionAgent
```python
# 在 enhanced_decision_agent.py 中新增:
# - decide_from_natural_language() 方法
# - _map_user_request_to_capabilities() 方法
# - 與 CapabilityRegistry 的深度集成
```

#### 3.2 實現智能編排策略
```python
# services/core/aiva_core/task_planning/intelligent_orchestration.py (新文件)
# 實現:
# - 基於能力依賴的任務排序
# - 並行任務識別和調度
# - 動態資源分配
# - 失敗恢復策略
```

#### 3.3 集成外部學習閉環
```python
# 確保 ExternalLoopConnector 與 CapabilityRegistry 聯動:
# - 執行結果影響能力評分
# - 成功/失敗統計更新到註冊表
# - 觸發能力優化建議
```

### Phase 4: 測試與驗證 (1-2 週)

#### 4.1 單元測試
```python
# tests/test_capability_executor.py
async def test_execute_python_capability():
    executor = CapabilityExecutor()
    result = await executor.execute_capability(
        capability_name="sql_injection_scan",
        parameters={"target": "http://testphp.vulnweb.com"}
    )
    assert result["success"] == True
    assert "vulnerabilities" in result["result"]

# tests/test_orchestrator_integration.py
async def test_intent_to_execution():
    agent = EnhancedDecisionAgent()
    orchestrator = AttackOrchestrator()
    
    context = DecisionContext()
    context.target_info = {"value": "example.com", "type": "url"}
    
    intent = agent.decide(context)
    plan = orchestrator.create_execution_plan_from_intent(intent)
    
    assert plan.task_sequence is not None
    assert len(plan.task_sequence.tasks) > 0
```

#### 4.2 集成測試
```bash
# 端到端測試腳本
python tests/integration/test_full_scan_flow.py
```

#### 4.3 性能測試
```python
# tests/performance/test_capability_execution_performance.py
# 測試:
# - 能力查詢速度 (目標: <50ms)
# - 編排決策速度 (目標: <200ms)
# - 並發執行能力 (目標: 50+ 並發任務)
```

### Phase 5: 部署與監控 (1 週)

#### 5.1 部署配置
```yaml
# deploy/docker-compose.yml
version: '3.8'
services:
  aiva-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - CAPABILITY_REGISTRY_PATH=/data/capabilities
      - AI_MODEL_PATH=/models
    volumes:
      - ./analysis_results:/data/capabilities
      - ./models:/models
```

#### 5.2 監控儀表板
```python
# services/core/aiva_core/monitoring/capability_dashboard.py
# 實時監控:
# - 能力調用統計
# - 執行成功率
# - 平均執行時間
# - AI 決策信心度分布
```

---

## 📊 預期成果

### 量化指標

| 指標 | 當前狀態 | 目標狀態 | 達成時間 |
|------|---------|---------|----------|
| **可對外能力數** | 35 (5%) | 692 (100%) | Phase 2 完成 |
| **API 端點數** | 11 | 50+ | Phase 2 完成 |
| **AI 編排覆蓋率** | 0% | 80%+ | Phase 3 完成 |
| **平均響應時間** | N/A | <5s | Phase 4 完成 |
| **並發處理能力** | N/A | 50+ 任務 | Phase 5 完成 |
| **用戶滿意度** | N/A | 4.5/5.0 | Phase 5 後 1 個月 |

### 質化成果

1. **完整的能力服務化**
   - 692 個內部能力全部可通過 API 調用
   - 統一的能力註冊、查詢、執行機制
   - 多語言能力的透明化執行

2. **智能的 AI 編排**
   - 自然語言理解用戶意圖
   - 智能選擇和組合能力
   - 動態優化執行策略

3. **閉環的學習機制**
   - 執行結果自動反饋到能力評分
   - 持續優化能力選擇和編排策略
   - AI 模型根據實際效果自我調整

4. **企業級的可用性**
   - RESTful API 接口
   - 完整的錯誤處理和重試機制
   - 詳細的監控和日誌
   - 水平擴展能力

---

## 🎯 關鍵成功因素

### 技術層面

1. **CapabilityRegistry 的穩定性**
   - 必須確保 692 個能力的完整載入
   - 查詢性能必須在 50ms 內
   - 支持動態更新和熱重載

2. **AI 編排的準確性**
   - 意圖理解準確率 >85%
   - 能力選擇準確率 >80%
   - 任務編排效率 >70%

3. **執行器的可靠性**
   - 多語言執行的成功率 >95%
   - 錯誤恢復和重試機制完善
   - 資源隔離和並發控制

### 流程層面

1. **漸進式實施**
   - 優先實現高頻使用的掃描類能力
   - 逐步擴展到分析、集成類能力
   - 最後完善AI增強功能

2. **持續驗證**
   - 每個 Phase 完成後進行完整測試
   - 收集早期用戶反饋
   - 基於數據進行調整優化

3. **文檔先行**
   - API 文檔與代碼同步更新
   - 提供豐富的使用示例
   - 維護troubleshooting指南

---

## 📚 參考文檔

### 現有架構文檔
- `services/core/aiva_core/README.md` - AIVA Core 架構總覽
- `services/core/aiva_core/cognitive_core/README.md` - 認知核心模組
- `services/core/aiva_core/task_planning/README.md` - 任務規劃模組
- `services/core/aiva_core/core_capabilities/README.md` - 核心能力模組

### 能力分析文檔
- `P0_IMPLEMENTATION_COMPLETION_REPORT.md` - P0 實施完成報告
- `VERIFIED_COMPLETE_GUIDE.md` - 完整驗證指南
- `analysis_results/baseline.json` - 能力基準數據

### API 設計參考
- `services/core/aiva_core/ui_panel/server.py` - 現有 API 服務器
- `services/core/aiva_core/ui_panel/ai_ui_schemas.py` - API 數據模式

---

**報告生成**: 2025-11-16 21:30:00  
**作者**: GitHub Copilot (Claude Sonnet 4.5)  
**狀態**: 📋 規劃完成，待評審和實施

**下一步行動**: 
1. ✅ 審閱本報告
2. 📋 確認實施優先級
3. 🚀 啟動 Phase 1 開發
