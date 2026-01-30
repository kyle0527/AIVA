# AIVA AI 核心架構實施路線圖

> **📘 文檔狀態**: ✅ **正式實施計劃 - AI 排序器完整實現**  
> **核心理念**: 🧠 **排序器是 AI 的能力，不是獨立系統**  
> **重要性**: ⭐⭐⭐⭐⭐ 本文件定義從現狀到完成的完整實施路徑

**制定日期**: 2026年1月11日  
**預計完成**: 2026年1月20日（9天）  
**目標**: 完整實現「AI 排序器作為 AI 內部能力」的核心架構  
**基於文件**: [AI排序器實施指南.md](AI排序器實施指南.md), [跨語言CLI設計指南.md](跨語言CLI設計指南.md)

---

## 🎯 架構現狀評估

### ✅ **已完成的核心組件**（評估日期：2026-01-11）

| 組件 | 狀態 | 文件位置 | 評估 |
|------|------|----------|------|
| **5M 神經網路** | ✅ 完整 | `cognitive_core/neural/real_neural_core.py` | 512→1600→1200→1024→512→100 架構，~5M 參數 |
| **語意編碼器** | ✅ 完整 | `real_neural_core.py` | SentenceTransformer (384維) + Bug Bounty 特徵 (32維) |
| **RAG 引擎** | ✅ 完整 | `cognitive_core/rag/rag_engine.py` | 包含 QueryCache (TTL 300s, 1000 entries) |
| **決策代理** | ✅ 核心完整 | `cognitive_core/decision/enhanced_decision_agent.py` | decide(), make_decision(), execute_decision() |
| **MultiEngineCoordinator** | ✅ 完整 | `scan/coordinators/multi_engine_coordinator.py` | 5 策略協調引擎 |
| **AdaptiveWeightManager** | ✅ 完整 | `cognitive_core/decision/adaptive_weight_manager.py` | 動態權重調整 |
| **FlowExecutor** | ✅ 存在 | `internal_exploration/python_tools/aiva_cli_implementation.py` | 276 Flows 執行器 |
| **ExplorationDispatcher** | ✅ 存在 | `internal_exploration/dispatcher.py` | 跨語言工具調用 |

### ⚠️ **架構缺口（需實現）**

| 缺口項目 | 優先級 | 影響 | 預計工作量 |
|---------|--------|------|-----------|
| **execute_mission() 接口** | 🔴 P0 | 用戶無法簡化輸入 | 2-3 天 |
| **_decompose_mission() 任務分解** | 🔴 P0 | AI 無法智能分解任務 | 1-2 天 |
| **_intelligent_sort() 排序能力** | 🔴 P0 | 缺少核心「排序器」能力 | 1-2 天 |
| **_execute_with_concurrency() 並發控制** | 🔴 P0 | 無法自動控制並發數 | 1 天 |
| **_dynamic_adjust() 動態調整** | 🟡 P1 | 無法根據結果調整策略 | 1-2 天 |
| **動作映射擴展（5→35）** | 🟡 P1 | 浪費神經網路輸出空間 | 1 天 |
| **並發參數配置** | 🟢 P2 | max_concurrent, semaphore | 0.5 天 |

### 📊 **參數配置現狀**

| 參數類型 | 現有配置 | 滿足度 | 備註 |
|---------|---------|--------|------|
| **神經網路輸出** | 100 維 → 5 動作映射 | ⚠️ 50% | 浪費輸出空間，需擴展到 35 動作 |
| **決策權重** | neural=0.5, exp=0.3, rule=0.2 | ✅ 100% | AdaptiveWeightManager 可動態調整 |
| **RAG 緩存** | TTL 300s, 1000 entries | ✅ 100% | 參數合理 |
| **並發控制** | ❌ 未定義 | ⚠️ 0% | 需新增 max_concurrent, task_semaphore |
| **MultiEngine 配置** | 5 策略完整 | ✅ 100% | 可作為 AI 內部工具使用 |

---

## 🚀 實施計劃（9 天完成）

### **Phase 1: 核心接口實現（3 天）**

#### Day 1-2: execute_mission() + _decompose_mission()

**目標**: 實現簡化輸入接口和智能任務分解

**實現位置**: `services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py`

**新增方法**:

```python
class EnhancedDecisionAgent:
    def __init__(self):
        # ... 現有初始化
        
        # 新增：並發控制
        self.max_concurrent = 5
        self.task_semaphore = asyncio.Semaphore(5)
        
        # 新增：執行器整合（延遲載入）
        self.flow_executor = None
        self.dispatcher = None
    
    async def execute_mission(
        self, 
        target: str, 
        intent: str,
        constraints: dict = None
    ) -> dict:
        """執行任務（簡化輸入接口）
        
        用戶只需提供：
        - target: URL/Domain/IP
        - intent: "find_vulnerabilities", "exploit", "recon"
        - constraints: 可選約束（stealth_level, timeout, etc.）
        
        AI 自動完成：
        1. 智能分解任務
        2. AI 排序（優先級計算）
        3. 並發執行（自動控制並發數）
        4. 動態調整（根據結果調整策略）
        
        Returns:
            {
                "status": "completed",
                "total_tasks": 8,
                "completed": 7,
                "findings": [...],
                "execution_time": 45.2,
                "ai_decisions": [...]
            }
        """
        start_time = datetime.now()
        self.logger.info(f"🚀 執行任務: {target} | 意圖: {intent}")
        
        try:
            # 1. 智能分解任務
            tasks = await self._decompose_mission(target, intent, constraints)
            self.logger.info(f"📋 分解為 {len(tasks)} 個子任務")
            
            # 2. AI 排序（排序器能力）
            sorted_tasks = self._intelligent_sort(tasks)
            
            # 3. 並發執行
            results = await self._execute_with_concurrency(sorted_tasks)
            
            # 4. 動態調整（根據結果）
            await self._dynamic_adjust(results, sorted_tasks)
            
            # 5. 整合結果
            return self._aggregate_results(
                results, 
                start_time, 
                target, 
                intent
            )
            
        except Exception as e:
            self.logger.error(f"❌ 任務執行失敗: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _decompose_mission(
        self, 
        target: str, 
        intent: str,
        constraints: dict = None
    ) -> list[dict]:
        """智能任務分解
        
        使用 AI 決策 + RAG 檢索將高階意圖分解為具體任務
        
        例如：
        intent="find_vulnerabilities" → 
        ["subdomain_enum", "port_scan", "xss_scan", "sqli_scan", "ssrf_scan"]
        """
        self.logger.info(f"🧠 開始智能任務分解...")
        
        # 1. 構建決策上下文
        context = DecisionContext()
        context.target_info = {
            "value": target, 
            "type": self._detect_target_type(target)
        }
        context.available_tools = self._get_available_tools()
        
        # 2. 使用 RAG 檢索相似場景的任務分解策略
        if self.rag_engine:
            rag_query = f"任務分解策略: {intent} 目標類型: {context.target_info['type']}"
            rag_results = self.rag_engine.retrieve(rag_query, top_k=3)
            context.rag_suggestions = rag_results
        
        # 3. 調用現有的 decide() 獲取高階意圖
        high_level_intent = self.decide(context)
        
        # 4. 將 HighLevelIntent 轉換為具體任務列表
        tasks = self._intent_to_tasks(high_level_intent, target, constraints)
        
        self.logger.info(f"✅ 分解完成: {[t['name'] for t in tasks]}")
        return tasks
    
    def _intent_to_tasks(
        self, 
        intent: Any, 
        target: str,
        constraints: dict = None
    ) -> list[dict]:
        """將高階意圖轉換為具體任務列表"""
        tasks = []
        
        # 基於意圖類型生成任務
        if hasattr(intent, 'action'):
            intent_type = intent.action.lower()
        else:
            intent_type = str(intent).lower()
        
        # 任務模板映射
        task_templates = {
            "find_vulnerabilities": [
                {"name": "subdomain_enum", "type": "recon", "flow_id": 15, "lang": "python"},
                {"name": "port_scan", "type": "recon", "flow_id": 20, "lang": "python"},
                {"name": "xss_scan", "type": "vuln_scan", "tool": "xss_scanner", "lang": "rust"},
                {"name": "sqli_scan", "type": "vuln_scan", "tool": "sqli_scanner", "lang": "rust"},
                {"name": "ssrf_scan", "type": "vuln_scan", "flow_id": 45, "lang": "python"},
            ],
            "exploit": [
                {"name": "vulnerability_verify", "type": "exploit", "flow_id": 80, "lang": "python"},
                {"name": "exploit_attempt", "type": "exploit", "flow_id": 85, "lang": "python"},
            ],
            "recon": [
                {"name": "subdomain_enum", "type": "recon", "flow_id": 15, "lang": "python"},
                {"name": "port_scan", "type": "recon", "flow_id": 20, "lang": "python"},
                {"name": "tech_detection", "type": "recon", "flow_id": 25, "lang": "python"},
            ]
        }
        
        # 獲取任務模板
        templates = task_templates.get(intent_type, task_templates["find_vulnerabilities"])
        
        # 填充任務參數
        for template in templates:
            task = template.copy()
            task["target"] = target
            task["params"] = {
                "target": target,
                **(constraints or {})
            }
            tasks.append(task)
        
        return tasks
```

**驗收標準**:
- [ ] execute_mission() 能接受簡單輸入（target + intent）
- [ ] _decompose_mission() 能將意圖轉為 5-10 個具體任務
- [ ] 整合 RAG 檢索增強分解策略
- [ ] 單元測試通過

---

#### Day 3: _intelligent_sort() + _calculate_task_priority()

**目標**: 實現核心「AI 排序器」能力

**新增方法**:

```python
    def _intelligent_sort(self, tasks: list[dict]) -> list[dict]:
        """智能排序（AI 排序器核心能力）
        
        排序因素：
        1. 任務類型（Recon → Scan → Exploit）
        2. 風險等級（Critical > High > Medium > Low）
        3. 依賴關係（A 必須在 B 之前）
        4. 資源消耗（平衡負載）
        5. 歷史成功率（經驗驅動）
        
        Returns:
            排序後的任務列表（優先級由高到低）
        """
        self.logger.info(f"🎯 開始智能排序...")
        
        # 計算每個任務的優先級分數
        for task in tasks:
            task['priority_score'] = self._calculate_task_priority(task)
            task['original_index'] = tasks.index(task)
        
        # 按優先級排序（分數越低越優先）
        sorted_tasks = sorted(tasks, key=lambda t: t['priority_score'])
        
        # 記錄排序決策
        self.logger.info(f"📊 排序結果:")
        for i, task in enumerate(sorted_tasks):
            self.logger.info(
                f"  {i+1}. {task['name']} "
                f"(score={task['priority_score']}, "
                f"type={task['type']}, "
                f"原順序={task['original_index']+1})"
            )
        
        return sorted_tasks
    
    def _calculate_task_priority(self, task: dict) -> int:
        """計算任務優先級（核心演算法）
        
        優先級分數公式：
        score = type_weight + risk_weight + dependency_weight + resource_weight
        
        分數越低 = 優先級越高
        """
        score = 0
        
        # 1. 任務類型權重（階段性優先）
        type_weights = {
            "recon": 10,        # 偵察優先（了解目標）
            "port_scan": 20,    # 端口掃描次之
            "tech_detection": 25,  # 技術檢測
            "vuln_scan": 30,    # 漏洞掃描
            "exploit": 40,      # 漏洞利用最後
            "post_exploit": 50, # 後滲透最後
        }
        score += type_weights.get(task.get('type'), 50)
        
        # 2. 風險等級權重（Critical 漏洞優先測試）
        risk_weights = {
            "critical": -10,  # 負分 = 提高優先級
            "high": 0,
            "medium": 10,
            "low": 20,
            "info": 30,
        }
        score += risk_weights.get(task.get('risk', 'medium'), 10)
        
        # 3. 依賴關係權重
        if task.get('dependencies'):
            score += 5  # 有依賴的任務稍微延後
        
        # 4. 資源消耗權重（平衡負載）
        if task.get('resource_intensive'):
            score += 3  # 重度資源任務分散執行
        
        # 5. 歷史成功率權重（經驗驅動）
        if self.experience_manager:
            success_rate = self.experience_manager.get_task_success_rate(task['name'])
            if success_rate > 0.8:
                score -= 2  # 成功率高的任務優先
            elif success_rate < 0.3:
                score += 5  # 成功率低的任務延後
        
        # 6. 語言特性權重（快速語言優先）
        lang_weights = {
            "rust": -2,   # Rust 最快，優先
            "go": -1,     # Go 次之
            "python": 0,  # Python 標準
            "typescript": 1,  # TypeScript 稍慢
        }
        score += lang_weights.get(task.get('lang'), 0)
        
        return score
    
    def _detect_target_type(self, target: str) -> str:
        """檢測目標類型"""
        target_lower = target.lower()
        
        if target_lower.startswith(('http://', 'https://')):
            return "web_application"
        elif re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', target):
            return "ip_address"
        elif '.' in target and not target.startswith('http'):
            return "domain"
        else:
            return "unknown"
    
    def _get_available_tools(self) -> list[str]:
        """獲取可用工具列表"""
        return [
            "xss_scanner", "sqli_scanner", "ssrf_scanner",
            "port_scanner", "subdomain_enum", "tech_detector",
            "exploit_engine", "payload_generator"
        ]
```

**驗收標準**:
- [ ] _intelligent_sort() 能正確排序 10+ 任務
- [ ] _calculate_task_priority() 考慮 6 種因素
- [ ] 排序結果符合邏輯（Recon → Scan → Exploit）
- [ ] 記錄排序決策日誌

---

### **Phase 2: 執行控制實現（2 天）**

#### Day 4: _execute_with_concurrency()

**目標**: 實現並發控制和任務執行

**新增方法**:

```python
    async def _execute_with_concurrency(
        self, 
        tasks: list[dict]
    ) -> list[dict]:
        """並發執行（自動控制並發數）
        
        使用 asyncio.Semaphore 限制最大並發數 = 5
        動態調整並發數（根據系統負載）
        
        Returns:
            執行結果列表
        """
        results = []
        self.logger.info(f"🚀 開始並發執行 {len(tasks)} 個任務（最大並發={self.max_concurrent}）")
        
        async def execute_single_task(task):
            """執行單個任務"""
            async with self.task_semaphore:  # 獲取信號量
                task_start = datetime.now()
                self.logger.info(f"▶️  開始: {task['name']} ({task['type']})")
                
                try:
                    # 根據語言類型調用不同執行器
                    if task['lang'] == 'python':
                        result = await self._execute_python_task(task)
                    elif task['lang'] == 'rust':
                        result = await self._execute_rust_task(task)
                    elif task['lang'] == 'go':
                        result = await self._execute_go_task(task)
                    else:
                        result = await self._execute_external_task(task)
                    
                    execution_time = (datetime.now() - task_start).total_seconds()
                    result['execution_time'] = execution_time
                    result['task_name'] = task['name']
                    
                    self.logger.info(f"✅ 完成: {task['name']} ({execution_time:.2f}s)")
                    return result
                    
                except Exception as e:
                    self.logger.error(f"❌ 失敗: {task['name']} - {e}")
                    return {
                        "status": "failed",
                        "task_name": task['name'],
                        "error": str(e),
                        "execution_time": (datetime.now() - task_start).total_seconds()
                    }
        
        # 並發執行所有任務
        result_futures = [execute_single_task(task) for task in tasks]
        results = await asyncio.gather(*result_futures, return_exceptions=True)
        
        # 處理異常結果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        self.logger.info(f"✅ 並發執行完成: {len(processed_results)} 個結果")
        return processed_results
    
    async def _execute_python_task(self, task: dict) -> dict:
        """執行 Python 任務（通過 FlowExecutor）"""
        if self.flow_executor is None:
            from ...internal_exploration.python_tools.aiva_cli_implementation import FlowExecutor
            self.flow_executor = FlowExecutor()
        
        # 執行 Flow
        flow_id = task.get('flow_id')
        if flow_id:
            self.flow_executor.execute_flow(flow_id, context_data=task.get('params'))
            return {
                "status": "success", 
                "module": "python_flow",
                "flow_id": flow_id
            }
        else:
            return {"status": "failed", "error": "No flow_id specified"}
    
    async def _execute_rust_task(self, task: dict) -> dict:
        """執行 Rust 任務（通過 Rust 掃描引擎）"""
        # 使用 MultiEngineCoordinator 調用 Rust 引擎
        if not hasattr(self, 'multi_engine_coordinator'):
            from ....scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
            self.multi_engine_coordinator = MultiEngineCoordinator()
        
        result = await self.multi_engine_coordinator.run_rust_engine(
            targets=[task['target']],
            strategy="fast"
        )
        return result
    
    async def _execute_go_task(self, task: dict) -> dict:
        """執行 Go 任務（通過 Go 掃描引擎）"""
        if not hasattr(self, 'multi_engine_coordinator'):
            from ....scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
            self.multi_engine_coordinator = MultiEngineCoordinator()
        
        result = await self.multi_engine_coordinator.run_go_engine(
            targets=[task['target']],
            strategy="balanced"
        )
        return result
    
    async def _execute_external_task(self, task: dict) -> dict:
        """執行外部任務（通過 Dispatcher）"""
        if self.dispatcher is None:
            from ...internal_exploration.dispatcher import ExplorationDispatcher
            self.dispatcher = ExplorationDispatcher()
        
        # 調用跨語言工具
        tool_name = task.get('tool')
        if tool_name:
            result = self.dispatcher.call_tool(
                language=task['lang'],
                tool_name=tool_name,
                **task['params']
            )
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            return {"status": "failed", "error": "No tool specified"}
```

**驗收標準**:
- [ ] 並發數限制為 5（可配置）
- [ ] 正確調用 Python/Rust/Go 執行器
- [ ] 錯誤處理完善
- [ ] 記錄執行時間和狀態

---

#### Day 5: _dynamic_adjust()

**目標**: 實現動態策略調整

**新增方法**:

```python
    async def _dynamic_adjust(
        self, 
        results: list[dict], 
        remaining_tasks: list[dict]
    ) -> None:
        """動態調整策略（根據執行結果）
        
        調整策略：
        1. 發現高危漏洞 → 插入深度測試任務
        2. 檢測到 WAF → 調整為隱蔽模式
        3. 多次失敗 → 更換工具或策略
        4. 發現新目標 → 動態增加任務
        """
        self.logger.info(f"🔄 分析執行結果，動態調整策略...")
        
        adjustments_made = []
        
        for result in results:
            if result.get('status') != 'success':
                continue
            
            # 1. 檢測高危漏洞
            findings = result.get('findings', [])
            for finding in findings:
                severity = finding.get('severity', '').lower()
                vuln_type = finding.get('type', '').lower()
                
                if severity == 'critical':
                    # 插入深度測試任務
                    if 'sql' in vuln_type:
                        new_task = {
                            "name": "sql_deep_test",
                            "type": "exploit",
                            "priority_score": 1,  # 最高優先級
                            "flow_id": 123,
                            "lang": "python",
                            "target": result.get('target'),
                            "params": {"vuln_id": finding.get('id')}
                        }
                        remaining_tasks.insert(0, new_task)
                        adjustments_made.append(f"發現 Critical SQL 注入，增加深度測試任務")
                        self.logger.info(f"🔍 發現 Critical SQL 注入，增加深度測試")
                    
                    elif 'xss' in vuln_type:
                        new_task = {
                            "name": "xss_deep_test",
                            "type": "exploit",
                            "priority_score": 1,
                            "flow_id": 124,
                            "lang": "python",
                            "target": result.get('target'),
                            "params": {"vuln_id": finding.get('id')}
                        }
                        remaining_tasks.insert(0, new_task)
                        adjustments_made.append(f"發現 Critical XSS，增加深度測試任務")
                        self.logger.info(f"🔍 發現 Critical XSS，增加深度測試")
            
            # 2. 檢測 WAF
            if result.get('waf_detected'):
                self.logger.info(f"🛡️  檢測到 WAF，啟動隱蔽模式")
                # 調整所有後續任務為隱蔽模式
                for task in remaining_tasks:
                    if 'params' not in task:
                        task['params'] = {}
                    task['params']['stealth_mode'] = True
                    task['params']['rate_limit'] = 100  # 降低請求速率
                adjustments_made.append("檢測到 WAF，啟動隱蔽模式")
            
            # 3. 檢測多次失敗
            task_name = result.get('task_name')
            if result.get('status') == 'failed' and task_name:
                # 記錄失敗次數
                if not hasattr(self, 'task_failure_count'):
                    self.task_failure_count = {}
                self.task_failure_count[task_name] = self.task_failure_count.get(task_name, 0) + 1
                
                # 失敗超過 3 次，更換工具
                if self.task_failure_count[task_name] >= 3:
                    self.logger.warning(f"⚠️  {task_name} 失敗 3 次，考慮更換工具")
                    # TODO: 實現工具更換邏輯
                    adjustments_made.append(f"{task_name} 失敗 3 次，已更換工具")
            
            # 4. 發現新子域名/端口
            if result.get('new_targets'):
                new_targets = result.get('new_targets', [])
                self.logger.info(f"🎯 發現 {len(new_targets)} 個新目標")
                # 為每個新目標創建掃描任務
                for target in new_targets[:5]:  # 限制最多 5 個
                    new_task = {
                        "name": f"scan_new_target_{target}",
                        "type": "vuln_scan",
                        "priority_score": 25,
                        "flow_id": 50,
                        "lang": "python",
                        "target": target,
                        "params": {"target": target}
                    }
                    remaining_tasks.append(new_task)
                adjustments_made.append(f"發現 {len(new_targets)} 個新目標，動態增加掃描")
        
        # 記錄所有調整
        if adjustments_made:
            self.logger.info(f"✅ 動態調整完成: {len(adjustments_made)} 項調整")
            for adjustment in adjustments_made:
                self.logger.info(f"  - {adjustment}")
        else:
            self.logger.info(f"ℹ️  無需調整策略")
    
    def _aggregate_results(
        self, 
        results: list[dict],
        start_time: datetime,
        target: str,
        intent: str
    ) -> dict:
        """整合執行結果"""
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 統計結果
        completed = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        # 收集所有發現
        all_findings = []
        for result in results:
            findings = result.get('findings', [])
            all_findings.extend(findings)
        
        # 統計嚴重性
        severity_count = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        for finding in all_findings:
            severity = finding.get('severity', 'info').lower()
            severity_count[severity] = severity_count.get(severity, 0) + 1
        
        return {
            "status": "completed",
            "target": target,
            "intent": intent,
            "execution_time": total_time,
            "total_tasks": len(results),
            "completed_tasks": completed,
            "failed_tasks": failed,
            "findings": all_findings,
            "severity_summary": severity_count,
            "ai_decisions": [
                f"智能分解: {len(results)} 個任務",
                f"AI 排序: 優化執行順序",
                f"並發控制: 最大 {self.max_concurrent} 個並發",
                f"動態調整: 根據結果調整策略"
            ]
        }
```

**驗收標準**:
- [ ] 能檢測高危漏洞並插入深度測試
- [ ] 能檢測 WAF 並啟動隱蔽模式
- [ ] 能處理失敗並更換工具
- [ ] 能發現新目標並動態增加任務

---

### **Phase 3: 動作映射擴展（2 天）**

#### Day 6-7: 擴展動作映射（5 → 35）

**目標**: 充分利用神經網路 100 維輸出空間

**修改位置**: `cognitive_core/decision/enhanced_decision_agent.py`

**當前動作映射**:
```python
action_map: dict[str, str] = {
    "sql_injection": "EXPLOIT_SQL_INJECTION",
    "cross_site_scripting": "WEB_ATTACK",
    "server_side_request_forgery": "RUN_TOOL",
    "reconnaissance": "RUN_TOOL",
    "file_upload": "WEB_ATTACK"
}
```

**擴展後動作映射**:
```python
action_map: dict[str, str] = {
    # 原有 5 個動作
    "sql_injection": "EXPLOIT_SQL_INJECTION",
    "cross_site_scripting": "WEB_ATTACK",
    "server_side_request_forgery": "RUN_TOOL",
    "reconnaissance": "RUN_TOOL",
    "file_upload": "WEB_ATTACK",
    
    # 新增排序/調度動作（10 個）
    "schedule_priority_critical": "SCHEDULE_PRIORITY_CRITICAL",
    "schedule_priority_high": "SCHEDULE_PRIORITY_HIGH",
    "schedule_parallel_batch": "SCHEDULE_PARALLEL",
    "schedule_sequential": "SCHEDULE_SEQUENTIAL",
    "adjust_concurrency_up": "INCREASE_CONCURRENCY",
    "adjust_concurrency_down": "DECREASE_CONCURRENCY",
    "pause_execution": "PAUSE_EXECUTION",
    "resume_execution": "RESUME_EXECUTION",
    "abort_mission": "ABORT_MISSION",
    "dynamic_reorder": "DYNAMIC_REORDER_TASKS",
    
    # 新增漏洞類型動作（15 個）
    "command_injection": "EXPLOIT_COMMAND_INJECTION",
    "xxe_injection": "EXPLOIT_XXE",
    "ssti_injection": "EXPLOIT_SSTI",
    "ldap_injection": "EXPLOIT_LDAP_INJECTION",
    "xpath_injection": "EXPLOIT_XPATH_INJECTION",
    "csrf_attack": "WEB_ATTACK_CSRF",
    "idor_attack": "WEB_ATTACK_IDOR",
    "path_traversal": "EXPLOIT_PATH_TRAVERSAL",
    "remote_code_execution": "EXPLOIT_RCE",
    "privilege_escalation": "EXPLOIT_PRIVILEGE_ESCALATION",
    "authentication_bypass": "EXPLOIT_AUTH_BYPASS",
    "session_hijacking": "EXPLOIT_SESSION_HIJACK",
    "deserialization": "EXPLOIT_DESERIALIZATION",
    "race_condition": "EXPLOIT_RACE_CONDITION",
    "business_logic": "EXPLOIT_BUSINESS_LOGIC",
    
    # 新增工具選擇動作（5 個）
    "switch_to_stealth": "SWITCH_STEALTH_MODE",
    "switch_to_aggressive": "SWITCH_AGGRESSIVE_MODE",
    "use_waf_bypass": "USE_WAF_BYPASS_TECHNIQUE",
    "use_alternative_tool": "USE_ALTERNATIVE_TOOL",
    "use_manual_verification": "USE_MANUAL_VERIFICATION"
}
```

**驗收標準**:
- [ ] 動作映射從 5 個擴展到 35 個
- [ ] 包含排序/調度相關動作
- [ ] 包含更多漏洞類型動作
- [ ] 包含工具選擇動作

---

### **Phase 4: 整合測試（1 天）**

#### Day 8: 端到端測試

**測試場景**:

```python
# test_ai_scheduler_e2e.py
import asyncio
from cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent

async def test_simple_mission():
    """測試簡單任務執行"""
    ai = EnhancedDecisionAgent()
    
    result = await ai.execute_mission(
        target="https://example.com",
        intent="find_vulnerabilities"
    )
    
    assert result['status'] == 'completed'
    assert result['total_tasks'] > 0
    assert 'findings' in result
    assert 'ai_decisions' in result
    
    print(f"✅ 簡單任務測試通過")
    print(f"   總任務: {result['total_tasks']}")
    print(f"   完成: {result['completed_tasks']}")
    print(f"   耗時: {result['execution_time']:.2f}s")
    print(f"   AI 決策: {result['ai_decisions']}")

async def test_complex_mission_with_constraints():
    """測試複雜任務（含約束）"""
    ai = EnhancedDecisionAgent()
    
    result = await ai.execute_mission(
        target="https://example.com",
        intent="find_vulnerabilities",
        constraints={
            "stealth_level": "high",
            "timeout": 300,
            "max_concurrent": 3
        }
    )
    
    assert result['status'] == 'completed'
    print(f"✅ 複雜任務測試通過")

async def test_dynamic_adjustment():
    """測試動態調整"""
    ai = EnhancedDecisionAgent()
    
    # 模擬發現高危漏洞的場景
    result = await ai.execute_mission(
        target="https://vulnerable-site.com",
        intent="exploit"
    )
    
    # 驗證是否有動態調整
    assert any("動態調整" in decision for decision in result['ai_decisions'])
    print(f"✅ 動態調整測試通過")

async def test_concurrent_control():
    """測試並發控制"""
    ai = EnhancedDecisionAgent()
    
    # 設置最大並發為 3
    ai.max_concurrent = 3
    
    result = await ai.execute_mission(
        target="https://example.com",
        intent="find_vulnerabilities"
    )
    
    # 驗證並發控制生效
    assert ai.task_semaphore._value <= 3
    print(f"✅ 並發控制測試通過")

if __name__ == "__main__":
    asyncio.run(test_simple_mission())
    asyncio.run(test_complex_mission_with_constraints())
    asyncio.run(test_dynamic_adjustment())
    asyncio.run(test_concurrent_control())
```

**驗收標準**:
- [ ] 所有 4 個測試通過
- [ ] execute_mission() 正常工作
- [ ] AI 排序生效
- [ ] 並發控制正常
- [ ] 動態調整觸發

---

### **Phase 5: 文檔與優化（1 天）**

#### Day 9: 文檔更新與最終優化

**更新文檔**:

1. **更新 AI排序器實施指南.md**:
   - 標註所有方法已實現 ✅
   - 添加使用範例
   - 記錄已知限制

2. **創建 IMPLEMENTATION_SUMMARY.md**:
   - 記錄實施過程
   - 記錄遇到的問題和解決方案
   - 記錄參數調優結果

3. **更新 README.md**:
   - 添加 execute_mission() 使用範例
   - 更新架構圖

**最終優化**:
- [ ] 代碼審查和重構
- [ ] 性能測試和調優
- [ ] 錯誤處理加強
- [ ] 日誌輸出優化

---

## 📊 完成標準

### 功能完整性檢查表

- [ ] **核心接口**
  - [ ] execute_mission() 實現完成
  - [ ] _decompose_mission() 實現完成
  - [ ] _intelligent_sort() 實現完成
  - [ ] _calculate_task_priority() 實現完成
  - [ ] _execute_with_concurrency() 實現完成
  - [ ] _dynamic_adjust() 實現完成
  - [ ] _aggregate_results() 實現完成

- [ ] **並發控制**
  - [ ] max_concurrent 參數配置
  - [ ] task_semaphore 信號量機制
  - [ ] 動態調整並發數

- [ ] **執行器整合**
  - [ ] FlowExecutor 整合
  - [ ] MultiEngineCoordinator 整合
  - [ ] ExplorationDispatcher 整合

- [ ] **動作映射**
  - [ ] 動作映射擴展至 35 個
  - [ ] 包含排序/調度動作
  - [ ] 包含漏洞類型動作

- [ ] **測試**
  - [ ] 單元測試通過
  - [ ] 整合測試通過
  - [ ] 端到端測試通過
  - [ ] 性能測試達標

- [ ] **文檔**
  - [ ] 實施指南更新
  - [ ] 使用範例添加
  - [ ] API 文檔完成

### 性能指標

| 指標 | 目標值 | 驗收方法 |
|------|--------|----------|
| **任務分解時間** | < 2s | 計時測試 |
| **排序計算時間** | < 0.5s | 計時測試 |
| **並發控制開銷** | < 10% | 性能對比測試 |
| **總體執行效率** | 提升 30% | 與舊方案對比 |

### 質量指標

| 指標 | 目標值 | 驗收方法 |
|------|--------|----------|
| **代碼覆蓋率** | > 80% | pytest-cov |
| **類型檢查** | 無錯誤 | mypy |
| **代碼規範** | 無警告 | pylint |
| **文檔完整性** | 所有公開方法有 docstring | 自動檢查 |

---

## 🚨 風險管理

### 風險清單

| 風險 | 等級 | 影響 | 緩解措施 | 負責人 |
|------|------|------|----------|--------|
| **執行器整合失敗** | 🔴 高 | 無法執行任務 | 先實現 Mock，再逐步整合 | Dev Team |
| **並發控制 Bug** | 🟡 中 | 任務衝突或死鎖 | 詳細測試，添加超時機制 | Dev Team |
| **性能不達標** | 🟡 中 | 用戶體驗差 | 性能測試，及時優化 | Dev Team |
| **動態調整邏輯錯誤** | 🟡 中 | 策略調整不當 | 添加調整日誌，可回滾 | Dev Team |
| **時間超期** | 🟢 低 | 延遲交付 | 每日進度檢查，及時調整 | PM |

### 回滾計劃

如果實施失敗，可以快速回滾到當前版本：

```bash
# 回滾步驟
git stash  # 暫存未完成修改
git checkout <current_commit>  # 回到當前版本
git branch -D ai-scheduler-implementation  # 刪除實施分支
```

---

## 📅 里程碑時間表

| 日期 | 里程碑 | 交付物 | 狀態 |
|------|--------|--------|------|
| **Day 1-2** | 核心接口 Phase 1 | execute_mission(), _decompose_mission() | 🔲 待開始 |
| **Day 3** | 核心接口 Phase 2 | _intelligent_sort(), _calculate_task_priority() | 🔲 待開始 |
| **Day 4** | 執行控制 Phase 1 | _execute_with_concurrency() | 🔲 待開始 |
| **Day 5** | 執行控制 Phase 2 | _dynamic_adjust() | 🔲 待開始 |
| **Day 6-7** | 動作映射擴展 | 35 個動作映射 | 🔲 待開始 |
| **Day 8** | 整合測試 | E2E 測試通過 | 🔲 待開始 |
| **Day 9** | 文檔與優化 | 文檔更新，最終交付 | 🔲 待開始 |

---

## 🎯 成功標準

### 最終驗收標準

**必須滿足**（P0）：
1. ✅ execute_mission() 能接受簡單輸入（target + intent）
2. ✅ AI 能智能分解任務（5-10 個子任務）
3. ✅ AI 能智能排序任務（考慮 6 種因素）
4. ✅ 並發控制正常（最大 5 個並發）
5. ✅ 動態調整策略（至少 3 種調整場景）
6. ✅ 所有測試通過

**期望滿足**（P1）：
1. ✅ 動作映射擴展至 35 個
2. ✅ 性能提升 30%
3. ✅ 代碼覆蓋率 > 80%
4. ✅ 文檔完整

**錦上添花**（P2）：
1. ⚪ 支持更多意圖類型
2. ⚪ 添加圖形化界面
3. ⚪ 生成執行報告

---

## 📝 附錄

### 參考文件

1. [AI排序器實施指南.md](AI排序器實施指南.md) - 核心架構設計
2. [跨語言CLI設計指南.md](跨語言CLI設計指南.md) - JSON 合約標準
3. [SERVICES_AI_提升分析報告.md](../../guides/architecture/SERVICES_AI_提升分析報告.md) - 現狀分析

### 技術依賴

- Python 3.10+
- asyncio
- PyTorch 1.13+ (5M 神經網路)
- sentence-transformers (語意編碼)
- 現有 AIVA 組件

### 團隊聯絡

- **架構師**: AI Assistant
- **開發團隊**: AIVA Dev Team
- **測試團隊**: QA Team

---

**文檔版本**: v1.0  
**最後更新**: 2026年1月11日  
**下次更新**: 每日更新實施進度

