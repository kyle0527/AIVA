# AIVA Schema 統一與優化方案

> **生成時間**: 2025-10-14
> **版本**: v1.0
> **狀態**: 規劃階段

---

## 📋 目錄

1. [現狀分析](#現狀分析)
2. [四大模組架構](#四大模組架構)
3. [Schema 統一規範](#schema-統一規範)
4. [缺少的 Schemas](#缺少的-schemas)
5. [命名重構計畫](#命名重構計畫)
6. [功能串接與流程優化](#功能串接與流程優化)
7. [實施步驟](#實施步驟)

---

## 現狀分析

### 統計數據

當前 `services/aiva_common/schemas.py` 包含：

- **總計**: 99 個 Schemas
- **共享基礎**: 20 個
- **Core AI 模組**: 27 個
- **Scan 模組**: 7 個
- **Function 模組**: 8 個
- **Integration 模組**: 37 個

### 已有的優秀命名模式

✅ **符合規範的命名**:

- `ScanStartPayload` / `ScanCompletedPayload` (Scan 模組)
- `FunctionTaskPayload` (Function 模組)
- `AITrainingStartPayload` / `AITrainingProgressPayload` / `AITrainingCompletedPayload` (AI 模組)
- `RAGQueryPayload` / `RAGResponsePayload` (RAG 模組)
- `AIExperienceCreatedEvent` / `AITraceCompletedEvent` / `AIModelUpdatedEvent` (AI 事件)
- `AIModelDeployCommand` (AI 命令)

### 需要改進的部分

⚠️ **命名不一致或模糊**:

- `FindingPayload` - 缺少模組前綴，難以區分所屬模組
- `EnhancedFindingPayload` - 與 `FindingPayload` 功能重疊
- `AttackStep` / `AttackPlan` - 應明確歸屬 Core AI 或 Integration
- `AssetLifecyclePayload` - 應明確歸屬 Scan 或 Integration

---

## 四大模組架構

AIVA 系統採用四大模組架構：

### 1. 🧠 Core 模組 (AI 核心引擎)

**職責**:

- AI 決策與規劃
- 強化學習訓練
- 攻擊計畫生成與執行
- 經驗學習與 RAG
- 任務編排

**核心組件**:

- BioNeuronRAGAgent (AI 決策引擎)
- PlanExecutor (計畫執行器)
- PlanComparator (AST/Trace 對比)
- ExperienceManager (經驗管理)
- ModelTrainer (模型訓練)
- RAG System (知識庫檢索)
- StandardScenarioManager (靶場場景管理)

### 2. 🔍 Scan 模組 (掃描引擎)

**職責**:

- 資產發現
- 指紋識別
- 漏洞掃描
- 信息收集

**技術棧**:

- Python Scanner
- TypeScript Scanner
- Rust Info Gatherer

### 3. ⚙️ Function 模組 (功能檢測)

**職責**:

- 漏洞驗證與利用
- 專項安全測試
- Payload 執行

**子模組**:

- XSS / SQLi / SSRF / IDOR 檢測
- SAST / SCA 分析
- AuthN / CSPM 檢查
- PostEx 測試

**技術棧**:

- Python Functions
- Go Functions
- Rust Functions

### 4. 🔗 Integration 模組 (整合服務)

**職責**:

- 漏洞相關性分析
- 攻擊路徑生成
- 風險評估
- 報告生成
- 外部系統整合

**核心組件**:

- VulnerabilityCorrelationAnalyzer
- AttackPathGenerator
- RiskAssessmentEngine
- ReportGenerator

---

## Schema 統一規範

### 命名規則

#### 1. **Payload 命名**

所有消息負載使用 `<Module><Action>Payload` 格式：

```python
# ✅ 正確
ScanStartPayload          # Scan 模組啟動
ScanProgressPayload       # Scan 模組進度
ScanCompletedPayload      # Scan 模組完成
ScanFailedPayload         # Scan 模組失敗

FunctionTaskPayload             # Function 模組任務
FunctionTaskProgressPayload     # Function 模組進度
FunctionTaskCompletedPayload    # Function 模組完成
FunctionTaskFailedPayload       # Function 模組失敗

AITrainingStartPayload          # AI 訓練啟動
AITrainingStopPayload           # AI 訓練停止
AITrainingProgressPayload       # AI 訓練進度
AITrainingCompletedPayload      # AI 訓練完成
AITrainingFailedPayload         # AI 訓練失敗

IntegrationAnalysisPayload      # Integration 分析
IntegrationReportPayload        # Integration 報告

# ❌ 錯誤
FindingPayload            # 缺少模組前綴
TaskPayload               # 太模糊
```

#### 2. **Event 命名**

所有事件使用 `<Module><EventName>Event` 格式：

```python
# ✅ 正確
AIExperienceCreatedEvent       # AI 經驗創建事件
AITraceCompletedEvent          # AI 追蹤完成事件
AIModelUpdatedEvent            # AI 模型更新事件
ScanAssetDiscoveredEvent       # Scan 資產發現事件
FunctionVulnFoundEvent         # Function 漏洞發現事件
IntegrationReportGeneratedEvent # Integration 報告生成事件
```

#### 3. **Command 命名**

所有命令使用 `<Module><CommandName>Command` 格式：

```python
# ✅ 正確
AIModelDeployCommand           # AI 模型部署命令
ScanCancelCommand              # Scan 取消命令
FunctionTaskCancelCommand      # Function 任務取消命令
IntegrationReportGenerateCommand # Integration 報告生成命令
```

#### 4. **Request/Response 命名**

請求-響應對使用統一格式：

```python
# ✅ 正確
RAGQueryPayload / RAGResponsePayload
AIVerificationRequest / AIVerificationResult
```

---

## 缺少的 Schemas

### Core AI 模組

```python
# ✅ 已有
class AITrainingStartPayload(BaseModel): ...
class AITrainingProgressPayload(BaseModel): ...
class AITrainingCompletedPayload(BaseModel): ...
class AIExperienceCreatedEvent(BaseModel): ...
class AITraceCompletedEvent(BaseModel): ...
class AIModelUpdatedEvent(BaseModel): ...
class AIModelDeployCommand(BaseModel): ...
class RAGQueryPayload(BaseModel): ...
class RAGResponsePayload(BaseModel): ...

# ⚠️ 需要新增
class AITrainingStopPayload(BaseModel):
    """AI 訓練停止請求"""
    session_id: str
    reason: str = "user_requested"
    save_checkpoint: bool = True

class AITrainingFailedPayload(BaseModel):
    """AI 訓練失敗通知"""
    session_id: str
    error_type: str
    error_message: str
    traceback: str | None = None
    failed_at: datetime

class AIScenarioLoadedEvent(BaseModel):
    """標準場景載入事件"""
    scenario_id: str
    scenario_name: str
    target_system: str
    expected_steps: int
```

### Scan 模組

```python
# ✅ 已有
class ScanStartPayload(BaseModel): ...
class ScanCompletedPayload(BaseModel): ...
class AssetLifecyclePayload(BaseModel): ...  # 應重命名為 ScanAssetLifecyclePayload

# ⚠️ 需要新增
class ScanProgressPayload(BaseModel):
    """掃描進度通知"""
    scan_id: str
    progress_percentage: float  # 0.0 - 100.0
    current_target: HttpUrl | None
    assets_discovered: int
    vulnerabilities_found: int
    estimated_time_remaining_seconds: int | None

class ScanFailedPayload(BaseModel):
    """掃描失敗通知"""
    scan_id: str
    error_type: str
    error_message: str
    failed_target: HttpUrl | None
    partial_results_available: bool

class ScanAssetDiscoveredEvent(BaseModel):
    """資產發現事件"""
    scan_id: str
    asset: Asset
    discovery_method: str  # "crawler", "dns", "port_scan" etc.
```

### Function 模組

```python
# ✅ 已有
class FunctionTaskPayload(BaseModel): ...
class FeedbackEventPayload(BaseModel): ...

# ⚠️ 需要新增
class FunctionTaskProgressPayload(BaseModel):
    """功能測試進度通知"""
    task_id: str
    scan_id: str
    progress_percentage: float
    tests_completed: int
    tests_total: int
    vulnerabilities_found: int

class FunctionTaskCompletedPayload(BaseModel):
    """功能測試完成通知"""
    task_id: str
    scan_id: str
    status: str  # "success", "partial", "failed"
    vulnerabilities_found: int
    tests_executed: int
    duration_seconds: float
    results: list[dict[str, Any]]

class FunctionTaskFailedPayload(BaseModel):
    """功能測試失敗通知"""
    task_id: str
    scan_id: str
    error_type: str
    error_message: str
    tests_completed: int
    tests_failed: int

class FunctionVulnFoundEvent(BaseModel):
    """漏洞發現事件"""
    task_id: str
    scan_id: str
    vulnerability: Vulnerability
    confidence: Confidence
```

### Integration 模組

```python
# ✅ 已有
class FindingPayload(BaseModel): ...  # 應重命名為 IntegrationFindingPayload
class EnhancedFindingPayload(BaseModel): ...  # 應合併至 IntegrationFindingPayload

# ⚠️ 需要新增
class IntegrationAnalysisStartPayload(BaseModel):
    """整合分析啟動請求"""
    analysis_id: str
    scan_id: str
    analysis_types: list[str]  # ["correlation", "attack_path", "risk"]
    findings: list[FindingPayload]

class IntegrationAnalysisProgressPayload(BaseModel):
    """整合分析進度通知"""
    analysis_id: str
    scan_id: str
    progress_percentage: float
    current_analysis_type: str
    correlations_found: int
    attack_paths_generated: int

class IntegrationAnalysisCompletedPayload(BaseModel):
    """整合分析完成通知"""
    analysis_id: str
    scan_id: str
    correlations: list[VulnerabilityCorrelation]
    attack_paths: list[AttackPathPayload]
    risk_assessment: RiskAssessmentResult
    recommendations: list[str]

class IntegrationReportGenerateCommand(BaseModel):
    """報告生成命令"""
    report_id: str
    scan_id: str
    report_format: str  # "pdf", "html", "json", "sarif"
    include_sections: list[str]

class IntegrationReportGeneratedEvent(BaseModel):
    """報告生成完成事件"""
    report_id: str
    scan_id: str
    report_format: str
    file_path: str | None
    download_url: str | None
```

---

## 命名重構計畫

### 1. 重命名現有 Schemas

```python
# Before → After
FindingPayload → IntegrationFindingPayload
EnhancedFindingPayload → (合併至 IntegrationFindingPayload)
AssetLifecyclePayload → ScanAssetLifecyclePayload
AttackStep → CoreAttackStep  # 明確歸屬
AttackPlan → CoreAttackPlan  # 明確歸屬
```

### 2. 統一 Topic 命名

```python
# Topic 命名格式: {category}.{module}.{action}

# Core AI
tasks.ai.training.start
tasks.ai.training.stop
results.ai.training.progress
results.ai.training.completed
results.ai.training.failed
events.ai.experience.created
events.ai.trace.completed
events.ai.model.updated
commands.ai.model.deploy

# Scan
tasks.scan.start
results.scan.progress
results.scan.completed
results.scan.failed
events.scan.asset.discovered

# Function
tasks.function.start
tasks.function.xss
tasks.function.sqli
tasks.function.ssrf
results.function.progress
results.function.completed
results.function.failed
events.function.vuln.found

# Integration
tasks.integration.analysis.start
results.integration.analysis.progress
results.integration.analysis.completed
commands.integration.report.generate
events.integration.report.generated
```

---

## 功能串接與流程優化

### 1. AST 解析與任務產生

#### 新增 Planner 模組

```python
class PlannerService:
    """攻擊計畫規劃器

    將 AST 攻擊流程圖轉換為可執行的任務序列
    """

    async def parse_ast_to_tasks(
        self,
        attack_plan: CoreAttackPlan,
        context: dict[str, Any]
    ) -> list[FunctionTaskPayload]:
        """將 AST 攻擊計畫轉換為任務列表

        Args:
            attack_plan: 攻擊計畫 AST
            context: 執行上下文

        Returns:
            任務負載列表
        """
        tasks = []
        for step in attack_plan.steps:
            task = FunctionTaskPayload(
                task_id=f"{attack_plan.plan_id}_step_{step.step_id}",
                scan_id=context["scan_id"],
                module=step.tool_type,
                test_type=step.action,
                targets=[{
                    "url": step.target.get("url"),
                    "params": step.target.get("params", {})
                }],
                config={
                    "session_id": attack_plan.session_id,
                    "plan_id": attack_plan.plan_id,
                    "step_id": step.step_id,
                    "dependencies": step.dependencies
                },
                metadata={
                    "mitre_technique": step.mitre_technique_id,
                    "expected_outcome": step.expected_outcome
                }
            )
            tasks.append(task)
        return tasks
```

### 2. 任務執行與 Trace 記錄

#### TraceLogger 擴充

```python
class TraceLogger:
    """執行追蹤記錄器

    訂閱 RabbitMQ 結果隊列，記錄所有任務執行詳情
    """

    async def subscribe_to_results(self):
        """訂閱結果隊列"""
        await self.broker.subscribe(
            exchange_name="aiva.topic",
            queue_name="trace.logger.results",
            routing_keys=[
                "results.function.completed",
                "results.function.failed",
                "events.function.vuln.found"
            ],
            callback=self.handle_result_message
        )

    async def handle_result_message(
        self,
        message: AivaMessage
    ):
        """處理結果消息並記錄 Trace"""
        trace_record = TraceRecord(
            plan_id=message.payload.get("config", {}).get("plan_id"),
            step_id=message.payload.get("config", {}).get("step_id"),
            task_id=message.payload["task_id"],
            tool_name=message.payload["module"],
            action=message.payload["test_type"],
            input_data=message.payload.get("targets"),
            output_data=message.payload.get("results"),
            timestamp=message.header.timestamp,
            success=message.payload.get("status") == "success",
            error_message=message.payload.get("error_message"),
            duration_seconds=message.payload.get("duration_seconds")
        )
        await self.storage.save_trace_record(trace_record)
```

### 3. AST/Trace 對比分析

#### PlanComparator 整合

```python
class PlanComparator:
    """AST 預期計畫與實際 Trace 對比分析器"""

    async def compare_plan_and_trace(
        self,
        plan: CoreAttackPlan,
        traces: list[TraceRecord]
    ) -> PlanExecutionResult:
        """對比分析

        Returns:
            包含差異指標的執行結果
        """
        # 步驟匹配
        matched_steps = self._match_steps(plan.steps, traces)

        # 順序檢查
        sequence_accuracy = self._calculate_sequence_accuracy(
            plan.steps,
            matched_steps
        )

        # 結果比較
        success_rate = sum(
            1 for trace in traces if trace.success
        ) / len(traces) if traces else 0

        # 計算差異指標
        completion_rate = len(matched_steps) / len(plan.steps)
        extra_actions = len(traces) - len(matched_steps)

        # 獎勵分數
        reward_score = self._calculate_reward_score(
            completion_rate=completion_rate,
            success_rate=success_rate,
            sequence_accuracy=sequence_accuracy,
            extra_actions=extra_actions
        )

        return PlanExecutionResult(
            plan_id=plan.plan_id,
            expected_steps=len(plan.steps),
            executed_steps=len(traces),
            matched_steps=len(matched_steps),
            completion_rate=completion_rate,
            sequence_accuracy=sequence_accuracy,
            success_rate=success_rate,
            extra_actions=extra_actions,
            reward_score=reward_score,
            metrics=PlanExecutionMetrics(...)
        )
```

### 4. 經驗樣本提取與存儲

#### ExperienceManager 整合

```python
class ExperienceManager:
    """經驗樣本管理器"""

    async def create_experience_from_execution(
        self,
        plan: CoreAttackPlan,
        traces: list[TraceRecord],
        comparison_result: PlanExecutionResult,
        context: dict[str, Any]
    ) -> ExperienceSample:
        """從執行結果創建經驗樣本

        Args:
            plan: 攻擊計畫
            traces: 執行追蹤記錄
            comparison_result: 對比分析結果
            context: 場景上下文

        Returns:
            經驗樣本
        """
        sample = ExperienceSample(
            sample_id=f"exp_{plan.plan_id}_{datetime.now(UTC).timestamp()}",
            session_id=plan.session_id,
            plan=plan,
            trace=traces,
            context={
                "target_info": context.get("target_info"),
                "vulnerability_types": context.get("vuln_types"),
                "environment": context.get("environment"),
                "business_criticality": context.get("business_criticality")
            },
            result=comparison_result,
            quality_score=self._calculate_quality_score(comparison_result),
            is_successful=comparison_result.reward_score >= 0.7,
            created_at=datetime.now(UTC),
            annotations={}
        )

        # 保存至資料庫
        await self.storage.save_experience_sample(sample)

        # 發佈事件
        await self.broker.publish(
            topic=Topic.AI_EXPERIENCE_CREATED,
            payload=AIExperienceCreatedEvent(
                sample_id=sample.sample_id,
                session_id=sample.session_id,
                quality_score=sample.quality_score,
                is_successful=sample.is_successful
            )
        )

        return sample
```

### 5. 模型微調與強化學習

#### ModelTrainer 自動化流程

```python
class ModelTrainer:
    """模型訓練器"""

    async def auto_training_pipeline(
        self,
        schedule: str = "weekly"  # daily, weekly, monthly
    ):
        """自動化訓練流程"""
        # 1. 提取經驗樣本
        samples = await self.experience_manager.get_high_quality_samples(
            min_quality_score=0.6,
            limit=1000
        )

        # 2. 構建訓練資料集
        dataset = await self._build_training_dataset(samples)

        # 3. 執行訓練
        result = await self.train_reinforcement(
            dataset=dataset,
            config=ModelTrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001,
                reward_discount=0.95
            )
        )

        # 4. 評估模型
        evaluation = await self.evaluate_model(
            model_path=result.model_path,
            test_scenarios=await self.scenario_manager.get_standard_scenarios()
        )

        # 5. 部署模型 (如果評估通過)
        if evaluation.accuracy >= 0.85:
            await self.deploy_model(
                model_path=result.model_path,
                deployment_env="production"
            )

            # 發佈模型更新事件
            await self.broker.publish(
                topic=Topic.AI_MODEL_UPDATED,
                payload=AIModelUpdatedEvent(
                    model_id=result.model_id,
                    version=result.version,
                    accuracy=evaluation.accuracy,
                    deployed_at=datetime.now(UTC)
                )
            )
```

---

## 實施步驟

### Phase 1: Schema 補全與統一 (Week 1-2)

1. **新增缺少的 Schemas**
   - [ ] 添加 `AITrainingStopPayload`
   - [ ] 添加 `AITrainingFailedPayload`
   - [ ] 添加 `ScanProgressPayload`
   - [ ] 添加 `ScanFailedPayload`
   - [ ] 添加 `FunctionTaskProgressPayload`
   - [ ] 添加 `FunctionTaskCompletedPayload`
   - [ ] 添加 `FunctionTaskFailedPayload`
   - [ ] 添加 Integration 模組完整 Schemas

2. **重命名現有 Schemas**
   - [ ] `FindingPayload` → `IntegrationFindingPayload`
   - [ ] `AssetLifecyclePayload` → `ScanAssetLifecyclePayload`
   - [ ] `AttackStep` → `CoreAttackStep`
   - [ ] `AttackPlan` → `CoreAttackPlan`

3. **更新 Topic 枚舉**
   - [ ] 添加所有新的 Topics
   - [ ] 確保命名一致性

### Phase 2: Planner 與 TraceLogger 實現 (Week 3-4)

1. **創建 PlannerService**
   - [ ] 實現 AST 解析邏輯
   - [ ] 實現任務生成邏輯
   - [ ] 整合 RabbitMQ 發佈

2. **擴充 TraceLogger**
   - [ ] 訂閱所有結果隊列
   - [ ] 實現完整 Trace 記錄
   - [ ] 整合 Storage Backend

### Phase 3: 對比分析與經驗管理 (Week 5-6)

1. **完善 PlanComparator**
   - [ ] 實現步驟匹配算法
   - [ ] 實現順序檢查
   - [ ] 實現獎勵計算

2. **完善 ExperienceManager**
   - [ ] 實現經驗樣本創建
   - [ ] 實現質量評分
   - [ ] 實現自動標註

### Phase 4: 訓練自動化 (Week 7-8)

1. **ModelTrainer 自動化**
   - [ ] 實現自動訓練流程
   - [ ] 實現標準場景集測試
   - [ ] 實現模型評估與部署

2. **TrainingOrchestrator**
   - [ ] 實現完整訓練編排
   - [ ] 整合所有組件
   - [ ] 實現 RabbitMQ 協調

### Phase 5: 測試與驗證 (Week 9-10)

1. **單元測試**
   - [ ] 所有新 Schemas 測試
   - [ ] Planner 測試
   - [ ] TraceLogger 測試
   - [ ] PlanComparator 測試

2. **整合測試**
   - [ ] 端到端流程測試
   - [ ] 標準場景測試
   - [ ] 性能測試

3. **文檔更新**
   - [ ] 更新 API 文檔
   - [ ] 更新架構圖
   - [ ] 更新開發者指南

---

## 成功指標

- ✅ 所有四大模組 Schemas 完整且命名統一
- ✅ AST → Tasks → Trace 流程完全自動化
- ✅ 經驗樣本自動提取並持久化
- ✅ 模型自動訓練與評估流程運行
- ✅ 所有測試通過，覆蓋率 ≥ 80%
- ✅ 文檔完整且最新

---

## 參考資料

- [AIVA 完整架構圖集](./COMPLETE_ARCHITECTURE_DIAGRAMS.md)
- [AI 系統總覽](./AI_SYSTEM_OVERVIEW.md)
- [模組通訊合約](./MODULE_COMMUNICATION_CONTRACTS.md)
- [數據存儲指南](./DATA_STORAGE_GUIDE.md)
