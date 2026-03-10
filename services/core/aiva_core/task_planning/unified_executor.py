"""統一攻擊執行器 - 靶場與實戰統一，持續學習

此模組實現統一執行流程，消除訓練模式和實戰模式的區分：
- 所有攻擊執行都使用相同邏輯
- 自動收集經驗並持續學習
- 累積足夠經驗後自動訓練模型
- 支持禁用學習模式（如需要）

架構理念：
> 靶場 = 實戰，每次執行都是學習機會

日期: 2025-12-17
版本: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

from aiva_common.utils import get_logger

logger = get_logger(__name__)

# ExecutionContext 是核心依賴，必須可用
from .executor.execution_status_monitor import ExecutionContext


# === 數據模型定義 ===

@dataclass
class AttackTarget:
    """攻擊目標"""
    url: str
    ip: Optional[str] = None
    domain: Optional[str] = None
    ports: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPlan:
    """攻擊計劃"""
    plan_id: str
    target: AttackTarget
    objective: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ExperienceSample:
    """經驗樣本 (State-Action-Reward)"""
    sample_id: str
    timestamp: datetime
    state: dict[str, Any]  # 環境狀態
    action: dict[str, Any]  # 執行的動作
    reward: float  # 獎勵分數
    next_state: dict[str, Any]  # 下一個狀態
    done: bool  # 是否完成
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0  # 品質評分
    confidence: float = 0.0  # 置信度


@dataclass
class ExecutionResult:
    """執行結果"""
    success: bool
    vulnerabilities: list[dict[str, Any]] = field(default_factory=list)
    attack_plan: Optional[AttackPlan] = None
    execution_details: dict[str, Any] = field(default_factory=dict)
    learning_info: Optional[dict[str, Any]] = None  # 學習狀態信息


@dataclass
class ModelTrainingConfig:
    """模型訓練配置"""
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2


@dataclass
class AttackFeedback:
    """攻擊反饋數據"""
    feedback_id: str
    attack_id: str
    timestamp: datetime
    success_rate: float  # 0.0 - 1.0
    vulnerabilities_found: int
    execution_time: float  # 秒
    strategies_used: list[str]
    effectiveness_score: float  # 綜合效能分數
    error_rate: float  # 錯誤率
    waf_triggered: bool  # 是否觸發 WAF
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyOptimization:
    """策略優化建議"""
    strategy_name: str
    current_score: float
    recommended_adjustments: dict[str, Any]
    priority: int  # 1-10，10 最高
    reason: str


# === 核心執行器 ===

class UnifiedAttackExecutor:
    """統一攻擊執行器

    設計理念：
    - 靶場和實戰使用完全相同的執行邏輯
    - 每次攻擊都自動收集經驗
    - 累積到閾值後自動訓練模型
    - 持續改進攻擊能力

    優勢：
    - 簡化架構：單一執行路徑
    - 數據利用：所有攻擊都成為訓練素材
    - 持續學習：自動優化模型
    - 強泛化：實戰數據提升魯棒性
    """

    def __init__(
        self,
        execution_mode: str = "cli",
        learning_enabled: bool = True,
        auto_train_threshold: int = 100,
        min_train_interval: int = 3600,
        data_directory: Optional[Path] = None
    ):
        """初始化統一執行器

        Args:
            execution_mode: 執行模式 "cli"（預設，新架構）或 "legacy"（舊架構，調試用）
            learning_enabled: 是否啟用學習（默認 True）
            auto_train_threshold: 自動訓練閾值（累積 N 個經驗後訓練）
            min_train_interval: 最小訓練間隔（秒）
            data_directory: 數據目錄

        架構升級（v2.0 - 2026-02-09）:
        - 支持 CLI 參數包驅動架構（execution_mode="cli"，預設）
        - 保留舊執行路徑供調試對比（execution_mode="legacy"）
        - 確認穩定後將刪除 legacy 模式
        """
        logger.info("🎯 Initializing Unified Attack Executor...")

        # 執行模式配置
        self.execution_mode = execution_mode
        if execution_mode == "cli":
            logger.info("🚀 使用 CLI 參數包驅動架構（新）")
        elif execution_mode == "legacy":
            logger.warning("⚠️ 使用舊執行路徑（僅供調試對比）")
        else:
            raise ValueError(f"無效的執行模式: {execution_mode}。應為 'cli' 或 'legacy'")

        self.learning_enabled = learning_enabled
        self.auto_train_threshold = auto_train_threshold
        self.min_train_interval = min_train_interval
        self.last_train_time = time.time()

        # 數據目錄
        self.data_directory = data_directory or Path("./data/unified_executor")
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # === 延遲初始化組件（避免循環導入）===
        self._rag_engine = None
        self._experience_manager = None
        self._model_trainer = None
        self._continuous_learning_engine = None  # 統一學習引擎
        self._message_broker = None  # 用於異步訓練任務
        self._feedback_optimizer = None  # 反饋優化引擎

        # 反饋歷史記錄
        self._feedback_history: list[AttackFeedback] = []
        self._strategy_performance: dict[str, list[float]] = {}  # 策略名稱 -> 效能分數列表

        logger.info(f"✅ Unified Executor initialized (learning={'ON' if learning_enabled else 'OFF'})")

    @property
    def rag_engine(self):
        """延遲加載 RAG Engine"""
        if self._rag_engine is None:
            from ..cognitive_core.rag import RAGEngine, KnowledgeBase, VectorStore
            vector_store = VectorStore(
                backend="memory",
                persist_directory=self.data_directory / "vectors"
            )
            knowledge_base = KnowledgeBase(vector_store=vector_store)
            self._rag_engine = RAGEngine(knowledge_base=knowledge_base)
        return self._rag_engine

    @property
    def experience_manager(self):
        """延遲加載 Experience Manager"""
        if self._experience_manager is None:
            # 模組整合: external_learning → cognitive_core/learning_system
            from ..cognitive_core.learning_system.experience_manager import ExperienceManager
            self._experience_manager = ExperienceManager(capacity=10000)
        return self._experience_manager

    @property
    def model_trainer(self):
        """延遲加載 Model Trainer"""
        if self._model_trainer is None:
            # 模組整合: external_learning → cognitive_core/learning_system
            from ..cognitive_core.learning_system.learning.model_trainer import ModelTrainer
            self._model_trainer = ModelTrainer()
        return self._model_trainer

    @property
    def continuous_learning_engine(self):
        """延遲加載 Continuous Learning Engine"""
        if self._continuous_learning_engine is None:
            from ..cognitive_core.learning_system.learning.continuous_learning import ContinuousLearningEngine
            self._continuous_learning_engine = ContinuousLearningEngine(
                experience_manager=self.experience_manager,
                model_trainer=self.model_trainer
            )
        return self._continuous_learning_engine

    @property
    def message_broker(self):
        """延遲載入 Message Broker

        Raises:
            ImportError: 如果 Message Broker 不可用
        """
        if self._message_broker is None:
            try:
                from ..service_backbone.messaging import get_broker
                self._message_broker = get_broker()
                logger.info("✅ Message Broker 初始化成功")
            except ImportError as e:
                raise ImportError(
                    "❌ 缺少必要依賴 MessageBroker\n"
                    "異步訓練功能依賴 Message Broker 進行任務佇列\n"
                    "請確認 services.service_backbone.messaging 模組已實現\n"
                    f"原始錯誤: {e}"
                ) from e
        return self._message_broker

    @property
    def feedback_optimizer(self):
        """延遲載入反饋優化引擎"""
        if self._feedback_optimizer is None:
            self._feedback_optimizer = FeedbackOptimizer(
                feedback_history=self._feedback_history,
                strategy_performance=self._strategy_performance
            )
        return self._feedback_optimizer
    async def execute(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None,
        cli_command: Optional['CLICommand'] = None
    ) -> ExecutionResult:
        """統一執行接口 - 靶場和實戰都調用這個

        架構升級（v2.0）:
        - 支持 CLI 參數包驅動（cli_command 參數）
        - 根據 execution_mode 選擇執行路徑
        - 預設使用新架構（CLI 模式）

        Args:
            target: 攻擊目標 (localhost:3000 或 example.com)
            objective: 攻擊目標 ("檢查 XSS 漏洞")
            scenario: 可選的場景配置（靶場模式會提供）
            constraints: 約束條件（速率限制、隱匿模式等）
            cli_command: CLI 命令（新架構，可選）

        Returns:
            ExecutionResult 包含攻擊結果和學習狀態
        """
        logger.info(f"🚀 Executing unified attack: {target} - {objective} [mode={self.execution_mode}]")

        # 根據執行模式選擇路徑
        if self.execution_mode == "cli":
            return await self._execute_via_cli(
                target=target,
                objective=objective,
                scenario=scenario,
                constraints=constraints,
                cli_command=cli_command
            )
        else:  # legacy
            return await self._execute_legacy(
                target=target,
                objective=objective,
                scenario=scenario,
                constraints=constraints
            )

    async def _execute_via_cli(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None,
        cli_command: Optional['CLICommand'] = None
    ) -> ExecutionResult:
        """CLI 參數包驅動執行（新架構）

        流程:
        1. 如果沒有提供 cli_command，使用 tool_selector 生成
        2. 執行 CLI 命令（subprocess）
        3. 解析結果
        4. 學習與反饋

        Args:
            同 execute()

        Returns:
            ExecutionResult
        """
        import subprocess
        import json
        from ..cognitive_core.schemas import CLICommand

        logger.info("🚀 CLI 模式執行開始")
        start_time = time.time()

        # 1. 生成或使用 CLI 命令
        if cli_command is None:
            from .planner.tool_selector import ToolSelector
            selector = ToolSelector()
            cli_command = selector.select_cli_command(
                intent=objective,
                target=target,
                context={
                    "intensity": constraints.get("intensity", 0.5) if constraints else 0.5,
                    "mode": constraints.get("mode", "normal") if constraints else "normal"
                }
            )

            if cli_command is None:
                logger.error("無法生成 CLI 命令")
                return ExecutionResult(
                    success=False,
                    vulnerabilities=[],
                    execution_details={"error": "無法生成 CLI 命令"},
                    learning_info={"mode": "cli", "error": "command_generation_failed"}
                )

        # 2. 執行 CLI 命令
        logger.info(f"📟 執行命令: {cli_command.to_shell_command()}")

        try:
            # 使用 asyncio subprocess 以配合 async 函式
            proc = await asyncio.create_subprocess_exec(
                *cli_command.to_cli_args(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent.parent.parent  # AIVA-git root
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=cli_command.timeout
                )
                stdout_text = stdout.decode() if stdout else ""
                stderr_text = stderr.decode() if stderr else ""
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise subprocess.TimeoutExpired(cli_command.to_cli_args(), cli_command.timeout)

            execution_time = time.time() - start_time

            # 3. 解析結果
            success = proc.returncode == 0
            vulnerabilities = self._parse_cli_output(stdout_text, stderr_text)

            logger.info(
                f"✅ CLI 執行完成: success={success}, "
                f"vulns={len(vulnerabilities)}, time={execution_time:.2f}s"
            )

            # 4. 學習與反饋
            learning_info = None
            if self.learning_enabled:
                try:
                    learning_info = await self._learn_from_cli_execution(
                        cli_command=cli_command,
                        success=success,
                        vulnerabilities=vulnerabilities,
                        execution_time=execution_time,
                        scenario=scenario
                    )
                except Exception as e:
                    logger.error(f"CLI 執行學習失敗: {e}")
                    learning_info = {"error": str(e)}

            return ExecutionResult(
                success=success,
                vulnerabilities=vulnerabilities,
                execution_details={
                    "cli_command": cli_command.to_shell_command(),
                    "returncode": result.returncode,
                    "stdout": result.stdout[:500],  # 截斷避免過長
                    "stderr": result.stderr[:500],
                    "execution_time": execution_time
                },
                learning_info=learning_info
            )

        except subprocess.TimeoutExpired:
            logger.error(f"CLI 命令超時: {cli_command.timeout}s")
            return ExecutionResult(
                success=False,
                vulnerabilities=[],
                execution_details={
                    "error": "timeout",
                    "timeout": cli_command.timeout
                },
                learning_info={"mode": "cli", "error": "timeout"}
            )
        except Exception as e:
            logger.error(f"CLI 執行異常: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                vulnerabilities=[],
                execution_details={"error": str(e)},
                learning_info={"mode": "cli", "error": "execution_failed"}
            )

    def _parse_cli_output(self, stdout: str, _stderr: str) -> list[dict]:  # stderr reserved for future debugging
        """解析 CLI 輸出提取漏洞信息

        Args:
            stdout: 標準輸出
            _stderr: 標準錯誤 (reserved for future debugging)

        Returns:
            漏洞列表
        """
        vulnerabilities = []

        # 嘗試解析 JSON 輸出
        try:
            if stdout.strip():
                data = json.loads(stdout)
                if isinstance(data, dict):
                    vulnerabilities = data.get("vulnerabilities", [])
                elif isinstance(data, list):
                    vulnerabilities = data
        except json.JSONDecodeError:
            # 降級：從文本中提取
            for line in stdout.split('\n'):
                line_lower = line.lower()
                if any(kw in line_lower for kw in ['vuln', 'issue', 'found', '發現']):
                    vulnerabilities.append({
                        "type": "unknown",
                        "description": line.strip(),
                        "severity": "medium"
                    })

        return vulnerabilities

    async def _learn_from_cli_execution(
        self,
        cli_command: 'CLICommand',
        success: bool,
        vulnerabilities: list,
        execution_time: float,
        scenario: Optional[dict] = None
    ) -> dict:
        """從 CLI 執行中學習

        Args:
            cli_command: 執行的 CLI 命令
            success: 是否成功
            vulnerabilities: 發現的漏洞
            execution_time: 執行時間
            scenario: 場景信息

        Returns:
            學習信息字典
        """
        # 構建經驗樣本
        sample = ExperienceSample(
            sample_id=f"cli_{int(time.time())}",
            timestamp=datetime.now(UTC),
            state={
                "flow_id": cli_command.flow_id,
                "target": cli_command.target,
                "flags": cli_command.flags,
                "metadata": cli_command.metadata
            },
            action={
                "type": "cli_execution",
                "command": cli_command.to_shell_command()
            },
            reward=1.0 if success and vulnerabilities else 0.0,
            next_state={"vulnerabilities_found": len(vulnerabilities)},
            done=True,
            metadata={
                "execution_time": execution_time,
                "is_training_scenario": scenario is not None,
                "flow_metadata": cli_command.metadata
            }
        )

        # 添加到經驗池
        await self.experience_manager.add(sample)

        # 檢查是否需要自動訓練
        buffer_size = len(self.experience_manager.experience_buffer)
        if buffer_size >= self.auto_train_threshold:
            logger.info(f"📚 經驗池達到閾值 ({buffer_size}），觸發訓練")
            await self._auto_train()

        return {
            "mode": "cli",
            "sample_id": sample.sample_id,
            "buffer_size": buffer_size,
            "auto_train_triggered": buffer_size >= self.auto_train_threshold
        }

    async def _execute_legacy(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None
    ) -> ExecutionResult:
        """舊執行路徑（保留供調試對比）

        ⚠️ 此路徑將在確認 CLI 模式穩定後刪除

        使用 CapabilityOrchestrator 進行規劃和執行。
        """
        logger.warning("⚠️ 使用 legacy 執行模式（調試用）")
        logger.info(f"🚀 Executing unified attack: {target} - {objective}")

        # 記錄開始時間
        start_time = time.time()

        # 1️⃣ 使用 CapabilityOrchestrator 生成執行計劃
        from ..cognitive_core.capability_orchestrator import CapabilityOrchestrator, TaskRequirement

        # 確保 RAG engine 已初始化，然後傳遞 knowledge base
        rag_kb = None
        try:
            rag_kb = self.rag_engine.knowledge_base
            logger.info("✅ 使用現有 RAG Knowledge Base")
        except Exception as e:
            logger.warning(f"⚠️ RAG engine 初始化問題: {e}")

        orchestrator = CapabilityOrchestrator(rag_kb=rag_kb)

        # 構建任務需求
        requirement = TaskRequirement(
            task_id=f"unified_{int(time.time())}",
            task_type="attack",  # 或根據 objective 推導
            target=target,
            objectives=[objective],
            constraints=constraints or {},
            priority=8 if scenario else 5  # 靶場模式優先級較低
        )

        # 生成計劃
        capability_plan = await orchestrator.plan(requirement)
        logger.info(f"📋 Generated capability plan: {capability_plan.plan_id} ({len(capability_plan.cli_commands)} commands)")

        # 記錄使用的策略
        strategies_used = []
        for cmd in capability_plan.cli_commands:
            if isinstance(cmd, dict):
                strategies_used.append(cmd.get("strategy", "default"))
            else:
                strategies_used.append("default")

        # 2️⃣ 執行計劃
        execution_result = await orchestrator.execute(capability_plan)

        # 計算執行時間
        execution_time = time.time() - start_time

        logger.info(f"⚡ Execution completed: success={execution_result.success} in {execution_time:.2f}s")

        # 3️⃣ 智能學習層（委派給 ContinuousLearningEngine）
        learning_info = None
        if self.learning_enabled:
            try:
                # 將 CapabilityPlan 轉換為學習系統需要的格式
                learning_context = {
                    "plan_id": capability_plan.plan_id,
                    "commands": capability_plan.cli_commands,
                    "execution_result": {
                        "success": execution_result.success,
                        "issues_found": execution_result.issues_found,
                        "total_duration": execution_result.total_duration
                    },
                    "scenario": scenario
                }

                learning_info = await self.continuous_learning_engine.learn_from_execution(
                    attack_plan=None,  # 不再使用 AttackPlan
                    execution_result=learning_context,
                    scenario=scenario
                )
            except Exception as e:
                logger.error(f"Learning failed: {e}")
                learning_info = {"error": str(e)}

        # 4️⃣ 返回統一結果
        result = ExecutionResult(
            success=execution_result.success,
            vulnerabilities=execution_result.issues_found,  # 將 issues 映射為 vulnerabilities
            attack_plan=None,  # 不再使用 AttackPlan
            execution_details={
                "capability_plan": capability_plan,
                "execution_result": execution_result
            },
            learning_info=learning_info
        )

        # 5️⃣ 收集反饋並優化（新增）
        if self.learning_enabled:
            try:
                await self.collect_feedback(
                    attack_id=capability_plan.plan_id,
                    execution_result=result,
                    execution_time=execution_time,
                    strategies_used=strategies_used
                )
            except Exception as e:
                logger.error(f"Feedback collection failed: {e}")

        return result

    async def execute_with_context(
        self,
        target: str,
        objective: str,
        execution_context: 'ExecutionContext',
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None
    ) -> ExecutionResult:
        """統一執行接口（v2.1 環境無關設計）

        設計理念：
        - 實際執行絕大部分是黑盒測試，不區分 sandbox/production
        - 策略選擇基於「目標敏感度」而非「執行環境」
        - 確保學習經驗可直接遷移

        Args:
            target: 攻擊目標
            objective: 攻擊目標
            execution_context: 執行上下文（提取 target_sensitivity）
            scenario: 可選的場景配置
            constraints: 約束條件

        Returns:
            ExecutionResult 包含攻擊結果和學習狀態
        """
        # v2.1: 從 context 提取目標敏感度
        target_sensitivity = getattr(execution_context, 'target_sensitivity', 0.5)
        risk_tolerance = getattr(execution_context, 'risk_tolerance', 0.5)

        logger.info(
            "🚀 Unified execution: %s - %s [sensitivity=%.1f, risk_tolerance=%.1f]",
            target, objective, target_sensitivity, risk_tolerance
        )

        # v2.1: 統一執行路徑，所有執行都走同一邏輯
        return await self._unified_execution(
            target, objective,
            target_sensitivity=target_sensitivity,
            risk_tolerance=risk_tolerance,
            scenario=scenario,
            constraints=constraints
        )

    async def _unified_execution(
        self,
        target: str,
        objective: str,
        target_sensitivity: float,
        risk_tolerance: float,  # 預留用於未來風險評估
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> ExecutionResult:
        """統一執行邏輯（v2.1 黑盒測試導向）

        根據目標敏感度自動調整策略，所有執行都學習：
        - 低敏感 (0.0-0.3): 激進策略，多策略並行
        - 中敏感 (0.3-0.7): 平衡策略
        - 高敏感 (0.7-1.0): 保守策略

        特點：
        - 無 sandbox/production 區分
        - 所有執行結果都可學習
        - 學習經驗直接可遷移

        Args:
            risk_tolerance: 預留參數，未來用於動態風險調整
        """
        # 使用 risk_tolerance 調整策略（預留邏輯）
        adjusted_sensitivity = min(1.0, target_sensitivity + (1.0 - risk_tolerance) * 0.1)

        # 根據敏感度決定探索程度
        if adjusted_sensitivity <= 0.3:
            # 低敏感：可以多策略探索
            num_strategies = 3
            logger.info("🔬 Low sensitivity target: exploring %d strategies", num_strategies)
        elif adjusted_sensitivity <= 0.7:
            # 中敏感：適度探索
            num_strategies = 2
            logger.info("⚖️ Medium sensitivity target: exploring %d strategies", num_strategies)
        else:
            # 高敏感：單一最佳策略
            num_strategies = 1
            logger.info("🛡️ High sensitivity target: using best strategy only")

        # 生成攻擊方案
        plans = []
        for strategy_variant in range(num_strategies):
            plan = await self._generate_enhanced_plan(
                target=target,
                objective=objective,
                scenario={
                    **(scenario or {}),
                    "strategy_variant": strategy_variant,
                    "target_sensitivity": target_sensitivity  # 傳遞敏感度，不是環境
                },
                constraints=constraints
            )
            plans.append(plan)

        logger.info(f"📋 Generated {len(plans)} attack plans")

        # 執行所有方案
        results = []
        for plan in plans:
            result = await self._execute_attack_plan(plan)
            results.append((plan, result))

        # 選擇最佳結果
        best_result = max(
            results,
            key=lambda x: len(x[1].get("vulnerabilities", []))
        )
        best_plan, execution_result = best_result

        logger.info(f"⚡ Best result: {len(execution_result.get('vulnerabilities', []))} vulns found")

        # 統一學習邏輯（所有執行都學習，不區分環境）
        learning_info = None
        if self.learning_enabled:
            all_samples = []
            for plan, result in results:
                samples = await self._learn_from_execution(
                    attack_plan=plan,
                    execution_result=result,
                    scenario={
                        **(scenario or {}),
                        "target_sensitivity": target_sensitivity  # 記錄敏感度而非環境
                    }
                )
                all_samples.append(samples)

            learning_info = {
                "samples_collected": sum(s.get("samples_collected", 0) for s in all_samples),
                "strategies_explored": len(plans),
                "target_sensitivity": target_sensitivity,
                "transferable": True  # 標記經驗可遷移
            }

        return ExecutionResult(
            success=execution_result.get("success", False),
            vulnerabilities=execution_result.get("vulnerabilities", []),
            attack_plan=best_plan,
            execution_details=execution_result,
            learning_info=learning_info
        )

    # === 向後兼容：保留舊方法但標記為已棄用 ===

    async def _sandbox_execution(
        self,
        target: str,
        objective: str,
        execution_context: 'ExecutionContext',
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> ExecutionResult:
        """[已棄用] 靶場環境執行

        警告：此方法基於 sandbox/production 區分，已不推薦使用
        請使用 execute_with_context() 的統一執行路徑
        """
        import warnings
        warnings.warn(
            "_sandbox_execution() is deprecated. "
            "Use execute_with_context() with target_sensitivity instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # 映射到統一執行：sandbox → 低敏感度
        return await self._unified_execution(
            target, objective,
            target_sensitivity=0.2,  # sandbox 映射為低敏感
            risk_tolerance=getattr(execution_context, 'risk_tolerance', 0.8),
            scenario=scenario,
            constraints=constraints
        )

    async def _production_execution(
        self,
        target: str,
        objective: str,
        execution_context: 'ExecutionContext',
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> ExecutionResult:
        """[已棄用] 生產環境執行

        警告：此方法基於 sandbox/production 區分，已不推薦使用
        請使用 execute_with_context() 的統一執行路徑
        """
        import warnings
        warnings.warn(
            "_production_execution() is deprecated. "
            "Use execute_with_context() with target_sensitivity instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # 映射到統一執行：production → 高敏感度
        return await self._unified_execution(
            target, objective,
            target_sensitivity=0.8,  # production 映射為高敏感
            risk_tolerance=getattr(execution_context, 'risk_tolerance', 0.3),
            scenario=scenario,
            constraints=constraints
        )

    # _should_learn_from_production 已移除
    # v2.1: 所有執行結果都學習，不區分環境
    # 學習經驗基於 target_sensitivity 分類，確保可遷移

    async def _generate_enhanced_plan(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> AttackPlan:
        """生成 RAG 增強的攻擊計劃

        現在支持策略變體:
        - variant 0: 保守策略 (低噪音、高隱蔽)
        - variant 1: 平衡策略 (中等覆蓋)
        - variant 2: 激進策略 (全面測試)
        """

        # 獲取策略配置
        from .strategy_profiles import get_strategy_profile
        strategy_variant = scenario.get("strategy_variant", 0) if scenario else 0
        strategy_profile = get_strategy_profile(strategy_variant)

        logger.info(f"🎯 Using strategy: {strategy_profile.name} (intensity={strategy_profile.attack_intensity})")

        # 查詢 RAG：尋找相似的成功案例（必需）
        rag_context = await self.rag_engine.retrieve_similar_cases(
            target=target,
            objective=objective
        )
        logger.info(f"🔍 RAG found {len(rag_context.get('similar_cases', []))} similar cases")

        # 使用 AI Commander 生成計劃 (注入策略配置)
        plan_context = {
            "target": target,
            "objective": objective,
            "rag_context": rag_context,
            "scenario": scenario,
            "constraints": constraints or {},
            "strategy_profile": {
                "name": strategy_profile.name,
                "intensity": strategy_profile.attack_intensity,
                "techniques": strategy_profile.techniques,
                "payload_count": strategy_profile.payload_count,
                "concurrent_attacks": strategy_profile.concurrent_attacks,
                "timing_delay": strategy_profile.timing_delay,
                "evasion_techniques": strategy_profile.evasion_techniques,
                "risk_level": strategy_profile.risk_level,
                "metadata": strategy_profile.metadata
            }
        }

        # 調用 AI Commander 生成計劃
        try:
            from .commander import AITaskType
            plan_result = await self.ai_commander.execute_command(
                task_type=AITaskType.ATTACK_PLANNING,
                context=plan_context
            )

            if plan_result.get("success"):
                plan_data = plan_result.get("plan", {})
                return AttackPlan(
                    plan_id=plan_data.get("plan_id", f"plan_{strategy_profile.name}_{int(time.time())}"),
                    target=AttackTarget(url=target),
                    objective=objective,
                    steps=plan_data.get("phases", []),
                    metadata={
                        "scenario_id": scenario.get("id") if scenario else None,
                        "rag_enhanced": bool(rag_context),
                        "constraints": constraints,
                        "strategy_profile": strategy_profile.name,
                        "attack_intensity": strategy_profile.attack_intensity,
                        "techniques": strategy_profile.techniques
                    }
                )
        except Exception as e:
            logger.warning(f"AI Commander plan generation failed: {e}, using strategy-based fallback")

        # 降級：創建基於策略配置的計劃
        steps = []

        # 根據策略配置生成步驟
        if strategy_profile.attack_intensity < 0.5:
            # 保守策略：基礎偵查 + 少量測試
            steps = [
                {"name": "passive_recon", "description": "Passive reconnaissance"},
                {"name": "basic_scan", "description": f"Basic {objective} scan",
                 "techniques": strategy_profile.techniques[:2]},  # 只用前2種技術
            ]
        elif strategy_profile.attack_intensity < 0.8:
            # 平衡策略：完整流程
            steps = [
                {"name": "reconnaissance", "description": "Active reconnaissance"},
                {"name": "vulnerability_scan", "description": f"Scan for {objective}",
                 "techniques": strategy_profile.techniques},
                {"name": "exploitation", "description": "Test found vulnerabilities"}
            ]
        else:
            # 激進策略：深度測試
            steps = [
                {"name": "comprehensive_recon", "description": "Comprehensive reconnaissance"},
                {"name": "deep_scan", "description": f"Deep scan for {objective}",
                 "techniques": strategy_profile.techniques},
                {"name": "advanced_exploitation", "description": "Advanced exploitation attempts"},
                {"name": "evasion_testing", "description": "Test evasion techniques",
                 "evasion_techniques": strategy_profile.evasion_techniques}
            ]

        return AttackPlan(
            plan_id=f"{strategy_profile.name}_fallback_{int(time.time())}",
            target=AttackTarget(url=target),
            objective=objective,
            steps=steps,
            metadata={
                "fallback": True,
                "strategy_profile": strategy_profile.name,
                "attack_intensity": strategy_profile.attack_intensity,
                "payload_count": strategy_profile.payload_count,
                "concurrent_attacks": strategy_profile.concurrent_attacks,
                "timing_delay": strategy_profile.timing_delay
            }
        )

    async def _execute_attack_plan(self, attack_plan: AttackPlan) -> dict[str, Any]:
        """執行攻擊計劃"""
        try:
            # 調用 Plan Executor
            result = await self.plan_executor.execute_plan(attack_plan)
            return result
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "traces": []
            }

    async def _learn_from_execution(
        self,
        attack_plan: AttackPlan,
        execution_result: dict,
        scenario: Optional[dict]
    ) -> dict:
        """從執行結果中學習"""

        logger.info("📚 Starting learning from execution...")

        # 1. 提取經驗樣本
        samples = self._extract_experience_samples(
            attack_plan=attack_plan,
            execution_result=execution_result,
            scenario=scenario
        )

        logger.info(f"📊 Extracted {len(samples)} experience samples")

        # 2. 保存到 Experience Buffer
        for sample in samples:
            try:
                self.experience_manager.add_sample(sample)
            except Exception as e:
                logger.warning(f"Failed to add sample: {e}")

        # 3. 更新 RAG 知識庫
        try:
            await self._update_rag_knowledge(samples)
        except Exception as e:
            logger.warning(f"RAG update failed: {e}")

        # 4. 判斷是否需要自動訓練
        buffer_size = len(self.experience_manager.buffer) if hasattr(self.experience_manager, 'buffer') else 0
        should_train = (
            buffer_size >= self.auto_train_threshold
            and (time.time() - self.last_train_time) >= self.min_train_interval
        )

        training_triggered = False
        if should_train:
            logger.info(f"🎓 Auto-training threshold reached ({buffer_size} samples)")
            training_triggered = await self._auto_train()

        return {
            "samples_collected": len(samples),
            "total_experiences": buffer_size,
            "training_triggered": training_triggered,
            "rag_updated": True
        }

    def _extract_experience_samples(
        self,
        attack_plan: AttackPlan,
        execution_result: dict,
        scenario: Optional[dict]
    ) -> list[ExperienceSample]:
        """提取訓練樣本 - 靶場和實戰使用相同邏輯"""

        samples = []
        traces = execution_result.get("traces", [])

        if not traces:
            logger.warning("No traces found in execution result")
            return samples

        # 計算基礎獎勵
        base_reward = 1.0 if execution_result.get("success") else -0.3

        # 從每個步驟提取樣本
        for i, trace in enumerate(traces):
            # 構建 State-Action-Reward
            sample = ExperienceSample(
                sample_id=f"exp_{attack_plan.plan_id}_{i}",
                timestamp=datetime.now(UTC),
                state={
                    "target": attack_plan.target.url,
                    "step_index": i,
                    "context": trace.get("context", {})
                },
                action={
                    "technique": trace.get("technique"),
                    "parameters": trace.get("parameters", {})
                },
                reward=self._calculate_step_reward(trace, base_reward),
                next_state=traces[i+1].get("context", {}) if i < len(traces)-1 else {},
                done=(i == len(traces)-1),
                metadata={
                    "target": attack_plan.target.url,
                    "is_training_scenario": scenario is not None,
                    "scenario_id": scenario.get("id") if scenario else None
                }
            )

            # 計算品質評分
            sample.quality_score = self._calculate_quality_score(sample)
            sample.confidence = self._calculate_confidence(sample)

            samples.append(sample)

        return samples

    def _calculate_step_reward(self, trace: dict, base_reward: float) -> float:
        """計算步驟獎勵"""
        reward = base_reward

        # 成功找到漏洞
        if trace.get("vulnerabilities_found"):
            vuln_count = len(trace["vulnerabilities_found"])
            reward += vuln_count * 0.5

        # 繞過防護
        if trace.get("bypassed_protection"):
            reward += 0.3

        # 觸發 WAF（負獎勵）
        if trace.get("triggered_waf"):
            reward -= 0.5

        return reward

    def _calculate_quality_score(self, sample: ExperienceSample) -> float:
        """計算樣本品質評分"""
        score = 0.5  # 基礎分���

        # 高獎勵樣本
        if sample.reward > 0.5:
            score += 0.3

        # 完整的狀態信息
        if len(sample.state) > 3:
            score += 0.1

        # 有明確技術
        if sample.action.get("technique"):
            score += 0.1

        return min(score, 1.0)

    def _calculate_confidence(self, sample: ExperienceSample) -> float:
        """計算置信度"""
        confidence = 0.5

        # 來自訓練場景
        if sample.metadata.get("is_training_scenario"):
            confidence += 0.2

        # 高品質樣本
        if sample.quality_score > 0.7:
            confidence += 0.2

        return min(confidence, 1.0)

    async def _update_rag_knowledge(self, samples: list[ExperienceSample]):
        """更新 RAG 知識庫"""
        for sample in samples:
            # 將經驗添加到 RAG
            try:
                await self.rag_engine.index_new_experience(sample)
            except Exception as e:
                logger.warning(f"Failed to index experience in RAG: {e}")

    async def _auto_train(self) -> bool:
        """異步傳遞訓練任務到獨立訓練服務

        重構目標：
        - 解決上帝物件問題：拆分 Training 職責
        - 避免 Event Loop 阻塞：訓練在獨立服務中進行
        - 避免 OOM 崩潰：大型模型訓練不占用主進程內存

        架構改進：
        UnifiedAttackExecutor (生產者)
            ↓ (publish training task)
        MessageBroker
            ↓ (subscribe)
        Training Worker (消費者) - 獨立進程/容器
            ↓ (train model)
        Model Storage
        """
        try:
            logger.info("🎓 Preparing training task...")

            # 從 buffer 採樣
            batch = self.experience_manager.sample(batch_size=32)

            if not batch:
                logger.warning("No samples available for training")
                return False

            # 強制使用 MessageBroker（不可用時拋出異常）
            # Fail Fast 原則：不降級，有錯就報錯
            broker = self.message_broker  # 觸發 property，不可用時會拋出 ImportError

            # 🎯 異步傳遞訓練任務
            from aiva_common.enums import Topic, ModuleName
            from aiva_common.schemas import AivaMessage, MessageHeader
            from uuid import uuid4

            training_task = {
                "task_id": f"train_{uuid4().hex[:12]}",
                "samples": [s.metadata for s in batch],  # 序列化樣本
                "config": {
                    "epochs": 10,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

            message = AivaMessage(
                header=MessageHeader(
                    message_id=str(uuid4()),
                    source_module=ModuleName.CORE,
                ),
                topic=Topic.TASK_AI_TRAINING_START,
                payload=training_task,
            )

            await broker.publish_message(
                exchange_name="aiva.tasks",
                routing_key="tasks.ai.training.start",
                message=message,
            )

            # 更新訓練時間（不等待結果）
            self.last_train_time = time.time()

            logger.info(f"✅ Training task published: {training_task['task_id']}")
            return True

        except ImportError:
            # MessageBroker 不可用，直接失敗不降級
            logger.error("❌ MessageBroker 不可用，無法執行異步訓練")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to publish training task: {e}", exc_info=True)
            return False

    def get_learning_status(self) -> dict:
        """獲取學習狀態"""
        buffer_size = len(self.experience_manager.buffer) if hasattr(self.experience_manager, 'buffer') else 0

        return {
            "learning_enabled": self.learning_enabled,
            "total_experiences": buffer_size,
            "auto_train_threshold": self.auto_train_threshold,
            "progress": f"{buffer_size}/{self.auto_train_threshold}",
            "last_train_time": datetime.fromtimestamp(self.last_train_time).isoformat(),
            "time_since_last_train": int(time.time() - self.last_train_time)
        }

    async def collect_feedback(
        self,
        attack_id: str,
        execution_result: ExecutionResult,
        execution_time: float,
        strategies_used: list[str]
    ) -> AttackFeedback:
        """收集攻擊反饋並進行優化分析

        Args:
            attack_id: 攻擊 ID
            execution_result: 執行結果
            execution_time: 執行時間（秒）
            strategies_used: 使用的策略列表

        Returns:
            AttackFeedback 反饋數據
        """
        logger.info(f"📊 Collecting feedback for attack: {attack_id}")

        # 計算成功率
        success_rate = 1.0 if execution_result.success else 0.0

        # 計算效能分數
        effectiveness_score = self._calculate_effectiveness_score(
            success_rate=success_rate,
            vulns_found=len(execution_result.vulnerabilities),
            execution_time=execution_time
        )

        # 檢查是否觸發 WAF
        waf_triggered = any(
            detail.get("waf_triggered", False)
            for detail in execution_result.execution_details.get("traces", [])
        )

        # 計算錯誤率
        error_rate = self._calculate_error_rate(execution_result.execution_details)

        # 創建反饋對象
        feedback = AttackFeedback(
            feedback_id=f"feedback_{int(time.time())}",
            attack_id=attack_id,
            timestamp=datetime.now(UTC),
            success_rate=success_rate,
            vulnerabilities_found=len(execution_result.vulnerabilities),
            execution_time=execution_time,
            strategies_used=strategies_used,
            effectiveness_score=effectiveness_score,
            error_rate=error_rate,
            waf_triggered=waf_triggered,
            metadata={
                "execution_details": execution_result.execution_details,
                "learning_info": execution_result.learning_info
            }
        )

        # 保存反饋歷史
        self._feedback_history.append(feedback)

        # 更新策略性能記錄
        for strategy in strategies_used:
            if strategy not in self._strategy_performance:
                self._strategy_performance[strategy] = []
            self._strategy_performance[strategy].append(effectiveness_score)

        # 觸發優化分析
        if len(self._feedback_history) >= 10:  # 累積 10 次反饋後進行優化
            await self._optimize_strategies()

        logger.info(
            f"✅ Feedback collected: effectiveness={effectiveness_score:.2f}, "
            f"success={success_rate}, vulns={len(execution_result.vulnerabilities)}"
        )

        return feedback

    def _calculate_effectiveness_score(
        self,
        success_rate: float,
        vulns_found: int,
        execution_time: float
    ) -> float:
        """計算攻擊效能分數

        考慮因素：
        - 成功率（40%）
        - 發現的漏洞數量（40%）
        - 執行效率（20%）
        """
        # 成功率分數
        success_score = success_rate * 0.4

        # 漏洞數量分數（歸一化）
        vuln_score = min(vulns_found / 5.0, 1.0) * 0.4

        # 效率分數（越快越好，假設理想時間為 10 秒）
        efficiency_score = max(0, 1.0 - (execution_time / 60.0)) * 0.2

        return success_score + vuln_score + efficiency_score

    def _calculate_error_rate(self, execution_details: dict) -> float:
        """計算錯誤率"""
        traces = execution_details.get("traces", [])
        if not traces:
            return 0.0

        error_count = sum(
            1 for trace in traces
            if trace.get("error") or trace.get("failed")
        )

        return error_count / len(traces)

    async def _optimize_strategies(self):
        """基於反饋優化攻擊策略"""
        logger.info("🔧 Optimizing attack strategies based on feedback...")

        try:
            # 使用優化引擎分析
            optimizations = await self.feedback_optimizer.analyze_and_optimize()

            if optimizations:
                logger.info(f"📈 Generated {len(optimizations)} optimization recommendations")

                # 應用優化建議
                for opt in optimizations:
                    logger.info(
                        f"  • {opt.strategy_name}: {opt.reason} "
                        f"(priority={opt.priority}, score={opt.current_score:.2f})"
                    )
                    await self._apply_optimization(opt)
            else:
                logger.info("✓ No optimizations needed at this time")

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}", exc_info=True)

    async def _apply_optimization(self, optimization: StrategyOptimization):
        """應用優化建議"""
        try:
            # 更新 RAG 知識庫中的策略配置
            await self.rag_engine.update_strategy_config(
                strategy_name=optimization.strategy_name,
                adjustments=optimization.recommended_adjustments
            )

            logger.info(f"✅ Applied optimization for: {optimization.strategy_name}")

        except Exception as e:
            logger.warning(f"Failed to apply optimization: {e}")

    def get_optimization_report(self) -> dict:
        """獲取優化報告"""
        report = {
            "total_feedbacks": len(self._feedback_history),
            "strategies_tracked": len(self._strategy_performance),
            "strategy_performance": {},
            "recent_optimizations": []
        }

        # 計算每個策略的平均性能
        for strategy, scores in self._strategy_performance.items():
            if scores:
                report["strategy_performance"][strategy] = {
                    "avg_score": sum(scores) / len(scores),
                    "best_score": max(scores),
                    "worst_score": min(scores),
                    "total_uses": len(scores),
                    "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
                }

        # 最近的反饋
        if self._feedback_history:
            recent = self._feedback_history[-5:]
            report["recent_feedbacks"] = [
                {
                    "attack_id": f.attack_id,
                    "effectiveness": f.effectiveness_score,
                    "success": f.success_rate,
                    "vulns_found": f.vulnerabilities_found,
                    "time": f.timestamp.isoformat()
                }
                for f in recent
            ]

        return report


class FeedbackOptimizer:
    """反饋優化引擎

    負責分析攻擊反饋並生成策略優化建議
    """

    def __init__(
        self,
        feedback_history: list[AttackFeedback],
        strategy_performance: dict[str, list[float]]
    ):
        self.feedback_history = feedback_history
        self.strategy_performance = strategy_performance
        self.logger = get_logger(__name__)

    def analyze_and_optimize(self) -> list[StrategyOptimization]:
        """分析反饋並生成優化建議"""
        optimizations = []

        # 1. 識別表現不佳的策略
        poor_strategies = self._identify_poor_strategies()
        for strategy in poor_strategies:
            opt = self._generate_optimization(strategy, "underperforming")
            if opt:
                optimizations.append(opt)

        # 2. 識別高頻錯誤模式
        error_patterns = self._identify_error_patterns()
        for pattern in error_patterns:
            opt = self._generate_error_fix(pattern)
            if opt:
                optimizations.append(opt)

        # 3. 識別 WAF 繞過機會
        waf_opportunities = self._identify_waf_bypass_opportunities()
        for opp in waf_opportunities:
            opt = self._generate_waf_bypass_optimization(opp)
            if opt:
                optimizations.append(opt)

        # 4. 按優先級排序
        optimizations.sort(key=lambda x: x.priority, reverse=True)

        return optimizations

    def _identify_poor_strategies(self) -> list[str]:
        """識別表現不佳的策略（平均分數 < 0.5）"""
        poor = []
        for strategy, scores in self.strategy_performance.items():
            if scores and sum(scores) / len(scores) < 0.5:
                poor.append(strategy)
        return poor

    def _identify_error_patterns(self) -> list[dict]:
        """識別高頻錯誤模式"""
        patterns = []

        # 分析最近的反饋
        recent = self.feedback_history[-20:] if len(self.feedback_history) > 20 else self.feedback_history

        # 統計錯誤類型
        error_types = {}
        for feedback in recent:
            if feedback.error_rate > 0.3:  # 錯誤率超過 30%
                for strategy in feedback.strategies_used:
                    if strategy not in error_types:
                        error_types[strategy] = 0
                    error_types[strategy] += 1

        # 識別高頻錯誤
        for strategy, count in error_types.items():
            if count >= 3:  # 至少出現 3 次
                patterns.append({
                    "strategy": strategy,
                    "error_count": count,
                    "frequency": count / len(recent)
                })

        return patterns

    def _identify_waf_bypass_opportunities(self) -> list[dict]:
        """識別 WAF 繞過機會"""
        opportunities = []

        # 分析最近的 WAF 觸發情況
        recent = self.feedback_history[-10:]
        waf_triggers = [f for f in recent if f.waf_triggered]

        if len(waf_triggers) >= 3:  # WAF 觸發頻率較高
            # 分析哪些策略容易觸發 WAF
            strategy_waf_rate = {}
            for feedback in recent:
                for strategy in feedback.strategies_used:
                    if strategy not in strategy_waf_rate:
                        strategy_waf_rate[strategy] = {"total": 0, "waf": 0}
                    strategy_waf_rate[strategy]["total"] += 1
                    if feedback.waf_triggered:
                        strategy_waf_rate[strategy]["waf"] += 1

            # 識別需要改進的策略
            for strategy, stats in strategy_waf_rate.items():
                waf_rate = stats["waf"] / stats["total"]
                if waf_rate > 0.5:  # WAF 觸發率超過 50%
                    opportunities.append({
                        "strategy": strategy,
                        "waf_rate": waf_rate,
                        "total_attempts": stats["total"]
                    })

        return opportunities

    def _generate_optimization(
        self,
        strategy: str,
        reason_type: str
    ) -> Optional[StrategyOptimization]:
        """生成策略優化建議"""
        scores = self.strategy_performance.get(strategy, [])
        if not scores:
            return None

        avg_score = sum(scores) / len(scores)

        # 根據原因類型生成調整建議
        if reason_type == "underperforming":
            adjustments = {
                "increase_timeout": True,
                "add_retry_logic": True,
                "adjust_payload_encoding": True,
                "diversify_attack_vectors": True
            }
            reason = f"策略平均效能低於預期 ({avg_score:.2f})"
            priority = 8
        else:
            adjustments = {}
            reason = "需要優化"
            priority = 5

        return StrategyOptimization(
            strategy_name=strategy,
            current_score=avg_score,
            recommended_adjustments=adjustments,
            priority=priority,
            reason=reason
        )

    def _generate_error_fix(self, pattern: dict) -> Optional[StrategyOptimization]:
        """生成錯誤修復建議"""
        strategy = pattern["strategy"]

        adjustments = {
            "add_error_handling": True,
            "validate_inputs": True,
            "add_fallback_strategy": True,
            "reduce_concurrency": True
        }

        return StrategyOptimization(
            strategy_name=strategy,
            current_score=0.0,  # 錯誤情況分數為 0
            recommended_adjustments=adjustments,
            priority=9,  # 高優先級
            reason=f"高頻錯誤模式 (出現 {pattern['error_count']} 次)"
        )

    def _generate_waf_bypass_optimization(
        self,
        opportunity: dict
    ) -> Optional[StrategyOptimization]:
        """生成 WAF 繞過優化建議"""
        strategy = opportunity["strategy"]
        waf_rate = opportunity["waf_rate"]

        adjustments = {
            "use_obfuscation": True,
            "add_delay_between_requests": True,
            "rotate_user_agents": True,
            "use_encoding_variations": True,
            "split_payload": True
        }

        return StrategyOptimization(
            strategy_name=strategy,
            current_score=1.0 - waf_rate,
            recommended_adjustments=adjustments,
            priority=10,  # 最高優先級
            reason=f"WAF 觸發率過高 ({waf_rate*100:.1f}%)"
        )

