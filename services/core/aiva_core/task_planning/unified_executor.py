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

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

# v2.0: 統一反饋架構支援
try:
    from .executor.execution_status_monitor import ExecutionContext, EnvironmentType
    _EXECUTION_CONTEXT_AVAILABLE = True
except ImportError:
    _EXECUTION_CONTEXT_AVAILABLE = False
    logger.warning("ExecutionContext not available, context-aware execution disabled")


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
        learning_enabled: bool = True,
        auto_train_threshold: int = 100,
        min_train_interval: int = 3600,
        data_directory: Optional[Path] = None
    ):
        """初始化統一執行器
        
        Args:
            learning_enabled: 是否啟用學習（默認 True）
            auto_train_threshold: 自動訓練閾值（累積 N 個經驗後訓練）
            min_train_interval: 最小訓練間隔（秒）
            data_directory: 數據目錄
        """
        logger.info("🎯 Initializing Unified Attack Executor...")
        
        self.learning_enabled = learning_enabled
        self.auto_train_threshold = auto_train_threshold
        self.min_train_interval = min_train_interval
        self.last_train_time = time.time()
        
        # 數據目錄
        self.data_directory = data_directory or Path("./data/unified_executor")
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # === 延遲初始化組件（避免循環導入）===
        self._ai_commander = None
        self._rag_engine = None
        self._experience_manager = None
        self._model_trainer = None
        self._plan_executor = None
        
        logger.info(f"✅ Unified Executor initialized (learning={'ON' if learning_enabled else 'OFF'})")
    
    @property
    def ai_commander(self):
        """延遲加載 AI Commander"""
        if self._ai_commander is None:
            from .ai_commander import AICommander
            self._ai_commander = AICommander(data_directory=self.data_directory / "ai_commander")
        return self._ai_commander
    
    @property
    def rag_engine(self):
        """延遲加載 RAG Engine"""
        if self._rag_engine is None:
            from ..cognitive_core.rag import RAGEngine, KnowledgeBase, VectorStore
            vector_store = VectorStore(
                backend="memory",
                persist_directory=self.data_directory / "vectors"
            )
            knowledge_base = KnowledgeBase(
                vector_store=vector_store,
                data_directory=self.data_directory / "knowledge"
            )
            self._rag_engine = RAGEngine(knowledge_base=knowledge_base)
        return self._rag_engine
    
    @property
    def experience_manager(self):
        """延遲加載 Experience Manager"""
        if self._experience_manager is None:
            from ..external_learning.experience_manager import ExperienceManager
            self._experience_manager = ExperienceManager(capacity=10000)
        return self._experience_manager
    
    @property
    def model_trainer(self):
        """延遲加載 Model Trainer"""
        if self._model_trainer is None:
            from ..external_learning.learning.model_trainer import ModelTrainer
            self._model_trainer = ModelTrainer()
        return self._model_trainer
    
    @property
    def plan_executor(self):
        """延遲加載 Plan Executor"""
        if self._plan_executor is None:
            from .executor.plan_executor import PlanExecutor
            self._plan_executor = PlanExecutor()
        return self._plan_executor
    
    async def execute(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None
    ) -> ExecutionResult:
        """統一執行接口 - 靶場和實戰都調用這個
        
        Args:
            target: 攻擊目標 (localhost:3000 或 example.com)
            objective: 攻擊目標 ("檢查 XSS 漏洞")
            scenario: 可選的場景配置（靶場模式會提供）
            constraints: 約束條件（速率限制、隱匿模式等）
        
        Returns:
            ExecutionResult 包含攻擊結果和學習狀態
        """
        logger.info(f"🚀 Executing unified attack: {target} - {objective}")
        
        # 1️⃣ 生成攻擊計劃（使用 RAG 增強）
        attack_plan = await self._generate_enhanced_plan(
            target=target,
            objective=objective,
            scenario=scenario,
            constraints=constraints
        )
        
        logger.info(f"📋 Generated attack plan: {attack_plan.plan_id} ({len(attack_plan.steps)} steps)")
        
        # 2️⃣ 執行攻擊（靶場和實戰完全相同）
        execution_result = await self._execute_attack_plan(attack_plan)
        
        logger.info(f"⚡ Execution completed: success={execution_result.get('success', False)}")
        
        # 3️⃣ 智能學習層（自動判斷是否學習）
        learning_info = None
        if self.learning_enabled:
            learning_info = await self._learn_from_execution(
                attack_plan=attack_plan,
                execution_result=execution_result,
                scenario=scenario
            )
        
        # 4️⃣ 返回統一結果
        return ExecutionResult(
            success=execution_result.get("success", False),
            vulnerabilities=execution_result.get("vulnerabilities", []),
            attack_plan=attack_plan,
            execution_details=execution_result,
            learning_info=learning_info
        )
    
    async def execute_with_context(
        self,
        target: str,
        objective: str,
        execution_context: 'ExecutionContext',
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None
    ) -> ExecutionResult:
        """環境感知執行接口（v2.0 統一反饋架構）
        
        根據執行環境（sandbox/production）自動調整執行策略：
        - Sandbox: 探索式執行（嘗試多種策略，即時學習）
        - Production: 保守式執行（使用最佳已知策略，選擇性學習）
        
        Args:
            target: 攻擊目標
            objective: 攻擊目標
            execution_context: 執行上下文（包含環境類型）
            scenario: 可選的場景配置
            constraints: 約束條件
        
        Returns:
            ExecutionResult 包含攻擊結果和學習狀態
        """
        if not _EXECUTION_CONTEXT_AVAILABLE:
            logger.warning("ExecutionContext not available, falling back to standard execute")
            return await self.execute(target, objective, scenario, constraints)
        
        env = execution_context.environment
        logger.info(
            f"🚀 Context-aware execution: {target} - {objective} "
            f"[env={env.value if env else 'unknown'}]"
        )
        
        # 根據環境選擇執行策略
        if env == EnvironmentType.SANDBOX:
            return await self._sandbox_execution(
                target, objective, execution_context, scenario, constraints
            )
        elif env == EnvironmentType.PRODUCTION:
            return await self._production_execution(
                target, objective, execution_context, scenario, constraints
            )
        else:
            # 未指定環境：使用標準執行
            logger.warning("Environment type not specified, using standard execution")
            return await self.execute(target, objective, scenario, constraints)
    
    async def _sandbox_execution(
        self,
        target: str,
        objective: str,
        execution_context: 'ExecutionContext',
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> ExecutionResult:
        """靶場環境執行：探索式策略
        
        特點：
        - 嘗試多種攻擊策略（探索）
        - 即時學習所有結果
        - 高風險容忍度
        """
        logger.info(
            f"🔬 Sandbox execution: exploratory mode "
            f"(risk_tolerance={execution_context.risk_tolerance})"
        )
        
        # 1. 生成多個攻擊方案（探索不同策略）
        plans = []
        for strategy_variant in range(3):  # 生成 3 種策略變體
            plan = await self._generate_enhanced_plan(
                target=target,
                objective=objective,
                scenario={**(scenario or {}), "strategy_variant": strategy_variant},
                constraints=constraints
            )
            plans.append(plan)
        
        logger.info(f"📋 Generated {len(plans)} strategy variants for exploration")
        
        # 2. 並行執行所有方案（靶場環境安全）
        results = []
        for plan in plans:
            result = await self._execute_attack_plan(plan)
            results.append((plan, result))
        
        # 3. 選擇最佳結果
        best_result = max(
            results,
            key=lambda x: len(x[1].get("vulnerabilities", []))
        )
        best_plan, execution_result = best_result
        
        logger.info(f"⚡ Best result: {len(execution_result.get('vulnerabilities', []))} vulns found")
        
        # 4. 即時學習（靶場環境總是學習）
        learning_info = None
        if self.learning_enabled:
            # 從所有嘗試中學習（包括失敗的）
            all_samples = []
            for plan, result in results:
                samples = await self._learn_from_execution(
                    attack_plan=plan,
                    execution_result=result,
                    scenario={**(scenario or {}), "environment": "sandbox"}
                )
                all_samples.append(samples)
            
            learning_info = {
                "samples_collected": sum(s.get("samples_collected", 0) for s in all_samples),
                "strategies_explored": len(plans),
                "immediate_learning": True
            }
        
        return ExecutionResult(
            success=execution_result.get("success", False),
            vulnerabilities=execution_result.get("vulnerabilities", []),
            attack_plan=best_plan,
            execution_details=execution_result,
            learning_info=learning_info
        )
    
    async def _production_execution(
        self,
        target: str,
        objective: str,
        execution_context: 'ExecutionContext',
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> ExecutionResult:
        """生產環境執行：保守式策略
        
        特點：
        - 使用最佳已知策略（不探索）
        - 僅在結果顯著偏離預期時學習
        - 低風險容忍度
        """
        logger.info(
            f"🛡️ Production execution: conservative mode "
            f"(risk_tolerance={execution_context.risk_tolerance})"
        )
        
        # 1. 生成單一最優方案（基於歷史最佳）
        attack_plan = await self._generate_enhanced_plan(
            target=target,
            objective=objective,
            scenario={**(scenario or {}), "use_best_strategy": True},
            constraints={**(constraints or {}), "conservative_mode": True}
        )
        
        logger.info(f"📋 Using best known strategy: {attack_plan.plan_id}")
        
        # 2. 執行攻擊
        execution_result = await self._execute_attack_plan(attack_plan)
        
        logger.info(f"⚡ Execution completed: success={execution_result.get('success', False)}")
        
        # 3. 選擇性學習（僅在顯著偏離時）
        learning_info = None
        if self.learning_enabled:
            # 評估結果是否值得學習
            should_learn = self._should_learn_from_production(execution_result)
            
            if should_learn:
                logger.info("📚 Production result worth learning, updating knowledge")
                learning_info = await self._learn_from_execution(
                    attack_plan=attack_plan,
                    execution_result=execution_result,
                    scenario={**(scenario or {}), "environment": "production"}
                )
            else:
                logger.info("✓ Production result as expected, no learning needed")
                learning_info = {"selective_learning": True, "learned": False}
        
        return ExecutionResult(
            success=execution_result.get("success", False),
            vulnerabilities=execution_result.get("vulnerabilities", []),
            attack_plan=attack_plan,
            execution_details=execution_result,
            learning_info=learning_info
        )
    
    def _should_learn_from_production(self, execution_result: dict) -> bool:
        """判斷生產環境結果是否值得學習
        
        學習條件：
        - 發現了意外的漏洞（新發現）
        - 出現了未預期的錯誤（失敗案例）
        - 性能顯著優於預期（效率提升）
        """
        # 簡化邏輯：發現漏洞或失敗都值得學習
        has_vulns = len(execution_result.get("vulnerabilities", [])) > 0
        has_errors = not execution_result.get("success", False)
        
        return has_vulns or has_errors

    async def _generate_enhanced_plan(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict],
        constraints: Optional[dict]
    ) -> AttackPlan:
        """生成 RAG 增強的攻擊計劃"""
        
        # 查詢 RAG：尋找相似的成功案例
        try:
            rag_context = await self.rag_engine.retrieve_similar_cases(
                target=target,
                objective=objective
            )
            logger.info(f"🔍 RAG found {len(rag_context.get('similar_cases', []))} similar cases")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}, using fallback")
            rag_context = {}
        
        # 使用 AI Commander 生成計劃
        plan_context = {
            "target": target,
            "objective": objective,
            "rag_context": rag_context,
            "scenario": scenario,
            "constraints": constraints or {}
        }
        
        # 調用 AI Commander 生成計劃
        try:
            from .ai_commander import AITaskType
            plan_result = await self.ai_commander.execute_command(
                task_type=AITaskType.ATTACK_PLANNING,
                context=plan_context
            )
            
            if plan_result.get("success"):
                plan_data = plan_result.get("plan", {})
                return AttackPlan(
                    plan_id=plan_data.get("plan_id", f"plan_{int(time.time())}"),
                    target=AttackTarget(url=target),
                    objective=objective,
                    steps=plan_data.get("phases", []),
                    metadata={
                        "scenario_id": scenario.get("id") if scenario else None,
                        "rag_enhanced": bool(rag_context),
                        "constraints": constraints
                    }
                )
        except Exception as e:
            logger.warning(f"AI Commander plan generation failed: {e}, using simple plan")
        
        # 降級：創建簡單計劃
        return AttackPlan(
            plan_id=f"simple_plan_{int(time.time())}",
            target=AttackTarget(url=target),
            objective=objective,
            steps=[
                {"name": "reconnaissance", "description": "Initial recon"},
                {"name": "vulnerability_scan", "description": f"Scan for {objective}"},
                {"name": "exploitation", "description": "Exploit found vulnerabilities"}
            ],
            metadata={"fallback": True}
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
        """自動訓練模型"""
        try:
            logger.info("🎓 Starting auto-training...")
            
            # 從 buffer 採樣
            batch = self.experience_manager.sample(batch_size=32)
            
            if not batch:
                logger.warning("No samples available for training")
                return False
            
            # 訓練模型
            config = ModelTrainingConfig(
                epochs=10,
                learning_rate=0.001,
                batch_size=32
            )
            
            result = await self.model_trainer.train(
                samples=batch,
                config=config
            )
            
            # 更新訓練時間
            self.last_train_time = time.time()
            
            logger.info(f"✅ Training completed: loss={result.get('final_loss', 'N/A')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Auto-training failed: {e}", exc_info=True)
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
