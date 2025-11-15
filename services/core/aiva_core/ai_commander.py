"""
AI Commander - AIVA 中央 AI 指揮系統

統一指揮所有 AI 組件：
1. BioNeuronRAGAgent（Python 主控 AI）
2. RAG Engine（知識檢索增強）
3. Training Orchestrator（訓練系統）
4. Multi-Language AI Modules（Go/Rust/TypeScript AI）

架構設計：
- AI Commander 作為最高指揮層
- 各語言 AI 作為專業執行層
- RAG 提供知識支持
- Training 提供持續學習
"""

from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any
import logging

# 在try-except之前先定義logger
logger = logging.getLogger(__name__)

try:
    from services.aiva_common.ai import AIVAExperienceManager as ExperienceManager
    from services.aiva_common.ai.interfaces import IExperienceManager
    
    from .ai_engine import BioNeuronRAGAgent
    from .learning.model_trainer import ModelTrainer
    from .multilang_coordinator import MultiLanguageAICoordinator
    from .rag import KnowledgeBase, RAGEngine, VectorStore
    from .training.training_orchestrator import TrainingOrchestrator
except ImportError as e:
    logger.warning(f"Failed to import AI components: {e}")
    # 回退到核心模組
    try:
        from services.core.aiva_core.ai_engine import BioNeuronRAGAgent
        from services.core.aiva_core.learning.model_trainer import ModelTrainer
        from services.core.aiva_core.multilang_coordinator import MultiLanguageAICoordinator
        from services.core.aiva_core.rag import KnowledgeBase, RAGEngine, VectorStore
        from services.core.aiva_core.training.training_orchestrator import (
            TrainingOrchestrator,
        )
        # 使用介面的預設實現
        ExperienceManager = None
        IExperienceManager = None
    except ImportError:
        logger.error("Unable to import core AI components")
        # 設定預設值
        ExperienceManager = None
        IExperienceManager = None

logger = logging.getLogger(__name__)


class AITaskType(str, Enum):
    """AI 任務類型"""

    # 決策類
    ATTACK_PLANNING = "attack_planning"  # 攻擊計畫生成
    STRATEGY_DECISION = "strategy_decision"  # 策略決策
    RISK_ASSESSMENT = "risk_assessment"  # 風險評估

    # 執行類
    VULNERABILITY_DETECTION = "vulnerability_detection"  # 漏洞檢測
    EXPLOIT_EXECUTION = "exploit_execution"  # 漏洞利用
    CODE_ANALYSIS = "code_analysis"  # 代碼分析

    # 學習類
    EXPERIENCE_LEARNING = "experience_learning"  # 經驗學習
    MODEL_TRAINING = "model_training"  # 模型訓練
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # 知識檢索

    # 協調類
    MULTI_LANG_COORDINATION = "multi_lang_coordination"  # 多語言協調
    TASK_DELEGATION = "task_delegation"  # 任務委派


class AIComponent(str, Enum):
    """AI 組件類型"""

    BIO_NEURON_AGENT = "bio_neuron_agent"  # Python 主控 AI
    RAG_ENGINE = "rag_engine"  # RAG 引擎
    TRAINING_SYSTEM = "training_system"  # 訓練系統
    MULTILANG_COORDINATOR = "multilang_coordinator"  # 多語言協調器

    # 語言專屬 AI
    GO_AI_MODULE = "go_ai_module"  # Go AI 模組
    RUST_AI_MODULE = "rust_ai_module"  # Rust AI 模組
    TS_AI_MODULE = "ts_ai_module"  # TypeScript AI 模組


class AICommander:
    """AI 指揮官

    統一管理和協調所有 AI 組件，負責：
    1. 任務分析和分配
    2. AI 組件協調
    3. 決策整合
    4. 經驗積累
    5. 持續學習
    """

    def __init__(
        self,
        codebase_path: str = "/workspaces/AIVA",
        data_directory: Path | None = None,
    ) -> None:
        """初始化 AI 指揮官

        Args:
            codebase_path: 代碼庫路徑
            data_directory: 數據目錄
        """
        logger.info("🎖️ Initializing AI Commander...")

        self.data_directory = data_directory or Path("./data/ai_commander")
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # === 核心 AI 組件 ===

        # 1. Python 主控 AI（BioNeuronRAGAgent）
        logger.info("  Loading BioNeuronRAGAgent...")
        self.bio_neuron_agent = BioNeuronRAGAgent(codebase_path)

        # 2. RAG 系統（知識增強）
        logger.info("  Loading RAG Engine...")
        vector_store = VectorStore(
            backend="memory",  # 可配置為 chroma/faiss
            persist_directory=self.data_directory / "vectors",
        )
        knowledge_base = KnowledgeBase(
            vector_store=vector_store,
            data_directory=self.data_directory / "knowledge",
        )
        self.rag_engine = RAGEngine(knowledge_base=knowledge_base)

        # 3. 經驗管理和模型訓練
        logger.info("  Loading Training System...")

        # 整合 ExperienceManager 與資料庫後端
        experience_db_path = self.data_directory / "experience_db"
        experience_db_path.mkdir(parents=True, exist_ok=True)

        # 使用簡單的 JSON 檔案儲存後端（可擴展為資料庫）
        class SimpleStorageBackend:
            """簡單的檔案儲存後端"""

            def __init__(self, storage_path: Path):
                self.storage_path = storage_path
                self.experiences_file = storage_path / "experiences.json"
                if not self.experiences_file.exists():
                    import json

                    with open(self.experiences_file, "w", encoding="utf-8") as f:
                        json.dump([], f)

            async def add_experience(self, experience_data: dict):
                """添加經驗記錄"""
                import json

                try:
                    with open(self.experiences_file, encoding="utf-8") as f:
                        experiences = json.load(f)
                    experiences.append(experience_data)
                    with open(self.experiences_file, "w", encoding="utf-8") as f:
                        json.dump(experiences, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Failed to save experience: {e}")

            async def get_experiences(self, limit: int = 100) -> list[dict]:
                """獲取經驗記錄"""
                import json

                try:
                    with open(self.experiences_file, encoding="utf-8") as f:
                        experiences = json.load(f)
                    return experiences[-limit:]  # 返回最近的記錄
                except Exception as e:
                    logger.error(f"Failed to load experiences: {e}")
                    return []

        storage_backend = SimpleStorageBackend(experience_db_path)
        self.experience_manager = ExperienceManager(
            storage_backend=storage_backend,
        )
        self.model_trainer = ModelTrainer(
            # 移除 model_config 參數避免與 Pydantic 衝突
            # 配置將在後續通過方法設置
        )

        # 4. 訓練編排器（整合 RAG 和訓練）
        try:
            from .execution.plan_executor import PlanExecutor
            from .messaging.message_broker import MessageBroker
            from .training.scenario_manager import ScenarioManager
        except ImportError:
            from services.core.aiva_core.execution.plan_executor import PlanExecutor
            from services.core.aiva_core.messaging.message_broker import MessageBroker
            from services.core.aiva_core.training.scenario_manager import (
                ScenarioManager,
            )

        scenario_manager = ScenarioManager()

        try:
            message_broker = MessageBroker()
            plan_executor = PlanExecutor(message_broker=message_broker)
        except TypeError:
            # 如果 PlanExecutor 不接受 message_broker 參數，使用無參數初始化
            plan_executor = PlanExecutor()

        self.training_orchestrator = TrainingOrchestrator(
            scenario_manager=scenario_manager,
            rag_engine=self.rag_engine,
            plan_executor=plan_executor,
            experience_manager=self.experience_manager,
            model_trainer=self.model_trainer,
        )

        # 5. 多語言協調器
        logger.info("  Loading Multi-Language Coordinator...")
        self.multilang_coordinator = MultiLanguageAICoordinator()

        # === 指揮狀態 ===
        self.command_history: list[dict[str, Any]] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.component_status: dict[str, bool] = {
            component.value: True for component in AIComponent
        }

        logger.info("✅ AI Commander initialized successfully")
        logger.info(f"   - BioNeuronRAGAgent: {self.bio_neuron_agent is not None}")
        logger.info(f"   - RAG Engine: {self.rag_engine is not None}")
        logger.info(f"   - Training System: {self.training_orchestrator is not None}")
        logger.info(
            f"   - Multi-Language Coordinator: {self.multilang_coordinator is not None}"
        )

    async def execute_command(
        self,
        task_type: AITaskType,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """執行 AI 指令

        Args:
            task_type: 任務類型
            context: 任務上下文

        Returns:
            執行結果
        """
        logger.info(f"🎯 Executing AI Command: {task_type.value}")

        # 記錄指令
        command_id = f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
        command_record = {
            "command_id": command_id,
            "task_type": task_type.value,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "status": "started",
        }
        self.command_history.append(command_record)
        self.active_tasks[command_id] = command_record

        try:
            # 根據任務類型分派
            if task_type == AITaskType.ATTACK_PLANNING:
                result = await self._plan_attack(context)

            elif task_type == AITaskType.STRATEGY_DECISION:
                result = await self._make_strategy_decision(context)

            elif task_type == AITaskType.VULNERABILITY_DETECTION:
                result = await self._detect_vulnerabilities(context)

            elif task_type == AITaskType.EXPERIENCE_LEARNING:
                result = await self._learn_from_experience(context)

            elif task_type == AITaskType.MODEL_TRAINING:
                result = await self._train_model(context)

            elif task_type == AITaskType.KNOWLEDGE_RETRIEVAL:
                result = await self._retrieve_knowledge(context)

            elif task_type == AITaskType.MULTI_LANG_COORDINATION:
                result = await self._coordinate_multilang(context)

            else:
                result = {
                    "success": False,
                    "error": f"Unsupported task type: {task_type.value}",
                }

            # 更新狀態
            command_record["status"] = "completed"
            command_record["result"] = result
            command_record["end_time"] = datetime.now().isoformat()

            logger.info(
                f"✅ Command {command_id} completed: "
                f"success={result.get('success', False)}"
            )

        except Exception as e:
            logger.error(f"❌ Command {command_id} failed: {e}", exc_info=True)
            command_record["status"] = "failed"
            command_record["error"] = str(e)
            result = {"success": False, "error": str(e)}

        finally:
            del self.active_tasks[command_id]

        return result

    async def _plan_attack(self, context: dict[str, Any]) -> dict[str, Any]:
        """生成攻擊計畫（RAG 增強）

        Args:
            context: 包含 target, objective 等

        Returns:
            攻擊計畫結果
        """
        logger.info("📋 Generating attack plan with RAG enhancement...")

        target = context.get("target")
        objective = context.get("objective", "Comprehensive security assessment")
        constraints = context.get("constraints", {})

        if not target:
            return {"success": False, "error": "Target not specified"}

        try:
            # 1. 使用 RAG 檢索相關知識
            rag_context = self.rag_engine.enhance_attack_plan(
                target=target,
                objective=objective,
            )

            # 2. 從經驗庫獲取歷史成功案例
            historical_experiences = (
                await self.experience_manager.storage.get_experiences(limit=50)
                if self.experience_manager.storage
                else []
            )

            # 3. 使用 BioNeuronRAGAgent 生成計畫
            plan_prompt = self._build_plan_generation_prompt(
                target=target,
                objective=objective,
                rag_context=rag_context,
                historical_experiences=historical_experiences,
                constraints=constraints,
            )

            # 調用 BioNeuron 生成計畫
            plan_response = await self.bio_neuron_agent.generate_structured_output(
                prompt=plan_prompt,
                output_schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "target": {"type": "string"},
                        "objective": {"type": "string"},
                        "phases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "steps": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "expected_duration": {"type": "string"},
                                },
                            },
                        },
                        "risk_assessment": {"type": "string"},
                        "success_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            )

            # 4. 構建完整的攻擊計畫
            from uuid import uuid4

            plan_id = f"plan_{uuid4().hex[:12]}"

            attack_plan = {
                "plan_id": plan_id,
                "target": target,
                "objective": objective,
                "phases": plan_response.get("phases", []),
                "risk_assessment": plan_response.get("risk_assessment", ""),
                "success_criteria": plan_response.get("success_criteria", []),
                "rag_context": {
                    "similar_techniques": rag_context.get("similar_techniques", []),
                    "successful_experiences_count": len(historical_experiences),
                },
                "confidence": self._calculate_plan_confidence(
                    rag_context, historical_experiences
                ),
                "created_at": datetime.now().isoformat(),
            }

            logger.info(
                f"✅ Generated plan {plan_id} with {len(attack_plan['phases'])} phases, "
                f"confidence: {attack_plan['confidence']:.2f}"
            )

            return {
                "success": True,
                "plan": attack_plan,
                "confidence": attack_plan["confidence"],
            }

        except Exception as e:
            logger.error(f"Failed to generate attack plan: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "fallback_message": "Plan generation failed, using basic strategy",
            }

    def _build_plan_generation_prompt(
        self,
        target: str,
        objective: str,
        rag_context: dict[str, Any],
        historical_experiences: list[dict],
        constraints: dict[str, Any],
    ) -> str:
        """構建計畫生成提示詞 (優化版)

        優化重點:
        - 更詳細的技術描述
        - 成功案例引用
        - 失敗經驗警示
        - 動態策略建議
        """
        # 1. 基本資訊
        prompt = f"""Generate a comprehensive security testing attack plan for:

🎯 Target: {target}
📋 Objective: {objective}

"""

        # 2. RAG 知識庫相似技術 (詳細版)
        similar_techs = rag_context.get("similar_techniques", [])
        if similar_techs:
            prompt += "🔍 Similar Techniques from Knowledge Base:\n"
            for idx, tech in enumerate(similar_techs[:5], 1):  # 增加到 5 個
                prompt += f"{idx}. {tech.get('name', 'N/A')}\n"
                prompt += f"   - Description: {tech.get('description', 'N/A')}\n"
                prompt += f"   - Relevance Score: {tech.get('score', 0):.2f}\n"
                if tech.get("tags"):
                    prompt += f"   - Tags: {', '.join(tech.get('tags', []))}\n"
            prompt += "\n"

        # 3. 歷史經驗統計
        if historical_experiences:
            success_exps = [
                e for e in historical_experiences if e.get("score", 0) > 0.7
            ]
            medium_exps = [
                e for e in historical_experiences if 0.4 <= e.get("score", 0) <= 0.7
            ]
            failed_exps = [e for e in historical_experiences if e.get("score", 0) < 0.4]

            prompt += "📊 Historical Performance Analysis:\n"
            prompt += f"   - Total Experiences: {len(historical_experiences)}\n"
            prompt += f"   - ✅ Success Rate: {len(success_exps)/len(historical_experiences)*100:.1f}%\n"
            prompt += f"   - ⚠️ Partial Success: {len(medium_exps)/len(historical_experiences)*100:.1f}%\n"
            prompt += f"   - ❌ Failure Rate: {len(failed_exps)/len(historical_experiences)*100:.1f}%\n"

            # 引用成功案例
            if success_exps:
                prompt += "\n🌟 Top Successful Cases:\n"
                for exp in success_exps[:2]:
                    context = exp.get("context", {})
                    action = exp.get("action", {})
                    prompt += f"   - Strategy: {action.get('decision', 'N/A')}\n"
                    prompt += f"     Score: {exp.get('score', 0):.2f}, Type: {context.get('objective', 'N/A')}\n"

            # 警示失敗經驗
            if failed_exps:
                prompt += "\n⚠️ Lessons from Failed Attempts:\n"
                for exp in failed_exps[:2]:
                    result = exp.get("result", {})
                    prompt += f"   - Avoid: {result.get('error', 'Unknown error')}\n"

            prompt += "\n"

        # 4. 約束條件
        if constraints:
            prompt += "🚧 Constraints:\n"
            for key, value in constraints.items():
                prompt += f"   - {key}: {value}\n"
            prompt += "\n"

        # 5. 動態策略建議
        prompt += """🎯 Required Output Structure:
1. **Multi-Phase Plan**:
   - Phase 1: Reconnaissance (information gathering)
   - Phase 2: Vulnerability Analysis (identify weaknesses)
   - Phase 3: Exploitation Planning (prepare attack vectors)
   - Phase 4: Validation & Reporting (verify findings)

2. **Risk Assessment**:
   - Identify potential risks for each phase
   - Categorize as Low/Medium/High/Critical
   - Suggest mitigation strategies

3. **Success Criteria**:
   - Measurable objectives for each phase
   - Clear indicators of completion
   - Fallback plans if primary approach fails

4. **Dynamic Adaptation**:
   - Conditional steps based on intermediate results
   - Alternative paths if obstacles encountered
   - Real-time adjustment triggers

⚖️ Focus on: Practical, safe, authorized security testing approaches.
🔒 Ensure: Compliance with ethical hacking standards and legal boundaries.
"""
        return prompt

    def _calculate_plan_confidence(
        self, rag_context: dict[str, Any], historical_experiences: list[dict]
    ) -> float:
        """計算計畫信心度 (優化版)

        考慮因素:
        - RAG 相似技術數量和分數
        - 歷史成功率
        - 經驗數量充足度
        - 時間新鮮度

        Returns:
            信心度分數 (0.3-0.95 範圍)
        """
        confidence = 0.3  # 最低基礎信心度

        # 1. RAG 相似技術加成 (最高 +0.25)
        similar_techs = rag_context.get("similar_techniques", [])
        if similar_techs:
            # 考慮技術數量
            tech_count_bonus = min(len(similar_techs) * 0.03, 0.15)

            # 考慮技術相關性分數
            avg_score = (
                sum(t.get("score", 0) for t in similar_techs) / len(similar_techs)
                if similar_techs
                else 0
            )
            score_bonus = avg_score * 0.1

            confidence += tech_count_bonus + score_bonus

        # 2. 歷史經驗加成 (最高 +0.35)
        if historical_experiences:
            # 經驗數量充足度 (至少 10 個經驗才有充分參考價值)
            exp_count = len(historical_experiences)
            count_factor = min(exp_count / 10, 1.0)

            # 成功率計算
            success_exps = [
                e for e in historical_experiences if e.get("score", 0) > 0.7
            ]
            success_rate = len(success_exps) / exp_count if exp_count > 0 else 0

            # 時間新鮮度 (最近的經驗權重更高)
            recent_bonus = 0
            if exp_count > 0:
                # 檢查最近 7 天內的經驗
                from datetime import timedelta

                recent_threshold = (datetime.now() - timedelta(days=7)).isoformat()
                recent_count = len(
                    [
                        e
                        for e in historical_experiences
                        if e.get("timestamp", "") > recent_threshold
                    ]
                )
                recent_bonus = min(recent_count / exp_count * 0.05, 0.05)

            # 綜合歷史因素
            historical_bonus = (success_rate * count_factor * 0.3) + recent_bonus
            confidence += historical_bonus

        # 3. 組合效應加成 (RAG + 歷史都強時額外獎勵)
        if len(similar_techs) >= 3 and len(historical_experiences) >= 5:
            success_rate = len(
                [e for e in historical_experiences if e.get("score", 0) > 0.7]
            ) / len(historical_experiences)
            if success_rate > 0.7:
                confidence += 0.05  # 高質量知識庫加成

        # 4. 確保範圍在 0.3-0.95 之間
        confidence = max(0.3, min(confidence, 0.95))

        logger.debug(
            f"Plan confidence calculated: {confidence:.3f} "
            f"(techs={len(similar_techs)}, exps={len(historical_experiences)})"
        )

        return confidence

    async def _make_strategy_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """策略決策 (優化版)

        增強功能:
        - 更詳細的風險評估
        - 多維度信心度計算
        - 決策追蹤和審計
        - 動態調整建議

        Args:
            context: 決策上下文

        Returns:
            決策結果
        """
        logger.info("🤔 Making strategic decision with enhanced risk assessment...")

        try:
            situation = context.get("situation", {})
            options = context.get("options", [])
            constraints = context.get("constraints", {})

            # 1. 從經驗庫獲取相似情況的歷史決策
            historical_decisions = await self._get_similar_decisions(situation)

            # 2. 風險預評估
            risk_factors = self._assess_risk_factors(situation, constraints)

            # 3. 構建增強型決策提示詞
            decision_prompt = self._build_strategy_decision_prompt(
                situation, options, constraints, historical_decisions, risk_factors
            )

            # 4. 使用 BioNeuronRAGAgent 進行決策
            decision_response = await self.bio_neuron_agent.generate_structured_output(
                prompt=decision_prompt,
                output_schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "confidence": {"type": "number"},
                        "alternative_options": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "risks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "severity": {"type": "string"},
                                    "mitigation": {"type": "string"},
                                },
                            },
                        },
                        "success_indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "fallback_plan": {"type": "string"},
                    },
                },
            )

            # 5. 多維度信心度計算
            ai_confidence = decision_response.get("confidence", 0.5)
            historical_confidence = self._calculate_historical_confidence(
                historical_decisions
            )
            risk_adjusted_confidence = self._adjust_confidence_by_risk(
                base_confidence=(ai_confidence * 0.6) + (historical_confidence * 0.4),
                risk_factors=risk_factors,
            )

            # 6. 構建完整決策結果
            result = {
                "success": True,
                "decision": decision_response.get("decision", "proceed_with_caution"),
                "confidence": risk_adjusted_confidence,
                "reasoning": decision_response.get("reasoning", "Based on AI analysis"),
                "alternative_options": decision_response.get("alternative_options", []),
                "risks": decision_response.get("risks", []),
                "success_indicators": decision_response.get("success_indicators", []),
                "fallback_plan": decision_response.get(
                    "fallback_plan", "Abort and reassess"
                ),
                "risk_assessment": {
                    "overall_risk": risk_factors.get("overall_risk", "medium"),
                    "key_factors": risk_factors.get("factors", []),
                    "mitigation_required": risk_factors.get(
                        "mitigation_required", False
                    ),
                },
                "historical_reference_count": len(historical_decisions),
                "decision_metadata": {
                    "ai_confidence": ai_confidence,
                    "historical_confidence": historical_confidence,
                    "risk_adjustment": risk_adjusted_confidence
                    - ((ai_confidence * 0.6) + (historical_confidence * 0.4)),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.info(
                f"✅ Decision made: {result['decision']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"risk: {risk_factors.get('overall_risk', 'unknown')})"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Decision making failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "decision": "abort",
                "confidence": 0.0,
                "reasoning": "Decision process encountered an error. Aborting for safety.",
                "fallback_plan": "Manual review required",
            }

    def _assess_risk_factors(
        self, situation: dict[str, Any], constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """評估風險因素

        Returns:
            風險評估結果
        """
        factors = []
        risk_score = 0

        # 1. 目標環境風險
        if situation.get("target_type") == "production":
            factors.append("Production environment - High impact potential")
            risk_score += 3
        elif situation.get("target_type") == "staging":
            factors.append("Staging environment - Medium impact")
            risk_score += 1

        # 2. 時間約束風險
        if constraints.get("time_limit"):
            factors.append("Time-constrained operation - Reduced testing window")
            risk_score += 2

        # 3. 授權範圍風險
        if not constraints.get("authorized"):
            factors.append("⚠️ CRITICAL: Unauthorized testing - Legal risk")
            risk_score += 5

        # 4. 資料敏感度風險
        if situation.get("contains_sensitive_data"):
            factors.append("Sensitive data present - Privacy concerns")
            risk_score += 2

        # 5. 系統關鍵度風險
        if situation.get("system_criticality") == "high":
            factors.append("Critical system - Service disruption risk")
            risk_score += 3

        # 計算總體風險等級
        if risk_score >= 7:
            overall_risk = "critical"
            mitigation_required = True
        elif risk_score >= 4:
            overall_risk = "high"
            mitigation_required = True
        elif risk_score >= 2:
            overall_risk = "medium"
            mitigation_required = False
        else:
            overall_risk = "low"
            mitigation_required = False

        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "factors": factors,
            "mitigation_required": mitigation_required,
        }

    def _build_strategy_decision_prompt(
        self,
        situation: dict[str, Any],
        options: list[str],
        constraints: dict[str, Any],
        historical_decisions: list[dict],
        risk_factors: dict[str, Any],
    ) -> str:
        """構建策略決策提示詞"""
        prompt = f"""Analyze the following situation and make a strategic decision:

📋 **Situation Analysis**:
{situation}

⚖️ **Available Options**:
"""
        for idx, option in enumerate(options, 1):
            prompt += f"{idx}. {option}\n"

        if constraints:
            prompt += "\n🚧 **Constraints**:\n"
            for key, value in constraints.items():
                prompt += f"   - {key}: {value}\n"

        # 風險評估
        prompt += "\n⚠️ **Risk Assessment**:\n"
        prompt += f"   - Overall Risk Level: {risk_factors.get('overall_risk', 'unknown').upper()}\n"
        prompt += f"   - Risk Score: {risk_factors.get('risk_score', 0)}/10\n"
        if risk_factors.get("factors"):
            prompt += "   - Key Risk Factors:\n"
            for factor in risk_factors["factors"]:
                prompt += f"     • {factor}\n"

        # 歷史決策
        if historical_decisions:
            success_rate = (
                len([d for d in historical_decisions if d.get("score", 0) > 0.7])
                / len(historical_decisions)
                * 100
            )
            prompt += "\n📊 **Historical Decisions** (similar situations):\n"
            prompt += f"   - Total References: {len(historical_decisions)}\n"
            prompt += f"   - Success Rate: {success_rate:.1f}%\n"
            prompt += "   - Top Cases:\n"
            for hist in historical_decisions[:2]:
                prompt += f"     • Decision: {hist.get('action', {}).get('decision', 'N/A')}\n"
                prompt += f"       Outcome: {'✅ Success' if hist.get('score', 0) > 0.7 else '⚠️ Partial'}\n"

        prompt += """
🎯 **Required Output**:
Please provide a comprehensive decision包含:
1. **Primary Decision**: Clear, actionable choice
2. **Reasoning**: Detailed explanation of decision logic
3. **Confidence Level**: 0.0-1.0 based on available information
4. **Alternative Options**: Backup choices if primary fails
5. **Risk Analysis**: Specific risks with severity (Low/Medium/High/Critical) and mitigation strategies
6. **Success Indicators**: Measurable criteria to validate decision effectiveness
7. **Fallback Plan**: What to do if decision leads to negative outcomes

⚖️ **Decision Criteria**:
- Prioritize safety and authorization compliance
- Balance effectiveness with risk level
- Consider time and resource constraints
- Learn from historical outcomes
"""
        return prompt

    def _adjust_confidence_by_risk(
        self, base_confidence: float, risk_factors: dict[str, Any]
    ) -> float:
        """根據風險因素調整信心度

        高風險情況下降低信心度,確保謹慎決策
        """
        overall_risk = risk_factors.get("overall_risk", "medium")

        if overall_risk == "critical":
            # 關鍵風險：大幅降低信心度
            adjustment = -0.2
        elif overall_risk == "high":
            # 高風險：中度降低信心度
            adjustment = -0.1
        elif overall_risk == "medium":
            # 中等風險：略微降低
            adjustment = -0.05
        else:
            # 低風險：不調整或略微提升
            adjustment = 0.0

        adjusted = base_confidence + adjustment
        return max(0.1, min(adjusted, 0.95))  # 確保在合理範圍內

    async def _get_similar_decisions(self, situation: dict[str, Any]) -> list[dict]:
        """獲取相似情況的歷史決策"""
        if not self.experience_manager.storage:
            return []

        try:
            all_experiences = await self.experience_manager.storage.get_experiences(
                limit=100
            )
            # 簡單的相似度匹配（可以使用更複雜的語義相似度）
            similar_decisions = [
                exp
                for exp in all_experiences
                if exp.get("context", {}).get("type") == situation.get("type")
            ]
            return similar_decisions[:10]  # 返回前 10 個最相似的
        except Exception as e:
            logger.error(f"Failed to retrieve similar decisions: {e}")
            return []

    def _calculate_historical_confidence(
        self, historical_decisions: list[dict]
    ) -> float:
        """根據歷史決策計算信心度"""
        if not historical_decisions:
            return 0.5  # 無歷史數據時的基準值

        # 計算歷史決策的平均成功率
        success_count = len(
            [d for d in historical_decisions if d.get("score", 0) > 0.7]
        )
        return (
            success_count / len(historical_decisions) if historical_decisions else 0.5
        )

    async def _detect_vulnerabilities(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢測漏洞（協調多語言模組）

        Args:
            context: 檢測上下文

        Returns:
            檢測結果
        """
        logger.info("🔍 Detecting vulnerabilities across languages...")

        # 協調多語言 AI 模組
        target = context.get("target")
        vuln_types = context.get("vulnerability_types", [])

        # TODO: 實際協調邏輯
        # results = await self.multilang_coordinator.coordinate_detection(
        #     target=target,
        #     vuln_types=vuln_types
        # )

        return {
            "success": True,
            "vulnerabilities_found": 0,
            "languages_coordinated": ["python", "go", "rust"],
        }

    async def _learn_from_experience(self, context: dict[str, Any]) -> dict[str, Any]:
        """從經驗中學習

        Args:
            context: 包含 experience_sample

        Returns:
            學習結果
        """
        logger.info("📚 Learning from experience...")

        sample = context.get("experience_sample")
        if not sample:
            return {"success": False, "error": "No experience sample provided"}

        # 1. 添加到經驗管理器
        self.experience_manager.add_sample(sample)

        # 2. 添加到 RAG 知識庫
        self.rag_engine.learn_from_experience(sample)

        return {
            "success": True,
            "sample_quality": sample.quality_score,
            "knowledge_updated": True,
        }

    async def _train_model(self, context: dict[str, Any]) -> dict[str, Any]:
        """訓練模型

        Args:
            context: 訓練配置

        Returns:
            訓練結果
        """
        logger.info("🎓 Training AI model...")

        # 使用訓練編排器
        result = await self.training_orchestrator.train_model(
            min_samples=context.get("min_samples", 100),
            model_type=context.get("model_type", "supervised"),
        )

        return result

    async def _retrieve_knowledge(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢索知識

        Args:
            context: 包含 query

        Returns:
            檢索結果
        """
        logger.info("🔎 Retrieving knowledge from RAG...")

        query = context.get("query", "")
        top_k = context.get("top_k", 5)

        results = self.rag_engine.knowledge_base.search(
            query=query,
            top_k=top_k,
        )

        return {
            "success": True,
            "results_count": len(results),
            "results": [
                {
                    "title": entry.title,
                    "type": entry.type.value,
                    "content": entry.content[:200],
                    "success_rate": entry.success_rate,
                }
                for entry in results
            ],
        }

    async def _coordinate_multilang(self, context: dict[str, Any]) -> dict[str, Any]:
        """協調多語言 AI 模組

        Args:
            context: 協調上下文

        Returns:
            協調結果
        """
        logger.info("🌐 Coordinating multi-language AI modules...")

        # 使用多語言協調器
        # TODO: 實際協調邏輯

        return {
            "success": True,
            "modules_coordinated": ["go", "rust", "typescript"],
            "tasks_distributed": 0,
        }

    async def run_training_session(
        self,
        scenario_ids: list[str] | None = None,
        episodes_per_scenario: int = 10,
    ) -> dict[str, Any]:
        """運行訓練會話

        Args:
            scenario_ids: 場景 ID 列表
            episodes_per_scenario: 每個場景的回合數

        Returns:
            訓練結果
        """
        logger.info("🎓 Starting training session...")

        result = await self.training_orchestrator.run_training_batch(
            scenario_ids=scenario_ids,
            episodes_per_scenario=episodes_per_scenario,
            use_rag=True,  # 使用 RAG 增強
        )

        return result

    def get_status(self) -> dict[str, Any]:
        """獲取 AI 指揮官狀態

        Returns:
            狀態信息
        """
        return {
            "component_status": self.component_status,
            "active_tasks": len(self.active_tasks),
            "total_commands": len(self.command_history),
            "successful_commands": sum(
                1 for cmd in self.command_history if cmd.get("status") == "completed"
            ),
            "training_stats": self.training_orchestrator.get_training_statistics(),
            "knowledge_stats": self.rag_engine.get_statistics(),
            "experience_stats": self.experience_manager.get_statistics(),
        }

    def save_state(self) -> None:
        """保存 AI 指揮官狀態"""
        logger.info("💾 Saving AI Commander state...")

        # 保存 RAG 知識庫
        self.rag_engine.save_knowledge()

        # 保存經驗數據
        self.experience_manager.export_to_jsonl(
            self.data_directory / "experiences.jsonl"
        )

        # 保存訓練會話
        self.training_orchestrator.save_session()

        logger.info("✅ AI Commander state saved")
