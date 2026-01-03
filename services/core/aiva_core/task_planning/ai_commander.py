"""
AI Commander - AIVA 中央 AI 指揮系統

統一指揮所有 AI 組件：
1. 5M 特化神經網路決策引擎（無 LLM/NLU 依賴）
2. RAG Engine（知識檢索增強）
3. Experience Manager（經驗管理）
4. Multi-Language AI Modules（Go/Rust/TypeScript）

架構設計：
- AI Commander 作為最高指揮層
- 5M 神經網路作為核心決策引擎
- RAG 提供歷史經驗檢索
- 持續學習優化決策能力
"""

from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

# 直接導入所需組件，遵循 aiva_common 規範：有錯就報錯
from ..cognitive_core.neural.real_neural_core import RealDecisionEngine
from ..external_learning.experience_manager import ExperienceManager
from ..external_learning.learning.model_trainer import ModelTrainer
from ..core_capabilities.multilang_coordinator import MultiLanguageAICoordinator
from ..cognitive_core.rag import KnowledgeBase, RAGEngine, VectorStore
# 統一執行器 - 替代 TrainingOrchestrator（已廢棄）
from .unified_executor import UnifiedAttackExecutor
from ..core_capabilities.task_context import TaskContext, parse_user_input_to_context
# v2.0: 統一反饋架構支援
from .mode_manager import ModeManager, get_mode_manager
from .executor.execution_status_monitor import ExecutionContext, EnvironmentType

# v3.0: 預定義能力選單系統 - 無 NLU/LLM 依賴
from aiva_common.enums.capabilities import (
    AttackCapability,
    ScanCapability,
    ReconCapability,
    AnalysisCapability,
    ForensicCapability,
    ExploitCapability,
    ReportCapability,
    CapabilityParameter,
    get_capability_config,
    get_all_capabilities,
    validate_capability_params,
)
from aiva_common.enums.capability_executor import (
    CapabilityExecutor,
    ExecutionResult,
)


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
    ATTACK_EXECUTION = "attack_execution"  # 攻擊執行
    TWO_PHASE_SCAN = "two_phase_scan"  # 兩階段掃描

    # 學習類
    EXPERIENCE_LEARNING = "experience_learning"  # 經驗學習
    MODEL_TRAINING = "model_training"  # 模型訓練
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # 知識檢索
    CAPABILITY_QUERY = "capability_query"  # 能力查詢 (v11.0)

    # 協調類
    MULTI_LANG_COORDINATION = "multi_lang_coordination"  # 多語言協調
    TASK_DELEGATION = "task_delegation"  # 任務委派


class AIComponent(str, Enum):
    """AI 組件類型"""

    DECISION_ENGINE_5M = "decision_engine_5m"  # 5M 參數決策引擎
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
        # 直接初始化，不使用降級邏輯
        # 1. 5M 特化神經網路決策引擎（無 LLM/NLU 依賴）
        logger.info("  Loading 5M Decision Engine...")
        weights_path = self.data_directory / "weights" / "aiva_real_weights.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.decision_engine = RealDecisionEngine(
            use_5m_model=True,
            weights_path=str(weights_path) if weights_path.exists() else None
        )
        logger.info("  ✅ 5M Decision Engine loaded")

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
        self.knowledge_base = knowledge_base  # 保存引用以供後續使用
        logger.info("  ✅ RAG Engine loaded")

        # 3. 經驗管理和模型訓練 - 簡化版
        logger.info("  Loading Training System...")

        # 整合 ExperienceManager 與資料庫後端
        experience_db_path = self.data_directory / "experience_db"
        experience_db_path.mkdir(parents=True, exist_ok=True)
        
        # 使用統一的儲存後端
        from ..external_learning.experience_manager import ExperienceManager
        storage_backend = None  # TODO: 整合實際儲存後端

        # 注意：SimpleStorageBackend 已在內部使用，無需在此處創建
        # storage_backend = SimpleStorageBackend(experience_db_path)
        
        # 3. Experience Manager 初始化
        logger.info("  Loading ExperienceManager...")
        self.experience_manager = ExperienceManager(
            capacity=10000,
            auto_persist=True,
            persist_batch_size=100,
            unified_manager=None  # 可擴展為整合資料庫
        )
        logger.info("  ✅ ExperienceManager loaded")
        
        # 4. Model Trainer 初始化
        logger.info("  Loading ModelTrainer...")
        self.model_trainer = ModelTrainer()
        logger.info("  ✅ ModelTrainer loaded")

        # 5. 統一執行器（替代 TrainingOrchestrator）
        logger.info("  Loading Unified Attack Executor...")
        self.unified_executor = UnifiedAttackExecutor(
            learning_enabled=True,
            auto_train_threshold=100,
            data_directory=self.data_directory / "unified_executor"
        )
        logger.info("  ✅ Unified Attack Executor loaded")

        # 6. 多語言協調器
        logger.info("  Loading Multi-Language Coordinator...")
        self.multilang_coordinator = MultiLanguageAICoordinator()
        logger.info("  ✅ Multi-Language Coordinator loaded")

        # 7. 內部循環連接器 (v11.0 新增)
        logger.info("  Loading Internal Loop Connector...")
        from ..cognitive_core.internal_loop_connector import InternalLoopConnector
        self.internal_loop = InternalLoopConnector(rag_kb=self.knowledge_base)
        logger.info("  ✅ Internal Loop Connector loaded")

        # 8. 模式管理器 (v2.0 統一反饋架構)
        logger.info("  Loading Mode Manager...")
        self.mode_manager = get_mode_manager(
            config_path=self.data_directory / "mode_config.json"
        )
        logger.info(f"  ✅ Mode Manager loaded: {self.mode_manager.get_mode().value}")

        # 9. 能力選單執行器 (v3.0 - 無 NLU/LLM)
        logger.info("  Loading Capability Executor...")
        self.capability_executor = CapabilityExecutor(
            decision_engine=self.decision_engine
        )
        logger.info("  ✅ Capability Executor loaded (menu-based, no NLU/LLM)")

        # === 指揮狀態 ===
        self.command_history: list[dict[str, Any]] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.component_status: dict[str, bool] = {
            component.value: True for component in AIComponent
        }

        # === v11.0: 系統啟動時自動同步能力 ===
        logger.info("🔄 Starting auto-sync of capabilities...")
        try:
            import asyncio
            # 創建事件循環來執行 async 方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sync_result = loop.run_until_complete(
                self.internal_loop.sync_capabilities_to_rag(
                    force_refresh=False,
                    target_scope="core",
                    target_module="core"
                )
            )
            loop.close()
            
            if sync_result.success:
                logger.info(f"✅ Auto-sync completed: {sync_result.total_synced} capabilities synced")
            else:
                logger.warning(f"⚠️ Auto-sync had issues: {sync_result.error_message}")
        except Exception as e:
            logger.error(f"❌ Auto-sync failed: {e}")
            # 不阻止系統啟動，繼續運行

        logger.info("✅ AI Commander initialized successfully")
        logger.info("   - All components loaded (no fallback mode)")
        logger.info("   - 5M Decision Engine: ✅")
        logger.info("   - Capability Executor: ✅ (menu-based)")
        logger.info("   - RAG Engine: ✅")
        logger.info("   - ExperienceManager: ✅")
        logger.info("   - ModelTrainer: ✅")
        logger.info("   - Unified Attack Executor: ✅")
        logger.info("   - Multi-Language Coordinator: ✅")
        logger.info("   - Internal Loop Connector: ✅")
        logger.info(f"   - Mode Manager: ✅ (mode={self.mode_manager.get_mode().value})")

    # =========================================================================
    # 選單式能力操作 (v3.0) - 無 NLU/LLM 依賴
    # =========================================================================

    def list_available_capabilities(
        self, category: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """列出所有可用能力選單
        
        Args:
            category: 類別過濾 (attack/scan/recon/analysis/forensic/exploit/report)
            
        Returns:
            按類別分組的能力列表
        """
        capabilities = self.capability_executor.list_capabilities(category)
        
        # 轉換為字典格式
        result = {}
        for cat, cap_list in capabilities.items():
            result[cat] = [
                {
                    "id": cap.id,
                    "name": cap.name,
                    "description": cap.description,
                    "required_params": cap.required_params,
                    "optional_params": cap.optional_params,
                    "risk_level": cap.risk_level,
                }
                for cap in cap_list
            ]
        
        return result

    async def execute_capability(
        self,
        capability: str,
        target: str,
        parameters: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """執行選定的能力
        
        選單式操作：能力從預定義列表選擇，只需輸入目標和參數
        
        Args:
            capability: 能力 ID (從 list_available_capabilities 選擇)
            target: 目標 (URL/IP/Domain/Path)
            parameters: 額外參數
            
        Returns:
            執行結果
            
        Example:
            >>> commander = AICommander()
            >>> # 1. 列出攻擊能力
            >>> caps = commander.list_available_capabilities("attack")
            >>> # 2. 選擇 SQL 注入
            >>> result = await commander.execute_capability(
            ...     capability="sql_injection",
            ...     target="https://example.com/api/users?id=1"
            ... )
        """
        logger.info(f"📋 Menu-based execution: {capability} -> {target}")
        
        # 使用能力執行器執行
        result = await self.capability_executor.execute(
            capability=capability,
            target=target,
            parameters=parameters,
            use_neural_optimization=True,  # 使用 5M 引擎優化
        )
        
        # 記錄到命令歷史
        self.command_history.append({
            "type": "capability_execution",
            "capability": capability,
            "target": target,
            "parameters": parameters,
            "success": result.success,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result

    def get_capability_info(self, capability: str) -> dict[str, Any] | None:
        """獲取單個能力的詳細信息
        
        Args:
            capability: 能力 ID
            
        Returns:
            能力詳細信息
        """
        cap_info = self.capability_executor.get_capability_info(capability)
        if cap_info:
            return {
                "id": cap_info.id,
                "name": cap_info.name,
                "description": cap_info.description,
                "category": cap_info.category,
                "required_params": cap_info.required_params,
                "optional_params": cap_info.optional_params,
                "risk_level": cap_info.risk_level,
                "default_timeout": cap_info.default_timeout,
            }
        return None

    def print_capability_menu(self, category: str | None = None) -> str:
        """生成可讀的能力選單
        
        Args:
            category: 類別過濾
            
        Returns:
            格式化的選單字串
        """
        return self.capability_executor.print_menu(category)

    def set_execution_mode(
        self,
        mode: str,
        persist: bool = True
    ) -> None:
        """設定執行模式（手動切換）
        
        Args:
            mode: 目標模式 ("sandbox" 或 "production")
            persist: 是否持久化到設定檔（預設 True）
        
        Raises:
            ValueError: 模式名稱無效
        
        使用範例:
            >>> commander.set_execution_mode("sandbox")  # 切換到靶場模式
            >>> commander.set_execution_mode("production")  # 切換到生產模式
        """
        self.mode_manager.set_mode(mode, persist=persist)
        logger.info(f"🔄 Execution mode set to: {mode}")
    
    def get_execution_mode(self) -> str:
        """獲取當前執行模式
        
        Returns:
            當前模式字串 ("sandbox" 或 "production")
        
        使用範例:
            >>> mode = commander.get_execution_mode()
            >>> print(f"Current mode: {mode}")
        """
        return self.mode_manager.get_mode().value

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

            elif task_type == AITaskType.ATTACK_EXECUTION:
                result = await self._execute_attack(context)

            elif task_type == AITaskType.TWO_PHASE_SCAN:
                result = await self._execute_two_phase_scan(context)

            elif task_type == AITaskType.EXPERIENCE_LEARNING:
                result = self._learn_from_experience(context)

            elif task_type == AITaskType.MODEL_TRAINING:
                result = await self._train_model(context)

            elif task_type == AITaskType.KNOWLEDGE_RETRIEVAL:
                result = self._retrieve_knowledge(context)

            elif task_type == AITaskType.CAPABILITY_QUERY:
                result = await self.query_my_capabilities(
                    query=context.get("query", ""),
                    filters=context.get("filters"),
                    top_k=context.get("top_k", 5)
                )

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

            # 3. 使用 5M Decision Engine 生成計畫
            # 將目標和歷史數據編碼為神經網路輸入
            target_features = self._encode_target_for_neural(target, rag_context)
            
            # 調用 5M 決策引擎生成攻擊向量和策略
            neural_decision = self.decision_engine.generate_decision(
                target_info=target_features,
                context={
                    "objective": objective,
                    "historical_count": len(historical_experiences),
                    "constraints": constraints,
                }
            )

            # 使用神經網路輸出構建計畫結構
            plan_response = self._build_plan_from_neural_decision(
                neural_decision=neural_decision,
                target=target,
                objective=objective,
                rag_context=rag_context,
                historical_experiences=historical_experiences,
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
        """構建計畫生成提示詞 (優化版 - 降低認知複雜度)

        優化重點:
        - 將複雜邏輯提取為獨立方法
        - 降低嵌套層次
        - 提高可讀性和維護性
        """
        # 構建基本提示詞
        prompt = self._build_base_prompt(target, objective)
        
        # 添加各個部分
        prompt += self._build_rag_section(rag_context)
        prompt += self._build_historical_section(historical_experiences)
        prompt += self._build_constraints_section(constraints)
        prompt += self._build_output_structure()
        
        return prompt

    def _build_base_prompt(self, target: str, objective: str) -> str:
        """構建基本提示詞部分"""
        return f"""Generate a comprehensive security testing attack plan for:

🎯 Target: {target}
📋 Objective: {objective}

"""

    def _build_rag_section(self, rag_context: dict[str, Any]) -> str:
        """構建 RAG 知識庫部分"""
        similar_techs = rag_context.get("similar_techniques", [])
        if not similar_techs:
            return ""
            
        section = "🔍 Similar Techniques from Knowledge Base:\n"
        for idx, tech in enumerate(similar_techs[:5], 1):
            section += f"{idx}. {tech.get('name', 'N/A')}\n"
            section += f"   - Description: {tech.get('description', 'N/A')}\n"
            section += f"   - Relevance Score: {tech.get('score', 0):.2f}\n"
            if tech.get("tags"):
                section += f"   - Tags: {', '.join(tech.get('tags', []))}\n"
        return section + "\n"

    def _build_historical_section(self, historical_experiences: list[dict]) -> str:
        """構建歷史經驗部分"""
        if not historical_experiences:
            return ""
            
        # 分類經驗
        categories = self._categorize_experiences(historical_experiences)
        
        # 構建統計信息
        section = self._build_experience_stats(historical_experiences, categories)
        
        # 添加成功案例和失敗教訓
        section += self._build_success_cases(categories['success'])
        section += self._build_failure_lessons(categories['failed'])
        
        return section + "\n"

    def _categorize_experiences(self, experiences: list[dict]) -> dict[str, list[dict]]:
        """將經驗按效果分類"""
        return {
            'success': [e for e in experiences if e.get("score", 0) > 0.7],
            'medium': [e for e in experiences if 0.4 <= e.get("score", 0) <= 0.7],
            'failed': [e for e in experiences if e.get("score", 0) < 0.4]
        }

    def _build_experience_stats(self, all_exps: list[dict], categories: dict[str, list[dict]]) -> str:
        """構建經驗統計信息"""
        total = len(all_exps)
        success_rate = len(categories['success']) / total * 100
        medium_rate = len(categories['medium']) / total * 100
        failure_rate = len(categories['failed']) / total * 100
        
        return f"""📊 Historical Performance Analysis:
   - Total Experiences: {total}
   - ✅ Success Rate: {success_rate:.1f}%
   - ⚠️ Partial Success: {medium_rate:.1f}%
   - ❌ Failure Rate: {failure_rate:.1f}%
"""

    def _build_success_cases(self, success_exps: list[dict]) -> str:
        """構建成功案例部分"""
        if not success_exps:
            return ""
            
        section = "\n🌟 Top Successful Cases:\n"
        for exp in success_exps[:2]:
            context = exp.get("context", {})
            action = exp.get("action", {})
            section += f"   - Strategy: {action.get('decision', 'N/A')}\n"
            section += f"     Score: {exp.get('score', 0):.2f}, Type: {context.get('objective', 'N/A')}\n"
        return section

    def _build_failure_lessons(self, failed_exps: list[dict]) -> str:
        """構建失敗教訓部分"""
        if not failed_exps:
            return ""
            
        section = "\n⚠️ Lessons from Failed Attempts:\n"
        for exp in failed_exps[:2]:
            result = exp.get("result", {})
            section += f"   - Avoid: {result.get('error', 'Unknown error')}\n"
        return section

    def _build_constraints_section(self, constraints: dict[str, Any]) -> str:
        """構建約束條件部分"""
        if not constraints:
            return ""
            
        section = "🚧 Constraints:\n"
        for key, value in constraints.items():
            section += f"   - {key}: {value}\n"
        return section + "\n"

    def _build_output_structure(self) -> str:
        """構建輸出結構要求部分"""
        return """🎯 Required Output Structure:
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

    # ===== 5M Neural Network 輔助方法 =====
    
    def _encode_target_for_neural(
        self, target: str, rag_context: dict[str, Any]
    ) -> dict[str, Any]:
        """將目標信息編碼為神經網路輸入特徵
        
        5M 參數模型使用預定義特徵向量，而非 LLM 文本理解
        """
        import hashlib
        
        # 從目標字串提取特徵
        target_hash = int(hashlib.md5(target.encode()).hexdigest()[:8], 16) / (16**8)
        
        # 從 RAG 上下文提取特徵
        similar_techs = rag_context.get("similar_techniques", [])
        tech_scores = [t.get("score", 0.5) for t in similar_techs[:5]]
        avg_similarity = sum(tech_scores) / len(tech_scores) if tech_scores else 0.5
        
        return {
            "target_hash": target_hash,
            "target_length": len(target) / 100.0,  # 正規化長度
            "has_ip": 1.0 if any(c.isdigit() for c in target.split('.')) else 0.0,
            "has_domain": 1.0 if '.' in target else 0.0,
            "rag_similarity": avg_similarity,
            "rag_tech_count": len(similar_techs) / 10.0,
        }
    
    def _encode_situation_for_neural(
        self, situation: dict, options: list, risk_factors: dict
    ) -> dict[str, Any]:
        """將情況編碼為神經網路決策輸入"""
        return {
            "option_count": len(options) / 10.0,
            "risk_level": risk_factors.get("overall_risk", 0.5),
            "urgency": situation.get("urgency", 0.5),
            "complexity": situation.get("complexity", 0.5),
            "resource_available": situation.get("resources", 1.0),
        }
    
    def _build_plan_from_neural_decision(
        self,
        neural_decision: dict,
        target: str,
        objective: str,
        rag_context: dict[str, Any],
        historical_experiences: list[dict],
    ) -> dict[str, Any]:
        """從神經網路輸出構建完整的攻擊計畫
        
        將 5M 模型的向量輸出轉換為結構化計畫
        """
        from uuid import uuid4
        
        attack_vector = neural_decision.get("attack_vector", "reconnaissance")
        confidence = neural_decision.get("confidence", 0.5)
        recommended_tools = neural_decision.get("recommended_tools", [])
        
        # 預定義的階段模板（5M 模型選擇而非生成）
        phase_templates = {
            "reconnaissance": {
                "name": "Reconnaissance",
                "description": "Information gathering and target enumeration",
                "steps": ["Domain/IP enumeration", "Port scanning", "Service detection", "Technology fingerprinting"],
                "expected_duration": "1-2 hours"
            },
            "vulnerability_scan": {
                "name": "Vulnerability Analysis",
                "description": "Identify potential vulnerabilities",
                "steps": ["Automated vulnerability scan", "Manual verification", "CVE mapping", "Risk assessment"],
                "expected_duration": "2-4 hours"
            },
            "exploitation": {
                "name": "Exploitation Planning",
                "description": "Plan and validate attack vectors",
                "steps": ["Exploit selection", "Payload preparation", "Environment setup", "Attack simulation"],
                "expected_duration": "3-6 hours"
            },
            "reporting": {
                "name": "Validation & Reporting",
                "description": "Document findings and recommendations",
                "steps": ["Finding documentation", "Evidence collection", "Risk scoring", "Remediation advice"],
                "expected_duration": "2-3 hours"
            }
        }
        
        # 根據攻擊向量選擇階段組合
        if attack_vector in ["recon", "reconnaissance", "information_gathering"]:
            selected_phases = ["reconnaissance", "vulnerability_scan"]
        elif attack_vector in ["exploit", "attack", "penetration"]:
            selected_phases = ["reconnaissance", "vulnerability_scan", "exploitation", "reporting"]
        else:
            selected_phases = ["reconnaissance", "vulnerability_scan", "reporting"]
        
        phases = [phase_templates[p] for p in selected_phases if p in phase_templates]
        
        # 根據推薦工具調整步驟
        if recommended_tools:
            for phase in phases:
                if "scan" in phase["name"].lower() and recommended_tools:
                    phase["steps"].insert(0, f"Use tools: {', '.join(recommended_tools[:3])}")
        
        # 風險評估（基於信心度）
        if confidence > 0.8:
            risk_assessment = "Low - High confidence in approach based on similar historical cases"
        elif confidence > 0.6:
            risk_assessment = "Medium - Moderate confidence, recommend careful monitoring"
        else:
            risk_assessment = "High - Lower confidence, consider alternative approaches"
        
        return {
            "plan_id": str(uuid4()),
            "target": target,
            "objective": objective,
            "phases": phases,
            "risk_assessment": risk_assessment,
            "success_criteria": [
                "Complete information gathering for target",
                "Identify at least 3 potential vulnerabilities",
                "Validate findings with evidence",
                "Generate comprehensive report"
            ],
            "neural_confidence": confidence,
            "attack_vector": attack_vector,
        }
    
    def _build_strategy_from_neural_decision(
        self,
        neural_decision: dict,
        situation: dict,
        options: list,
        historical_decisions: list[dict],
        risk_factors: dict,
    ) -> dict[str, Any]:
        """從神經網路輸出構建策略決策響應
        
        將 5M 模型輸出轉換為決策結構
        """
        attack_vector = neural_decision.get("attack_vector", "conservative")
        confidence = neural_decision.get("confidence", 0.5)
        recommended_tools = neural_decision.get("recommended_tools", [])
        
        # 根據神經網路輸出選擇最佳選項
        if options:
            # 使用信心度加權選擇
            selected_idx = int(confidence * len(options)) % len(options)
            selected_option = options[selected_idx]
        else:
            selected_option = attack_vector
        
        # 預定義的推理模板
        reasoning_templates = {
            "high_confidence": f"基於 5M 決策引擎分析，{attack_vector} 策略有 {confidence:.1%} 信心度。歷史數據支持此決策。",
            "medium_confidence": f"5M 引擎建議 {attack_vector} 策略，信心度 {confidence:.1%}。建議謹慎執行並監控結果。",
            "low_confidence": f"當前情況複雜，5M 引擎信心度為 {confidence:.1%}。建議採取保守的 {attack_vector} 策略。"
        }
        
        if confidence > 0.7:
            reasoning = reasoning_templates["high_confidence"]
        elif confidence > 0.4:
            reasoning = reasoning_templates["medium_confidence"]
        else:
            reasoning = reasoning_templates["low_confidence"]
        
        # 構建風險列表
        overall_risk = risk_factors.get("overall_risk", 0.5)
        risks = []
        if overall_risk > 0.7:
            risks.append({
                "description": "High-risk operation detected",
                "severity": "high",
                "mitigation": "Implement staged rollout with manual checkpoints"
            })
        if confidence < 0.5:
            risks.append({
                "description": "Low confidence decision",
                "severity": "medium", 
                "mitigation": "Prepare fallback strategies before execution"
            })
        
        return {
            "decision": selected_option,
            "reasoning": reasoning,
            "confidence": confidence,
            "alternative_options": [opt for opt in options if opt != selected_option][:3],
            "risks": risks,
            "success_indicators": [
                f"成功執行 {attack_vector} 策略",
                "達成預期目標",
                "無意外錯誤或警報"
            ],
            "fallback_plan": f"如果 {selected_option} 失敗，考慮 {options[0] if options else 'conservative approach'}",
            "recommended_tools": recommended_tools,
        }

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

            # 3. 使用 5M Decision Engine 進行策略決策
            # 將情況編碼為神經網路輸入
            situation_features = self._encode_situation_for_neural(
                situation, options, risk_factors
            )
            
            # 4. 調用 5M 決策引擎
            neural_decision = self.decision_engine.generate_decision(
                target_info=situation_features,
                context={
                    "options": options,
                    "constraints": constraints,
                    "historical_count": len(historical_decisions),
                }
            )
            
            # 從神經網路輸出構建決策響應
            decision_response = self._build_strategy_from_neural_decision(
                neural_decision=neural_decision,
                situation=situation,
                options=options,
                historical_decisions=historical_decisions,
                risk_factors=risk_factors,
            )

            # 5. 多維度信心度計算
            ai_confidence = decision_response.get("confidence", neural_decision.get("confidence", 0.5))
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
        """檢測漏洞（調用功能模組）

        Args:
            context: 檢測上下文 {
                "target": str,  # 目標 URL
                "vulnerability_types": list[str],  # ["sqli", "xss", "ssrf", "idor"]
                "deep_scan": bool,  # 是否深度掃描
            }

        Returns:
            檢測結果
        """
        logger.info("🔍 AI 控制: 開始漏洞檢測...")

        target = context.get("target")
        if not target:
            return {"success": False, "error": "No target specified"}

        vuln_types = context.get("vulnerability_types", ["sqli", "xss", "ssrf", "idor"])
        deep_scan = context.get("deep_scan", False)

        results = {
            "success": True,
            "target": target,
            "vulnerabilities_found": [],
            "modules_executed": [],
            "total_findings": 0,
        }

        try:
            # 動態導入功能模組（按需加載）
            module_map = {
                "sqli": "services.features.function_sqli.worker",
                "xss": "services.features.function_xss.worker",
                "ssrf": "services.features.function_ssrf.worker",
                "idor": "services.features.function_idor.worker",
            }

            for vuln_type in vuln_types:
                if vuln_type not in module_map:
                    logger.warning(f"⚠️ 未知漏洞類型: {vuln_type}")
                    continue

                try:
                    # 動態導入 Worker
                    module_path = module_map[vuln_type]
                    module = __import__(module_path, fromlist=["*"])
                    
                    # 獲取 Worker 類
                    worker_class_name = f"{vuln_type.capitalize()}WorkerService"
                    if vuln_type == "sqli":
                        worker_class_name = "SqliWorkerService"
                    elif vuln_type == "xss":
                        worker_class_name = "XssWorkerService"
                    elif vuln_type == "ssrf":
                        worker_class_name = "SsrfWorkerService"
                    elif vuln_type == "idor":
                        worker_class_name = "IdorWorkerService"
                    
                    worker_class = getattr(module, worker_class_name)
                    worker = worker_class()

                    # 構建任務
                    from services.aiva_common.schemas.tasks import (
                        FunctionTaskPayload,
                        FunctionTaskTarget,
                    )

                    task = FunctionTaskPayload(
                        task_id=f"task_ai_{vuln_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        scan_id=f"scan_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        target=FunctionTaskTarget(
                            url=target,
                            method="GET",
                        ),
                        priority=8 if deep_scan else 5,
                    )

                    # 執行檢測
                    logger.info(f"   🎯 執行 {vuln_type.upper()} 檢測...")
                    detection_result = await worker.process_task(task)

                    # 收集結果
                    if detection_result:
                        findings_count = len(detection_result.get("findings", []))
                        results["vulnerabilities_found"].extend(
                            detection_result.get("findings", [])
                        )
                        results["modules_executed"].append(vuln_type)
                        results["total_findings"] += findings_count
                        logger.info(f"   ✅ {vuln_type.upper()}: 發現 {findings_count} 個漏洞")

                except Exception as e:
                    logger.error(f"   ❌ {vuln_type.upper()} 模組執行失敗: {e}")
                    results[vuln_type + "_error"] = str(e)

            logger.info(
                f"✅ AI 控制: 漏洞檢測完成 - 共發現 {results['total_findings']} 個漏洞"
            )

        except Exception as e:
            logger.error(f"❌ AI 控制: 漏洞檢測失敗 - {e}", exc_info=True)
            results["success"] = False
            results["error"] = str(e)

        return results

    def _learn_from_experience(self, context: dict[str, Any]) -> dict[str, Any]:
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

    def _retrieve_knowledge(self, context: dict[str, Any]) -> dict[str, Any]:
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
    
    async def query_my_capabilities(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5
    ) -> dict[str, Any]:
        """查詢自身能力（v11.0 - 內部循環增強）
        
        使用內部循環連接器從 RAG 查詢 AIVA 自身的能力數據
        
        Args:
            query: 查詢文本（例如："如何執行掃描？"、"神經網路訓練流程"）
            filters: 過濾條件 (例如 {'module': 'cognitive_core', 'scope': 'CORE'})
            top_k: 返回結果數量
            
        Returns:
            包含匹配能力的結果字典
            
        Example:
            >>> result = await ai_commander.query_my_capabilities(
            ...     query="如何執行神經網路訓練？",
            ...     filters={"module": "cognitive_core"},
            ...     top_k=3
            ... )
            >>> print(f"Found {result['total_found']} capabilities")
            >>> for cap in result['capabilities']:
            ...     print(f"  - {cap['name']}: {cap['description']}")
        """
        logger.info(f"🧠 AI Commander querying self capabilities: '{query}'")
        
        try:
            # 使用內部循環連接器查詢
            rag_result = await self.internal_loop.query_capabilities(
                query=query,
                filters=filters,
                top_k=top_k
            )
            
            # 轉換為適合返回的格式
            capabilities = []
            for result in rag_result.results:
                cap_data = result.get('metadata', {})
                capabilities.append({
                    "name": cap_data.get('name', 'unknown'),
                    "description": cap_data.get('description', ''),
                    "module": cap_data.get('module', 'unknown'),
                    "category": cap_data.get('category', 'unknown'),
                    "file_path": cap_data.get('file_path', ''),
                    "relevance_score": result.get('score', 0.0)
                })
            
            logger.info(f"✅ Found {len(capabilities)} matching capabilities")
            
            return {
                "success": True,
                "query": query,
                "total_found": rag_result.total_found,
                "capabilities": capabilities,
                "timestamp": rag_result.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to query capabilities: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "total_found": 0,
                "capabilities": []
            }

    async def _coordinate_multilang(self, context: dict[str, Any]) -> dict[str, Any]:
        """協調掃描引擎（Python/TypeScript/Rust/Go）

        Args:
            context: 協調上下文 {
                "targets": list[str],  # 目標列表
                "scan_strategy": str,  # fast/balanced/comprehensive/aggressive/smart
                "scan_id": str,  # 掃描 ID
                "max_depth": int,  # 最大深度
                "timeout": int,  # 超時時間
            }

        Returns:
            掃描結果
        """
        logger.info("🌐 AI 控制: 協調多引擎掃描...")

        targets = context.get("targets", [])
        if not targets:
            return {"success": False, "error": "No targets specified"}

        strategy = context.get("scan_strategy", "balanced")
        scan_id = context.get("scan_id", f"scan_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        max_depth = context.get("max_depth", 3)
        # 注意：timeout 參數已獲取但在當前實現中未使用
        # timeout = context.get("timeout", 600)

        try:
            # 導入 MultiEngineCoordinator
            from services.scan.coordinators.multi_engine_coordinator import (
                MultiEngineCoordinator,
            )

            # 初始化協調器
            coordinator = MultiEngineCoordinator()
            await coordinator.initialize()

            logger.info(f"   🎯 使用策略: {strategy}")
            logger.info(f"   🎯 目標數量: {len(targets)}")

            # 根據策略選擇執行方法
            strategy_methods = {
                "fast": coordinator.execute_strategy_fast,
                "balanced": coordinator.execute_strategy_balanced,
                "comprehensive": coordinator.execute_strategy_comprehensive,
                "aggressive": coordinator.execute_strategy_aggressive,
                "smart": coordinator.execute_strategy_smart,
            }

            if strategy not in strategy_methods:
                logger.warning(f"⚠️ 未知策略 '{strategy}'，使用 'balanced'")
                strategy = "balanced"

            # 執行掃描
            scan_method = strategy_methods[strategy]
            result = await scan_method(
                scan_id=scan_id,
                targets=targets,
                max_depth=max_depth,
            )

            # 解析結果 (Phase1CompletedPayload 是 Pydantic model)
            urls_found = result.summary.urls_found if result.summary else 0
            assets_found = len(result.assets) if result.assets else 0
            execution_time = result.execution_time if hasattr(result, "execution_time") else 0
            engines_used = [
                engine for engine, data in result.engine_results.items()
                if data.get("status") == "completed"
            ] if result.engine_results else []

            logger.info(
                f"✅ AI 控制: 掃描完成 - 發現 {urls_found} 個 URL, {assets_found} 個資產, 耗時 {execution_time:.2f}s"
            )

            return {
                "success": True,
                "scan_id": scan_id,
                "strategy_used": strategy,
                "targets_scanned": len(targets),
                "urls_found": urls_found,
                "assets_found": assets_found,
                "execution_time": execution_time,
                "engines_used": engines_used,
                "full_result": result,
            }

        except Exception as e:
            logger.error(f"❌ AI 控制: 多引擎掃描失敗 - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "scan_id": scan_id,
            }

    async def _execute_attack(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行攻擊計畫（調用 AttackExecutor）

        Args:
            context: 執行上下文 {
                "plan": dict | AttackPlan,  # 攻擊計畫
                "target": dict | AttackTarget,  # 攻擊目標
                "mode": str,  # 執行模式: safe/testing/aggressive
                "ai_analysis": dict,  # AI 分析結果（可選）
            }

        Returns:
            執行結果
        """
        logger.info("⚔️ AI 控制: 執行攻擊計畫...")

        plan = context.get("plan")
        target = context.get("target")

        if not plan or not target:
            return {"success": False, "error": "Missing plan or target"}

        try:
            # 導入 AttackExecutor
            from services.core.aiva_core.core_capabilities.attack.attack_executor import (
                AttackExecutor,
                ExecutionMode,
            )

            # 解析執行模式
            mode_str = context.get("mode", "testing").lower()
            mode_map = {
                "safe": ExecutionMode.SAFE,
                "testing": ExecutionMode.TESTING,
                "aggressive": ExecutionMode.AGGRESSIVE,
            }
            mode = mode_map.get(mode_str, ExecutionMode.TESTING)

            # 初始化執行器
            executor = AttackExecutor(
                mode=mode,
                max_concurrent=context.get("max_concurrent", 5),
                timeout=context.get("timeout", 300),
                safety_enabled=context.get("safety_enabled", True),
            )

            # 獲取 AI 分析結果
            ai_analysis = context.get("ai_analysis")

            # 執行攻擊計畫
            logger.info(f"   🎯 執行模式: {mode.value}")
            logger.info(f"   🎯 安全檢查: {'啟用' if executor.safety_enabled else '禁用'}")

            execution_result = await executor.execute_plan_with_ai_analysis(
                plan=plan,
                target=target,
                ai_analysis_results=ai_analysis,
            )

            # 解析結果
            if hasattr(execution_result, "success"):
                success = execution_result.success
                steps_completed = execution_result.steps_completed
                steps_failed = execution_result.steps_failed
            else:
                success = execution_result.get("success", False)
                steps_completed = execution_result.get("steps_completed", 0)
                steps_failed = execution_result.get("steps_failed", 0)

            logger.info(
                f"✅ AI 控制: 攻擊執行完成 - 成功: {success}, "
                f"完成步驟: {steps_completed}, 失敗步驟: {steps_failed}"
            )

            return {
                "success": success,
                "mode": mode.value,
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "execution_result": execution_result
                if isinstance(execution_result, dict)
                else execution_result.__dict__,
            }

        except Exception as e:
            logger.error(f"❌ AI 控制: 攻擊執行失敗 - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def _execute_two_phase_scan(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行兩階段掃描（調用 TwoPhaseScanOrchestrator）

        Args:
            context: 掃描上下文 {
                "targets": list[str],  # 目標列表
                "trace_id": str,  # 追蹤 ID
                "max_depth": int,  # Phase1 最大深度
                "max_urls": int,  # Phase1 最大 URL 數量
                "broker": AbstractBroker,  # RabbitMQ Broker (必需)
            }

        Returns:
            掃描結果
        """
        logger.info("🔍 AI 控制: 執行兩階段掃描...")

        targets = context.get("targets", [])
        if not targets:
            return {"success": False, "error": "No targets specified"}

        broker = context.get("broker")
        if not broker:
            return {
                "success": False,
                "error": "RabbitMQ broker is required for two-phase scan",
            }

        try:
            # 導入 TwoPhaseScanOrchestrator
            from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import (
                TwoPhaseScanOrchestrator,
            )

            # 初始化編排器
            orchestrator = TwoPhaseScanOrchestrator(broker=broker)

            trace_id = context.get("trace_id", f"ai_scan_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            max_depth = context.get("max_depth", 3)
            max_urls = context.get("max_urls", 1000)

            logger.info(f"   🎯 目標數量: {len(targets)}")
            logger.info(f"   🎯 最大深度: {max_depth}")
            logger.info(f"   🎯 最大 URL: {max_urls}")

            # 執行兩階段掃描
            result = await orchestrator.execute_two_phase_scan(
                targets=targets,
                trace_id=trace_id,
                max_depth=max_depth,
                max_urls=max_urls,
            )

            # 解析結果
            urls_found = result.summary.urls_found if hasattr(result, "summary") else 0
            assets_found = len(result.assets) if hasattr(result, "assets") else 0
            execution_time = (
                result.execution_time if hasattr(result, "execution_time") else 0
            )

            logger.info(
                f"✅ AI 控制: 兩階段掃描完成 - 發現 {urls_found} 個 URL, "
                f"{assets_found} 個資產, 耗時 {execution_time:.2f}s"
            )

            return {
                "success": True,
                "trace_id": trace_id,
                "targets_scanned": len(targets),
                "urls_found": urls_found,
                "assets_found": assets_found,
                "execution_time": execution_time,
                "phase0_summary": result.phase0_summary
                if hasattr(result, "phase0_summary")
                else {},
                "full_result": result.__dict__ if hasattr(result, "__dict__") else result,
            }

        except Exception as e:
            logger.error(f"❌ AI 控制: 兩階段掃描失敗 - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def run_training_session(
        self,
        scenario_ids: list[str] | None = None,
        episodes_per_scenario: int = 10,
    ) -> dict[str, Any]:
        """運行訓練會話（已廢棄 - 由 UnifiedExecutor 自動管理）

        Args:
            scenario_ids: 場景 ID 列表
            episodes_per_scenario: 每個場景的回合數

        Returns:
            訓練結果
        """
        logger.warning("⚠️ run_training_session is deprecated. Training is now automatic in UnifiedExecutor.")
        
        # 返回當前學習狀態
        status = self.unified_executor.get_learning_status()
        
        return {
            "success": True,
            "message": "Training is now automatic. Current learning status:",
            "learning_status": status
        }
    
    async def process_scan_command(
        self,
        user_input: str
    ) -> dict[str, Any]:
        """處理用戶掃描命令 - 完整的指揮鏈流程
        
        流程: 用戶輸入 → TaskContext → AI 決策 → Orchestrator 執行
        
        Args:
            user_input: 用戶自然語言輸入，如 "快速掃描 192.168.1.1"
        
        Returns:
            掃描結果
        """
        try:
            # 步驟 1: 解析用戶輸入為標準任務參數包
            from ..core_capabilities.task_context import parse_user_input_to_context
            scan_context = parse_user_input_to_context(user_input)
            
            logger.info(f"📋 已解析任務參數: target={scan_context.target}, "
                       f"intent={scan_context.intent}")
            
            # 步驟 2: AI 決策 - 選擇掃描工具和參數
            from ..cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
            decision_agent = EnhancedDecisionAgent()
            ai_decision = decision_agent.decide_scan_strategy(scan_context)
            
            # 將 AI 決策結果填充到 context 中
            scan_context.ai_decision.selected_tool = ai_decision["selected_tool"]
            scan_context.ai_decision.confidence_score = ai_decision["confidence"]
            scan_context.ai_decision.reasoning = ai_decision["reasoning"]
            
            logger.info(f"🎯 AI 決策: {ai_decision['selected_tool']} "
                       f"(信心度 {ai_decision['confidence']:.2f})")
            
            # 步驟 3: 傳給 Orchestrator 執行
            from ..core_capabilities.orchestration.two_phase_scan_orchestrator import TwoPhaseScanOrchestrator
            orchestrator = TwoPhaseScanOrchestrator(broker=self.broker)
            result = await orchestrator.execute_scan_with_context(scan_context)
            
            logger.info(f"✅ 掃描完成: {result.total_found if hasattr(result, 'total_found') else 'N/A'} 個發現")
            
            return {
                "status": "success",
                "task_id": scan_context.task_id,
                "ai_decision": ai_decision,
                "scan_result": result.to_dict() if hasattr(result, 'to_dict') else str(result)
            }
        
        except Exception as e:
            logger.error(f"❌ 掃描命令處理失敗: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def unified_attack(
        self,
        target: str,
        objective: str,
        user_input: Optional[str] = None
    ) -> dict[str, Any]:
        """統一攻擊執行接口 - 簡化版
        
        使用統一執行器執行攻擊，自動學習。
        
        Args:
            target: 目標 URL/IP
            objective: 攻擊目標描述
            user_input: 原始用戶輸入（可選）
        
        Returns:
            執行結果
        """
        logger.info(f"🚀 Unified Attack: {target} - {objective}")
        
        try:
            # 使用統一執行器
            result = await self.unified_executor.execute(
                target=target,
                objective=objective,
                scenario=None,  # 實戰模式
                constraints=None
            )
            
            # 格式化返回結果
            return {
                "success": result.success,
                "vulnerabilities": result.vulnerabilities,
                "learning_info": result.learning_info,
                "execution_details": result.execution_details
            }
        except Exception as e:
            logger.error(f"Unified attack failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "vulnerabilities": []
            }
    
    def get_learning_status(self) -> dict[str, Any]:
        """獲取學習狀態"""
        try:
            return self.unified_executor.get_learning_status()
        except Exception as e:
            logger.error(f"Failed to get learning status: {e}")
            return {"error": str(e)}
    
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
            "learning_status": self.get_learning_status(),
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

        # 注意：UnifiedExecutor 會自動管理學習狀態
        # 經驗已通過 experience_manager 保存

        logger.info("✅ AI Commander state saved")
