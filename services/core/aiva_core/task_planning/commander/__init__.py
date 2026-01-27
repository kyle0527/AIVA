"""AI Commander 子模組

將 AICommander 拆解為以下子模組：
- types: 枚舉類型定義 (AITaskType, AIComponent)
- capability_manager: 能力選單管理
- plan_builder: 攻擊計劃建構
- strategy_engine: 策略決策引擎
- attack_coordinator: 攻擊執行協調
- learning_adapter: 學習系統適配

外部使用方式：
    from .commander import CommanderCoordinator, AITaskType
    coordinator = CommanderCoordinator(data_directory=...)
    result = await coordinator.execute_command(task_type=AITaskType.ATTACK_PLANNING, context={...})
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .types import AITaskType, AIComponent
from .capability_manager import CapabilityManager
from .plan_builder import PlanBuilder
from .strategy_engine import StrategyEngine
from .attack_coordinator import AttackCoordinator
from .learning_adapter import LearningAdapter

logger = logging.getLogger(__name__)


class CommanderCoordinator:
    """AI 指揮官協調器
    
    協調所有子模組，提供統一的 execute_command 介面。
    這是 AICommander 的完全替代品（方案 B）。
    """
    
    def __init__(
        self,
        data_directory: Optional[Path] = None,
        learning_enabled: bool = True,
    ):
        """初始化協調器
        
        Args:
            data_directory: 數據目錄
            learning_enabled: 是否啟用學習
        """
        self.data_directory = data_directory or Path("./data/commander")
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.learning_enabled = learning_enabled
        
        # 延遲初始化子模組
        self._capability_manager: Optional[CapabilityManager] = None
        self._plan_builder: Optional[PlanBuilder] = None
        self._strategy_engine: Optional[StrategyEngine] = None
        self._attack_coordinator: Optional[AttackCoordinator] = None
        self._learning_adapter: Optional[LearningAdapter] = None
        
        logger.info(f"✅ Commander Coordinator initialized")
    
    @property
    def capability_manager(self) -> CapabilityManager:
        """延遲加載能力管理器"""
        if self._capability_manager is None:
            self._capability_manager = CapabilityManager()
        return self._capability_manager
    
    @property
    def plan_builder(self) -> PlanBuilder:
        """延遲加載計劃建構器"""
        if self._plan_builder is None:
            self._plan_builder = PlanBuilder(data_directory=self.data_directory / "plans")
        return self._plan_builder
    
    @property
    def strategy_engine(self) -> StrategyEngine:
        """延遲加載策略引擎"""
        if self._strategy_engine is None:
            self._strategy_engine = StrategyEngine(data_directory=self.data_directory / "strategy")
        return self._strategy_engine
    
    @property
    def attack_coordinator(self) -> AttackCoordinator:
        """延遲加載攻擊協調器"""
        if self._attack_coordinator is None:
            # 使用 CLI 執行架構：傳入 data_directory 和 dispatcher
            from ..dispatcher import TaskDispatcher
            dispatcher = TaskDispatcher(source_module="commander")
            self._attack_coordinator = AttackCoordinator(
                data_directory=self.data_directory / "attacks",
                dispatcher=dispatcher
            )
        return self._attack_coordinator
    
    @property
    def learning_adapter(self) -> LearningAdapter:
        """延遲加載學習適配器"""
        if self._learning_adapter is None:
            self._learning_adapter = LearningAdapter(
                data_directory=self.data_directory / "learning",
                enabled=self.learning_enabled
            )
        return self._learning_adapter
    
    async def execute_command(
        self,
        task_type: AITaskType,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """執行 AI 指令
        
        統一入口，根據任務類型路由到對應的子模組。
        
        Args:
            task_type: 任務類型
            context: 任務上下文
            **kwargs: 額外參數
            
        Returns:
            執行結果字典
        """
        logger.info(f"📍 CommanderCoordinator.execute_command: {task_type}")
        
        try:
            # 決策類任務
            if task_type == AITaskType.ATTACK_PLANNING:
                return await self.plan_builder.build_attack_plan(context)
            
            elif task_type == AITaskType.STRATEGY_DECISION:
                return await self.strategy_engine.make_decision(context)
            
            elif task_type == AITaskType.RISK_ASSESSMENT:
                return await self.strategy_engine.assess_risk(context)
            
            # 執行類任務
            elif task_type == AITaskType.VULNERABILITY_DETECTION:
                return await self.attack_coordinator.detect_vulnerabilities(context)
            
            elif task_type == AITaskType.EXPLOIT_EXECUTION:
                return await self.attack_coordinator.execute_exploit(context)
            
            elif task_type == AITaskType.CODE_ANALYSIS:
                return await self.attack_coordinator.analyze_code(context)
            
            elif task_type == AITaskType.ATTACK_EXECUTION:
                return await self.attack_coordinator.execute_attack(context)
            
            elif task_type == AITaskType.TWO_PHASE_SCAN:
                return await self.attack_coordinator.execute_two_phase_scan(context)
            
            # 學習類任務
            elif task_type == AITaskType.EXPERIENCE_LEARNING:
                return await self.learning_adapter.learn_from_experience(context)
            
            elif task_type == AITaskType.MODEL_TRAINING:
                return await self.learning_adapter.train_model(context)
            
            elif task_type == AITaskType.KNOWLEDGE_RETRIEVAL:
                return await self.learning_adapter.retrieve_knowledge(context)
            
            # 能力查詢
            elif task_type == AITaskType.CAPABILITY_QUERY:
                return self.capability_manager.query_capabilities(context)
            
            # 協調類任務
            elif task_type == AITaskType.MULTI_LANG_COORDINATION:
                return await self.attack_coordinator.coordinate_multilang(context)
            
            elif task_type == AITaskType.TASK_DELEGATION:
                return await self.strategy_engine.delegate_task(context)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}",
                    "task_type": str(task_type)
                }
                
        except Exception as e:
            logger.error(f"❌ execute_command failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_type": str(task_type)
            }


# 為向後兼容，別名 AICommander 為 CommanderCoordinator
AICommander = CommanderCoordinator


__all__ = [
    # 主協調器（替代原 AICommander）
    "CommanderCoordinator",
    "AICommander",  # 向後兼容別名
    # 類型
    "AITaskType",
    "AIComponent",
    # 子模組
    "CapabilityManager",
    "PlanBuilder",
    "StrategyEngine",
    "AttackCoordinator",
    "LearningAdapter",
]
