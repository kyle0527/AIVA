"""
AIVA Core - 核心引擎模組

這是 AIVA 的核心處理引擎，基於六大模組架構設計：

六大模組架構 (v3.0):
1. 🧠 cognitive_core/      - AI 認知核心 (神經網路、RAG、決策、反幻覺)
2. 🧭 internal_exploration/ - 對內探索 (自我認知、能力分析)
3. 📋 task_planning/        - 任務規劃與執行 (規劃器、執行器、指揮官)
4. 🌍 external_learning/    - 對外學習 (分析、追蹤、訓練、模型)
5. 🎯 core_capabilities/    - 核心能力 (攻擊鏈、業務邏輯、對話、插件)
6. 🏗️ service_backbone/     - 服務骨幹 (API、協調、消息、存儲、狀態)

UI 層:
- 🎨 ui_panel/             - 使用者介面 (控制台、儀表板、CLI)

整合增強功能:
- migration_controller: Strangler Fig 遷移控制器
- plugins: 增強插件系統，整合能力註冊和智能編排
"""

__version__ = "3.0.0-alpha"

import logging
from typing import Any, Dict, Optional, Set, List
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== Strangler Fig 遷移控制器 ====================

class MigrationPhase(Enum):
    """遷移階段定義"""
    LEGACY = "legacy"           # 純舊系統
    TRANSITION = "transition"   # 過渡期 - 雙重運行
    MODERN = "modern"           # 新系統主導
    COMPLETE = "complete"       # 遷移完成

class FeatureFlag(Enum):
    """功能開關定義"""
    V1_CAPABILITY_REGISTRY = "v1_capability_registry"
    AI_MODULE_ORCHESTRATION = "ai_module_orchestration"
    ENHANCED_MESSAGE_BROKER = "enhanced_message_broker"
    RISK_CONTROL_SYSTEM = "risk_control_system"
    TOPOLOGICAL_SORTING = "topological_sorting"

class StranglerFigMigrationController:
    """Strangler Fig 模式遷移控制器 - 整合自 AI 模組"""
    
    def __init__(self):
        self.current_phase = MigrationPhase.TRANSITION
        self.feature_flags: Dict[FeatureFlag, bool] = {
            FeatureFlag.V1_CAPABILITY_REGISTRY: True,
            FeatureFlag.AI_MODULE_ORCHESTRATION: True,
            FeatureFlag.ENHANCED_MESSAGE_BROKER: True,
            FeatureFlag.RISK_CONTROL_SYSTEM: True,
            FeatureFlag.TOPOLOGICAL_SORTING: True,
        }
        
        # 遷移統計
        self.migration_stats = {
            'features_migrated': 0,
            'features_in_transition': 0,
            'legacy_calls': 0,
            'modern_calls': 0,
            'migration_started': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }
        
        # 路由表 - 決定使用新舊系統
        self.routing_rules: Dict[str, Dict[str, Any]] = {
            'capability_registry': {
                'legacy_path': 'aiva_common.plugins',
                'modern_path': 'aiva_core.plugins.ai_summary_plugin.global_capability_registry',
                'feature_flag': FeatureFlag.V1_CAPABILITY_REGISTRY,
                'fallback_strategy': 'legacy_first'
            },
            'message_broker': {
                'legacy_path': 'aiva_core.messaging.message_router',
                'modern_path': 'aiva_core.messaging.message_broker.enhanced_broker',
                'feature_flag': FeatureFlag.ENHANCED_MESSAGE_BROKER,
                'fallback_strategy': 'modern_first'
            },
            'risk_control': {
                'legacy_path': 'aiva_core.authz.base_authz',
                'modern_path': 'aiva_core.authz.permission_matrix.RiskGuard',
                'feature_flag': FeatureFlag.RISK_CONTROL_SYSTEM,
                'fallback_strategy': 'modern_first'
            }
        }
        
        logger.info(f"🔄 Strangler Fig 遷移控制器啟動 - 當前階段: {self.current_phase.value}")
    
    def route_request(self, service_name: str, operation: str, **kwargs) -> Any:
        """智能路由請求到新舊系統"""
        
        if service_name not in self.routing_rules:
            logger.warning(f"⚠️ 未知服務: {service_name}")
            return None
            
        rule = self.routing_rules[service_name]
        feature_flag = rule['feature_flag']
        
        # 檢查功能開關
        use_modern = self.feature_flags.get(feature_flag, False)
        
        try:
            if use_modern:
                # 嘗試使用新系統
                result = self._call_modern_system(rule['modern_path'], operation, **kwargs)
                self.migration_stats['modern_calls'] += 1
                return result
            else:
                # 使用舊系統
                result = self._call_legacy_system(rule['legacy_path'], operation, **kwargs)
                self.migration_stats['legacy_calls'] += 1
                return result
                
        except Exception as e:
            logger.error(f"❌ {service_name} 調用失敗: {e}")
            
            # 降級策略
            if rule['fallback_strategy'] == 'legacy_first':
                return self._call_legacy_system(rule['legacy_path'], operation, **kwargs)
            else:
                return self._call_modern_system(rule['modern_path'], operation, **kwargs)
    
    def _call_modern_system(self, modern_path: str, operation: str, **kwargs) -> Any:
        """調用新系統"""
        # 這裡實際上會動態導入和調用現代化的系統
        logger.info(f"🚀 調用新系統: {modern_path}.{operation}")
        return {"status": "modern_system", "path": modern_path, "operation": operation}
    
    def _call_legacy_system(self, legacy_path: str, operation: str, **kwargs) -> Any:
        """調用舊系統"""
        logger.info(f"🏛️ 調用舊系統: {legacy_path}.{operation}")
        return {"status": "legacy_system", "path": legacy_path, "operation": operation}
    
    def enable_feature(self, feature: FeatureFlag) -> None:
        """啟用功能"""
        self.feature_flags[feature] = True
        self._update_migration_stats()
        logger.info(f"✅ 功能已啟用: {feature.value}")
    
    def disable_feature(self, feature: FeatureFlag) -> None:
        """禁用功能"""
        self.feature_flags[feature] = False
        self._update_migration_stats()
        logger.info(f"❌ 功能已禁用: {feature.value}")
    
    def advance_migration_phase(self) -> bool:
        """推進遷移階段"""
        phase_order = [MigrationPhase.LEGACY, MigrationPhase.TRANSITION, 
                      MigrationPhase.MODERN, MigrationPhase.COMPLETE]
        
        current_index = phase_order.index(self.current_phase)
        
        if current_index < len(phase_order) - 1:
            self.current_phase = phase_order[current_index + 1]
            self._update_migration_stats()
            logger.info(f"🔄 遷移階段推進到: {self.current_phase.value}")
            return True
        
        logger.info("✅ 遷移已完成")
        return False
    
    def _update_migration_stats(self) -> None:
        """更新遷移統計"""
        self.migration_stats['features_migrated'] = sum(self.feature_flags.values())
        self.migration_stats['features_in_transition'] = len(self.feature_flags) - self.migration_stats['features_migrated']
        self.migration_stats['last_update'] = datetime.now().isoformat()
    
    def get_migration_status(self) -> Dict[str, Any]:
        """獲取遷移狀態"""
        return {
            'current_phase': self.current_phase.value,
            'feature_flags': {flag.value: enabled for flag, enabled in self.feature_flags.items()},
            'stats': self.migration_stats,
            'routing_rules': {name: {
                'legacy_path': rule['legacy_path'],
                'modern_path': rule['modern_path'],
                'feature_enabled': self.feature_flags[rule['feature_flag']],
                'fallback_strategy': rule['fallback_strategy']
            } for name, rule in self.routing_rules.items()}
        }

# 全域遷移控制器實例
migration_controller = StranglerFigMigrationController()


# ==================== 便利函數 ====================

def route_to_system(service: str, operation: str, **kwargs) -> Any:
    """便利函數 - 智能路由到新舊系統"""
    return migration_controller.route_request(service, operation, **kwargs)

def is_feature_enabled(feature: FeatureFlag) -> bool:
    """檢查功能是否啟用"""
    return migration_controller.feature_flags.get(feature, False)

def get_migration_phase() -> MigrationPhase:
    """獲取當前遷移階段"""
    return migration_controller.current_phase


# ==================== 原有導入保持不變 ====================

# 從 aiva_common 導入共享基礎設施 (修復: 移除錯誤的 services. 前綴)
from aiva_common.enums import (
    ComplianceFramework,
    Confidence,
    ModuleName,
    RemediationStatus,
    RemediationType,
    RiskLevel,
    Severity,
    TaskStatus,
    Topic,
)
from aiva_common.schemas import CVEReference, CVSSv3Metrics, CWEReference

# 從 core.ai_models 導入 AI 系統相關模型 (修復: ai_models 在 services/core/ 目錄)
# 使用更智能的條件導入處理相對導入問題
try:
    # 首先嘗試相對導入 (當作為正確包導入時)
    from ..ai_models import (
        AIExperienceCreatedEvent,
        AIModelDeployCommand,
        AIModelUpdatedEvent,
        AITraceCompletedEvent,
        AITrainingCompletedPayload,
        AITrainingProgressPayload,
        AITrainingStartPayload,
        AIVACommand,
        AIVAEvent,
        AIVARequest,
        AIVAResponse,
        AIVerificationRequest,
        AIVerificationResult,
        AttackPlan,
        AttackStep,
        ExperienceSample,
        ModelTrainingConfig,
        ModelTrainingResult,
        PlanExecutionMetrics,
        PlanExecutionResult,
        RAGKnowledgeUpdatePayload,
        RAGQueryPayload,
        RAGResponsePayload,
        ScenarioTestResult,
        SessionState,
        StandardScenario,
        TraceRecord,
    )
except (ImportError, ValueError):
    # 如果相對導入失敗，使用絕對導入
    import sys
    import os
    # 添加項目根目錄到 sys.path
    _current_path = os.path.dirname(os.path.abspath(__file__))  # aiva_core 目錄
    _services_core_path = os.path.dirname(_current_path)  # services/core 目錄
    _services_path = os.path.dirname(_services_core_path)  # services 目錄  
    _project_root = os.path.dirname(_services_path)  # 項目根目錄
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    
    from services.core.ai_models import (
        AIExperienceCreatedEvent,
        AIModelDeployCommand,
        AIModelUpdatedEvent,
        AITraceCompletedEvent,
        AITrainingCompletedPayload,
        AITrainingProgressPayload,
        AITrainingStartPayload,
        AIVACommand,
        AIVAEvent,
        AIVARequest,
        AIVAResponse,
        AIVerificationRequest,
        AIVerificationResult,
        AttackPlan,
        AttackStep,
        ExperienceSample,
        ModelTrainingConfig,
        ModelTrainingResult,
        PlanExecutionMetrics,
        PlanExecutionResult,
        RAGKnowledgeUpdatePayload,
        RAGQueryPayload,
        RAGResponsePayload,
        ScenarioTestResult,
        SessionState,
        StandardScenario,
        TraceRecord,
    )

# 從 core.models 導入核心業務邏輯模型
# 從 aiva_common.schemas 導入共享標準模型
from aiva_common.schemas import (
    ConfigUpdatePayload,
    FeedbackEventPayload,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    HeartbeatPayload,
    ModuleStatus,
    RemediationGeneratePayload,
    RemediationResultPayload,
    Target,
    TaskUpdatePayload,
)

# 從 core.models 導入核心擴展模型（conftest.py 已將 services/core/ 加入 Python 路徑）
# 使用智能條件導入處理相對導入問題
try:
    # 首先嘗試相對導入
    from ..models import (
        AttackPathEdge,
        AttackPathNode,
        AttackPathPayload,
        AttackPathRecommendation,
        CodeLevelRootCause,
        EnhancedAttackPath,
        EnhancedAttackPathNode,
        EnhancedFindingPayload,
        EnhancedModuleStatus,
        EnhancedRiskAssessment,
        EnhancedTaskExecution,
        EnhancedVulnerability,
        EnhancedVulnerabilityCorrelation,
        RiskAssessmentContext,
        RiskAssessmentResult,
        RiskFactor,
        RiskTrendAnalysis,
        SystemOrchestration,
        TaskDependency,
        TaskQueue,
        TestStrategy,
        VulnerabilityCorrelation,
    )
except (ImportError, ValueError):
    # 如果相對導入失敗，使用絕對導入
    from services.core.models import (
        AttackPathEdge,
        AttackPathNode,
        AttackPathPayload,
        AttackPathRecommendation,
        CodeLevelRootCause,
        EnhancedAttackPath,
        EnhancedAttackPathNode,
        EnhancedFindingPayload,
        EnhancedModuleStatus,
        EnhancedRiskAssessment,
        EnhancedTaskExecution,
        EnhancedVulnerability,
        EnhancedVulnerabilityCorrelation,
        RiskAssessmentContext,
        RiskAssessmentResult,
        RiskFactor,
        RiskTrendAnalysis,
        SystemOrchestration,
        TaskDependency,
        TaskQueue,
        TestStrategy,
        VulnerabilityCorrelation,
    )

# 從新遷移的核心服務組件導入 (從 aiva_core_v2 遷移而來)
from .task_planning.command_router import (
    CommandContext,
    CommandRouter,
    CommandType,
    ExecutionMode,
    ExecutionResult,
    get_command_router,
)
from .task_planning.context_manager import ContextManager, get_context_manager
from .core_service_coordinator import (
    AIVACoreServiceCoordinator,
    get_core_service_coordinator,
    initialize_core_module,
    process_command,
    shutdown_core_module,
)
from .decision.skill_graph import AIVASkillGraph, skill_graph

# 從新增核心組件導入 (附件要求實現)
from .dialog.assistant import AIVADialogAssistant, dialog_assistant
from .task_planning.execution_planner import ExecutionPlanner, get_execution_planner

# capability_evaluator 現在使用 aiva_common.ai.capability_evaluator 統一實現






__all__ = [
    # 遷移控制器組件 (整合增強功能)
    "StranglerFigMigrationController",
    "migration_controller",
    "MigrationPhase",
    "FeatureFlag",
    "route_to_system",
    "is_feature_enabled",
    "get_migration_phase",
    # 新增核心組件 (附件要求實現)
    "AIVADialogAssistant",
    "dialog_assistant",
    "AIVASkillGraph",
    "skill_graph",
    # capability_evaluator 已移至 aiva_common.ai
    # 從 aiva_core_v2 遷移的核心服務組件
    "CommandRouter",
    "get_command_router",
    "CommandType",
    "ExecutionMode",
    "CommandContext",
    "ExecutionResult",
    "ContextManager",
    "get_context_manager",
    "ExecutionPlanner",
    "get_execution_planner",
    "AIVACoreServiceCoordinator",
    "get_core_service_coordinator",
    "process_command",
    "initialize_core_module",
    "shutdown_core_module",
    # 來自 aiva_common
    "CVEReference",
    "CVSSv3Metrics",
    "CWEReference",
    "ComplianceFramework",
    "Confidence",
    "ModuleName",
    "RemediationStatus",
    "RemediationType",
    "RiskLevel",
    "Severity",
    "TaskStatus",
    "Topic",
    # AI 系統模型
    "AIExperienceCreatedEvent",
    "AIModelDeployCommand",
    "AIModelUpdatedEvent",
    "AITraceCompletedEvent",
    "AITrainingCompletedPayload",
    "AITrainingProgressPayload",
    "AITrainingStartPayload",
    "AIVACommand",
    "AIVAEvent",
    "AIVARequest",
    "AIVAResponse",
    "AIVerificationRequest",
    "AIVerificationResult",
    "AttackPlan",
    "AttackStep",
    "ExperienceSample",
    "ModelTrainingConfig",
    "ModelTrainingResult",
    "PlanExecutionMetrics",
    "PlanExecutionResult",
    "RAGKnowledgeUpdatePayload",
    "RAGQueryPayload",
    "RAGResponsePayload",
    "ScenarioTestResult",
    "SessionState",
    "StandardScenario",
    "TraceRecord",
    # 核心業務邏輯模型
    "AttackPathEdge",
    "AttackPathNode",
    "AttackPathPayload",
    "AttackPathRecommendation",
    "CodeLevelRootCause",
    "ConfigUpdatePayload",
    "EnhancedAttackPath",
    "EnhancedAttackPathNode",
    "EnhancedFindingPayload",
    "EnhancedModuleStatus",
    "EnhancedRiskAssessment",
    "EnhancedTaskExecution",
    "EnhancedVulnerability",
    "EnhancedVulnerabilityCorrelation",
    "FeedbackEventPayload",
    "FindingEvidence",
    "FindingImpact",
    "FindingPayload",
    "FindingRecommendation",
    "HeartbeatPayload",
    "ModuleStatus",
    "RemediationGeneratePayload",
    "RemediationResultPayload",
    "RiskAssessmentContext",
    "RiskAssessmentResult",
    "RiskFactor",
    "RiskTrendAnalysis",

    "SystemOrchestration",
    "Target",
    "TaskDependency",
    "TaskQueue",
    "TaskUpdatePayload",
    "TestStrategy",
    "VulnerabilityCorrelation",
]
