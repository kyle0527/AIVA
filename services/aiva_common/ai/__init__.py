"""
AIVA Common AI Module - AI 共享組件模組 (可插拔設計)

此模組提供 AIVA 五大模組架構中 AI 相關的共享組件和工具類，
基於 2024-2025 年 Python 最佳實踐實現，支援可插拔架構。

五大模組架構支援:
├── Integration Layer (整合層)
│   └── AI 整合介面和適配器
├── Core Layer (核心層) 
│   └── AI 核心邏輯和狀態管理
├── Scan Features (掃描功能層)
│   └── AI 增強的掃描能力
├── Common (共享層)
│   └── AI 通用組件和資料模型
└── Services (服務層)
    └── AI 微服務和 API 介面

核心 AI 組件 (可插拔):
- 對話助手介面 (IDialogAssistant) - 可替換的對話實現
- 計劃執行器介面 (IPlanExecutor) - 可替換的執行引擎  
- 經驗管理器介面 (IExperienceManager) - 可替換的學習機制
- 能力評估器介面 (ICapabilityEvaluator) - 可替換的評估策略
- 跨語言橋接器介面 (ICrossLanguageBridge) - 可替換的語言支援
- RAG 檢索代理介面 (IRAGAgent) - 可替換的知識檢索
- 技能圖分析器介面 (ISkillGraphAnalyzer) - 可替換的圖分析

設計原則:
- 介面分離原則 (Interface Segregation)
- 依賴倒置原則 (Dependency Inversion) 
- 開放封閉原則 (Open/Closed)
- 插件架構模式 (Plugin Architecture)
- 策略模式 (Strategy Pattern)

符合標準:
- AIVA Common 設計規範
- 五大模組架構相容性
- Pydantic v2 數據模型
- 異步編程模式
- 現代化 Python 架構
"""



import contextlib
from typing import TYPE_CHECKING

# 版本信息
__version__ = "1.0.0"
__author__ = "AIVA Team"
__description__ = "AIVA Common AI Module - AI 共享組件模組"

# 條件導入核心組件
_has_dialog = False
_has_plan_executor = False
_has_experience_manager = False
_has_capability_evaluator = False
_has_cross_language_bridge = False
_has_rag_agent = False
_has_skill_graph = False

# 嘗試導入對話助手
with contextlib.suppress(ImportError):
    from .dialog_assistant import AIVADialogAssistant, DialogIntent
    _has_dialog = True

# 嘗試導入計劃執行器
with contextlib.suppress(ImportError):
    from .plan_executor import AIVAPlanExecutor, ExecutionConfig
    _has_plan_executor = True

# 嘗試導入經驗管理器
with contextlib.suppress(ImportError):
    from .experience_manager import AIVAExperienceManager, LearningSession, create_experience_manager
    _has_experience_manager = True

# 嘗試導入能力評估器
with contextlib.suppress(ImportError):
    from .capability_evaluator import AIVACapabilityEvaluator, CapabilityEvidence, create_capability_evaluator
    _has_capability_evaluator = True

# 嘗試導入跨語言橋接器
with contextlib.suppress(ImportError):
    from .cross_language_bridge import AIVACrossLanguageBridge, BridgeConfig
    _has_cross_language_bridge = True

# 嘗試導入 RAG 代理
with contextlib.suppress(ImportError):
    from .rag_agent import BioNeuronRAGAgent, RAGConfig
    _has_rag_agent = True

# 嘗試導入技能圖分析器
with contextlib.suppress(ImportError):
    from .skill_graph_analyzer import AIVASkillGraphAnalyzer, SkillNode
    _has_skill_graph = True

# 導入介面和註冊表 (始終可用)
from .interfaces import (
    IDialogAssistant,
    IPlanExecutor,
    IExperienceManager,
    ICapabilityEvaluator,
    ICrossLanguageBridge,
    IRAGAgent,
    ISkillGraphAnalyzer,
    IAIComponentFactory,
    IAIComponentRegistry,
    IAIContext,
)

from .registry import (
    AIVAComponentRegistry,
    AIVAComponentFactory,
    AIVAContext,
    get_global_registry,
    set_global_registry,
    aiva_ai_context,
    register_builtin_components,
)

# 基礎 __all__ 列表
__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    # 介面定義
    "IDialogAssistant",
    "IPlanExecutor", 
    "IExperienceManager",
    "ICapabilityEvaluator",
    "ICrossLanguageBridge",
    "IRAGAgent",
    "ISkillGraphAnalyzer",
    "IAIComponentFactory",
    "IAIComponentRegistry",
    "IAIContext",
    # 註冊表實現
    "AIVAComponentRegistry",
    "AIVAComponentFactory",
    "AIVAContext",
    "get_global_registry",
    "set_global_registry",
    "aiva_ai_context",
    "register_builtin_components",
]

# 動態添加可用組件到 __all__
if _has_dialog:
    __all__.extend(["AIVADialogAssistant", "DialogIntent"])

if _has_plan_executor:
    __all__.extend(["AIVAPlanExecutor", "ExecutionConfig"])

if _has_experience_manager:
    __all__.extend([
        "AIVAExperienceManager", "LearningSession", 
        "create_default_experience_manager", "get_default_experience_manager"
    ])

if _has_capability_evaluator:
    __all__.extend([
        "AIVACapabilityEvaluator", "CapabilityEvidence",
        "create_default_capability_evaluator", "get_default_capability_evaluator"
    ])

if _has_cross_language_bridge:
    __all__.extend(["AIVACrossLanguageBridge", "BridgeConfig"])

if _has_rag_agent:
    __all__.extend(["BioNeuronRAGAgent", "RAGConfig"])

if _has_skill_graph:
    __all__.extend(["AIVASkillGraphAnalyzer", "SkillNode"])

# 輔助函數
def get_available_components() -> dict[str, bool]:
    """獲取可用組件狀態
    
    Returns:
        組件可用性字典
    """
    return {
        "dialog_assistant": _has_dialog,
        "plan_executor": _has_plan_executor,
        "experience_manager": _has_experience_manager,
        "capability_evaluator": _has_capability_evaluator,
        "cross_language_bridge": _has_cross_language_bridge,
        "rag_agent": _has_rag_agent,
        "skill_graph_analyzer": _has_skill_graph,
    }

def get_ai_module_info() -> dict[str, str]:
    """獲取 AI 模組信息
    
    Returns:
        模組信息字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_components": str(sum(get_available_components().values())),
        "total_components": "7",
    }

# 工廠函數 (便利創建函數)
def create_default_capability_evaluator(config=None):
    """創建預設能力評估器實例 (工廠函數)"""
    if _has_capability_evaluator:
        from .capability_evaluator import AIVACapabilityEvaluator
        if config:
            return AIVACapabilityEvaluator(config=config)
        return AIVACapabilityEvaluator()
    raise ImportError("CapabilityEvaluator not available")

def create_default_experience_manager(config=None):
    """創建預設經驗管理器實例 (工廠函數)"""
    if _has_experience_manager:
        from .experience_manager import AIVAExperienceManager
        if config:
            return AIVAExperienceManager(config=config)
        return AIVAExperienceManager()
    raise ImportError("ExperienceManager not available")

def create_default_dialog_assistant(config=None):
    """創建預設對話助手實例 (工廠函數)"""
    if _has_dialog:
        from .dialog_assistant import AIVADialogAssistant
        if config:
            return AIVADialogAssistant(config=config)
        return AIVADialogAssistant()
    raise ImportError("DialogAssistant not available")

# 全域實例管理
_default_instances = {}

def get_default_capability_evaluator():
    """獲取預設的能力評估器實例 (單例)"""
    if "capability_evaluator" not in _default_instances:
        _default_instances["capability_evaluator"] = create_default_capability_evaluator()
    return _default_instances["capability_evaluator"]

def get_default_experience_manager():
    """獲取預設的經驗管理器實例 (單例)"""
    if "experience_manager" not in _default_instances:
        _default_instances["experience_manager"] = create_default_experience_manager()
    return _default_instances["experience_manager"]

# 類型檢查時的導入
if TYPE_CHECKING:
    # 用於類型提示的導入，不影響運行時
    from .dialog_assistant import AIVADialogAssistant, DialogIntent
    from .plan_executor import AIVAPlanExecutor, ExecutionConfig
    from .experience_manager import AIVAExperienceManager, LearningSession
    from .capability_evaluator import AIVACapabilityEvaluator, CapabilityEvidence
    from .cross_language_bridge import AIVACrossLanguageBridge, BridgeConfig
    from .rag_agent import BioNeuronRAGAgent, RAGConfig
    from .skill_graph_analyzer import AIVASkillGraphAnalyzer, SkillNode