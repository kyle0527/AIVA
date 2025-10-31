"""
AIVA AI 組件註冊表 - 可插拔 AI 架構的核心管理器

此文件實現 AI 組件的註冊、發現和管理功能，支援可插拔架構。
各個模組可以註冊自己的 AI 組件實現，系統會自動選擇最適合的實現。

設計特點:
- 支援動態組件註冊和替換
- 自動依賴解析和組件初始化
- 組件生命週期管理
- 配置驗證和預設值處理
- 線程安全的組件管理
"""

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from .interfaces import (
    IAIComponentFactory,
    IAIComponentRegistry,
    IAIContext,
    ICapabilityEvaluator,
    ICrossLanguageBridge,
    IDialogAssistant,
    IExperienceManager,
    IPlanExecutor,
    IRAGAgent,
    ISkillGraphAnalyzer,
)

logger = logging.getLogger(__name__)


class AIVAComponentRegistry(IAIComponentRegistry):
    """AIVA AI 組件註冊表實現"""

    def __init__(self):
        """初始化註冊表"""
        self._components: dict[str, dict[str, dict[str, Any]]] = {
            "dialog_assistant": {},
            "plan_executor": {},
            "experience_manager": {},
            "capability_evaluator": {},
            "cross_language_bridge": {},
            "rag_agent": {},
            "skill_graph_analyzer": {},
        }
        self._instances: dict[str, Any] = {}
        self._default_components: dict[str, str] = {}
        self._lock = asyncio.Lock()

        logger.info("AIVAComponentRegistry initialized")

    def register_component(
        self,
        component_type: str,
        component_name: str,
        component_class: type[Any],
        config: dict[str, Any] | None = None,
        is_default: bool = False,
    ) -> bool:
        """註冊 AI 組件

        Args:
            component_type: 組件類型
            component_name: 組件名稱
            component_class: 組件類別
            config: 預設配置
            is_default: 是否設為預設組件

        Returns:
            是否註冊成功
        """
        try:
            # 驗證組件類型
            if component_type not in self._components:
                logger.error(f"Unsupported component type: {component_type}")
                return False

            # 驗證組件類別實現了正確的介面
            if not self._validate_component_interface(component_type, component_class):
                logger.error(
                    f"Component {component_name} does not implement "
                    f"required interface for {component_type}"
                )
                return False

            # 註冊組件
            self._components[component_type][component_name] = {
                "class": component_class,
                "config": config or {},
                "registered_at": datetime.now(UTC).isoformat(),
                "is_default": is_default,
            }

            # 設置為預設組件
            if is_default or component_type not in self._default_components:
                self._default_components[component_type] = component_name

            logger.info(
                f"Registered {component_type} component: {component_name} "
                f"(default: {is_default})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register component {component_name}: {e}")
            return False

    def get_component(
        self,
        component_type: str,
        component_name: str | None = None,
        config_override: dict[str, Any] | None = None,
    ) -> Any | None:
        """獲取 AI 組件實例

        Args:
            component_type: 組件類型
            component_name: 組件名稱 (None 表示使用預設)
            config_override: 覆蓋配置

        Returns:
            組件實例
        """
        try:
            # 使用預設組件名稱
            if component_name is None:
                component_name = self._default_components.get(component_type)
                if not component_name:
                    logger.error(f"No default component for type: {component_type}")
                    return None

            # 檢查組件是否已註冊
            if (
                component_type not in self._components
                or component_name not in self._components[component_type]
            ):
                logger.error(f"Component not found: {component_type}.{component_name}")
                return None

            # 檢查是否已有實例 (單例模式)
            instance_key = f"{component_type}.{component_name}"
            if instance_key in self._instances:
                logger.debug(f"Returning existing instance: {instance_key}")
                return self._instances[instance_key]

            # 創建新實例
            component_info = self._components[component_type][component_name]
            component_class = component_info["class"]

            # 合併配置
            final_config = component_info["config"].copy()
            if config_override:
                final_config.update(config_override)

            # 實例化組件
            if final_config:
                instance = component_class(**final_config)
            else:
                instance = component_class()

            # 快取實例
            self._instances[instance_key] = instance

            logger.info(f"Created new instance: {instance_key}")
            return instance

        except Exception as e:
            logger.error(f"Failed to get component {component_name}: {e}")
            return None

    def list_components(
        self, component_type: str | None = None
    ) -> dict[str, list[str]]:
        """列出可用組件

        Args:
            component_type: 組件類型 (None 表示所有類型)

        Returns:
            組件類型到組件名稱列表的映射
        """
        if component_type:
            if component_type in self._components:
                return {component_type: list(self._components[component_type].keys())}
            else:
                return {}

        return {
            comp_type: list(comp_dict.keys())
            for comp_type, comp_dict in self._components.items()
        }

    def unregister_component(self, component_type: str, component_name: str) -> bool:
        """取消註冊組件

        Args:
            component_type: 組件類型
            component_name: 組件名稱

        Returns:
            是否取消成功
        """
        try:
            if (
                component_type not in self._components
                or component_name not in self._components[component_type]
            ):
                logger.warning(
                    f"Component not found for unregistration: "
                    f"{component_type}.{component_name}"
                )
                return False

            # 移除註冊信息
            del self._components[component_type][component_name]

            # 移除實例快取
            instance_key = f"{component_type}.{component_name}"
            if instance_key in self._instances:
                del self._instances[instance_key]

            # 如果是預設組件，需要重新設置預設
            if (
                component_type in self._default_components
                and self._default_components[component_type] == component_name
            ):
                remaining_components = list(self._components[component_type].keys())
                if remaining_components:
                    self._default_components[component_type] = remaining_components[0]
                else:
                    del self._default_components[component_type]

            logger.info(f"Unregistered component: {component_type}.{component_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister component {component_name}: {e}")
            return False

    def get_component_info(
        self, component_type: str, component_name: str
    ) -> dict[str, Any] | None:
        """獲取組件信息

        Args:
            component_type: 組件類型
            component_name: 組件名稱

        Returns:
            組件信息字典
        """
        if (
            component_type not in self._components
            or component_name not in self._components[component_type]
        ):
            return None

        info = self._components[component_type][component_name].copy()
        info["is_instantiated"] = (
            f"{component_type}.{component_name}" in self._instances
        )
        return info

    def get_registry_stats(self) -> dict[str, Any]:
        """獲取註冊表統計信息

        Returns:
            統計信息字典
        """
        stats = {
            "total_component_types": len(self._components),
            "total_registered_components": sum(
                len(comp_dict) for comp_dict in self._components.values()
            ),
            "total_instantiated_components": len(self._instances),
            "component_counts_by_type": {
                comp_type: len(comp_dict)
                for comp_type, comp_dict in self._components.items()
            },
            "default_components": self._default_components.copy(),
        }
        return stats

    def _validate_component_interface(
        self, component_type: str, component_class: type[Any]
    ) -> bool:
        """驗證組件是否實現了正確的介面

        Args:
            component_type: 組件類型
            component_class: 組件類別

        Returns:
            是否實現了正確的介面
        """
        interface_map = {
            "dialog_assistant": IDialogAssistant,
            "plan_executor": IPlanExecutor,
            "experience_manager": IExperienceManager,
            "capability_evaluator": ICapabilityEvaluator,
            "cross_language_bridge": ICrossLanguageBridge,
            "rag_agent": IRAGAgent,
            "skill_graph_analyzer": ISkillGraphAnalyzer,
        }

        expected_interface = interface_map.get(component_type)
        if not expected_interface:
            return False

        # 檢查是否繼承自正確的介面
        return issubclass(component_class, expected_interface)

    async def clear_instances(self, component_type: str | None = None) -> None:
        """清理組件實例快取

        Args:
            component_type: 組件類型 (None 表示清理所有)
        """
        async with self._lock:
            if component_type:
                # 清理特定類型的實例
                keys_to_remove = [
                    key
                    for key in self._instances.keys()
                    if key.startswith(f"{component_type}.")
                ]
                for key in keys_to_remove:
                    del self._instances[key]
                logger.info(
                    f"Cleared {len(keys_to_remove)} instances for {component_type}"
                )
            else:
                # 清理所有實例
                instance_count = len(self._instances)
                self._instances.clear()
                logger.info(f"Cleared all {instance_count} instances")


class AIVAComponentFactory(IAIComponentFactory):
    """AIVA AI 組件工廠實現"""

    def __init__(self, registry: AIVAComponentRegistry | None = None):
        """初始化工廠

        Args:
            registry: 組件註冊表 (None 表示使用全域註冊表)
        """
        self.registry = registry or get_global_registry()
        logger.info("AIVAComponentFactory initialized")

    def create_dialog_assistant(
        self, config: dict[str, Any] | None = None
    ) -> IDialogAssistant:
        """創建對話助手實例"""
        component = self.registry.get_component(
            "dialog_assistant", config_override=config
        )
        if component is None:
            raise RuntimeError("Failed to create dialog_assistant component")
        return component

    def create_plan_executor(
        self, config: dict[str, Any] | None = None
    ) -> IPlanExecutor:
        """創建計劃執行器實例"""
        component = self.registry.get_component("plan_executor", config_override=config)
        if component is None:
            raise RuntimeError("Failed to create plan_executor component")
        return component

    def create_experience_manager(
        self, config: dict[str, Any] | None = None
    ) -> IExperienceManager:
        """創建經驗管理器實例"""
        component = self.registry.get_component(
            "experience_manager", config_override=config
        )
        if component is None:
            raise RuntimeError("Failed to create experience_manager component")
        return component

    def create_capability_evaluator(
        self, config: dict[str, Any] | None = None
    ) -> ICapabilityEvaluator:
        """創建能力評估器實例"""
        component = self.registry.get_component(
            "capability_evaluator", config_override=config
        )
        if component is None:
            raise RuntimeError("Failed to create capability_evaluator component")
        return component

    def create_cross_language_bridge(
        self, config: dict[str, Any] | None = None
    ) -> ICrossLanguageBridge:
        """創建跨語言橋接器實例"""
        component = self.registry.get_component(
            "cross_language_bridge", config_override=config
        )
        if component is None:
            raise RuntimeError("Failed to create cross_language_bridge component")
        return component

    def create_rag_agent(self, config: dict[str, Any] | None = None) -> IRAGAgent:
        """創建 RAG 代理實例"""
        component = self.registry.get_component("rag_agent", config_override=config)
        if component is None:
            raise RuntimeError("Failed to create rag_agent component")
        return component

    def create_skill_graph_analyzer(
        self, config: dict[str, Any] | None = None
    ) -> ISkillGraphAnalyzer:
        """創建技能圖分析器實例"""
        component = self.registry.get_component(
            "skill_graph_analyzer", config_override=config
        )
        if component is None:
            raise RuntimeError("Failed to create skill_graph_analyzer component")
        return component


class AIVAContext(IAIContext):
    """AIVA AI 上下文管理器實現"""

    def __init__(
        self,
        factory: AIVAComponentFactory | None = None,
        auto_init_components: bool = True,
    ):
        """初始化上下文

        Args:
            factory: 組件工廠
            auto_init_components: 是否自動初始化組件
        """
        self.factory = factory or AIVAComponentFactory()
        self.auto_init_components = auto_init_components
        self._components: dict[str, Any] = {}
        self._initialized = False

        logger.info("AIVAContext initialized")

    async def __aenter__(self) -> "AIVAContext":
        """進入上下文管理器"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """退出上下文管理器"""
        await self.cleanup()

    async def initialize(self) -> None:
        """初始化 AI 上下文"""
        if self._initialized:
            logger.warning("AIVAContext already initialized")
            return

        try:
            if self.auto_init_components:
                # 自動初始化所有可用組件
                component_types = [
                    "dialog_assistant",
                    "plan_executor",
                    "experience_manager",
                    "capability_evaluator",
                    "cross_language_bridge",
                    "rag_agent",
                    "skill_graph_analyzer",
                ]

                for comp_type in component_types:
                    try:
                        component = getattr(self.factory, f"create_{comp_type}")()
                        if component:
                            self._components[comp_type] = component
                            logger.debug(f"Initialized {comp_type} component")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {comp_type}: {e}")

            self._initialized = True
            logger.info(
                f"AIVAContext initialized with {len(self._components)} components"
            )

        except Exception as e:
            logger.error(f"Failed to initialize AIVAContext: {e}")
            raise

    async def cleanup(self) -> None:
        """清理 AI 上下文"""
        if not self._initialized:
            return

        try:
            # 清理所有組件
            for comp_type, component in self._components.items():
                try:
                    # 如果組件有清理方法，調用它
                    if hasattr(component, "cleanup") and callable(component.cleanup):
                        if inspect.iscoroutinefunction(component.cleanup):
                            await component.cleanup()
                        else:
                            component.cleanup()
                    logger.debug(f"Cleaned up {comp_type} component")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {comp_type}: {e}")

            self._components.clear()
            self._initialized = False
            logger.info("AIVAContext cleaned up")

        except Exception as e:
            logger.error(f"Failed to cleanup AIVAContext: {e}")
            raise

    def get_component(self, component_type: str) -> Any | None:
        """從上下文中獲取組件"""
        return self._components.get(component_type)

    def set_component(self, component_type: str, component: Any) -> None:
        """設置組件到上下文"""
        self._components[component_type] = component
        logger.debug(f"Set {component_type} component in context")

    def is_initialized(self) -> bool:
        """檢查上下文是否已初始化"""
        return self._initialized

    def get_available_components(self) -> list[str]:
        """獲取可用組件列表"""
        return list(self._components.keys())


# ============================================================================
# Global Registry (全域註冊表)
# ============================================================================

_global_registry: AIVAComponentRegistry | None = None


def get_global_registry() -> AIVAComponentRegistry:
    """獲取全域組件註冊表

    Returns:
        全域註冊表實例
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AIVAComponentRegistry()
    return _global_registry


def set_global_registry(registry: AIVAComponentRegistry) -> None:
    """設置全域組件註冊表

    Args:
        registry: 註冊表實例
    """
    global _global_registry
    _global_registry = registry


@asynccontextmanager
async def aiva_ai_context(
    factory: AIVAComponentFactory | None = None, auto_init_components: bool = True
):
    """AIVA AI 上下文管理器的便捷函數

    Args:
        factory: 組件工廠
        auto_init_components: 是否自動初始化組件

    Yields:
        AIVAContext 實例

    Example:
        async with aiva_ai_context() as ai_ctx:
            dialog = ai_ctx.get_component("dialog_assistant")
            response = await dialog.process_user_input("現在系統會什麼？")
    """
    context = AIVAContext(factory, auto_init_components)
    try:
        await context.initialize()
        yield context
    finally:
        await context.cleanup()


# ============================================================================
# Utility Functions (工具函數)
# ============================================================================


def register_builtin_components() -> None:
    """註冊內建的 AI 組件 (如果可用)"""
    registry = get_global_registry()

    # 嘗試註冊來自 services.core.aiva_core 的組件
    try:
        from typing import Any, Type, cast

        from services.core.aiva_core.dialog.assistant import (
            dialog_assistant,  # type: ignore
        )

        if dialog_assistant is not None:
            # 明確轉換動態導入的組件類型
            dialog_assistant_typed = cast(Any, dialog_assistant)
            component_class = cast(type[Any], type(dialog_assistant_typed))
            registry.register_component(
                "dialog_assistant", "aiva_core_dialog", component_class, is_default=True
            )
            logger.info("Registered aiva_core dialog assistant")
    except (ImportError, AttributeError) as e:
        logger.debug(f"aiva_core dialog assistant not available: {e}")

    # 嘗試註冊其他內建組件...
    # (這裡可以添加更多內建組件的註冊邏輯)

    logger.info("Builtin component registration completed")
