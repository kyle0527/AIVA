"""Capability Registry - 能力註冊表

基於 internal_exploration 的能力分析結果，提供動態能力註冊和查詢

Architecture Fix Note:
- 創建日期: 2025-11-16
- 目的: 解決問題四「規劃器如何實際調用工具」
- 功能: 動態註冊和查詢系統能力，支持 UnifiedFunctionCaller 調用
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityInfo:
    """能力信息"""

    def __init__(
        self,
        name: str,
        module: str,
        description: str,
        parameters: list[str],
        file_path: str,
        return_type: str | None = None,
        is_async: bool = False,
    ):
        self.name = name
        self.module = module
        self.description = description
        self.parameters = parameters
        self.file_path = file_path
        self.return_type = return_type
        self.is_async = is_async
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "parameters": self.parameters,
            "file_path": self.file_path,
            "return_type": self.return_type,
            "is_async": self.is_async,
            "metadata": self.metadata,
        }


class CapabilityRegistry:
    """能力註冊表 (Singleton)

    職責：
    1. 從 internal_exploration 載入能力分析結果
    2. 提供能力註冊和查詢接口
    3. 支持 UnifiedFunctionCaller 動態調用
    4. 管理能力元數據和索引
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._capabilities: dict[str, CapabilityInfo] = {}
        self._module_index: dict[str, list[str]] = {}  # module -> [capability_names]
        self._initialized = True
        logger.info("CapabilityRegistry initialized (Singleton)")

    async def load_from_exploration(self) -> dict[str, Any]:
        """從 internal_exploration 載入能力
        
        使用 internal_loop_connector 獲取能力分析結果
        
        Returns:
            {
                "capabilities_loaded": int,
                "modules_indexed": int,
                "errors": list
            }
        """
        logger.info("🔄 Loading capabilities from internal_exploration...")

        try:
            # 導入 internal_loop_connector
            from services.core.aiva_core.cognitive_core.internal_loop_connector import (
                InternalLoopConnector,
            )
            from services.core.aiva_core.cognitive_core.rag.knowledge_base import (
                KnowledgeBase,
            )
            from services.core.aiva_core.cognitive_core.rag.unified_vector_store import (
                UnifiedVectorStore,
            )

            # 初始化 vector_store 和 knowledge_base
            vector_store = UnifiedVectorStore()
            kb = KnowledgeBase(vector_store=vector_store)
            connector = InternalLoopConnector(rag_knowledge_base=kb)

            # 獲取能力分析結果
            result = await connector.sync_capabilities_to_rag(force_refresh=False)

            capabilities_data = result.get("capabilities", [])

            # 註冊能力
            for cap_data in capabilities_data:
                await self.register_capability(
                    name=cap_data.get("name"),
                    module=cap_data.get("module"),
                    description=cap_data.get("description", ""),
                    parameters=cap_data.get("parameters", []),
                    file_path=cap_data.get("file_path"),
                    return_type=cap_data.get("return_type"),
                    is_async=cap_data.get("is_async", False),
                )

            logger.info(
                f"✅ Loaded {len(self._capabilities)} capabilities from "
                f"{len(self._module_index)} modules"
            )

            return {
                "capabilities_loaded": len(self._capabilities),
                "modules_indexed": len(self._module_index),
                "errors": [],
            }

        except Exception as e:
            error_msg = f"Failed to load capabilities: {e}"
            logger.error(error_msg)
            return {
                "capabilities_loaded": 0,
                "modules_indexed": 0,
                "errors": [error_msg],
            }

    async def register_capability(
        self,
        name: str,
        module: str,
        description: str = "",
        parameters: list[str] | None = None,
        file_path: str | None = None,
        return_type: str | None = None,
        is_async: bool = False,
        **metadata,
    ) -> bool:
        """註冊能力
        
        Args:
            name: 能力名稱
            module: 模組名稱
            description: 描述
            parameters: 參數列表
            file_path: 文件路徑
            return_type: 返回類型
            is_async: 是否異步
            **metadata: 額外元數據
            
        Returns:
            註冊是否成功
        """
        try:
            capability = CapabilityInfo(
                name=name,
                module=module,
                description=description,
                parameters=parameters or [],
                file_path=file_path or "",
                return_type=return_type,
                is_async=is_async,
            )
            capability.metadata.update(metadata)

            # 註冊到主字典
            self._capabilities[name] = capability

            # 更新模組索引
            if module not in self._module_index:
                self._module_index[module] = []
            if name not in self._module_index[module]:
                self._module_index[module].append(name)

            logger.debug(f"Registered capability: {name} (module: {module})")
            return True

        except Exception as e:
            logger.error(f"Failed to register capability {name}: {e}")
            return False

    def get_capability(self, name: str) -> CapabilityInfo | None:
        """獲取能力信息
        
        Args:
            name: 能力名稱
            
        Returns:
            能力信息或 None
        """
        return self._capabilities.get(name)

    def list_capabilities(
        self, module: str | None = None, filter_func=None
    ) -> list[CapabilityInfo]:
        """列出能力
        
        Args:
            module: 可選的模組過濾
            filter_func: 可選的自定義過濾函數
            
        Returns:
            能力列表
        """
        capabilities = []

        if module:
            # 按模組過濾
            capability_names = self._module_index.get(module, [])
            capabilities = [self._capabilities[name] for name in capability_names]
        else:
            # 全部能力
            capabilities = list(self._capabilities.values())

        # 應用自定義過濾
        if filter_func:
            capabilities = [cap for cap in capabilities if filter_func(cap)]

        return capabilities

    def list_modules(self) -> list[str]:
        """列出所有模組
        
        Returns:
            模組名稱列表
        """
        return list(self._module_index.keys())

    def search_capabilities(self, keyword: str) -> list[CapabilityInfo]:
        """搜索能力
        
        Args:
            keyword: 搜索關鍵字 (在名稱或描述中搜索)
            
        Returns:
            匹配的能力列表
        """
        keyword_lower = keyword.lower()
        results = []

        for capability in self._capabilities.values():
            if (
                keyword_lower in capability.name.lower()
                or keyword_lower in capability.description.lower()
            ):
                results.append(capability)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息
        
        Returns:
            {
                "total_capabilities": int,
                "total_modules": int,
                "async_capabilities": int,
                "capabilities_by_module": dict
            }
        """
        async_count = sum(
            1 for cap in self._capabilities.values() if cap.is_async
        )

        capabilities_by_module = {
            module: len(caps) for module, caps in self._module_index.items()
        }

        return {
            "total_capabilities": len(self._capabilities),
            "total_modules": len(self._module_index),
            "async_capabilities": async_count,
            "capabilities_by_module": capabilities_by_module,
        }

    def clear(self):
        """清空註冊表 (用於測試)"""
        self._capabilities.clear()
        self._module_index.clear()
        logger.info("CapabilityRegistry cleared")


# 全局實例
_global_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    """獲取全局能力註冊表實例 (Singleton)
    
    Returns:
        CapabilityRegistry 實例
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry


async def initialize_capability_registry(force_refresh: bool = False) -> dict[str, Any]:
    """初始化能力註冊表
    
    這個函數應該在應用啟動時調用
    
    Args:
        force_refresh: 是否強制刷新
        
    Returns:
        初始化結果
    """
    registry = get_capability_registry()

    if not registry._capabilities or force_refresh:
        result = await registry.load_from_exploration()
        return result
    else:
        logger.info("CapabilityRegistry already initialized, skipping load")
        return {
            "capabilities_loaded": len(registry._capabilities),
            "modules_indexed": len(registry._module_index),
            "errors": [],
            "skipped": True,
        }


# 測試代碼
if __name__ == "__main__":
    async def test_registry():
        """測試能力註冊表"""
        print("🧪 Testing CapabilityRegistry...")

        # 獲取實例
        registry = get_capability_registry()

        # 載入能力
        result = await registry.load_from_exploration()
        print(f"\n📊 Load Result:")
        print(f"   - Capabilities loaded: {result['capabilities_loaded']}")
        print(f"   - Modules indexed: {result['modules_indexed']}")
        print(f"   - Errors: {result['errors']}")

        # 獲取統計
        stats = registry.get_statistics()
        print(f"\n📈 Statistics:")
        print(f"   - Total capabilities: {stats['total_capabilities']}")
        print(f"   - Total modules: {stats['total_modules']}")
        print(f"   - Async capabilities: {stats['async_capabilities']}")

        # 列出模組
        modules = registry.list_modules()
        print(f"\n📦 Modules ({len(modules)}):")
        for module in modules[:5]:  # 只顯示前 5 個
            caps = registry.list_capabilities(module=module)
            print(f"   - {module}: {len(caps)} capabilities")

        # 搜索能力
        search_results = registry.search_capabilities("sql")
        print(f"\n🔍 Search 'sql': {len(search_results)} results")
        for cap in search_results[:3]:  # 只顯示前 3 個
            print(f"   - {cap.name} ({cap.module})")

        print("\n✅ Test completed!")

    asyncio.run(test_registry())
