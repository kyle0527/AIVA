"""Capability Registry - 能力註冊表代理

基於 aiva_common 單一數據來源 (SOT) 原則，
此模組作為 services.integration.capability.CapabilityRegistry 的代理。

Architecture Fix Note:
- 創建日期: 2025-11-16 
- 最後更新: 2025-12-16 (統一為代理模式)
- 目的: 統一 aiva_core 和 integration 的能力註冊系統
- 設計原則: 遵循 aiva_common 單一數據來源原則
- 功能: 動態註冊和查詢系統能力，支持 UnifiedFunctionCaller 調用
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CapabilityInfo:
    """能力信息 - 向後兼容的包裝類
    
    為了保持 aiva_core 代碼的向後兼容性，提供簡化的 CapabilityInfo。
    內部實際使用 integration.CapabilityRecord。
    """

    def __init__(
        self,
        name: str,
        module: str,
        description: str,
        parameters: list[str],
        file_path: str,
        return_type: str | None = None,
        is_async: bool = False,
        capability_id: str | None = None,
        language: str = "python",
    ):
        self.name = name
        self.module = module
        self.description = description
        self.parameters = parameters
        self.file_path = file_path
        self.return_type = return_type
        self.is_async = is_async
        self.id = capability_id or f"{module}.{name}"
        self.language = language
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "id": self.id,
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "parameters": self.parameters,
            "file_path": self.file_path,
            "return_type": self.return_type,
            "is_async": self.is_async,
            "language": self.language,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_capability_record(cls, record):
        """從 CapabilityRecord 創建 CapabilityInfo
        
        Args:
            record: services.integration.capability.models.CapabilityRecord
            
        Returns:
            CapabilityInfo 實例
        """
        # 從 record.config 提取參數
        parameters = []
        if record.config:
            params = record.config.get('parameters', [])
            if isinstance(params, list):
                parameters = params
        
        return cls(
            name=record.name,
            module=record.module,
            description=record.description or "",
            parameters=parameters,
            file_path=record.entrypoint,
            return_type=record.config.get('return_type') if record.config else None,
            is_async=record.config.get('is_async', False) if record.config else False,
            capability_id=record.id,
            language=record.language if isinstance(record.language, str) else record.language.value,
        )


class CapabilityRegistry:
    """能力註冊表代理 (Singleton)
    
    遵循 aiva_common 單一數據來源 (SOT) 原則。
    此類作為 services.integration.capability.CapabilityRegistry 的代理，
    提供向後兼容的接口，同時所有數據操作都委託給 integration 模組。

    職責：
    1. 代理到 integration.CapabilityRegistry
    2. 提供向後兼容的 API
    3. 轉換數據格式 (CapabilityRecord ↔ CapabilityInfo)
    4. 保持 aiva_core 代碼的可用性
    """

    _instance: Optional['CapabilityRegistry'] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 導入 integration.CapabilityRegistry（延遲導入避免循環依賴）
        from services.integration.capability.registry import (
            registry as integration_registry,
        )
        
        self._integration_registry = integration_registry
        self._capabilities: dict[str, CapabilityInfo] = {}  # 本地緩存
        self._module_index: dict[str, list[str]] = {}  # module -> [capability_names]
        self._initialized = True
        logger.info("✅ CapabilityRegistry proxy initialized (delegates to integration.capability)")

    async def load_from_exploration(self) -> dict[str, Any]:
        """從 internal_exploration 載入能力並同步到 integration registry
        
        此方法會：
        1. 使用 internal_loop_connector 從 Internal Exploration 分析能力
        2. 將能力註冊到 integration.CapabilityRegistry (單一數據源)
        3. 同步到 RAG 向量數據庫供 AI 查詢
        
        Returns:
            {
                "capabilities_loaded": int,
                "modules_indexed": int,
                "errors": list
            }
        """
        logger.info("🔄 Loading capabilities from internal_exploration and syncing to integration registry...")

        try:
            # 導入必要模組
            from datetime import datetime

            from aiva_common.enums.modules import ProgrammingLanguage

            from services.core.aiva_core.cognitive_core.internal_loop_connector import (
                InternalLoopConnector,
            )
            from services.core.aiva_core.cognitive_core.rag.knowledge_base import (
                KnowledgeBase,
            )
            from services.core.aiva_core.cognitive_core.rag.unified_vector_store import (
                UnifiedVectorStore,
            )
            from services.integration.capability.models import (
                CapabilityRecord,
                CapabilityStatus,
                CapabilityType,
            )

            # 初始化 vector_store 和 knowledge_base
            vector_store = UnifiedVectorStore()
            kb = KnowledgeBase(vector_store=vector_store)
            connector = InternalLoopConnector(rag_knowledge_base=kb)

            # 獲取能力分析結果
            result = await connector.sync_capabilities_to_rag(force_refresh=False)

            # InternalLoopSyncResult 是 Pydantic 模型，使用屬性訪問
            capabilities_data = getattr(result, "capabilities", [])

            # 註冊到 integration registry（單一數據源）
            registered_count = 0
            for cap_data in capabilities_data:
                try:
                    # 創建 CapabilityRecord
                    cap_record = CapabilityRecord(
                        id=cap_data.get("id", f"{cap_data.get('module')}.{cap_data.get('name')}"),
                        name=cap_data.get("name"),
                        description=cap_data.get("description", ""),
                        module=cap_data.get("module"),
                        language=ProgrammingLanguage.PYTHON,  # 從 exploration 來的都是 Python
                        entrypoint=cap_data.get("file_path", ""),
                        capability_type=CapabilityType.UTILITY,  # 修正：使用正確的枚舉值
                        status=CapabilityStatus.HEALTHY,  # 修正：使用正確的枚舉值
                        category=cap_data.get("category", "general"),  # 必要參數
                        topic=cap_data.get("topic", "default"),  # 必要參數
                        last_probe=datetime.now(),  # 必要參數
                        last_success=datetime.now(),  # 必要參數
                        environment_vars=cap_data.get("environment_vars", {}),  # 必要參數
                        # v2.1: 去語意化反射引擎參數
                        rag_trigger=cap_data.get("rag_trigger"),  # 可選：環境特徵權重表
                        feature_signature=cap_data.get("feature_signature"),  # 可選：特徵簽名列表
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        config={
                            "parameters": cap_data.get("parameters", []),
                            "return_type": cap_data.get("return_type"),
                            "is_async": cap_data.get("is_async", False),
                        }
                    )
                    
                    # 註冊到 integration registry
                    await self._integration_registry.register_capability(cap_record)
                    
                    # 同時更新本地緩存（向後兼容）
                    cap_info = CapabilityInfo.from_capability_record(cap_record)
                    self._capabilities[cap_info.name] = cap_info
                    
                    # 更新模組索引
                    if cap_info.module not in self._module_index:
                        self._module_index[cap_info.module] = []
                    if cap_info.name not in self._module_index[cap_info.module]:
                        self._module_index[cap_info.module].append(cap_info.name)
                    
                    registered_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to register capability {cap_data.get('name')}: {e}")
                    continue

            logger.info(
                f"✅ Loaded {registered_count} capabilities from {len(self._module_index)} modules"
            )
            logger.info("   All capabilities synced to integration.CapabilityRegistry (SOT)")

            return {
                "capabilities_loaded": registered_count,
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

    def register_capability(
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
        """註冊能力（代理到 integration registry）
        
        遵循 aiva_common 單一數據來源原則，所有註冊操作委託給 integration.CapabilityRegistry。
        
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
            # 創建本地 CapabilityInfo（向後兼容）
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

            # 更新本地緩存
            self._capabilities[name] = capability

            # 更新模組索引
            if module not in self._module_index:
                self._module_index[module] = []
            if name not in self._module_index[module]:
                self._module_index[module].append(name)

            # 注意：同步註冊到 integration registry 需要在調用方處理（異步操作）
            # 此處僅更新本地緩存以保持性能
            logger.debug(f"Registered capability (local cache): {name} (module: {module})")
            return True

        except Exception as e:
            logger.error(f"Failed to register capability {name}: {e}")
            return False

    def get_capability(self, name: str) -> CapabilityInfo | None:
        """獲取能力信息（優先查詢本地緩存）
        
        Args:
            name: 能力名稱
            
        Returns:
            能力信息或 None
        """
        # 優先返回本地緩存（性能優化）
        if name in self._capabilities:
            return self._capabilities.get(name)
        
        # 注意：如需查詢 integration registry，請使用 sync_from_integration_registry() 方法
        # 此處僅返回本地緩存以保持性能
        logger.debug(f"Capability {name} not found in local cache")
        return None

    def list_capabilities(
        self, module: str | None = None, filter_func=None
    ) -> list[CapabilityInfo]:
        """列出能力（從本地緩存）
        
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
            capabilities = [self._capabilities[name] for name in capability_names if name in self._capabilities]
        else:
            # 全部能力
            capabilities = list(self._capabilities.values())

        # 應用自定義過濾
        if filter_func:
            capabilities = [cap for cap in capabilities if filter_func(cap)]

        return capabilities
    
    async def list_capabilities_async(
        self, module: str | None = None, status: str | None = None
    ) -> list[CapabilityInfo]:
        """異步列出能力（查詢 integration registry）
        
        Args:
            module: 可選的模組過濾
            status: 可選的狀態過濾
            
        Returns:
            能力列表
        """
        try:
            # 查詢 integration registry
            cap_records = await self._integration_registry.list_capabilities()
            
            # 轉換為 CapabilityInfo
            capabilities = []
            for record in cap_records:
                # 應用過濾
                if module and record.module != module:
                    continue
                if status and str(record.status.value) != status:
                    continue
                    
                cap_info = CapabilityInfo.from_capability_record(record)
                capabilities.append(cap_info)
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to list capabilities from integration registry: {e}")
            raise RuntimeError(
                f"能力列表查詢失敗: {e}。"
                "請確認 integration registry 服務已正確初始化。"
            ) from e

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
        print("\n📊 Load Result:")
        print(f"   - Capabilities loaded: {result['capabilities_loaded']}")
        print(f"   - Modules indexed: {result['modules_indexed']}")
        print(f"   - Errors: {result['errors']}")

        # 獲取統計
        stats = registry.get_statistics()
        print("\n📈 Statistics:")
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

