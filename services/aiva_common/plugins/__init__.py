"""
現代化插件架構系統
基於 entry_points 和動態加載的最佳實踐
"""



import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar, Union

from pydantic import BaseModel, Field, validator

# 類型變數
T = TypeVar('T')
PluginType = TypeVar('PluginType', bound='BasePlugin')


class PluginMetadata(BaseModel):
    """插件元數據"""
    name: str = Field(..., description="插件名稱")
    version: str = Field(..., description="插件版本")
    description: str = Field("", description="插件描述")
    author: str = Field("", description="插件作者")
    license: str = Field("", description="許可證")
    dependencies: List[str] = Field(default_factory=list, description="依賴項")
    entry_point: str = Field(..., description="入口點")
    category: str = Field("general", description="插件類別")
    tags: List[str] = Field(default_factory=list, description="標籤")
    min_aiva_version: str = Field("0.1.0", description="最小 AIVA 版本")
    max_aiva_version: Optional[str] = Field(None, description="最大 AIVA 版本")
    enabled: bool = Field(True, description="是否啟用")
    priority: int = Field(0, description="優先級，數字越大優先級越高")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Plugin name must be alphanumeric with underscores or hyphens")
        return v


class PluginConfig(BaseModel):
    """插件配置"""
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)
    auto_load: bool = True
    lazy_load: bool = False


class PluginHook(Protocol):
    """插件鉤子協議"""
    def __call__(self, *args, **kwargs) -> Any:
        ...


class BasePlugin(ABC):
    """基礎插件類"""
    
    def __init__(self, metadata: PluginMetadata, config: Optional[PluginConfig] = None):
        self.metadata = metadata
        self.config = config or PluginConfig()
        self._initialized = False
        self._logger = logging.getLogger(f"plugin.{metadata.name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化插件"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理插件資源"""
        pass
    
    async def configure(self, settings: Dict[str, Any]) -> None:
        """配置插件"""
        self.config.settings.update(settings)
    
    def is_initialized(self) -> bool:
        """檢查是否已初始化"""
        return self._initialized
    
    def get_hooks(self) -> Dict[str, PluginHook]:
        """獲取插件提供的鉤子"""
        hooks = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_is_hook'):
                hooks[name] = method
        return hooks
    
    @property
    def logger(self) -> logging.Logger:
        """獲取插件日誌器"""
        return self._logger


def plugin_hook(name: Optional[str] = None):
    """插件鉤子裝飾器"""
    def decorator(func):
        func._is_hook = True
        func._hook_name = name or func.__name__
        return func
    return decorator


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_configs: Dict[str, PluginConfig] = {}
        self._hooks: Dict[str, List[PluginHook]] = {}
        self._logger = logging.getLogger("plugin_manager")
        self._plugin_paths: List[Path] = []
    
    def add_plugin_path(self, path: Union[str, Path]) -> None:
        """添加插件搜索路徑"""
        path = Path(path)
        if path.exists() and path.is_dir():
            self._plugin_paths.append(path)
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """發現插件"""
        discovered = []
        
        # 從搜索路徑中發現插件
        for plugin_path in self._plugin_paths:
            for plugin_file in plugin_path.glob("**/plugin.toml"):
                try:
                    metadata = self._load_plugin_metadata(plugin_file)
                    discovered.append(metadata)
                except Exception as e:
                    self._logger.warning(f"Failed to load plugin metadata from {plugin_file}: {e}")
        
        # 從 entry_points 發現插件
        try:
            import pkg_resources
            for entry_point in pkg_resources.iter_entry_points('aiva.plugins'):
                try:
                    metadata = self._create_metadata_from_entry_point(entry_point)
                    discovered.append(metadata)
                except Exception as e:
                    self._logger.warning(f"Failed to load plugin from entry point {entry_point}: {e}")
        except ImportError:
            pass  # pkg_resources 不可用
        
        return discovered
    
    def _load_plugin_metadata(self, plugin_file: Path) -> PluginMetadata:
        """從 TOML 文件加載插件元數據"""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        
        with open(plugin_file, 'rb') as f:
            data = tomllib.load(f)
        
        plugin_data = data.get('plugin', {})
        return PluginMetadata(**plugin_data)
    
    def _create_metadata_from_entry_point(self, entry_point) -> PluginMetadata:
        """從 entry_point 創建元數據"""
        plugin_class = entry_point.load()
        
        # 嘗試獲取插件的元數據
        if hasattr(plugin_class, 'METADATA'):
            return plugin_class.METADATA
        else:
            # 創建默認元數據
            return PluginMetadata(
                name=entry_point.name,
                version="1.0.0",
                description="Auto-generated plugin",
                author="AIVA",
                license="MIT",
                category="general",
                min_aiva_version="1.0.0",
                max_aiva_version="2.0.0",
                enabled=True,
                priority=0,
                entry_point=f"{plugin_class.__module__}:{plugin_class.__name__}"
            )
    
    async def load_plugin(self, metadata: PluginMetadata) -> Optional[BasePlugin]:
        """加載插件"""
        if metadata.name in self._plugins:
            self._logger.warning(f"Plugin {metadata.name} already loaded")
            return self._plugins[metadata.name]
        
        try:
            # 動態導入插件
            module_path, class_name = metadata.entry_point.split(':')
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            
            # 檢查是否為有效的插件類
            if not issubclass(plugin_class, BasePlugin):
                raise ValueError(f"Plugin class {class_name} must inherit from BasePlugin")
            
            # 創建插件實例
            config = self._plugin_configs.get(metadata.name, PluginConfig())
            plugin = plugin_class(metadata, config)
            
            # 初始化插件
            if not config.lazy_load:
                await plugin.initialize()
                plugin._initialized = True
            
            # 註冊插件
            self._plugins[metadata.name] = plugin
            
            # 註冊鉤子
            hooks = plugin.get_hooks()
            for hook_name, hook_func in hooks.items():
                if hook_name not in self._hooks:
                    self._hooks[hook_name] = []
                self._hooks[hook_name].append(hook_func)
            
            self._logger.info(f"Plugin {metadata.name} loaded successfully")
            return plugin
            
        except Exception as e:
            self._logger.error(f"Failed to load plugin {metadata.name}: {e}")
            return None
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """卸載插件"""
        if plugin_name not in self._plugins:
            return False
        
        try:
            plugin = self._plugins[plugin_name]
            
            # 清理插件資源
            await plugin.cleanup()
            
            # 移除鉤子
            hooks = plugin.get_hooks()
            for hook_name, hook_func in hooks.items():
                if hook_name in self._hooks:
                    try:
                        self._hooks[hook_name].remove(hook_func)
                        if not self._hooks[hook_name]:
                            del self._hooks[hook_name]
                    except ValueError:
                        pass
            
            # 移除插件
            del self._plugins[plugin_name]
            
            self._logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """重新加載插件"""
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        metadata = plugin.metadata
        
        # 卸載插件
        if await self.unload_plugin(plugin_name):
            # 重新加載插件
            new_plugin = await self.load_plugin(metadata)
            return new_plugin is not None
        
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """獲取插件實例"""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_category(self, category: str) -> List[BasePlugin]:
        """按類別獲取插件"""
        return [
            plugin for plugin in self._plugins.values()
            if plugin.metadata.category == category
        ]
    
    def get_enabled_plugins(self) -> List[BasePlugin]:
        """獲取啟用的插件"""
        return [
            plugin for plugin in self._plugins.values()
            if plugin.metadata.enabled and plugin.config.enabled
        ]
    
    async def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """調用鉤子"""
        results = []
        if hook_name in self._hooks:
            # 按優先級排序
            hooks = sorted(
                self._hooks[hook_name],
                key=lambda h: getattr(getattr(h, "metadata", None), "priority", 0),
                reverse=True
            )
            
            for hook in hooks:
                try:
                    if inspect.iscoroutinefunction(hook):
                        result = await hook(*args, **kwargs)
                    else:
                        result = hook(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    self._logger.error(f"Error calling hook {hook_name}: {e}")
        
        return results
    
    def configure_plugin(self, plugin_name: str, settings: Dict[str, Any]) -> bool:
        """配置插件"""
        if plugin_name in self._plugin_configs:
            self._plugin_configs[plugin_name].settings.update(settings)
        else:
            self._plugin_configs[plugin_name] = PluginConfig(settings=settings)
        
        # 如果插件已加載，更新配置
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            plugin.config.settings.update(settings)
            return True
        
        return False
    
    def get_plugin_list(self) -> List[Dict[str, Any]]:
        """獲取插件列表"""
        return [
            {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "description": plugin.metadata.description,
                "category": plugin.metadata.category,
                "enabled": plugin.metadata.enabled and plugin.config.enabled,
                "initialized": plugin.is_initialized(),
            }
            for plugin in self._plugins.values()
        ]
    
    async def initialize_all(self) -> None:
        """初始化所有插件"""
        for plugin in self._plugins.values():
            if not plugin.is_initialized() and plugin.config.enabled:
                try:
                    await plugin.initialize()
                    plugin._initialized = True
                except Exception as e:
                    self._logger.error(f"Failed to initialize plugin {plugin.metadata.name}: {e}")
    
    async def cleanup_all(self) -> None:
        """清理所有插件"""
        for plugin in self._plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                self._logger.error(f"Failed to cleanup plugin {plugin.metadata.name}: {e}")


class PluginRegistry:
    """插件註冊表"""
    
    def __init__(self):
        self._registry: Dict[str, Type[BasePlugin]] = {}
    
    def register(self, name: str, plugin_class: Type[BasePlugin]) -> None:
        """註冊插件類"""
        if not issubclass(plugin_class, BasePlugin):
            raise ValueError("Plugin class must inherit from BasePlugin")
        
        self._registry[name] = plugin_class
    
    def unregister(self, name: str) -> bool:
        """取消註冊插件類"""
        if name in self._registry:
            del self._registry[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Type[BasePlugin]]:
        """獲取插件類"""
        return self._registry.get(name)
    
    def list_registered(self) -> List[str]:
        """列出已註冊的插件"""
        return list(self._registry.keys())


# 全域實例
default_plugin_manager = PluginManager()
default_plugin_registry = PluginRegistry()


# 裝飾器
def register_plugin(name: str):
    """插件註冊裝飾器"""
    def decorator(plugin_class: Type[BasePlugin]) -> Type[BasePlugin]:
        default_plugin_registry.register(name, plugin_class)
        return plugin_class
    return decorator


__all__ = [
    "BasePlugin",
    "PluginMetadata",
    "PluginConfig",
    "PluginHook",
    "PluginManager",
    "PluginRegistry",
    "plugin_hook",
    "register_plugin",
    "default_plugin_manager",
    "default_plugin_registry",
]