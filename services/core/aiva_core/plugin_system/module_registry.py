#!/usr/bin/env python3
"""
AI 模組註冊中心

負責插件的註冊、發現、管理和生命週期控制。
設計參考 Kubernetes API Aggregation 和 Service Registry Pattern。

核心功能:
1. 插件註冊和驗證
2. 自動發現 (*_plugin.py)
3. 能力查詢和路由
4. 健康檢查和監控
5. 優雅關閉管理

使用方式:
    registry = ModuleRegistry(Path("data"))
    
    # 手動註冊
    await registry.register_plugin(BioNeuronPlugin())
    
    # 自動發現
    discovered = await registry.discover_plugins(Path("services/plugins"))
    
    # 獲取插件
    plugin = registry.get_plugin("bio_neuron")
    result = await plugin.execute_task(task)
"""

import logging
import asyncio
import importlib.util
import inspect
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_plugin import AIModulePlugin, AITask

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """AI 模組註冊中心
    
    負責管理所有 AI 插件的生命週期：
    - 註冊和驗證
    - 自動發現
    - 查詢和路由
    - 健康監控
    
    設計參考 Kubernetes API Aggregation Pattern
    """
    
    def __init__(self, data_directory: Path):
        """初始化註冊中心
        
        Args:
            data_directory: 數據目錄，用於存儲配置和日誌
        """
        self.data_directory = data_directory
        self.plugins: Dict[str, AIModulePlugin] = {}
        self._lock = asyncio.Lock()
        
        # 創建配置目錄
        self.config_dir = data_directory / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModuleRegistry initialized with data_directory={data_directory}")
    
    async def register_plugin(
        self,
        plugin: AIModulePlugin,
        weight_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """註冊 AI 模組插件
        
        執行流程:
        1. 檢查是否已註冊
        2. 驗證插件接口
        3. 載入配置
        4. 初始化插件
        5. 載入權重 (如果需要)
        6. 健康檢查
        7. 註冊成功
        
        Args:
            plugin: 插件實例
            weight_path: 權重路徑 (如果需要)
            config: 插件配置 (可選，否則從文件載入)
        
        Returns:
            註冊是否成功
        """
        async with self._lock:
            try:
                module_id = plugin.module_id
                
                # 1. 檢查是否已註冊
                if module_id in self.plugins:
                    logger.warning(f"Plugin {module_id} already registered, skipping")
                    return False
                
                # 2. 驗證插件接口
                if not self._validate_plugin(plugin):
                    raise ValueError(f"Plugin {module_id} does not implement required interface")
                
                # 3. 載入配置
                if config is None:
                    config = self._load_plugin_config(module_id)
                
                # 4. 初始化插件
                logger.info(f"Initializing plugin: {module_id}")
                if not await plugin.initialize(config):
                    raise RuntimeError(f"Plugin {module_id} initialization failed")
                
                # 5. 載入權重 (如果需要)
                if plugin.requires_weights:
                    if weight_path is None:
                        logger.warning(f"Plugin {module_id} requires weights but none provided")
                    else:
                        logger.info(f"Loading weights for {module_id}: {weight_path}")
                        if not await plugin.load_weights(weight_path):
                            raise RuntimeError(f"Failed to load weights for {module_id}")
                
                # 6. 健康檢查
                if not await plugin.health_check():
                    raise RuntimeError(f"Plugin {module_id} failed health check")
                
                # 7. 註冊成功
                self.plugins[module_id] = plugin
                
                logger.info(f"✅ Plugin registered successfully: {module_id}")
                logger.info(f"   Version: {plugin.version}")
                logger.info(f"   Capabilities: {plugin.capabilities}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to register plugin {plugin.module_id}: {e}", exc_info=True)
                return False
    
    def _validate_plugin(self, plugin: AIModulePlugin) -> bool:
        """驗證插件是否實現必需的接口
        
        Args:
            plugin: 插件實例
        
        Returns:
            驗證是否通過
        """
        required_attrs = [
            'module_id', 'capabilities', 'version',
            'initialize', 'execute_task', 'health_check', 'shutdown'
        ]
        
        for attr in required_attrs:
            if not hasattr(plugin, attr):
                logger.error(f"Plugin missing required attribute: {attr}")
                return False
        
        # 驗證 module_id 格式
        module_id = plugin.module_id
        if not module_id or not module_id.replace('_', '').isalnum():
            logger.error(f"Invalid module_id format: {module_id}")
            return False
        
        # 驗證 capabilities 不為空
        if not plugin.capabilities:
            logger.error(f"Plugin {module_id} has no capabilities")
            return False
        
        return True
    
    def _load_plugin_config(self, module_id: str) -> Dict[str, Any]:
        """載入插件配置文件
        
        支持格式: YAML, JSON
        
        Args:
            module_id: 模組 ID
        
        Returns:
            配置字典，如果不存在則返回默認配置
        """
        # 嘗試載入 YAML 配置
        yaml_config = self.config_dir / f"{module_id}.yaml"
        if yaml_config.exists():
            try:
                import yaml
                with open(yaml_config, encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded config from {yaml_config}")
                    return config
            except Exception as e:
                logger.error(f"Failed to load YAML config: {e}")
        
        # 嘗試載入 JSON 配置
        json_config = self.config_dir / f"{module_id}.json"
        if json_config.exists():
            try:
                import json
                with open(json_config, encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"Loaded config from {json_config}")
                    return config
            except Exception as e:
                logger.error(f"Failed to load JSON config: {e}")
        
        # 返回默認配置
        logger.info(f"No config file found for {module_id}, using defaults")
        return {
            "enabled": True,
            "log_level": "INFO",
            "timeout": 300,
        }
    
    async def discover_plugins(self, plugin_dir: Path) -> List[str]:
        """自動發現插件
        
        掃描指定目錄下的 *_plugin.py 文件，自動載入並註冊
        
        Args:
            plugin_dir: 插件目錄
        
        Returns:
            成功發現的插件 ID 列表
        """
        discovered = []
        
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return discovered
        
        logger.info(f"Discovering plugins in {plugin_dir}...")
        
        for plugin_file in plugin_dir.glob("*_plugin.py"):
            try:
                # 動態導入模組
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec is None or spec.loader is None:
                    logger.error(f"Failed to load spec for {plugin_file}")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找實現 AIModulePlugin 的類
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_plugin_class(obj):
                        logger.info(f"Found plugin class: {name} in {plugin_file.name}")
                        
                        # 實例化並註冊
                        plugin = obj()
                        if await self.register_plugin(plugin):
                            discovered.append(plugin.module_id)
                        
            except Exception as e:
                logger.error(f"Failed to discover plugin {plugin_file}: {e}", exc_info=True)
        
        logger.info(f"✅ Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def _is_plugin_class(self, obj) -> bool:
        """檢查是否為有效的插件類
        
        Args:
            obj: 待檢查對象
        
        Returns:
            是否為插件類
        """
        try:
            return (
                inspect.isclass(obj) and
                issubclass(obj, AIModulePlugin) and
                obj is not AIModulePlugin  # 排除基類本身
            )
        except TypeError:
            return False
    
    def get_plugin(self, module_id: str) -> Optional[AIModulePlugin]:
        """獲取已註冊的插件
        
        Args:
            module_id: 模組 ID
        
        Returns:
            插件實例，如果不存在則返回 None
        """
        return self.plugins.get(module_id)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已註冊插件
        
        Returns:
            插件信息列表
        """
        plugins_info = []
        
        for plugin in self.plugins.values():
            try:
                # 同步執行健康檢查
                is_healthy = asyncio.run(plugin.health_check())
                status = "active" if is_healthy else "unhealthy"
            except Exception as e:
                status = "error"
                logger.error(f"Health check failed for {plugin.module_id}: {e}")
            
            plugins_info.append({
                "module_id": plugin.module_id,
                "version": plugin.version,
                "capabilities": plugin.capabilities,
                "requires_weights": plugin.requires_weights,
                "status": status,
                "metadata": plugin.get_metadata()
            })
        
        return plugins_info
    
    def get_plugins_by_capability(self, capability: str) -> List[AIModulePlugin]:
        """根據能力查找插件
        
        Args:
            capability: 能力名稱 (如 "scan", "analyze")
        
        Returns:
            支持該能力的插件列表
        """
        return [
            plugin for plugin in self.plugins.values()
            if capability in plugin.capabilities
        ]
    
    async def health_check_all(self) -> Dict[str, bool]:
        """對所有插件執行健康檢查
        
        Returns:
            插件 ID -> 健康狀態 映射
        """
        results = {}
        
        for module_id, plugin in self.plugins.items():
            try:
                is_healthy = await plugin.health_check()
                results[module_id] = is_healthy
                
                status_icon = "✅" if is_healthy else "❌"
                logger.info(f"{status_icon} Plugin {module_id}: {'healthy' if is_healthy else 'unhealthy'}")
                
            except Exception as e:
                results[module_id] = False
                logger.error(f"❌ Health check error for {module_id}: {e}")
        
        return results
    
    async def shutdown_all(self):
        """優雅關閉所有插件
        
        按照註冊順序的反序關閉，確保依賴關係正確
        """
        logger.info("Shutting down all plugins...")
        
        # 反序關閉
        for module_id in reversed(list(self.plugins.keys())):
            plugin = self.plugins[module_id]
            try:
                await plugin.shutdown()
                logger.info(f"✅ Plugin {module_id} shut down successfully")
            except Exception as e:
                logger.error(f"❌ Error shutting down plugin {module_id}: {e}")
        
        # 清空註冊表
        self.plugins.clear()
        logger.info("All plugins shut down")
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取註冊中心統計信息
        
        Returns:
            統計信息字典
        """
        total_capabilities = set()
        for plugin in self.plugins.values():
            total_capabilities.update(plugin.capabilities)
        
        return {
            "total_plugins": len(self.plugins),
            "total_capabilities": len(total_capabilities),
            "plugins": [p.module_id for p in self.plugins.values()],
            "capabilities": sorted(total_capabilities),
        }
