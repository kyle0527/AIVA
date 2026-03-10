# AIVA AI 模組整合實施計劃

[![Implementation Guide](https://img.shields.io/badge/Implementation-Guide-orange.svg)](https://github.com/)
[![Ready to Deploy](https://img.shields.io/badge/Ready%20to-Deploy-success.svg)](https://github.com/)

> **快速參考**: 從架構設計到生產部署的完整實施路線圖  
> **版本**: v1.0 | **最後更新**: 2025年11月29日

## 📑 目錄

- [實施策略](#實施策略)
  - [漸進式遷移原則](#漸進式遷移原則)
- [目錄結構](#目錄結構)
- [核心代碼實現](#核心代碼實現)
  - [1. AIModulePlugin 基礎接口](#1-aimoduleplugin-基礎接口)
  - [2. ModuleRegistry 實現](#2-moduleregistry-實現)
  - [3. WeightManager 實現](#3-weightmanager-實現)
  - [4. 示例插件: BioNeuronPlugin](#4-示例插件-bioneuronplugin)
- [啟動腳本](#啟動腳本)
  - [1. 權重註冊腳本](#1-權重註冊腳本)
  - [2. 快速啟動腳本](#2-快速啟動腳本)
- [測試計劃](#測試計劃)
  - [單元測試示例](#單元測試示例)
- [下一步行動](#下一步行動)

---

## 實施策略

### 漸進式遷移原則

```
階段 0: 基礎設施搭建 (不影響現有系統)
   ↓
階段 1: 新舊並存 (V1 和 V2 同時運行)
   ↓
階段 2: 逐步遷移 (一次一個模組)
   ↓
階段 3: 完全切換 (移除 V1 代碼)
```

**關鍵原則**: 
- ✅ 每個階段獨立可測試
- ✅ 隨時可回退到上一階段
- ✅ 不破壞現有功能
- ✅ 保持數據完整性

---

## 目錄結構

```
AIVA-git/
├── services/
│   ├── core/aiva_core/
│   │   ├── plugin_system/          # 新增: 插件系統
│   │   │   ├── __init__.py
│   │   │   ├── base_plugin.py      # AIModulePlugin 接口定義
│   │   │   ├── module_registry.py  # 模組註冊中心
│   │   │   └── weight_manager.py   # 權重管理器
│   │   │
│   │   ├── task_planning/
│   │   │   ├── ai_commander.py     # 保留 V1 (向後兼容)
│   │   │   ├── ai_commander_v2.py  # 新增: V2 插件化版本
│   │   │   └── coordinators/       # 新增: 領域協調器
│   │   │       ├── __init__.py
│   │   │       ├── base_coordinator.py
│   │   │       ├── attack_coordinator.py
│   │   │       ├── defense_coordinator.py
│   │   │       ├── analysis_coordinator.py
│   │   │       └── training_coordinator.py
│   │   │
│   │   └── plugins/                # 新增: 核心模組插件實現
│   │       ├── __init__.py
│   │       ├── bio_neuron_plugin.py
│   │       ├── scanner_plugin.py
│   │       ├── exploiter_plugin.py
│   │       ├── data_hub_plugin.py
│   │       └── learning_plugin.py
│   │
│   ├── integration/aiva_integration/
│   │   ├── unified_data_manager.py      # 保留 V1
│   │   ├── unified_data_manager_v2.py   # 新增: V2 統一數據管理
│   │   ├── ai_operation_recorder.py     # 保留 V1
│   │   ├── ai_operation_recorder_v2.py  # 已存在: V2 適配器
│   │   └── training_dataset_manager.py  # 新增: 訓練數據集管理
│   │
│   └── api/
│       ├── main.py                      # 更新: 添加 V2 Lifespan
│       └── routes/
│           └── ai_tasks.py              # 新增: AI 任務 API
│
├── data/                                # 數據目錄
│   ├── weights/                         # 權重存儲
│   │   ├── bio_neuron/
│   │   │   ├── v1.0.0/
│   │   │   │   ├── model.safetensors
│   │   │   │   ├── config.json
│   │   │   │   └── metadata.yaml
│   │   │   └── latest -> v1.0.0
│   │   ├── embeddings/
│   │   └── registry.json
│   │
│   ├── operations/                      # AI 操作記錄
│   └── integration/                     # Integration Module 數據
│
├── scripts/                             # 工具腳本
│   ├── register_weights.py              # 權重註冊腳本
│   ├── validate_plugins.py              # 插件驗證腳本
│   └── migrate_to_v2.py                 # 遷移腳本
│
└── docs/
    ├── AI_MODULE_INTEGRATION_ARCHITECTURE.md  # 架構文檔
    ├── AI_MODULE_INTEGRATION_IMPLEMENTATION_PLAN.md  # 本文檔
    └── plugin_development_guide.md      # 插件開發指南
```

---

## 核心代碼實現

### 1. AIModulePlugin 基礎接口

```python
# services/core/aiva_core/plugin_system/base_plugin.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class AITaskType(Enum):
    """AI 任務類型"""
    SCAN = "scan"
    ANALYZE = "analyze"
    EXPLOIT = "exploit"
    DEFEND = "defend"
    TRAIN = "train"
    LEARN = "learn"

@dataclass
class AITask:
    """AI 任務定義"""
    task_type: AITaskType
    target: Optional[str] = None
    parameters: Dict[str, Any] = None
    description: str = ""
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class AIResult:
    """AI 任務結果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    trace: Optional[Dict] = None
    metrics: Optional[Dict] = None
    feedback: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "trace": self.trace,
            "metrics": self.metrics,
            "feedback": self.feedback
        }

class AIModulePlugin(ABC):
    """AI 模組插件接口
    
    所有 AI 模組必須實現此接口才能被註冊到系統中
    設計參考 Kubernetes Device Plugin Pattern
    """
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """模組唯一標識符
        
        Example: "bio_neuron", "scanner", "exploiter"
        """
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """模組支援的能力列表
        
        Example: ["scan", "analyze_vulnerabilities", "generate_report"]
        """
        pass
    
    @property
    def requires_weights(self) -> bool:
        """是否需要載入權重
        
        Returns:
            True: 此模組需要 ML 模型權重
            False: 此模組基於規則或不需要權重
        """
        return False
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化模組
        
        Args:
            config: 模組配置字典
            
        Returns:
            初始化是否成功
        """
        pass
    
    async def load_weights(self, weight_path: Path) -> bool:
        """載入模型權重 (可選實現)
        
        Args:
            weight_path: 權重文件路徑
            
        Returns:
            載入是否成功
        """
        if self.requires_weights:
            raise NotImplementedError(
                f"Plugin {self.module_id} requires weights but load_weights() not implemented"
            )
        return True
    
    @abstractmethod
    async def execute_task(self, task: AITask) -> AIResult:
        """執行 AI 任務
        
        Args:
            task: AI 任務對象
            
        Returns:
            任務執行結果
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康檢查
        
        Returns:
            模組是否健康
        """
        pass
    
    async def shutdown(self) -> None:
        """優雅關閉
        
        清理資源、保存狀態等
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取模組元數據
        
        Returns:
            包含版本、作者、依賴等信息的字典
        """
        return {
            "module_id": self.module_id,
            "capabilities": self.capabilities,
            "requires_weights": self.requires_weights,
            "version": getattr(self, '__version__', '1.0.0')
        }
```

### 2. ModuleRegistry 實現

```python
# services/core/aiva_core/plugin_system/module_registry.py
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import importlib.util
import inspect
import asyncio

from .base_plugin import AIModulePlugin, AITask

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """AI 模組註冊中心
    
    負責：
    1. 插件註冊和管理
    2. 插件發現 (自動掃描)
    3. 插件驗證
    4. 插件生命週期管理
    """
    
    def __init__(self, data_directory: Path):
        self.data_directory = data_directory
        self.plugins: Dict[str, AIModulePlugin] = {}
        self._lock = asyncio.Lock()
        
        logger.info("ModuleRegistry initialized")
    
    async def register_plugin(
        self,
        plugin: AIModulePlugin,
        weight_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """註冊 AI 模組插件
        
        Args:
            plugin: 插件實例
            weight_path: 權重路徑 (如果需要)
            config: 插件配置
            
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
                        raise ValueError(f"Plugin {module_id} requires weights but none provided")
                    
                    logger.info(f"Loading weights for {module_id}: {weight_path}")
                    if not await plugin.load_weights(weight_path):
                        raise RuntimeError(f"Failed to load weights for {module_id}")
                
                # 6. 健康檢查
                if not await plugin.health_check():
                    raise RuntimeError(f"Plugin {module_id} failed health check")
                
                # 7. 註冊成功
                self.plugins[module_id] = plugin
                
                logger.info(f"✅ Plugin registered successfully: {module_id}")
                logger.info(f"   Capabilities: {plugin.capabilities}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to register plugin {plugin.module_id}: {e}")
                return False
    
    def _validate_plugin(self, plugin: AIModulePlugin) -> bool:
        """驗證插件是否實現必需的接口"""
        required_attrs = ['module_id', 'capabilities', 'initialize', 
                          'execute_task', 'health_check', 'shutdown']
        
        for attr in required_attrs:
            if not hasattr(plugin, attr):
                logger.error(f"Plugin missing required attribute: {attr}")
                return False
        
        return True
    
    def _load_plugin_config(self, module_id: str) -> Dict[str, Any]:
        """載入插件配置文件"""
        config_file = self.data_directory / "configs" / f"{module_id}.yaml"
        
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                return yaml.safe_load(f)
        
        # 返回默認配置
        return {
            "enabled": True,
            "log_level": "INFO"
        }
    
    async def discover_plugins(self, plugin_dir: Path) -> List[str]:
        """自動發現插件
        
        掃描指定目錄下的 *_plugin.py 文件
        
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
                logger.error(f"Failed to discover plugin {plugin_file}: {e}")
        
        logger.info(f"✅ Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def _is_plugin_class(self, obj) -> bool:
        """檢查是否為有效的插件類"""
        try:
            return (
                inspect.isclass(obj) and
                issubclass(obj, AIModulePlugin) and
                obj is not AIModulePlugin  # 排除基類本身
            )
        except TypeError:
            return False
    
    def get_plugin(self, module_id: str) -> Optional[AIModulePlugin]:
        """獲取已註冊的插件"""
        return self.plugins.get(module_id)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已註冊插件"""
        plugins_info = []
        
        for plugin in self.plugins.values():
            try:
                is_healthy = asyncio.run(plugin.health_check())
                status = "active" if is_healthy else "unhealthy"
            except:
                status = "error"
            
            plugins_info.append({
                "module_id": plugin.module_id,
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
    
    async def shutdown_all(self):
        """優雅關閉所有插件"""
        logger.info("Shutting down all plugins...")
        
        for module_id, plugin in self.plugins.items():
            try:
                await plugin.shutdown()
                logger.info(f"✅ Plugin {module_id} shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down plugin {module_id}: {e}")
```

### 3. WeightManager 實現

```python
# services/core/aiva_core/plugin_system/weight_manager.py
import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class WeightManager:
    """AI 模型權重管理器
    
    功能：
    1. 本地權重存儲和版本管理
    2. 權重完整性驗證 (SHA256)
    3. 語義化版本控制
    4. 雲端同步 (可選)
    """
    
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = weights_dir / "registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"WeightManager initialized: {weights_dir}")
    
    def register_weights(
        self,
        module_id: str,
        version: str,
        weight_path: Path,
        metadata: Optional[Dict] = None
    ) -> bool:
        """註冊模組權重
        
        Args:
            module_id: 模組 ID (如 "bio_neuron")
            version: 版本號 (如 "v1.0.0")
            weight_path: 權重文件路徑
            metadata: 額外元數據 (作者、訓練時間等)
            
        Returns:
            註冊是否成功
        """
        try:
            if not weight_path.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_path}")
            
            logger.info(f"Registering weights: {module_id} v{version}")
            
            # 1. 計算 SHA256 校驗和
            checksum = self._calculate_checksum(weight_path)
            logger.info(f"Weight checksum: {checksum[:16]}...")
            
            # 2. 創建版本目錄
            target_dir = self.weights_dir / module_id / version
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. 複製權重文件
            target_file = target_dir / weight_path.name
            shutil.copy2(weight_path, target_file)
            logger.info(f"Weight file copied to: {target_file}")
            
            # 4. 生成配置文件
            config = {
                "module_id": module_id,
                "version": version,
                "weight_file": weight_path.name,
                "checksum": checksum,
                "size_mb": weight_path.stat().st_size / (1024 * 1024),
                "registered_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            config_file = target_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # 5. 生成元數據 YAML
            metadata_content = {
                "module": module_id,
                "version": version,
                "checksum": checksum,
                "description": metadata.get("description", ""),
                "author": metadata.get("author", "AIVA Team"),
                "training_date": metadata.get("training_date", ""),
                "architecture": metadata.get("architecture", ""),
                "parameters": metadata.get("parameters", 0),
                "metrics": metadata.get("metrics", {})
            }
            
            metadata_file = target_dir / "metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata_content, f, default_flow_style=False)
            
            # 6. 更新註冊表
            if module_id not in self.registry:
                self.registry[module_id] = {"versions": []}
            
            version_info = {
                "version": version,
                "path": str(target_file),
                "checksum": checksum,
                "size_mb": round(config["size_mb"], 2),
                "registered_at": config["registered_at"]
            }
            
            # 檢查是否已存在此版本
            existing_versions = [v["version"] for v in self.registry[module_id]["versions"]]
            if version in existing_versions:
                # 更新現有版本
                for i, v in enumerate(self.registry[module_id]["versions"]):
                    if v["version"] == version:
                        self.registry[module_id]["versions"][i] = version_info
                        break
            else:
                # 添加新版本
                self.registry[module_id]["versions"].append(version_info)
            
            self._save_registry()
            
            # 7. 更新 latest 符號鏈接
            self._update_latest_link(module_id, version)
            
            logger.info(f"✅ Weights registered successfully: {module_id} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register weights: {e}")
            return False
    
    def get_weights(
        self,
        module_id: str,
        version: str = "latest"
    ) -> Optional[Path]:
        """獲取模組權重路徑
        
        Args:
            module_id: 模組 ID
            version: 版本號 (default: "latest")
            
        Returns:
            權重文件路徑，如果不存在則返回 None
        """
        try:
            weight_dir = self.weights_dir / module_id / version
            
            if not weight_dir.exists():
                logger.warning(f"Weight directory not found: {weight_dir}")
                return None
            
            # 查找權重文件 (支持多種格式)
            for ext in ['.safetensors', '.pt', '.pth', '.onnx', '.h5']:
                weight_files = list(weight_dir.glob(f"*{ext}"))
                if weight_files:
                    logger.info(f"Found weight file: {weight_files[0]}")
                    return weight_files[0]
            
            logger.warning(f"No weight file found in {weight_dir}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting weights: {e}")
            return None
    
    def verify_weights(self, module_id: str, version: str = "latest") -> bool:
        """驗證權重完整性
        
        Args:
            module_id: 模組 ID
            version: 版本號
            
        Returns:
            驗證是否通過
        """
        try:
            weight_path = self.get_weights(module_id, version)
            if not weight_path:
                return False
            
            # 計算當前校驗和
            current_checksum = self._calculate_checksum(weight_path)
            
            # 讀取註冊時的校驗和
            config_file = weight_path.parent / "config.json"
            with open(config_file) as f:
                config = json.load(f)
                registered_checksum = config["checksum"]
            
            if current_checksum == registered_checksum:
                logger.info(f"✅ Weight integrity verified: {module_id} v{version}")
                return True
            else:
                logger.error(f"❌ Weight integrity check failed: {module_id} v{version}")
                logger.error(f"Expected: {registered_checksum[:16]}...")
                logger.error(f"Got: {current_checksum[:16]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying weights: {e}")
            return False
    
    def list_weights(self, module_id: Optional[str] = None) -> List[Dict]:
        """列出所有權重
        
        Args:
            module_id: 如果指定，只列出該模組的權重
            
        Returns:
            權重信息列表
        """
        weights = []
        
        if module_id:
            # 只列出指定模組
            if module_id in self.registry:
                for version_info in self.registry[module_id]["versions"]:
                    weights.append({
                        "module_id": module_id,
                        **version_info
                    })
        else:
            # 列出所有模組
            for mod_id, mod_data in self.registry.items():
                for version_info in mod_data["versions"]:
                    weights.append({
                        "module_id": mod_id,
                        **version_info
                    })
        
        return weights
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """計算文件 SHA256 校驗和"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _load_registry(self) -> Dict:
        """載入權重註冊表"""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """保存權重註冊表"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _update_latest_link(self, module_id: str, version: str):
        """更新 latest 符號鏈接"""
        try:
            latest_link = self.weights_dir / module_id / "latest"
            
            # 移除舊鏈接
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            # 創建新鏈接
            latest_link.symlink_to(version, target_is_directory=True)
            logger.info(f"Updated latest link: {module_id}/latest -> {version}")
            
        except Exception as e:
            logger.warning(f"Failed to update latest link: {e}")
```

### 4. 示例插件: BioNeuronPlugin

```python
# services/core/aiva_core/plugins/bio_neuron_plugin.py
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any

from ..plugin_system.base_plugin import AIModulePlugin, AITask, AIResult, AITaskType

logger = logging.getLogger(__name__)


class BioNeuronPlugin(AIModulePlugin):
    """BioNeuron AI 插件
    
    功能：
    - 代碼分析和漏洞檢測
    - 攻擊路徑規劃
    - 決策推理
    """
    
    def __init__(self):
        self.model = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @property
    def module_id(self) -> str:
        return "bio_neuron"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "analyze_code",
            "detect_vulnerabilities",
            "plan_attack_path",
            "make_decision",
            "learn_from_experience"
        ]
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化 BioNeuron"""
        try:
            self.config = config
            
            # 初始化模型架構 (5M 參數)
            from services.core.aiva_core.cognitive_core.bio_neuron_core import BioNeuronCore
            
            self.model = BioNeuronCore(
                input_dim=config.get("input_dim", 768),
                hidden_dim=config.get("hidden_dim", 2048),
                output_dim=config.get("output_dim", 512),
                num_layers=config.get("num_layers", 4)
            )
            
            self.model.to(self.device)
            
            logger.info(f"✅ BioNeuron initialized on {self.device}")
            logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BioNeuron: {e}")
            return False
    
    async def load_weights(self, weight_path: Path) -> bool:
        """載入 BioNeuron 權重"""
        try:
            logger.info(f"Loading BioNeuron weights from {weight_path}")
            
            # 使用 safetensors 載入權重
            from safetensors.torch import load_file
            
            state_dict = load_file(str(weight_path))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # 驗證載入
            param_count = sum(p.numel() for p in self.model.parameters())
            weight_size_mb = weight_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"✅ BioNeuron weights loaded successfully")
            logger.info(f"   Parameters: {param_count:,}")
            logger.info(f"   Weight size: {weight_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行 AI 任務"""
        import time
        start_time = time.time()
        
        try:
            if task.task_type == AITaskType.ANALYZE:
                result_data = await self._analyze_code(task.parameters)
            elif task.task_type == AITaskType.SCAN:
                result_data = await self._detect_vulnerabilities(task.parameters)
            else:
                result_data = {"message": f"Task type {task.task_type} not implemented"}
            
            execution_time = time.time() - start_time
            
            return AIResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                metrics={"inference_time_ms": execution_time * 1000}
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return AIResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _analyze_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """分析代碼"""
        code = parameters.get("code", "")
        
        # 簡化示例：實際應該使用模型推理
        with torch.no_grad():
            # TODO: 實現完整的代碼分析邏輯
            pass
        
        return {
            "analysis": "Code analysis complete",
            "vulnerabilities_found": 3,
            "risk_level": "medium"
        }
    
    async def _detect_vulnerabilities(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """檢測漏洞"""
        target = parameters.get("target", "")
        
        # 簡化示例：實際應該使用模型推理
        return {
            "target": target,
            "vulnerabilities": [
                {"type": "XSS", "severity": "high", "confidence": 0.87},
                {"type": "SQLi", "severity": "critical", "confidence": 0.92}
            ]
        }
    
    async def health_check(self) -> bool:
        """健康檢查"""
        try:
            if self.model is None:
                return False
            
            # 測試推理
            with torch.no_grad():
                test_input = torch.randn(1, self.config.get("input_dim", 768)).to(self.device)
                _ = self.model(test_input)
            
            return True
            
        except:
            return False
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("BioNeuron plugin shut down")
```

---

## 啟動腳本

### 1. 權重註冊腳本

```python
# scripts/register_weights.py
#!/usr/bin/env python3
"""
權重註冊腳本

用法:
    python scripts/register_weights.py \\
        --module bio_neuron \\
        --version v1.0.0 \\
        --weight-file /path/to/model.safetensors \\
        --description "Initial BioNeuron weights"
"""

import argparse
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.core.aiva_core.plugin_system.weight_manager import WeightManager


def main():
    parser = argparse.ArgumentParser(description="Register AI model weights")
    parser.add_argument("--module", required=True, help="Module ID (e.g., bio_neuron)")
    parser.add_argument("--version", required=True, help="Version (e.g., v1.0.0)")
    parser.add_argument("--weight-file", required=True, help="Path to weight file")
    parser.add_argument("--description", default="", help="Weight description")
    parser.add_argument("--author", default="AIVA Team", help="Author name")
    parser.add_argument("--parameters", type=int, default=0, help="Number of parameters")
    parser.add_argument("--weights-dir", default="data/weights", help="Weights directory")
    
    args = parser.parse_args()
    
    # 初始化 WeightManager
    weights_dir = Path(args.weights_dir)
    weight_manager = WeightManager(weights_dir)
    
    # 準備元數據
    metadata = {
        "description": args.description,
        "author": args.author,
        "parameters": args.parameters
    }
    
    # 註冊權重
    success = weight_manager.register_weights(
        module_id=args.module,
        version=args.version,
        weight_path=Path(args.weight_file),
        metadata=metadata
    )
    
    if success:
        print(f"✅ Weights registered successfully: {args.module} v{args.version}")
        print(f"   Path: {weights_dir / args.module / args.version}")
        
        # 驗證完整性
        if weight_manager.verify_weights(args.module, args.version):
            print("✅ Weight integrity verified")
        else:
            print("❌ Weight integrity check failed!")
            sys.exit(1)
    else:
        print("❌ Failed to register weights")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 2. 快速啟動腳本

```powershell
# scripts/start_aiva_ai.ps1
# AIVA AI 系統快速啟動腳本

Write-Host "🚀 Starting AIVA AI System..." -ForegroundColor Cyan

# 1. 檢查 Python 環境
Write-Host "Checking Python environment..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    exit 1
}

# 2. 檢查依賴
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$required_packages = @("torch", "fastapi", "uvicorn", "safetensors", "pyyaml")

foreach ($package in $required_packages) {
    python -c "import $package" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing $package..." -ForegroundColor Yellow
        pip install $package
    }
}

# 3. 檢查權重文件
Write-Host "Checking weight files..." -ForegroundColor Yellow
$weights_dir = "data/weights"

if (-Not (Test-Path $weights_dir)) {
    Write-Host "Creating weights directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $weights_dir | Out-Null
}

# 檢查 BioNeuron 權重
$bio_neuron_weights = "$weights_dir/bio_neuron/latest"
if (-Not (Test-Path $bio_neuron_weights)) {
    Write-Host "⚠️  BioNeuron weights not found!" -ForegroundColor Yellow
    Write-Host "   Please register weights using: python scripts/register_weights.py" -ForegroundColor Yellow
}

# 4. 啟動 FastAPI 服務
Write-Host "Starting FastAPI service..." -ForegroundColor Green
Write-Host ""

python -m uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 測試計劃

### 單元測試示例

```python
# tests/unit/test_module_registry.py
import pytest
import asyncio
from pathlib import Path
from services.core.aiva_core.plugin_system.module_registry import ModuleRegistry
from services.core.aiva_core.plugin_system.base_plugin import AIModulePlugin, AITask, AIResult

class MockPlugin(AIModulePlugin):
    """測試用的模擬插件"""
    
    @property
    def module_id(self) -> str:
        return "mock_plugin"
    
    @property
    def capabilities(self) -> List[str]:
        return ["test_capability"]
    
    async def initialize(self, config):
        return True
    
    async def execute_task(self, task):
        return AIResult(success=True, data={"message": "test"})
    
    async def health_check(self):
        return True
    
    async def shutdown(self):
        pass

@pytest.mark.asyncio
async def test_register_plugin():
    """測試插件註冊"""
    registry = ModuleRegistry(Path("./test_data"))
    plugin = MockPlugin()
    
    success = await registry.register_plugin(plugin)
    
    assert success == True
    assert "mock_plugin" in registry.plugins
    assert registry.get_plugin("mock_plugin") is not None

@pytest.mark.asyncio
async def test_plugin_capabilities():
    """測試插件能力查詢"""
    registry = ModuleRegistry(Path("./test_data"))
    plugin = MockPlugin()
    
    await registry.register_plugin(plugin)
    
    plugins = registry.get_plugins_by_capability("test_capability")
    assert len(plugins) == 1
    assert plugins[0].module_id == "mock_plugin"
```

---

## 下一步行動

1. **立即可做**:
   - 創建 `services/core/aiva_core/plugin_system/` 目錄
   - 實現 `base_plugin.py` 接口定義
   - 實現 `module_registry.py` 註冊中心
   - 實現 `weight_manager.py` 權重管理器

2. **第一個插件**:
   - 改造 `bio_neuron` 為插件
   - 測試權重載入
   - 驗證健康檢查

3. **測試驗證**:
   - 編寫單元測試
   - 手動測試註冊流程
   - 驗證權重完整性

4. **文檔完善**:
   - 插件開發指南
   - API 文檔
   - 部署文檔
