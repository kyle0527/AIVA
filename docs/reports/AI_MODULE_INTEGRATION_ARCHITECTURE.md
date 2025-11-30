# AIVA AI 模組整合架構設計

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture v2.0](https://img.shields.io/badge/Architecture-v2.0-brightgreen.svg)](https://github.com/)
[![Plugin System](https://img.shields.io/badge/Plugin%20System-Ready-blue.svg)](https://github.com/)

> **目標**: 將六大 AI 模組、權重系統、指揮架構整合為統一的插件化系統，支援未來能力擴展  
> **版本**: v2.0 | **最後更新**: 2025年11月29日

## 📑 目錄

- [執行摘要](#執行摘要)
- [1. 整體架構概覽](#1-整體架構概覽)
- [2. 核心設計原則](#2-核心設計原則)
  - [2.1 單一事實來源 (Single Source of Truth)](#21-單一事實來源-single-source-of-truth)
  - [2.2 插件化設計 (Plugin Architecture)](#22-插件化設計-plugin-architecture)
  - [2.3 分層指揮架構 (Hierarchical Command)](#23-分層指揮架構-hierarchical-command)
- [3. 六大模組插件化改造](#3-六大模組插件化改造)
  - [3.1 模組對應表](#31-模組對應表)
  - [3.2 插件註冊機制 (Dynamic Registration)](#32-插件註冊機制-dynamic-registration)
- [4. 權重管理系統](#4-權重管理系統)
  - [4.1 權重存儲結構](#41-權重存儲結構)
  - [4.2 權重管理器 (Weight Manager)](#42-權重管理器-weight-manager)
  - [4.3 權重載入流程](#43-權重載入流程-參考-ray-serve)
- [5. AI 指揮系統升級](#5-ai-指揮系統升級)
  - [5.1 AICommander V2 架構](#51-aicommander-v2-架構)
  - [5.2 領域協調器 (Domain Coordinators)](#52-領域協調器-domain-coordinators)
- [6. Integration Module 作為數據中心](#6-integration-module-作為數據中心)
  - [6.1 數據流架構](#61-數據流架構)
  - [6.2 統一數據管理器升級](#62-統一數據管理器升級)
- [7. 未來擴展機制](#7-未來擴展機制)
  - [7.1 新能力添加流程](#71-新能力添加流程)
  - [7.2 版本兼容性管理](#72-版本兼容性管理)
- [8. 部署和啟動流程](#8-部署和啟動流程)
  - [8.1 FastAPI Lifespan 整合](#81-fastapi-lifespan-整合)
  - [8.2 啟動命令](#82-啟動命令)
- [9. 參考架構和最佳實踐](#9-參考架構和最佳實踐)
  - [9.1 Kubernetes 擴展模式](#91-kubernetes-擴展模式)
  - [9.2 Kubeflow Pipelines](#92-kubeflow-pipelines)
  - [9.3 Ray Serve](#93-ray-serve)
  - [9.4 FastAPI 最佳實踐](#94-fastapi-最佳實踐)
- [10. 實施檢查清單](#10-實施檢查清單)
- [11. 成功指標](#11-成功指標)
- [12. 總結](#12-總結)

---

## 執行摘要

本文檔設計了 AIVA 的完整 AI 模組整合架構，結合：
1. **插件化架構** (參考 Kubernetes、Kubeflow 設計模式)
2. **統一指揮層** (AICommander 升級)
3. **集中式數據中心** (Integration Module)
4. **權重管理系統** (本地 + 雲端)
5. **未來擴展機制** (動態註冊、版本管理)

---

## 1. 整體架構概覽

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI Commander (指揮層)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  任務分析 → 模組選擇 → 指令分發 → 結果整合 → 經驗學習    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────┐
│  Module Plugin  │  │   Module    │  │   Module    │
│   Registry      │◄─┤   Loader    │◄─┤  Validator  │
└─────────────────┘  └─────────────┘  └─────────────┘
         │
         ├─► 六大核心模組 (AI Plugins)
         │   ├─ core (Cognitive Core, Task Planning, Core Capabilities)
         │   ├─ scan (Passive Scanner, Active Scanner)
         │   ├─ features (XSS, SQLi, CSRF, Path Traversal, etc.)
         │   ├─ integration (AI Operation Recorder, Attack Path Analyzer)
         │   ├─ aiva_common (Shared Utilities)
         │   └─ external_learning (未來擴展)
         │
         └─► 權重系統 (Weight Management)
             ├─ 本地權重倉庫 (data/weights/)
             ├─ 雲端同步 (optional)
             └─ 版本控制 (semantic versioning)

┌─────────────────────────────────────────────────────────────────┐
│              Integration Module (數據中心)                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  • AI Operation Recorder (操作記錄)                     │    │
│  │  • Experience Repository (經驗存儲)                     │    │
│  │  • Attack Path Storage (攻擊路徑)                       │    │
│  │  • Training Dataset Manager (訓練數據集)                │    │
│  │  • Unified Data Manager (統一數據管理)                  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心設計原則

### 2.1 單一事實來源 (Single Source of Truth)
- **Integration Module** 是所有數據的唯一真相來源
- 所有 AI 模組通過 Integration Module 存取數據
- 消除數據分散和不一致問題

### 2.2 插件化設計 (Plugin Architecture)
參考 **Kubernetes 擴展模式**:
```python
# 統一插件接口 (參考 Kubernetes Device Plugin)
class AIModulePlugin(Protocol):
    """AI 模組插件接口 - 受 Kubernetes Plugin Pattern 啟發"""
    
    @property
    def module_id(self) -> str:
        """模組唯一標識符"""
        ...
    
    @property
    def capabilities(self) -> List[str]:
        """模組支援的能力列表"""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化模組"""
        ...
    
    async def load_weights(self, weight_path: Path) -> bool:
        """載入權重 (如果需要)"""
        ...
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行 AI 任務"""
        ...
    
    async def health_check(self) -> bool:
        """健康檢查"""
        ...
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        ...
```

### 2.3 分層指揮架構 (Hierarchical Command)
參考 **Kubeflow Pipelines 的 Orchestration**:
```
AI Commander (最高指揮)
    ↓
Domain Coordinators (領域協調器)
    ├─ Attack Coordinator (攻擊協調)
    ├─ Defense Coordinator (防禦協調)
    ├─ Analysis Coordinator (分析協調)
    └─ Training Coordinator (訓練協調)
        ↓
Module Plugins (執行單元)
```

---

## 3. 六大模組插件化改造

### 3.1 模組對應表

| 原始模組 | 插件化後角色 | 主要能力 | 權重需求 |
|---------|-------------|---------|---------|
| **core** | Core AI Plugin | 認知、任務規劃、決策 | ✅ BioNeuron 5M params |
| **scan** | Scanner Plugin | 被動/主動掃描、漏洞發現 | ❌ Rule-based |
| **features** | Feature Exploiter Plugin | XSS, SQLi, CSRF 等攻擊 | ❌ Rule-based |
| **integration** | Data Hub Plugin | 數據管理、經驗存儲 | ❌ Coordinator |
| **aiva_common** | Shared Library | 工具函數、共享資源 | ❌ Utilities |
| **external_learning** | Learning Plugin | 外部知識學習、RAG | ✅ Embedding Model |

### 3.2 插件註冊機制 (Dynamic Registration)

參考 **FastAPI Lifespan Pattern + Kubernetes Admission Control**:

```python
# services/core/aiva_core/plugin_system/module_registry.py
from typing import Dict, List, Optional
from pathlib import Path
import importlib
import inspect

class ModuleRegistry:
    """AI 模組註冊中心 - 參考 Kubernetes API Aggregation"""
    
    def __init__(self, data_directory: Path):
        self.data_directory = data_directory
        self.plugins: Dict[str, AIModulePlugin] = {}
        self.coordinators: Dict[str, BaseCoordinator] = {}
        
    async def register_plugin(
        self, 
        plugin: AIModulePlugin,
        weight_path: Optional[Path] = None
    ) -> bool:
        """註冊 AI 模組插件
        
        Args:
            plugin: 模組插件實例
            weight_path: 權重文件路徑 (如果需要)
            
        Returns:
            註冊是否成功
        """
        try:
            # 1. 驗證插件接口
            if not self._validate_plugin(plugin):
                raise ValueError(f"Plugin {plugin.module_id} 不符合接口規範")
            
            # 2. 初始化插件
            config = self._load_plugin_config(plugin.module_id)
            if not await plugin.initialize(config):
                raise RuntimeError(f"Plugin {plugin.module_id} 初始化失敗")
            
            # 3. 載入權重 (如果需要)
            if weight_path and hasattr(plugin, 'load_weights'):
                if not await plugin.load_weights(weight_path):
                    logger.warning(f"Plugin {plugin.module_id} 權重載入失敗")
            
            # 4. 註冊到註冊表
            self.plugins[plugin.module_id] = plugin
            
            logger.info(f"✅ Plugin registered: {plugin.module_id}")
            logger.info(f"   Capabilities: {plugin.capabilities}")
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin registration failed: {e}")
            return False
    
    def _validate_plugin(self, plugin: AIModulePlugin) -> bool:
        """驗證插件是否實現必需接口"""
        required_methods = [
            'module_id', 'capabilities', 'initialize', 
            'execute_task', 'health_check', 'shutdown'
        ]
        return all(hasattr(plugin, method) for method in required_methods)
    
    async def discover_plugins(self, plugin_dir: Path) -> List[str]:
        """自動發現插件 - 參考 Kubernetes Device Plugin Discovery"""
        discovered = []
        
        for plugin_file in plugin_dir.glob("*_plugin.py"):
            try:
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找實現 AIModulePlugin 的類
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_plugin_class(obj):
                        plugin = obj()
                        if await self.register_plugin(plugin):
                            discovered.append(plugin.module_id)
                            
            except Exception as e:
                logger.error(f"Failed to discover plugin {plugin_file}: {e}")
        
        return discovered
    
    def get_plugin(self, module_id: str) -> Optional[AIModulePlugin]:
        """獲取已註冊的插件"""
        return self.plugins.get(module_id)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已註冊插件"""
        return [
            {
                "module_id": plugin.module_id,
                "capabilities": plugin.capabilities,
                "status": "active" if asyncio.run(plugin.health_check()) else "unhealthy"
            }
            for plugin in self.plugins.values()
        ]
```

---

## 4. 權重管理系統

### 4.1 權重存儲結構

```
data/
└── weights/
    ├── bio_neuron/
    │   ├── v1.0.0/
    │   │   ├── model.safetensors  (5M parameters)
    │   │   ├── config.json
    │   │   └── metadata.yaml
    │   ├── v1.1.0/
    │   └── latest -> v1.1.0/
    ├── embeddings/
    │   ├── sentence-transformers-v2/
    │   └── custom-embeddings-v1/
    └── registry.json  (權重元數據註冊表)
```

### 4.2 權重管理器 (Weight Manager)

參考 **HuggingFace Model Hub + TensorFlow Serving**:

```python
# services/core/aiva_core/plugin_system/weight_manager.py
from pathlib import Path
import hashlib
import yaml
from typing import Dict, Optional

class WeightManager:
    """AI 模組權重管理器
    
    功能：
    1. 本地權重存儲和版本管理
    2. 權重完整性驗證 (SHA256)
    3. 雲端同步 (optional, 可用 S3/Azure Blob)
    4. 語義化版本控制 (semantic versioning)
    """
    
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = weights_dir / "registry.json"
        self.registry = self._load_registry()
    
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
            metadata: 額外元數據
        """
        try:
            # 1. 計算 SHA256 校驗和
            checksum = self._calculate_checksum(weight_path)
            
            # 2. 複製權重到版本目錄
            target_dir = self.weights_dir / module_id / version
            target_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            target_file = target_dir / weight_path.name
            shutil.copy2(weight_path, target_file)
            
            # 3. 保存元數據
            metadata_file = target_dir / "metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump({
                    "module_id": module_id,
                    "version": version,
                    "checksum": checksum,
                    "size_mb": weight_path.stat().st_size / (1024 * 1024),
                    "registered_at": datetime.now().isoformat(),
                    **metadata or {}
                }, f)
            
            # 4. 更新註冊表
            if module_id not in self.registry:
                self.registry[module_id] = {"versions": []}
            
            self.registry[module_id]["versions"].append({
                "version": version,
                "path": str(target_file),
                "checksum": checksum,
                "registered_at": datetime.now().isoformat()
            })
            
            # 5. 更新 latest 符號鏈接
            latest_link = self.weights_dir / module_id / "latest"
            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(version, target_is_directory=True)
            
            self._save_registry()
            
            logger.info(f"✅ Weights registered: {module_id} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register weights: {e}")
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
        weight_dir = self.weights_dir / module_id / version
        if not weight_dir.exists():
            logger.error(f"Weights not found: {module_id} v{version}")
            return None
        
        # 查找 .safetensors 或 .pt 文件
        for ext in ['.safetensors', '.pt', '.pth', '.onnx']:
            weight_files = list(weight_dir.glob(f"*{ext}"))
            if weight_files:
                return weight_files[0]
        
        return None
    
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
            import json
            with open(self.registry_file) as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """保存權重註冊表"""
        import json
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
```

### 4.3 權重載入流程 (參考 Ray Serve)

```python
# 在插件初始化時自動載入權重
class BioNeuronPlugin(AIModulePlugin):
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化並載入權重"""
        weight_manager = WeightManager(Path("data/weights"))
        
        # 1. 獲取權重路徑
        weight_path = weight_manager.get_weights(
            module_id="bio_neuron",
            version=config.get("version", "latest")
        )
        
        if not weight_path:
            logger.error("BioNeuron weights not found!")
            return False
        
        # 2. 載入權重到模型
        await self.load_weights(weight_path)
        
        logger.info(f"✅ BioNeuron initialized with weights from {weight_path}")
        return True
    
    async def load_weights(self, weight_path: Path) -> bool:
        """載入 BioNeuron 權重"""
        try:
            from safetensors.torch import load_file
            
            # 載入權重到 PyTorch 模型
            state_dict = load_file(str(weight_path))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logger.info(f"✅ Weights loaded: {weight_path.stat().st_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False
```

---

## 5. AI 指揮系統升級

### 5.1 AICommander V2 架構

參考 **Kubernetes Controller Pattern + Kubeflow Orchestration**:

```python
# services/core/aiva_core/task_planning/ai_commander_v2.py
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

class AITaskType(Enum):
    """AI 任務類型"""
    ATTACK_PLANNING = "attack_planning"
    VULNERABILITY_DETECTION = "vulnerability_detection"
    EXPLOIT_GENERATION = "exploit_generation"
    DEFENSE_ANALYSIS = "defense_analysis"
    MODEL_TRAINING = "model_training"
    EXPERIENCE_LEARNING = "experience_learning"

class AICommanderV2:
    """AI 指揮官 V2 - 統一指揮所有 AI 模組
    
    設計參考：
    - Kubernetes Controller: 調和實際狀態與期望狀態
    - Kubeflow Pipelines: 複雜任務編排
    - Ray Serve: 分布式 AI 服務調度
    """
    
    def __init__(self, data_directory: Path):
        self.data_directory = data_directory
        
        # 1. 初始化插件註冊中心
        self.module_registry = ModuleRegistry(data_directory)
        
        # 2. 初始化權重管理器
        self.weight_manager = WeightManager(data_directory / "weights")
        
        # 3. 初始化領域協調器 (Domain Coordinators)
        self.coordinators = {
            "attack": AttackCoordinator(self.module_registry),
            "defense": DefenseCoordinator(self.module_registry),
            "analysis": AnalysisCoordinator(self.module_registry),
            "training": TrainingCoordinator(self.module_registry)
        }
        
        # 4. 連接 Integration Module (數據中心)
        from services.integration.aiva_integration.ai_operation_recorder import AIOperationRecorderV2
        from services.integration.aiva_integration.reception.experience_repository import ExperienceRepository
        
        self.operation_recorder = AIOperationRecorderV2(
            output_dir=str(data_directory / "operations")
        )
        self.experience_repo = ExperienceRepository(
            database_url="sqlite:///aiva_operations.sqlite"
        )
        
        logger.info("🎖️ AICommander V2 initialized with plugin system")
    
    async def startup(self):
        """啟動 AI Commander - 使用 FastAPI Lifespan Pattern"""
        logger.info("🚀 Starting AI Commander V2...")
        
        # 1. 自動發現和註冊插件
        discovered = await self.module_registry.discover_plugins(
            Path("services/plugins")
        )
        logger.info(f"✅ Discovered {len(discovered)} plugins: {discovered}")
        
        # 2. 手動註冊核心模組
        await self._register_core_modules()
        
        # 3. 健康檢查
        await self._health_check_all()
        
        logger.info("✅ AI Commander V2 ready")
    
    async def _register_core_modules(self):
        """註冊六大核心模組"""
        
        # 1. Core AI Plugin (需要權重)
        bio_neuron_plugin = BioNeuronPlugin()
        bio_neuron_weights = self.weight_manager.get_weights("bio_neuron", "latest")
        await self.module_registry.register_plugin(
            plugin=bio_neuron_plugin,
            weight_path=bio_neuron_weights
        )
        
        # 2. Scanner Plugin (無需權重)
        scanner_plugin = ScannerPlugin()
        await self.module_registry.register_plugin(scanner_plugin)
        
        # 3. Feature Exploiter Plugin (無需權重)
        exploiter_plugin = FeatureExploiterPlugin()
        await self.module_registry.register_plugin(exploiter_plugin)
        
        # 4. Data Hub Plugin (Integration Module)
        data_hub_plugin = DataHubPlugin()
        await self.module_registry.register_plugin(data_hub_plugin)
        
        # 5. Shared Library (aiva_common)
        # 不需要註冊為插件，直接作為 Python 模組導入
        
        # 6. Learning Plugin (需要 Embedding 權重)
        learning_plugin = LearningPlugin()
        embedding_weights = self.weight_manager.get_weights("embeddings", "latest")
        await self.module_registry.register_plugin(
            plugin=learning_plugin,
            weight_path=embedding_weights
        )
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行 AI 任務 - Kubernetes Controller 風格
        
        工作流程：
        1. 分析任務類型
        2. 選擇合適的領域協調器
        3. 協調器選擇並調用插件
        4. 整合結果
        5. 記錄操作和經驗
        """
        try:
            # 1. 記錄任務開始
            operation_id = self.operation_recorder.record_operation(
                command=f"execute_task:{task.task_type.value}",
                description=task.description,
                parameters=task.parameters,
                operation_type="ai_task"
            )
            
            # 2. 選擇協調器
            coordinator = self._select_coordinator(task.task_type)
            
            # 3. 執行任務 (協調器會自動選擇插件)
            result = await coordinator.execute(task)
            
            # 4. 記錄結果
            self.operation_recorder.record_operation(
                command=f"task_completed:{task.task_type.value}",
                description=f"Task {operation_id} completed",
                result=result.to_dict(),
                duration=result.execution_time,
                success=result.success
            )
            
            # 5. 保存經驗 (用於未來訓練)
            if result.success:
                await self.experience_repo.save_experience(
                    plan_id=operation_id,
                    attack_type=task.task_type.value,
                    execution_trace=result.trace,
                    metrics=result.metrics,
                    feedback=result.feedback
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return AIResult(success=False, error=str(e))
    
    def _select_coordinator(self, task_type: AITaskType) -> BaseCoordinator:
        """根據任務類型選擇協調器"""
        mapping = {
            AITaskType.ATTACK_PLANNING: "attack",
            AITaskType.VULNERABILITY_DETECTION: "analysis",
            AITaskType.EXPLOIT_GENERATION: "attack",
            AITaskType.DEFENSE_ANALYSIS: "defense",
            AITaskType.MODEL_TRAINING: "training",
        }
        coordinator_name = mapping.get(task_type, "analysis")
        return self.coordinators[coordinator_name]
    
    async def _health_check_all(self):
        """健康檢查所有插件"""
        plugins = self.module_registry.list_plugins()
        for plugin_info in plugins:
            status = "✅" if plugin_info["status"] == "active" else "❌"
            logger.info(f"{status} {plugin_info['module_id']}: {plugin_info['capabilities']}")
```

### 5.2 領域協調器 (Domain Coordinators)

參考 **Kubernetes Operator Pattern**:

```python
# services/core/aiva_core/task_planning/coordinators/attack_coordinator.py
class AttackCoordinator(BaseCoordinator):
    """攻擊任務協調器
    
    負責協調掃描、漏洞利用等攻擊相關插件
    """
    
    async def execute(self, task: AITask) -> AIResult:
        """執行攻擊任務"""
        
        # 1. 獲取所需插件
        scanner = self.registry.get_plugin("scanner")
        exploiter = self.registry.get_plugin("exploiter")
        bio_neuron = self.registry.get_plugin("bio_neuron")
        
        # 2. 編排執行流程 (類似 Kubeflow Pipeline)
        # Step 1: 掃描階段
        scan_result = await scanner.execute_task(
            AITask(task_type="scan", target=task.target)
        )
        
        # Step 2: BioNeuron 分析漏洞
        analysis = await bio_neuron.execute_task(
            AITask(
                task_type="analyze_vulnerabilities",
                data=scan_result.vulnerabilities
            )
        )
        
        # Step 3: 生成利用代碼
        exploits = await exploiter.execute_task(
            AITask(
                task_type="generate_exploit",
                vulnerabilities=analysis.high_priority_vulns
            )
        )
        
        # 3. 整合結果
        return AIResult(
            success=True,
            data={
                "scan": scan_result,
                "analysis": analysis,
                "exploits": exploits
            },
            execution_time=scan_result.time + analysis.time + exploits.time
        )
```

---

## 6. Integration Module 作為數據中心

### 6.1 數據流架構

```
所有 AI 操作
    ↓
AI Operation Recorder (V2)
    ↓
┌──────────────────────────────────────┐
│     Integration Module (數據中心)     │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Experience Repository         │ │
│  │  (SQLite + 單一事實原則)       │ │
│  └────────────────────────────────┘ │
│           ↓        ↓        ↓        │
│    ┌─────────┐ ┌─────────┐ ┌──────┐│
│    │ Attack  │ │Training │ │ RAG  ││
│    │ Paths   │ │Datasets │ │  KB  ││
│    └─────────┘ └─────────┘ └──────┘│
└──────────────────────────────────────┘
           ↓
    統一查詢接口
           ↓
    AI 模組消費數據
```

### 6.2 統一數據管理器升級

```python
# services/integration/aiva_integration/unified_data_manager_v2.py
class UnifiedDataManagerV2:
    """統一數據管理器 V2
    
    作為 Integration Module 的核心，提供：
    1. 單一事實來源存儲
    2. 跨模組數據共享
    3. 訓練數據集管理
    4. 經驗檢索和查詢
    """
    
    def __init__(self, database_url: str):
        self.experience_repo = ExperienceRepository(database_url)
        self.attack_path_storage = AttackPathStorage(database_url)
        self.training_dataset_manager = TrainingDatasetManager(database_url)
    
    async def save_ai_operation(
        self,
        operation_id: str,
        operation_data: Dict[str, Any]
    ) -> bool:
        """保存 AI 操作記錄"""
        return await self.experience_repo.save_experience(
            plan_id=operation_id,
            **operation_data
        )
    
    async def query_experiences(
        self,
        attack_type: Optional[str] = None,
        min_score: float = 0.0,
        limit: int = 100
    ) -> List[Dict]:
        """查詢經驗記錄 - 供 AI 學習使用"""
        return await self.experience_repo.query_experiences(
            attack_type=attack_type,
            min_score=min_score,
            limit=limit
        )
    
    async def prepare_training_dataset(
        self,
        task_type: str,
        min_samples: int = 1000
    ) -> Path:
        """為 AI 訓練準備數據集"""
        experiences = await self.query_experiences(
            attack_type=task_type,
            limit=min_samples
        )
        
        return self.training_dataset_manager.create_dataset(
            task_type=task_type,
            experiences=experiences
        )
```

---

## 7. 未來擴展機制

### 7.1 新能力添加流程

```python
# 1. 實現插件接口
class NewCapabilityPlugin(AIModulePlugin):
    @property
    def module_id(self) -> str:
        return "new_capability"
    
    @property
    def capabilities(self) -> List[str]:
        return ["capability_a", "capability_b"]
    
    async def initialize(self, config: Dict) -> bool:
        # 初始化邏輯
        return True
    
    async def execute_task(self, task: AITask) -> AIResult:
        # 執行邏輯
        pass

# 2. 註冊插件 (自動或手動)
await ai_commander.module_registry.register_plugin(
    NewCapabilityPlugin()
)

# 3. 立即可用，無需修改核心代碼
```

### 7.2 版本兼容性管理

```yaml
# services/plugins/new_capability/metadata.yaml
plugin:
  id: new_capability
  version: 1.0.0
  api_version: v2  # 插件 API 版本
  min_aiva_version: 2.0.0
  
  dependencies:
    - bio_neuron>=1.0.0
    - scanner>=2.1.0
  
  capabilities:
    - capability_a
    - capability_b
  
  weights:
    required: true
    source: huggingface://aiva/new-capability-weights
    checksum: sha256:abcd1234...
```

---

## 8. 部署和啟動流程

### 8.1 FastAPI Lifespan 整合

參考 **FastAPI + Ray Serve 最佳實踐**:

```python
# services/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    
    # 啟動階段
    logger.info("🚀 Starting AIVA AI System...")
    
    # 1. 初始化 AI Commander
    ai_commander = AICommanderV2(data_directory=Path("data"))
    await ai_commander.startup()
    
    # 2. 註冊權重
    weight_manager = ai_commander.weight_manager
    
    # 如果本地沒有權重，從雲端下載
    if not weight_manager.get_weights("bio_neuron", "latest"):
        logger.info("Downloading BioNeuron weights...")
        # await download_weights_from_cloud()
    
    # 3. 健康檢查
    plugins = ai_commander.module_registry.list_plugins()
    logger.info(f"✅ {len(plugins)} plugins ready")
    
    # 將 commander 注入到 app state
    app.state.ai_commander = ai_commander
    
    yield  # 應用運行
    
    # 關閉階段
    logger.info("Shutting down AI Commander...")
    for plugin in ai_commander.module_registry.plugins.values():
        await plugin.shutdown()
    
    logger.info("✅ AIVA AI System stopped")

app = FastAPI(lifespan=lifespan)

@app.post("/api/ai/execute")
async def execute_ai_task(task: AITaskRequest):
    """執行 AI 任務端點"""
    ai_commander = app.state.ai_commander
    result = await ai_commander.execute_task(task.to_ai_task())
    return result.to_dict()
```

### 8.2 啟動命令

```powershell
# 1. 啟動 AIVA AI 服務
python -m services.api.main

# 2. 自動完成：
#    ✅ 發現和註冊所有插件
#    ✅ 載入必需的權重
#    ✅ 健康檢查
#    ✅ 啟動 API 服務

# 3. 測試 AI 指令
curl -X POST http://localhost:8000/api/ai/execute `
  -H "Content-Type: application/json" `
  -d '{
    "task_type": "attack_planning",
    "target": "http://example.com",
    "description": "Scan and exploit vulnerabilities"
  }'
```

---

## 9. 參考架構和最佳實踐

### 9.1 Kubernetes 擴展模式
- **Custom Resource Definitions**: 定義 AI 模組作為資源
- **Controller Pattern**: AICommander 作為控制器調和狀態
- **Admission Webhooks**: 插件驗證和註冊機制

### 9.2 Kubeflow Pipelines
- **DAG Orchestration**: 複雜任務編排 (掃描 → 分析 → 利用)
- **Pipeline Components**: 每個 AI 模組作為管道組件
- **Artifact Storage**: Integration Module 存儲中間結果

### 9.3 Ray Serve
- **Dynamic Model Loading**: 權重動態載入
- **Replica Management**: 插件實例管理
- **Request Routing**: 任務路由到合適的插件

### 9.4 FastAPI 最佳實踐
- **Lifespan Events**: 優雅啟動和關閉
- **Dependency Injection**: 插件依賴注入
- **Async/Await**: 非同步任務執行

---

## 10. 實施檢查清單

### Phase 1: 插件系統基礎設施 (1-2 週)
- [ ] 實現 `AIModulePlugin` 接口
- [ ] 實現 `ModuleRegistry` 註冊中心
- [ ] 實現 `WeightManager` 權重管理器
- [ ] 實現插件自動發現機制

### Phase 2: 核心模組插件化 (2-3 週)
- [ ] 改造 `core` 模組為 `BioNeuronPlugin`
- [ ] 改造 `scan` 模組為 `ScannerPlugin`
- [ ] 改造 `features` 模組為 `FeatureExploiterPlugin`
- [ ] 改造 `integration` 模組為 `DataHubPlugin`
- [ ] 實現 `LearningPlugin` (external_learning)

### Phase 3: AICommander V2 升級 (1-2 週)
- [ ] 實現 `AICommanderV2` 類
- [ ] 實現領域協調器 (AttackCoordinator, DefenseCoordinator, etc.)
- [ ] 整合 Integration Module 數據中心
- [ ] 實現任務編排和結果整合

### Phase 4: Integration Module 升級 (1 週)
- [ ] 升級 `UnifiedDataManagerV2`
- [ ] 升級 `AIOperationRecorderV2` (已完成)
- [ ] 實現 `TrainingDatasetManager`
- [ ] 統一數據查詢接口

### Phase 5: 測試和部署 (1-2 週)
- [ ] 單元測試 (每個插件)
- [ ] 集成測試 (完整任務流程)
- [ ] 性能測試 (並發任務)
- [ ] 健康檢查和監控
- [ ] 文檔和培訓

---

## 11. 成功指標

1. **模組化**: 所有 6 個模組成功轉換為插件
2. **權重管理**: BioNeuron 5M 參數正確載入
3. **指揮系統**: AICommander 可協調所有插件執行複雜任務
4. **數據中心**: Integration Module 成為單一數據來源
5. **可擴展性**: 新插件可在 15 分鐘內添加並運行
6. **性能**: 任務執行延遲 < 100ms (不含模型推理)
7. **穩定性**: 單插件故障不影響整體系統運行

---

## 12. 總結

本架構設計通過借鑒 **Kubernetes、Kubeflow、Ray Serve 和 FastAPI** 的最佳實踐，實現了：

✅ **插件化**: 所有 AI 模組統一為插件接口  
✅ **權重管理**: 語義化版本控制 + 完整性驗證  
✅ **統一指揮**: AICommander V2 協調所有組件  
✅ **數據中心**: Integration Module 單一事實來源  
✅ **未來擴展**: 動態註冊、版本兼容、自動發現  

這是一個**生產級、可擴展、容錯**的 AI 系統架構，為 AIVA 未來持續演進奠定堅實基礎。
