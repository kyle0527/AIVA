# 協調器引擎整合設計規範 (基於 AIVA Common 數據合約)

**文檔創建日期**: 2025-11-19  
**設計目標**: 協調器能自由搭配各語言引擎進行掃描,且能調整所有引擎參數  
**核心原則**: 基於數據合約的插件式架構,參數完全可配置

---

## 📋 執行摘要

本文檔基於 **aiva_common** 數據合約,設計一個支持 Python、TypeScript、Rust、Go 四引擎的統一協調器架構。

**關鍵特性**:
- ✅ **統一數據合約**: 所有引擎遵循 aiva_common Schema
- ✅ **參數完全可配置**: 每個引擎的所有參數都可動態調整
- ✅ **插件式架構**: 引擎作為可插拔組件
- ✅ **核心模組決策**: Core 模組決定使用哪些引擎和參數

---

## 🏗️ 架構設計原則

### 1. 基於 Abstract Factory 模式

參考: [Refactoring Guru - Abstract Factory](https://refactoring.guru/design-patterns/abstract-factory)

```
                  ┌─────────────────────────┐
                  │  EngineFactory          │
                  │  (Abstract Factory)     │
                  └───────────┬─────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ↓                     ↓                     ↓
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ PythonEngine  │     │ RustEngine    │     │ GoEngine      │
│ Factory       │     │ Factory       │     │ Factory       │
└───────────────┘     └───────────────┘     └───────────────┘
```

**優點**:
- 引擎間解耦
- 易於添加新引擎
- 統一接口管理

### 2. 基於數據合約的通訊

**AIVA Common Schema 作為統一語言**:

```python
# services/aiva_common/schemas/tasks.py

# Phase 0 啟動 (Rust 快速偵察)
class Phase0StartPayload(BaseModel):
    scan_id: str
    targets: list[HttpUrl]
    scope: ScanScope
    authentication: Authentication
    rate_limit: RateLimit
    custom_headers: dict[str, str]
    max_depth: int = 3
    timeout: int = 600

# Phase 1 啟動 (多引擎深度掃描)
class Phase1StartPayload(BaseModel):
    scan_id: str
    targets: list[HttpUrl]
    selected_engines: list[str]  # ["python", "rust", "go", "typescript"]
    strategy: str
    rate_limit: RateLimit
    max_depth: int = 5
    max_pages: int = 1000
    timeout: int = 1800
```

**所有引擎都接受相同的數據合約** → 參數一致性

---

## 🎯 設計方案: 配置驅動的引擎協調器

### 核心概念

**協調器不決定策略,只執行策略**

```
┌─────────────────────────────────────────────────────────┐
│                      Core 模組                           │
│  • 分析 Phase 0 結果                                      │
│  • 決定使用哪些引擎 (["python", "rust", "go"])            │
│  • 決定每個引擎的參數配置                                  │
│  • 生成 EngineCoordinationRequest                        │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              MultiEngineCoordinator                      │
│  • 接收 Core 的 EngineCoordinationRequest                 │
│  • 根據配置調用對應引擎                                    │
│  • 並行/串行執行                                          │
│  • 聚合結果返回給 Core                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 數據合約設計

### 1. 引擎配置 Schema

```python
# services/scan/coordinators/engine_schemas.py

from pydantic import BaseModel, Field
from typing import Literal, Any
from services.aiva_common.schemas import (
    RateLimit, 
    Authentication, 
    ScanScope
)

class EngineConfig(BaseModel):
    """單個引擎的配置
    
    這個 Schema 封裝了每個引擎可以調整的所有參數
    """
    
    engine_name: Literal["python", "typescript", "rust", "go"]
    enabled: bool = True
    
    # 掃描參數 (所有引擎通用)
    max_depth: int = Field(default=5, ge=1, le=10)
    max_pages: int = Field(default=1000, ge=10, le=10000)
    timeout: int = Field(default=1800, ge=60, le=7200)
    
    # 速率控制
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    
    # 認證配置
    authentication: Authentication = Field(default_factory=Authentication)
    
    # 引擎特定參數 (靈活擴展)
    engine_specific: dict[str, Any] = Field(default_factory=dict)
    
    # 示例:
    # Python Engine:
    #   engine_specific = {
    #       "enable_dynamic_rendering": True,
    #       "playwright_browser": "chromium",
    #       "screenshot_on_error": False
    #   }
    # 
    # Rust Engine:
    #   engine_specific = {
    #       "mode": "deep_analysis",
    #       "js_analysis_depth": 3,
    #       "memory_limit_mb": 512
    #   }
    #
    # Go Engine:
    #   engine_specific = {
    #       "enable_ssrf": True,
    #       "enable_cspm": True,
    #       "enable_sca": False,
    #       "ssrf_timeout": 60,
    #       "cspm_cloud_providers": ["aws", "azure"]
    #   }


class ExecutionStrategy(BaseModel):
    """執行策略配置"""
    
    mode: Literal["parallel", "sequential", "hybrid"] = "parallel"
    
    # 並行模式配置
    parallel_config: dict[str, Any] | None = Field(
        None,
        description="並行執行配置: {'max_concurrent': 3, 'timeout_per_engine': 600}"
    )
    
    # 串行模式配置
    sequential_config: dict[str, Any] | None = Field(
        None,
        description="串行執行順序: {'order': ['rust', 'python', 'go'], 'pass_results': True}"
    )
    
    # 混合模式配置
    hybrid_config: dict[str, Any] | None = Field(
        None,
        description="混合執行: {'parallel_group_1': ['rust', 'go'], 'then': ['python']}"
    )


class EngineCoordinationRequest(BaseModel):
    """協調器請求 - Core 模組發送給協調器的完整配置
    
    這是核心與協調器之間的數據合約
    """
    
    # 掃描基本資訊 (繼承自 Phase1StartPayload)
    scan_id: str
    targets: list[str]
    scope: ScanScope = Field(default_factory=ScanScope)
    
    # 引擎配置列表
    engine_configs: list[EngineConfig] = Field(
        ...,
        description="要使用的引擎及其配置"
    )
    
    # 執行策略
    execution_strategy: ExecutionStrategy = Field(
        default_factory=ExecutionStrategy
    )
    
    # 全局配置 (覆蓋單個引擎配置)
    global_config: dict[str, Any] = Field(
        default_factory=dict,
        description="全局配置,優先級高於引擎配置"
    )
    
    # Phase 0 結果 (可選,用於串行模式)
    phase0_result: dict[str, Any] | None = Field(
        None,
        description="Phase 0 結果,供引擎參考"
    )


class EngineExecutionResult(BaseModel):
    """單個引擎的執行結果"""
    
    engine_name: str
    status: Literal["success", "failed", "timeout", "skipped"]
    execution_time: float
    
    # 發現的資產 (符合 aiva_common.schemas.Asset)
    assets: list[dict] = Field(default_factory=list)
    
    # 引擎元數據
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # 錯誤資訊
    error: str | None = None


class EngineCoordinationResponse(BaseModel):
    """協調器響應 - 返回給 Core 模組
    
    符合 Phase1CompletedPayload 格式
    """
    
    scan_id: str
    status: Literal["success", "partial", "failed"]
    execution_time: float
    
    # 所有引擎結果
    engine_results: list[EngineExecutionResult]
    
    # 聚合後的資產 (已去重)
    aggregated_assets: list[dict] = Field(default_factory=list)
    
    # 統計資訊
    summary: dict[str, Any] = Field(default_factory=dict)
```

---

### 2. 引擎工廠介面

```python
# services/scan/coordinators/engine_factory.py

from abc import ABC, abstractmethod
from typing import Protocol
import asyncio


class IEngine(Protocol):
    """引擎接口 (協議)
    
    所有引擎必須實現這個接口
    """
    
    async def execute(
        self, 
        config: EngineConfig,
        targets: list[str],
        context: dict[str, Any]
    ) -> EngineExecutionResult:
        """執行掃描
        
        Args:
            config: 引擎配置
            targets: 目標列表
            context: 上下文信息 (Phase 0 結果等)
        
        Returns:
            引擎執行結果
        """
        ...
    
    async def health_check(self) -> bool:
        """健康檢查"""
        ...
    
    def get_capabilities(self) -> dict[str, Any]:
        """返回引擎能力"""
        ...


class EngineFactory(ABC):
    """引擎工廠抽象類"""
    
    @abstractmethod
    def create_engine(self, engine_name: str) -> IEngine:
        """創建引擎實例
        
        Args:
            engine_name: 引擎名稱 ("python", "rust", "go", "typescript")
        
        Returns:
            引擎實例
        """
        pass
    
    @abstractmethod
    def get_available_engines(self) -> list[str]:
        """獲取可用引擎列表"""
        pass


class DefaultEngineFactory(EngineFactory):
    """默認引擎工廠實現"""
    
    def __init__(self):
        self._engines: dict[str, type[IEngine]] = {}
        self._register_default_engines()
    
    def _register_default_engines(self):
        """註冊默認引擎"""
        from .adapters import (
            PythonEngineAdapter,
            RustEngineAdapter,
            GoEngineAdapter,
            TypeScriptEngineAdapter
        )
        
        self._engines = {
            "python": PythonEngineAdapter,
            "rust": RustEngineAdapter,
            "go": GoEngineAdapter,
            "typescript": TypeScriptEngineAdapter
        }
    
    def create_engine(self, engine_name: str) -> IEngine:
        """創建引擎實例"""
        if engine_name not in self._engines:
            raise ValueError(f"Unknown engine: {engine_name}")
        
        return self._engines[engine_name]()
    
    def get_available_engines(self) -> list[str]:
        """獲取可用引擎列表"""
        return list(self._engines.keys())
    
    def register_engine(self, name: str, engine_class: type[IEngine]):
        """動態註冊新引擎"""
        self._engines[name] = engine_class
```

---

### 3. 引擎適配器實現

```python
# services/scan/coordinators/adapters/python_adapter.py

from ..engine_factory import IEngine, EngineConfig, EngineExecutionResult
from services.scan.engines.python_engine import ScanOrchestrator
from services.aiva_common.schemas import Phase1StartPayload, Asset
import time


class PythonEngineAdapter(IEngine):
    """Python 引擎適配器
    
    將 Python Engine 適配到統一的 IEngine 接口
    """
    
    def __init__(self):
        self.orchestrator = ScanOrchestrator()
    
    async def execute(
        self, 
        config: EngineConfig,
        targets: list[str],
        context: dict[str, Any]
    ) -> EngineExecutionResult:
        """執行 Python 引擎掃描"""
        start_time = time.time()
        
        try:
            # 構建 Phase1StartPayload (符合 aiva_common 合約)
            request = Phase1StartPayload(
                scan_id=context.get("scan_id", "unknown"),
                targets=targets,
                selected_engines=["python"],
                strategy=context.get("strategy", "deep"),
                rate_limit=config.rate_limit,
                authentication=config.authentication,
                max_depth=config.max_depth,
                max_pages=config.max_pages,
                timeout=config.timeout
            )
            
            # 調用 Python Engine
            result = await self.orchestrator.execute_phase1(request)
            
            # 轉換為 EngineExecutionResult
            return EngineExecutionResult(
                engine_name="python",
                status="success" if result.status == "success" else "failed",
                execution_time=time.time() - start_time,
                assets=[asset.model_dump() for asset in result.assets],
                metadata={
                    "urls_found": result.summary.urls_found,
                    "forms_found": result.summary.forms_found,
                    "enable_dynamic_rendering": config.engine_specific.get(
                        "enable_dynamic_rendering", True
                    )
                }
            )
            
        except Exception as exc:
            return EngineExecutionResult(
                engine_name="python",
                status="failed",
                execution_time=time.time() - start_time,
                error=str(exc)
            )
    
    async def health_check(self) -> bool:
        """健康檢查"""
        try:
            # 簡單檢查 orchestrator 是否可用
            return self.orchestrator is not None
        except:
            return False
    
    def get_capabilities(self) -> dict[str, Any]:
        """返回引擎能力"""
        return {
            "engine": "python",
            "version": "2.0",
            "features": [
                "static_crawling",
                "dynamic_rendering", 
                "form_discovery",
                "api_detection"
            ],
            "configurable_params": [
                "max_depth",
                "max_pages",
                "enable_dynamic_rendering",
                "playwright_browser",
                "screenshot_on_error"
            ]
        }


# services/scan/coordinators/adapters/rust_adapter.py

class RustEngineAdapter(IEngine):
    """Rust 引擎適配器"""
    
    def __init__(self):
        from services.scan.engines.rust_engine.python_bridge import (
            RustInfoGatherer
        )
        self.gatherer = RustInfoGatherer()
    
    async def execute(
        self, 
        config: EngineConfig,
        targets: list[str],
        context: dict[str, Any]
    ) -> EngineExecutionResult:
        """執行 Rust 引擎掃描"""
        start_time = time.time()
        
        try:
            # 檢查可用性
            if not self.gatherer.check_availability():
                return EngineExecutionResult(
                    engine_name="rust",
                    status="skipped",
                    execution_time=0,
                    error="Rust binary not available"
                )
            
            # 準備配置
            rust_config = {
                "mode": config.engine_specific.get("mode", "deep_analysis"),
                "timeout": config.timeout,
                "max_depth": config.max_depth,
                "js_analysis_depth": config.engine_specific.get(
                    "js_analysis_depth", 3
                ),
                "memory_limit_mb": config.engine_specific.get(
                    "memory_limit_mb", 512
                )
            }
            
            # 並行掃描所有目標
            all_assets = []
            for target in targets:
                result = await asyncio.to_thread(
                    self.gatherer.scan_target,
                    target,
                    rust_config
                )
                
                # 轉換為 Asset
                for endpoint in result.get("endpoints", []):
                    asset = {
                        "asset_id": f"rust_{endpoint['path']}",
                        "type": "endpoint",
                        "value": endpoint['path'],
                        "parameters": endpoint.get('parameters', [])
                    }
                    all_assets.append(asset)
            
            return EngineExecutionResult(
                engine_name="rust",
                status="success",
                execution_time=time.time() - start_time,
                assets=all_assets,
                metadata={
                    "mode": rust_config["mode"],
                    "js_findings": len([a for a in all_assets if 'js' in a.get('type', '')])
                }
            )
            
        except Exception as exc:
            return EngineExecutionResult(
                engine_name="rust",
                status="failed",
                execution_time=time.time() - start_time,
                error=str(exc)
            )
    
    async def health_check(self) -> bool:
        return self.gatherer.check_availability()
    
    def get_capabilities(self) -> dict[str, Any]:
        return {
            "engine": "rust",
            "version": "1.0",
            "features": [
                "fast_discovery",
                "js_analysis",
                "endpoint_detection",
                "high_performance"
            ],
            "configurable_params": [
                "mode",
                "js_analysis_depth",
                "memory_limit_mb",
                "max_depth"
            ]
        }


# services/scan/coordinators/adapters/go_adapter.py

class GoEngineAdapter(IEngine):
    """Go 引擎適配器"""
    
    def __init__(self):
        from pathlib import Path
        self.go_engine_path = Path(__file__).parent.parent.parent / "engines" / "go_engine"
    
    async def execute(
        self, 
        config: EngineConfig,
        targets: list[str],
        context: dict[str, Any]
    ) -> EngineExecutionResult:
        """執行 Go 引擎掃描"""
        start_time = time.time()
        
        try:
            # 檢查可用的 Go 掃描器
            available = await self._check_scanners()
            
            if not available:
                return EngineExecutionResult(
                    engine_name="go",
                    status="skipped",
                    execution_time=0,
                    error="No Go scanners available"
                )
            
            # 根據配置決定啟用哪些掃描器
            tasks = []
            if config.engine_specific.get("enable_ssrf", True) and available.get("ssrf"):
                tasks.append(self._run_ssrf_scanner(targets, config))
            
            if config.engine_specific.get("enable_cspm", True) and available.get("cspm"):
                tasks.append(self._run_cspm_scanner(targets, config))
            
            if config.engine_specific.get("enable_sca", False) and available.get("sca"):
                tasks.append(self._run_sca_scanner(targets, config))
            
            # 並行執行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 聚合結果
            all_assets = []
            scanners_used = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    continue
                if isinstance(result, list):
                    all_assets.extend(result)
                    scanners_used.append(["ssrf", "cspm", "sca"][i])
            
            return EngineExecutionResult(
                engine_name="go",
                status="success",
                execution_time=time.time() - start_time,
                assets=all_assets,
                metadata={
                    "scanners_used": scanners_used,
                    "ssrf_findings": len([a for a in all_assets if 'ssrf' in a.get('type', '')]),
                    "cspm_findings": len([a for a in all_assets if 'cspm' in a.get('type', '')]),
                    "sca_findings": len([a for a in all_assets if 'sca' in a.get('type', '')])
                }
            )
            
        except Exception as exc:
            return EngineExecutionResult(
                engine_name="go",
                status="failed",
                execution_time=time.time() - start_time,
                error=str(exc)
            )
    
    async def _check_scanners(self) -> dict[str, bool]:
        """檢查 Go 掃描器可用性"""
        scanners = {
            "ssrf": self.go_engine_path / "ssrf_scanner" / "worker.exe",
            "cspm": self.go_engine_path / "cspm_scanner" / "worker.exe",
            "sca": self.go_engine_path / "sca_scanner" / "worker.exe"
        }
        return {name: path.exists() for name, path in scanners.items()}
    
    async def _run_ssrf_scanner(self, targets, config) -> list[dict]:
        """調用 SSRF 掃描器"""
        # 實現細節 (參考 go_engine/worker.py)
        ...
    
    async def health_check(self) -> bool:
        available = await self._check_scanners()
        return any(available.values())
    
    def get_capabilities(self) -> dict[str, Any]:
        return {
            "engine": "go",
            "version": "1.0",
            "features": [
                "ssrf_detection",
                "cspm_scanning",
                "sca_analysis",
                "high_concurrency"
            ],
            "configurable_params": [
                "enable_ssrf",
                "enable_cspm",
                "enable_sca",
                "ssrf_timeout",
                "cspm_cloud_providers",
                "sca_include_dev_deps"
            ]
        }
```

---

### 4. 協調器核心實現

```python
# services/scan/coordinators/multi_engine_coordinator.py (重構版)

from typing import List, Dict, Any
import asyncio
import time
from .engine_factory import DefaultEngineFactory, IEngine
from .engine_schemas import (
    EngineCoordinationRequest,
    EngineCoordinationResponse,
    EngineExecutionResult,
    ExecutionStrategy
)


class MultiEngineCoordinator:
    """多引擎協調器 - 配置驅動版本
    
    核心職責:
    1. 接收 Core 模組的 EngineCoordinationRequest
    2. 根據配置創建並執行引擎
    3. 聚合結果返回給 Core
    """
    
    def __init__(self, engine_factory: DefaultEngineFactory | None = None):
        self.factory = engine_factory or DefaultEngineFactory()
        self.logger = logging.getLogger(__name__)
    
    async def execute(
        self, 
        request: EngineCoordinationRequest
    ) -> EngineCoordinationResponse:
        """執行協調掃描
        
        Args:
            request: 協調請求 (包含所有配置)
        
        Returns:
            協調響應 (包含所有引擎結果)
        """
        start_time = time.time()
        self.logger.info(f"🚀 開始協調掃描: {request.scan_id}")
        
        # 1. 驗證引擎配置
        valid_configs = await self._validate_configs(request.engine_configs)
        if not valid_configs:
            return EngineCoordinationResponse(
                scan_id=request.scan_id,
                status="failed",
                execution_time=time.time() - start_time,
                engine_results=[],
                summary={"error": "No valid engine configurations"}
            )
        
        # 2. 根據執行策略調度
        strategy = request.execution_strategy
        
        if strategy.mode == "parallel":
            results = await self._execute_parallel(
                valid_configs, 
                request.targets,
                request
            )
        elif strategy.mode == "sequential":
            results = await self._execute_sequential(
                valid_configs,
                request.targets,
                request
            )
        elif strategy.mode == "hybrid":
            results = await self._execute_hybrid(
                valid_configs,
                request.targets,
                request
            )
        else:
            raise ValueError(f"Unknown execution mode: {strategy.mode}")
        
        # 3. 聚合結果
        aggregated_assets = self._aggregate_assets(results)
        
        # 4. 構建響應
        execution_time = time.time() - start_time
        status = self._determine_status(results)
        
        response = EngineCoordinationResponse(
            scan_id=request.scan_id,
            status=status,
            execution_time=execution_time,
            engine_results=results,
            aggregated_assets=aggregated_assets,
            summary=self._build_summary(results, execution_time)
        )
        
        self.logger.info(
            f"✅ 協調掃描完成: {request.scan_id}, "
            f"狀態={status}, 時間={execution_time:.2f}s"
        )
        
        return response
    
    async def _validate_configs(
        self, 
        configs: List[EngineConfig]
    ) -> List[EngineConfig]:
        """驗證引擎配置"""
        valid_configs = []
        available_engines = self.factory.get_available_engines()
        
        for config in configs:
            # 檢查引擎是否可用
            if config.engine_name not in available_engines:
                self.logger.warning(
                    f"引擎 {config.engine_name} 不可用,跳過"
                )
                continue
            
            # 檢查引擎是否啟用
            if not config.enabled:
                self.logger.info(
                    f"引擎 {config.engine_name} 未啟用,跳過"
                )
                continue
            
            # 健康檢查
            engine = self.factory.create_engine(config.engine_name)
            if await engine.health_check():
                valid_configs.append(config)
                self.logger.info(f"✓ 引擎 {config.engine_name} 已就緒")
            else:
                self.logger.warning(
                    f"引擎 {config.engine_name} 健康檢查失敗,跳過"
                )
        
        return valid_configs
    
    async def _execute_parallel(
        self,
        configs: List[EngineConfig],
        targets: List[str],
        request: EngineCoordinationRequest
    ) -> List[EngineExecutionResult]:
        """並行執行引擎"""
        self.logger.info(f"⚡ 並行執行 {len(configs)} 個引擎")
        
        tasks = []
        for config in configs:
            engine = self.factory.create_engine(config.engine_name)
            context = {
                "scan_id": request.scan_id,
                "phase0_result": request.phase0_result,
                "global_config": request.global_config
            }
            task = engine.execute(config, targets, context)
            tasks.append(task)
        
        # 使用 TaskGroup (Python 3.11+) 或 gather
        try:
            async with asyncio.TaskGroup() as tg:
                result_tasks = [tg.create_task(task) for task in tasks]
            results = [await task for task in result_tasks]
        except AttributeError:
            # Python < 3.11 fallback
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [
                r if not isinstance(r, Exception) 
                else EngineExecutionResult(
                    engine_name="unknown",
                    status="failed",
                    execution_time=0,
                    error=str(r)
                )
                for r in results
            ]
        
        return results
    
    async def _execute_sequential(
        self,
        configs: List[EngineConfig],
        targets: List[str],
        request: EngineCoordinationRequest
    ) -> List[EngineExecutionResult]:
        """串行執行引擎"""
        seq_config = request.execution_strategy.sequential_config or {}
        order = seq_config.get("order", [c.engine_name for c in configs])
        pass_results = seq_config.get("pass_results", False)
        
        self.logger.info(f"🔄 串行執行引擎: {' → '.join(order)}")
        
        results = []
        context = {
            "scan_id": request.scan_id,
            "phase0_result": request.phase0_result,
            "global_config": request.global_config
        }
        
        for engine_name in order:
            # 找到對應配置
            config = next((c for c in configs if c.engine_name == engine_name), None)
            if not config:
                continue
            
            self.logger.info(f"  ▶️ 執行 {engine_name} 引擎...")
            
            # 創建引擎並執行
            engine = self.factory.create_engine(engine_name)
            result = await engine.execute(config, targets, context)
            results.append(result)
            
            # 如果啟用結果傳遞,將當前結果添加到上下文
            if pass_results and result.status == "success":
                context[f"{engine_name}_result"] = result.model_dump()
                self.logger.info(
                    f"  📊 {engine_name} 發現 {len(result.assets)} 個資產"
                )
        
        return results
    
    async def _execute_hybrid(
        self,
        configs: List[EngineConfig],
        targets: List[str],
        request: EngineCoordinationRequest
    ) -> List[EngineExecutionResult]:
        """混合執行引擎 (部分並行,部分串行)"""
        hybrid_config = request.execution_strategy.hybrid_config or {}
        
        # 示例: {"parallel_group_1": ["rust", "go"], "then": ["python"]}
        parallel_group = hybrid_config.get("parallel_group_1", [])
        sequential_group = hybrid_config.get("then", [])
        
        self.logger.info("🔀 混合執行模式")
        
        results = []
        
        # 1. 先執行並行組
        if parallel_group:
            parallel_configs = [
                c for c in configs if c.engine_name in parallel_group
            ]
            parallel_results = await self._execute_parallel(
                parallel_configs, targets, request
            )
            results.extend(parallel_results)
        
        # 2. 再執行串行組
        if sequential_group:
            # 將並行結果添加到上下文
            request.phase0_result = request.phase0_result or {}
            for r in results:
                if r.status == "success":
                    request.phase0_result[f"{r.engine_name}_assets"] = r.assets
            
            sequential_configs = [
                c for c in configs if c.engine_name in sequential_group
            ]
            sequential_results = await self._execute_sequential(
                sequential_configs, targets, request
            )
            results.extend(sequential_results)
        
        return results
    
    def _aggregate_assets(
        self, 
        results: List[EngineExecutionResult]
    ) -> List[Dict]:
        """聚合並去重資產"""
        seen = set()
        unique_assets = []
        
        for result in results:
            for asset in result.assets:
                # 使用 (type, value) 作為唯一標識
                key = (asset.get("type"), asset.get("value"))
                if key not in seen:
                    seen.add(key)
                    # 添加來源信息
                    asset["discovered_by"] = result.engine_name
                    unique_assets.append(asset)
        
        self.logger.info(f"  🔍 去重後共 {len(unique_assets)} 個唯一資產")
        return unique_assets
    
    def _determine_status(
        self, 
        results: List[EngineExecutionResult]
    ) -> str:
        """判斷整體狀態"""
        if not results:
            return "failed"
        
        success_count = sum(1 for r in results if r.status == "success")
        failed_count = sum(1 for r in results if r.status == "failed")
        
        if success_count == len(results):
            return "success"
        elif success_count > 0:
            return "partial"
        else:
            return "failed"
    
    def _build_summary(
        self, 
        results: List[EngineExecutionResult],
        total_time: float
    ) -> Dict[str, Any]:
        """構建摘要統計"""
        return {
            "total_engines": len(results),
            "successful_engines": sum(1 for r in results if r.status == "success"),
            "failed_engines": sum(1 for r in results if r.status == "failed"),
            "total_execution_time": total_time,
            "average_execution_time": total_time / len(results) if results else 0,
            "total_assets_before_dedup": sum(len(r.assets) for r in results),
            "engines_used": [r.engine_name for r in results]
        }
```

---

## 🎮 使用示例

### 示例 1: Core 模組調用協調器 (最小配置)

```python
from services.scan.coordinators import MultiEngineCoordinator
from services.scan.coordinators.engine_schemas import (
    EngineCoordinationRequest,
    EngineConfig,
    ExecutionStrategy
)

# Core 模組準備配置
request = EngineCoordinationRequest(
    scan_id="scan_001",
    targets=["https://example.com"],
    
    # 配置兩個引擎
    engine_configs=[
        EngineConfig(
            engine_name="rust",
            max_depth=3,
            timeout=600
        ),
        EngineConfig(
            engine_name="python",
            max_depth=5,
            timeout=1800
        )
    ],
    
    # 並行執行
    execution_strategy=ExecutionStrategy(mode="parallel")
)

# 協調器執行
coordinator = MultiEngineCoordinator()
response = await coordinator.execute(request)

# 結果
print(f"狀態: {response.status}")
print(f"發現資產: {len(response.aggregated_assets)} 個")
print(f"引擎結果: {[r.engine_name for r in response.engine_results]}")
```

### 示例 2: 高級配置 - 所有參數可控

```python
request = EngineCoordinationRequest(
    scan_id="scan_002",
    targets=["https://juice-shop.example.com"],
    
    # 配置四個引擎,每個引擎都有詳細配置
    engine_configs=[
        # Rust 引擎 - 快速偵察模式
        EngineConfig(
            engine_name="rust",
            enabled=True,
            max_depth=2,
            timeout=300,
            rate_limit=RateLimit(requests_per_second=50, burst=100),
            engine_specific={
                "mode": "fast_discovery",
                "js_analysis_depth": 2,
                "memory_limit_mb": 256
            }
        ),
        
        # Python 引擎 - 深度爬取模式
        EngineConfig(
            engine_name="python",
            enabled=True,
            max_depth=7,
            max_pages=5000,
            timeout=3600,
            rate_limit=RateLimit(requests_per_second=25, burst=50),
            authentication=Authentication(
                method="bearer",
                credentials={"token": "xxxx"}
            ),
            engine_specific={
                "enable_dynamic_rendering": True,
                "playwright_browser": "chromium",
                "screenshot_on_error": True,
                "wait_for_navigation": True
            }
        ),
        
        # Go 引擎 - 專業掃描器
        EngineConfig(
            engine_name="go",
            enabled=True,
            timeout=1200,
            engine_specific={
                "enable_ssrf": True,
                "enable_cspm": True,
                "enable_sca": False,
                "ssrf_timeout": 60,
                "cspm_cloud_providers": ["aws", "azure", "gcp"],
                "ssrf_bypass_techniques": ["dns_rebinding", "redirect_chain"]
            }
        ),
        
        # TypeScript 引擎 - SPA 渲染
        EngineConfig(
            engine_name="typescript",
            enabled=False,  # 暫時禁用
            max_depth=5,
            timeout=1800,
            engine_specific={
                "enable_spa_routing": True,
                "intercept_ajax": True,
                "wait_for_idle": True
            }
        )
    ],
    
    # 執行策略 - 混合模式
    execution_strategy=ExecutionStrategy(
        mode="hybrid",
        hybrid_config={
            # 第一組: Rust 和 Go 並行 (快速偵察)
            "parallel_group_1": ["rust", "go"],
            # 第二組: Python 串行 (基於第一組結果)
            "then": ["python"]
        }
    ),
    
    # 全局配置 (覆蓋單個引擎配置)
    global_config={
        "verbose": True,
        "save_screenshots": True,
        "output_format": "json"
    }
)

# 執行
response = await coordinator.execute(request)
```

### 示例 3: 串行協同模式 (Rust → Python)

```python
request = EngineCoordinationRequest(
    scan_id="scan_003",
    targets=["https://example.com"],
    
    engine_configs=[
        EngineConfig(engine_name="rust", max_depth=2, timeout=300),
        EngineConfig(engine_name="python", max_depth=5, timeout=1800)
    ],
    
    # 串行執行,並傳遞結果
    execution_strategy=ExecutionStrategy(
        mode="sequential",
        sequential_config={
            "order": ["rust", "python"],
            "pass_results": True  # Python 可以看到 Rust 的結果
        }
    )
)

response = await coordinator.execute(request)

# Rust 發現 100 個 URL
# Python 基於這 100 個 URL 進行深度爬取
```

---

## 📊 參數配置完整清單

### 通用參數 (所有引擎)

| 參數 | 類型 | 默認值 | 說明 | 來源 |
|------|------|--------|------|------|
| `max_depth` | int | 5 | 最大爬取深度 | aiva_common.Phase1StartPayload |
| `max_pages` | int | 1000 | 最大頁面數 | aiva_common.Phase1StartPayload |
| `timeout` | int | 1800 | 超時時間(秒) | aiva_common.Phase1StartPayload |
| `rate_limit.requests_per_second` | int | 25 | 每秒請求數 | aiva_common.RateLimit |
| `rate_limit.burst` | int | 50 | 突發請求數 | aiva_common.RateLimit |
| `authentication.method` | str | "none" | 認證方法 | aiva_common.Authentication |
| `authentication.credentials` | dict | {} | 認證憑證 | aiva_common.Authentication |

### Python Engine 特定參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `enable_dynamic_rendering` | bool | True | 啟用 Playwright 動態渲染 |
| `playwright_browser` | str | "chromium" | 瀏覽器類型 (chromium/firefox/webkit) |
| `screenshot_on_error` | bool | False | 錯誤時截圖 |
| `wait_for_navigation` | bool | True | 等待頁面導航完成 |
| `wait_for_idle` | bool | False | 等待網絡閒置 |
| `intercept_ajax` | bool | True | 攔截 AJAX 請求 |

### Rust Engine 特定參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `mode` | str | "deep_analysis" | 掃描模式 (fast_discovery/deep_analysis) |
| `js_analysis_depth` | int | 3 | JS 分析深度 |
| `memory_limit_mb` | int | 512 | 內存限制 (MB) |
| `enable_js_deobfuscation` | bool | True | 啟用 JS 反混淆 |
| `parallel_workers` | int | 4 | 並行工作數 |

### Go Engine 特定參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `enable_ssrf` | bool | True | 啟用 SSRF 掃描器 |
| `enable_cspm` | bool | True | 啟用 CSPM 掃描器 |
| `enable_sca` | bool | False | 啟用 SCA 掃描器 |
| `ssrf_timeout` | int | 60 | SSRF 掃描超時 |
| `cspm_cloud_providers` | list | ["aws"] | 雲服務商列表 |
| `sca_include_dev_deps` | bool | False | 包含開發依賴 |
| `ssrf_bypass_techniques` | list | [] | SSRF 繞過技術 |

### TypeScript Engine 特定參數

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `enable_spa_routing` | bool | True | 啟用 SPA 路由檢測 |
| `intercept_ajax` | bool | True | 攔截 AJAX |
| `wait_for_idle` | bool | True | 等待閒置 |
| `headless` | bool | True | 無頭模式 |

---

## 🔄 與 Core 模組的交互流程

```
┌──────────────────────────────────────────────────────────────┐
│                        Core 模組                              │
│                                                               │
│  1. 接收 Phase 0 結果 (Rust 快速偵察)                         │
│  2. AI 分析決策:                                              │
│     • 檢測到 100 個端點 → 需要 Python 爬蟲                    │
│     • 發現 50 個 JS 文件 → 需要 Rust JS 分析                  │
│     • 檢測到 AWS S3 URL → 需要 Go CSPM 掃描                   │
│  3. 生成 EngineCoordinationRequest                           │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
                       📨 發送 Request
                       ↓
┌──────────────────────────────────────────────────────────────┐
│              MultiEngineCoordinator                           │
│                                                               │
│  1. 驗證引擎配置                                              │
│  2. 創建引擎實例 (透過 Factory)                               │
│  3. 根據執行策略調度:                                         │
│     • Mode = "hybrid":                                       │
│       - 並行: [Rust, Go]                                     │
│       - 串行: [Python] (基於上一步結果)                       │
│  4. 執行掃描                                                  │
│  5. 聚合並去重資產                                           │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
                       📨 返回 Response
                       ↓
┌──────────────────────────────────────────────────────────────┐
│                        Core 模組                              │
│                                                               │
│  1. 接收 EngineCoordinationResponse                          │
│  2. 分析結果:                                                 │
│     • aggregated_assets: 500 個唯一資產                       │
│     • engine_results: Rust/Go/Python 詳細結果                │
│  3. 將資產分發給 Function 模組測試                            │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 實施步驟

### Step 1: 創建引擎配置 Schema

```bash
# 創建新文件
services/scan/coordinators/engine_schemas.py
```

實現上面定義的所有 Schema 類。

### Step 2: 實現引擎工廠

```bash
# 創建工廠模塊
services/scan/coordinators/engine_factory.py
```

實現 `IEngine` 協議和 `DefaultEngineFactory`。

### Step 3: 創建引擎適配器

```bash
# 創建適配器目錄
services/scan/coordinators/adapters/
    __init__.py
    python_adapter.py
    rust_adapter.py
    go_adapter.py
    typescript_adapter.py
```

每個適配器將對應引擎適配到 `IEngine` 接口。

### Step 4: 重構協調器

```bash
# 修改現有協調器
services/scan/coordinators/multi_engine_coordinator.py
```

使用配置驅動的新實現替換舊代碼。

### Step 5: Core 模組集成

```bash
# 修改 Core 模組
services/core/cognitive_core/decision_engine.py
```

讓 Core 模組生成 `EngineCoordinationRequest` 並調用協調器。

---

## 🧪 測試用例

### 測試 1: 單引擎執行

```python
async def test_single_engine():
    request = EngineCoordinationRequest(
        scan_id="test_001",
        targets=["https://juice-shop.local"],
        engine_configs=[
            EngineConfig(engine_name="python", max_depth=3)
        ],
        execution_strategy=ExecutionStrategy(mode="parallel")
    )
    
    coordinator = MultiEngineCoordinator()
    response = await coordinator.execute(request)
    
    assert response.status == "success"
    assert len(response.engine_results) == 1
    assert response.engine_results[0].engine_name == "python"
```

### 測試 2: 多引擎並行

```python
async def test_multi_engine_parallel():
    request = EngineCoordinationRequest(
        scan_id="test_002",
        targets=["https://juice-shop.local"],
        engine_configs=[
            EngineConfig(engine_name="rust", max_depth=2),
            EngineConfig(engine_name="python", max_depth=5),
            EngineConfig(engine_name="go", timeout=600)
        ],
        execution_strategy=ExecutionStrategy(mode="parallel")
    )
    
    response = await coordinator.execute(request)
    
    assert len(response.engine_results) == 3
    assert response.status in ["success", "partial"]
```

### 測試 3: 引擎特定參數

```python
async def test_engine_specific_params():
    request = EngineCoordinationRequest(
        scan_id="test_003",
        targets=["https://example.com"],
        engine_configs=[
            EngineConfig(
                engine_name="go",
                engine_specific={
                    "enable_ssrf": True,
                    "enable_cspm": False,
                    "enable_sca": False
                }
            )
        ]
    )
    
    response = await coordinator.execute(request)
    
    # 驗證只有 SSRF 掃描器被調用
    go_result = response.engine_results[0]
    assert "ssrf" in go_result.metadata["scanners_used"]
    assert "cspm" not in go_result.metadata["scanners_used"]
```

---

## 📚 關鍵技術參考

### 1. AIVA Common Schema

- **路徑**: `services/aiva_common/schemas/`
- **關鍵文件**:
  - `tasks.py`: Phase0/Phase1 Payload 定義
  - `base.py`: 基礎模型 (Asset, RateLimit, Authentication)
  - `assets.py`: 資產相關 Schema

### 2. Abstract Factory 模式

- **參考**: https://refactoring.guru/design-patterns/abstract-factory
- **核心思想**: 通過工廠創建產品族 (引擎)
- **優點**: 解耦、易擴展

### 3. Asyncio 並發模式

- **參考**: https://docs.python.org/3/library/asyncio-task.html
- **關鍵技術**:
  - `asyncio.gather()`: 並行執行
  - `asyncio.TaskGroup()`: 結構化並發 (Python 3.11+)
  - `asyncio.to_thread()`: 調用同步代碼

### 4. Protocol (Structural Subtyping)

- **PEP 544**: https://peps.python.org/pep-0544/
- **用途**: 定義引擎接口而不需要繼承
- **優點**: 更靈活的類型檢查

---

## 🎯 設計優勢

### 1. 完全配置驅動

**所有參數都可配置**, Core 模組完全控制:
- 使用哪些引擎
- 每個引擎的所有參數
- 執行順序和模式

### 2. 統一數據合約

基於 **aiva_common Schema**, 確保:
- 所有模組使用相同數據格式
- 類型安全 (Pydantic 驗證)
- 易於序列化和傳輸

### 3. 插件式架構

添加新引擎只需:
1. 實現 `IEngine` 協議
2. 註冊到工廠
3. 無需修改協調器代碼

### 4. 靈活的執行策略

支持三種模式:
- **Parallel**: 所有引擎同時執行
- **Sequential**: 引擎依次執行,可傳遞結果
- **Hybrid**: 部分並行,部分串行

### 5. 引擎隔離

每個引擎:
- 獨立實現
- 獨立配置
- 獨立錯誤處理
- 不影響其他引擎

---

## 📋 TODO 清單

### Phase 1: 基礎架構 (1-2 週)

- [ ] 創建引擎配置 Schema
- [ ] 實現引擎工廠
- [ ] 實現 Python Engine 適配器
- [ ] 實現 Rust Engine 適配器
- [ ] 重構協調器核心

### Phase 2: 引擎整合 (2-3 週)

- [ ] 實現 Go Engine 適配器
- [ ] 實現 TypeScript Engine 適配器
- [ ] 添加參數驗證邏輯
- [ ] 實現結果去重算法
- [ ] 添加執行策略支持

### Phase 3: Core 模組集成 (1 週)

- [ ] 修改 Core 決策引擎
- [ ] 生成 EngineCoordinationRequest
- [ ] 處理 EngineCoordinationResponse
- [ ] 集成測試

### Phase 4: 測試與優化 (1-2 週)

- [ ] 單元測試 (每個適配器)
- [ ] 集成測試 (多引擎協同)
- [ ] 性能測試 (並行 vs 串行)
- [ ] 壓力測試 (大規模目標)
- [ ] 文檔完善

---

## 🏁 總結

本設計方案提供了一個 **完全配置驅動、基於數據合約、支持所有引擎參數調整** 的協調器架構。

**核心特點**:
1. ✅ **統一數據合約**: 基於 aiva_common Schema
2. ✅ **參數完全可配置**: 所有引擎參數都可動態調整
3. ✅ **插件式架構**: 易於添加新引擎
4. ✅ **靈活執行策略**: 並行/串行/混合
5. ✅ **Core 模組控制**: 協調器只執行,不決策

**下一步**: 按照實施步驟依次實現各個組件。

---

**文檔版本**: 1.0  
**最後更新**: 2025-11-19  
**作者**: AIVA Architecture Team
