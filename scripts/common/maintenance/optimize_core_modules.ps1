# AIVA 核心模組優化腳本
# 執行核心模組的關鍵優化任務

param(
    [Parameter(Position=0)]
    [ValidateSet("unify-ai", "refactor-app", "split-optimized", "monitor", "all", "help")]
    [string]$Action = "help",
    
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

# 設定顏色輸出
function Write-ColorOutput($ForegroundColor) {
    if ($Host.UI.RawUI.ForegroundColor) {
        $fc = $Host.UI.RawUI.ForegroundColor
        $Host.UI.RawUI.ForegroundColor = $ForegroundColor
        if ($args) {
            Write-Output $args
        } else {
            $input | Write-Output
        }
        $Host.UI.RawUI.ForegroundColor = $fc
    } else {
        if ($args) {
            Write-Output $args
        } else {
            $input | Write-Output
        }
    }
}

function Write-Info($Message) {
    Write-ColorOutput Cyan "ℹ️  $Message"
}

function Write-Success($Message) {
    Write-ColorOutput Green "✅ $Message"
}

function Write-Warning($Message) {
    Write-ColorOutput Yellow "⚠️  $Message"
}

function Write-Error($Message) {
    Write-ColorOutput Red "❌ $Message"
}

# 檢查前置條件
function Test-Prerequisites {
    Write-Info "檢查前置條件..."
    
    $coreDir = "services\core\aiva_core"
    if (-not (Test-Path $coreDir)) {
        Write-Error "找不到核心模組目錄: $coreDir"
        return $false
    }
    
    # 檢查Git狀態
    try {
        $gitStatus = git status --porcelain
        if ($gitStatus) {
            Write-Warning "Git工作區有未提交的變更，建議先提交"
        }
    } catch {
        Write-Warning "無法檢查Git狀態"
    }
    
    Write-Success "前置條件檢查完成"
    return $true
}

# 統一AI引擎
function Invoke-UnifyAIEngine {
    Write-Info "開始統一AI引擎..."
    
    $aiEngineDir = "services\core\aiva_core\ai_engine"
    
    if (-not (Test-Path $aiEngineDir)) {
        Write-Error "AI引擎目錄不存在: $aiEngineDir"
        return
    }
    
    Push-Location $aiEngineDir
    
    try {
        # 創建必要目錄
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path "unified" -Force | Out-Null
            New-Item -ItemType Directory -Path "legacy" -Force | Out-Null
        }
        
        # 備份舊版本
        $backupFiles = @("bio_neuron_core.py.backup", "knowledge_base.py.backup")
        foreach ($file in $backupFiles) {
            if (Test-Path $file) {
                Write-Info "備份文件: $file -> legacy\"
                if (-not $DryRun) {
                    Move-Item $file "legacy\" -Force
                }
            }
        }
        
        # 分析版本差異
        if ((Test-Path "bio_neuron_core.py") -and (Test-Path "bio_neuron_core_v2.py")) {
            Write-Info "分析AI引擎版本差異..."
            if (-not $DryRun) {
                git diff --no-index bio_neuron_core.py bio_neuron_core_v2.py > ai_engine_diff.txt 2>$null
            }
            Write-Success "差異分析完成，結果保存至 ai_engine_diff.txt"
        }
        
        # 統一版本邏輯
        if (Test-Path "bio_neuron_core_v2.py") {
            $v2Size = (Get-Item "bio_neuron_core_v2.py").Length
            $v1Size = if (Test-Path "bio_neuron_core.py") { (Get-Item "bio_neuron_core.py").Length } else { 0 }
            
            if ($v2Size -gt $v1Size) {
                Write-Info "使用 v2 作為統一版本 (更大的文件)"
                if (-not $DryRun) {
                    Copy-Item "bio_neuron_core_v2.py" "unified\bio_neuron_core.py"
                    Move-Item "bio_neuron_core.py" "legacy\bio_neuron_core_v1.py" -Force -ErrorAction SilentlyContinue
                    Copy-Item "unified\bio_neuron_core.py" "bio_neuron_core.py" -Force
                    Remove-Item "bio_neuron_core_v2.py" -Force
                }
            } else {
                Write-Info "保留 v1 作為主版本"
                if (-not $DryRun) {
                    Move-Item "bio_neuron_core_v2.py" "legacy\bio_neuron_core_v2.py" -Force
                }
            }
        }
        
        Write-Success "AI引擎統一完成"
        
    } finally {
        Pop-Location
    }
}

# 重構app.py
function Invoke-RefactorApp {
    Write-Info "開始重構 app.py..."
    
    $appFile = "services\core\aiva_core\app.py"
    $bootstrapDir = "services\core\aiva_core\bootstrap"
    
    if (-not (Test-Path $appFile)) {
        Write-Error "找不到 app.py: $appFile"
        return
    }
    
    # 創建bootstrap目錄
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $bootstrapDir -Force | Out-Null
    }
    
    # 創建依賴注入容器
    $containerContent = @"
"""
依賴注入容器 - 管理核心模組的依賴關係
"""
from typing import Any, Dict, Type, TypeVar, Protocol
import asyncio
from abc import ABC, abstractmethod

T = TypeVar('T')

class ServiceProtocol(Protocol):
    '''服務協議基類'''
    pass

class DependencyContainer:
    '''依賴注入容器'''
    
    def __init__(self):
        self._services: Dict[Type, Type] = {}
        self._instances: Dict[Type, Any] = {}
        self._singletons: Dict[Type, bool] = {}
    
    def register(self, interface: Type[T], implementation: Type[T], singleton: bool = True) -> None:
        '''註冊服務'''
        self._services[interface] = implementation
        self._singletons[interface] = singleton
    
    def get(self, interface: Type[T]) -> T:
        '''獲取服務實例'''
        if interface in self._instances:
            return self._instances[interface]
        
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
        
        implementation = self._services[interface]
        instance = implementation()
        
        if self._singletons.get(interface, True):
            self._instances[interface] = instance
        
        return instance
    
    def clear(self):
        '''清空容器'''
        self._instances.clear()

# 全域容器實例
container = DependencyContainer()
"@

    if (-not $DryRun) {
        $containerContent | Out-File -FilePath "$bootstrapDir\dependency_container.py" -Encoding UTF8
    }
    
    Write-Success "依賴注入容器創建完成"
    
    # 創建組件工廠
    $factoryContent = @"
"""
組件工廠 - 創建和配置核心組件
"""
from services.core.aiva_core.bootstrap.dependency_container import container
from services.aiva_common.config import get_settings

class ComponentFactory:
    '''組件工廠類'''
    
    @staticmethod
    def setup_core_components():
        '''設置核心組件'''
        from services.core.aiva_core.ingestion.scan_module_interface import ScanModuleInterface
        from services.core.aiva_core.analysis.initial_surface import InitialAttackSurface
        from services.core.aiva_core.execution.task_generator import TaskGenerator
        
        # 註冊核心服務
        container.register(ScanModuleInterface, ScanModuleInterface)
        container.register(InitialAttackSurface, InitialAttackSurface)
        container.register(TaskGenerator, TaskGenerator)
        
        return container
    
    @staticmethod
    def get_app_config():
        '''獲取應用配置'''
        settings = get_settings()
        return {
            'title': "AIVA Core Engine - 智慧分析與協調中心",
            'description': "核心分析引擎：攻擊面分析、策略生成、任務協調",
            'version': "2.0.0",
            'monitor_interval': settings.core_monitor_interval,
            'enable_strategy_gen': settings.enable_strategy_generator,
        }

# 初始化工廠
factory = ComponentFactory()
"@

    if (-not $DryRun) {
        $factoryContent | Out-File -FilePath "$bootstrapDir\component_factory.py" -Encoding UTF8
        New-Item -ItemType File -Path "$bootstrapDir\__init__.py" -Force | Out-Null
    }
    
    Write-Success "組件工廠創建完成"
    Write-Info "建議手動更新 app.py 以使用新的依賴注入系統"
}

# 拆分optimized_core.py
function Invoke-SplitOptimizedCore {
    Write-Info "開始拆分 optimized_core.py..."
    
    $optimizedFile = "services\core\aiva_core\optimized_core.py"
    $optimizationDir = "services\core\aiva_core\optimization"
    
    if (-not (Test-Path $optimizedFile)) {
        Write-Warning "找不到 optimized_core.py，跳過拆分"
        return
    }
    
    # 創建optimization目錄
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $optimizationDir -Force | Out-Null
    }
    
    # 讀取原始文件內容
    $content = Get-Content $optimizedFile -Raw
    
    # 創建各個專業化模組
    $modules = @{
        "parallel_processing.py" = "ParallelMessageProcessor"
        "neural_optimization.py" = "OptimizedBioNet"
        "memory_management.py" = "MemoryManager"
        "metrics_collection.py" = "MetricsCollector"
        "component_pooling.py" = "ComponentPool"
    }
    
    foreach ($module in $modules.Keys) {
        $className = $modules[$module]
        Write-Info "創建模組: $module (包含 $className)"
        
        if (-not $DryRun) {
            $moduleContent = @"
"""
$($module.Replace('.py','').Replace('_',' ').ToTitleCase()) - 從 optimized_core.py 重構而來
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# TODO: 從 optimized_core.py 中提取 $className 相關代碼
# 這個模組需要手動完成重構

class ${className}Placeholder:
    '''$className 的佔位符實現'''
    
    def __init__(self):
        logger.warning(f"使用 $className 的佔位符實現，需要完成重構")
    
    async def placeholder_method(self):
        '''佔位符方法'''
        pass

# 導出主要類
__all__ = ['${className}Placeholder']
"@
            $moduleContent | Out-File -FilePath "$optimizationDir\$module" -Encoding UTF8
        }
    }
    
    if (-not $DryRun) {
        # 創建 __init__.py
        $initContent = @"
"""
核心模組優化包
"""
from .parallel_processing import ParallelMessageProcessorPlaceholder
from .neural_optimization import OptimizedBioNetPlaceholder
from .memory_management import MemoryManagerPlaceholder
from .metrics_collection import MetricsCollectorPlaceholder
from .component_pooling import ComponentPoolPlaceholder

__all__ = [
    'ParallelMessageProcessorPlaceholder',
    'OptimizedBioNetPlaceholder',
    'MemoryManagerPlaceholder',
    'MetricsCollectorPlaceholder',
    'ComponentPoolPlaceholder',
]
"@
        $initContent | Out-File -FilePath "$optimizationDir\__init__.py" -Encoding UTF8
    }
    
    Write-Success "優化模組拆分完成"
    Write-Warning "需要手動將 optimized_core.py 中的具體實現遷移到新模組中"
}

# 設置監控系統
function Invoke-SetupMonitoring {
    Write-Info "設置效能監控系統..."
    
    $monitoringDir = "services\core\aiva_core\monitoring"
    
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $monitoringDir -Force | Out-Null
    }
    
    # 創建效能監控器
    $monitorContent = @"
"""
核心模組效能監控器
"""
import asyncio
import psutil
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    '''效能指標'''
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Optional[Dict[str, str]] = None

class CorePerformanceMonitor:
    '''核心模組效能監控器'''
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.is_running = False
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'response_time': 1000.0,
        }
    
    async def start_monitoring(self):
        '''開始效能監控'''
        if self.is_running:
            logger.warning("監控已在運行中")
            return
        
        self.is_running = True
        logger.info("開始核心模組效能監控")
        
        while self.is_running:
            try:
                await self._collect_metrics()
                await self._check_thresholds()
                await asyncio.sleep(30)  # 每30秒收集一次
            except Exception as e:
                logger.error(f"監控過程中發生錯誤: {e}")
                await asyncio.sleep(60)  # 錯誤後等待1分鐘再試
    
    async def _collect_metrics(self):
        '''收集效能指標'''
        now = datetime.now()
        
        # 系統指標
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metrics = [
            PerformanceMetric(now, 'cpu_usage', cpu_percent, 'percent'),
            PerformanceMetric(now, 'memory_usage', memory_info.percent, 'percent'),
            PerformanceMetric(now, 'memory_available', memory_info.available / 1024**3, 'GB'),
        ]
        
        self.metrics.extend(metrics)
        
        # 保留最近1000個指標
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    async def _check_thresholds(self):
        '''檢查閾值並發出告警'''
        if not self.metrics:
            return
        
        latest_metrics = {m.metric_name: m.value for m in self.metrics[-10:] if m.metric_name in self.thresholds}
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in latest_metrics:
                current_value = latest_metrics[metric_name]
                if current_value > threshold:
                    logger.warning(f"效能告警: {metric_name} = {current_value:.1f} 超過閾值 {threshold}")
    
    def stop_monitoring(self):
        '''停止效能監控'''
        self.is_running = False
        logger.info("停止核心模組效能監控")
    
    def get_metrics_summary(self) -> Dict[str, float]:
        '''獲取指標摘要'''
        if not self.metrics:
            return {}
        
        recent_metrics = self.metrics[-60:]  # 最近60個指標
        summary = {}
        
        for metric_name in ['cpu_usage', 'memory_usage']:
            values = [m.value for m in recent_metrics if m.metric_name == metric_name]
            if values:
                summary[f'{metric_name}_avg'] = sum(values) / len(values)
                summary[f'{metric_name}_max'] = max(values)
        
        return summary

# 全域監控實例
monitor = CorePerformanceMonitor()
"@

    if (-not $DryRun) {
        $monitorContent | Out-File -FilePath "$monitoringDir\performance_monitor.py" -Encoding UTF8
        New-Item -ItemType File -Path "$monitoringDir\__init__.py" -Force | Out-Null
    }
    
    Write-Success "效能監控系統設置完成"
}

# 執行所有優化
function Invoke-AllOptimizations {
    Write-Info "開始執行所有核心模組優化..."
    
    if (-not (Test-Prerequisites)) {
        Write-Error "前置條件檢查失敗，停止執行"
        return
    }
    
    Write-Info "步驟 1/4: 統一AI引擎"
    Invoke-UnifyAIEngine
    
    Write-Info "步驟 2/4: 重構app.py"
    Invoke-RefactorApp
    
    Write-Info "步驟 3/4: 拆分optimized_core.py"
    Invoke-SplitOptimizedCore
    
    Write-Info "步驟 4/4: 設置監控系統"
    Invoke-SetupMonitoring
    
    Write-Success "所有核心模組優化完成！"
    Write-Info "後續步驟："
    Write-Info "1. 手動完成 optimized_core.py 的程式碼遷移"
    Write-Info "2. 更新 app.py 以使用新的依賴注入系統"
    Write-Info "3. 運行測試確保功能正常"
    Write-Info "4. 查看優化建議報告: reports\ANALYSIS_REPORTS\CORE_MODULE_OPTIMIZATION_RECOMMENDATIONS.md"
}

# 顯示幫助
function Show-Help {
    Write-ColorOutput White @"
AIVA 核心模組優化工具

用法: .\optimize_core_modules.ps1 [動作] [選項]

動作:
  unify-ai        統一AI引擎版本
  refactor-app    重構app.py依賴注入
  split-optimized 拆分optimized_core.py
  monitor         設置效能監控
  all             執行所有優化 (推薦)
  help            顯示此幫助

選項:
  -DryRun         預覽模式，不實際修改檔案
  -Verbose        顯示詳細輸出

範例:
  .\optimize_core_modules.ps1 all              # 執行所有優化
  .\optimize_core_modules.ps1 unify-ai -DryRun # 預覽AI引擎統一過程
  .\optimize_core_modules.ps1 monitor          # 只設置監控系統

詳細說明請查看：
reports\ANALYSIS_REPORTS\CORE_MODULE_OPTIMIZATION_RECOMMENDATIONS.md
"@
}

# 主執行邏輯
switch ($Action) {
    "unify-ai" { 
        if (Test-Prerequisites) { Invoke-UnifyAIEngine }
    }
    "refactor-app" { 
        if (Test-Prerequisites) { Invoke-RefactorApp }
    }
    "split-optimized" { 
        if (Test-Prerequisites) { Invoke-SplitOptimizedCore }
    }
    "monitor" { 
        if (Test-Prerequisites) { Invoke-SetupMonitoring }
    }
    "all" { 
        Invoke-AllOptimizations 
    }
    "help" { 
        Show-Help 
    }
    default { 
        Write-Error "未知動作: $Action"
        Show-Help 
    }
}