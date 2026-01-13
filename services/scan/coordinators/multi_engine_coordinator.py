"""Multi-Engine Coordinator - 多引擎協調器

AI 控制的多語言掃描引擎統一入口

架構設計：
    AI Core (task_planning) → MultiEngineCoordinator → CLI Engines (Rust/Go/TS/Python)

特點：
    - 🎯 輕量級：專注任務分發與結果收集
    - ⚡ 高效能：保持 CLI 直接調用的效能優勢
    - 🔄 AI 友好：提供結構化輸出和錯誤恢復
    - 🧩 功能兼顧：同時支援掃描任務和功能模組

實現日期: 2026-01-11
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScanStrategy(str, Enum):
    """掃描策略"""
    FAST = "fast"                      # 快速：僅 Rust 引擎
    BALANCED = "balanced"              # 平衡：Rust + Python
    COMPREHENSIVE = "comprehensive"    # 全面：全部引擎
    AGGRESSIVE = "aggressive"          # 激進：並行 + 深度
    SMART = "smart"                    # 智能：AI 動態決策


class EngineStatus(str, Enum):
    """引擎狀態"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class EngineConfig:
    """引擎配置"""
    name: str
    language: str
    cli_command: str
    timeout: int = 300
    priority: int = 5
    capabilities: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """掃描結果"""
    scan_id: str
    engine_id: str
    success: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MultiEngineCoordinator:
    """多引擎協調器
    
    統一管理 Rust/Go/TypeScript/Python 掃描引擎
    通過 subprocess + JSON 標準化通訊
    
    使用示例:
        coordinator = MultiEngineCoordinator()
        await coordinator.initialize()
        
        result = await coordinator.execute_strategy_balanced(
            scan_id="scan_001",
            targets=["https://example.com"],
            max_depth=3
        )
    """
    
    # 引擎配置表
    ENGINES: dict[str, EngineConfig] = {
        "rust_scanner": EngineConfig(
            name="Rust Scanner",
            language="rust",
            cli_command="services/scan/rust_engine/target/release/rust_scanner",
            timeout=120,
            priority=10,  # 最高優先級
            capabilities=["port_scan", "service_detection", "web_crawl"]
        ),
        "python_passive": EngineConfig(
            name="Python Passive Analyzer",
            language="python",
            cli_command="python -m services.scan.python_engine.passive_analyzer",
            timeout=180,
            priority=7,
            capabilities=["passive_analysis", "pattern_matching", "data_extraction"]
        ),
        "python_xxe": EngineConfig(
            name="Python XXE Detector",
            language="python",
            cli_command="python -m services.scan.python_engine.xxe_detector",
            timeout=120,
            priority=6,
            capabilities=["xxe_detection"]
        ),
        "python_deserial": EngineConfig(
            name="Python Deserialization Detector",
            language="python",
            cli_command="python -m services.scan.python_engine.deserialization_detector",
            timeout=150,
            priority=6,
            capabilities=["deserialization_detection"]
        ),
    }
    
    def __init__(self):
        """初始化多引擎協調器"""
        self._initialized = False
        self._active_scans: dict[str, asyncio.Task] = {}
        self._engine_status: dict[str, EngineStatus] = {}
        self._scan_history: list[ScanResult] = []
        
        # 基礎路徑
        self._base_path = Path(__file__).parent.parent
        
        logger.info("MultiEngineCoordinator created")
    
    async def initialize(self) -> None:
        """初始化協調器，檢查各引擎健康狀態"""
        logger.info("🔧 Initializing MultiEngineCoordinator...")
        
        for engine_id, config in self.ENGINES.items():
            status = await self._check_engine_health(engine_id, config)
            self._engine_status[engine_id] = status
            status_icon = "✅" if status == EngineStatus.AVAILABLE else "❌"
            logger.info(f"   {status_icon} Engine '{config.name}': {status.value}")
        
        available_count = sum(
            1 for s in self._engine_status.values() 
            if s == EngineStatus.AVAILABLE
        )
        
        self._initialized = True
        logger.info(f"✅ MultiEngineCoordinator initialized: {available_count}/{len(self.ENGINES)} engines available")
    
    async def _check_engine_health(self, engine_id: str, config: EngineConfig) -> EngineStatus:
        """檢查引擎健康狀態"""
        try:
            if config.language == "rust":
                # Rust 引擎：檢查二進制檔案是否存在
                binary_path = self._base_path / config.cli_command.replace("services/scan/", "")
                if binary_path.exists():
                    return EngineStatus.AVAILABLE
                
                # 嘗試相對路徑
                alt_path = Path(config.cli_command)
                if alt_path.exists():
                    return EngineStatus.AVAILABLE
                
                logger.warning(f"Rust binary not found: {binary_path}")
                return EngineStatus.UNAVAILABLE
                
            elif config.language == "python":
                # Python 引擎：嘗試 import 檢查
                module_path = config.cli_command.replace("python -m ", "")
                try:
                    __import__(module_path)
                    return EngineStatus.AVAILABLE
                except ImportError:
                    # 檢查檔案是否存在
                    file_path = self._base_path / module_path.replace(".", "/") + ".py"
                    if file_path.exists():
                        return EngineStatus.AVAILABLE
                    return EngineStatus.UNAVAILABLE
                    
            return EngineStatus.AVAILABLE
            
        except Exception as e:
            logger.error(f"Health check failed for {engine_id}: {e}")
            return EngineStatus.ERROR
    
    def get_available_engines(self) -> list[str]:
        """獲取可用引擎列表"""
        return [
            engine_id for engine_id, status in self._engine_status.items()
            if status == EngineStatus.AVAILABLE
        ]
    
    def get_engine_capabilities(self, engine_id: str) -> list[str]:
        """獲取引擎能力列表"""
        config = self.ENGINES.get(engine_id)
        return config.capabilities if config else []
    
    # ==================== 掃描策略方法 ====================
    
    async def execute_strategy_fast(
        self,
        scan_id: str,
        targets: list[str],
        max_depth: int = 3
    ) -> dict[str, Any]:
        """快速掃描策略
        
        特點：
        - 僅使用高優先級引擎 (Rust)
        - 最小化延遲
        - 適合初步探測
        """
        logger.info(f"🚀 Executing FAST strategy for scan {scan_id}")
        
        engines = ["rust_scanner"]
        available = [e for e in engines if self._engine_status.get(e) == EngineStatus.AVAILABLE]
        
        if not available:
            logger.warning("Fast strategy: Rust engine unavailable, falling back to Python")
            available = ["python_passive"]
        
        return await self._execute_scan(
            scan_id=scan_id,
            targets=targets,
            engines=available,
            max_depth=max_depth,
            strategy="fast"
        )
    
    async def execute_strategy_balanced(
        self,
        scan_id: str,
        targets: list[str],
        max_depth: int = 3
    ) -> dict[str, Any]:
        """平衡掃描策略
        
        特點：
        - 使用 Rust + Python 引擎
        - 效能與覆蓋度平衡
        - 推薦的預設策略
        """
        logger.info(f"⚖️ Executing BALANCED strategy for scan {scan_id}")
        
        engines = ["rust_scanner", "python_passive"]
        available = [e for e in engines if self._engine_status.get(e) == EngineStatus.AVAILABLE]
        
        return await self._execute_scan(
            scan_id=scan_id,
            targets=targets,
            engines=available,
            max_depth=max_depth,
            strategy="balanced"
        )
    
    async def execute_strategy_comprehensive(
        self,
        scan_id: str,
        targets: list[str],
        max_depth: int = 5
    ) -> dict[str, Any]:
        """全面掃描策略
        
        特點：
        - 使用所有可用引擎
        - 深度掃描
        - 最大化覆蓋
        """
        logger.info(f"🔍 Executing COMPREHENSIVE strategy for scan {scan_id}")
        
        available = self.get_available_engines()
        
        return await self._execute_scan(
            scan_id=scan_id,
            targets=targets,
            engines=available,
            max_depth=max_depth,
            strategy="comprehensive"
        )
    
    async def execute_strategy_aggressive(
        self,
        scan_id: str,
        targets: list[str],
        max_depth: int = 10
    ) -> dict[str, Any]:
        """激進掃描策略
        
        特點：
        - 並行執行所有引擎
        - 最大深度
        - 高資源消耗
        """
        logger.info(f"⚡ Executing AGGRESSIVE strategy for scan {scan_id}")
        
        available = self.get_available_engines()
        
        # 並行執行所有引擎
        tasks = []
        for engine_id in available:
            task = self._execute_single_engine(
                scan_id=f"{scan_id}_{engine_id}",
                targets=targets,
                engine_id=engine_id,
                max_depth=max_depth
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # 合併結果
        merged = self._merge_results(results, scan_id, available)
        merged["strategy"] = "aggressive"
        merged["duration"] = duration
        merged["parallel"] = True
        
        return merged
    
    async def execute_strategy_smart(
        self,
        scan_id: str,
        targets: list[str],
        max_depth: int = 5
    ) -> dict[str, Any]:
        """智能掃描策略
        
        特點：
        - AI 動態決策
        - 根據初步結果調整深度
        - 自適應資源分配
        """
        logger.info(f"🧠 Executing SMART strategy for scan {scan_id}")
        
        # Phase 1: 快速探測
        logger.info("   Phase 1: Quick reconnaissance...")
        initial_result = await self.execute_strategy_fast(
            scan_id=f"{scan_id}_phase1",
            targets=targets,
            max_depth=2
        )
        
        findings_count = initial_result.get("findings_count", 0)
        logger.info(f"   Phase 1 complete: {findings_count} potential findings")
        
        # Phase 2: 根據結果決定策略
        if findings_count > 5:
            # 發現多個潛在漏洞，進行全面掃描
            logger.info("   Phase 2: Multiple findings detected, switching to COMPREHENSIVE")
            final_result = await self.execute_strategy_comprehensive(
                scan_id=f"{scan_id}_phase2",
                targets=targets,
                max_depth=max_depth
            )
        elif findings_count > 0:
            # 發現少量漏洞，使用平衡策略
            logger.info("   Phase 2: Some findings detected, using BALANCED")
            final_result = await self.execute_strategy_balanced(
                scan_id=f"{scan_id}_phase2",
                targets=targets,
                max_depth=max_depth
            )
        else:
            # 無發現，保持快速模式
            logger.info("   Phase 2: No findings, staying with FAST")
            final_result = initial_result
        
        # 合併兩階段結果
        combined = {
            "scan_id": scan_id,
            "strategy": "smart",
            "phases": [
                {"phase": 1, "strategy": "fast", "findings_count": findings_count},
                {"phase": 2, "strategy": final_result.get("strategy", "fast")}
            ],
            "findings": initial_result.get("findings", []) + final_result.get("findings", []),
            "findings_count": initial_result.get("findings_count", 0) + final_result.get("findings_count", 0),
            "engines_used": list(set(
                initial_result.get("engines_used", []) + 
                final_result.get("engines_used", [])
            )),
            "duration": initial_result.get("duration", 0) + final_result.get("duration", 0),
            "success": initial_result.get("success", False) or final_result.get("success", False)
        }
        
        return combined
    
    # ==================== 內部方法 ====================
    
    async def _execute_scan(
        self,
        scan_id: str,
        targets: list[str],
        engines: list[str],
        max_depth: int,
        strategy: str
    ) -> dict[str, Any]:
        """執行掃描（依序執行引擎）"""
        results: dict[str, Any] = {
            "scan_id": scan_id,
            "strategy": strategy,
            "targets": targets,
            "engines_used": engines,
            "findings": [],
            "errors": [],
            "duration": 0.0,
            "success": False
        }
        
        start_time = time.time()
        
        for engine_id in engines:
            engine_result = await self._execute_single_engine(
                scan_id=scan_id,
                targets=targets,
                engine_id=engine_id,
                max_depth=max_depth
            )
            
            if engine_result.success:
                results["findings"].extend(engine_result.findings)
                results["success"] = True
            else:
                results["errors"].extend(engine_result.errors)
            
            # 記錄歷史
            self._scan_history.append(engine_result)
        
        results["duration"] = time.time() - start_time
        results["findings_count"] = len(results["findings"])
        
        logger.info(
            f"✅ Scan {scan_id} complete: "
            f"{results['findings_count']} findings in {results['duration']:.2f}s"
        )
        
        return results
    
    async def _execute_single_engine(
        self,
        scan_id: str,
        targets: list[str],
        engine_id: str,
        max_depth: int
    ) -> ScanResult:
        """執行單一引擎"""
        config = self.ENGINES.get(engine_id)
        
        if not config:
            return ScanResult(
                scan_id=scan_id,
                engine_id=engine_id,
                success=False,
                errors=[f"Unknown engine: {engine_id}"]
            )
        
        if self._engine_status.get(engine_id) != EngineStatus.AVAILABLE:
            return ScanResult(
                scan_id=scan_id,
                engine_id=engine_id,
                success=False,
                errors=[f"Engine {engine_id} is not available"]
            )
        
        logger.info(f"   🔧 Executing {config.name}...")
        start_time = time.time()
        
        try:
            if config.language == "rust":
                result = await self._execute_rust_engine(config, targets, max_depth)
            elif config.language == "python":
                result = await self._execute_python_engine(config, targets, max_depth)
            else:
                result = {"success": False, "error": f"Unsupported language: {config.language}"}
            
            duration = time.time() - start_time
            
            if result.get("success"):
                findings = result.get("findings", [])
                logger.info(f"   ✅ {config.name}: {len(findings)} findings in {duration:.2f}s")
                return ScanResult(
                    scan_id=scan_id,
                    engine_id=engine_id,
                    success=True,
                    findings=findings,
                    duration=duration
                )
            else:
                error = result.get("error", "Unknown error")
                logger.warning(f"   ⚠️ {config.name}: {error}")
                return ScanResult(
                    scan_id=scan_id,
                    engine_id=engine_id,
                    success=False,
                    errors=[error],
                    duration=duration
                )
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(f"   ❌ {config.name}: Timeout after {config.timeout}s")
            return ScanResult(
                scan_id=scan_id,
                engine_id=engine_id,
                success=False,
                errors=[f"Timeout after {config.timeout}s"],
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"   ❌ {config.name}: {e}")
            return ScanResult(
                scan_id=scan_id,
                engine_id=engine_id,
                success=False,
                errors=[str(e)],
                duration=duration
            )
    
    async def _execute_rust_engine(
        self,
        config: EngineConfig,
        targets: list[str],
        max_depth: int
    ) -> dict[str, Any]:
        """執行 Rust 引擎"""
        # 構建命令
        cmd = [
            config.cli_command,
            "--targets", ",".join(targets),
            "--depth", str(max_depth),
            "--output-format", "json"
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )
            
            if process.returncode == 0:
                try:
                    output = json.loads(stdout.decode())
                    return {"success": True, "findings": output.get("findings", [])}
                except json.JSONDecodeError:
                    # 如果不是 JSON，嘗試解析文本輸出
                    return {"success": True, "findings": [], "raw_output": stdout.decode()}
            else:
                return {"success": False, "error": stderr.decode() or "Non-zero exit code"}
                
        except FileNotFoundError:
            return {"success": False, "error": f"Binary not found: {config.cli_command}"}
    
    async def _execute_python_engine(
        self,
        config: EngineConfig,
        targets: list[str],
        max_depth: int
    ) -> dict[str, Any]:
        """執行 Python 引擎"""
        # 嘗試直接 import 並調用
        module_path = config.cli_command.replace("python -m ", "")
        
        try:
            # 動態導入模組
            import importlib
            module = importlib.import_module(module_path)
            
            # 檢查是否有標準化的 scan 方法
            if hasattr(module, "scan"):
                result = await module.scan(targets=targets, depth=max_depth)
                return {"success": True, "findings": result.get("findings", [])}
            elif hasattr(module, "analyze"):
                result = await module.analyze(targets=targets, depth=max_depth)
                return {"success": True, "findings": result.get("findings", [])}
            else:
                # 沒有標準方法，嘗試 subprocess
                return await self._execute_python_subprocess(config, targets, max_depth)
                
        except ImportError:
            # 模組無法導入，使用 subprocess
            return await self._execute_python_subprocess(config, targets, max_depth)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_python_subprocess(
        self,
        config: EngineConfig,
        targets: list[str],
        max_depth: int
    ) -> dict[str, Any]:
        """使用 subprocess 執行 Python 引擎"""
        cmd = config.cli_command.split() + [
            "--targets", ",".join(targets),
            "--depth", str(max_depth),
            "--output-format", "json"
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )
            
            if process.returncode == 0:
                try:
                    output = json.loads(stdout.decode())
                    return {"success": True, "findings": output.get("findings", [])}
                except json.JSONDecodeError:
                    return {"success": True, "findings": [], "raw_output": stdout.decode()}
            else:
                return {"success": False, "error": stderr.decode() or "Non-zero exit code"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _merge_results(
        self,
        results: list,
        scan_id: str,
        engines: list[str]
    ) -> dict[str, Any]:
        """合併多引擎結果"""
        merged: dict[str, Any] = {
            "scan_id": scan_id,
            "findings": [],
            "errors": [],
            "engines_used": engines,
            "engines_completed": 0,
            "success": False
        }
        
        for result in results:
            if isinstance(result, Exception):
                merged["errors"].append(str(result))
            elif isinstance(result, ScanResult):
                if result.success:
                    merged["findings"].extend(result.findings)
                    merged["engines_completed"] += 1
                    merged["success"] = True
                else:
                    merged["errors"].extend(result.errors)
            elif isinstance(result, dict):
                if result.get("success"):
                    merged["findings"].extend(result.get("findings", []))
                    merged["engines_completed"] += 1
                    merged["success"] = True
                else:
                    merged["errors"].append(result.get("error", "Unknown error"))
        
        # 去重 findings
        seen = set()
        unique_findings = []
        for finding in merged["findings"]:
            finding_key = json.dumps(finding, sort_keys=True)
            if finding_key not in seen:
                seen.add(finding_key)
                unique_findings.append(finding)
        
        merged["findings"] = unique_findings
        merged["findings_count"] = len(unique_findings)
        
        return merged
    
    # ==================== 統計和監控 ====================
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取掃描統計"""
        if not self._scan_history:
            return {"total_scans": 0}
        
        total = len(self._scan_history)
        successful = sum(1 for r in self._scan_history if r.success)
        total_findings = sum(len(r.findings) for r in self._scan_history)
        avg_duration = sum(r.duration for r in self._scan_history) / total
        
        return {
            "total_scans": total,
            "successful_scans": successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_findings": total_findings,
            "avg_duration": avg_duration,
            "engines_status": {
                engine_id: status.value 
                for engine_id, status in self._engine_status.items()
            }
        }
    
    def clear_history(self) -> None:
        """清除掃描歷史"""
        self._scan_history.clear()
        logger.info("Scan history cleared")


# ==================== 導出 ====================

__all__ = [
    "MultiEngineCoordinator",
    "ScanStrategy",
    "EngineStatus",
    "EngineConfig",
    "ScanResult"
]
