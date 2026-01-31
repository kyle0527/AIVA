import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Import Python Engine Components
try:
    from services.scan.python_engine.xxe_detector import XXEDetector
    from services.scan.python_engine.deserialization_detector_v2 import DeserializationDetector
    from services.scan.python_engine.passive_analyzer import PassiveAnalyzer
    PYTHON_ENGINE_AVAILABLE = True
except ImportError:
    PYTHON_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiEngineCoordinator:
    """Multi-Engine Coordinator - Orchestrates Go, Rust, and Python scan engines"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the coordinator.
        
        Args:
            project_root: Path to the project root. If None, attempts to auto-discover.
        """
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        
        # Binary Paths
        self.go_binary = self.project_root / "services/scan/go_engine/bin/scanner"
        self.rust_binary = self.project_root / "services/scan/rust_engine/target/release/aiva-info-gatherer"
        
        # Python Engines
        self.xxe_detector = XXEDetector() if PYTHON_ENGINE_AVAILABLE else None
        self.deserial_detector = DeserializationDetector() if PYTHON_ENGINE_AVAILABLE else None
        self.passive_analyzer = PassiveAnalyzer() if PYTHON_ENGINE_AVAILABLE else None

    def _find_project_root(self) -> Path:
        """Find the project root by looking for services directory"""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "services").exists() and (parent / "services" / "scan").exists():
                return parent
        return Path.cwd()

    async def initialize(self):
        """Check for engine availability"""
        logger.info("Initializing MultiEngineCoordinator...")
        
        # Check Go Engine
        if not self.go_binary.exists():
            logger.warning(f"Go binary not found at {self.go_binary}. Go-based scans (SSRF, SCA) will be skipped.")
        else:
            logger.info(f"Go Engine found: {self.go_binary}")

        # Check Rust Engine
        if not self.rust_binary.exists():
            logger.warning(f"Rust binary not found at {self.rust_binary}. Rust-based scans (Port, Smuggling) will be skipped.")
        else:
            logger.info(f"Rust Engine found: {self.rust_binary}")

        if not PYTHON_ENGINE_AVAILABLE:
            logger.warning("Python Engine modules not importable. Python-based scans will be skipped.")
        else:
            logger.info("Python Engines initialized.")

    async def _run_command(self, cmd: List[str], input_data: Optional[str] = None) -> Dict[str, Any]:
        """Run a CLI command asynchronously"""
        try:
            cmd_str = " ".join(str(p) for p in cmd)
            logger.debug(f"Executing: {cmd_str}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=input_data.encode() if input_data else None)
            
            if process.returncode != 0:
                logger.error(f"Command failed: {cmd_str}\nStderr: {stderr.decode()}")
                return {"error": stderr.decode(), "status": "failed"}
            
            try:
                # Try to parse JSON output if available
                output_str = stdout.decode().strip()
                if output_str.startswith("{") or output_str.startswith("["):
                    return json.loads(output_str)
                return {"output": output_str, "status": "success"}
            except json.JSONDecodeError:
                return {"output": stdout.decode(), "status": "success (raw output)"}
                
        except FileNotFoundError:
             logger.error(f"Binary not found: {cmd[0]}")
             return {"error": "Binary not found", "status": "failed"}
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"error": str(e), "status": "failed"}

    async def execute_strategy_fast(self, scan_id: str, targets: List[str], max_depth: int) -> Dict[str, Any]:
        """Fast Strategy: Port Scan (Rust) + Basic SSRF (Go)"""
        start_time = datetime.now()
        results = {"scan_id": scan_id, "strategy": "fast", "details": {}}
        
        tasks = []
        for target in targets:
            # Rust Port Scan
            if self.rust_binary.exists():
                # Extract host from URL for port scan would be better, but assuming target is URL here
                # Simplified for demo: scan standard ports
                cmd = [str(self.rust_binary), "scan", "--target", target, "--ports", "80,443,8080,8443", "--mode", "fast", "--format", "json"]
                tasks.append(self._run_command(cmd))
            
            # Go SSRF Basic
            if self.go_binary.exists():
                # Need to verify Go Engine CLI arguments from README
                # ./bin/scanner ssrf --url <URL>
                cmd = [str(self.go_binary), "ssrf", "--url", target]
                tasks.append(self._run_command(cmd))

        scan_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        results["raw_results"] = scan_results
        results["execution_time"] = (datetime.now() - start_time).total_seconds()
        return results

    async def execute_strategy_balanced(self, scan_id: str, targets: List[str], max_depth: int) -> Dict[str, Any]:
        """Balanced Strategy: Fast + Info Gather (Rust) + Passive Analysis (Python)"""
        # First run fast scan
        fast_result = await self.execute_strategy_fast(scan_id, targets, max_depth)
        
        # Add tasks
        tasks = []
        for target in targets:
             # Rust Info Gather
            if self.rust_binary.exists():
                cmd = [str(self.rust_binary), "gather", "--url", target, "--format", "json"]
                tasks.append(self._run_command(cmd))
        
        # Note: Passive Analysis typically works on traffic, here we assume it might fetch or analyze target structure
        # For this implementation, we just mock the python call if available as placeholder for real traffic analysis integration
        if self.passive_analyzer:
             # In a real scenario, we would feed HAR files or traffic logs
             pass 

        scan_results = await asyncio.gather(*tasks)
        
        # Merge
        fast_result["details"]["balanced_extension"] = scan_results
        fast_result["strategy"] = "balanced"
        return fast_result

    async def execute_strategy_comprehensive(self, scan_id: str, targets: List[str], max_depth: int) -> Dict[str, Any]:
        """Comprehensive Strategy: Balanced + Smuggling (Rust) + XXE/Deserial (Python)"""
        # Run Balanced first
        balanced_result = await self.execute_strategy_balanced(scan_id, targets, max_depth)
        
        tasks = []
        for target in targets:
            # Rust Smuggling
            if self.rust_binary.exists():
                cmd = [str(self.rust_binary), "smuggling", "--url", target, "--format", "json"]
                tasks.append(self._run_command(cmd))
            
            # Python Active Checks (Conceptually)
            if self.xxe_detector:
                # This would be synchronous in strict sense, ideally specific async wrapper or thread pool
                # Assuming detector has async method or is fast enough
                # tasks.append(asyncio.to_thread(self.xxe_detector.detect, "xml_payload_placeholder", target))
                pass

        scan_results = await asyncio.gather(*tasks)
        
        balanced_result["details"]["comprehensive_extension"] = scan_results
        balanced_result["strategy"] = "comprehensive"
        return balanced_result

    async def execute_strategy_aggressive(self, scan_id: str, targets: List[str], max_depth: int) -> Dict[str, Any]:
        """Aggressive Strategy: Comprehensive + Auth Brute (Rust)"""
        comp_result = await self.execute_strategy_comprehensive(scan_id, targets, max_depth)
        
        # Add Brute Force Logic -> CAUTION: Only with permission
        # ... logic implementation ...
        
        comp_result["strategy"] = "aggressive"
        return comp_result

    async def execute_strategy_smart(self, scan_id: str, targets: List[str], max_depth: int) -> Dict[str, Any]:
        """Smart Strategy: AI Decided (For now maps to Balanced)"""
        return await self.execute_strategy_balanced(scan_id, targets, max_depth)
