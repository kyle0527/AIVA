"""
統一掃描引擎 - 整合AIVA主系統和掃描套件的功能

提供統一的掃描介面，結合：
- AIVA主系統的掃描協調器
- 掃描套件的高級功能
- 插件系統支援
"""

from __future__ import annotations
import sys
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# 添加掃描套件到路徑
SCAN_SUITE_PATH = r'c:\Users\User\Downloads\AIVA_scan_suite_20251019'
if SCAN_SUITE_PATH not in sys.path:
    sys.path.append(SCAN_SUITE_PATH)

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

@dataclass
class UnifiedScanConfig:
    """統一掃描配置"""
    targets: List[str]
    scan_type: str = "comprehensive"  # fast, comprehensive, aggressive
    max_depth: int = 3
    max_pages: int = 100
    enable_plugins: bool = True
    output_format: str = "json"
    
class UnifiedScanEngine:
    """統一掃描引擎"""
    
    def __init__(self, config: UnifiedScanConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化AIVA主系統掃描器
        self.aiva_scanner = None
        try:
            from .aiva_scan.scan_orchestrator import ScanOrchestrator
            self.aiva_scanner = ScanOrchestrator
            self.logger.info("AIVA主系統掃描器已載入")
        except ImportError as e:
            self.logger.warning(f"AIVA主系統掃描器載入失敗: {e}")
        
        # 初始化掃描套件
        self.suite_scanner = None
        self.suite_config_class = None
        try:
            from services.scan.aiva_scan import ScanOrchestrator as SuiteScanOrchestrator
            from services.scan.aiva_scan import OrchestratorConfig
            self.suite_scanner = SuiteScanOrchestrator
            self.suite_config_class = OrchestratorConfig
            self.logger.info("掃描套件已載入")
        except ImportError as e:
            self.logger.warning(f"掃描套件載入失敗: {e}")
    
    async def run_comprehensive_scan(self) -> Dict[str, Any]:
        """執行綜合掃描 - 整合 Phase I 模組"""
        results = {
            "scan_id": f"unified_scan_{int(__import__('time').time())}",
            "targets": self.config.targets,
            "results": [],
            "summary": {},
            "phase_i_findings": []
        }
        
        # 使用掃描套件執行基礎掃描
        if self.suite_scanner and self.suite_config_class:
            self.logger.info("使用掃描套件執行基礎掃描")
            suite_config = self.suite_config_class(
                seeds=self.config.targets,
                max_depth=self.config.max_depth,
                max_pages=self.config.max_pages,
                strategy=self.config.scan_type.upper()
            )
            
            orchestrator = self.suite_scanner(suite_config)
            suite_results = await orchestrator.run()
            
            results["suite_results"] = suite_results
            results["summary"]["suite_pages"] = suite_results.get("pages", 0)
        
        # 整合 Phase I 高價值功能模組
        phase_i_results = await self._execute_phase_i_modules()
        results["phase_i_findings"].extend(phase_i_results)
        results["summary"]["phase_i_findings_count"] = len(phase_i_results)
        
        return results
    
    async def _execute_phase_i_modules(self) -> List[Dict[str, Any]]:
        """執行 Phase I 高價值模組"""
        findings = []
        
        try:
            # 1. 客戶端授權繞過檢測
            from services.features.client_side_auth_bypass.client_side_auth_bypass_worker import ClientSideAuthBypassWorker
            from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget, FunctionTaskContext
            
            for target_url in self.config.targets:
                self.logger.info(f"執行客戶端授權繞過檢測: {target_url}")
                
                # 構建任務 payload
                task_payload = FunctionTaskPayload(
                    task_id=f"csab_{int(__import__('time').time())}",
                    module_name="FUNC_CLIENT_AUTH_BYPASS",
                    target=FunctionTaskTarget(url=target_url),
                    context=FunctionTaskContext(session_id=f"scan_{int(__import__('time').time())}")
                )
                
                # 執行檢測
                worker = ClientSideAuthBypassWorker()
                result = await worker.execute_task(task_payload)
                
                if result.success and result.findings:
                    findings.extend([{
                        "module": "client_auth_bypass",
                        "target": target_url,
                        "finding": finding
                    } for finding in result.findings])
            
            # 2. 進階 SSRF 檢測 (如果有 Go 模組可用)
            # TODO: 整合 Go SSRF 模組
            
        except Exception as e:
            self.logger.error(f"Phase I 模組執行錯誤: {e}")
        
        return findings
    
    def get_available_scanners(self) -> Dict[str, bool]:
        """獲取可用掃描器狀態"""
        return {
            "aiva_main_scanner": self.aiva_scanner is not None,
            "scan_suite": self.suite_scanner is not None,
        }
    
    @classmethod
    def create_fast_scan(cls, targets: List[str]) -> 'UnifiedScanEngine':
        """創建快速掃描配置"""
        config = UnifiedScanConfig(
            targets=targets,
            scan_type="fast",
            max_depth=1,
            max_pages=20
        )
        return cls(config)
    
    @classmethod  
    def create_comprehensive_scan(cls, targets: List[str]) -> 'UnifiedScanEngine':
        """創建綜合掃描配置"""
        config = UnifiedScanConfig(
            targets=targets,
            scan_type="comprehensive",
            max_depth=3,
            max_pages=100
        )
        return cls(config)


def main():
    """測試統一掃描引擎"""
    import asyncio
    
    # 創建掃描引擎
    engine = UnifiedScanEngine.create_fast_scan(["http://example.com"])
    
    print("=== 統一掃描引擎測試 ===")
    print(f"可用掃描器: {engine.get_available_scanners()}")
    
    # 如果有可用的掃描器，執行測試掃描
    if any(engine.get_available_scanners().values()):
        print("執行測試掃描...")
        results = asyncio.run(engine.run_comprehensive_scan())
        print(f"掃描完成，掃描ID: {results['scan_id']}")
        print(f"掃描目標: {results['targets']}")
    else:
        print("⚠️  沒有可用的掃描器")

if __name__ == "__main__":
    main()