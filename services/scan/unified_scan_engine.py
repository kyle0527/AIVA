"""
統一掃描引擎 - 異步消息隊列架構版本

遵循 aiva_common 架構模式，將同步檔案操作改為異步消息隊列架構：
- 使用 MessageBroker 進行消息通信
- 實施異步任務派發和結果收集
- 遵循 12-factor app 原則
- 符合 AIVA 統一架構規範
"""


import asyncio
import logging

from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

from services.aiva_common.config import get_settings
from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader, 
    ScanStartPayload,
    FunctionTaskPayload,
    FunctionTaskTarget,
    FunctionTaskContext,
)
from services.aiva_common.schemas.tasks import FunctionTaskSchema
from services.core.aiva_core.messaging.message_broker import MessageBroker
from services.core.aiva_core.messaging.task_dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)

class UnifiedScanConfig(BaseModel):
    """統一掃描配置 - 遵循 Pydantic 模式"""
    targets: List[str]
    scan_type: str = "comprehensive"  # fast, comprehensive, aggressive
    max_depth: int = 3
    max_pages: int = 100
    enable_plugins: bool = True
    output_format: str = "json"
    session_id: Optional[str] = None
    scan_id: Optional[str] = None
    
class UnifiedScanEngine:
    """統一掃描引擎 - 異步消息隊列架構"""
    
    def __init__(self, config: UnifiedScanConfig):
        self.config = config
        self.logger = logger
        self.settings = get_settings()
        
        # 初始化消息代理和任務派发器
        self.broker = MessageBroker(ModuleName.SCAN)
        self.dispatcher = TaskDispatcher(self.broker, ModuleName.SCAN)
        
        # 任務追蹤
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.scan_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # 生成掃描 ID
        if not self.config.scan_id:
            self.config.scan_id = f"unified_scan_{uuid4().hex[:12]}"
        
        if not self.config.session_id:
            self.config.session_id = f"session_{uuid4().hex[:12]}"
    
    async def run_comprehensive_scan(self) -> Dict[str, Any]:
        """執行綜合掃描 - 整合 Phase I 模組"""
        results = {
            "scan_id": f"unified_scan_{int(__import__('time').time())}",
            "targets": self.config.targets,
            "results": [],
            "summary": {},
            "phase_i_findings": []
        }
        
        # 使用異步消息隊列執行基礎掃描
        self.logger.info("使用消息隊列派發基礎掃描任務")
        
        # 創建掃描套件任務
        suite_task = FunctionTaskSchema(
            task_id=f"suite_scan_{uuid4().hex[:8]}",
            module_name=ModuleName.SCAN,
            target=FunctionTaskTarget(urls=self.config.targets),
            context=FunctionTaskContext(
                session_id=self.config.session_id,
                scan_id=self.config.scan_id,
                metadata={
                    "scan_type": "suite",
                    "max_depth": self.config.max_depth,
                    "max_pages": self.config.max_pages,
                    "strategy": self.config.scan_type.upper()
                }
            )
        )
        
        # 派發任務並等待結果
        suite_results = await self.dispatcher.dispatch_task(suite_task)
        results["suite_results"] = suite_results or {}
        results["summary"]["suite_pages"] = results["suite_results"].get("pages", 0)
        
        # 使用消息隊列執行 Phase I 高價值功能模組
        phase_i_results = await self._dispatch_phase_i_modules()
        results["phase_i_findings"].extend(phase_i_results)
        results["summary"]["phase_i_findings_count"] = len(phase_i_results)
        
        return results
    
    async def _dispatch_phase_i_modules(self) -> List[Dict[str, Any]]:
        """使用消息隊列派發 Phase I 高價值模組任務"""
        findings = []
        task_futures = []
        
        try:
            # 為每個目標創建掃描任務
            for target_url in self.config.targets:
                self.logger.info(f"派發任務給目標: {target_url}")
                
                # 1. 客戶端授權繞過檢測任務
                csab_task = FunctionTaskSchema(
                    task_id=f"csab_{uuid4().hex[:8]}",
                    module_name=ModuleName.FUNC_CLIENT_AUTH_BYPASS,
                    target=FunctionTaskTarget(url=target_url),
                    context=FunctionTaskContext(
                        session_id=self.config.session_id,
                        scan_id=self.config.scan_id
                    )
                )
                task_futures.append(self.dispatcher.dispatch_task(csab_task))
                
                # 2. SSRF 檢測任務 (Go)
                ssrf_task = FunctionTaskSchema(
                    task_id=f"ssrf_{uuid4().hex[:8]}",
                    module_name=ModuleName.FUNC_SSRF_GO,
                    target=FunctionTaskTarget(url=target_url),
                    context=FunctionTaskContext(
                        session_id=self.config.session_id,
                        scan_id=self.config.scan_id
                    )
                )
                task_futures.append(self.dispatcher.dispatch_task(ssrf_task))
            
            # 等待所有任務完成並收集結果
            task_results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            for result in task_results:
                if isinstance(result, Exception):
                    self.logger.error(f"任務執行錯誤: {result}")
                    continue
                    
                if hasattr(result, 'success') and result.success and hasattr(result, 'findings'):
                    findings.extend([{
                        "module": result.module_name,
                        "target": result.target.url if hasattr(result, 'target') else 'unknown',
                        "finding": finding,
                        "task_id": result.task_id if hasattr(result, 'task_id') else 'unknown'
                    } for finding in result.findings])
            
        except Exception as e:
            self.logger.error(f"Phase I 模組派發錯誤: {e}")
        
        return findings
    
    async def _dispatch_scan_task(self, target: str, scan_type: str) -> None:
        """派發單個掃描任務"""
        task_id = f"{scan_type}_scan_{uuid4().hex[:8]}"
        
        try:
            if scan_type == 'aiva':
                # 派發AIVA主系統掃描任務
                task = FunctionTaskSchema(
                    task_id=task_id,
                    module_name=ModuleName.SCAN,
                    target=FunctionTaskTarget(url=target),
                    context=FunctionTaskContext(
                        session_id=self.config.session_id,
                        scan_id=self.config.scan_id,
                        metadata={"scan_type": "aiva_main"}
                    )
                )
            elif scan_type == 'suite':
                # 派發掃描套件任務
                task = FunctionTaskSchema(
                    task_id=task_id,
                    module_name=ModuleName.SCAN,
                    target=FunctionTaskTarget(url=target),
                    context=FunctionTaskContext(
                        session_id=self.config.session_id,
                        scan_id=self.config.scan_id,
                        metadata={"scan_type": "suite"}
                    )
                )
            else:
                self.logger.warning(f"未知的掃描類型: {scan_type}")
                return
                
            # 追蹤活動任務
            self.active_tasks[task_id] = {
                "task": task,
                "status": "dispatched",
                "timestamp": datetime.now()
            }
            
            # 派發任務
            await self.dispatcher.dispatch_task(task)
            self.logger.info(f"已派發 {scan_type} 掃描任務: {task_id}")
            
        except Exception as e:
            self.logger.error(f"派發 {scan_type} 掃描任務失敗: {e}")
    
    async def _wait_for_completion(self, timeout: float = 300.0) -> None:
        """等待所有任務完成"""
        start_time = datetime.now()
        
        while self.active_tasks:
            # 檢查超時
            if (datetime.now() - start_time).total_seconds() > timeout:
                self.logger.warning(f"掃描任務超時，剩餘任務: {len(self.active_tasks)}")
                break
                
            # 檢查任務狀態
            completed_tasks = []
            for task_id, task_info in self.active_tasks.items():
                # 這裡應該檢查消息隊列中的任務狀態
                # 實際實現中會監聽任務完成消息
                pass
                
            # 移除已完成的任務
            for task_id in completed_tasks:
                del self.active_tasks[task_id]
                
            await asyncio.sleep(1.0)
    
    async def _collect_and_merge_results(self, target: str) -> Dict[str, Any]:
        """收集並合併掃描結果"""
        return {
            "scan_id": self.config.scan_id,
            "session_id": self.config.session_id,
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "scans": self.scan_results.get(self.config.scan_id, []),
            "summary": {
                "total_findings": sum(len(scan.get("findings", [])) 
                                    for scan in self.scan_results.get(self.config.scan_id, [])),
                "scan_types_completed": len(self.scan_results.get(self.config.scan_id, []))
            }
        }
    
    def get_available_scanners(self) -> Dict[str, bool]:
        """獲取可用掃描器狀態 - 異步消息隊列版本"""
        return {
            "message_broker": self.broker is not None,
            "task_dispatcher": self.dispatcher is not None,
            "async_architecture": True,
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