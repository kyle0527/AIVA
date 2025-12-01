"""Task Manager - 任務管理器 (P0-3 實現)

實現任務調度、狀態管理和斷點續傳功能，符合 aiva_common v2.0 規範

設計理念：
1. 啟動恢復：程序啟動時自動恢復未完成的任務
2. 狀態追蹤：實時更新任務狀態到 SQLite
3. 進度保存：關鍵步驟完成後立即保存進度
4. 雙閉環整合：記錄執行軌跡供 ExternalLoopConnector 分析

功能：
- 任務創建和調度
- 狀態自動持久化
- 斷點續傳支持
- 執行軌跡記錄
- 漏洞發現管理

參考：
- 內容截斷問題分析.md § 第四階段：狀態持久化與斷點續傳
- AI自動化閉環完整實施計劃.md § 架構設計方案
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC

# ✅ 使用 aiva_common 統一日誌和錯誤處理
from aiva_common.utils.logging import get_logger
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

from .task_storage import TaskStorage, TaskStatus

logger = get_logger(__name__)


class TaskManager:
    """任務管理器 (v2.0 合規版本)
    
    功能：
    1. 任務生命週期管理（創建、調度、完成）
    2. 狀態自動持久化到 SQLite
    3. 斷點續傳（啟動時恢復未完成任務）
    4. 進度追蹤和報告
    5. 執行軌跡記錄（支持雙閉環偏差分析）
    
    與雙閉環的整合：
    - 內閉環：使用 CapabilityRegistry 查詢可用能力
    - 外閉環：記錄執行軌跡供 ExternalLoopConnector 分析
    """
    
    def __init__(
        self,
        storage: Optional[TaskStorage] = None,
        max_concurrent_tasks: int = 5
    ):
        """初始化任務管理器
        
        Args:
            storage: 任務存儲實例（如果為 None 則創建新實例）
            max_concurrent_tasks: 最大並發任務數
        """
        self.storage = storage or TaskStorage()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: Dict[int, asyncio.Task] = {}
        self.task_results: Dict[int, Dict[str, Any]] = {}
        
        logger.info(f"✅ TaskManager initialized (max_concurrent: {max_concurrent_tasks})")
    
    def start(self) -> None:
        """啟動任務管理器（恢復未完成的任務）
        
        這是斷點續傳的關鍵方法：
        1. 查詢所有 PENDING 和 SCANNING 狀態的任務
        2. 按優先級排序
        3. 重新調度執行
        """
        logger.info("🔄 TaskManager starting...")
        
        # 獲取待處理的任務
        pending_targets = self.storage.get_pending_targets()
        
        if pending_targets:
            logger.info(f"📋 Found {len(pending_targets)} pending tasks, resuming...")
            
            for target in pending_targets:
                target_id = target['id']
                url = target['url']
                
                # 檢查是否已在運行
                if target_id in self.active_tasks:
                    logger.warning(f"⚠️  Task {target_id} already running, skipping")
                    continue
                
                # 恢復任務
                logger.info(f"🔄 Resuming task {target_id}: {url}")
                
                # 獲取之前的進度
                progress = self.storage.get_scan_progress(target_id)
                if progress:
                    logger.info(f"   Previous progress: {progress['current_stage']}, "
                              f"{progress['payloads_tried']} payloads tried")
                
                # 重新調度（這裡需要與實際的掃描執行器整合）
                # 暫時標記為 PENDING，等待 execute_scan 調用
                if target['status'] == TaskStatus.SCANNING:
                    self.storage.update_target_status(target_id, TaskStatus.PENDING)
        else:
            logger.info("✅ No pending tasks found")
    
    async def create_scan_task(
        self,
        url: str,
        priority: int = 5,
        scan_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """創建新的掃描任務
        
        Args:
            url: 目標 URL
            priority: 優先級 (1-10, 預設 5)
            scan_config: 掃描配置
            metadata: 額外元數據
            
        Returns:
            int: 任務 ID
        """
        # 驗證 URL
        if not url or not url.strip():
            raise ValueError("無效的 URL")
            
        # 異步偵測網路連接（簡單測試）
        try:
            await asyncio.wait_for(self._test_url_connectivity(url), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ URL 連接測試逾時: {url}")
        except Exception as e:
            logger.debug(f"URL 連接測試失敗: {e}")
            
        # 異步偵測網路連接（簡單測試）
        try:
            await asyncio.wait_for(self._test_url_connectivity(url), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ URL 連接測試逾時: {url}")
        except Exception as e:
            logger.debug(f"URL 連接測試失敗: {e}")
        
        # 創建目標記錄
        target_id = self.storage.create_target(url, priority, metadata)
        
        # 初始化進度記錄
        self.storage.update_scan_progress(
            target_id=target_id,
            current_stage="initialized",
            scan_config=scan_config
        )
        
        logger.info(f"✅ Scan task created: {url} (ID: {target_id}, Priority: {priority})")
        return target_id
    
    async def execute_scan(
        self,
        target_id: int,
        scanner_executor,
        plan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行掃描任務（帶狀態持久化）
        
        Args:
            target_id: 目標 ID
            scanner_executor: 掃描執行器（callable，接收 url 和 config）
            plan_id: 執行計劃 ID（用於雙閉環追蹤）
            
        Returns:
            掃描結果
        """
        # 獲取目標信息
        targets = self.storage.get_pending_targets()
        target = next((t for t in targets if t['id'] == target_id), None)
        
        if not target:
            raise AIVAError(
                message=f"Target {target_id} not found or not in pending state",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH
            )
        
        url = target['url']
        
        try:
            # ✅ 狀態更新：開始掃描
            self.storage.update_target_status(target_id, TaskStatus.SCANNING)
            self.storage.update_scan_progress(
                target_id=target_id,
                current_stage="scanning"
            )
            
            logger.info(f"🚀 Starting scan for target {target_id}: {url}")
            
            # 記錄執行步驟（雙閉環支持）
            step_start = asyncio.get_event_loop().time()
            
            # ✅ 執行實際掃描
            scan_result = await scanner_executor(url, target.get('metadata', {}))
            
            step_end = asyncio.get_event_loop().time()
            execution_time = step_end - step_start
            
            # ✅ 記錄執行軌跡（支持雙閉環偏差分析）
            self.storage.record_execution_step(
                target_id=target_id,
                plan_id=plan_id,
                step_index=1,
                action="scan_execution",
                result=scan_result,
                success=True,
                execution_time=execution_time
            )
            
            # ✅ 處理掃描結果
            vulnerabilities = scan_result.get('vulnerabilities', [])
            
            # 記錄發現的漏洞
            for vuln in vulnerabilities:
                self.storage.record_finding(
                    target_id=target_id,
                    vulnerability_type=vuln.get('type', 'unknown'),
                    severity=vuln.get('severity', 'info'),
                    confidence=vuln.get('confidence', 0.5),
                    evidence=vuln.get('evidence'),
                    payload=vuln.get('payload')
                )
            
            # ✅ 更新進度
            self.storage.update_scan_progress(
                target_id=target_id,
                current_stage="completed",
                findings_count=len(vulnerabilities)
            )
            
            # ✅ 狀態更新：完成
            self.storage.update_target_status(target_id, TaskStatus.COMPLETED)
            
            logger.info(f"✅ Scan completed for target {target_id}: {len(vulnerabilities)} findings")
            
            # 保存結果
            self.task_results[target_id] = scan_result
            
            return scan_result
            
        except Exception as e:
            logger.error(f"❌ Scan failed for target {target_id}: {e}", exc_info=True)
            
            # ✅ 記錄失敗軌跡
            self.storage.record_execution_step(
                target_id=target_id,
                plan_id=plan_id,
                step_index=1,
                action="scan_execution",
                result={"error": str(e)},
                success=False,
                execution_time=0.0
            )
            
            # ✅ 狀態更新：失敗
            self.storage.update_target_status(
                target_id,
                TaskStatus.FAILED,
                error_message=str(e)
            )
            
            raise
    
    async def execute_with_checkpoint(
        self,
        target_id: int,
        steps: List[Dict[str, Any]],
        executor_map: Dict[str, Any],
        plan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行多步驟任務並保存檢查點（斷點續傳支持）
        
        Args:
            target_id: 目標 ID
            steps: 執行步驟列表
            executor_map: 執行器映射
            plan_id: 執行計劃 ID
            
        Returns:
            執行結果
        """
        # 獲取之前的進度
        progress = self.storage.get_scan_progress(target_id)
        
        # 確定從哪個步驟開始
        start_step = 0
        if progress and progress['current_stage'] != 'initialized':
            # 找到上次執行到的步驟
            traces = self.storage.get_execution_trace(target_id, plan_id)
            if traces:
                start_step = max(t['step_index'] for t in traces) + 1
                logger.info(f"🔄 Resuming from step {start_step}")
        
        results = []
        
        for i, step in enumerate(steps[start_step:], start=start_step):
            step_name = step.get('name', f'step_{i}')
            executor_name = step.get('executor', 'default')  # 提供默認值
            
            logger.info(f"📍 Executing step {i}: {step_name}")
            
            # ✅ 更新進度（檢查點）
            self.storage.update_scan_progress(
                target_id=target_id,
                current_stage=step_name,
                payloads_tried=i
            )
            
            try:
                # 獲取執行器
                if isinstance(executor_name, str):
                    executor = executor_map.get(executor_name)
                else:
                    executor = None
                if not executor:
                    raise ValueError(f"Executor '{executor_name}' not found")
                
                # 執行步驟
                step_start = asyncio.get_event_loop().time()
                result = await executor(step.get('params', {}))
                step_end = asyncio.get_event_loop().time()
                
                # ✅ 記錄執行軌跡
                self.storage.record_execution_step(
                    target_id=target_id,
                    plan_id=plan_id,
                    step_index=i,
                    action=step_name,
                    result=result,
                    success=True,
                    execution_time=step_end - step_start
                )
                
                results.append({
                    "step": step_name,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"❌ Step {i} failed: {e}")
                
                # ✅ 記錄失敗
                self.storage.record_execution_step(
                    target_id=target_id,
                    plan_id=plan_id,
                    step_index=i,
                    action=step_name,
                    result={"error": str(e)},
                    success=False,
                    execution_time=0.0
                )
                
                # 根據策略決定是否繼續
                if not step.get('continue_on_error', False):
                    raise
                
                results.append({
                    "step": step_name,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "steps_completed": len(results),
            "results": results,
            "success": all(r.get('success', False) for r in results)
        }
    
    def get_task_status(self, target_id: int) -> Optional[Dict[str, Any]]:
        """獲取任務狀態
        
        Args:
            target_id: 目標 ID
            
        Returns:
            任務狀態信息
        """
        targets = self.storage.get_pending_targets()
        target = next((t for t in targets if t['id'] == target_id), None)
        
        if target:
            progress = self.storage.get_scan_progress(target_id)
            findings = self.storage.get_findings(target_id)
            
            return {
                "target": target,
                "progress": progress,
                "findings_count": len(findings),
                "findings": findings[:5]  # 只返回前 5 個
            }
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息
        
        Returns:
            統計數據
        """
        stats = self.storage.get_statistics()
        stats['active_tasks'] = len(self.active_tasks)
        return stats
    
    def pause_task(self, target_id: int) -> None:
        """暫停任務
        
        Args:
            target_id: 目標 ID
        """
        self.storage.update_target_status(target_id, TaskStatus.PAUSED)
        
        if target_id in self.active_tasks:
            self.active_tasks[target_id].cancel()
            del self.active_tasks[target_id]
        
        logger.info(f"⏸️  Task {target_id} paused")
    
    def resume_task(self, target_id: int) -> None:
        """恢復任務
        
        Args:
            target_id: 目標 ID
        """
        self.storage.update_target_status(target_id, TaskStatus.PENDING)
        logger.info(f"▶️  Task {target_id} resumed")
    
    def cleanup(self) -> None:
        """清理資源"""
        # 取消所有活動任務
        for task in self.active_tasks.values():
            task.cancel()
        
        self.active_tasks.clear()
        self.task_results.clear()
        
        logger.info("✅ TaskManager cleaned up")

    async def _test_url_connectivity(self, url: str) -> bool:
        """測試 URL 連接性
        
        Args:
            url: 要測試的 URL
            
        Returns:
            bool: 連接是否成功
        """
        try:
            # 簡單的連接測試
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                    return response.status < 400
        except Exception:
            # 如果 aiohttp 不可用，使用基本的 socket 測試
            import socket
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            host = parsed.hostname or 'localhost'
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), 
                    timeout=3.0
                )
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                return False
