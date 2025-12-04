#!/usr/bin/env python3
"""
Attack Coordinator

協調攻擊相關任務
負責規劃和執行攻擊鏈

整合 TaskManager 實現 checkpoint 恢復與狀態持久化
"""

import logging
from typing import Dict, Any, List, Optional

from .base_coordinator import BaseCoordinator, CoordinatorTask
from aiva_core.task_planning.persistence import TaskManager, TaskStatus

logger = logging.getLogger(__name__)


class AttackCoordinator(BaseCoordinator):
    """攻擊協調器
    
    負責協調：
    - 攻擊路徑規劃 (BioNeuron)
    - 掃描和偵察 (Scanner)
    - 漏洞利用 (Exploiter)
    
    整合 TaskManager：
    - 持久化任務狀態到 SQLite
    - 支持 checkpoint 恢復
    - 記錄執行軌跡（雙閉環支持）
    async def decompose_task(self, task: CoordinatorTask) -> List[Dict[str, Any]]:
        """分解攻擊任務
        
        攻擊流程：
        1. BioNeuron 規劃攻擊路徑
        2. Scanner 掃描目標
        3. BioNeuron 決策下一步
        4. Exploiter 執行利用
        
        整合 TaskManager：
        - 創建持久化任務記錄
        - 啟動 checkpoint 恢復（首次）
        
        Args:
            task: 攻擊任務
        
        Returns:
            子任務列表
        """
        # 確保 TaskManager 已啟動（checkpoint 恢復）
        if not self._task_manager_started:
            await self.task_manager.start()
            self._task_manager_started = True
        
        target = task.parameters.get("target", "")
        attack_type = task.parameters.get("attack_type", "auto")
        
        # 創建持久化任務記錄
        try:
            scan_config = {
                "attack_type": attack_type,
                "scan_depth": task.parameters.get("scan_depth", "basic"),
                "stealth": task.parameters.get("stealth", False),
                "risk_level": task.parameters.get("risk_level", "medium")
            }
            task_id = await self.task_manager.create_scan_task(
                url=target,
                priority=task.parameters.get("priority", 5),
                scan_config=scan_config
            )
            # 記錄到任務元數據
            task.parameters["persistent_task_id"] = task_id
            logger.info(f"Created persistent task {task_id} for target {target}")
        except Exception as e:
            logger.error(f"Failed to create persistent task: {e}")
            # 繼續執行，即使持久化失敗
        
        logger.info(f"Decomposing attack task for target: {target}")
        
        Returns:
            子任務列表
        """
        target = task.parameters.get("target", "")
        attack_type = task.parameters.get("attack_type", "auto")
        
        logger.info(f"Decomposing attack task for target: {target}")
        
        subtasks = []
        
        # 1. 使用 BioNeuron 規劃攻擊
        subtasks.append({
            "module_id": "bio_neuron",
            "parameters": {
                "task_type": "PLAN",
                "description": f"Plan attack strategy for {target}",
                "target": target,
                "attack_type": attack_type,
                "objective": "reconnaissance"
            }
        })
        
        # 2. 使用 Scanner 進行掃描
        scan_mode = "passive" if task.parameters.get("stealth", False) else "active"
        subtasks.append({
            "module_id": "scanner",
            "parameters": {
                "task_type": "SCAN",
                "description": f"Scan target {target}",
                "target": target,
                "scan_type": scan_mode,
                "depth": task.parameters.get("scan_depth", "basic")
            }
        })
        
        # 3. 使用 BioNeuron 分析掃描結果並決策
        subtasks.append({
            "module_id": "bio_neuron",
            "parameters": {
                "task_type": "ANALYZE",
                "description": "Analyze scan results and decide exploitation strategy",
                "context": "attack_planning",
                "previous_results": "scan_data"
            }
        })
        
        # 4. 使用 Exploiter 執行漏洞利用
        if not task.parameters.get("scan_only", False):
            subtasks.append({
                "module_id": "exploiter",
                "parameters": {
                    "task_type": "EXPLOIT",
                    "description": f"Execute exploit against {target}",
                    "target": target,
                    "exploit_type": attack_type,
                    "risk_level": task.parameters.get("risk_level", "medium")
                }
            })
        
        return subtasks
    
    async def aggregate_results(
        self, 
        task: CoordinatorTask, 
        subtask_results: List[Any]
    ) -> Any:
        """聚合攻擊結果
        
        整合 TaskManager：
        - 更新任務狀態為 COMPLETED
        - 記錄執行軌跡（雙閉環支持）
        
        Args:
            task: 原始任務
            subtask_results: 子任務結果列表
        
        Returns:
            聚合後的攻擊報告
        """
        logger.info(f"Aggregating attack results for task {task.task_id}")
        
        attack_report = {
            "task_id": task.task_id,
            "target": task.parameters.get("target", ""),
            "attack_type": task.parameters.get("attack_type", "auto"),
            "stages": []
        }
        
        # 解析子任務結果
        for i, result in enumerate(subtask_results):
            stage_name = self._get_stage_name(i)
            
            if hasattr(result, 'data'):
                # AIResult 對象
                stage_data = {
                    "stage": stage_name,
                    "success": result.success,
                    "data": result.data,
                    "execution_time": result.execution_time,
                    "error": result.error
                }
            else:
                # 字典或其他類型
                stage_data = {
                    "stage": stage_name,
                    "success": not ("error" in str(result).lower()),
                    "data": result
                }
            
            attack_report["stages"].append(stage_data)
        
        # 判斷整體成功
        attack_report["overall_success"] = all(
            stage.get("success", False) 
            for stage in attack_report["stages"]
        )
        
        # 提取關鍵發現
        attack_report["findings"] = self._extract_findings(attack_report["stages"])
        
        # 更新持久化任務狀態
        persistent_task_id = task.parameters.get("persistent_task_id")
        if persistent_task_id:
            try:
                final_status = TaskStatus.COMPLETED if attack_report["overall_success"] else TaskStatus.FAILED
                await self.task_manager.storage.update_target_status(
                    target_id=persistent_task_id,
                    status=final_status,
                    error_message=None if attack_report["overall_success"] else "Attack stages failed"
                )
                logger.info(f"Updated persistent task {persistent_task_id} to {final_status}")
            except Exception as e:
                logger.error(f"Failed to update persistent task status: {e}")
        
        return attack_report
    
    def _get_stage_name(self, index: int) -> str:
        """獲取階段名稱"""
        stage_names = [
            "planning",
            "scanning",
            "analysis",
            "exploitation"
        ]
        return stage_names[index] if index < len(stage_names) else f"stage_{index}"
    
    def _extract_findings(self, stages: List[Dict[str, Any]]) -> List[str]:
        """從階段結果中提取關鍵發現"""
        findings = []
        
        for stage in stages:
            stage_name = stage.get("stage", "unknown")
            data = stage.get("data", {})
            
            if stage_name == "scanning":
                # 提取掃描發現
                if isinstance(data, dict):
                    scan_results = data.get("scan_results", [])
                    if scan_results:
                        findings.append(f"Found {len(scan_results)} potential vulnerabilities")
            
            elif stage_name == "exploitation":
                # 提取利用結果
                if isinstance(data, dict):
                    exploits = data.get("exploits", [])
                    successful = [e for e in exploits if e.get("success")]
                    if successful:
                        findings.append(f"Successfully exploited {len(successful)} vulnerabilities")
        
        return findings
