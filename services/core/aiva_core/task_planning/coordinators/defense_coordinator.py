#!/usr/bin/env python3
"""
Defense Coordinator

協調防禦相關任務
負責系統安全監控和防護
"""

import logging
from typing import Dict, Any, List

from .base_coordinator import BaseCoordinator, CoordinatorTask

logger = logging.getLogger(__name__)


class DefenseCoordinator(BaseCoordinator):
    """防禦協調器
    
    負責協調：
    - 被動掃描和監控 (Scanner)
    - 威脅分析 (BioNeuron)
    - 防護措施建議
    """
    
    def __init__(self, module_registry):
        super().__init__(module_registry, name="DefenseCoordinator")
    
    async def decompose_task(self, task: CoordinatorTask) -> List[Dict[str, Any]]:
        """分解防禦任務
        
        防禦流程：
        1. Scanner 被動監控
        2. BioNeuron 分析威脅
        3. 生成防護建議
        
        Args:
            task: 防禦任務
        
        Returns:
            子任務列表
        """
        target = task.parameters.get("target", "localhost")
        
        logger.info(f"Decomposing defense task for target: {target}")
        
        subtasks = []
        
        # 1. 被動掃描監控
        subtasks.append({
            "module_id": "scanner",
            "parameters": {
                "task_type": "SCAN",
                "description": f"Passive monitoring of {target}",
                "target": target,
                "scan_type": "passive",
                "monitoring_mode": True
            }
        })
        
        # 2. BioNeuron 威脅分析
        subtasks.append({
            "module_id": "bio_neuron",
            "parameters": {
                "task_type": "ANALYZE",
                "description": "Analyze potential threats",
                "context": "defense",
                "analysis_type": "threat_detection"
            }
        })
        
        # 3. 生成防護建議
        subtasks.append({
            "module_id": "bio_neuron",
            "parameters": {
                "task_type": "DECIDE",
                "description": "Generate defense recommendations",
                "context": "defense_strategy",
                "objective": "protection"
            }
        })
        
        return subtasks
    
    async def aggregate_results(
        self, 
        task: CoordinatorTask, 
        subtask_results: List[Any]
    ) -> Any:
        """聚合防禦結果
        
        Args:
            task: 原始任務
            subtask_results: 子任務結果列表
        
        Returns:
            聚合後的防禦報告
        """
        logger.info(f"Aggregating defense results for task {task.task_id}")
        
        defense_report = {
            "task_id": task.task_id,
            "target": task.parameters.get("target", ""),
            "monitoring_results": {},
            "threat_analysis": {},
            "recommendations": []
        }
        
        # 解析子任務結果
        if len(subtask_results) > 0:
            # 監控結果
            monitoring = subtask_results[0]
            if hasattr(monitoring, 'data'):
                defense_report["monitoring_results"] = monitoring.data
        
        if len(subtask_results) > 1:
            # 威脅分析
            threat = subtask_results[1]
            if hasattr(threat, 'data'):
                defense_report["threat_analysis"] = threat.data
        
        if len(subtask_results) > 2:
            # 防護建議
            recommendations = subtask_results[2]
            if hasattr(recommendations, 'data'):
                data = recommendations.data
                if isinstance(data, dict):
                    defense_report["recommendations"] = data.get("recommendations", [])
        
        # 評估風險等級
        defense_report["risk_level"] = self._assess_risk_level(defense_report)
        
        return defense_report
    
    def _assess_risk_level(self, report: Dict[str, Any]) -> str:
        """評估風險等級"""
        threat_analysis = report.get("threat_analysis", {})
        
        # 簡單的風險評估邏輯
        if isinstance(threat_analysis, dict):
            threats = threat_analysis.get("threats_detected", 0)
            if threats > 10:
                return "critical"
            elif threats > 5:
                return "high"
            elif threats > 0:
                return "medium"
        
        return "low"
