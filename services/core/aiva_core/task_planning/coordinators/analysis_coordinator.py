#!/usr/bin/env python3
"""
Analysis Coordinator

協調分析相關任務
負責代碼分析、漏洞檢測、報告生成
"""

import logging
from typing import Dict, Any, List

from .base_coordinator import BaseCoordinator, CoordinatorTask

logger = logging.getLogger(__name__)


class AnalysisCoordinator(BaseCoordinator):
    """分析協調器
    
    負責協調：
    - 代碼分析 (BioNeuron)
    - 漏洞檢測 (Scanner + BioNeuron)
    - 數據查詢 (DataHub)
    - 報告生成
    """
    
    def __init__(self, module_registry):
        super().__init__(module_registry, name="AnalysisCoordinator")
    
    async def decompose_task(self, task: CoordinatorTask) -> List[Dict[str, Any]]:
        """分解分析任務
        
        Args:
            task: 分析任務
        
        Returns:
            子任務列表
        """
        analysis_type = task.parameters.get("analysis_type", "code")
        target = task.parameters.get("target", "")
        
        logger.info(f"Decomposing analysis task: {analysis_type}")
        
        subtasks = []
        
        if analysis_type == "code":
            # 代碼分析流程
            subtasks.extend(self._create_code_analysis_subtasks(target, task.parameters))
        
        elif analysis_type == "vulnerability":
            # 漏洞分析流程
            subtasks.extend(self._create_vulnerability_analysis_subtasks(target, task.parameters))
        
        elif analysis_type == "experience":
            # 經驗數據分析
            subtasks.extend(self._create_experience_analysis_subtasks(task.parameters))
        
        else:
            # 通用分析
            subtasks.append({
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "ANALYZE",
                    "description": f"Analyze {analysis_type}",
                    "target": target,
                    "analysis_type": analysis_type
                }
            })
        
        return subtasks
    
    def _create_code_analysis_subtasks(
        self, 
        target: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """創建代碼分析子任務"""
        return [
            {
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "ANALYZE",
                    "description": f"Analyze code structure of {target}",
                    "target": target,
                    "analysis_type": "code_structure",
                    "depth": parameters.get("depth", "basic")
                }
            },
            {
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "ANALYZE",
                    "description": f"Detect code vulnerabilities in {target}",
                    "target": target,
                    "analysis_type": "vulnerability_detection"
                }
            }
        ]
    
    def _create_vulnerability_analysis_subtasks(
        self, 
        target: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """創建漏洞分析子任務"""
        return [
            {
                "module_id": "scanner",
                "parameters": {
                    "task_type": "SCAN",
                    "description": f"Scan {target} for vulnerabilities",
                    "target": target,
                    "scan_type": "active",
                    "focus": "vulnerabilities"
                }
            },
            {
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "ANALYZE",
                    "description": "Analyze vulnerability scan results",
                    "context": "vulnerability_analysis",
                    "severity_threshold": parameters.get("severity_threshold", "medium")
                }
            }
        ]
    
    def _create_experience_analysis_subtasks(
        self, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """創建經驗數據分析子任務"""
        return [
            {
                "module_id": "data_hub",
                "parameters": {
                    "task_type": "QUERY",
                    "description": "Query experience data",
                    "query_type": "experiences",
                    "filters": parameters.get("filters", {})
                }
            },
            {
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "ANALYZE",
                    "description": "Analyze experience patterns",
                    "context": "experience_analysis"
                }
            }
        ]
    
    async def aggregate_results(
        self, 
        task: CoordinatorTask, 
        subtask_results: List[Any]
    ) -> Any:
        """聚合分析結果
        
        Args:
            task: 原始任務
            subtask_results: 子任務結果列表
        
        Returns:
            聚合後的分析報告
        """
        logger.info(f"Aggregating analysis results for task {task.task_id}")
        
        analysis_report = {
            "task_id": task.task_id,
            "analysis_type": task.parameters.get("analysis_type", "unknown"),
            "target": task.parameters.get("target", ""),
            "findings": [],
            "summary": {},
            "recommendations": []
        }
        
        # 提取所有結果中的數據
        all_data = []
        for result in subtask_results:
            if hasattr(result, 'data'):
                all_data.append(result.data)
            elif isinstance(result, dict):
                all_data.append(result)
        
        # 根據分析類型處理結果
        analysis_type = task.parameters.get("analysis_type", "code")
        
        if analysis_type == "code":
            analysis_report["findings"] = self._extract_code_findings(all_data)
        elif analysis_type == "vulnerability":
            analysis_report["findings"] = self._extract_vulnerability_findings(all_data)
        elif analysis_type == "experience":
            analysis_report["findings"] = self._extract_experience_findings(all_data)
        
        # 生成摘要
        analysis_report["summary"] = {
            "total_findings": len(analysis_report["findings"]),
            "critical_issues": len([f for f in analysis_report["findings"] if f.get("severity") == "critical"]),
            "analysis_date": "now"
        }
        
        return analysis_report
    
    def _extract_code_findings(self, data_list: List[Any]) -> List[Dict[str, Any]]:
        """從代碼分析數據中提取發現"""
        findings = []
        
        for data in data_list:
            if isinstance(data, dict):
                # 提取漏洞
                vulnerabilities = data.get("vulnerabilities", [])
                for vuln in vulnerabilities:
                    findings.append({
                        "type": "vulnerability",
                        "severity": vuln.get("severity", "unknown"),
                        "description": vuln.get("description", "")
                    })
                
                # 提取代碼問題
                issues = data.get("issues", [])
                for issue in issues:
                    findings.append({
                        "type": "code_issue",
                        "severity": issue.get("severity", "low"),
                        "description": issue.get("description", "")
                    })
        
        return findings
    
    def _extract_vulnerability_findings(self, data_list: List[Any]) -> List[Dict[str, Any]]:
        """從漏洞分析數據中提取發現"""
        findings = []
        
        for data in data_list:
            if isinstance(data, dict):
                scan_results = data.get("scan_results", [])
                for result in scan_results:
                    findings.append({
                        "type": "vulnerability",
                        "severity": result.get("severity", "medium"),
                        "cve": result.get("cve", "N/A"),
                        "description": result.get("description", "")
                    })
        
        return findings
    
    def _extract_experience_findings(self, data_list: List[Any]) -> List[Dict[str, Any]]:
        """從經驗數據中提取發現"""
        findings = []
        
        for data in data_list:
            if isinstance(data, dict):
                experiences = data.get("experiences", [])
                for exp in experiences:
                    findings.append({
                        "type": "experience_pattern",
                        "attack_type": exp.get("attack_type", ""),
                        "success_rate": exp.get("success_rate", 0.0),
                        "description": exp.get("description", "")
                    })
        
        return findings
