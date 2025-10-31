"""
HackingTool SQL 注入檢測引擎
整合 HackingTool 的 SQL 注入工具到 AIVA 的檢測流程中

這個引擎作為 AIVA function_sqli 模組的擴展，
提供基於 HackingTool 工具的額外檢測能力。
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import subprocess
from pathlib import Path

import httpx

from services.aiva_common.utils.logging import get_logger
from services.aiva_common.utils.ids import new_id
from services.aiva_common.enums import Severity, Confidence
from services.aiva_common.schemas import (
    FindingEvidence, FindingImpact, FindingRecommendation, 
    FindingTarget, Vulnerability, FunctionTaskPayload
)

from ..detection_models import DetectionResult
from ..schemas import SqliDetectionResult, SqliTelemetry
from ..config import SqliConfig
from ..hackingtool_config import (
    HackingToolSQLIntegrator, SQLToolType, 
    HACKINGTOOL_SQL_CONFIGS, sql_integrator
)

logger = get_logger(__name__)


class HackingToolDetectionEngine:
    """HackingTool SQL 注入檢測引擎"""
    
    def __init__(self, config: SqliConfig):
        self.config = config
        self.integrator = sql_integrator
        self.trace_id = new_id("hackingtool_sqli")
        
        logger.info("HackingTool SQL 檢測引擎已初始化", trace_id=self.trace_id)
    
    async def detect(self, task: FunctionTaskPayload, client: httpx.AsyncClient) -> List[DetectionResult]:
        """執行 HackingTool SQL 注入檢測"""
        results = []
        
        # 從 task 中提取目標 URL
        target = task.url
        
        logger.info("開始 HackingTool SQL 檢測", 
                   extra={"target": target, "trace_id": self.trace_id})
        
        # 獲取啟用的工具
        enabled_tools = self.integrator.get_enabled_tools()
        
        if not enabled_tools:
            logger.warning("沒有可用的 HackingTool SQL 工具", trace_id=self.trace_id)
            return results
        
        # 並行執行工具檢測
        detection_tasks = []
        for tool_name in enabled_tools:
            detection_task = asyncio.create_task(
                self._run_tool_detection(tool_name, target, task)
            )
            detection_tasks.append(detection_task)
        
        # 等待所有檢測完成
        tool_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # 處理結果，轉換為 DetectionResult
        for i, result in enumerate(tool_results):
            tool_name = enabled_tools[i]
            
            if isinstance(result, Exception):
                logger.error(f"工具 {tool_name} 檢測失敗: {result}", 
                           trace_id=self.trace_id)
                continue
            
            if result:
                # 轉換 SqliDetectionResult 為 DetectionResult
                for sqli_result in result:
                    detection_result = self._convert_to_detection_result(sqli_result)
                    if detection_result:
                        results.append(detection_result)
        
        logger.info(f"HackingTool 檢測完成，發現 {len(results)} 個結果", 
                   trace_id=self.trace_id)
        
        return results
    
    async def _run_tool_detection(self, tool_name: str, target: str, task: FunctionTaskPayload) -> List[SqliDetectionResult]:
        """執行單個工具的檢測"""
        config = HACKINGTOOL_SQL_CONFIGS.get(tool_name)
        if not config:
            return []
        
        logger.debug(f"執行工具檢測: {tool_name}", 
                    extra={"target": target, "trace_id": self.trace_id})
        
        try:
            # 執行工具
            execution_result = await self._execute_tool(tool_name, target)
            
            if not execution_result.get("success", False):
                logger.warning(f"工具 {tool_name} 執行失敗: {execution_result.get('error', 'Unknown error')}")
                return []
            
            # 解析結果
            detection_results = self._parse_tool_output(
                tool_name, execution_result, target
            )
            
            return detection_results
            
        except Exception as e:
            logger.error(f"工具 {tool_name} 檢測過程異常: {e}", trace_id=self.trace_id)
            return []
    
    async def _execute_tool(self, tool_name: str, target: str) -> Dict[str, Any]:
        """異步執行工具命令"""
        config = HACKINGTOOL_SQL_CONFIGS[tool_name]
        
        if not config.run_commands:
            return {"success": False, "error": "No run commands defined"}
        
        try:
            # 格式化命令
            cmd = config.run_commands[0].format(target=target)
            
            # 使用 asyncio.create_subprocess_shell 異步執行
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # 等待執行完成，設置超時
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {"success": False, "error": f"Execution timeout after {config.timeout_seconds}s"}
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "returncode": process.returncode,
                "command": cmd
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_tool_output(self, tool_name: str, execution_result: Dict[str, Any], target: str) -> List[SqliDetectionResult]:
        """解析工具輸出，生成檢測結果"""
        config = HACKINGTOOL_SQL_CONFIGS[tool_name]
        results = []
        
        stdout = execution_result.get("stdout", "")
        stderr = execution_result.get("stderr", "")
        combined_output = f"{stdout}\n{stderr}"
        
        # 使用正則表達式匹配漏洞
        vulnerabilities_found = []
        
        for pattern_name, pattern in config.result_patterns.items():
            matches = re.finditer(pattern, combined_output, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                vulnerability_info = {
                    "type": pattern_name,
                    "match": match.group(0),
                    "groups": match.groups(),
                    "position": match.span()
                }
                vulnerabilities_found.append(vulnerability_info)
        
        # 為每個發現的漏洞創建檢測結果
        for vuln_info in vulnerabilities_found:
            result = self._create_detection_result(
                tool_name, config, vuln_info, target, combined_output
            )
            if result:
                results.append(result)
        
        return results
    
    def _create_detection_result(
        self, 
        tool_name: str, 
        config: Any,
        vuln_info: Dict[str, Any], 
        target: str, 
        full_output: str
    ) -> Optional[SqliDetectionResult]:
        """創建標準化的檢測結果"""
        
        try:
            # 確定嚴重程度
            severity = self._determine_severity(vuln_info["type"])
            
            # 確定置信度
            confidence_score = config.confidence_mapping.get(
                vuln_info["type"], 0.7
            )
            
            # 提取載荷資訊
            payload_used = self._extract_payload(vuln_info, full_output)
            
            # 提取數據庫指紋
            db_fingerprint = self._extract_db_fingerprint(full_output)
            
            # 創建漏洞對象
            vulnerability = Vulnerability(
                id=new_id("vuln"),
                cve_id=None,  # HackingTool 通常不提供 CVE
                title=f"SQL Injection detected by {config.title}",
                description=f"SQL injection vulnerability found using {tool_name}",
                severity=severity,
                confidence=Confidence.HIGH if confidence_score >= 0.8 else Confidence.MEDIUM,
                references=[config.project_url] if config.project_url else []
            )
            
            # 創建證據
            evidence = FindingEvidence(
                description=f"Tool {tool_name} detected SQL injection",
                raw_output=full_output[:1000],  # 限制輸出長度
                detection_method=f"hackingtool_{tool_name}",
                timestamps={"detected_at": datetime.now().isoformat()}
            )
            
            # 創建影響評估
            impact = FindingImpact(
                confidentiality="HIGH",
                integrity="HIGH", 
                availability="MEDIUM",
                description="SQL injection can lead to unauthorized data access, modification, or deletion"
            )
            
            # 創建修復建議
            recommendation = FindingRecommendation(
                description="Implement proper input validation and parameterized queries",
                remediation_steps=[
                    "Use parameterized queries or prepared statements",
                    "Validate and sanitize all user input",
                    "Implement proper error handling",
                    "Apply principle of least privilege to database accounts",
                    "Regular security testing and code review"
                ],
                references=[
                    "https://owasp.org/www-community/attacks/SQL_Injection",
                    "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html"
                ]
            )
            
            # 創建目標資訊
            finding_target = FindingTarget(
                url=target,
                parameter=self._extract_parameter(vuln_info),
                method="GET"  # 默認，實際應根據工具結果確定
            )
            
            # 創建檢測結果
            result = SqliDetectionResult(
                is_vulnerable=True,
                vulnerability=vulnerability,
                evidence=evidence,
                impact=impact,
                recommendation=recommendation,
                target=finding_target,
                detection_method=f"hackingtool_{tool_name}",
                payload_used=payload_used,
                confidence_score=confidence_score,
                db_fingerprint=db_fingerprint,
                response_time=0.0  # HackingTool 通常不提供響應時間
            )
            
            return result
            
        except Exception as e:
            logger.error(f"創建檢測結果時發生錯誤: {e}", trace_id=self.trace_id)
            return None
    
    def _determine_severity(self, vuln_type: str) -> Severity:
        """根據漏洞類型確定嚴重程度"""
        severity_map = {
            "vulnerable": Severity.HIGH,
            "injectable": Severity.HIGH,
            "time_based": Severity.MEDIUM,
            "blind_sqli": Severity.MEDIUM,
            "error_based": Severity.HIGH,
            "nosql_injection": Severity.HIGH,
            "possible": Severity.LOW
        }
        
        return severity_map.get(vuln_type, Severity.MEDIUM)
    
    def _extract_payload(self, vuln_info: Dict[str, Any], full_output: str) -> str:
        """從輸出中提取使用的載荷"""
        # 嘗試從匹配組中獲取載荷
        if vuln_info["groups"]:
            for group in vuln_info["groups"]:
                if group and len(group) > 5:  # 假設載荷長度大於 5
                    return group
        
        # 嘗試從完整輸出中提取載荷
        payload_patterns = [
            r"Payload:\s*(.+)",
            r"Using payload:\s*(.+)",
            r"Injected:\s*(.+)"
        ]
        
        for pattern in payload_patterns:
            match = re.search(pattern, full_output, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown payload"
    
    def _extract_db_fingerprint(self, full_output: str) -> Optional[str]:
        """從輸出中提取數據庫指紋"""
        db_patterns = [
            r"back-end DBMS:\s*(.+)",
            r"Database:\s*(.+)",
            r"DBMS:\s*(.+)",
            r"database management system:\s*(.+)"
        ]
        
        for pattern in db_patterns:
            match = re.search(pattern, full_output, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_parameter(self, vuln_info: Dict[str, Any]) -> Optional[str]:
        """從漏洞資訊中提取參數名稱"""
        if vuln_info["groups"]:
            for group in vuln_info["groups"]:
                if group and len(group) < 50 and ' ' not in group:  # 假設參數名不含空格且較短
                    return group
        
        return None
    
    def get_tool_status(self) -> Dict[str, Any]:
        """獲取所有工具的狀態"""
        status = {
            "available_tools": [],
            "enabled_tools": [],
            "tool_details": {}
        }
        
        for tool_name, config in HACKINGTOOL_SQL_CONFIGS.items():
            is_available = self.integrator.check_tool_availability(tool_name)
            
            if is_available:
                status["available_tools"].append(tool_name)
                
                if config.enable_by_default:
                    status["enabled_tools"].append(tool_name)
            
            status["tool_details"][tool_name] = {
                "title": config.title,
                "type": config.tool_type.value,
                "available": is_available,
                "enabled": config.enable_by_default,
                "priority": config.priority,
                "project_url": config.project_url
            }
        
        return status
    
    def install_missing_tools(self) -> Dict[str, bool]:
        """安裝缺失的工具"""
        installation_results = {}
        
        for tool_name in HACKINGTOOL_SQL_CONFIGS:
            if not self.integrator.check_tool_availability(tool_name):
                logger.info(f"嘗試安裝工具: {tool_name}", trace_id=self.trace_id)
                
                success = self.integrator.install_tool(tool_name)
                installation_results[tool_name] = success
                
                if success:
                    logger.info(f"工具 {tool_name} 安裝成功", trace_id=self.trace_id)
                else:
                    logger.error(f"工具 {tool_name} 安裝失敗", trace_id=self.trace_id)
        
        return installation_results
    
    def _convert_to_detection_result(self, sqli_result: SqliDetectionResult) -> Optional[DetectionResult]:
        """將 SqliDetectionResult 轉換為 DetectionResult"""
        try:
            # 使用正確的 DetectionResult 結構
            detection_result = DetectionResult(
                is_vulnerable=sqli_result.is_vulnerable,
                vulnerability=sqli_result.vulnerability,
                evidence=sqli_result.evidence,
                impact=sqli_result.impact,
                recommendation=sqli_result.recommendation,
                target=sqli_result.target,
                detection_method=sqli_result.detection_method,
                payload_used=sqli_result.payload_used,
                confidence_score=sqli_result.confidence_score,
                db_fingerprint=sqli_result.db_fingerprint,
                response_time=sqli_result.response_time
            )
            
            return detection_result
            
        except Exception as e:
            logger.error(f"轉換檢測結果時發生錯誤: {e}", trace_id=self.trace_id)
            return None