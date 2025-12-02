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
from services.aiva_common.enums.common import Severity, Confidence
from services.aiva_common.enums.security import VulnerabilityType
from services.aiva_common.schemas import (
    FindingEvidence, FindingImpact, FindingRecommendation, 
    FindingTarget, Vulnerability, FunctionTaskPayload
)

from ..detection_models import DetectionResult
# 修復: schemas 模組不存在，這些類型實際在其他文件中定義
# SqliDetectionResult 使用 DetectionResult，SqliTelemetry 從 telemetry 導入
try:
    from ..telemetry import SqliExecutionTelemetry as SqliTelemetry
except ImportError:
    # 如果 telemetry 模組不存在，使用基礎遙測類
    SqliTelemetry = None  # type: ignore

from ..config import SqliConfig
from ..hackingtool_config import (
    HackingToolSQLIntegrator, SQLToolType, 
    HACKINGTOOL_SQL_CONFIGS, sql_integrator
)

logger = get_logger(__name__)


class HackingToolDetectionEngine:
    """HackingTool SQL 注入檢測引擎
    
    ✅ 修復: 添加工具可用性檢查,防止偽陰性
    """
    
    def __init__(self, config: SqliConfig):
        self.config = config
        self.integrator = sql_integrator
        self.trace_id = new_id("hackingtool_sqli")
        self._initialized = False
        
        logger.info(f"HackingTool SQL 檢測引擎已創建 [trace_id={self.trace_id}]")
    
    def _validate_tools_availability(self, enabled_tools: List[str]) -> List[str]:
        """驗證工具可用性——降低initialize函數複雜度"""
        available_tools = []
        
        for tool_name in enabled_tools:
            if self._check_tool_availability(tool_name):
                available_tools.append(tool_name)
                
        return available_tools
    
    def _check_tool_availability(self, tool_name: str) -> bool:
        """檢查單一工具可用性"""
        config = HACKINGTOOL_SQL_CONFIGS.get(tool_name)
        if not config:
            logger.warning(f"⚠️  工具 {tool_name} 配置不存在,已跳過")
            return False
        
        tool_binary = tool_name.lower()
        if not self._is_tool_installed(tool_binary):
            logger.warning(f"⚠️  {tool_name} 未安裝 (未在 PATH 中)")
            return False
            
        if self._check_tool_version(tool_binary, tool_name):
            logger.debug(f"✅ {tool_name} 可用")
            return True
            
        return False
    
    def _is_tool_installed(self, tool_binary: str) -> bool:
        """檢查工具是否安裝"""
        return subprocess.run(
            ["which", tool_binary],
            capture_output=True,
            text=True
        ).returncode == 0
    
    def _check_tool_version(self, tool_binary: str, tool_name: str) -> bool:
        """檢查工具版本"""
        try:
            version_result = subprocess.run(
                [tool_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if version_result.returncode == 0:
                return True
            else:
                logger.warning(f"⚠️  {tool_name} 版本檢查失敗")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️  {tool_name} 版本檢查超時")
        except Exception as e:
            logger.warning(f"⚠️  {tool_name} 版本檢查異常: {e}")
        
        return False
    
    def initialize(self) -> bool:
        """初始化檢測引擎,驗證工具可用性
        
        ✅ 新增: 檢查 sqlmap 等工具是否安裝
        
        Raises:
            RuntimeError: 當必要工具未安裝時
        """
        enabled_tools = self.integrator.get_enabled_tools()
        
        if not enabled_tools:
            raise RuntimeError(
                "❌ SQL 注入檢測引擎初始化失敗: 沒有啟用的工具\n"
                "\n"
                "請在配置中啟用至少一個 SQL 工具:\n"
                "  - sqlmap\n"
                "  - ghauri\n"
                "  - NoSQLMap\n"
                "\n"
                "配置文件: services/features/function_sqli/hackingtool_config.py"
            )
        
        # 驗證工具可執行性——使用輔助函數降低複雜度
        available_tools = self._validate_tools_availability(enabled_tools)
        
        if not available_tools:
            raise RuntimeError(
                f"❌ SQL 注入檢測引擎初始化失敗: 沒有可用的工具\n"
                f"\n"
                f"已啟用但不可用的工具: {', '.join(enabled_tools)}\n"
                f"\n"
                f"安裝方法:\n"
                f"  sqlmap:   apt-get install sqlmap  或  pip install sqlmap\n"
                f"  ghauri:   pip install ghauri\n"
                f"  NoSQLMap: git clone https://github.com/codingo/NoSQLMap\n"
            )
        
        self._initialized = True
        logger.info(
            f"✅ SQL 檢測引擎初始化成功 "
            f"({len(available_tools)} 個工具可用: {', '.join(available_tools)}) "
            f"[trace_id={self.trace_id}]"
        )
        return True
    
    async def detect(self, task: FunctionTaskPayload) -> List[DetectionResult]:
        """執行 HackingTool SQL 注入檢測"""
        results = []
        
        # 從 task 中提取目標 URL
        target = task.target.url
        
        logger.info("開始 HackingTool SQL 檢測", 
                   extra={"target": target})
        
        # 獲取啟用的工具
        enabled_tools = self.integrator.get_enabled_tools()
        
        if not enabled_tools:
            logger.warning(f"沒有可用的 HackingTool SQL 工具 [trace_id={self.trace_id}]")
            return results
        
        # 並行執行工具檢測
        detection_tasks = []
        for tool_name in enabled_tools:
            detection_task = asyncio.create_task(
                self._run_tool_detection(tool_name, target)
            )
            detection_tasks.append(detection_task)
        
        # 等待所有檢測完成
        tool_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # 處理結果，轉換為 DetectionResult
        for i, result in enumerate(tool_results):
            tool_name = enabled_tools[i]
            
            if isinstance(result, Exception):
                logger.error(f"工具 {tool_name} 檢測失敗: {result}")
                continue
            
            if result and not isinstance(result, Exception):
                # 確保 result 是可疊代的
                if hasattr(result, '__iter__'):
                    # 轉換 SqliDetectionResult 為 DetectionResult
                    for sqli_result in result:
                        detection_result = self._convert_to_detection_result(sqli_result)
                        if detection_result:
                            results.append(detection_result)
                else:
                    # 單個結果的情況
                    detection_result = self._convert_to_detection_result(result)
                    if detection_result:
                        results.append(detection_result)
        
        logger.info(f"HackingTool 檢測完成，發現 {len(results)} 個結果")
        
        return results
    
    async def _run_tool_detection(self, tool_name: str, target: str) -> List[DetectionResult]:
        """執行單個工具的檢測
        
        ✅ 修復: 工具執行失敗時拋出異常,不再回傳空列表
        """
        config = HACKINGTOOL_SQL_CONFIGS.get(tool_name)
        if not config:
            raise RuntimeError(
                f"❌ SQL 工具配置不存在: {tool_name}\n"
                f"可用工具: {list(HACKINGTOOL_SQL_CONFIGS.keys())}"
            )
        
        logger.debug(f"執行工具檢測: {tool_name}", 
                    extra={"target": target, "trace_id": self.trace_id})
        
        # ✅ 修復: 不再捕捉異常,讓異常向上傳播
        # 執行工具 (失敗時會拋出異常)
        execution_result = await self._execute_tool(tool_name, target)
        
        # 檢查執行結果
        if not execution_result.get("success", False):
            # 這個分支理論上不會執行,因為 _execute_tool 失敗會拋異常
            # 但保留作為雙重保險
            raise RuntimeError(
                f"❌ {tool_name} 執行失敗但未拋出異常 (不應該發生)\n"
                f"錯誤: {execution_result.get('error', 'Unknown error')}"
            )
        
        # 解析結果
        detection_results = self._parse_tool_output(
            tool_name, execution_result, target
        )
        
        return detection_results
    
    async def _execute_tool(self, tool_name: str, target: str) -> Dict[str, Any]:
        """異步執行工具命令
        
        ✅ 修復: 執行失敗時拋出異常,移除偽陰性邏輯
        
        Raises:
            RuntimeError: 當工具執行失敗或超時時
        """
        config = HACKINGTOOL_SQL_CONFIGS[tool_name]
        
        if not config.run_commands:
            raise RuntimeError(
                f"❌ {tool_name} 工具配置錯誤: 沒有定義執行命令\n"
                f"請檢查 HACKINGTOOL_SQL_CONFIGS 配置"
            )
        
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
            
            # 等待執行完成,設置超時
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                # ✅ 修復: 拋出異常而非回傳 {"success": False}
                raise RuntimeError(
                    f"❌ {tool_name} 執行超時\n"
                    f"目標: {target}\n"
                    f"超時時間: {config.timeout_seconds}s\n"
                    f"命令: {cmd}\n"
                    f"建議: 增加 timeout 或檢查目標可訪問性"
                ) from None
            
            # ✅ 修復: 執行失敗時拋出異常
            if process.returncode != 0:
                stderr_str = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(
                    f"❌ {tool_name} 執行失敗\n"
                    f"目標: {target}\n"
                    f"返回碼: {process.returncode}\n"
                    f"錯誤輸出: {stderr_str[:500]}\n"
                    f"命令: {cmd}\n"
                    f"請檢查:\n"
                    f"  1. {tool_name} 是否正確安裝\n"
                    f"  2. 目標 URL 是否可訪問\n"
                    f"  3. 命令參數是否正確"
                )
            
            return {
                "success": True,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "returncode": process.returncode,
                "command": cmd
            }
            
        except asyncio.TimeoutError:
            # 已在上面處理
            raise
        except RuntimeError:
            # 已在上面處理的運行時錯誤,直接傳播
            raise
        except Exception as e:
            # ✅ 修復: 其他異常也拋出,不回傳字典
            raise RuntimeError(
                f"❌ {tool_name} 執行過程異常\n"
                f"目標: {target}\n"
                f"異常類型: {type(e).__name__}\n"
                f"異常信息: {str(e)}"
            ) from e
    
    def _parse_tool_output(self, tool_name: str, execution_result: Dict[str, Any], target: str) -> List[DetectionResult]:
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
    ) -> Optional[DetectionResult]:
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
                name=VulnerabilityType.SQLI,
                cve=None,  # HackingTool 通常不提供 CVE
                severity=severity,
                confidence=Confidence.FIRM if confidence_score >= 0.8 else Confidence.POSSIBLE,
                description=f"SQL injection vulnerability found using {tool_name}"
            )
            
            # 創建證據
            evidence = FindingEvidence(
                payload=payload_used,
                proof=full_output[:1000],  # 限制輸出長度
                request=f"Tool: {tool_name}",
                response=f"Detection result at {datetime.now().isoformat()}"
            )
            
            # 創建影響評估
            impact = FindingImpact(
                description="SQL injection can lead to unauthorized data access, modification, or deletion",
                business_impact="HIGH - Potential data breach and unauthorized access",
                technical_impact="HIGH - Database compromise and data manipulation"
            )
            
            # 創建修復建議
            recommendation = FindingRecommendation(
                fix="Implement proper input validation and parameterized queries",
                priority="HIGH",
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
            result = DetectionResult(
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
            logger.error(f"創建檢測結果時發生錯誤: {e} [trace_id={self.trace_id}]")
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
                logger.info(f"嘗試安裝工具: {tool_name} [trace_id={self.trace_id}]")
                
                success = self.integrator.install_tool(tool_name)
                installation_results[tool_name] = success
                
                if success:
                    logger.info(f"工具 {tool_name} 安裝成功 [trace_id={self.trace_id}]")
                else:
                    logger.error(f"工具 {tool_name} 安裝失敗 [trace_id={self.trace_id}]")
        
        return installation_results
    
    def _convert_to_detection_result(self, sqli_result: DetectionResult) -> Optional[DetectionResult]:
        """將 DetectionResult 轉換為 DetectionResult（已經是正確類型）"""
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
            logger.error(f"轉換檢測結果時發生錯誤: {e} [trace_id={self.trace_id}]")
            return None