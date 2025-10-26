# services/features/client_side_auth_bypass/client_side_auth_bypass_worker.py
"""
客戶端授權繞過檢測 Worker

Compliance Note (遵循 aiva_common 設計原則):
- 移除 fallback 導入機制，確保使用 aiva_common 標準
- Severity, Confidence 從 aiva_common.enums 導入
- 修正日期: 2025-10-25
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio
import re
from urllib.parse import urljoin

# 直接導入，不使用 fallback（確保 aiva_common 可用）
from services.aiva_common.schemas.generated.tasks import FunctionTaskPayload
from services.aiva_common.schemas.telemetry import FunctionExecutionResult
from services.aiva_common.schemas.generated.findings import FindingPayload
from services.aiva_common.enums.common import Severity, Confidence
from services.features.base.feature_base import FeatureBase
from .js_analysis_engine import JavaScriptAnalysisEngine


logger = logging.getLogger(__name__)

class ClientSideAuthBypassWorker(FeatureBase):
    """
    執行客戶端授權繞過檢測的 Worker。
    
    Constants:
        DEFAULT_TIMEOUT: 默認 HTTP 請求超時時間（秒）
        MIN_SCRIPT_LENGTH: 最小腳本長度，低於此長度的腳本將被忽略
        SCRIPT_SRC_PATTERN: 提取外部腳本 URL 的正則表達式
        SCRIPT_INLINE_PATTERN: 提取內聯腳本的正則表達式
    """
    
    # 類級別常量定義
    DEFAULT_TIMEOUT = 30
    MIN_SCRIPT_LENGTH = 10
    SCRIPT_SRC_PATTERN = r'<script[^>]*src=["\'](.*?)["\'][^>]*>'
    SCRIPT_INLINE_PATTERN = r'<script[^>]*>(.*?)</script>'

    def __init__(self, mq_channel=None, http_client=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(mq_channel, http_client, config)
        
        self.js_analyzer = JavaScriptAnalysisEngine(http_client=self.http_client)
        self.timeout = self.config.get('timeout', self.DEFAULT_TIMEOUT) if self.config else self.DEFAULT_TIMEOUT
        logger.info("ClientSideAuthBypassWorker initialized.")

    async def execute_task(self, payload: FunctionTaskPayload) -> FunctionExecutionResult:
        """
        執行單個客戶端授權繞過檢測任務。

        Args:
            payload: 包含目標、上下文和策略的任務 Payload。

        Returns:
            包含檢測結果的 FunctionExecutionResult。
        """
        target_url = payload.target.url if payload.target else None
        if not target_url:
            return FunctionExecutionResult(
                findings=[],
                telemetry={"error": "Target URL is missing."},
                errors=[{"message": "Target URL is missing."}]
            )

        findings: List[FindingPayload] = []
        logger.info(f"Starting Client-Side Auth Bypass check for task {payload.task_id} on {target_url}")

        try:
            # 步驟 1: 獲取頁面內容和 JavaScript 腳本
            logger.debug("Fetching page content and scripts...")
            html_content, scripts_on_page = await self._fetch_page_and_scripts(target_url)
            
            if not scripts_on_page:
                logger.warning(f"No JavaScript found on {target_url}")
                return FunctionExecutionResult(
                    findings=[],
                    telemetry={"scripts_analyzed": 0, "potential_issues": 0}
                )

            # 步驟 2: 使用 JavaScript 分析引擎進行靜態分析
            logger.debug("Analyzing JavaScript for auth bypass patterns...")
            js_issues = await self.js_analyzer.analyze(target_url, scripts_on_page)
            
            # 步驟 3: 分析DOM操作相關問題
            logger.debug("Analyzing DOM manipulation patterns...")
            dom_issues = await self.js_analyzer.analyze_dom_manipulation(html_content)
            
            # 合併所有問題
            all_issues = js_issues + dom_issues

            # 步驟 4: 處理分析結果並生成 Findings
            for issue in all_issues:
                logger.warning(f"Potential client-side auth issue found: {issue['description']}")
                
                # 映射嚴重性
                severity = self._map_severity(issue.get('severity', 'low'))
                confidence = self._determine_confidence(issue)
                
                finding = FindingPayload(
                    finding_id=f"{payload.task_id}_csab_{len(findings)+1}",
                    scan_id=payload.context.session_id if payload.context else "unknown",
                    target_url=target_url,
                    vulnerability_type="Client-Side Authorization Bypass",
                    title=f"客戶端授權繞過: {issue['description']}",
                    description=self._create_detailed_description(issue),
                    severity=severity,
                    confidence=confidence,
                    evidence={
                        "issue_type": issue['type'],
                        "script_identifier": issue.get('script_identifier', 'unknown'),
                        "line_number": issue.get('line_number', 0),
                        "matched_text": issue.get('matched_text', ''),
                        "code_snippet": issue.get('snippet', ''),
                        "context": issue.get('context', '')
                    },
                    module_name="FUNC_CLIENT_AUTH_BYPASS",
                    recommendations=self.js_analyzer.get_recommendations([issue])
                )
                findings.append(finding)

            # 步驟 5: 生成總結性Finding（如果發現多個問題）
            if len(findings) > 3:
                summary_finding = self._create_summary_finding(payload, target_url, findings)
                findings.append(summary_finding)

            logger.info(f"Client-side auth bypass check completed for {target_url}. Found {len(findings)} findings.")

        except Exception as e:
            logger.error(f"Error during client-side auth bypass check for {target_url}: {e}", exc_info=True)
            return FunctionExecutionResult(
                findings=[],
                telemetry={"error": str(e)},
                errors=[{"message": f"An unexpected error occurred: {e}"}]
            )

        return FunctionExecutionResult(
            findings=[finding.dict() for finding in findings],
            telemetry={
                "scripts_analyzed": len(scripts_on_page),
                "potential_issues": len(all_issues),
                "findings_generated": len(findings)
            }
        )

    async def _fetch_page_and_scripts(self, url: str) -> tuple[str, List[str]]:
        """
        獲取目標 URL 頁面內容和所有 JavaScript 腳本內容。
        
        Returns:
            tuple: (html_content, scripts_list)
        """
        scripts = []
        html_content = ""
        
        try:
            # 獲取主頁面
            response = await self.http_client.get(
                url, 
                follow_redirects=True, 
                timeout=self.timeout
            )
            response.raise_for_status()
            html_content = response.text

            # 提取外部腳本 URL
            src_matches = re.findall(self.SCRIPT_SRC_PATTERN, html_content, re.IGNORECASE)
            
            # 獲取外部腳本內容
            for src in src_matches:
                if src.startswith('//'):
                    src = 'https:' + src
                elif not src.startswith(('http://', 'https://')):
                    src = urljoin(url, src)
                
                try:
                    script_resp = await self.http_client.get(
                        src, 
                        timeout=self.timeout
                    )
                    if script_resp.status_code == 200:
                        scripts.append(script_resp.text)
                        logger.debug(f"Fetched external script: {src}")
                except Exception as e:
                    logger.warning(f"Failed to fetch external script {src}: {e}")

            # 提取內聯腳本
            inline_matches = re.findall(
                self.SCRIPT_INLINE_PATTERN, 
                html_content, 
                re.IGNORECASE | re.DOTALL
            )
            
            for inline_script in inline_matches:
                cleaned_script = inline_script.strip()
                if cleaned_script and len(cleaned_script) > self.MIN_SCRIPT_LENGTH:
                    scripts.append(cleaned_script)
                    logger.debug("Found inline script.")

        except Exception as e:
            logger.error(f"Failed to fetch or parse scripts from {url}: {e}")

        logger.info(f"Fetched {len(scripts)} script blocks from {url}")
        return html_content, scripts

    def _map_severity(self, issue_severity: str) -> str:
        """映射問題嚴重性到標準嚴重性"""
        mapping = {
            'high': Severity.HIGH,
            'medium': Severity.MEDIUM,
            'low': Severity.LOW
        }
        return mapping.get(issue_severity.lower(), Severity.LOW)

    def _determine_confidence(self, issue: Dict[str, Any]) -> str:
        """根據問題類型確定置信度"""
        high_confidence_types = ['hardcoded_admin', 'client_side_validation']
        medium_confidence_types = ['localStorage_auth', 'sessionStorage_auth', 'jwt_client_decode']
        
        issue_type = issue.get('type', '')
        
        if issue_type in high_confidence_types:
            return Confidence.HIGH
        elif issue_type in medium_confidence_types:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW

    def _create_detailed_description(self, issue: Dict[str, Any]) -> str:
        """創建詳細的問題描述"""
        base_desc = issue.get('description', '')
        
        details = f"{base_desc}\n\n"
        details += f"檢測類型: {issue.get('type', 'unknown')}\n"
        
        if issue.get('line_number'):
            details += f"發現位置: 第 {issue['line_number']} 行\n"
            
        if issue.get('matched_text'):
            details += f"匹配代碼: {issue['matched_text']}\n"
            
        # 添加風險說明
        risk_descriptions = {
            'localStorage_auth': '攻擊者可以通過瀏覽器開發者工具或XSS攻擊修改本地存儲的授權信息',
            'hardcoded_admin': '硬編碼的角色檢查可以被攻擊者通過修改JavaScript代碼繞過',
            'client_side_validation': '僅客戶端的權限驗證可以被攻擊者完全繞過',
            'hidden_elements': '隱藏的管理功能可能被攻擊者通過修改CSS或JavaScript重新顯示'
        }
        
        risk_desc = risk_descriptions.get(issue.get('type', ''))
        if risk_desc:
            details += f"\n風險說明: {risk_desc}"
            
        return details

    def _create_summary_finding(self, payload: FunctionTaskPayload, target_url: str, findings: List[FindingPayload]) -> FindingPayload:
        """創建總結性發現"""
        high_severity_count = sum(1 for f in findings if f.severity == Severity.HIGH)
        medium_severity_count = sum(1 for f in findings if f.severity == Severity.MEDIUM)
        
        summary_desc = f"在目標應用程序中發現多個客戶端授權繞過風險點。"
        summary_desc += f"\n- 高風險問題: {high_severity_count} 個"
        summary_desc += f"\n- 中風險問題: {medium_severity_count} 個"
        summary_desc += f"\n\n建議進行全面的授權架構審查，確保所有關鍵授權檢查都在服務端實現。"
        
        return FindingPayload(
            finding_id=f"{payload.task_id}_csab_summary",
            scan_id=payload.context.session_id if payload.context else "unknown",
            target_url=target_url,
            vulnerability_type="Client-Side Authorization Bypass (Summary)",
            title="客戶端授權繞過風險總結",
            description=summary_desc,
            severity=Severity.HIGH if high_severity_count > 0 else Severity.MEDIUM,
            confidence=Confidence.HIGH,
            evidence={
                "total_issues": len(findings),
                "high_severity_count": high_severity_count,
                "medium_severity_count": medium_severity_count,
                "analysis_summary": "Multiple client-side authorization bypass risks detected"
            },
            module_name="FUNC_CLIENT_AUTH_BYPASS"
        )