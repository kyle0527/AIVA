"""Knowledge Decision Mixin - 知識驅動決策混入

從 enhanced_decision_agent.py 拆分而來。
負責 embedded_knowledge 和雙 CLI 架構的知識整合。

功能:
- 查詢內部能力庫 (InternalLoopConnector)
- 記錄執行反饋 (ExternalLoopConnector)
- 漏洞分析 (VulnerabilityDetector)
- CVE 識別 (CVEIdentifier)
- WAF 繞過技術 (WAFBypassEngine)
- Web 架構分析 (WebArchitectureAnalyzer)

Architecture:
    - 作為 Mixin 混入 EnhancedDecisionAgent
    - 依賴 internal_connector / external_connector / embedded_knowledge
    - 符合 aiva_common 數據合約規範

Date: 2026-02-09 (拆分自 enhanced_decision_agent.py)
"""

from datetime import datetime
import logging
from typing import Any, Protocol

from aiva_common.utils import get_logger

logger = get_logger(__name__)


# ==================== 類型協議定義 ====================

class _MixinHostProtocol(Protocol):
    """Mixin 宿主類別應實現的協議
    
    用於型別檢查，確保宿主類別提供必要的屬性。
    """
    logger: logging.Logger
    internal_connector: Any  # InternalLoopConnector
    external_connector: Any  # ExternalLoopConnector


# ==================== Mixin 類別 ====================

class KnowledgeDecisionMixin:
    """知識驅動決策混入

    提供 embedded_knowledge 和雙 CLI 架構的知識整合功能。
    設計為透過多重繼承混入 EnhancedDecisionAgent。

    需要宿主類別提供:
    - self.internal_connector  (InternalLoopConnector)
    - self.external_connector  (ExternalLoopConnector)
    - self.logger  (logging.Logger)
    """
    
    # 聲明 Mixin 期望的屬性（供型別檢查）
    logger: logging.Logger
    internal_connector: Any
    external_connector: Any

    # ========== 雙CLI架構的方法 ==========

    def query_internal_capabilities(
        self,
        query: str,
        scope_filter: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """AI 查詢內部能力庫

        使用 InternalLoopConnector 查詢可用的內部能力。

        Args:
            query: 查詢字串（如 "SQL注入測試"）
            scope_filter: 可選的範圍過濾（如 "internal", "external"）
            top_k: 返回的最大結果數

        Returns:
            list[dict]: 匹配的能力列表
        """
        if not getattr(self, "internal_connector", None):
            self.logger.warning("⚠️ InternalLoopConnector 未初始化")
            return []

        try:
            filters = {"scope": scope_filter} if scope_filter else None
            result = self.internal_connector.query_capabilities(
                query=query,
                filters=filters,
                top_k=top_k,
            )

            self.logger.info(f"🔍 查詢內部能力 '{query}': 找到 {result.total_found} 個匹配")

            return [
                {
                    "document_id": item.get("document_id", f"doc_{i}"),
                    "content": str(item.get("content", ""))[:200],
                    "relevance_score": item.get("relevance_score", 0.0),
                    "metadata": item.get("metadata", {}),
                }
                for i, item in enumerate(result.results)
            ]
        except Exception as e:
            self.logger.error(f"❌ 查詢內部能力失敗: {e}")
            return []

    async def record_execution_feedback(
        self,
        plan_id: str,
        trace_data: list[dict[str, Any]],
        objective: str = "Task execution",
    ) -> dict[str, Any]:
        """AI 記錄執行結果（用於學習）

        使用 ExternalLoopConnector 將執行結果反饋到學習系統。

        Args:
            plan_id: 執行計劃 ID
            trace_data: 執行軌跡列表（簡化格式）
            objective: 任務目標描述

        Returns:
            dict: 處理結果
        """
        if not getattr(self, "external_connector", None):
            self.logger.warning("⚠️ ExternalLoopConnector 未初始化")
            return {"success": False, "error": "ExternalLoopConnector not available"}

        try:
            from aiva_common.schemas.dual_loop import (
                ExecutionPlan,
                ExecutionStep,
                ExecutionTrace,
            )

            now = datetime.now()

            steps = [
                ExecutionStep(
                    step_id=t.get("step_id", f"step_{i}"),
                    action=t.get("action", "execute"),
                    capability_id=t.get("capability_id"),
                    parameters=t.get("parameters", {}),
                    expected_duration=t.get("expected_duration"),
                )
                for i, t in enumerate(trace_data)
            ]

            plan = ExecutionPlan(
                plan_id=plan_id,
                objective=objective,
                steps=steps,
                expected_duration=None,
                metadata={"source": "enhanced_decision_agent"},
            )

            traces = [
                ExecutionTrace(
                    trace_id=f"trace_{plan_id}_{i}",
                    step_id=t.get("step_id", f"step_{i}"),
                    capability_id=t.get("capability_id"),
                    status=t.get("status", "success"),
                    duration=t.get("duration", 0.0),
                    start_time=now,
                    end_time=now,
                    output=t.get("output"),
                    error=t.get("error"),
                    metrics=t.get("metrics"),
                )
                for i, t in enumerate(trace_data)
            ]

            result = await self.external_connector.process_execution_result(plan, traces)

            self.logger.info(
                f"📝 執行反饋已處理: {plan_id} - 發現 {result.deviations_found} 偏差"
            )

            return {
                "success": result.success,
                "deviations_found": result.deviations_found,
                "training_triggered": result.training_triggered,
                "weights_updated": result.weights_updated,
            }
        except Exception as e:
            self.logger.error(f"❌ 記錄執行反饋失敗: {e}")
            return {"success": False, "error": str(e)}

    # ========== embedded_knowledge 方法 ==========

    def analyze_target_vulnerabilities(
        self,
        target_url: str,
        response_body: str = "",
        response_time: float = 0.0,
    ) -> dict[str, Any]:
        """AI 分析目標漏洞

        使用 embedded_knowledge 的 VulnerabilityDetector 進行分析。

        Args:
            target_url: 目標 URL
            response_body: HTTP 響應體
            response_time: 響應時間

        Returns:
            dict: 漏洞分析結果
        """
        # 延遲導入，避免循環依賴
        from ..embedded_knowledge import (
            AttackContext,
            VulnerabilityDetector,
        )

        results: list[dict[str, Any]] = []

        try:
            context = AttackContext(
                target_url=target_url,
                injection_point="parameter",
            )

            sqli_result = VulnerabilityDetector.check_sqli(
                response_body=response_body,
                response_time=response_time,
                context=context,
            )

            if sqli_result.detected:
                results.append(
                    {
                        "type": sqli_result.vulnerability_type.value,
                        "confidence": sqli_result.confidence_score,
                        "evidence": sqli_result.evidence,
                        "recommendations": sqli_result.recommendations,
                    }
                )

            xss_result = VulnerabilityDetector.check_xss(
                response_body=response_body,
                payload_used="<script>alert(1)</script>",
                context=context,
            )

            if xss_result.detected:
                results.append(
                    {
                        "type": xss_result.vulnerability_type.value,
                        "confidence": xss_result.confidence_score,
                        "evidence": xss_result.evidence,
                        "recommendations": xss_result.recommendations,
                    }
                )

            self.logger.info(
                f"🔍 漏洞分析完成: {target_url} - 發現 {len(results)} 個潛在漏洞"
            )

            return {
                "target": target_url,
                "vulnerabilities": results,
                "risk_count": len(results),
                "scan_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"❌ 漏洞分析失敗: {e}")
            return {"error": str(e)}

    def identify_high_risk_cves(
        self,
        url: str = "",
        response_headers: dict[str, str] | None = None,
        response_body: str = "",
    ) -> list[dict[str, Any]]:
        """AI 識別高風險 CVE

        Args:
            url: 目標 URL
            response_headers: 響應 headers
            response_body: 響應體

        Returns:
            list[dict]: 匹配的 CVE 列表
        """
        from ..embedded_knowledge import CVEIdentifier

        try:
            cves = CVEIdentifier.identify(
                url=url,
                response_headers=response_headers,
                response_body=response_body,
            )

            self.logger.info(f"🚨 CVE 識別完成: 發現 {len(cves)} 個高危 CVE")

            return [
                {
                    "cve_id": cve.cve_id,
                    "confidence_score": cve.confidence_score,
                    "matched_indicators": cve.matched_indicators,
                }
                for cve in cves
            ]
        except Exception as e:
            self.logger.error(f"❌ CVE 識別失敗: {e}")
            return []

    def generate_waf_bypass_payloads(
        self,
        vuln_type: str,
        waf_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """AI 生成 WAF 繞過 payload

        Args:
            vuln_type: 漏洞類型（如 "sqli", "xss"）
            waf_type: 可選的 WAF 類型（如 "cloudflare", "aws_waf"）

        Returns:
            list[dict]: 繞過技術列表
        """
        from ..embedded_knowledge import WAFBypassEngine
        from ..embedded_knowledge.base import WAFVendor

        try:
            waf_vendor = WAFVendor.UNKNOWN
            if waf_type:
                try:
                    waf_vendor = WAFVendor(waf_type.lower())
                except ValueError:
                    waf_vendor = WAFVendor.UNKNOWN

            techniques = WAFBypassEngine.get_bypass_techniques(
                waf_vendor=waf_vendor,
                attack_type=vuln_type,
                min_success_rate=0.3,
            )

            self.logger.info(f"🛡️ WAF 繞過技術獲取: {vuln_type} - {len(techniques)} 個")

            return [
                {
                    "name": tech.name,
                    "category": tech.category.value,
                    "description": tech.description,
                    "payloads": tech.payloads[:5],
                    "success_rate": tech.success_rate,
                }
                for tech in techniques[:10]
            ]
        except Exception as e:
            self.logger.error(f"❌ WAF 繞過技術獲取失敗: {e}")
            return []

    def analyze_web_architecture(
        self,
        response_headers: dict[str, str],
        response_body: str = "",
        endpoints: list[str] | None = None,
    ) -> dict[str, Any]:
        """AI 分析 Web 架構

        Args:
            response_headers: HTTP 響應頭
            response_body: 響應體
            endpoints: 已知端點列表

        Returns:
            dict: 架構分析結果
        """
        from ..embedded_knowledge import WebArchitectureAnalyzer

        try:
            fingerprint = WebArchitectureAnalyzer.identify_architecture(
                response_headers=response_headers,
                response_body=response_body,
                endpoints=endpoints,
            )

            self.logger.info(f"🌐 Web 架構分析完成: {fingerprint.arch_type.value}")

            return {
                "architecture_type": fingerprint.arch_type.value,
                "confidence": fingerprint.confidence.name,
                "indicators": fingerprint.indicators,
                "endpoints": fingerprint.endpoints,
                "metadata": fingerprint.metadata,
            }
        except Exception as e:
            self.logger.error(f"❌ Web 架構分析失敗: {e}")
            return {"error": str(e)}
