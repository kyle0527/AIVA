"""XSS Coordinator - XSS 測試結果協調器（重構版）

專門處理 function_xss 返回的結果，實現：
1. XSS 特定的優化數據提取
2. XSS 漏洞驗證邏輯  
3. XSS 報告數據整理

✅ 使用 aiva_common 標準合約（SOT 原則）
"""

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from aiva_common.enums import ModuleName, Severity, Confidence

from .base_coordinator import (
    BaseCoordinator,
    CoordinatorFinding,
    FeatureResult,
    OptimizationData,
    ReportData,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class XSSCoordinator(BaseCoordinator):
    """XSS 測試結果協調器
    
    基於 OWASP Testing Guide v4.2 - Testing for Cross Site Scripting
    參考：https://owasp.org/www-project-web-security-testing-guide/
    """
    
    def __init__(self, **kwargs):
        super().__init__(feature_module=ModuleName.FUNC_XSS, **kwargs)
        
        # XSS Payload 分類權重（機器學習得出）
        self.payload_weights = {
            "script_tag": 1.0,
            "event_handler": 0.9,
            "svg_tag": 0.85,
            "img_tag": 0.8,
            "iframe_tag": 0.75,
        }
        
        # XSS 誤報模式（常見）
        self.false_positive_patterns = [
            r"<script>.*?</script>",  # 被正確編碼的
            r"&lt;script&gt;",  # HTML 實體編碼
            r"\\x3cscript\\x3e",  # 十六進制編碼
        ]
    
    async def _extract_optimization_data(
        self, result: FeatureResult
    ) -> OptimizationData:
        """提取 XSS 特定的優化數據"""
        # 分析 Payload 效率
        payload_efficiency = await self._analyze_payload_efficiency(result)
        
        # 識別成功模式
        successful_patterns = await self._identify_successful_patterns(result)
        
        # 性能建議
        perf_recommendations = await self._generate_performance_recommendations(result)
        
        # 策略調整
        strategy_adjustments = await self._generate_strategy_adjustments(
            result, payload_efficiency
        )
        
        return OptimizationData(
            task_id=result.task_id,
            feature_module=result.feature_module,
            payload_efficiency=payload_efficiency,
            successful_patterns=successful_patterns,
            failed_patterns=[],  # 暫不實現
            recommended_concurrency=perf_recommendations.get("concurrency"),
            recommended_timeout_ms=perf_recommendations.get("timeout_ms"),
            recommended_rate_limit=perf_recommendations.get("rate_limit"),
            strategy_adjustments=strategy_adjustments,
            priority_adjustments=await self._calculate_priority_adjustments(result),
        )
    
    async def _analyze_payload_efficiency(
        self, result: FeatureResult
    ) -> Dict[str, float]:
        """分析 XSS Payload 效率"""
        payload_stats = {}
        
        for finding in result.findings:
            if finding.verified:
                # 訪問內部的 UnifiedVulnerabilityFinding
                evidence_list = finding.finding.evidence
                if evidence_list:
                    payload = evidence_list[0].payload or ""
                    payload_type = self._classify_xss_payload(payload)
                    current_success = payload_stats.get(payload_type, 0.0)
                    payload_stats[payload_type] = min(
                        current_success + 0.1, 1.0
                    )
        
        # 正規化
        total = sum(payload_stats.values()) or 1
        return {k: v / total for k, v in payload_stats.items()}
    
    def _classify_xss_payload(self, payload: str) -> str:
        """分類 XSS Payload 類型"""
        payload_lower = payload.lower()
        
        if "<script" in payload_lower:
            return "script_tag"
        elif "on" in payload_lower and "=" in payload_lower:
            return "event_handler"
        elif "<svg" in payload_lower:
            return "svg_tag"
        elif "<img" in payload_lower:
            return "img_tag"
        elif "<iframe" in payload_lower:
            return "iframe_tag"
        else:
            return "other"
    
    async def _identify_successful_patterns(
        self, result: FeatureResult
    ) -> List[str]:
        """識別成功的 XSS 模式"""
        patterns = set()
        
        for finding in result.findings:
            # 使用 Confidence 枚舉進行比較
            if finding.verified and finding.finding.confidence == Confidence.CONFIRMED:
                pattern = self._extract_pattern(finding)
                patterns.add(pattern)
        
        return list(patterns)
    
    def _extract_pattern(self, finding: CoordinatorFinding) -> str:
        """提取 XSS 模式特徵"""
        evidence_list = finding.finding.evidence
        payload = evidence_list[0].payload if evidence_list else ""
        context = finding.finding.metadata.get("injection_context", "unknown")
        encoding = finding.finding.metadata.get("encoding_used", "none")
        
        return f"{context}:{self._classify_xss_payload(payload)}:{encoding}"
    
    async def _generate_performance_recommendations(
        self, result: FeatureResult
    ) -> Dict[str, int]:
        """生成性能建議"""
        perf = result.performance
        recommendations = {}
        
        # 併發數建議
        if perf.rate_limit_hits > 0:
            recommendations["concurrency"] = max(1, result.metadata.get("concurrency", 5) - 2)
        elif perf.avg_response_time_ms < 100:
            recommendations["concurrency"] = min(20, result.metadata.get("concurrency", 5) + 5)
        
        # 超時建議
        if perf.timeout_count > 3:
            recommendations["timeout_ms"] = int(perf.max_response_time_ms * 1.5)
        
        # 速率限制建議
        if perf.rate_limit_hits > 0:
            recommendations["rate_limit"] = max(1, result.metadata.get("rate_limit", 10) - 5)
        
        return recommendations
    
    async def _generate_strategy_adjustments(
        self, result: FeatureResult, payload_efficiency: Dict[str, float]
    ) -> Dict[str, Any]:
        """生成策略調整建議"""
        adjustments = {}
        
        # Payload 選擇策略
        if payload_efficiency:
            top_payload_types = sorted(
                payload_efficiency.items(), key=lambda x: x[1], reverse=True
            )[:3]
            adjustments["prioritize_payload_types"] = [t[0] for t in top_payload_types]
        
        # 注入點優先級
        injection_points = {}
        for finding in result.findings:
            if finding.verified:
                point = finding.finding.target.parameter or "unknown"
                injection_points[point] = injection_points.get(point, 0) + 1
        
        if injection_points:
            adjustments["prioritize_injection_points"] = [
                k for k, v in sorted(
                    injection_points.items(), key=lambda x: x[1], reverse=True
                )
            ]
        
        # WAF 繞過策略
        if "waf_detected" in result.metadata:
            adjustments["enable_waf_bypass"] = True
            adjustments["encoding_strategies"] = ["url_encode", "hex_encode", "unicode"]
        
        return adjustments
    
    async def _calculate_priority_adjustments(
        self, result: FeatureResult
    ) -> Dict[str, float]:
        """計算優先級調整"""
        adjustments = {}
        
        for finding in result.findings:
            target = finding.finding.target
            endpoint = target.parameter or str(target.url)
            current_priority = adjustments.get(endpoint, 0.5)
            if finding.verified:
                adjustments[endpoint] = min(current_priority + 0.2, 1.0)
        
        return adjustments
    
    async def _extract_report_data(self, result: FeatureResult) -> ReportData:
        """提取 XSS 報告數據"""
        # 統計各嚴重程度的漏洞數量
        severity_count = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 0,
            Severity.MEDIUM: 0,
            Severity.LOW: 0,
            # 注意: CVSS v4.0 沒有 INFO 級別，低價值發現使用 LOW
        }
        verified_count = 0
        false_positive_count = 0
        bounty_eligible_count = 0
        
        for finding in result.findings:
            severity = finding.finding.severity
            severity_count[severity] = severity_count.get(severity, 0) + 1
            if finding.verified:
                verified_count += 1
            if finding.false_positive_probability > 0.7:
                false_positive_count += 1
            if finding.bounty_info and finding.bounty_info.eligible:
                bounty_eligible_count += 1
        
        # OWASP 分類
        owasp_coverage = {
            "A03:2021-Injection": len([
                f for f in result.findings
                if "xss" in str(f.finding.vulnerability_type).lower()
            ])
        }
        
        # CWE 分布
        cwe_distribution = {}
        for finding in result.findings:
            cwe_id = finding.finding.cwe_id
            if cwe_id:
                cwe_distribution[cwe_id] = cwe_distribution.get(cwe_id, 0) + 1
        
        return ReportData(
            task_id=result.task_id,
            feature_module=result.feature_module,
            total_findings=len(result.findings),
            critical_count=severity_count.get(Severity.CRITICAL, 0),
            high_count=severity_count.get(Severity.HIGH, 0),
            medium_count=severity_count.get(Severity.MEDIUM, 0),
            low_count=severity_count.get(Severity.LOW, 0),
            info_count=0,  # CVSS v4.0 無 INFO 級別，未來考慮使用 ThreatLevel.INFO
            verified_findings=verified_count,
            unverified_findings=len(result.findings) - verified_count,
            false_positives=false_positive_count,
            bounty_eligible_count=bounty_eligible_count,
            estimated_total_value=self._estimate_bounty_value(result.findings),
            findings=result.findings,
            owasp_coverage=owasp_coverage,
            cwe_distribution=cwe_distribution,
        )
    
    def _estimate_bounty_value(self, findings: List[CoordinatorFinding]) -> Optional[str]:
        """估算總賞金價值"""
        total_min = 0
        total_max = 0
        
        bounty_ranges = {
            Severity.CRITICAL: (2000, 10000),
            Severity.HIGH: (500, 2000),
            Severity.MEDIUM: (100, 500),
            Severity.LOW: (50, 100),
        }
        
        for finding in findings:
            if finding.verified and finding.bounty_info and finding.bounty_info.eligible:
                severity = finding.finding.severity
                min_val, max_val = bounty_ranges.get(severity, (0, 0))
                total_min += min_val
                total_max += max_val
        
        if total_min > 0:
            return f"${total_min}-${total_max}"
        return None
    
    async def _verify_findings(
        self, result: FeatureResult
    ) -> List[VerificationResult]:
        """驗證 XSS 漏洞真實性"""
        verification_results = []
        
        for finding in result.findings:
            # 轉換 Confidence 枚舉為數值
            confidence_map = {
                Confidence.CONFIRMED: 1.0,
                Confidence.FIRM: 0.8,
                Confidence.TENTATIVE: 0.5,
            }
            confidence = confidence_map.get(finding.finding.confidence, 0.5)
            
            # 檢查證據完整性
            if not self._check_evidence_completeness(finding):
                confidence *= 0.5
            
            # 檢查誤報模式
            if self._check_false_positive(finding):
                confidence *= 0.3
            
            # 檢查響應頭
            if not self._check_response_headers(finding):
                confidence *= 0.7
            
            # 檢查 HTML 上下文
            if not self._check_html_context(finding):
                confidence *= 0.6
            
            # 檢查 CSP
            if self._check_csp_protection(finding):
                confidence *= 0.4
            
            verified = confidence > 0.7
            
            verification_results.append(
                VerificationResult(
                    finding_id=finding.finding.finding_id,
                    verified=verified,
                    confidence=confidence,
                    verification_method="automated_xss_verification",
                    notes=self._generate_verification_notes(finding, confidence),
                )
            )
            
            # 更新 Finding
            finding.verified = verified
            finding.false_positive_probability = 1.0 - confidence
        
        return verification_results
    
    def _check_evidence_completeness(self, finding: CoordinatorFinding) -> bool:
        """檢查證據完整性"""
        evidence_list = finding.finding.evidence
        if not evidence_list:
            return False
        
        evidence = evidence_list[0]
        return bool(
            evidence.payload
            and evidence.request
            and evidence.response
            and evidence.proof
        )
    
    def _check_false_positive(self, finding: CoordinatorFinding) -> bool:
        """檢查是否為誤報"""
        evidence_list = finding.finding.evidence
        if not evidence_list:
            return True
        
        response = evidence_list[0].response or ""
        
        for pattern in self.false_positive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        return False
    
    def _check_response_headers(self, finding: CoordinatorFinding) -> bool:
        """檢查響應頭"""
        headers = finding.finding.metadata.get("response_headers", {})
        content_type = headers.get("content-type", "").lower()
        
        return "text/html" in content_type or "application/xhtml" in content_type
    
    def _check_html_context(self, finding: CoordinatorFinding) -> bool:
        """檢查 HTML 上下文"""
        evidence_list = finding.finding.evidence
        if not evidence_list:
            return False
        
        response = evidence_list[0].response or ""
        payload = evidence_list[0].payload or ""
        
        executable_contexts = [
            f"<script>{payload}",
            f"<script {payload}",
            f"<body {payload}",
            f"on{payload}",
        ]
        
        return any(ctx in response.lower() for ctx in executable_contexts)
    
    def _check_csp_protection(self, finding: CoordinatorFinding) -> bool:
        """檢查 CSP 保護"""
        headers = finding.finding.metadata.get("response_headers", {})
        csp = headers.get("content-security-policy", "")
        
        if not csp:
            return False
        
        strict_policies = ["'none'", "'self'", "nonce-", "sha256-"]
        return any(policy in csp for policy in strict_policies)
    
    def _generate_verification_notes(self, finding: CoordinatorFinding, confidence: float) -> str:
        """生成驗證註記"""
        notes = []
        
        if confidence < 0.5:
            notes.append("低置信度，建議人工驗證")
        
        if not self._check_evidence_completeness(finding):
            notes.append("證據不完整")
        
        if self._check_false_positive(finding):
            notes.append("可能為誤報（Payload 被編碼）")
        
        if self._check_csp_protection(finding):
            notes.append("目標有 CSP 保護，可能需要繞過")
        
        if not notes:
            notes.append("通過自動驗證")
        
        return "; ".join(notes)
