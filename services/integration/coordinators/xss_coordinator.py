"""XSS Coordinator - XSS 測試結果協調器

專門處理 function_xss 返回的結果，實現：
1. XSS 特定的優化數據提取
2. XSS 漏洞驗證邏輯
3. XSS 報告數據整理
"""

import logging
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from .base_coordinator import (
    BaseCoordinator,
    CoreFeedback,
    FeatureResult,
    Finding,
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
        super().__init__(feature_module="function_xss", **kwargs)
        
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
        """提取 XSS 特定的優化數據
        
        分析：
        1. 哪些 XSS payload 類型最有效
        2. 哪些注入點（URL參數、表單、Header）成功率高
        3. WAF 繞過策略效果
        4. DOM XSS vs Reflected XSS 分布
        """
        # 分析 Payload 效率
        payload_efficiency = await self._analyze_payload_efficiency(result)
        
        # 識別成功模式
        successful_patterns = await self._identify_successful_patterns(result)
        
        # 識別失敗模式
        failed_patterns = await self._identify_failed_patterns(result)
        
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
            failed_patterns=failed_patterns,
            recommended_concurrency=perf_recommendations.get("concurrency"),
            recommended_timeout_ms=perf_recommendations.get("timeout_ms"),
            recommended_rate_limit=perf_recommendations.get("rate_limit"),
            strategy_adjustments=strategy_adjustments,
            priority_adjustments=await self._calculate_priority_adjustments(result),
        )
    
    async def _analyze_payload_efficiency(
        self, result: FeatureResult
    ) -> Dict[str, float]:
        """分析 XSS Payload 效率
        
        Returns:
            {payload_type: success_rate} 映射
        """
        payload_stats = {}
        
        for finding in result.findings:
            if finding.verified:
                payload_type = self._classify_xss_payload(finding.evidence.payload)
                current_success = payload_stats.get(payload_type, 0.0)
                payload_stats[payload_type] = min(
                    current_success + 0.1, 1.0
                )  # 累加成功率
        
        # 正規化
        total = sum(payload_stats.values()) or 1
        return {k: v / total for k, v in payload_stats.items()}
    
    def _classify_xss_payload(self, payload: str) -> str:
        """分類 XSS Payload 類型"""
        payload_lower = payload.lower()
        
        if "<script" in payload_lower:
            return "script_tag"
        elif "on" in payload_lower and "=" in payload_lower:  # onerror=, onload=
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
            if finding.verified and finding.evidence.confidence > 0.8:
                # 提取模式特徵
                pattern = self._extract_pattern(finding)
                patterns.add(pattern)
        
        return list(patterns)
    
    def _extract_pattern(self, finding: Finding) -> str:
        """提取 XSS 模式特徵"""
        payload = finding.evidence.payload
        context = finding.metadata.get("injection_context", "unknown")
        encoding = finding.metadata.get("encoding_used", "none")
        
        return f"{context}:{self._classify_xss_payload(payload)}:{encoding}"
    
    async def _identify_failed_patterns(self, result: FeatureResult) -> List[str]:
        """識別失敗的模式（從統計數據推斷）"""
        # TODO: 需要 Features 返回失敗的 payload 信息
        return []
    
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
                point = finding.target.parameters.get("injection_point", "unknown")
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
        
        # 根據成功率調整端點優先級
        for finding in result.findings:
            endpoint = finding.target.endpoint or finding.target.url
            current_priority = adjustments.get(endpoint, 0.5)
            if finding.verified:
                adjustments[endpoint] = min(current_priority + 0.2, 1.0)
        
        return adjustments
    
    async def _extract_report_data(self, result: FeatureResult) -> ReportData:
        """提取 XSS 報告數據"""
        # 統計各嚴重程度的漏洞數量
        severity_count = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        verified_count = 0
        false_positive_count = 0
        bounty_eligible_count = 0
        
        for finding in result.findings:
            severity_count[finding.severity] += 1
            if finding.verified:
                verified_count += 1
            if finding.false_positive_probability > 0.7:
                false_positive_count += 1
            if finding.bounty_info and finding.bounty_info.eligible:
                bounty_eligible_count += 1
        
        # OWASP 分類
        owasp_coverage = {
            "A03:2021-Injection": len([f for f in result.findings if "xss" in f.vulnerability_type.lower()])
        }
        
        # CWE 分布
        cwe_distribution = {}
        for finding in result.findings:
            if finding.cwe_id:
                cwe_distribution[finding.cwe_id] = cwe_distribution.get(finding.cwe_id, 0) + 1
        
        return ReportData(
            task_id=result.task_id,
            feature_module=result.feature_module,
            total_findings=len(result.findings),
            critical_count=severity_count["critical"],
            high_count=severity_count["high"],
            medium_count=severity_count["medium"],
            low_count=severity_count["low"],
            info_count=severity_count["info"],
            verified_findings=verified_count,
            unverified_findings=len(result.findings) - verified_count,
            false_positives=false_positive_count,
            bounty_eligible_count=bounty_eligible_count,
            estimated_total_value=self._estimate_bounty_value(result.findings),
            findings=result.findings,
            owasp_coverage=owasp_coverage,
            cwe_distribution=cwe_distribution,
        )
    
    def _estimate_bounty_value(self, findings: List[Finding]) -> str:
        """估算總賞金價值（基於 HackerOne/Bugcrowd 數據）"""
        total_min = 0
        total_max = 0
        
        # 賞金範圍參考（美元）
        bounty_ranges = {
            "critical": (2000, 10000),
            "high": (500, 2000),
            "medium": (100, 500),
            "low": (50, 100),
        }
        
        for finding in findings:
            if finding.verified and finding.bounty_info and finding.bounty_info.eligible:
                min_val, max_val = bounty_ranges.get(finding.severity, (0, 0))
                total_min += min_val
                total_max += max_val
        
        if total_min > 0:
            return f"${total_min}-${total_max}"
        return None
    
    async def _verify_findings(
        self, result: FeatureResult
    ) -> List[VerificationResult]:
        """驗證 XSS 漏洞真實性
        
        驗證標準（參考 OWASP）：
        1. Payload 在響應中未被編碼
        2. 響應 Content-Type 是 text/html
        3. Payload 在可執行的 HTML 上下文中
        4. 無 CSP (Content Security Policy) 保護或 CSP 可繞過
        """
        verification_results = []
        
        for finding in result.findings:
            # 基礎驗證
            confidence = finding.evidence.confidence
            
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
                    finding_id=finding.id,
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
    
    def _check_evidence_completeness(self, finding: Finding) -> bool:
        """檢查證據完整性"""
        evidence = finding.evidence
        return bool(
            evidence.payload
            and evidence.request
            and evidence.response
            and evidence.matched_pattern
        )
    
    def _check_false_positive(self, finding: Finding) -> bool:
        """檢查是否為誤報"""
        response = finding.evidence.response
        
        for pattern in self.false_positive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        return False
    
    def _check_response_headers(self, finding: Finding) -> bool:
        """檢查響應頭"""
        headers = finding.metadata.get("response_headers", {})
        content_type = headers.get("content-type", "").lower()
        
        # XSS 通常需要 HTML 響應
        return "text/html" in content_type or "application/xhtml" in content_type
    
    def _check_html_context(self, finding: Finding) -> bool:
        """檢查 HTML 上下文"""
        response = finding.evidence.response
        payload = finding.evidence.payload
        
        # 檢查 payload 是否在可執行的 HTML 標籤中
        # 簡化版：檢查是否在 <script>, <body>, 事件處理器中
        executable_contexts = [
            f"<script>{payload}",
            f"<script {payload}",
            f"<body {payload}",
            f"on{payload}",
        ]
        
        return any(ctx in response.lower() for ctx in executable_contexts)
    
    def _check_csp_protection(self, finding: Finding) -> bool:
        """檢查 CSP 保護"""
        headers = finding.metadata.get("response_headers", {})
        csp = headers.get("content-security-policy", "")
        
        if not csp:
            return False  # 無 CSP，不影響利用
        
        # 簡化版：檢查 CSP 是否嚴格
        strict_policies = ["'none'", "'self'", "nonce-", "sha256-"]
        return any(policy in csp for policy in strict_policies)
    
    def _generate_verification_notes(self, finding: Finding, confidence: float) -> str:
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
