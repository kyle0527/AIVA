"""
SARIF 2.1.0 轉換器

將 AIVA 掃描結果轉換為 SARIF (Static Analysis Results Interchange Format) 2.1.0 標準格式。

SARIF 是業界標準的靜態分析結果格式，支援：
- GitHub Security Code Scanning
- Azure DevOps
- 各種 IDE 和安全工具

參考: https://docs.oasis-open.org/sarif/sarif/v2.1.0/
"""



from datetime import UTC, datetime
import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.aiva_common.enums import Severity
from services.aiva_common.schemas import (
    CVSSv3Metrics,
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRule,
    SARIFRun,
    SARIFTool,
    Vulnerability,
)
from services.scan.models import VulnerabilityDiscovery


class SARIFConverter:
    """SARIF 格式轉換器"""

    TOOL_NAME = "AIVA Scanner"
    TOOL_VERSION = "1.0.0"
    TOOL_URI = "https://github.com/kyle0527/AIVA"

    @classmethod
    def severity_to_sarif_level(cls, severity: Severity) -> str:
        """
        將 AIVA Severity 映射到 SARIF level

        SARIF levels: "error", "warning", "note", "none"
        """
        mapping = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "note",
        }
        return mapping.get(severity, "warning")

    @classmethod
    def cvss_to_rank(cls, cvss_metrics: CVSSv3Metrics | None) -> float:
        """
        將 CVSS 評分轉換為 SARIF rank (0.0-1.0)

        CVSS 0-10 -> SARIF 0.0-1.0
        """
        if cvss_metrics and cvss_metrics.base_score is not None:
            return cvss_metrics.base_score / 10.0
        return 0.5  # 預設中等風險

    @classmethod
    def create_sarif_location(
        cls,
        url: str,
        line: int | None = None,
        parameter: str | None = None,
    ) -> SARIFLocation:
        """創建 SARIF 位置信息"""
        location = SARIFLocation(
            physicalLocation={
                "artifactLocation": {"uri": url},
            }
        )

        if line is not None:
            location.physicalLocation["region"] = {
                "startLine": line,
            }

        if parameter:
            location.logicalLocations = [
                {
                    "name": parameter,
                    "kind": "parameter",
                }
            ]

        return location

    @classmethod
    def vulnerability_to_sarif_result(cls, vuln: Vulnerability) -> SARIFResult:
        """
        將 AIVA Vulnerability 轉換為 SARIF Result

        Args:
            vuln: AIVA 漏洞對象

        Returns:
            SARIF Result 對象
        """
        # 構建 rule ID
        rule_id = f"AIVA-{vuln.vuln_type.value if hasattr(vuln.vuln_type, 'value') else vuln.vuln_type}"
        if vuln.cwe_ids:
            rule_id = vuln.cwe_ids[0]  # 優先使用 CWE ID

        # 創建位置信息
        locations = [cls.create_sarif_location(
            url=str(vuln.url),
            parameter=vuln.parameter,
        )]

        # 構建消息
        message = vuln.description
        if vuln.evidence:
            message += "\n\n證據:\n" + "\n".join(f"- {e}" for e in vuln.evidence[:3])

        # 創建 SARIF Result
        result = SARIFResult(
            ruleId=rule_id,
            level=cls.severity_to_sarif_level(vuln.severity),
            message={"text": message},
            locations=locations,
        )

        # 添加 rank (基於 CVSS)
        if vuln.cvss_metrics:
            result.rank = cls.cvss_to_rank(vuln.cvss_metrics)

        # 添加修復建議
        if vuln.remediation:
            result.fixes = [{
                "description": {"text": vuln.remediation},
            }]

        # 添加相關位置 (如果有多個證據)
        if len(vuln.evidence) > 1:
            result.relatedLocations = [
                {"message": {"text": evidence}}
                for evidence in vuln.evidence[:5]
            ]

        # 添加屬性
        result.properties = {
            "confidence": vuln.confidence.value if hasattr(vuln.confidence, 'value') else str(vuln.confidence),
            "method": vuln.method,
            "discovered_at": vuln.discovered_at.isoformat(),
        }

        if vuln.payload:
            result.properties["payload"] = vuln.payload

        return result

    @classmethod
    def vulnerability_discovery_to_sarif_result(
        cls, discovery: VulnerabilityDiscovery
    ) -> SARIFResult:
        """
        將 VulnerabilityDiscovery 轉換為 SARIF Result

        Args:
            discovery: AIVA 漏洞發現對象

        Returns:
            SARIF Result 對象
        """
        # 構建 rule ID
        rule_id = f"AIVA-{discovery.vulnerability_type}"
        if discovery.cwe_ids:
            rule_id = discovery.cwe_ids[0]

        # 創建位置信息
        locations = [cls.create_sarif_location(
            url=discovery.asset_id,  # 使用 asset_id 作為 URI
        )]

        # 構建消息
        message = discovery.description
        if discovery.evidence:
            message += "\n\n證據:\n" + "\n".join(f"- {e}" for e in discovery.evidence[:3])

        # 創建 SARIF Result
        result = SARIFResult(
            ruleId=rule_id,
            level=cls.severity_to_sarif_level(discovery.severity),
            message={"text": message},
            locations=locations,
        )

        # 添加 rank
        if discovery.cvss_metrics:
            result.rank = cls.cvss_to_rank(discovery.cvss_metrics)

        # 添加修復建議
        if discovery.remediation_advice:
            result.fixes = [{
                "description": {"text": discovery.remediation_advice},
            }]

        # 添加屬性
        result.properties = {
            "confidence": discovery.confidence.value if hasattr(discovery.confidence, 'value') else str(discovery.confidence),
            "detection_method": discovery.detection_method,
            "scanner_name": discovery.scanner_name,
            "discovered_at": discovery.discovered_at.isoformat(),
            "false_positive_likelihood": discovery.false_positive_likelihood,
        }

        if discovery.proof_of_concept:
            result.properties["proof_of_concept"] = discovery.proof_of_concept

        return result

    @classmethod
    def create_sarif_rule(cls, vuln_type: str, cwe_ids: list[str]) -> SARIFRule:
        """創建 SARIF Rule 定義"""
        rule_id = cwe_ids[0] if cwe_ids else f"AIVA-{vuln_type}"

        rule = SARIFRule(
            id=rule_id,
            name=vuln_type,
            shortDescription={"text": f"{vuln_type} vulnerability"},
            fullDescription={"text": f"Detected {vuln_type} vulnerability"},
        )

        # 添加 help URI
        if cwe_ids:
            cwe_id = cwe_ids[0].replace("CWE-", "")
            rule.helpUri = f"https://cwe.mitre.org/data/definitions/{cwe_id}.html"

        return rule

    @classmethod
    def vulnerabilities_to_sarif(
        cls,
        vulnerabilities: list[Vulnerability | VulnerabilityDiscovery],
        scan_id: str,
    ) -> SARIFReport:
        """
        將漏洞列表轉換為完整的 SARIF Report

        Args:
            vulnerabilities: 漏洞列表
            scan_id: 掃描 ID

        Returns:
            完整的 SARIF 2.1.0 報告
        """
        # 創建工具信息
        tool = SARIFTool(
            driver={
                "name": cls.TOOL_NAME,
                "version": cls.TOOL_VERSION,
                "informationUri": cls.TOOL_URI,
                "rules": [],
            }
        )

        # 創建結果列表
        results = []
        rules_map: dict[str, SARIFRule] = {}

        for vuln in vulnerabilities:
            # 轉換為 SARIF Result
            if isinstance(vuln, Vulnerability):
                result = cls.vulnerability_to_sarif_result(vuln)
                vuln_type = str(vuln.vuln_type)
                cwe_ids = vuln.cwe_ids
            else:  # VulnerabilityDiscovery
                result = cls.vulnerability_discovery_to_sarif_result(vuln)
                vuln_type = vuln.vulnerability_type
                cwe_ids = vuln.cwe_ids

            results.append(result)

            # 添加 rule 定義（避免重複）
            if result.ruleId not in rules_map:
                rules_map[result.ruleId] = cls.create_sarif_rule(vuln_type, cwe_ids)

        # 添加所有 rules
        tool.driver["rules"] = list(rules_map.values())

        # 創建 Run
        run = SARIFRun(
            tool=tool,
            results=results,
            properties={
                "scan_id": scan_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # 創建完整報告
        report = SARIFReport(
            version="2.1.0",
            schema="https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            runs=[run],
        )

        return report

    @classmethod
    def to_json(
        cls,
        vulnerabilities: list[Vulnerability | VulnerabilityDiscovery],
        scan_id: str,
    ) -> str:
        """
        將漏洞列表轉換為 SARIF JSON 字符串

        Args:
            vulnerabilities: 漏洞列表
            scan_id: 掃描 ID

        Returns:
            SARIF JSON 字符串
        """
        report = cls.vulnerabilities_to_sarif(vulnerabilities, scan_id)
        return report.model_dump_json(indent=2, exclude_none=True)


__all__ = ["SARIFConverter"]
