"""
AIVA 官方安全標準實現

包含所有官方安全標準的完整實現：
- CVSS v3.1: Common Vulnerability Scoring System
- SARIF v2.1.0: Static Analysis Results Interchange Format
- CVE: Common Vulnerabilities and Exposures
- CWE: Common Weakness Enumeration
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ==================== CVSS v3.1 漏洞評分系統 ====================


class CVSSv3Metrics(BaseModel):
    """CVSS v3.1 評分指標

    符合標準: CVSS v3.1 Specification (https://www.first.org/cvss/v3.1/specification-document)
    """

    # Base Metrics (基礎指標)
    attack_vector: str = Field(
        default="N",
        description="攻擊向量: N(Network), A(Adjacent), L(Local), P(Physical)",
        pattern=r"^[NALP]$",
    )
    attack_complexity: str = Field(
        default="L", description="攻擊複雜度: L(Low), H(High)", pattern=r"^[LH]$"
    )
    privileges_required: str = Field(
        default="N",
        description="所需權限: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )
    user_interaction: str = Field(
        default="N", description="用戶交互: N(None), R(Required)", pattern=r"^[NR]$"
    )
    scope: str = Field(
        default="U", description="影響範圍: U(Unchanged), C(Changed)", pattern=r"^[UC]$"
    )
    confidentiality_impact: str = Field(
        default="N",
        description="機密性影響: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )
    integrity_impact: str = Field(
        default="N",
        description="完整性影響: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )
    availability_impact: str = Field(
        default="N",
        description="可用性影響: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )

    # Temporal Metrics (時間指標 - 可選)
    exploit_code_maturity: str | None = Field(
        default=None,
        description="漏洞利用程式碼成熟度: X(Not Defined), U(Unproven), P(Proof-of-Concept), F(Functional), H(High)",
        pattern=r"^[XUPFH]$",
    )
    remediation_level: str | None = Field(
        default=None,
        description="修復級別: X(Not Defined), O(Official Fix), T(Temporary Fix), W(Workaround), U(Unavailable)",
        pattern=r"^[XOTWU]$",
    )
    report_confidence: str | None = Field(
        default=None,
        description="報告可信度: X(Not Defined), U(Unknown), R(Reasonable), C(Confirmed)",
        pattern=r"^[XURC]$",
    )

    def calculate_base_score(self) -> float:
        """計算 CVSS v3.1 基礎分數

        Returns:
            基礎分數 (0.0 - 10.0)
        """
        # Impact Sub-Score (ISC)
        impact_values = {"N": 0.0, "L": 0.22, "H": 0.56}
        isc_base = 1 - (1 - impact_values[self.confidentiality_impact]) * (
            1 - impact_values[self.integrity_impact]
        ) * (1 - impact_values[self.availability_impact])

        if self.scope == "U":
            impact = 6.42 * isc_base
        else:  # Changed
            impact = 7.52 * (isc_base - 0.029) - 3.25 * ((isc_base - 0.02) ** 15)

        # Exploitability Sub-Score (ESS)
        av_values = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
        ac_values = {"L": 0.77, "H": 0.44}
        pr_values_unchanged = {"N": 0.85, "L": 0.62, "H": 0.27}
        pr_values_changed = {"N": 0.85, "L": 0.68, "H": 0.5}
        ui_values = {"N": 0.85, "R": 0.62}

        pr_values = pr_values_unchanged if self.scope == "U" else pr_values_changed

        exploitability = (
            8.22
            * av_values[self.attack_vector]
            * ac_values[self.attack_complexity]
            * pr_values[self.privileges_required]
            * ui_values[self.user_interaction]
        )

        # Base Score
        if impact <= 0:
            return 0.0
        elif self.scope == "U":
            base_score = min(impact + exploitability, 10.0)
        else:
            base_score = min(1.08 * (impact + exploitability), 10.0)

        # Round up to one decimal place
        return round(base_score * 10) / 10

    def to_vector_string(self) -> str:
        """生成 CVSS v3.1 向量字串

        Returns:
            CVSS 向量字串，例如: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        """
        vector = (
            f"CVSS:3.1/AV:{self.attack_vector}/AC:{self.attack_complexity}/"
            f"PR:{self.privileges_required}/UI:{self.user_interaction}/"
            f"S:{self.scope}/C:{self.confidentiality_impact}/"
            f"I:{self.integrity_impact}/A:{self.availability_impact}"
        )

        # 添加時間指標（如果定義）
        if self.exploit_code_maturity and self.exploit_code_maturity != "X":
            vector += f"/E:{self.exploit_code_maturity}"
        if self.remediation_level and self.remediation_level != "X":
            vector += f"/RL:{self.remediation_level}"
        if self.report_confidence and self.report_confidence != "X":
            vector += f"/RC:{self.report_confidence}"

        return vector


# ==================== CVE & CWE 標準 ====================


class CVEReference(BaseModel):
    """CVE 參考資訊

    符合標準: CVE Numbering Authority (https://www.cve.org/)
    """

    cve_id: str = Field(
        description="CVE ID (格式: CVE-YYYY-NNNNN)",
        pattern=r"^CVE-\d{4}-\d{4,}$",
    )
    description: str | None = None
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0)
    cvss_vector: str | None = None
    references: list[str] = Field(default_factory=list)
    published_date: datetime | None = None
    last_modified_date: datetime | None = None


class CWEReference(BaseModel):
    """CWE 參考資訊

    符合標準: Common Weakness Enumeration (https://cwe.mitre.org/)
    """

    cwe_id: str = Field(description="CWE ID (格式: CWE-XXX)", pattern=r"^CWE-\d+$")
    name: str | None = None
    description: str | None = None
    weakness_category: str | None = None  # "Class", "Base", "Variant", "Compound"
    likelihood_of_exploit: str | None = None  # "High", "Medium", "Low"


# ==================== SARIF v2.1.0 標準 ====================


class SARIFLocation(BaseModel):
    """SARIF 位置資訊

    符合標準: SARIF v2.1.0 (https://docs.oasis-open.org/sarif/sarif/v2.1.0/)
    """

    uri: str  # 檔案 URI
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    snippet: str | None = None  # 代碼片段


class SARIFResult(BaseModel):
    """SARIF 結果項

    符合標準: SARIF v2.1.0
    """

    rule_id: str  # 規則 ID (可以是 CWE ID 或自定義規則)
    level: str = Field(
        default="warning",
        description="嚴重性級別",
        pattern=r"^(none|note|warning|error)$",
    )
    message: str  # 訊息文本
    locations: list[SARIFLocation] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class SARIFReport(BaseModel):
    """SARIF 報告

    符合標準: SARIF v2.1.0
    完整規範: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
    """

    version: str = "2.1.0"
    schema_uri: str = (
        "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/"
        "master/Schemata/sarif-schema-2.1.0.json"
    )
    runs: list[dict[str, Any]] = Field(default_factory=list)

    def add_run(
        self,
        tool_name: str,
        tool_version: str,
        results: list[SARIFResult],
    ) -> None:
        """添加一個掃描運行結果

        Args:
            tool_name: 工具名稱
            tool_version: 工具版本
            results: 結果列表
        """
        run = {
            "tool": {
                "driver": {
                    "name": tool_name,
                    "version": tool_version,
                    "informationUri": "https://github.com/kyle0527/AIVA",
                }
            },
            "results": [
                {
                    "ruleId": r.rule_id,
                    "level": r.level,
                    "message": {"text": r.message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": loc.uri},
                                "region": {
                                    "startLine": loc.start_line,
                                    "startColumn": loc.start_column,
                                    "endLine": loc.end_line,
                                    "endColumn": loc.end_column,
                                    "snippet": (
                                        {"text": loc.snippet} if loc.snippet else None
                                    ),
                                },
                            }
                        }
                        for loc in r.locations
                    ],
                    "properties": r.properties,
                }
                for r in results
            ],
        }
        self.runs.append(run)


__all__ = [
    "CVSSv3Metrics",
    "CVEReference",
    "CWEReference",
    "SARIFLocation",
    "SARIFResult",
    "SARIFReport",
]
