"""
AIVA Common Models - 共享基礎模型

此文件包含所有模組共享的核心基礎設施和官方標準實現。
這些定義是跨模組的基礎，不屬於任何特定業務領域。

包含內容：
1. 核心消息協議 (MessageHeader, AivaMessage)
2. 通用認證和限流 (Authentication, RateLimit)
3. 官方安全標準 (CVSS v3.1, SARIF v2.1.0, CVE/CWE)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .enums import ModuleName, Topic

# ==================== 核心消息協議 ====================


class MessageHeader(BaseModel):
    """AIVA 統一消息頭

    所有模組間通信都必須使用此消息頭格式。
    """

    message_id: str = Field(description="唯一消息ID")
    trace_id: str = Field(description="追蹤ID，用於關聯相關消息")
    correlation_id: str | None = Field(default=None, description="關聯ID")
    source_module: ModuleName = Field(description="來源模組")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="消息時間戳"
    )
    version: str = Field(default="1.0", description="消息格式版本")


class AivaMessage(BaseModel):
    """AIVA 統一消息格式

    所有模組間的消息都必須遵循此格式。
    """

    header: MessageHeader = Field(description="消息頭")
    topic: Topic = Field(description="消息主題")
    schema_version: str = Field(default="1.0", description="Schema版本")
    payload: dict[str, Any] = Field(description="消息載荷")


# ==================== 通用認證和控制 ====================


class Authentication(BaseModel):
    """通用認證配置"""

    method: str = Field(default="none", description="認證方法")
    credentials: dict[str, str] | None = Field(default=None, description="認證憑據")


class RateLimit(BaseModel):
    """速率限制配置"""

    requests_per_second: int = Field(default=25, description="每秒請求數")
    burst: int = Field(default=50, description="突發流量限制")

    @field_validator("requests_per_second", "burst")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate limit must be non-negative")
        return v


# ==================== CVSS v3.1 官方標準 ====================


class CVSSv3Metrics(BaseModel):
    """CVSS v3.1 評分指標

    符合標準: CVSS v3.1 Specification (https://www.first.org/cvss/v3.1/specification-document)
    完整實現了 CVSS v3.1 的所有評分維度。
    """

    # Base Metrics (基礎指標) - 必填
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

    # Temporal Metrics (時間指標) - 可選
    exploit_code_maturity: str | None = Field(
        default=None,
        description="利用代碼成熟度: X(Not Defined), U(Unproven), P(Proof-of-Concept), F(Functional), H(High)",
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

    # Environmental Metrics (環境指標) - 可選
    confidentiality_requirement: str | None = Field(
        default=None,
        description="機密性要求: X(Not Defined), L(Low), M(Medium), H(High)",
        pattern=r"^[XLMH]$",
    )
    integrity_requirement: str | None = Field(
        default=None,
        description="完整性要求: X(Not Defined), L(Low), M(Medium), H(High)",
        pattern=r"^[XLMH]$",
    )
    availability_requirement: str | None = Field(
        default=None,
        description="可用性要求: X(Not Defined), L(Low), M(Medium), H(High)",
        pattern=r"^[XLMH]$",
    )

    # Calculated Scores (計算得分)
    base_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="基礎評分 (0.0-10.0)"
    )
    temporal_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="時間評分 (0.0-10.0)"
    )
    environmental_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="環境評分 (0.0-10.0)"
    )

    # Vector String
    vector_string: str | None = Field(
        default=None, description="CVSS向量字符串，如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    )


# ==================== CVE/CWE/CAPEC 官方標準 ====================


class CVEReference(BaseModel):
    """CVE (Common Vulnerabilities and Exposures) 引用

    符合標準: CVE Numbering Authority
    """

    cve_id: str = Field(description="CVE標識符", pattern=r"^CVE-\d{4}-\d{4,}$")
    description: str | None = Field(default=None, description="CVE描述")
    published_date: datetime | None = Field(default=None, description="發布日期")
    last_modified: datetime | None = Field(default=None, description="最後修改日期")
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0, description="CVSS評分")
    references: list[str] = Field(default_factory=list, description="參考鏈接")


class CWEReference(BaseModel):
    """CWE (Common Weakness Enumeration) 引用

    符合標準: MITRE CWE
    """

    cwe_id: str = Field(description="CWE標識符", pattern=r"^CWE-\d+$")
    name: str | None = Field(default=None, description="CWE名稱")
    description: str | None = Field(default=None, description="CWE描述")
    weakness_type: str | None = Field(default=None, description="弱點類型")


class CAPECReference(BaseModel):
    """CAPEC (Common Attack Pattern Enumeration and Classification) 引用

    符合標準: MITRE CAPEC
    """

    capec_id: str = Field(description="CAPEC標識符", pattern=r"^CAPEC-\d+$")
    name: str | None = Field(default=None, description="CAPEC名稱")
    description: str | None = Field(default=None, description="攻擊模式描述")
    related_cwes: list[str] = Field(default_factory=list, description="相關CWE列表")


# ==================== SARIF v2.1.0 官方標準 ====================


class SARIFLocation(BaseModel):
    """SARIF 位置信息

    符合標準: SARIF v2.1.0 (Static Analysis Results Interchange Format)
    """

    uri: str = Field(description="文件URI")
    start_line: int = Field(ge=1, description="起始行號")
    start_column: int | None = Field(default=None, ge=1, description="起始列號")
    end_line: int | None = Field(default=None, ge=1, description="結束行號")
    end_column: int | None = Field(default=None, ge=1, description="結束列號")


class SARIFResult(BaseModel):
    """SARIF 結果

    符合標準: SARIF v2.1.0
    """

    rule_id: str = Field(description="規則ID")
    level: str = Field(description="嚴重程度", pattern=r"^(error|warning|note|none)$")
    message: str = Field(description="消息內容")
    locations: list[SARIFLocation] = Field(description="位置列表")

    # 可選欄位
    kind: str = Field(default="fail", description="結果種類")
    properties: dict[str, Any] = Field(default_factory=dict, description="自定義屬性")


class SARIFRule(BaseModel):
    """SARIF 規則定義

    符合標準: SARIF v2.1.0
    """

    id: str = Field(description="規則ID")
    name: str | None = Field(default=None, description="規則名稱")
    short_description: str | None = Field(default=None, description="簡短描述")
    full_description: str | None = Field(default=None, description="完整描述")
    help_uri: str | None = Field(default=None, description="幫助鏈接")
    properties: dict[str, Any] = Field(default_factory=dict, description="規則屬性")


class SARIFTool(BaseModel):
    """SARIF 工具信息

    符合標準: SARIF v2.1.0
    """

    name: str = Field(description="工具名稱")
    version: str | None = Field(default=None, description="工具版本")
    information_uri: str | None = Field(default=None, description="工具信息URI")
    rules: list[SARIFRule] = Field(default_factory=list, description="規則列表")


class SARIFRun(BaseModel):
    """SARIF 運行記錄

    符合標準: SARIF v2.1.0
    """

    tool: SARIFTool = Field(description="工具信息")
    results: list[SARIFResult] = Field(description="結果列表")

    # 可選欄位
    invocations: list[dict] = Field(default_factory=list, description="調用信息")
    properties: dict[str, Any] = Field(default_factory=dict, description="運行屬性")


class SARIFReport(BaseModel):
    """SARIF 報告

    符合標準: SARIF v2.1.0 完整格式
    官方規範: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
    """

    version: str = Field(default="2.1.0", description="SARIF版本")
    schema: str = Field(
        default="https://json.schemastore.org/sarif-2.1.0.json",
        description="JSON Schema URI"
    )
    runs: list[SARIFRun] = Field(description="運行列表")

    # 元數據
    properties: dict[str, Any] = Field(default_factory=dict, description="報告屬性")


__all__ = [
    # 核心消息協議
    "MessageHeader",
    "AivaMessage",
    # 通用認證和控制
    "Authentication",
    "RateLimit",
    # CVSS v3.1
    "CVSSv3Metrics",
    # CVE/CWE/CAPEC
    "CVEReference",
    "CWEReference",
    "CAPECReference",
    # SARIF v2.1.0
    "SARIFLocation",
    "SARIFResult",
    "SARIFRule",
    "SARIFTool",
    "SARIFRun",
    "SARIFReport",
]
