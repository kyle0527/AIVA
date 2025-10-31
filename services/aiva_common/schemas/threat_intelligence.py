"""
威脅情報 Schema 模型 - 基於 STIX v2.1 和 TAXII v2.1 官方標準

此模組實現了完整的 STIX Domain Objects (SDO) 和 STIX Relationship Objects (SRO)，
以及 TAXII 2.1 傳輸協議支持，用於威脅情報的標準化處理和交換。

參考標準：
- STIX v2.1 (https://docs.oasis-open.org/cti/stix/v2.1/stix-v2.1.html)
- TAXII v2.1 (https://docs.oasis-open.org/cti/taxii/v2.1/taxii-v2.1.html)
"""

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from ..enums.security import AttackTactic, AttackTechnique, IntelSource, IOCType
from .base import MessageHeader

# ==================== STIX 基礎類型 ====================


class STIXDomainObject(BaseModel):
    """STIX Domain Object 基礎類 - 所有 STIX 物件的基類"""

    type: str = Field(description="STIX 物件類型")
    spec_version: str = Field(default="2.1", description="STIX 規範版本")
    id: str = Field(description="STIX 物件唯一標識符，格式: {type}--{UUID}")
    created: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="建立時間"
    )
    modified: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="最後修改時間"
    )
    created_by_ref: str | None = Field(default=None, description="建立者引用")
    revoked: bool = Field(default=False, description="是否已撤銷")
    labels: list[str] = Field(default_factory=list, description="標籤列表")
    confidence: int | None = Field(
        default=None, ge=0, le=100, description="置信度 (0-100)"
    )
    lang: str | None = Field(default=None, description="語言代碼")
    external_references: list["ExternalReference"] = Field(
        default_factory=list, description="外部參考"
    )
    object_marking_refs: list[str] = Field(default_factory=list, description="標記參考")
    granular_markings: list["GranularMarking"] = Field(
        default_factory=list, description="細粒度標記"
    )

    @field_validator("id")
    @classmethod
    def validate_stix_id(cls, v: str) -> str:
        """驗證 STIX ID 格式"""
        if not v or "--" not in v:
            raise ValueError("STIX ID 必須格式為 {type}--{UUID}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: int | None) -> int | None:
        """驗證置信度範圍"""
        if v is not None and not (0 <= v <= 100):
            raise ValueError("置信度必須在 0-100 之間")
        return v


class STIXRelationshipObject(BaseModel):
    """STIX Relationship Object 基礎類"""

    type: Literal["relationship"] = "relationship"
    spec_version: str = Field(default="2.1", description="STIX 規範版本")
    id: str = Field(description="關係物件唯一標識符")
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified: datetime = Field(default_factory=lambda: datetime.now(UTC))
    relationship_type: str = Field(description="關係類型")
    source_ref: str = Field(description="來源物件引用")
    target_ref: str = Field(description="目標物件引用")
    confidence: int | None = Field(default=None, ge=0, le=100)
    description: str | None = Field(default=None, description="關係描述")
    revoked: bool = Field(default=False)
    external_references: list["ExternalReference"] = Field(default_factory=list)
    object_marking_refs: list[str] = Field(default_factory=list)
    granular_markings: list["GranularMarking"] = Field(default_factory=list)


class ExternalReference(BaseModel):
    """外部參考"""

    source_name: str = Field(description="來源名稱")
    description: str | None = Field(default=None, description="描述")
    url: str | None = Field(default=None, description="URL")
    hashes: dict[str, str] | None = Field(default=None, description="雜湊值")
    external_id: str | None = Field(default=None, description="外部ID")


class GranularMarking(BaseModel):
    """細粒度標記"""

    marking_ref: str = Field(description="標記引用")
    selectors: list[str] = Field(description="選擇器列表")


class KillChainPhase(BaseModel):
    """Kill Chain 階段"""

    kill_chain_name: str = Field(description="Kill Chain 名稱")
    phase_name: str = Field(description="階段名稱")


# ==================== STIX Domain Objects ====================


class AttackPattern(STIXDomainObject):
    """攻擊模式 - STIX Domain Object"""

    type: Literal["attack-pattern"] = "attack-pattern"
    name: str = Field(description="攻擊模式名稱")
    description: str | None = Field(default=None, description="攻擊模式描述")
    aliases: list[str] = Field(default_factory=list, description="別名列表")
    kill_chain_phases: list[KillChainPhase] = Field(
        default_factory=list, description="Kill Chain 階段"
    )

    # MITRE ATT&CK 支援
    mitre_attack_id: str | None = Field(default=None, description="MITRE ATT&CK ID")
    tactic: AttackTactic | None = Field(
        default=None, description="MITRE ATT&CK 戰術"
    )
    technique: AttackTechnique | None = Field(
        default=None, description="MITRE ATT&CK 技術"
    )


class Malware(STIXDomainObject):
    """惡意軟體 - STIX Domain Object"""

    type: Literal["malware"] = "malware"
    name: str = Field(description="惡意軟體名稱")
    description: str | None = Field(default=None, description="惡意軟體描述")
    malware_types: list[str] = Field(description="惡意軟體類型")
    is_family: bool = Field(default=False, description="是否為惡意軟體家族")
    aliases: list[str] = Field(default_factory=list, description="別名列表")
    kill_chain_phases: list[KillChainPhase] = Field(
        default_factory=list, description="Kill Chain 階段"
    )
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後發現時間")
    operating_system_refs: list[str] = Field(
        default_factory=list, description="作業系統引用"
    )
    architecture_execution_envs: list[str] = Field(
        default_factory=list, description="架構執行環境"
    )
    implementation_languages: list[str] = Field(
        default_factory=list, description="實現語言"
    )
    capabilities: list[str] = Field(default_factory=list, description="能力列表")


class Indicator(STIXDomainObject):
    """指標 - STIX Domain Object"""

    type: Literal["indicator"] = "indicator"
    name: str | None = Field(default=None, description="指標名稱")
    description: str | None = Field(default=None, description="指標描述")
    pattern: str = Field(description="STIX 模式表達式")
    pattern_type: str = Field(default="stix", description="模式類型")
    pattern_version: str | None = Field(default=None, description="模式版本")
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="有效起始時間"
    )
    valid_until: datetime | None = Field(default=None, description="有效結束時間")
    kill_chain_phases: list[KillChainPhase] = Field(
        default_factory=list, description="Kill Chain 階段"
    )

    # IOC 相關欄位
    ioc_type: IOCType | None = Field(default=None, description="IOC 類型")
    ioc_value: str | None = Field(default=None, description="IOC 值")


class ThreatActor(STIXDomainObject):
    """威脅行為者 - STIX Domain Object"""

    type: Literal["threat-actor"] = "threat-actor"
    name: str = Field(description="威脅行為者名稱")
    description: str | None = Field(default=None, description="威脅行為者描述")
    threat_actor_types: list[str] = Field(description="威脅行為者類型")
    aliases: list[str] = Field(default_factory=list, description="別名列表")
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後活動時間")
    roles: list[str] = Field(default_factory=list, description="角色列表")
    goals: list[str] = Field(default_factory=list, description="目標列表")
    sophistication: str | None = Field(default=None, description="複雜度")
    resource_level: str | None = Field(default=None, description="資源水平")
    primary_motivation: str | None = Field(default=None, description="主要動機")
    secondary_motivations: list[str] = Field(
        default_factory=list, description="次要動機"
    )
    personal_motivations: list[str] = Field(
        default_factory=list, description="個人動機"
    )


class IntrusionSet(STIXDomainObject):
    """入侵集合 - STIX Domain Object"""

    type: Literal["intrusion-set"] = "intrusion-set"
    name: str = Field(description="入侵集合名稱")
    description: str | None = Field(default=None, description="入侵集合描述")
    aliases: list[str] = Field(default_factory=list, description="別名列表")
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後活動時間")
    goals: list[str] = Field(default_factory=list, description="目標列表")
    resource_level: str | None = Field(default=None, description="資源水平")
    primary_motivation: str | None = Field(default=None, description="主要動機")
    secondary_motivations: list[str] = Field(
        default_factory=list, description="次要動機"
    )


class Campaign(STIXDomainObject):
    """攻擊活動 - STIX Domain Object"""

    type: Literal["campaign"] = "campaign"
    name: str = Field(description="攻擊活動名稱")
    description: str | None = Field(default=None, description="攻擊活動描述")
    aliases: list[str] = Field(default_factory=list, description="別名列表")
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後活動時間")
    objective: str | None = Field(default=None, description="目標")


class CourseOfAction(STIXDomainObject):
    """行動方案 - STIX Domain Object"""

    type: Literal["course-of-action"] = "course-of-action"
    name: str = Field(description="行動方案名稱")
    description: str | None = Field(default=None, description="行動方案描述")
    action_type: str | None = Field(default=None, description="行動類型")
    os_execution_envs: list[str] = Field(
        default_factory=list, description="作業系統執行環境"
    )
    action_bin: str | None = Field(default=None, description="行動二進制")
    action_reference: ExternalReference | None = Field(
        default=None, description="行動參考"
    )


class Vulnerability(STIXDomainObject):
    """漏洞 - STIX Domain Object"""

    type: Literal["vulnerability"] = "vulnerability"
    name: str = Field(description="漏洞名稱")
    description: str | None = Field(default=None, description="漏洞描述")

    # CVE 相關欄位
    cve_id: str | None = Field(default=None, description="CVE ID")
    cvss_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="CVSS 分數"
    )
    cvss_vector: str | None = Field(default=None, description="CVSS 向量")

    # CWE 相關欄位
    cwe_id: str | None = Field(default=None, description="CWE ID")

    # 額外屬性
    severity: str | None = Field(default=None, description="嚴重程度")
    exploitability: str | None = Field(default=None, description="可利用性")
    remediation_available: bool = Field(default=False, description="是否有修復方案")


class Tool(STIXDomainObject):
    """工具 - STIX Domain Object"""

    type: Literal["tool"] = "tool"
    name: str = Field(description="工具名稱")
    description: str | None = Field(default=None, description="工具描述")
    tool_types: list[str] = Field(description="工具類型")
    aliases: list[str] = Field(default_factory=list, description="別名列表")
    tool_version: str | None = Field(default=None, description="工具版本")
    kill_chain_phases: list[KillChainPhase] = Field(
        default_factory=list, description="Kill Chain 階段"
    )


class ObservedData(STIXDomainObject):
    """觀察數據 - STIX Domain Object"""

    type: Literal["observed-data"] = "observed-data"
    first_observed: datetime = Field(description="首次觀察時間")
    last_observed: datetime = Field(description="最後觀察時間")
    number_observed: int = Field(ge=1, description="觀察次數")
    objects: dict[str, Any] = Field(description="觀察到的對象")
    object_refs: list[str] = Field(default_factory=list, description="對象引用")


class Report(STIXDomainObject):
    """報告 - STIX Domain Object"""

    type: Literal["report"] = "report"
    name: str = Field(description="報告名稱")
    description: str | None = Field(default=None, description="報告描述")
    report_types: list[str] = Field(description="報告類型")
    published: datetime = Field(description="發布時間")
    object_refs: list[str] = Field(description="引用對象列表")


# ==================== STIX 關係物件 ====================


class Relationship(STIXRelationshipObject):
    """標準關係物件"""

    # 常見關係類型
    USES: ClassVar[str] = "uses"
    INDICATES: ClassVar[str] = "indicates"
    TARGETS: ClassVar[str] = "targets"
    ATTRIBUTED_TO: ClassVar[str] = "attributed-to"
    MITIGATES: ClassVar[str] = "mitigates"
    DERIVED_FROM: ClassVar[str] = "derived-from"
    RELATED_TO: ClassVar[str] = "related-to"
    VARIANT_OF: ClassVar[str] = "variant-of"


class Sighting(BaseModel):
    """目擊物件 - STIX Relationship Object"""

    type: Literal["sighting"] = "sighting"
    spec_version: str = Field(default="2.1")
    id: str = Field(description="目擊物件唯一標識符")
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sighting_of_ref: str = Field(description="目擊對象引用")
    observed_data_refs: list[str] = Field(
        default_factory=list, description="觀察數據引用"
    )
    where_sighted_refs: list[str] = Field(
        default_factory=list, description="目擊位置引用"
    )
    first_seen: datetime | None = Field(default=None, description="首次目擊時間")
    last_seen: datetime | None = Field(default=None, description="最後目擊時間")
    count: int | None = Field(default=None, ge=1, description="目擊次數")
    summary: bool = Field(default=False, description="是否為摘要")
    confidence: int | None = Field(default=None, ge=0, le=100, description="置信度")
    revoked: bool = Field(default=False)
    external_references: list[ExternalReference] = Field(default_factory=list)


# ==================== STIX Bundle ====================


class Bundle(BaseModel):
    """STIX Bundle - 用於批量傳輸 STIX 物件"""

    type: Literal["bundle"] = "bundle"
    id: str = Field(description="Bundle 唯一標識符")
    objects: list[
        AttackPattern | Malware | Indicator | ThreatActor | IntrusionSet | Campaign | CourseOfAction | Vulnerability | Tool | ObservedData | Report | Relationship | Sighting
    ] = Field(description="STIX 物件列表")

    @classmethod
    def create_bundle(cls, objects: list[Any]) -> "Bundle":
        """建立新的 Bundle"""
        bundle_id = f"bundle--{uuid4()}"
        return cls(id=bundle_id, objects=objects)


# ==================== TAXII 2.1 支援 ====================


class TAXIICollection(BaseModel):
    """TAXII Collection 物件"""

    id: str = Field(description="Collection 唯一標識符")
    title: str = Field(description="Collection 標題")
    description: str | None = Field(default=None, description="Collection 描述")
    alias: str | None = Field(default=None, description="Collection 別名")
    can_read: bool = Field(default=True, description="是否可讀")
    can_write: bool = Field(default=False, description="是否可寫")
    media_types: list[str] = Field(
        default_factory=lambda: ["application/stix+json;version=2.1"],
        description="支援的媒體類型",
    )


class TAXIIManifestEntry(BaseModel):
    """TAXII Manifest 條目"""

    id: str = Field(description="物件 ID")
    date_added: datetime | None = Field(default=None, description="新增日期")
    version: str | None = Field(default=None, description="物件版本")
    media_type: str | None = Field(default=None, description="媒體類型")


class TAXIIManifest(BaseModel):
    """TAXII Manifest 回應"""

    objects: list[TAXIIManifestEntry] = Field(description="物件清單")
    more: bool = Field(default=False, description="是否有更多物件")
    next: str | None = Field(default=None, description="下一頁 URL")


class TAXIIErrorMessage(BaseModel):
    """TAXII 錯誤訊息"""

    title: str = Field(description="錯誤標題")
    description: str | None = Field(default=None, description="錯誤描述")
    error_id: str | None = Field(default=None, description="錯誤 ID")
    error_code: str | None = Field(default=None, description="錯誤代碼")
    http_status: int | None = Field(default=None, description="HTTP 狀態碼")
    external_details: str | None = Field(default=None, description="外部詳細資訊")
    details: dict[str, Any] | None = Field(default=None, description="詳細資訊")


class TAXIIStatus(BaseModel):
    """TAXII 狀態物件 - 用於非同步操作"""

    id: str = Field(description="狀態 ID")
    status: str = Field(description="狀態：pending、complete、error")
    request_timestamp: datetime | None = Field(
        default=None, description="請求時間戳"
    )
    total_count: int = Field(description="總物件數")
    success_count: int = Field(description="成功處理數")
    failure_count: int = Field(description="失敗處理數")
    pending_count: int = Field(description="待處理數")


# ==================== 威脅情報整合模型 ====================


class ThreatIntelligenceReport(BaseModel):
    """威脅情報報告 - AIVA 內部使用"""

    header: MessageHeader = Field(description="訊息標頭")
    report_id: str = Field(description="報告唯一標識符")
    title: str = Field(description="報告標題")
    description: str | None = Field(default=None, description="報告描述")

    # STIX 物件
    indicators: list[Indicator] = Field(default_factory=list, description="指標列表")
    attack_patterns: list[AttackPattern] = Field(
        default_factory=list, description="攻擊模式列表"
    )
    malware: list[Malware] = Field(default_factory=list, description="惡意軟體列表")
    threat_actors: list[ThreatActor] = Field(
        default_factory=list, description="威脅行為者列表"
    )
    vulnerabilities: list[Vulnerability] = Field(
        default_factory=list, description="漏洞列表"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="關係列表"
    )

    # 元資料
    confidence: int = Field(ge=0, le=100, description="整體置信度")
    severity: str = Field(description="威脅嚴重程度")
    source: IntelSource = Field(description="情報來源")
    tlp_marking: str = Field(default="TLP:WHITE", description="TLP 標記")

    # 時間資訊
    intelligence_date: datetime = Field(description="情報日期")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # TAXII 相關
    collection_id: str | None = Field(
        default=None, description="TAXII Collection ID"
    )

    def to_stix_bundle(self) -> Bundle:
        """轉換為 STIX Bundle"""
        all_objects = []
        all_objects.extend(self.indicators)
        all_objects.extend(self.attack_patterns)
        all_objects.extend(self.malware)
        all_objects.extend(self.threat_actors)
        all_objects.extend(self.vulnerabilities)
        all_objects.extend(self.relationships)

        return Bundle.create_bundle(all_objects)


class IOCEnrichment(BaseModel):
    """IOC 豐富化資料"""

    ioc_value: str = Field(description="IOC 值")
    ioc_type: IOCType = Field(description="IOC 類型")

    # 豐富化資訊
    reputation_score: int | None = Field(
        default=None, ge=0, le=100, description="聲譽分數"
    )
    is_malicious: bool | None = Field(default=None, description="是否惡意")
    first_seen: datetime | None = Field(default=None, description="首次發現")
    last_seen: datetime | None = Field(default=None, description="最後發現")

    # 地理位置資訊（針對 IP）
    country: str | None = Field(default=None, description="國家")
    city: str | None = Field(default=None, description="城市")
    asn: str | None = Field(default=None, description="ASN")

    # 威脅情報來源
    sources: list[IntelSource] = Field(default_factory=list, description="情報來源列表")
    tags: list[str] = Field(default_factory=list, description="標籤")

    # STIX 指標
    stix_indicator: Indicator | None = Field(
        default=None, description="對應的 STIX 指標"
    )


# ==================== HackerOne 優化相關 ====================


class LowValueVulnerabilityPattern(BaseModel):
    """低價值高概率漏洞模式 - 用於穩定收入策略"""

    pattern_id: str = Field(description="模式唯一標識符")
    name: str = Field(description="模式名稱")
    description: str = Field(description="模式描述")

    # 獎金預估
    min_bounty: int = Field(description="最低獎金預估（美元）")
    max_bounty: int = Field(description="最高獎金預估（美元）")
    avg_bounty: int = Field(description="平均獎金（美元）")
    success_rate: float = Field(ge=0.0, le=1.0, description="發現成功率")

    # 檢測模式
    detection_patterns: list[str] = Field(description="檢測模式列表")
    test_vectors: list[str] = Field(description="測試向量")

    # STIX 對應
    attack_pattern: AttackPattern | None = Field(
        default=None, description="對應的攻擊模式"
    )
    indicators: list[Indicator] = Field(default_factory=list, description="相關指標")

    # 時間估算
    avg_discovery_time_hours: float = Field(description="平均發現時間（小時）")
    effort_level: str = Field(description="努力程度：low、medium、high")

    # 程式分類
    suitable_program_types: list[str] = Field(description="適合的程式類型")
    excluded_domains: list[str] = Field(default_factory=list, description="排除域名")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BugBountyIntelligence(BaseModel):
    """Bug Bounty 情報模型"""

    intelligence_id: str = Field(description="情報唯一標識符")
    program_name: str = Field(description="程式名稱")
    program_url: str = Field(description="程式 URL")

    # 程式資訊
    program_type: str = Field(description="程式類型：web、mobile、api、cloud等")
    max_bounty: int = Field(description="最高獎金")
    avg_bounty: int = Field(description="平均獎金")
    response_time_days: int | None = Field(
        default=None, description="平均回應時間（天）"
    )

    # 技術棧
    technologies: list[str] = Field(default_factory=list, description="技術棧")
    frameworks: list[str] = Field(default_factory=list, description="框架")
    languages: list[str] = Field(default_factory=list, description="程式語言")

    # 威脅情報整合
    common_vulnerabilities: list[LowValueVulnerabilityPattern] = Field(
        default_factory=list, description="常見漏洞模式"
    )
    stix_bundle: Bundle | None = Field(default=None, description="相關 STIX Bundle")

    # 成功策略
    recommended_approach: str = Field(description="推薦方法")
    priority_score: float = Field(ge=0.0, le=10.0, description="優先級分數")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
