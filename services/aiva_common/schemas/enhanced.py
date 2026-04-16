"""
Enhanced 版本 Schemas

此模組定義了各種增強版本的資料模型,提供更詳細的字段和擴展功能。
"""

from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, HttpUrl, field_validator

from ..enums import ModuleName, Severity, TestStatus
from .ai import CVSSv3Metrics, EnhancedVulnerability, SARIFLocation, SARIFResult
from .base import RiskFactor, TaskDependency
from .findings import FindingEvidence, FindingImpact, FindingRecommendation, Target

# ============================================================================
# Enhanced 版本
# ============================================================================


class EnhancedFindingPayload(BaseModel):
    """增強版漏洞發現 Payload - 集成所有業界標準

    此 Schema 擴展了基礎 FindingPayload，添加了完整的標準支持
    """

    finding_id: str
    task_id: str
    scan_id: str
    status: str

    # 使用增強版漏洞資訊
    vulnerability: EnhancedVulnerability

    target: Target
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None

    # SARIF 格式支持
    sarif_result: SARIFResult | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("finding_id")
    @classmethod
    def validate_finding_id(cls, v: str) -> str:
        if not v.startswith("finding_"):
            raise ValueError("finding_id must start with 'finding_'")
        return v

    def to_sarif_result(self) -> SARIFResult:
        """轉換為 SARIF 結果格式

        Returns:
            SARIF 結果項
        """
        if self.sarif_result:
            return self.sarif_result

        # 構建 SARIF 結果
        level_mapping = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "warning",
            "informational": "note",
        }

        locations = []
        if self.target.url:
            locations.append(SARIFLocation(uri=str(self.target.url)))

        return SARIFResult(
            rule_id=(
                self.vulnerability.vulnerability_id
                if self.vulnerability.vulnerability_id
                else f"AIVA-{self.vulnerability.title}"
            ),
            level=cast(
                Literal["error", "warning", "info", "note"],
                level_mapping.get(
                    (
                        self.vulnerability.severity.lower()
                        if self.vulnerability.severity
                        else "medium"
                    ),
                    "warning",
                ),
            ),
            message=self.vulnerability.description
            or f"{self.vulnerability.title} detected",
            locations=locations,
            properties={
                "finding_id": self.finding_id,
                "confidence": self.vulnerability.ai_confidence,
                "cvss_score": (
                    getattr(self.vulnerability.cvss_metrics, "base_score", None)
                    if self.vulnerability.cvss_metrics
                    else None
                ),
                "exploitability_score": self.vulnerability.exploitability_score,
            },
        )


# ==================== AI 訓練與學習合約 ====================


class EnhancedScanScope(BaseModel):
    """增強掃描範圍定義"""

    included_hosts: list[str] = Field(default_factory=list, description="包含的主機")
    excluded_hosts: list[str] = Field(default_factory=list, description="排除的主機")
    included_paths: list[str] = Field(default_factory=list, description="包含的路徑")
    excluded_paths: list[str] = Field(default_factory=list, description="排除的路徑")
    max_depth: int = Field(default=5, ge=1, le=20, description="最大掃描深度")


class EnhancedScanRequest(BaseModel):
    """增強掃描請求"""

    scan_id: str = Field(description="掃描ID", pattern=r"^scan_[a-zA-Z0-9_]+$")
    targets: list[HttpUrl] = Field(description="目標URL列表", min_length=1)
    scope: EnhancedScanScope = Field(description="掃描範圍")
    strategy: str = Field(description="掃描策略", pattern=r"^[a-zA-Z0-9_]+$")
    priority: int = Field(default=5, ge=1, le=10, description="優先級 1-10")
    max_duration: int = Field(default=3600, ge=60, description="最大執行時間(秒)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")

    @field_validator("scan_id")
    @classmethod
    def validate_scan_id(cls, v: str) -> str:
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        return v


class EnhancedFunctionTaskTarget(BaseModel):
    """增強功能測試目標"""

    url: HttpUrl = Field(description="目標URL")
    method: str = Field(default="GET", description="HTTP方法")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP標頭")
    cookies: dict[str, str] = Field(default_factory=dict, description="Cookie")
    parameters: dict[str, str] = Field(default_factory=dict, description="參數")
    body: str | None = Field(default=None, description="請求體")
    auth_required: bool = Field(default=False, description="是否需要認證")


class EnhancedIOCRecord(BaseModel):
    """增強威脅指標記錄 (Indicator of Compromise)"""

    ioc_id: str = Field(description="IOC唯一標識符")
    ioc_type: str = Field(
        description="IOC類型"
    )  # "ip", "domain", "url", "hash", "email"
    value: str = Field(description="IOC值")

    # 威脅信息
    threat_type: str | None = Field(default=None, description="威脅類型")
    malware_family: str | None = Field(default=None, description="惡意軟體家族")
    campaign: str | None = Field(default=None, description="攻擊活動")

    # 評級信息
    severity: Severity = Field(description="嚴重程度")
    confidence: int = Field(ge=0, le=100, description="可信度 0-100")
    reputation_score: int = Field(ge=0, le=100, description="聲譽分數")

    # 時間信息
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後發現時間")
    expires_at: datetime | None = Field(default=None, description="過期時間")

    # 標籤和分類
    tags: list[str] = Field(default_factory=list, description="標籤")
    mitre_techniques: list[str] = Field(
        default_factory=list, description="MITRE ATT&CK技術"
    )

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedRiskAssessment(BaseModel):
    """增強風險評估"""

    assessment_id: str = Field(description="評估ID")
    target_id: str = Field(description="目標ID")

    # 風險評分
    overall_risk_score: float = Field(ge=0.0, le=10.0, description="總體風險評分")
    likelihood_score: float = Field(ge=0.0, le=10.0, description="可能性評分")
    impact_score: float = Field(ge=0.0, le=10.0, description="影響評分")

    # 風險分級
    risk_level: Severity = Field(description="風險級別")
    risk_category: str = Field(description="風險分類")

    # 風險因子
    risk_factors: list[RiskFactor] = Field(description="風險因子列表")

    # CVSS 整合
    cvss_metrics: CVSSv3Metrics | None = Field(default=None, description="CVSS評分")

    # 業務影響
    business_impact: str | None = Field(default=None, description="業務影響描述")
    affected_assets: list[str] = Field(default_factory=list, description="受影響資產")

    # 緩解措施
    mitigation_strategies: list[str] = Field(
        default_factory=list, description="緩解策略"
    )
    residual_risk: float = Field(ge=0.0, le=10.0, description="殘餘風險")

    # 時間戳
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = Field(default=None, description="有效期限")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedAttackPathNode(BaseModel):
    """增強攻擊路徑節點"""

    node_id: str = Field(description="節點ID")
    node_type: str = Field(
        description="節點類型"
    )  # "asset", "vulnerability", "technique"
    name: str = Field(description="節點名稱")
    description: str | None = Field(default=None, description="節點描述")

    # 節點屬性
    exploitability: float = Field(ge=0.0, le=10.0, description="可利用性")
    impact: float = Field(ge=0.0, le=10.0, description="影響度")
    difficulty: float = Field(ge=0.0, le=10.0, description="難度")

    # MITRE ATT&CK 映射
    mitre_technique: str | None = Field(default=None, description="MITRE技術ID")
    mitre_tactic: str | None = Field(default=None, description="MITRE戰術")

    # 前置條件和後果
    prerequisites: list[str] = Field(default_factory=list, description="前置條件")
    consequences: list[str] = Field(default_factory=list, description="後果")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedAttackPath(BaseModel):
    """增強攻擊路徑"""

    path_id: str = Field(description="路徑ID")
    target_asset: str = Field(description="目標資產")

    # 路徑信息
    nodes: list[EnhancedAttackPathNode] = Field(description="路徑節點")
    edges: list[dict[str, str]] = Field(description="邊關係")

    # 路徑評估
    path_feasibility: float = Field(ge=0.0, le=1.0, description="路徑可行性")
    estimated_time: int = Field(ge=0, description="估計時間(分鐘)")
    skill_level_required: str = Field(description="所需技能等級")

    # 風險評估
    success_probability: float = Field(ge=0.0, le=1.0, description="成功概率")
    detection_probability: float = Field(ge=0.0, le=1.0, description="被檢測概率")
    overall_risk: float = Field(ge=0.0, le=10.0, description="總體風險")

    # 緩解措施
    blocking_controls: list[str] = Field(default_factory=list, description="阻斷控制")
    detection_controls: list[str] = Field(default_factory=list, description="檢測控制")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedTaskExecution(BaseModel):
    """增強任務執行"""

    task_id: str = Field(description="任務ID")
    task_type: str = Field(description="任務類型")
    module_name: ModuleName = Field(description="執行模組")

    # 任務配置
    priority: int = Field(ge=1, le=10, description="優先級")
    timeout: int = Field(default=3600, ge=60, description="超時時間(秒)")
    retry_count: int = Field(default=3, ge=0, description="重試次數")

    # 依賴關係
    dependencies: list[TaskDependency] = Field(
        default_factory=list, description="任務依賴"
    )

    # 執行狀態
    status: TestStatus = Field(description="執行狀態")
    progress: float = Field(ge=0.0, le=1.0, description="執行進度")

    # 結果信息
    result_data: dict[str, Any] = Field(default_factory=dict, description="結果數據")
    error_message: str | None = Field(default=None, description="錯誤消息")

    # 資源使用
    cpu_usage: float | None = Field(default=None, description="CPU使用率")
    memory_usage: int | None = Field(default=None, description="內存使用(MB)")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = Field(default=None, description="開始時間")
    completed_at: datetime | None = Field(default=None, description="完成時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v


class EnhancedVulnerabilityCorrelation(BaseModel):
    """增強漏洞關聯分析"""

    correlation_id: str = Field(description="關聯分析ID")
    primary_vulnerability: str = Field(description="主要漏洞ID")

    # 關聯漏洞
    related_vulnerabilities: list[str] = Field(description="相關漏洞列表")
    correlation_strength: float = Field(ge=0.0, le=1.0, description="關聯強度")

    # 關聯類型
    correlation_type: str = Field(description="關聯類型")

    # 組合影響
    combined_risk_score: float = Field(ge=0.0, le=10.0, description="組合風險評分")
    exploitation_complexity: float = Field(ge=0.0, le=10.0, description="利用複雜度")

    # 攻擊場景
    attack_scenarios: list[str] = Field(default_factory=list, description="攻擊場景")
    recommended_order: list[str] = Field(
        default_factory=list, description="建議利用順序"
    )

    # 緩解建議
    coordinated_mitigation: list[str] = Field(
        default_factory=list, description="協調緩解措施"
    )
    priority_ranking: list[str] = Field(default_factory=list, description="優先級排序")

    # 時間戳
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
