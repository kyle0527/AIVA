"""
自動化創建 schemas 剩餘檔案的腳本
"""

import sys
from pathlib import Path

# 設定路徑 - 使用相對路徑計算
SCHEMAS_DIR = Path(__file__).parent.parent.parent / "services" / "aiva_common" / "schemas"

# 定義剩餘檔案的內容模板
REMAINING_FILES = {
    "telemetry.py": '''"""
遙測與監控相關 Schema

此模組定義了模組狀態、心跳、性能指標等監控相關的資料模型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..enums import ModuleName


class HeartbeatPayload(BaseModel):
    """心跳 Payload"""

    module: ModuleName
    worker_id: str
    capacity: int


class ModuleStatus(BaseModel):
    """模組狀態報告 - 用於模組健康檢查和監控"""

    module: ModuleName
    status: str  # "running", "stopped", "error", "initializing"
    worker_id: str
    worker_count: int = 1
    queue_size: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: float = 0.0

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"running", "stopped", "error", "initializing", "degraded"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class FunctionTelemetry(BaseModel):
    """功能模組遙測數據基礎類"""

    payloads_sent: int = 0
    detections: int = 0
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式"""
        details: dict[str, Any] = {
            "payloads_sent": self.payloads_sent,
            "detections": self.detections,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
        }
        if findings_count is not None:
            details["findings"] = findings_count
        if self.errors:
            details["errors"] = self.errors
        return details


class FunctionExecutionResult(BaseModel):
    """功能模組執行結果統一格式"""

    findings: list[dict[str, Any]]
    telemetry: dict[str, Any]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OastEvent(BaseModel):
    """OAST (Out-of-Band Application Security Testing) 事件數據合約"""

    event_id: str
    probe_token: str
    event_type: str  # "http", "dns", "smtp", "ftp"
    source_ip: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    protocol: str | None = None
    raw_request: str | None = None
    raw_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    def validate_event_type(cls, v: str) -> str:
        allowed = {"http", "dns", "smtp", "ftp", "ldap", "other"}
        if v not in allowed:
            raise ValueError(f"Invalid event_type: {v}. Must be one of {allowed}")
        return v


class OastProbe(BaseModel):
    """OAST 探針數據合約"""

    probe_id: str
    token: str
    callback_url: str
    task_id: str
    scan_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    status: str = "active"

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"active", "triggered", "expired", "cancelled"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class SIEMEventPayload(BaseModel):
    """SIEM 事件 Payload"""

    event_id: str
    event_type: str
    severity: str
    source: str
    destination: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NotificationPayload(BaseModel):
    """通知 Payload - 用於 Slack/Teams/Email"""

    notification_id: str
    notification_type: str  # "slack", "teams", "email", "webhook"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    recipients: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
''',

    "ai.py": '''"""
AI 相關 Schema

此模組定義了所有 AI 驅動功能相關的資料模型，包括訓練、強化學習、
攻擊計劃、執行追蹤等。

注意: 此檔案整合了原 ai_schemas.py 的內容
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..enums import RiskLevel, Severity, TestStatus, VulnerabilityType


# ==================== CVSS v3.1 標準 ====================


class CVSSv3Metrics(BaseModel):
    """CVSS v3.1 標準指標"""

    model_config = {"str_strip_whitespace": True}

    # Base Metrics (Required)
    attack_vector: Literal["N", "A", "L", "P"] = Field(description="攻擊向量")
    attack_complexity: Literal["L", "H"] = Field(description="攻擊複雜度")
    privileges_required: Literal["N", "L", "H"] = Field(description="所需權限")
    user_interaction: Literal["N", "R"] = Field(description="用戶交互")
    scope: Literal["U", "C"] = Field(description="範圍")
    confidentiality: Literal["N", "L", "H"] = Field(description="機密性影響")
    integrity: Literal["N", "L", "H"] = Field(description="完整性影響")
    availability: Literal["N", "L", "H"] = Field(description="可用性影響")

    # Temporal Metrics (Optional)
    exploit_code_maturity: Literal["X", "H", "F", "P", "U"] = Field(
        default="X", description="漏洞利用代碼成熟度"
    )
    remediation_level: Literal["X", "U", "W", "T", "O"] = Field(default="X", description="修復級別")
    report_confidence: Literal["X", "C", "R", "U"] = Field(default="X", description="報告置信度")

    # Environmental Metrics (Optional)
    confidentiality_requirement: Literal["X", "L", "M", "H"] = Field(
        default="X", description="機密性要求"
    )
    integrity_requirement: Literal["X", "L", "M", "H"] = Field(
        default="X", description="完整性要求"
    )
    availability_requirement: Literal["X", "L", "M", "H"] = Field(
        default="X", description="可用性要求"
    )

    # Calculated Scores
    base_score: float | None = Field(default=None, ge=0.0, le=10.0, description="基本分數")
    temporal_score: float | None = Field(default=None, ge=0.0, le=10.0, description="時間分數")
    environmental_score: float | None = Field(default=None, ge=0.0, le=10.0, description="環境分數")
    vector_string: str | None = Field(default=None, description="CVSS 向量字符串")

    def calculate_base_score(self) -> float:
        """計算 CVSS v3.1 基本分數"""
        av_weights = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
        ac_weights = {"L": 0.77, "H": 0.44}
        if self.scope == "C":
            pr_weights = {"N": 0.85, "L": 0.68, "H": 0.50}
        else:
            pr_weights = {"N": 0.85, "L": 0.62, "H": 0.27}
        ui_weights = {"N": 0.85, "R": 0.62}
        cia_weights = {"N": 0.0, "L": 0.22, "H": 0.56}

        impact = 1 - (1 - cia_weights[self.confidentiality]) * (1 - cia_weights[self.integrity]) * (
            1 - cia_weights[self.availability]
        )

        if self.scope == "C":
            impact_adjusted = 7.52 * (impact - 0.029) - 3.25 * pow(impact - 0.02, 15)
        else:
            impact_adjusted = 6.42 * impact

        exploitability = (
            8.22
            * av_weights[self.attack_vector]
            * ac_weights[self.attack_complexity]
            * pr_weights[self.privileges_required]
            * ui_weights[self.user_interaction]
        )

        if impact_adjusted <= 0:
            return 0.0
        elif self.scope == "U":
            base = impact_adjusted + exploitability
        else:
            base = 1.08 * (impact_adjusted + exploitability)

        return min(10.0, round(base, 1))


# ==================== 攻擊計劃和步驟 ====================


class AttackStep(BaseModel):
    """攻擊步驟 (整合 MITRE ATT&CK)"""

    step_id: str
    action: str
    tool_type: str
    target: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_result: str | None = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    mitre_technique_id: str | None = Field(
        default=None,
        pattern=r"^T\\d{4}(\\.\\d{3})?$",
    )
    mitre_tactic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackPlan(BaseModel):
    """攻擊計畫"""

    plan_id: str
    scan_id: str
    attack_type: VulnerabilityType
    steps: list[AttackStep]
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    target_info: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "ai_planner"
    mitre_techniques: list[str] = Field(default_factory=list)
    mitre_tactics: list[str] = Field(default_factory=list)
    capec_id: str | None = Field(default=None, pattern=r"^CAPEC-\\d+$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        if not v.startswith("plan_"):
            raise ValueError("plan_id must start with 'plan_'")
        return v


class TraceRecord(BaseModel):
    """執行追蹤記錄"""

    trace_id: str
    plan_id: str
    step_id: str
    session_id: str
    tool_name: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    status: str
    error_message: str | None = None
    execution_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    environment_response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"success", "failed", "timeout", "skipped", "error"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class PlanExecutionMetrics(BaseModel):
    """計畫執行指標"""

    plan_id: str
    session_id: str
    expected_steps: int
    executed_steps: int
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    extra_actions: int
    completion_rate: float
    success_rate: float
    sequence_accuracy: float
    goal_achieved: bool
    reward_score: float
    total_execution_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlanExecutionResult(BaseModel):
    """計畫執行結果"""

    result_id: str
    plan_id: str
    session_id: str
    plan: AttackPlan
    trace: list[TraceRecord]
    metrics: PlanExecutionMetrics
    findings: list[dict[str, Any]] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    status: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"completed", "partial", "failed", "aborted"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


# ==================== AI 訓練相關 ====================


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    config_id: str
    model_type: str
    training_mode: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 3
    reward_function: str = "completion_rate"
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AITrainingStartPayload(BaseModel):
    """AI 訓練啟動請求"""

    training_id: str
    training_type: str
    scenario_id: str | None = None
    target_vulnerability: str | None = None
    config: ModelTrainingConfig
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("training_id")
    def validate_training_id(cls, v: str) -> str:
        if not v.startswith("training_"):
            raise ValueError("training_id must start with 'training_'")
        return v


class AITrainingProgressPayload(BaseModel):
    """AI 訓練進度報告"""

    training_id: str
    episode_number: int
    total_episodes: int
    successful_episodes: int = 0
    failed_episodes: int = 0
    total_samples: int = 0
    high_quality_samples: int = 0
    avg_reward: float | None = None
    avg_quality: float | None = None
    best_reward: float | None = None
    model_metrics: dict[str, float] = Field(default_factory=dict)
    status: str = "running"
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RAGKnowledgeUpdatePayload(BaseModel):
    """RAG 知識庫更新請求"""

    knowledge_type: str
    content: str
    source_id: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    related_cve: str | None = None
    related_cwe: str | None = None
    mitre_techniques: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryPayload(BaseModel):
    """RAG 查詢請求"""

    query_id: str
    query_text: str
    top_k: int = Field(default=5, ge=1, le=100)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    knowledge_types: list[str] | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
''',

    "assets.py": '''"""
資產與 EASM 相關 Schema

此模組定義了資產探索、資產生命週期管理、EASM 等相關的資料模型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ..enums import (
    AssetExposure,
    AssetType,
    BusinessCriticality,
    ComplianceFramework,
    Confidence,
    DataSensitivity,
    Environment,
    Exploitability,
    Severity,
    VulnerabilityStatus,
    VulnerabilityType,
)


class AssetLifecyclePayload(BaseModel):
    """資產生命週期管理 Payload"""

    asset_id: str
    asset_type: AssetType
    value: str
    environment: Environment
    business_criticality: BusinessCriticality
    data_sensitivity: DataSensitivity | None = None
    asset_exposure: AssetExposure | None = None
    owner: str | None = None
    team: str | None = None
    compliance_tags: list[ComplianceFramework] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VulnerabilityLifecyclePayload(BaseModel):
    """漏洞生命週期管理 Payload"""

    vulnerability_id: str
    finding_id: str
    asset_id: str
    vulnerability_type: VulnerabilityType
    severity: Severity
    confidence: Confidence
    status: VulnerabilityStatus
    exploitability: Exploitability | None = None
    assigned_to: str | None = None
    due_date: datetime | None = None
    first_detected: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolution_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VulnerabilityUpdatePayload(BaseModel):
    """漏洞狀態更新 Payload"""

    vulnerability_id: str
    status: VulnerabilityStatus
    assigned_to: str | None = None
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_by: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DiscoveredAsset(BaseModel):
    """探索到的資產"""

    asset_id: str
    asset_type: AssetType
    value: str
    discovery_method: str
    confidence: Confidence
    metadata: dict[str, Any] = Field(default_factory=dict)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
''',

    "risk.py": '''"""
風險評估與攻擊路徑分析 Schema

此模組定義了風險評估、攻擊路徑分析等相關的資料模型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ..enums import (
    AssetExposure,
    AttackPathEdgeType,
    AttackPathNodeType,
    BusinessCriticality,
    ComplianceFramework,
    DataSensitivity,
    Environment,
    RiskLevel,
)


class RiskAssessmentContext(BaseModel):
    """風險評估上下文"""

    environment: Environment
    business_criticality: BusinessCriticality
    data_sensitivity: DataSensitivity | None = None
    asset_exposure: AssetExposure | None = None
    compliance_tags: list[ComplianceFramework] = Field(default_factory=list)
    asset_value: float | None = None
    user_base: int | None = None
    sla_hours: int | None = None


class RiskAssessmentResult(BaseModel):
    """風險評估結果"""

    finding_id: str
    technical_risk_score: float
    business_risk_score: float
    risk_level: RiskLevel
    priority_score: float
    context_multiplier: float
    business_impact: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    estimated_effort: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskTrendAnalysis(BaseModel):
    """風險趨勢分析"""

    period_start: datetime
    period_end: datetime
    total_vulnerabilities: int
    risk_distribution: dict[str, int]
    average_risk_score: float
    trend: str
    improvement_percentage: float | None = None
    top_risks: list[dict[str, Any]] = Field(default_factory=list)


class AttackPathNode(BaseModel):
    """攻擊路徑節點"""

    node_id: str
    node_type: AttackPathNodeType
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class AttackPathEdge(BaseModel):
    """攻擊路徑邊"""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: AttackPathEdgeType
    risk_score: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class AttackPathPayload(BaseModel):
    """攻擊路徑 Payload"""

    path_id: str
    scan_id: str
    source_node: AttackPathNode
    target_node: AttackPathNode
    nodes: list[AttackPathNode]
    edges: list[AttackPathEdge]
    total_risk_score: float
    path_length: int
    description: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AttackPathRecommendation(BaseModel):
    """攻擊路徑推薦"""

    path_id: str
    risk_level: RiskLevel
    priority_score: float
    executive_summary: str
    technical_explanation: str
    business_impact: str
    remediation_steps: list[str]
    quick_wins: list[str] = Field(default_factory=list)
    affected_assets: list[str] = Field(default_factory=list)
    estimated_effort: str
    estimated_risk_reduction: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
''',

    "api_testing.py": '''"""
API 安全測試相關 Schema

此模組定義了 API 安全測試相關的資料模型。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# API 測試相關的 Schema 已在 tasks.py 中定義
# 此檔案保留作為擴展使用

__all__ = []
''',
}

def create_file_safe(file_path: Path, content: str):
    """安全地創建檔案"""
    try:
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ 成功創建: {file_path.name}")
        return True
    except Exception as e:
        print(f"❌ 創建失敗 {file_path.name}: {e}")
        return False

def main():
    """主函數"""
    print(f"📁 目標目錄: {SCHEMAS_DIR}")
    print(f"📝 準備創建 {len(REMAINING_FILES)} 個檔案\n")
    
    success_count = 0
    for filename, content in REMAINING_FILES.items():
        file_path = SCHEMAS_DIR / filename
        if create_file_safe(file_path, content):
            success_count += 1
    
    print(f"\n✨ 完成! 成功創建 {success_count}/{len(REMAINING_FILES)} 個檔案")
    return success_count == len(REMAINING_FILES)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
