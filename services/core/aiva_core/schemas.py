"""Core 模組專用數據合約

⚠️ DEPRECATION WARNING - V1 架構 ⚠️
此文件為 V1 架構的 schema 定義，正在逐步遷移到 V2 統一架構。

V2 單一事實來源 (Single Source of Truth):
- 檔案位置: services/aiva_common/core_schema_sot.yaml
- 說明文件: services/aiva_common/README_SCHEMA.md
- 生成工具: tools/schema_codegen_tool.py

未來此文件將被廢棄，請優先使用 V2 架構的 schema。
新功能開發請使用 core_schema_sot.yaml 定義 schema。

定義核心引擎的內部數據結構，包括：
- 攻擊面分析結果
- 測試策略
- 任務定義
- 策略調整記錄
"""

import warnings
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 發出運行時警告，提醒開發者遷移到 V2
warnings.warn(
    "services/core/aiva_core/schemas.py is deprecated (V1 architecture). "
    "Please migrate to V2 unified schema: services/aiva_common/core_schema_sot.yaml",
    DeprecationWarning,
    stacklevel=2,
)

# ============================================================================
# 攻擊面分析相關
# ============================================================================


class AssetAnalysis(BaseModel):
    """資產分析結果"""

    asset_id: str = Field(description="資產唯一識別碼")
    url: str = Field(description="資產 URL")
    asset_type: str = Field(
        description="資產類型", examples=["endpoint", "form", "api"]
    )
    risk_score: int = Field(ge=0, le=100, description="風險評分 (0-100)")
    categories: list[str] = Field(default_factory=list, description="資產分類標籤")
    parameters: list[str] = Field(default_factory=list, description="可測試的參數列表")
    has_form: bool = Field(default=False, description="是否包含表單")
    method: str = Field(default="GET", description="HTTP 方法")

    @field_validator("risk_score")
    @classmethod
    def validate_risk_score(cls, v: int) -> int:
        """驗證風險評分範圍"""
        if not 0 <= v <= 100:
            raise ValueError("Risk score must be between 0 and 100")
        return v


class VulnerabilityCandidate(BaseModel):
    """漏洞候選基類"""

    asset_url: str = Field(description="目標資產 URL")
    parameter: str | None = Field(default=None, description="測試參數名稱")
    location: Literal["query", "body", "header", "path", "cookie"] = Field(
        default="query", description="參數位置"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 (0.0-1.0)")
    reasons: list[str] = Field(default_factory=list, description="判斷為候選的原因列表")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """驗證置信度範圍"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class XssCandidate(VulnerabilityCandidate):
    """XSS 漏洞候選"""

    xss_type: Literal["reflected", "stored", "dom"] = Field(
        default="reflected", description="XSS 類型"
    )
    context: str | None = Field(
        default=None,
        description="參數出現的上下文",
        examples=["html", "javascript", "attribute"],
    )


class SqliCandidate(VulnerabilityCandidate):
    """SQLi 漏洞候選"""

    database_hints: list[str] = Field(
        default_factory=list, description="數據庫類型提示"
    )
    error_based_possible: bool = Field(
        default=False, description="是否可能進行錯誤注入"
    )


class SsrfCandidate(VulnerabilityCandidate):
    """SSRF 漏洞候選"""

    target_type: Literal["url_parameter", "redirect", "webhook", "import"] = Field(
        default="url_parameter", description="SSRF 目標類型"
    )
    protocols: list[str] = Field(
        default_factory=lambda: ["http", "https"], description="支援的協議"
    )


class IdorCandidate(VulnerabilityCandidate):
    """IDOR 漏洞候選"""

    resource_type: str | None = Field(
        default=None, description="資源類型", examples=["user", "document", "order"]
    )
    id_pattern: str | None = Field(
        default=None, description="ID 模式", examples=["numeric", "uuid", "hash"]
    )
    requires_auth: bool = Field(default=True, description="是否需要認證")


class AttackSurfaceAnalysis(BaseModel):
    """完整的攻擊面分析結果"""

    scan_id: str = Field(description="掃描任務 ID")
    total_assets: int = Field(ge=0, description="總資產數量")
    forms: int = Field(default=0, ge=0, description="表單數量")
    parameters: int = Field(default=0, ge=0, description="參數總數")
    waf_detected: bool = Field(default=False, description="是否檢測到 WAF")

    # 資產風險分層
    high_risk_assets: list[AssetAnalysis] = Field(
        default_factory=list, description="高風險資產 (score >= 70)"
    )
    medium_risk_assets: list[AssetAnalysis] = Field(
        default_factory=list, description="中風險資產 (40 <= score < 70)"
    )
    low_risk_assets: list[AssetAnalysis] = Field(
        default_factory=list, description="低風險資產 (score < 40)"
    )

    # 漏洞候選
    xss_candidates: list[XssCandidate] = Field(
        default_factory=list, description="XSS 漏洞候選列表"
    )
    sqli_candidates: list[SqliCandidate] = Field(
        default_factory=list, description="SQLi 漏洞候選列表"
    )
    ssrf_candidates: list[SsrfCandidate] = Field(
        default_factory=list, description="SSRF 漏洞候選列表"
    )
    idor_candidates: list[IdorCandidate] = Field(
        default_factory=list, description="IDOR 漏洞候選列表"
    )

    @property
    def total_candidates(self) -> int:
        """總候選數量"""
        return (
            len(self.xss_candidates)
            + len(self.sqli_candidates)
            + len(self.ssrf_candidates)
            + len(self.idor_candidates)
        )


# ============================================================================
# 測試策略相關
# ============================================================================


class TestTask(BaseModel):
    """單個測試任務定義"""

    vulnerability_type: Literal["xss", "sqli", "ssrf", "idor"] = Field(
        description="漏洞類型"
    )
    asset: str = Field(description="目標資產 URL")
    parameter: str | None = Field(default=None, description="測試參數")
    method: str = Field(default="GET", description="HTTP 方法")
    location: Literal["query", "body", "header", "path", "cookie"] = Field(
        default="query", description="參數位置"
    )
    priority: int = Field(ge=1, le=10, default=5, description="優先級 (1-10)")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.5, description="成功可能性 (0.0-1.0)"
    )
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="額外元數據"
    )


class TestStrategy(BaseModel):
    """完整的測試策略"""

    scan_id: str = Field(description="掃描任務 ID")
    strategy_type: Literal["comprehensive", "targeted", "quick", "custom"] = Field(
        default="comprehensive", description="策略類型"
    )

    # 各類型任務列表
    xss_tasks: list[TestTask] = Field(default_factory=list, description="XSS 測試任務")
    sqli_tasks: list[TestTask] = Field(
        default_factory=list, description="SQLi 測試任務"
    )
    ssrf_tasks: list[TestTask] = Field(
        default_factory=list, description="SSRF 測試任務"
    )
    idor_tasks: list[TestTask] = Field(
        default_factory=list, description="IDOR 測試任務"
    )

    # 策略元數據
    estimated_duration_seconds: int = Field(
        default=0, ge=0, description="預估執行時間（秒）"
    )
    max_concurrent_tasks: int = Field(
        default=5, ge=1, le=50, description="最大並發任務數"
    )

    @property
    def total_tasks(self) -> int:
        """總任務數量"""
        return (
            len(self.xss_tasks)
            + len(self.sqli_tasks)
            + len(self.ssrf_tasks)
            + len(self.idor_tasks)
        )

    @property
    def all_tasks(self) -> list[TestTask]:
        """獲取所有任務的平面列表"""
        return self.xss_tasks + self.sqli_tasks + self.ssrf_tasks + self.idor_tasks


# ============================================================================
# 策略調整相關
# ============================================================================


class StrategyAdjustment(BaseModel):
    """策略調整記錄"""

    adjustment_type: Literal["waf", "success_rate", "tech_stack", "findings"] = Field(
        description="調整類型"
    )
    applied_rules: list[str] = Field(
        default_factory=list, description="已應用的調整規則"
    )
    priority_changes: dict[str, int] = Field(
        default_factory=dict, description="優先級變更記錄 {task_id: new_priority}"
    )
    timing_adjustments: dict[str, float] = Field(
        default_factory=dict, description="時間調整記錄 {task_id: delay_multiplier}"
    )
    reason: str = Field(default="", description="調整原因說明")


class LearningFeedback(BaseModel):
    """學習反饋數據"""

    scan_id: str = Field(description="掃描任務 ID")
    task_id: str = Field(description="任務 ID")
    module: Literal["xss", "sqli", "ssrf", "idor"] = Field(description="模組名稱")
    success: bool = Field(description="是否成功檢測到漏洞")
    vulnerability_type: str = Field(description="漏洞類型")
    confidence: float = Field(ge=0.0, le=1.0, description="檢測置信度")
    execution_time_seconds: float = Field(default=0.0, ge=0.0, description="執行時間")
    waf_detected: bool = Field(default=False, description="是否遇到 WAF")
    error_occurred: bool = Field(default=False, description="是否發生錯誤")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="額外反饋數據"
    )


# ============================================================================
# 配置相關
# ============================================================================


class StrategyGenerationConfig(BaseModel):
    """策略生成配置"""

    # 任務生成限制
    max_tasks_per_scan: int = Field(default=1000, ge=1, le=10000)
    max_tasks_per_asset: int = Field(default=20, ge=1, le=100)
    max_tasks_per_parameter: int = Field(default=4, ge=1, le=10)

    # 優先級配置
    high_risk_priority: int = Field(default=8, ge=1, le=10)
    medium_risk_priority: int = Field(default=5, ge=1, le=10)
    low_risk_priority: int = Field(default=3, ge=1, le=10)

    # 置信度閾值
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # 時間預估（秒）
    avg_xss_task_duration: int = Field(default=30, ge=1)
    avg_sqli_task_duration: int = Field(default=45, ge=1)
    avg_ssrf_task_duration: int = Field(default=60, ge=1)
    avg_idor_task_duration: int = Field(default=20, ge=1)
