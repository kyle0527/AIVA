"""
SQLi 檢測結果模型
統一的檢測結果和錯誤模型，與實際引擎使用保持一致
"""

from __future__ import annotations

from dataclasses import dataclass

from services.aiva_common.schemas import (
    FindingEvidence,
    FindingImpact,
    FindingRecommendation,
    FindingTarget,
    Vulnerability,
)


@dataclass
class DetectionResult:
    """SQLi檢測結果 - 與引擎實際使用保持一致的完整模型"""

    is_vulnerable: bool
    vulnerability: Vulnerability
    evidence: FindingEvidence
    impact: FindingImpact
    recommendation: FindingRecommendation
    target: FindingTarget
    detection_method: str
    payload_used: str
    confidence_score: float

    # 向後兼容的字段（如果需要）
    db_fingerprint: str | None = None
    response_time: float = 0.0


@dataclass
class DetectionError:
    """檢測過程中的錯誤"""

    payload: str
    vector: str
    message: str
    attempts: int = 1
    engine_name: str = ""

    def __str__(self) -> str:
        return f"[{self.engine_name}] {self.message} (payload: {self.payload[:50]}...)"
