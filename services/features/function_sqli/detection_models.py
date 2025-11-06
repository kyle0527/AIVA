"""
SQLi 檢測結果模型
統一的檢測結果和錯誤模型，與實際引擎使用保持一致
"""



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


class DetectionModels:
    """SQLi 檢測模型集合類"""
    
    @staticmethod
    def create_detection_result(
        is_vulnerable: bool,
        vulnerability: Vulnerability,
        evidence: FindingEvidence,
        detection_method: str,
        payload_used: str,
        confidence_score: float = 0.8
    ) -> DetectionResult:
        """創建檢測結果"""
        return DetectionResult(
            is_vulnerable=is_vulnerable,
            vulnerability=vulnerability,
            evidence=evidence,
            impact=FindingImpact(description="SQL injection vulnerability detected"),
            recommendation=FindingRecommendation(
                description="Use parameterized queries to prevent SQL injection"
            ),
            target=FindingTarget(url="", parameter=""),
            detection_method=detection_method,
            payload_used=payload_used,
            confidence_score=confidence_score
        )
    
    @staticmethod
    def create_detection_error(
        payload: str,
        vector: str,
        message: str,
        engine_name: str = "SQLi"
    ) -> DetectionError:
        """創建檢測錯誤"""
        return DetectionError(
            payload=payload,
            vector=vector,
            message=message,
            engine_name=engine_name
        )
