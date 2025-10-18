"""
PostEx (Post-Exploitation) 模組專用數據合約
定義後滲透測試相關的所有數據結構，基於 Pydantic v2.12.0
僅用於授權的滲透測試環境
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from services.aiva_common.enums import PostExTestType, ThreatLevel
from services.aiva_common.schemas import (
    FindingPayload,
    FunctionTelemetry,
)


class PostExTestVector(BaseModel):
    """後滲透測試向量 - 官方 Pydantic BaseModel"""

    vector_id: str
    test_type: PostExTestType
    target_system: str
    payload: str
    technique_id: str  # MITRE ATT&CK Technique ID (如 T1055, T1078)
    tactic: str  # MITRE ATT&CK Tactic (如 "Privilege Escalation", "Lateral Movement")
    safe_mode: bool = True
    requires_authorization: bool = True
    risk_level: ThreatLevel = ThreatLevel.LOW
    timeout_seconds: int = Field(ge=1, le=300, default=30)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("test_type")
    def validate_test_type(cls, v: PostExTestType) -> PostExTestType:
        """驗證測試類型"""
        return v

    @field_validator("technique_id")
    def validate_technique_id(cls, v: str) -> str:
        """驗證 MITRE ATT&CK 技術 ID 格式"""
        import re

        if not re.match(r"^T\d{4}(\.\d{3})?$", v):
            raise ValueError(
                f"Invalid MITRE ATT&CK Technique ID: {v}. Expected format: T1234 or T1234.001"
            )
        return v

    @field_validator("tactic")
    def validate_tactic(cls, v: str) -> str:
        """驗證 MITRE ATT&CK 戰術名稱"""
        allowed_tactics = {
            "Initial Access",
            "Execution",
            "Persistence",
            "Privilege Escalation",
            "Defense Evasion",
            "Credential Access",
            "Discovery",
            "Lateral Movement",
            "Collection",
            "Command and Control",
            "Exfiltration",
            "Impact",
        }
        if v not in allowed_tactics:
            raise ValueError(f"Invalid MITRE ATT&CK tactic: {v}")
        return v


class PostExDetectionResult(BaseModel):
    """後滲透測試檢測結果 - 官方 Pydantic BaseModel"""

    test_vector: PostExTestVector
    execution_successful: bool = False
    privileges_escalated: bool = False
    lateral_movement_achieved: bool = False
    persistence_established: bool = False
    data_exfiltrated: bool = False
    detection_evaded: bool = False
    impact_severity: ThreatLevel = ThreatLevel.LOW
    execution_time_ms: float = Field(ge=0.0, default=0.0)
    output: str = ""
    error_message: str | None = None
    artifacts_created: list[str] = Field(default_factory=list)
    network_connections: list[str] = Field(default_factory=list)
    processes_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    registry_changes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    def calculate_risk_score(self) -> float:
        """計算風險評分 (0.0-1.0)"""
        score = 0.0

        if self.execution_successful:
            score += 0.2

        if self.privileges_escalated:
            score += 0.25

        if self.lateral_movement_achieved:
            score += 0.2

        if self.persistence_established:
            score += 0.15

        if self.data_exfiltrated:
            score += 0.3

        if self.detection_evaded:
            score += 0.1

        # 根據影響嚴重程度調整
        severity_multiplier = {
            ThreatLevel.LOW: 0.5,
            ThreatLevel.MEDIUM: 0.7,
            ThreatLevel.HIGH: 0.9,
            ThreatLevel.CRITICAL: 1.0,
        }

        return min(
            score
            * severity_multiplier.get(self.impact_severity, 0.5)
            * self.confidence,
            1.0,
        )

    def get_mitre_techniques(self) -> list[str]:
        """獲取涉及的 MITRE ATT&CK 技術"""
        techniques = [self.test_vector.technique_id]

        # 根據結果添加額外的技術 ID
        if self.privileges_escalated:
            techniques.extend(["T1055", "T1078"])  # Process Injection, Valid Accounts

        if self.lateral_movement_achieved:
            techniques.extend(
                ["T1021", "T1570"]
            )  # Remote Services, Lateral Tool Transfer

        if self.persistence_established:
            techniques.extend(
                ["T1053", "T1547"]
            )  # Scheduled Task, Boot or Logon Autostart

        if self.data_exfiltrated:
            techniques.extend(
                ["T1041", "T1567"]
            )  # Exfiltration Over C2, Exfiltration Over Web

        return list(set(techniques))  # 去重


class SystemFingerprint(BaseModel):
    """系統指紋信息 - 官方 Pydantic BaseModel"""

    os_type: str  # "windows", "linux", "macos"
    os_version: str
    architecture: str  # "x64", "x86", "arm64"
    hostname: str
    domain: str | None = None
    installed_software: list[str] = Field(default_factory=list)
    running_services: list[str] = Field(default_factory=list)
    network_interfaces: list[dict[str, str]] = Field(default_factory=list)
    security_products: list[str] = Field(default_factory=list)
    patch_level: str | None = None
    privileged_users: list[str] = Field(default_factory=list)

    @field_validator("os_type")
    def validate_os_type(cls, v: str) -> str:
        """驗證作業系統類型"""
        allowed = {"windows", "linux", "macos", "unix", "other"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid OS type: {v}")
        return v.lower()


class PostExTelemetry(FunctionTelemetry):
    """PostEx 專用遙測數據 - 繼承自 FunctionTelemetry"""

    tests_executed: int = 0
    successful_executions: int = 0
    privilege_escalations: int = 0
    lateral_movements: int = 0
    persistence_attempts: int = 0
    data_exfiltrations: int = 0
    detection_evasions: int = 0
    techniques_used: dict[str, int] = Field(
        default_factory=dict
    )  # technique_id -> count
    tactics_covered: list[str] = Field(default_factory=list)
    systems_compromised: int = 0
    artifacts_left: int = 0
    average_execution_time_ms: float = 0.0

    def record_test_result(self, result: PostExDetectionResult) -> None:
        """記錄測試結果"""
        self.tests_executed += 1

        if result.execution_successful:
            self.successful_executions += 1

        if result.privileges_escalated:
            self.privilege_escalations += 1

        if result.lateral_movement_achieved:
            self.lateral_movements += 1

        if result.persistence_established:
            self.persistence_attempts += 1

        if result.data_exfiltrated:
            self.data_exfiltrations += 1

        if result.detection_evaded:
            self.detection_evasions += 1

        # 記錄使用的技術
        technique_id = result.test_vector.technique_id
        self.techniques_used[technique_id] = (
            self.techniques_used.get(technique_id, 0) + 1
        )

        # 記錄涵蓋的戰術
        tactic = result.test_vector.tactic
        if tactic not in self.tactics_covered:
            self.tactics_covered.append(tactic)

        # 計算留下的痕跡
        self.artifacts_left += len(result.artifacts_created)

        # 更新平均執行時間
        if self.tests_executed == 1:
            self.average_execution_time_ms = result.execution_time_ms
        else:
            self.average_execution_time_ms = (
                self.average_execution_time_ms * (self.tests_executed - 1)
                + result.execution_time_ms
            ) / self.tests_executed

    def get_success_rate(self) -> float:
        """獲取成功率"""
        if self.tests_executed == 0:
            return 0.0
        return self.successful_executions / self.tests_executed

    def get_stealth_score(self) -> float:
        """獲取隱蔽性評分 (0.0-1.0, 1.0 = 完全隱蔽)"""
        if self.tests_executed == 0:
            return 0.0

        evasion_rate = self.detection_evasions / self.tests_executed
        artifacts_per_test = self.artifacts_left / self.tests_executed

        # 規避率高且留下痕跡少 = 隱蔽性高
        stealth_score = evasion_rate * (1.0 - min(artifacts_per_test / 10.0, 1.0))
        return stealth_score

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式"""
        details = super().to_details(findings_count)
        details.update(
            {
                "tests_executed": self.tests_executed,
                "successful_executions": self.successful_executions,
                "privilege_escalations": self.privilege_escalations,
                "lateral_movements": self.lateral_movements,
                "persistence_attempts": self.persistence_attempts,
                "data_exfiltrations": self.data_exfiltrations,
                "detection_evasions": self.detection_evasions,
                "success_rate": self.get_success_rate(),
                "stealth_score": self.get_stealth_score(),
                "techniques_used": self.techniques_used,
                "tactics_covered": len(self.tactics_covered),
                "systems_compromised": self.systems_compromised,
                "artifacts_left": self.artifacts_left,
                "average_execution_time_ms": round(self.average_execution_time_ms, 2),
            }
        )
        return details


class TaskExecutionResult(BaseModel):
    """PostEx 任務執行結果 - 官方 Pydantic BaseModel"""

    findings: list[FindingPayload]
    telemetry: PostExTelemetry
    system_fingerprint: SystemFingerprint | None = None
    authorization_verified: bool = False
    safe_mode_enabled: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# 導出所有公共類
__all__ = [
    "PostExTestVector",
    "PostExDetectionResult",
    "SystemFingerprint",
    "PostExTelemetry",
    "TaskExecutionResult",
]
