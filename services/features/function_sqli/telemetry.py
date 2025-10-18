"""
SQLi 執行遙測數據模型
分離遙測邏輯以避免循環導入
"""

from dataclasses import dataclass, field


@dataclass
class SqliExecutionTelemetry:
    """SQLi檢測執行過程的遙測數據"""

    payloads_sent: int = 0
    detections: int = 0
    errors: list[str] = field(default_factory=list)
    engines_run: list[str] = field(default_factory=list)

    def to_details(self, findings_count: int) -> dict[str, object]:
        """轉換為詳細的遙測報告格式"""
        details: dict[str, object] = {
            "findings": findings_count,
            "payloads_sent": self.payloads_sent,
            "detections": self.detections,
        }
        if self.engines_run:
            details["engines"] = list(dict.fromkeys(self.engines_run))
        if self.errors:
            details["errors"] = self.errors
        return details

    def record_engine_execution(self, engine_name: str) -> None:
        """記錄引擎執行"""
        self.engines_run.append(engine_name)

    def record_payload_sent(self) -> None:
        """記錄載荷發送"""
        self.payloads_sent += 1

    def record_detection(self) -> None:
        """記錄檢測結果"""
        self.detections += 1

    def record_error(self, error_message: str) -> None:
        """記錄錯誤訊息"""
        self.errors.append(error_message)

    # 向後兼容的別名方法
    def add_engine(self, engine_name: str) -> None:
        """記錄引擎執行（向後兼容別名）"""
        self.record_engine_execution(engine_name)

    def add_error(self, error_message: str) -> None:
        """記錄錯誤訊息（向後兼容別名）"""
        self.record_error(error_message)
