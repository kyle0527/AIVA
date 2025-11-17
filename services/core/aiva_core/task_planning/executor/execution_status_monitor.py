import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """執行上下文 - 追蹤任務執行的環境信息"""
    
    session_id: str
    task_id: str
    start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "start_time": self.start_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionContext":
        """從字典創建"""
        return cls(
            session_id=data["session_id"],
            task_id=data["task_id"],
            start_time=data.get("start_time", datetime.now(UTC).isoformat()),
            metadata=data.get("metadata", {})
        )


class ExecutionMonitor:
    """執行監控器的別名類，保持向後兼容"""
    pass


class ExecutionStatusMonitor:
    """執行狀態監控器

    監控任務執行狀態、Worker心跳檢測、SLA追蹤等，
    提供系統健康狀態評估與異常警報。
    """

    def __init__(self) -> None:
        self._worker_heartbeats: dict[str, datetime] = {}
        self._task_metrics: dict[str, dict[str, Any]] = {}
        self._system_metrics: dict[str, Any] = {
            "start_time": datetime.now(UTC),
            "total_tasks_processed": 0,
            "failed_tasks": 0,
            "avg_task_duration": 0.0,
            "active_workers": 0,
        }
        self._alerts: list[dict[str, Any]] = []

    def record_worker_heartbeat(self, worker_id: str, status: str = "healthy") -> None:
        """記錄Worker心跳

        Args:
            worker_id: Worker標識符
            status: Worker狀態
        """
        self._worker_heartbeats[worker_id] = datetime.now(UTC)
        logger.debug(f"Heartbeat received from worker {worker_id}: {status}")

    def record_task_start(self, task_id: str, worker_id: str) -> None:
        """記錄任務開始執行

        Args:
            task_id: 任務ID
            worker_id: 執行Worker ID
        """
        self._task_metrics[task_id] = {
            "worker_id": worker_id,
            "start_time": datetime.now(UTC),
            "status": "running",
        }
        logger.info(f"Task {task_id} started on worker {worker_id}")

    def record_task_completion(
        self, task_id: str, success: bool, duration_seconds: float | None = None
    ) -> None:
        """記錄任務完成

        Args:
            task_id: 任務ID
            success: 是否成功
            duration_seconds: 執行時長（秒）
        """
        if task_id not in self._task_metrics:
            return

        task_metric = self._task_metrics[task_id]
        end_time = datetime.now(UTC)

        if not duration_seconds:
            duration_seconds = (end_time - task_metric["start_time"]).total_seconds()

        task_metric.update(
            {
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "success": success,
                "status": "completed" if success else "failed",
            }
        )

        # 更新系統指標
        self._system_metrics["total_tasks_processed"] += 1
        if not success:
            self._system_metrics["failed_tasks"] += 1

        # 更新平均執行時間
        total_tasks = self._system_metrics["total_tasks_processed"]
        old_avg = self._system_metrics["avg_task_duration"]
        self._system_metrics["avg_task_duration"] = (
            old_avg * (total_tasks - 1) + duration_seconds
        ) / total_tasks

        status_text = "success" if success else "failed"
        logger.info(
            f"Task {task_id} completed: {status_text} in {duration_seconds:.2f}s"
        )

    def get_system_health(self) -> dict[str, Any]:
        """獲取系統健康狀態

        Returns:
            系統健康狀態資訊
        """
        now = datetime.now(UTC)

        # 檢查Worker健康狀態
        active_workers = 0
        unhealthy_workers = []

        for worker_id, last_heartbeat in self._worker_heartbeats.items():
            if (now - last_heartbeat).total_seconds() < 120:  # 2分鐘內有心跳
                active_workers += 1
            else:
                unhealthy_workers.append(
                    {
                        "worker_id": worker_id,
                        "last_heartbeat": last_heartbeat.isoformat(),
                        "offline_duration": (now - last_heartbeat).total_seconds(),
                    }
                )

        self._system_metrics["active_workers"] = active_workers

        # 計算失敗率
        total_tasks = self._system_metrics["total_tasks_processed"]
        failed_tasks = self._system_metrics["failed_tasks"]
        failure_rate = (failed_tasks / max(total_tasks, 1)) * 100

        # 判斷系統健康狀態
        if failure_rate > 50:
            status = "critical"
        elif failure_rate > 20 or active_workers == 0:
            status = "warning"
        elif len(unhealthy_workers) > 0:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "timestamp": now.isoformat(),
            "uptime_seconds": (
                now - self._system_metrics["start_time"]
            ).total_seconds(),
            "active_workers": active_workers,
            "unhealthy_workers": unhealthy_workers,
            "total_tasks_processed": total_tasks,
            "failed_tasks": failed_tasks,
            "failure_rate_percent": failure_rate,
            "avg_task_duration_seconds": self._system_metrics["avg_task_duration"],
            "alerts": self._get_recent_alerts(),
        }

    def check_sla_violations(self) -> list[dict[str, Any]]:
        """檢查SLA違規

        Returns:
            SLA違規列表
        """
        violations = []
        now = datetime.now(UTC)

        for task_id, metric in self._task_metrics.items():
            if metric["status"] == "running":
                runtime = (now - metric["start_time"]).total_seconds()

                # 任務執行時間超過10分鐘視為SLA違規
                if runtime > 600:
                    violations.append(
                        {
                            "task_id": task_id,
                            "worker_id": metric["worker_id"],
                            "runtime_seconds": runtime,
                            "violation_type": "execution_timeout",
                        }
                    )

        return violations

    def _get_recent_alerts(self) -> list[dict[str, Any]]:
        """獲取最近的警報"""
        # 返回最近1小時的警報
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        recent_alerts = [
            alert
            for alert in self._alerts
            if datetime.fromisoformat(alert["timestamp"].replace("Z", "+00:00"))
            > one_hour_ago
        ]
        return recent_alerts[-10:]  # 最多返回10個

    def add_alert(
        self, level: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """新增警報

        Args:
            level: 警報級別 (info, warning, error, critical)
            message: 警報訊息
            details: 詳細資訊
        """
        alert = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "details": details or {},
        }

        self._alerts.append(alert)

        # 保持最近1000個警報
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        logger.warning(f"Alert [{level.upper()}]: {message}")

    async def start_monitoring(self) -> None:
        """開始監控循環"""
        logger.info("Starting execution status monitoring...")

        while True:
            try:
                # 每30秒檢查一次
                await asyncio.sleep(30)

                # 檢查SLA違規
                violations = self.check_sla_violations()
                for violation in violations:
                    self.add_alert(
                        "warning",
                        f"Task {violation['task_id']} execution timeout",
                        violation,
                    )

                # 檢查Worker健康狀態
                health = self.get_system_health()
                if health["status"] in ["critical", "warning"]:
                    self.add_alert(
                        health["status"],
                        f"System health degraded: {health['status']}",
                        {"health_data": health},
                    )

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
