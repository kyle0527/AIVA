from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from queue import PriorityQueue
from typing import Any

from services.aiva_common.enums import Topic
from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class TaskQueueManager:
    """
    任務佇列管理器

    負責管理功能模組的任務佇列、優先級排序、執行狀態追蹤等。
    實現智慧化的任務調度與負載平衡。
    """

    def __init__(self) -> None:
        # 按掃描ID分組的任務佇列
        self._task_queues: dict[str, PriorityQueue] = defaultdict(PriorityQueue)
        # 任務執行狀態
        self._task_status: dict[str, dict[str, Any]] = {}
        # 掃描任務統計
        self._scan_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "total": 0,
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
            }
        )

    def enqueue_task(self, topic: Topic, task_payload: FunctionTaskPayload) -> None:
        """
        將任務加入佇列

        Args:
            topic: 任務主題
            task_payload: 任務負載
        """
        scan_id = task_payload.scan_id
        task_id = task_payload.task_id
        priority = task_payload.priority

        # 使用負優先級值實現高優先級優先
        queue_item = (
            -priority,
            datetime.now(UTC).timestamp(),
            topic,
            task_payload,
        )

        self._task_queues[scan_id].put(queue_item)

        # 更新任務狀態
        self._task_status[task_id] = {
            "scan_id": scan_id,
            "topic": topic,
            "status": "pending",
            "priority": priority,
            "created_at": datetime.now(UTC).isoformat(),
            "payload": task_payload,
        }

        # 更新統計
        self._scan_stats[scan_id]["total"] += 1
        self._scan_stats[scan_id]["pending"] += 1

        logger.info(
            f"Enqueued task {task_id} for scan {scan_id} with priority {priority}"
        )

    def get_pending_tasks(self, scan_id: str) -> list[dict[str, Any]]:
        """
        獲取待執行任務列表

        Args:
            scan_id: 掃描ID

        Returns:
            待執行任務列表
        """
        pending_tasks = []

        for task_id, task_info in self._task_status.items():
            if task_info["scan_id"] == scan_id and task_info["status"] == "pending":
                pending_tasks.append(
                    {
                        "task_id": task_id,
                        "topic": task_info["topic"],
                        "priority": task_info["priority"],
                        "created_at": task_info["created_at"],
                    }
                )

        # 按優先級排序
        pending_tasks.sort(key=lambda x: x["priority"], reverse=True)
        return pending_tasks

    def mark_task_running(self, task_id: str) -> None:
        """標記任務為執行中"""
        if task_id in self._task_status:
            task_info = self._task_status[task_id]
            old_status = task_info["status"]
            task_info["status"] = "running"
            task_info["started_at"] = datetime.now(UTC).isoformat()

            scan_id = task_info["scan_id"]
            if old_status == "pending":
                self._scan_stats[scan_id]["pending"] -= 1
                self._scan_stats[scan_id]["running"] += 1

            logger.info(f"Task {task_id} is now running")

    def mark_task_completed(
        self, task_id: str, result: dict[str, Any] | None = None
    ) -> None:
        """標記任務為完成"""
        if task_id in self._task_status:
            task_info = self._task_status[task_id]
            old_status = task_info["status"]
            task_info["status"] = "completed"
            task_info["completed_at"] = datetime.now(UTC).isoformat()

            if result:
                task_info["result"] = result

            scan_id = task_info["scan_id"]
            if old_status == "running":
                self._scan_stats[scan_id]["running"] -= 1
                self._scan_stats[scan_id]["completed"] += 1

            logger.info(f"Task {task_id} completed")

    def mark_task_failed(self, task_id: str, error: str) -> None:
        """標記任務為失敗"""
        if task_id in self._task_status:
            task_info = self._task_status[task_id]
            old_status = task_info["status"]
            task_info["status"] = "failed"
            task_info["failed_at"] = datetime.now(UTC).isoformat()
            task_info["error"] = error

            scan_id = task_info["scan_id"]
            if old_status in ["pending", "running"]:
                self._scan_stats[scan_id][old_status] -= 1
                self._scan_stats[scan_id]["failed"] += 1

            logger.warning(f"Task {task_id} failed: {error}")

    def get_scan_progress(self, scan_id: str) -> dict[str, Any]:
        """
        獲取掃描進度

        Args:
            scan_id: 掃描ID

        Returns:
            進度資訊
        """
        stats = self._scan_stats[scan_id]
        total = stats["total"]
        completed = stats["completed"]
        failed = stats["failed"]

        return {
            "scan_id": scan_id,
            "total_tasks": total,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "pending_tasks": stats["pending"],
            "running_tasks": stats["running"],
            "success_rate": completed / max(total, 1) * 100,
            "progress_percentage": (completed + failed) / max(total, 1) * 100,
        }

    def update_task_queue(
        self, scan_id: str, updated_tasks: list[dict[str, Any]]
    ) -> None:
        """
        更新任務佇列

        Args:
            scan_id: 掃描ID
            updated_tasks: 更新的任務列表
        """
        # 暫時實作：記錄更新請求
        task_count = len(updated_tasks)
        logger.info(
            f"Task queue update requested for scan {scan_id} with {task_count} tasks"
        )

    def get_queue_status(self) -> dict[str, Any]:
        """獲取佇列整體狀態"""
        total_scans = len(self._scan_stats)
        total_tasks = sum(stats["total"] for stats in self._scan_stats.values())
        pending_tasks = sum(stats["pending"] for stats in self._scan_stats.values())
        running_tasks = sum(stats["running"] for stats in self._scan_stats.values())

        return {
            "total_scans": total_scans,
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "running_tasks": running_tasks,
            "queue_health": "healthy" if pending_tasks < 1000 else "overloaded",
        }
