"""Attack Chain - 攻擊鏈編排器

管理和編排複雜的多步驟攻擊鏈
"""

from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChainStatus(str, Enum):
    """攻擊鏈狀態"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AttackChain:
    """攻擊鏈編排器

    管理複雜的多步驟攻擊序列，包括:
    - 依賴關係管理
    - 執行順序編排
    - 條件分支
    - 結果傳遞
    """

    def __init__(self, chain_id: str):
        """初始化攻擊鏈

        Args:
            chain_id: 攻擊鏈 ID
        """
        self.chain_id = chain_id
        self.steps: list[dict[str, Any]] = []
        self.dependencies: dict[str, list[str]] = {}
        self.results: dict[str, Any] = {}
        self.status = ChainStatus.PENDING

        logger.info(f"AttackChain created: {chain_id}")

    def add_step(
        self,
        step_id: str,
        action: str,
        parameters: dict[str, Any],
        depends_on: list[str] = None,
    ):
        """添加攻擊步驟

        Args:
            step_id: 步驟 ID
            action: 動作類型
            parameters: 參數
            depends_on: 依賴的步驟列表
        """
        step = {
            "step_id": step_id,
            "action": action,
            "parameters": parameters,
            "status": "pending",
        }

        self.steps.append(step)

        if depends_on:
            self.dependencies[step_id] = depends_on

        logger.debug(f"Step added: {step_id}, depends_on={depends_on}")

    def can_execute_step(self, step_id: str) -> bool:
        """檢查步驟是否可以執行

        Args:
            step_id: 步驟 ID

        Returns:
            是否可以執行
        """
        # 檢查依賴是否都已完成
        dependencies = self.dependencies.get(step_id, [])

        for dep_id in dependencies:
            dep_result = self.results.get(dep_id)
            if not dep_result or not dep_result.get("success"):
                return False

        return True

    def get_next_steps(self) -> list[dict[str, Any]]:
        """獲取下一批可執行的步驟"""
        next_steps = []

        for step in self.steps:
            if step["status"] == "pending" and self.can_execute_step(step["step_id"]):
                next_steps.append(step)

        return next_steps

    def mark_step_completed(
        self,
        step_id: str,
        result: dict[str, Any],
    ):
        """標記步驟完成

        Args:
            step_id: 步驟 ID
            result: 執行結果
        """
        # 更新步驟狀態
        for step in self.steps:
            if step["step_id"] == step_id:
                step["status"] = "completed" if result.get("success") else "failed"
                break

        # 保存結果
        self.results[step_id] = result

        logger.debug(f"Step completed: {step_id}, success={result.get('success')}")

    def is_completed(self) -> bool:
        """檢查攻擊鏈是否完成"""
        return all(step["status"] in ["completed", "failed"] for step in self.steps)

    def get_progress(self) -> dict[str, Any]:
        """獲取執行進度"""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s["status"] == "completed")
        failed = sum(1 for s in self.steps if s["status"] == "failed")
        pending = sum(1 for s in self.steps if s["status"] == "pending")

        return {
            "chain_id": self.chain_id,
            "status": self.status.value,
            "total_steps": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
        }

    def get_execution_path(self) -> list[str]:
        """獲取執行路徑"""
        return [step["step_id"] for step in self.steps if step["status"] == "completed"]

    def get_summary(self) -> dict[str, Any]:
        """獲取摘要信息"""
        return {
            "chain_id": self.chain_id,
            "progress": self.get_progress(),
            "execution_path": self.get_execution_path(),
            "results_summary": {
                step_id: {
                    "success": result.get("success"),
                    "findings": len(result.get("findings", [])),
                }
                for step_id, result in self.results.items()
            },
        }
