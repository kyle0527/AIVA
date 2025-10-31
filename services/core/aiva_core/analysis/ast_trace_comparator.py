"""AST 與 Trace 對比分析模組

比較預期的攻擊流程圖 (AST) 與實際執行 trace，
計算差異指標作為強化學習回饋信號。
"""

from dataclasses import asdict, dataclass, field
import logging
from typing import Any

from ..execution_tracer.trace_recorder import ExecutionTrace, TraceType
from ..planner.ast_parser import AttackFlowGraph, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """比較指標

    AST 與 Trace 的差異統計
    """

    # 完成率指標
    expected_steps: int  # 預期步驟數
    completed_steps: int  # 完成步驟數
    completion_rate: float  # 完成率 (0.0-1.0)

    # 順序指標
    sequence_match_rate: float  # 順序匹配率 (0.0-1.0)
    out_of_order_steps: int  # 亂序步驟數

    # 步驟差異
    missing_steps: list[str]  # 缺失的步驟
    extra_steps: list[str]  # 額外的步驟

    # 執行質量
    success_steps: int  # 成功的步驟
    failed_steps: int  # 失敗的步驟
    error_count: int  # 錯誤數量

    # 時間指標
    total_duration_seconds: float | None = None  # 總執行時間
    avg_step_duration_seconds: float | None = None  # 平均步驟時間

    # 綜合評分 (0.0-1.0)
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return asdict(self)


@dataclass
class StepComparison:
    """單步比較結果"""

    expected_step_id: str
    expected_action: str
    actual_executed: bool
    execution_status: str | None = None  # success, failed, skipped
    timing_info: dict[str, Any] = field(default_factory=dict)


class ASTTraceComparator:
    """AST 與 Trace 對比分析器

    比較預期的攻擊流程與實際執行軌跡
    """

    def __init__(self) -> None:
        """初始化對比分析器"""
        logger.info("ASTTraceComparator initialized")

    def compare(
        self, expected_graph: AttackFlowGraph, actual_trace: ExecutionTrace
    ) -> ComparisonMetrics:
        """比較 AST 與 Trace

        Args:
            expected_graph: 預期的攻擊流程圖
            actual_trace: 實際執行軌跡

        Returns:
            比較指標
        """
        logger.info(
            f"Comparing AST graph '{expected_graph.graph_id}' "
            f"with trace '{actual_trace.trace_session_id}'"
        )

        # 1. 提取預期步驟
        expected_steps = self._extract_expected_steps(expected_graph)

        # 2. 提取實際步驟
        actual_steps = self._extract_actual_steps(actual_trace)

        # 3. 計算完成率
        completed, missing = self._calculate_completion(expected_steps, actual_steps)
        completion_rate = (
            len(completed) / len(expected_steps) if expected_steps else 0.0
        )

        # 4. 計算順序匹配率
        sequence_match_rate, out_of_order = self._calculate_sequence_match(
            expected_steps, actual_steps
        )

        # 5. 找出額外步驟
        extra_steps = self._find_extra_steps(expected_steps, actual_steps)

        # 6. 統計成功/失敗步驟
        success_count, failed_count = self._count_success_failure(actual_trace)

        # 7. 統計錯誤
        error_count = len(actual_trace.get_entries_by_type(TraceType.ERROR))

        # 8. 計算時間指標
        duration, avg_duration = self._calculate_timing(actual_trace)

        # 9. 計算綜合評分
        overall_score = self._calculate_overall_score(
            completion_rate=completion_rate,
            sequence_match_rate=sequence_match_rate,
            error_rate=error_count / max(len(expected_steps), 1),
        )

        metrics = ComparisonMetrics(
            expected_steps=len(expected_steps),
            completed_steps=len(completed),
            completion_rate=completion_rate,
            sequence_match_rate=sequence_match_rate,
            out_of_order_steps=out_of_order,
            missing_steps=missing,
            extra_steps=extra_steps,
            success_steps=success_count,
            failed_steps=failed_count,
            error_count=error_count,
            total_duration_seconds=duration,
            avg_step_duration_seconds=avg_duration,
            overall_score=overall_score,
        )

        logger.info(
            f"Comparison complete: completion={completion_rate:.2%}, "
            f"score={overall_score:.2f}"
        )
        return metrics

    def _extract_expected_steps(self, graph: AttackFlowGraph) -> list[dict[str, Any]]:
        """從 AST 提取預期步驟

        Args:
            graph: 攻擊流程圖

        Returns:
            預期步驟列表
        """
        steps = []
        for node in graph.nodes.values():
            # 跳過 START 和 END 節點
            if node.node_type in (NodeType.START, NodeType.END):
                continue

            steps.append(
                {
                    "node_id": node.node_id,
                    "action": node.action,
                    "type": node.node_type.value,
                    "parameters": node.parameters,
                }
            )

        return steps

    def _extract_actual_steps(self, trace: ExecutionTrace) -> list[dict[str, Any]]:
        """從 Trace 提取實際步驟

        Args:
            trace: 執行軌跡

        Returns:
            實際步驟列表
        """
        steps = []
        task_starts = trace.get_entries_by_type(TraceType.TASK_START)

        for entry in task_starts:
            steps.append(
                {
                    "task_id": entry.task_id,
                    "action": entry.content.get("action", ""),
                    "type": entry.content.get("task_type", ""),
                    "timestamp": entry.timestamp,
                }
            )

        return steps

    def _calculate_completion(
        self, expected: list[dict[str, Any]], actual: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """計算完成情況

        Args:
            expected: 預期步驟
            actual: 實際步驟

        Returns:
            (已完成步驟列表, 缺失步驟列表)
        """
        expected_actions = {step["action"] for step in expected}
        actual_actions = {step["action"] for step in actual}

        completed = list(expected_actions & actual_actions)
        missing = list(expected_actions - actual_actions)

        return completed, missing

    def _calculate_sequence_match(
        self, expected: list[dict[str, Any]], actual: list[dict[str, Any]]
    ) -> tuple[float, int]:
        """計算順序匹配率

        Args:
            expected: 預期步驟
            actual: 實際步驟

        Returns:
            (匹配率, 亂序步驟數)
        """
        if not expected or not actual:
            return 0.0, 0

        # 簡化版本：檢查相鄰步驟是否保持相對順序
        expected_sequence = [s["action"] for s in expected]
        actual_sequence = [s["action"] for s in actual]

        # 找出最長公共子序列（LCS）
        lcs_length = self._longest_common_subsequence(
            expected_sequence, actual_sequence
        )

        match_rate = lcs_length / len(expected_sequence)
        out_of_order = len(expected_sequence) - lcs_length

        return match_rate, out_of_order

    def _longest_common_subsequence(self, seq1: list[str], seq2: list[str]) -> int:
        """計算最長公共子序列長度

        Args:
            seq1: 序列 1
            seq2: 序列 2

        Returns:
            LCS 長度
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _find_extra_steps(
        self, expected: list[dict[str, Any]], actual: list[dict[str, Any]]
    ) -> list[str]:
        """找出額外的步驟

        Args:
            expected: 預期步驟
            actual: 實際步驟

        Returns:
            額外步驟列表
        """
        expected_actions = {step["action"] for step in expected}
        actual_actions = {step["action"] for step in actual}

        extra = list(actual_actions - expected_actions)
        return extra

    def _count_success_failure(self, trace: ExecutionTrace) -> tuple[int, int]:
        """統計成功和失敗的步驟

        Args:
            trace: 執行軌跡

        Returns:
            (成功數, 失敗數)
        """
        task_ends = trace.get_entries_by_type(TraceType.TASK_END)

        success_count = 0
        failed_count = 0

        for entry in task_ends:
            if entry.content.get("success", False):
                success_count += 1
            else:
                failed_count += 1

        return success_count, failed_count

    def _calculate_timing(
        self, trace: ExecutionTrace
    ) -> tuple[float | None, float | None]:
        """計算時間指標

        Args:
            trace: 執行軌跡

        Returns:
            (總時間, 平均步驟時間)
        """
        if not trace.end_time:
            return None, None

        duration = (trace.end_time - trace.start_time).total_seconds()

        task_starts = trace.get_entries_by_type(TraceType.TASK_START)
        avg_duration = duration / len(task_starts) if task_starts else None

        return duration, avg_duration

    def _calculate_overall_score(
        self, completion_rate: float, sequence_match_rate: float, error_rate: float
    ) -> float:
        """計算綜合評分

        Args:
            completion_rate: 完成率
            sequence_match_rate: 順序匹配率
            error_rate: 錯誤率

        Returns:
            綜合評分 (0.0-1.0)
        """
        # 加權平均
        score = (
            0.5 * completion_rate  # 完成率佔 50%
            + 0.3 * sequence_match_rate  # 順序匹配率佔 30%
            + 0.2 * max(0, 1.0 - error_rate)  # 低錯誤率佔 20%
        )

        return min(1.0, max(0.0, score))

    def generate_feedback(self, metrics: ComparisonMetrics) -> dict[str, Any]:
        """生成回饋信號

        用於強化學習的獎勵/懲罰信號

        Args:
            metrics: 比較指標

        Returns:
            回饋字典
        """
        # 計算獎勵值 (-1.0 到 1.0)
        reward = metrics.overall_score * 2 - 1.0

        # 生成改進建議
        suggestions = []
        if metrics.completion_rate < 0.8:
            suggestions.append("增加步驟完成率")
        if metrics.sequence_match_rate < 0.7:
            suggestions.append("改善步驟執行順序")
        if metrics.error_count > 0:
            suggestions.append(f"減少錯誤（當前：{metrics.error_count}）")
        if metrics.failed_steps > 0:
            suggestions.append(f"提高步驟成功率（失敗：{metrics.failed_steps}）")

        feedback = {
            "reward": reward,
            "overall_score": metrics.overall_score,
            "completion_rate": metrics.completion_rate,
            "sequence_match_rate": metrics.sequence_match_rate,
            "suggestions": suggestions,
            "metrics_summary": {
                "completed": f"{metrics.completed_steps}/{metrics.expected_steps}",
                "success_rate": (
                    metrics.success_steps
                    / max(metrics.success_steps + metrics.failed_steps, 1)
                ),
                "error_count": metrics.error_count,
            },
        }

        return feedback
