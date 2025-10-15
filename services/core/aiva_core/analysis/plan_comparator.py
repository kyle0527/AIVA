"""
Plan Comparator - 攻擊計畫對比分析器

負責對比 AST 預期計畫與實際執行 Trace，計算差異指標和獎勵分數
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any

from aiva_common.schemas import (
    AttackPlan,
    AttackStep,
    PlanExecutionMetrics,
    TraceRecord,
)

logger = logging.getLogger(__name__)


class StepMatch:
    """步驟匹配結果"""

    def __init__(
        self,
        expected_step: AttackStep | None,
        actual_trace: TraceRecord | None,
        match_type: str,
        similarity_score: float = 0.0,
    ):
        self.expected_step = expected_step
        self.actual_trace = actual_trace
        self.match_type = match_type  # "exact", "partial", "missing", "extra"
        self.similarity_score = similarity_score


class PlanComparator:
    """攻擊計畫對比分析器

    對比 AST 預期計畫與實際執行 Trace，生成詳細的差異分析報告
    """

    def __init__(self) -> None:
        """初始化對比分析器"""
        logger.info("PlanComparator initialized")

    def compare(
        self,
        plan: AttackPlan,
        trace_records: list[TraceRecord],
        session_id: str,
    ) -> PlanExecutionMetrics:
        """對比計畫與執行軌跡

        Args:
            plan: 攻擊計畫
            trace_records: 執行追蹤記錄
            session_id: 會話 ID

        Returns:
            執行指標
        """
        logger.info(
            f"Comparing plan {plan.plan_id} with {len(trace_records)} trace records"
        )

        # 1. 步驟匹配分析
        matches = self._match_steps(plan.steps, trace_records)

        # 2. 計算基本指標
        expected_steps = len(plan.steps)
        executed_steps = len(trace_records)
        completed_steps = sum(1 for t in trace_records if t.status == "success")
        failed_steps = sum(1 for t in trace_records if t.status == "failed")
        skipped_steps = sum(1 for t in trace_records if t.status == "skipped")

        # 3. 計算額外動作數量
        extra_actions = self._count_extra_actions(matches)

        # 4. 計算完成率和成功率
        completion_rate = (
            completed_steps / expected_steps if expected_steps > 0 else 0.0
        )
        success_rate = completed_steps / executed_steps if executed_steps > 0 else 0.0

        # 5. 計算順序準確度
        sequence_accuracy = self._calculate_sequence_accuracy(plan, trace_records)

        # 6. 判斷目標達成
        goal_achieved = self._evaluate_goal_achievement(
            plan, trace_records, completion_rate, success_rate
        )

        # 7. 計算獎勵分數
        reward_score = self._calculate_reward_score(
            completion_rate=completion_rate,
            success_rate=success_rate,
            sequence_accuracy=sequence_accuracy,
            goal_achieved=goal_achieved,
            matches=matches,
        )

        # 8. 計算總執行時間
        total_execution_time = sum(t.execution_time_seconds for t in trace_records)

        metrics = PlanExecutionMetrics(
            plan_id=plan.plan_id,
            session_id=session_id,
            expected_steps=expected_steps,
            executed_steps=executed_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            extra_actions=extra_actions,
            completion_rate=completion_rate,
            success_rate=success_rate,
            sequence_accuracy=sequence_accuracy,
            goal_achieved=goal_achieved,
            reward_score=reward_score,
            total_execution_time=total_execution_time,
            timestamp=datetime.now(UTC),
        )

        logger.info(
            f"Comparison complete: completion={completion_rate:.2%}, "
            f"success={success_rate:.2%}, reward={reward_score:.3f}"
        )

        return metrics

    def _match_steps(
        self,
        expected_steps: list[AttackStep],
        trace_records: list[TraceRecord],
    ) -> list[StepMatch]:
        """匹配預期步驟與實際執行

        Args:
            expected_steps: 預期步驟列表
            trace_records: 實際執行記錄

        Returns:
            匹配結果列表
        """
        matches: list[StepMatch] = []

        # 建立 step_id 到 trace 的映射
        trace_map = {t.step_id: t for t in trace_records}

        # 匹配預期步驟
        for step in expected_steps:
            if step.step_id in trace_map:
                trace = trace_map[step.step_id]
                similarity = self._calculate_step_similarity(step, trace)

                if similarity >= 0.9:
                    match_type = "exact"
                elif similarity >= 0.5:
                    match_type = "partial"
                else:
                    match_type = "missing"

                matches.append(
                    StepMatch(
                        expected_step=step,
                        actual_trace=trace,
                        match_type=match_type,
                        similarity_score=similarity,
                    )
                )
            else:
                # 步驟未執行
                matches.append(
                    StepMatch(
                        expected_step=step,
                        actual_trace=None,
                        match_type="missing",
                        similarity_score=0.0,
                    )
                )

        # 檢測額外的動作（不在計畫中的執行）
        expected_step_ids = {s.step_id for s in expected_steps}
        for trace in trace_records:
            if trace.step_id not in expected_step_ids:
                matches.append(
                    StepMatch(
                        expected_step=None,
                        actual_trace=trace,
                        match_type="extra",
                        similarity_score=0.0,
                    )
                )

        return matches

    def _calculate_step_similarity(self, step: AttackStep, trace: TraceRecord) -> float:
        """計算步驟相似度

        Args:
            step: 預期步驟
            trace: 實際執行記錄

        Returns:
            相似度 (0.0-1.0)
        """
        similarity = 0.0

        # 工具類型匹配 (40%)
        if step.tool_type == trace.tool_name:
            similarity += 0.4

        # 動作匹配 (30%)
        if step.action.lower() in trace.output_data.get("action", "").lower():
            similarity += 0.3

        # 執行狀態 (30%)
        if trace.status == "success":
            similarity += 0.3
        elif trace.status in {"failed", "error"}:
            similarity += 0.1

        return min(similarity, 1.0)

    def _count_extra_actions(self, matches: list[StepMatch]) -> int:
        """計算額外動作數量

        Args:
            matches: 匹配結果

        Returns:
            額外動作數量
        """
        return sum(1 for m in matches if m.match_type == "extra")

    def _calculate_sequence_accuracy(
        self, plan: AttackPlan, trace_records: list[TraceRecord]
    ) -> float:
        """計算順序準確度

        使用最長公共子序列 (LCS) 算法計算順序相似度

        Args:
            plan: 攻擊計畫
            trace_records: 執行記錄

        Returns:
            順序準確度 (0.0-1.0)
        """
        expected_order = [step.step_id for step in plan.steps]
        actual_order = [trace.step_id for trace in trace_records]

        if not expected_order or not actual_order:
            return 0.0

        # 計算最長公共子序列長度
        lcs_length = self._lcs_length(expected_order, actual_order)

        # 準確度 = LCS長度 / 預期順序長度
        accuracy = lcs_length / len(expected_order)

        return accuracy

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """計算最長公共子序列長度

        Args:
            seq1: 序列1
            seq2: 序列2

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

    def _evaluate_goal_achievement(
        self,
        plan: AttackPlan,
        trace_records: list[TraceRecord],
        completion_rate: float,
        success_rate: float,
    ) -> bool:
        """評估目標達成

        Args:
            plan: 攻擊計畫
            trace_records: 執行記錄
            completion_rate: 完成率
            success_rate: 成功率

        Returns:
            是否達成目標
        """
        # 基本條件：完成率 >= 80% 且成功率 >= 85%
        if completion_rate < 0.8 or success_rate < 0.85:
            return False

        # 檢查是否發現漏洞（對於漏洞檢測類型的計畫）
        if plan.attack_type.value in {"SQLI", "XSS", "SSRF", "IDOR"}:
            has_findings = any(
                trace.output_data.get("findings")
                for trace in trace_records
                if trace.status == "success"
            )
            if not has_findings:
                return False

        # 檢查關鍵步驟是否完成
        return self._check_critical_steps(plan, trace_records)

    def _check_critical_steps(
        self, plan: AttackPlan, trace_records: list[TraceRecord]
    ) -> bool:
        """檢查關鍵步驟是否完成

        Args:
            plan: 攻擊計畫
            trace_records: 執行記錄

        Returns:
            關鍵步驟是否都完成
        """
        # 識別關鍵步驟（通常是 exploit 類型）
        critical_step_ids = {
            step.step_id
            for step in plan.steps
            if "exploit" in step.action.lower() or step.parameters.get("critical")
        }

        if not critical_step_ids:
            return True

        # 檢查關鍵步驟是否都成功執行
        completed_critical = {
            trace.step_id
            for trace in trace_records
            if trace.step_id in critical_step_ids and trace.status == "success"
        }

        return len(completed_critical) == len(critical_step_ids)

    def _calculate_reward_score(
        self,
        completion_rate: float,
        success_rate: float,
        sequence_accuracy: float,
        goal_achieved: bool,
        matches: list[StepMatch],
    ) -> float:
        """計算獎勵分數（用於強化學習）

        Args:
            completion_rate: 完成率
            success_rate: 成功率
            sequence_accuracy: 順序準確度
            goal_achieved: 是否達成目標
            matches: 匹配結果

        Returns:
            獎勵分數 (0.0-1.0)
        """
        # 基礎分數：加權平均
        base_score = (
            completion_rate * 0.3  # 完成率權重 30%
            + success_rate * 0.3  # 成功率權重 30%
            + sequence_accuracy * 0.2  # 順序準確度權重 20%
        )

        # 目標達成獎勵 (20%)
        goal_bonus = 0.2 if goal_achieved else 0.0

        # 計算步驟質量加成
        quality_bonus = self._calculate_quality_bonus(matches)

        # 總獎勵分數
        reward = base_score + goal_bonus + quality_bonus

        # 懲罰額外動作
        extra_penalty = min(
            sum(1 for m in matches if m.match_type == "extra") * 0.05, 0.2
        )

        reward = max(0.0, min(reward - extra_penalty, 1.0))

        return reward

    def _calculate_quality_bonus(self, matches: list[StepMatch]) -> float:
        """計算步驟質量加成

        Args:
            matches: 匹配結果

        Returns:
            質量加成分數
        """
        if not matches:
            return 0.0

        # 精確匹配的步驟越多，加成越高
        exact_matches = sum(1 for m in matches if m.match_type == "exact")
        total_expected = sum(1 for m in matches if m.expected_step is not None)

        if total_expected == 0:
            return 0.0

        quality_ratio = exact_matches / total_expected
        return min(quality_ratio * 0.1, 0.1)  # 最高 10% 加成

    def generate_comparison_report(
        self,
        plan: AttackPlan,
        trace_records: list[TraceRecord],
        metrics: PlanExecutionMetrics,
    ) -> dict[str, Any]:
        """生成詳細的對比分析報告

        Args:
            plan: 攻擊計畫
            trace_records: 執行記錄
            metrics: 執行指標

        Returns:
            對比報告
        """
        matches = self._match_steps(plan.steps, trace_records)

        report: dict[str, Any] = {
            "plan_id": plan.plan_id,
            "session_id": metrics.session_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": {
                "expected_steps": metrics.expected_steps,
                "executed_steps": metrics.executed_steps,
                "completed_steps": metrics.completed_steps,
                "failed_steps": metrics.failed_steps,
                "skipped_steps": metrics.skipped_steps,
                "extra_actions": metrics.extra_actions,
                "completion_rate": metrics.completion_rate,
                "success_rate": metrics.success_rate,
                "sequence_accuracy": metrics.sequence_accuracy,
                "goal_achieved": metrics.goal_achieved,
                "reward_score": metrics.reward_score,
            },
            "step_analysis": [],
            "recommendations": [],
        }

        # 步驟詳細分析
        for match in matches:
            step_info: dict[str, Any] = {
                "match_type": match.match_type,
                "similarity_score": match.similarity_score,
            }

            if match.expected_step:
                step_info["expected"] = {
                    "step_id": match.expected_step.step_id,
                    "action": match.expected_step.action,
                    "tool_type": match.expected_step.tool_type,
                }

            if match.actual_trace:
                step_info["actual"] = {
                    "step_id": match.actual_trace.step_id,
                    "tool_name": match.actual_trace.tool_name,
                    "status": match.actual_trace.status,
                    "execution_time": match.actual_trace.execution_time_seconds,
                }

            report["step_analysis"].append(step_info)

        # 生成建議
        report["recommendations"] = self._generate_recommendations(metrics, matches)

        return report

    def _generate_recommendations(
        self, metrics: PlanExecutionMetrics, matches: list[StepMatch]
    ) -> list[str]:
        """生成改進建議

        Args:
            metrics: 執行指標
            matches: 匹配結果

        Returns:
            建議列表
        """
        recommendations = []

        if metrics.completion_rate < 0.7:
            recommendations.append(
                f"完成率僅 {metrics.completion_rate:.1%}，建議優化計畫步驟或增加容錯機制"
            )

        if metrics.success_rate < 0.8:
            recommendations.append(
                f"成功率僅 {metrics.success_rate:.1%}，建議檢查工具配置和目標可達性"
            )

        if metrics.sequence_accuracy < 0.7:
            recommendations.append(
                f"順序準確度僅 {metrics.sequence_accuracy:.1%}，建議重新評估步驟依賴關係"
            )

        if metrics.extra_actions > 0:
            recommendations.append(
                f"檢測到 {metrics.extra_actions} 個額外動作，建議檢查執行邏輯是否符合預期"
            )

        missing_steps = sum(1 for m in matches if m.match_type == "missing")
        if missing_steps > 0:
            recommendations.append(
                f"有 {missing_steps} 個步驟未執行，建議檢查依賴關係和錯誤處理"
            )

        if not metrics.goal_achieved:
            recommendations.append("未達成目標，建議調整攻擊策略或改進檢測邏輯")

        if not recommendations:
            recommendations.append("執行表現良好，無需特別調整")

        return recommendations
