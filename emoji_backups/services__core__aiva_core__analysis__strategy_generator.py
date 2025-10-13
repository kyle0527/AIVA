"""
基於規則的測試策略生成器

從攻擊面分析結果生成測試策略，包括：
- XSS 測試任務生成
- SQLi 測試任務生成
- SSRF 測試任務生成
- IDOR 測試任務生成
"""

import logging

from services.aiva_common.schemas import ScanCompletedPayload
from services.core.aiva_core.schemas import (
    AttackSurfaceAnalysis,
    IdorCandidate,
    SqliCandidate,
    SsrfCandidate,
    StrategyGenerationConfig,
    TestStrategy,
    TestTask,
    XssCandidate,
)

logger = logging.getLogger(__name__)


class RuleBasedStrategyGenerator:
    """
    基於規則的策略生成器

    根據攻擊面分析結果和配置規則，生成針對性的測試策略。
    使用啟發式規則判斷每個資產的測試優先級。
    """

    def __init__(self, config: StrategyGenerationConfig | None = None) -> None:
        """
        初始化策略生成器

        Args:
            config: 策略生成配置，如果為 None 則使用默認配置
        """
        self.config = config or StrategyGenerationConfig()
        logger.info("🎯 RuleBasedStrategyGenerator initialized")

    def generate(
        self,
        attack_surface: AttackSurfaceAnalysis,
        scan_payload: ScanCompletedPayload,
    ) -> TestStrategy:
        """
        從攻擊面分析生成測試策略

        Args:
            attack_surface: 攻擊面分析結果
            scan_payload: 原始掃描完成負載

        Returns:
            完整的測試策略
        """
        logger.info(f"🎯 Generating test strategy for scan {attack_surface.scan_id}")
        logger.info(f"   - Total candidates: {attack_surface.total_candidates}")

        # 生成各類型任務
        xss_tasks = self._generate_xss_tasks(attack_surface.xss_candidates)
        sqli_tasks = self._generate_sqli_tasks(attack_surface.sqli_candidates)
        ssrf_tasks = self._generate_ssrf_tasks(attack_surface.ssrf_candidates)
        idor_tasks = self._generate_idor_tasks(attack_surface.idor_candidates)

        # 應用任務數量限制
        total_tasks = (
            len(xss_tasks) + len(sqli_tasks) + len(ssrf_tasks) + len(idor_tasks)
        )
        if total_tasks > self.config.max_tasks_per_scan:
            logger.warning(
                f"⚠️  Total tasks ({total_tasks}) exceeds limit "
                f"({self.config.max_tasks_per_scan}), prioritizing..."
            )
            # 優先保留高優先級任務
            xss_tasks = self._prioritize_tasks(xss_tasks)
            sqli_tasks = self._prioritize_tasks(sqli_tasks)
            ssrf_tasks = self._prioritize_tasks(ssrf_tasks)
            idor_tasks = self._prioritize_tasks(idor_tasks)

        # 計算預估執行時間
        estimated_duration = self._estimate_duration(
            len(xss_tasks), len(sqli_tasks), len(ssrf_tasks), len(idor_tasks)
        )

        strategy = TestStrategy(
            scan_id=attack_surface.scan_id,
            strategy_type="comprehensive" if total_tasks > 50 else "targeted",
            xss_tasks=xss_tasks,
            sqli_tasks=sqli_tasks,
            ssrf_tasks=ssrf_tasks,
            idor_tasks=idor_tasks,
            estimated_duration_seconds=estimated_duration,
        )

        logger.info(
            f"✅ Strategy generated: {strategy.total_tasks} tasks "
            f"(XSS:{len(xss_tasks)}, SQLi:{len(sqli_tasks)}, "
            f"SSRF:{len(ssrf_tasks)}, IDOR:{len(idor_tasks)})"
        )

        return strategy

    def _generate_xss_tasks(self, candidates: list[XssCandidate]) -> list[TestTask]:
        """生成 XSS 測試任務"""
        tasks = []
        for candidate in candidates:
            if candidate.confidence < self.config.min_confidence_threshold:
                continue

            priority = self._calculate_priority(
                candidate.confidence,
                "high_risk" if candidate.confidence > 0.7 else "medium_risk",
            )

            task = TestTask(
                vulnerability_type="xss",
                asset=candidate.asset_url,
                parameter=candidate.parameter,
                location=candidate.location,
                priority=priority,
                confidence=candidate.confidence,
                metadata={
                    "xss_type": candidate.xss_type,
                    "context": candidate.context or "unknown",
                    "reasons": ",".join(candidate.reasons),
                },
            )
            tasks.append(task)

        return tasks

    def _generate_sqli_tasks(self, candidates: list[SqliCandidate]) -> list[TestTask]:
        """生成 SQLi 測試任務"""
        tasks = []
        for candidate in candidates:
            if candidate.confidence < self.config.min_confidence_threshold:
                continue

            # SQLi 通常優先級更高
            priority = self._calculate_priority(candidate.confidence, "high_risk")

            task = TestTask(
                vulnerability_type="sqli",
                asset=candidate.asset_url,
                parameter=candidate.parameter,
                location=candidate.location,
                priority=min(priority + 1, 10),  # SQLi 提升 1 級優先級
                confidence=candidate.confidence,
                metadata={
                    "database_hints": ",".join(candidate.database_hints),
                    "error_based_possible": candidate.error_based_possible,
                    "reasons": ",".join(candidate.reasons),
                },
            )
            tasks.append(task)

        return tasks

    def _generate_ssrf_tasks(self, candidates: list[SsrfCandidate]) -> list[TestTask]:
        """生成 SSRF 測試任務"""
        tasks = []
        for candidate in candidates:
            if candidate.confidence < self.config.min_confidence_threshold:
                continue

            priority = self._calculate_priority(candidate.confidence, "high_risk")

            task = TestTask(
                vulnerability_type="ssrf",
                asset=candidate.asset_url,
                parameter=candidate.parameter,
                location=candidate.location,
                priority=priority,
                confidence=candidate.confidence,
                metadata={
                    "target_type": candidate.target_type,
                    "protocols": ",".join(candidate.protocols),
                    "reasons": ",".join(candidate.reasons),
                },
            )
            tasks.append(task)

        return tasks

    def _generate_idor_tasks(self, candidates: list[IdorCandidate]) -> list[TestTask]:
        """生成 IDOR 測試任務"""
        tasks = []
        for candidate in candidates:
            if candidate.confidence < self.config.min_confidence_threshold:
                continue

            priority = self._calculate_priority(candidate.confidence, "medium_risk")

            task = TestTask(
                vulnerability_type="idor",
                asset=candidate.asset_url,
                parameter=candidate.parameter,
                location=candidate.location,
                priority=priority,
                confidence=candidate.confidence,
                metadata={
                    "resource_type": candidate.resource_type or "unknown",
                    "id_pattern": candidate.id_pattern or "unknown",
                    "requires_auth": candidate.requires_auth,
                    "reasons": ",".join(candidate.reasons),
                },
            )
            tasks.append(task)

        return tasks

    def _calculate_priority(self, confidence: float, risk_level: str) -> int:
        """
        計算任務優先級

        Args:
            confidence: 候選置信度 (0.0-1.0)
            risk_level: 風險等級 ("high_risk", "medium_risk", "low_risk")

        Returns:
            優先級 (1-10)
        """
        # 基礎優先級
        base_priority = {
            "high_risk": self.config.high_risk_priority,
            "medium_risk": self.config.medium_risk_priority,
            "low_risk": self.config.low_risk_priority,
        }.get(risk_level, 5)

        # 根據置信度調整
        if confidence >= self.config.high_confidence_threshold:
            base_priority = min(base_priority + 2, 10)
        elif confidence >= 0.5:
            base_priority = min(base_priority + 1, 10)

        return base_priority

    def _prioritize_tasks(self, tasks: list[TestTask]) -> list[TestTask]:
        """
        按優先級和置信度排序任務，保留最重要的部分

        Args:
            tasks: 任務列表

        Returns:
            排序後的任務列表
        """
        # 按優先級（降序）和置信度（降序）排序
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (t.priority, t.confidence),
            reverse=True,
        )

        # 保留最多 max_tasks_per_scan / 4 個任務（假設 4 種類型均分）
        limit = self.config.max_tasks_per_scan // 4
        return sorted_tasks[:limit]

    def _estimate_duration(
        self, xss_count: int, sqli_count: int, ssrf_count: int, idor_count: int
    ) -> int:
        """
        預估總執行時間（秒）

        Args:
            xss_count: XSS 任務數量
            sqli_count: SQLi 任務數量
            ssrf_count: SSRF 任務數量
            idor_count: IDOR 任務數量

        Returns:
            預估時間（秒）
        """
        total = (
            xss_count * self.config.avg_xss_task_duration
            + sqli_count * self.config.avg_sqli_task_duration
            + ssrf_count * self.config.avg_ssrf_task_duration
            + idor_count * self.config.avg_idor_task_duration
        )

        # 考慮並發執行，假設 5 個並發
        concurrent_factor = 5
        return total // concurrent_factor
