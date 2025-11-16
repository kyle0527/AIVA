"""基於規則的測試策略生成器

從攻擊面分析結果生成測試策略，包括：
- XSS 測試任務生成
- SQLi 測試任務生成
- SSRF 測試任務生成
- IDOR 測試任務生成

Architecture Fix Note:
- 修復日期: 2025-11-16
- 修復項目: 問題三「決策交接不明確」
- 新增: generate_from_intent() 方法接收 HighLevelIntent 並生成 AttackPlan (AST)
- 職責劃分: StrategyGenerator 負責「怎麼做」(How) - 將高階意圖轉換為具體的 AST
"""

import logging

from services.aiva_common.schemas import ScanCompletedPayload, HighLevelIntent, IntentType
from services.core.aiva_core.business_schemas import (
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
    """基於規則的策略生成器

    根據攻擊面分析結果和配置規則，生成針對性的測試策略。
    使用啟發式規則判斷每個資產的測試優先級。
    """

    def __init__(self, config: StrategyGenerationConfig | None = None) -> None:
        """初始化策略生成器

        Args:
            config: 策略生成配置，如果為 None 則使用默認配置
        """
        self.config = config or StrategyGenerationConfig()
        logger.info("[目標] RuleBasedStrategyGenerator initialized")

    def generate(
        self,
        attack_surface: AttackSurfaceAnalysis,
        scan_payload: ScanCompletedPayload,
    ) -> TestStrategy:
        """從攻擊面分析生成測試策略

        Args:
            attack_surface: 攻擊面分析結果
            scan_payload: 原始掃描完成負載

        Returns:
            完整的測試策略
        """
        logger.info(
            f"[目標] Generating test strategy for scan {attack_surface.scan_id}"
        )
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
                f"[警告]  Total tasks ({total_tasks}) exceeds limit "
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
            f"[已] Strategy generated: {strategy.total_tasks} tasks "
            f"(XSS:{len(xss_tasks)}, SQLi:{len(sqli_tasks)}, "
            f"SSRF:{len(ssrf_tasks)}, IDOR:{len(idor_tasks)})"
        )

        return strategy

    def generate_from_intent(self, intent: HighLevelIntent) -> TestStrategy:
        """從高階意圖生成測試策略 (問題三修復)
        
        這是 cognitive_core → task_planning 的標準接口
        
        職責劃分：
        - cognitive_core (EnhancedDecisionAgent): 決定「做什麼」(What) - 輸出 HighLevelIntent
        - task_planning (StrategyGenerator): 決定「怎麼做」(How) - 生成 AST (AttackPlan)
        
        Args:
            intent: 高階意圖 (來自 cognitive_core)
            
        Returns:
            TestStrategy: 測試策略 (AST 格式)
        """
        logger.info(
            f"[目標] Generating strategy from intent: {intent.intent_type.value} "
            f"(confidence: {intent.confidence:.2f})"
        )
        
        # 根據意圖類型生成不同的策略
        if intent.intent_type == IntentType.TEST_VULNERABILITY:
            return self._generate_vulnerability_test_strategy(intent)
        elif intent.intent_type == IntentType.SCAN_SURFACE:
            return self._generate_surface_scan_strategy(intent)
        elif intent.intent_type == IntentType.EXPLOIT_TARGET:
            return self._generate_exploit_strategy(intent)
        elif intent.intent_type == IntentType.ANALYZE_RESULTS:
            return self._generate_analysis_strategy(intent)
        else:
            # 默認策略
            return self._generate_default_strategy(intent)
    
    def _generate_vulnerability_test_strategy(self, intent: HighLevelIntent) -> TestStrategy:
        """生成漏洞測試策略"""
        vuln_type = intent.parameters.get("vulnerability_type", "unknown")
        test_depth = intent.parameters.get("test_depth", "basic")
        payload_count = intent.parameters.get("payload_count", 10)
        
        # 根據漏洞類型生成任務
        tasks = []
        if vuln_type == "sql_injection":
            tasks = self._generate_sqli_tasks_from_intent(intent, test_depth, payload_count)
        elif vuln_type == "xss":
            tasks = self._generate_xss_tasks_from_intent(intent, test_depth, payload_count)
        elif vuln_type == "ssrf":
            tasks = self._generate_ssrf_tasks_from_intent(intent, test_depth, payload_count)
        
        strategy = TestStrategy(
            scan_id=intent.intent_id,
            strategy_type="vulnerability_test",
            xss_tasks=tasks if vuln_type == "xss" else [],
            sqli_tasks=tasks if vuln_type == "sql_injection" else [],
            ssrf_tasks=tasks if vuln_type == "ssrf" else [],
            idor_tasks=[],
            estimated_duration_seconds=len(tasks) * 5,  # 估算每個任務 5 秒
        )
        
        logger.info(
            f"[已] Vulnerability test strategy generated: {len(tasks)} tasks "
            f"for {vuln_type}"
        )
        
        return strategy
    
    def _generate_surface_scan_strategy(self, intent: HighLevelIntent) -> TestStrategy:
        """生成攻擊面掃描策略"""
        # 簡化實現：生成基本掃描任務
        strategy = TestStrategy(
            scan_id=intent.intent_id,
            strategy_type="surface_scan",
            xss_tasks=[],
            sqli_tasks=[],
            ssrf_tasks=[],
            idor_tasks=[],
            estimated_duration_seconds=60,
        )
        
        logger.info("[已] Surface scan strategy generated")
        
        return strategy
    
    def _generate_exploit_strategy(self, intent: HighLevelIntent) -> TestStrategy:
        """生成攻擊利用策略"""
        strategy = TestStrategy(
            scan_id=intent.intent_id,
            strategy_type="exploit",
            xss_tasks=[],
            sqli_tasks=[],
            ssrf_tasks=[],
            idor_tasks=[],
            estimated_duration_seconds=120,
        )
        
        logger.info("[已] Exploit strategy generated")
        
        return strategy
    
    def _generate_analysis_strategy(self, intent: HighLevelIntent) -> TestStrategy:
        """生成分析策略"""
        strategy = TestStrategy(
            scan_id=intent.intent_id,
            strategy_type="analysis",
            xss_tasks=[],
            sqli_tasks=[],
            ssrf_tasks=[],
            idor_tasks=[],
            estimated_duration_seconds=30,
        )
        
        logger.info("[已] Analysis strategy generated")
        
        return strategy
    
    def _generate_default_strategy(self, intent: HighLevelIntent) -> TestStrategy:
        """生成默認策略"""
        strategy = TestStrategy(
            scan_id=intent.intent_id,
            strategy_type="default",
            xss_tasks=[],
            sqli_tasks=[],
            ssrf_tasks=[],
            idor_tasks=[],
            estimated_duration_seconds=60,
        )
        
        logger.info("[已] Default strategy generated")
        
        return strategy
    
    def _generate_sqli_tasks_from_intent(
        self, intent: HighLevelIntent, test_depth: str, payload_count: int
    ) -> list[TestTask]:
        """從意圖生成 SQLi 測試任務"""
        task = TestTask(
            vulnerability_type="sqli",
            asset=intent.target.target_value,
            parameter=intent.parameters.get("parameter", "unknown"),
            location=intent.target.target_value,
            priority=int(intent.confidence * 10),
            confidence=intent.confidence,
            metadata={
                "test_depth": test_depth,
                "payload_count": payload_count,
                "reasoning": intent.reasoning,
            },
        )
        return [task]
    
    def _generate_xss_tasks_from_intent(
        self, intent: HighLevelIntent, test_depth: str, payload_count: int
    ) -> list[TestTask]:
        """從意圖生成 XSS 測試任務"""
        task = TestTask(
            vulnerability_type="xss",
            asset=intent.target.target_value,
            parameter=intent.parameters.get("parameter", "unknown"),
            location=intent.target.target_value,
            priority=int(intent.confidence * 10),
            confidence=intent.confidence,
            metadata={
                "test_depth": test_depth,
                "payload_count": payload_count,
                "reasoning": intent.reasoning,
            },
        )
        return [task]
    
    def _generate_ssrf_tasks_from_intent(
        self, intent: HighLevelIntent, test_depth: str, payload_count: int
    ) -> list[TestTask]:
        """從意圖生成 SSRF 測試任務"""
        task = TestTask(
            vulnerability_type="ssrf",
            asset=intent.target.target_value,
            parameter=intent.parameters.get("parameter", "unknown"),
            location=intent.target.target_value,
            priority=int(intent.confidence * 10),
            confidence=intent.confidence,
            metadata={
                "test_depth": test_depth,
                "payload_count": payload_count,
                "reasoning": intent.reasoning,
            },
        )
        return [task]

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
        """計算任務優先級

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
        """按優先級和置信度排序任務，保留最重要的部分

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
        """預估總執行時間（秒）

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
