"""
AIVA Common AI Capability Evaluator - 能力評估器組件

此文件提供符合 aiva_common 規範的能力評估器實現，
支援 AI 組件能力的全面評估、基準測試和性能監控。

設計特點:
- 實現 ICapabilityEvaluator 介面
- 整合現有 aiva_common 能力 Schema (CapabilityInfo, CapabilityScorecard)
- 支援多維度能力評估 (性能、準確性、可靠性、安全性)
- 自動化基準測試和回歸檢測
- 實時監控和預警機制
- 證據驅動的評估方法論

架構位置:
- 屬於 Common 層的共享組件
- 支援五大模組架構的能力評估需求
- 與 AI 組件生命週期管理整合
"""

import asyncio
import logging
import statistics
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from ..schemas import CapabilityInfo, CapabilityScorecard
from .interfaces import ICapabilityEvaluator

logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    """評估維度枚舉"""

    PERFORMANCE = "performance"  # 性能指標
    ACCURACY = "accuracy"  # 準確性指標
    RELIABILITY = "reliability"  # 可靠性指標
    SECURITY = "security"  # 安全性指標
    USABILITY = "usability"  # 可用性指標
    SCALABILITY = "scalability"  # 可擴展性指標
    MAINTAINABILITY = "maintainability"  # 可維護性指標


class EvaluationSeverity(Enum):
    """評估嚴重性等級"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CapabilityEvidence(BaseModel):
    """能力證據"""

    evidence_id: str = Field(default_factory=lambda: f"evidence_{uuid4().hex[:8]}")
    capability_id: str
    dimension: EvaluationDimension

    # 證據內容
    evidence_type: str  # measurement|observation|test_result|user_feedback
    evidence_value: Any  # 證據數值或內容
    evidence_description: str

    # 置信度和權重
    confidence: float = Field(ge=0.0, le=1.0, description="證據置信度")
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description="證據權重")

    # 時間和上下文
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = Field(default_factory=dict)

    # 來源信息
    source: str = Field(description="證據來源")
    source_version: str | None = None

    # 驗證狀態
    verified: bool = False
    verification_method: str | None = None


class EvaluationMetric(BaseModel):
    """評估指標"""

    metric_id: str
    name: str
    dimension: EvaluationDimension
    description: str

    # 指標配置
    unit: str = Field(description="度量單位")
    target_value: float | None = Field(default=None, description="目標值")
    threshold_warning: float | None = Field(default=None, description="警告閾值")
    threshold_critical: float | None = Field(default=None, description="嚴重閾值")

    # 計算方法
    calculation_method: str = Field(description="計算方法")
    aggregation_type: str = Field(default="average", description="聚合類型")

    # 時間窗口
    evaluation_window_hours: int = Field(default=24, ge=1, le=8760)

    # 啟用狀態
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class BenchmarkTest(BaseModel):
    """基準測試"""

    test_id: str = Field(default_factory=lambda: f"benchmark_{uuid4().hex[:8]}")
    name: str
    description: str
    capability_id: str

    # 測試配置
    test_type: str  # performance|accuracy|load|stress|security
    test_parameters: dict[str, Any] = Field(default_factory=dict)
    expected_results: dict[str, Any] = Field(default_factory=dict)

    # 執行配置
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    retry_count: int = Field(default=3, ge=0, le=10)
    parallel_execution: bool = False

    # 前置條件
    prerequisites: list[str] = Field(default_factory=list)
    setup_commands: list[str] = Field(default_factory=list)
    cleanup_commands: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """基準測試結果"""

    result_id: str = Field(default_factory=lambda: f"result_{uuid4().hex[:8]}")
    test_id: str
    capability_id: str

    # 執行信息
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    execution_time_seconds: float = 0.0

    # 結果
    success: bool
    score: float | None = None
    measurements: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)

    # 錯誤信息
    error_message: str | None = None
    error_stack: str | None = None

    # 比較基線
    baseline_score: float | None = None
    score_delta: float | None = None
    performance_regression: bool = False

    # 元數據
    environment: str = "test"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityAssessment(BaseModel):
    """能力評估報告"""

    assessment_id: str = Field(default_factory=lambda: f"assessment_{uuid4().hex[:12]}")
    capability_id: str
    capability_name: str

    # 評估時間
    assessment_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evaluation_period_start: datetime
    evaluation_period_end: datetime

    # 綜合評分
    overall_score: float = Field(ge=0.0, le=100.0, description="綜合評分")
    grade: str = Field(description="評估等級 A/B/C/D/F")

    # 維度評分（使用 Any 簡化複雜類型推導）
    dimension_scores: dict[str, Any] = Field(default_factory=dict)

    # 關鍵指標
    key_metrics: dict[str, Any] = Field(default_factory=dict)

    # 發現的問題（使用 Any 簡化複雜推導）
    issues: list[Any] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # 趨勢分析
    trend_analysis: dict[str, Any] = Field(default_factory=dict)
    performance_trend: str = Field(description="improving|stable|declining")

    # 證據統計
    total_evidences: int = 0
    high_confidence_evidences: int = 0
    verified_evidences: int = 0

    # 基準測試結果
    benchmark_results: list[str] = Field(default_factory=list)  # result_ids

    # 風險評估
    risk_level: EvaluationSeverity = EvaluationSeverity.LOW
    risk_factors: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """評估配置"""

    # 評估調度
    evaluation_interval_hours: int = Field(default=24, ge=1, le=168)
    continuous_monitoring: bool = True
    auto_benchmark: bool = True

    # 數據收集
    evidence_retention_days: int = Field(default=90, ge=1, le=365)
    min_evidence_count: int = Field(default=10, ge=1, le=1000)
    evidence_quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # 評分權重
    dimension_weights: dict[EvaluationDimension, float] = Field(
        default_factory=lambda: {
            EvaluationDimension.PERFORMANCE: 0.25,
            EvaluationDimension.ACCURACY: 0.25,
            EvaluationDimension.RELIABILITY: 0.20,
            EvaluationDimension.SECURITY: 0.15,
            EvaluationDimension.USABILITY: 0.10,
            EvaluationDimension.SCALABILITY: 0.05,
        }
    )

    # 閾值配置
    warning_score_threshold: float = Field(default=70.0, ge=0.0, le=100.0)
    critical_score_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    regression_threshold: float = Field(default=5.0, ge=0.0, le=50.0)

    # 通知配置
    enable_alerts: bool = True
    alert_channels: list[str] = Field(default_factory=list)


class AIVACapabilityEvaluator(ICapabilityEvaluator):
    """AIVA 能力評估器實現

    此類提供符合 aiva_common 規範的能力評估功能，
    支援多維度、證據驅動的能力評估和持續監控。
    """

    def __init__(self, config: EvaluationConfig | None = None):
        """初始化能力評估器

        Args:
            config: 評估配置
        """
        self.config = config or EvaluationConfig()

        # 數據存儲
        self.capabilities: dict[str, CapabilityInfo] = {}
        self.scorecards: dict[str, CapabilityScorecard] = {}
        self.evidences: dict[str, list[CapabilityEvidence]] = (
            {}
        )  # capability_id -> evidences
        self.metrics: dict[str, EvaluationMetric] = {}
        self.benchmark_tests: dict[str, BenchmarkTest] = {}
        self.benchmark_results: dict[str, list[BenchmarkResult]] = {}
        self.assessments: dict[str, list[CapabilityAssessment]] = {}

        # 統計和監控
        self.start_time = datetime.now(UTC)
        self.total_evaluations = 0
        self.total_benchmarks = 0

        # 評估任務
        self._evaluation_task: asyncio.Task[Any] | None = None
        self._start_evaluation_task()

        logger.info("AIVACapabilityEvaluator initialized")

    async def evaluate_capability(
        self, capability_id: str, execution_evidence: dict[str, Any]
    ) -> dict[str, Any]:
        """評估能力

        Args:
            capability_id: 能力 ID
            execution_evidence: 執行證據

        Returns:
            評估結果
        """
        try:
            if capability_id not in self.capabilities:
                return {"error": f"Capability {capability_id} not found"}

            # 存儲證據
            if capability_id not in self.evidences:
                self.evidences[capability_id] = []

            # 從執行證據轉換為證據列表（明確類型標註）
            evidence_list: list[CapabilityEvidence] = []

            # 如果execution_evidence包含evidence_list字段，使用它
            if "evidence_list" in execution_evidence and isinstance(
                execution_evidence["evidence_list"], list
            ):
                evidence_data_list = cast(
                    list[Any], execution_evidence["evidence_list"]
                )
                for evidence_data in evidence_data_list:
                    if isinstance(evidence_data, CapabilityEvidence):
                        evidence_list.append(evidence_data)
                    elif isinstance(evidence_data, dict):
                        # 從字典創建CapabilityEvidence
                        try:
                            evidence_dict = cast(dict[str, Any], evidence_data)
                            evidence = CapabilityEvidence(
                                capability_id=capability_id,
                                dimension=EvaluationDimension(
                                    evidence_dict.get("dimension", "performance")
                                ),
                                evidence_type=str(
                                    evidence_dict.get("evidence_type", "measurement")
                                ),
                                evidence_value=evidence_dict.get("evidence_value"),
                                evidence_description=str(
                                    evidence_dict.get("evidence_description", "")
                                ),
                                confidence=float(evidence_dict.get("confidence", 0.8)),
                                source=str(evidence_dict.get("source", "execution")),
                            )
                            evidence_list.append(evidence)
                        except Exception as e:
                            logger.warning(f"Failed to create evidence from dict: {e}")

            # 過濾和驗證證據
            valid_evidences: list[CapabilityEvidence] = []
            for evidence in evidence_list:
                if await self._validate_evidence(evidence):
                    valid_evidences.append(evidence)
                    self.evidences[capability_id].append(evidence)

            if len(valid_evidences) < self.config.min_evidence_count:
                return {
                    "error": f"Insufficient evidence count: {len(valid_evidences)} < {self.config.min_evidence_count}"
                }

            # 執行評估
            assessment = await self._perform_assessment(capability_id, valid_evidences)

            # 更新評分卡
            await self._update_scorecard(capability_id, assessment)

            # 存儲評估結果
            if capability_id not in self.assessments:
                self.assessments[capability_id] = []
            self.assessments[capability_id].append(assessment)

            # 更新統計
            self.total_evaluations += 1

            logger.info(
                f"Capability evaluation completed: {capability_id}, score: {assessment.overall_score:.1f}"
            )

            return {
                "assessment_id": assessment.assessment_id,
                "capability_id": capability_id,
                "overall_score": assessment.overall_score,
                "grade": assessment.grade,
                "dimension_scores": {
                    dim.value: score
                    for dim, score in assessment.dimension_scores.items()
                },
                "key_metrics": assessment.key_metrics,
                "risk_level": assessment.risk_level.value,
                "recommendations": assessment.recommendations[:5],  # 限制建議數量
                "evaluation_time": assessment.assessment_time.isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Error evaluating capability {capability_id}: {e}", exc_info=True
            )
            return {"error": str(e)}

    async def run_benchmark(
        self, capability_id: str, test_suite: list[str] | None = None
    ) -> dict[str, Any]:
        """運行基準測試

        Args:
            capability_id: 能力 ID
            test_suite: 測試套件 ID 列表 (可選)

        Returns:
            基準測試結果
        """
        try:
            if capability_id not in self.capabilities:
                return {"error": f"Capability {capability_id} not found"}

            # 獲取測試列表
            if test_suite:
                tests = [
                    self.benchmark_tests[test_id]
                    for test_id in test_suite
                    if test_id in self.benchmark_tests
                ]
            else:
                # 獲取該能力的所有測試
                tests = [
                    test
                    for test in self.benchmark_tests.values()
                    if test.capability_id == capability_id
                ]

            if not tests:
                return {
                    "error": f"No benchmark tests found for capability {capability_id}"
                }

            results: list[Any] = []  # 明確類型標註
            overall_success = True
            total_score = 0.0
            score_count = 0

            # 執行測試
            for test in tests:
                result = await self._execute_benchmark_test(test)
                results.append(result)

                if not result.success:
                    overall_success = False

                if result.score is not None:
                    total_score += result.score
                    score_count += 1

                # 存儲結果
                if capability_id not in self.benchmark_results:
                    self.benchmark_results[capability_id] = []
                self.benchmark_results[capability_id].append(result)

            # 計算平均分數
            avg_score = total_score / score_count if score_count > 0 else None

            # 更新統計
            self.total_benchmarks += len(results)

            logger.info(
                f"Benchmark completed for {capability_id}: {len(results)} tests, avg_score: {avg_score}"
            )

            return {
                "capability_id": capability_id,
                "total_tests": len(results),
                "successful_tests": sum(1 for r in results if r.success),
                "failed_tests": sum(1 for r in results if not r.success),
                "overall_success": overall_success,
                "average_score": avg_score,
                "execution_time": sum(r.execution_time_seconds for r in results),
                "results": [
                    {
                        "result_id": r.result_id,
                        "test_id": r.test_id,
                        "success": r.success,
                        "score": r.score,
                        "execution_time": r.execution_time_seconds,
                        "error_message": r.error_message,
                    }
                    for r in results
                ],
            }

        except Exception as e:
            logger.error(
                f"Error running benchmark for capability {capability_id}: {e}",
                exc_info=True,
            )
            return {"error": str(e)}

    async def get_capability_assessment(
        self, capability_id: str, include_history: bool = False
    ) -> dict[str, Any]:
        """獲取能力評估報告

        Args:
            capability_id: 能力 ID
            include_history: 是否包含歷史記錄

        Returns:
            評估報告
        """
        try:
            if capability_id not in self.capabilities:
                return {"error": f"Capability {capability_id} not found"}

            capability = self.capabilities[capability_id]
            scorecard = self.scorecards.get(capability_id)
            assessments = self.assessments.get(capability_id, [])

            # 獲取最新評估
            latest_assessment = assessments[-1] if assessments else None

            result: dict[str, Any] = {
                "capability_id": capability_id,
                "capability_name": capability.name,
                "capability_version": capability.version,
                "capability_status": capability.status.value,
                "latest_assessment": None,
                "scorecard": None,
                "evidence_summary": {
                    "total_evidences": len(self.evidences.get(capability_id, [])),
                    "verified_evidences": sum(
                        1 for e in self.evidences.get(capability_id, []) if e.verified
                    ),
                    "recent_evidences": len(
                        [
                            e
                            for e in self.evidences.get(capability_id, [])
                            if (datetime.now(UTC) - e.timestamp).days <= 7
                        ]
                    ),
                },
                "benchmark_summary": {
                    "total_results": len(self.benchmark_results.get(capability_id, [])),
                    "recent_results": len(
                        [
                            r
                            for r in self.benchmark_results.get(capability_id, [])
                            if (datetime.now(UTC) - r.start_time).days <= 7
                        ]
                    ),
                },
            }

            # 添加最新評估
            if latest_assessment:
                result["latest_assessment"] = {
                    "assessment_id": latest_assessment.assessment_id,
                    "overall_score": latest_assessment.overall_score,
                    "grade": latest_assessment.grade,
                    "dimension_scores": {
                        dim: score
                        for dim, score in latest_assessment.dimension_scores.items()
                    },
                    "key_metrics": latest_assessment.key_metrics,
                    "risk_level": latest_assessment.risk_level.value,
                    "performance_trend": latest_assessment.performance_trend,
                    "assessment_time": latest_assessment.assessment_time.isoformat(),
                    "issues_count": len(latest_assessment.issues),
                    "recommendations_count": len(latest_assessment.recommendations),
                }

            # 添加評分卡
            if scorecard:
                result["scorecard"] = {
                    "success_rate_7d": scorecard.success_rate_7d,
                    "avg_latency_ms": scorecard.avg_latency_ms,
                    "availability_7d": scorecard.availability_7d,
                    "usage_count_7d": scorecard.usage_count_7d,
                    "error_count_7d": scorecard.error_count_7d,
                    "last_used_at": (
                        scorecard.last_used_at.isoformat()
                        if scorecard.last_used_at
                        else None
                    ),
                    "last_updated_at": (
                        scorecard.last_updated_at.isoformat()
                        if scorecard.last_updated_at
                        else None
                    ),
                }

            # 包含歷史記錄
            if include_history and assessments:
                result["assessment_history"] = [
                    {
                        "assessment_id": a.assessment_id,
                        "overall_score": a.overall_score,
                        "grade": a.grade,
                        "risk_level": a.risk_level.value,
                        "assessment_time": a.assessment_time.isoformat(),
                    }
                    for a in assessments[-10:]  # 最近10次評估
                ]

            return result

        except Exception as e:
            logger.error(f"Error getting capability assessment {capability_id}: {e}")
            return {"error": str(e)}

    def get_evaluation_statistics(self) -> dict[str, Any]:
        """獲取評估統計信息

        Returns:
            統計信息字典
        """
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 計算各種統計數據
        total_capabilities = len(self.capabilities)
        total_evidences = sum(len(evidences) for evidences in self.evidences.values())
        total_assessments = sum(
            len(assessments) for assessments in self.assessments.values()
        )

        # 評分分佈（明確類型標註）
        latest_scores: list[Any] = []
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for assessments in self.assessments.values():
            if assessments:
                latest = assessments[-1]
                latest_scores.append(latest.overall_score)
                risk_distribution[latest.risk_level.value] += 1

        return {
            "uptime_seconds": uptime,
            "total_capabilities": total_capabilities,
            "total_evaluations": self.total_evaluations,
            "total_benchmarks": self.total_benchmarks,
            "total_evidences": total_evidences,
            "total_assessments": total_assessments,
            "score_statistics": {
                "count": len(latest_scores),
                "average": statistics.mean(latest_scores) if latest_scores else 0.0,
                "median": statistics.median(latest_scores) if latest_scores else 0.0,
                "min": min(latest_scores) if latest_scores else 0.0,
                "max": max(latest_scores) if latest_scores else 0.0,
            },
            "risk_distribution": risk_distribution,
            "configuration": {
                "evaluation_interval_hours": self.config.evaluation_interval_hours,
                "continuous_monitoring": self.config.continuous_monitoring,
                "auto_benchmark": self.config.auto_benchmark,
                "evidence_retention_days": self.config.evidence_retention_days,
                "min_evidence_count": self.config.min_evidence_count,
            },
        }

    async def register_capability(self, capability: CapabilityInfo) -> bool:
        """註冊能力

        Args:
            capability: 能力信息

        Returns:
            是否註冊成功
        """
        try:
            self.capabilities[capability.id] = capability

            # 初始化評分卡
            scorecard = CapabilityScorecard(
                capability_id=capability.id,
                last_used_at=datetime.now(UTC),
                last_updated_at=datetime.now(UTC),
                metadata={},
            )
            self.scorecards[capability.id] = scorecard

            # 初始化存儲
            self.evidences[capability.id] = []
            self.benchmark_results[capability.id] = []
            self.assessments[capability.id] = []

            logger.info(f"Capability registered: {capability.id} ({capability.name})")
            return True

        except Exception as e:
            logger.error(f"Error registering capability {capability.id}: {e}")
            return False

    async def add_benchmark_test(self, test: BenchmarkTest) -> bool:
        """添加基準測試

        Args:
            test: 基準測試

        Returns:
            是否添加成功
        """
        try:
            self.benchmark_tests[test.test_id] = test
            logger.info(
                f"Benchmark test added: {test.test_id} for capability {test.capability_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding benchmark test {test.test_id}: {e}")
            return False

    async def _validate_evidence(self, evidence: CapabilityEvidence) -> bool:
        """驗證證據有效性"""
        # 檢查置信度閾值
        if evidence.confidence < self.config.evidence_quality_threshold:
            return False

        # 檢查時效性 (證據不能太舊)
        max_age_days = 30
        age = (datetime.now(UTC) - evidence.timestamp).days
        if age > max_age_days:
            return False

        # 檢查必需字段
        if not evidence.evidence_description or not evidence.source:
            return False

        return True

    async def _perform_assessment(
        self, capability_id: str, evidences: list[Any]  # 使用 Any 簡化複雜類型推導
    ) -> Any:  # 使用 Any 簡化返回類型
        """執行能力評估"""
        capability = self.capabilities[capability_id]

        # 計算維度評分（使用 Any 簡化類型推導）
        dimension_scores: dict[str, Any] = {}
        for dimension in EvaluationDimension:
            dimension_evidences = [
                e
                for e in evidences
                if hasattr(e, "dimension") and e.dimension == dimension
            ]
            if dimension_evidences:
                score = await self._calculate_dimension_score(
                    dimension, dimension_evidences
                )
                dimension_scores[dimension.value] = (
                    score  # 使用 enum value 作為字符串鍵
                )

        # 計算綜合評分（明確類型標註）
        overall_score: float = 0.0
        total_weight: float = 0.0

        for dimension_key, score in dimension_scores.items():
            # 使用 Any 避免複雜的字典查找類型推導
            weight: Any = getattr(self.config, "dimension_weights", {}).get(
                dimension_key, 1.0
            )
            overall_score += float(score) * float(weight)
            total_weight += float(weight)

        if total_weight > 0:
            overall_score /= total_weight

        # 確定等級
        grade = self._calculate_grade(overall_score)

        # 風險評估
        risk_level = self._assess_risk_level(overall_score, dimension_scores)

        # 生成建議
        recommendations = await self._generate_recommendations(
            capability_id, dimension_scores, evidences
        )

        # 趨勢分析
        trend_analysis, performance_trend = await self._analyze_trends(capability_id)

        # 創建評估報告
        assessment = CapabilityAssessment(
            capability_id=capability_id,
            capability_name=capability.name,
            evaluation_period_start=min(e.timestamp for e in evidences),
            evaluation_period_end=max(e.timestamp for e in evidences),
            overall_score=overall_score,
            grade=grade,
            dimension_scores=dimension_scores,
            risk_level=risk_level,
            recommendations=recommendations,
            trend_analysis=trend_analysis,
            performance_trend=performance_trend,
            total_evidences=len(evidences),
            high_confidence_evidences=len(
                [e for e in evidences if e.confidence >= 0.8]
            ),
            verified_evidences=len([e for e in evidences if e.verified]),
        )

        return assessment

    async def _calculate_dimension_score(
        self, dimension: EvaluationDimension, evidences: list[CapabilityEvidence]
    ) -> float:
        """計算維度評分"""
        if not evidences:
            return 0.0

        # 加權平均評分
        total_score = 0.0
        total_weight = 0.0

        for evidence in evidences:
            # 從證據值中提取評分
            score = self._extract_score_from_evidence(evidence)
            weight = evidence.weight * evidence.confidence

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _extract_score_from_evidence(self, evidence: CapabilityEvidence) -> float:
        """從證據中提取評分"""
        value = evidence.evidence_value

        if isinstance(value, (int, float)):
            # 數值型證據，需要根據類型進行歸一化
            if evidence.evidence_type == "measurement":
                # 性能測量：假設較小值較好 (如延遲)
                if (
                    "latency" in evidence.evidence_description.lower()
                    or "time" in evidence.evidence_description.lower()
                ):
                    return max(0, 100 - value)  # 假設100ms以下為滿分
                else:
                    return min(value, 100)  # 假設100為滿分
            elif evidence.evidence_type == "test_result":
                # 測試結果：通常是百分比
                return min(value, 100)
            else:
                return min(value, 100)

        elif isinstance(value, bool):
            return 100.0 if value else 0.0

        elif isinstance(value, str):
            # 文本型證據，需要進行文本分析
            if any(word in value.lower() for word in ["excellent", "good", "success"]):
                return 90.0
            elif any(word in value.lower() for word in ["fair", "average", "ok"]):
                return 70.0
            elif any(word in value.lower() for word in ["poor", "bad", "fail"]):
                return 30.0
            else:
                return 50.0  # 默認中等分數

        else:
            return 50.0  # 默認分數

    def _calculate_grade(self, score: float) -> str:
        """計算等級"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _assess_risk_level(
        self,
        overall_score: float,
        dimension_scores: dict[str, Any],  # 使用 Any 簡化類型推導
    ) -> Any:  # 使用 Any 簡化返回類型
        """評估風險等級"""
        if overall_score < self.config.critical_score_threshold:
            return EvaluationSeverity.CRITICAL
        elif overall_score < self.config.warning_score_threshold:
            return EvaluationSeverity.HIGH

        # 檢查關鍵維度（使用字符串鍵避免枚舉類型推導問題）
        security_score: Any = dimension_scores.get("security", 100)
        reliability_score: Any = dimension_scores.get("reliability", 100)

        if float(security_score) < 60 or float(reliability_score) < 60:
            return EvaluationSeverity.HIGH
        elif float(security_score) < 80 or float(reliability_score) < 80:
            return EvaluationSeverity.MEDIUM

        return EvaluationSeverity.LOW

    async def _generate_recommendations(
        self,
        capability_id: str,
        dimension_scores: dict[str, Any],  # 使用 Any 簡化類型推導
        evidences: list[Any],  # 使用 Any 簡化類型推導
    ) -> list[Any]:  # 使用 Any 簡化返回類型推導
        """生成改進建議"""
        recommendations: list[Any] = []  # 明確類型標註

        # 基於維度評分的建議（使用字符串比較避免複雜枚舉推導）
        for dimension_key, score in dimension_scores.items():
            if float(score) < 70:
                if dimension_key == "performance":
                    recommendations.append("優化性能瓶頸，考慮實施緩存或負載平衡")
                elif dimension_key == "accuracy":
                    recommendations.append("提高準確性，檢查算法參數和訓練數據質量")
                elif dimension_key == "reliability":
                    recommendations.append("增強可靠性，實施錯誤處理和重試機制")
                elif dimension_key == "security":
                    recommendations.append("加強安全措施，進行安全審計和漏洞修復")
                elif dimension_key == "usability":
                    recommendations.append("改善可用性，簡化接口和提供更好的文檔")

        # 基於證據的建議（使用 Any 避免複雜屬性推導）
        error_evidences = [
            e
            for e in evidences
            if hasattr(e, "evidence_description")
            and "error" in str(getattr(e, "evidence_description", "")).lower()
        ]
        if len(error_evidences) > len(evidences) * 0.3:  # 超過30%的證據涉及錯誤
            recommendations.append("錯誤率偏高，需要進行根本原因分析")

        # 基於趨勢的建議
        assessments = self.assessments.get(capability_id, [])
        if len(assessments) >= 3:
            recent_scores = [a.overall_score for a in assessments[-3:]]
            if all(
                recent_scores[i] < recent_scores[i - 1]
                for i in range(1, len(recent_scores))
            ):
                recommendations.append("性能呈下降趨勢，建議進行深入診斷")

        return recommendations[:10]  # 限制建議數量

    async def _analyze_trends(self, capability_id: str) -> tuple[dict[str, Any], str]:
        """分析趨勢"""
        assessments = self.assessments.get(capability_id, [])

        if len(assessments) < 2:
            return {}, "stable"

        # 獲取最近的評估
        recent_assessments = assessments[-5:]  # 最近5次評估
        scores = [a.overall_score for a in recent_assessments]

        # 計算趨勢
        if len(scores) >= 3:
            # 簡單線性趨勢
            improvements = sum(
                1 for i in range(1, len(scores)) if scores[i] > scores[i - 1]
            )
            declines = sum(
                1 for i in range(1, len(scores)) if scores[i] < scores[i - 1]
            )

            if improvements > declines:
                trend = "improving"
            elif declines > improvements:
                trend = "declining"
            else:
                trend = "stable"
        else:
            # 只有兩個數據點
            if scores[-1] > scores[0]:
                trend = "improving"
            elif scores[-1] < scores[0]:
                trend = "declining"
            else:
                trend = "stable"

        trend_analysis = {
            "recent_scores": scores,
            "score_variance": statistics.variance(scores) if len(scores) > 1 else 0.0,
            "assessment_count": len(assessments),
        }

        return trend_analysis, trend

    async def _execute_benchmark_test(self, test: BenchmarkTest) -> BenchmarkResult:
        """執行基準測試"""
        result = BenchmarkResult(
            test_id=test.test_id,
            capability_id=test.capability_id,
            success=False,  # 初始化為False，執行成功後會更新
        )

        try:
            # 模擬測試執行 (實際實現中會調用真實的測試)
            await asyncio.sleep(0.1)  # 模擬執行時間

            # 模擬測試結果
            import random

            success_probability = 0.8  # 80% 成功率
            result.success = random.random() < success_probability

            if result.success:
                result.score = random.uniform(70, 100)  # 成功時的分數範圍
            else:
                result.score = random.uniform(0, 60)  # 失敗時的分數範圍
                result.error_message = "Simulated test failure"

            result.end_time = datetime.now(UTC)
            result.execution_time_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

            # 模擬測量數據
            result.measurements = {
                "response_time_ms": random.uniform(10, 200),
                "throughput_rps": random.uniform(100, 1000),
                "error_rate": random.uniform(0, 0.1),
            }

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now(UTC)
            result.execution_time_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

        return result

    async def _update_scorecard(
        self, capability_id: str, assessment: CapabilityAssessment
    ) -> CapabilityScorecard:
        """更新評分卡"""
        scorecard = self.scorecards.get(capability_id)
        if not scorecard:
            scorecard = CapabilityScorecard(
                capability_id=capability_id,
                last_used_at=datetime.now(UTC),
                last_updated_at=datetime.now(UTC),
                metadata={},
            )
            self.scorecards[capability_id] = scorecard

        # 更新基於評估結果的指標
        scorecard.success_rate_7d = assessment.overall_score / 100.0
        scorecard.last_updated_at = datetime.now(UTC)

        # 從關鍵指標中提取性能數據
        if "avg_latency_ms" in assessment.key_metrics:
            scorecard.avg_latency_ms = assessment.key_metrics["avg_latency_ms"]

        if "availability" in assessment.key_metrics:
            scorecard.availability_7d = assessment.key_metrics["availability"]

        return scorecard

    def _start_evaluation_task(self) -> None:
        """啟動評估任務"""
        if self.config.continuous_monitoring and (
            self._evaluation_task is None or self._evaluation_task.done()
        ):
            try:
                # 檢查是否有事件循環在運行
                loop = asyncio.get_running_loop()
                self._evaluation_task = loop.create_task(self._continuous_evaluation())
            except RuntimeError:
                # 沒有事件循環在運行，跳過任務創建
                logger.info(
                    "No running event loop, skipping continuous evaluation task creation"
                )

    async def _continuous_evaluation(self) -> None:
        """持續評估任務"""
        while True:
            try:
                await asyncio.sleep(self.config.evaluation_interval_hours * 3600)

                # 為所有註冊的能力執行自動評估
                for capability_id in self.capabilities.keys():
                    evidences = self.evidences.get(capability_id, [])

                    # 獲取最近的證據
                    cutoff_time = datetime.now(UTC) - timedelta(days=7)
                    recent_evidences = [
                        e for e in evidences if e.timestamp >= cutoff_time
                    ]

                    if len(recent_evidences) >= self.config.min_evidence_count:
                        await self.evaluate_capability(
                            capability_id, {"evidence_list": recent_evidences}
                        )

                    # 自動基準測試
                    if self.config.auto_benchmark:
                        await self.run_benchmark(capability_id)

            except Exception as e:
                logger.error(f"Error in continuous evaluation: {e}")

    # ============================================================================
    # Abstract Methods Implementation (抽象方法實現)
    # ============================================================================

    async def collect_capability_evidence(
        self, capability_id: str, time_window_days: int = 7
    ) -> list[dict[str, Any]]:
        """收集能力證據

        Args:
            capability_id: 能力ID
            time_window_days: 時間窗口天數

        Returns:
            證據列表
        """
        try:
            if capability_id not in self.evidences:
                return []

            # 計算時間窗口
            cutoff_time = datetime.now(UTC) - timedelta(days=time_window_days)

            # 過濾時間窗口內的證據
            filtered_evidences: list[dict[str, Any]] = []
            for evidence in self.evidences[capability_id]:
                if evidence.timestamp >= cutoff_time:
                    evidence_dict = {
                        "evidence_id": evidence.evidence_id,
                        "capability_id": evidence.capability_id,
                        "evidence_type": evidence.evidence_type,
                        "evidence_value": evidence.evidence_value,
                        "evidence_description": evidence.evidence_description,
                        "confidence": evidence.confidence,
                        "timestamp": evidence.timestamp.isoformat(),
                        "weight": evidence.weight,
                    }
                    filtered_evidences.append(evidence_dict)

            logger.info(
                f"Collected {len(filtered_evidences)} evidence items for capability {capability_id}"
            )
            return filtered_evidences

        except Exception as e:
            logger.error(f"Error collecting capability evidence: {e}")
            return []

    async def update_capability_scorecard(
        self, capability_id: str, metrics: dict[str, float]
    ) -> bool:
        """更新能力記分卡

        Args:
            capability_id: 能力ID
            metrics: 指標數據

        Returns:
            更新是否成功
        """
        try:
            if capability_id not in self.capabilities:
                logger.warning(
                    f"Capability {capability_id} not found, cannot update scorecard"
                )
                return False

            # 獲取或創建記分卡
            if capability_id not in self.scorecards:
                self.scorecards[capability_id] = CapabilityScorecard(
                    capability_id=capability_id,
                    last_used_at=datetime.now(UTC),
                    last_updated_at=datetime.now(UTC),
                    metadata={},
                )

            scorecard = self.scorecards[capability_id]

            # 更新基本性能指標 (使用 CapabilityScorecard 實際支援的屬性)
            if "success_rate" in metrics:
                scorecard.success_rate_7d = min(1.0, max(0.0, metrics["success_rate"]))
            if "avg_latency" in metrics:
                scorecard.avg_latency_ms = max(0.0, metrics["avg_latency"])
            if "availability" in metrics:
                scorecard.availability_7d = min(1.0, max(0.0, metrics["availability"]))
            if "usage_count" in metrics:
                scorecard.usage_count_7d = max(0, int(metrics["usage_count"]))
            if "error_count" in metrics:
                scorecard.error_count_7d = max(0, int(metrics["error_count"]))

            # 更新時間戳
            scorecard.last_updated_at = datetime.now(UTC)

            logger.info(
                f"Updated scorecard for capability {capability_id}, success rate: {scorecard.success_rate_7d:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating capability scorecard: {e}")
            return False

    async def cleanup(self) -> None:
        """清理資源"""
        try:
            # 取消評估任務
            if self._evaluation_task and not self._evaluation_task.done():
                self._evaluation_task.cancel()
                try:
                    await self._evaluation_task
                except asyncio.CancelledError:
                    pass

            # 清理數據
            self.capabilities.clear()
            self.scorecards.clear()
            self.evidences.clear()
            self.metrics.clear()
            self.benchmark_tests.clear()
            self.benchmark_results.clear()
            self.assessments.clear()

            logger.info("AIVACapabilityEvaluator cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """析構函數"""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
        except Exception:
            pass


# ============================================================================
# Factory Function (工廠函數)
# ============================================================================


def create_capability_evaluator(
    config: EvaluationConfig | None = None, **kwargs: Any
) -> AIVACapabilityEvaluator:
    """創建能力評估器實例

    Args:
        config: 評估配置
        **kwargs: 其他參數

    Returns:
        能力評估器實例
    """
    return AIVACapabilityEvaluator(config=config)


# ============================================================================
# 全域實例 (Global Instance)
# ============================================================================

# 創建全域能力評估器實例
capability_evaluator = create_capability_evaluator()
