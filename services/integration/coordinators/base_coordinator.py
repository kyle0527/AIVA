"""Integration Base Coordinator - 雙閉環協調器基類

基於業界最佳實踐：
- SARIF (Static Analysis Results Interchange Format)
- OWASP Testing Guide
- Bug Bounty 平台標準（HackerOne、Bugcrowd）
- 可觀測性標準（OpenTelemetry）
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models - 資料驗證 Schema
# ============================================================================

class TargetInfo(BaseModel):
    """目標信息"""
    url: str
    endpoint: Optional[str] = None
    method: str = "GET"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvidenceData(BaseModel):
    """證據數據"""
    payload: str
    request: str
    response: str
    matched_pattern: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PoCData(BaseModel):
    """Proof of Concept 數據"""
    steps: List[str]
    curl_command: Optional[str] = None
    exploit_code: Optional[str] = None
    screenshot_path: Optional[str] = None
    video_path: Optional[str] = None


class ImpactAssessment(BaseModel):
    """影響評估（CVSS 標準）"""
    confidentiality: str = Field(regex="^(high|medium|low|none)$")
    integrity: str = Field(regex="^(high|medium|low|none)$")
    availability: str = Field(regex="^(high|medium|low|none)$")
    scope_changed: bool = False


class RemediationAdvice(BaseModel):
    """修復建議"""
    recommendation: str
    references: List[str] = Field(default_factory=list)
    effort: str = Field(regex="^(low|medium|high)$", default="medium")
    priority: str = Field(regex="^(critical|high|medium|low)$", default="medium")


class BountyInfo(BaseModel):
    """Bug Bounty 信息"""
    eligible: bool = True
    estimated_value: Optional[str] = None
    program_relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    submission_ready: bool = False


class Finding(BaseModel):
    """單個漏洞發現"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    vulnerability_type: str
    severity: str = Field(regex="^(critical|high|medium|low|info)$")
    cvss_score: Optional[float] = Field(ge=0.0, le=10.0, default=None)
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    
    title: str
    description: str
    evidence: EvidenceData
    poc: Optional[PoCData] = None
    impact: ImpactAssessment
    remediation: RemediationAdvice
    bounty_info: Optional[BountyInfo] = None
    
    # 內部追蹤
    false_positive_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    verified: bool = False
    verification_notes: Optional[str] = None


class StatisticsData(BaseModel):
    """統計信息（內循環優化用）"""
    payloads_tested: int = 0
    requests_sent: int = 0
    false_positives_filtered: int = 0
    time_per_payload_ms: float = 0.0
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0)


class PerformanceMetrics(BaseModel):
    """性能指標（內循環優化用）"""
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    rate_limit_hits: int = 0
    retries: int = 0
    network_errors: int = 0
    timeout_count: int = 0


class ErrorInfo(BaseModel):
    """錯誤信息"""
    code: str
    message: str
    recoverable: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureResult(BaseModel):
    """Features 模組返回的完整結果"""
    task_id: str
    feature_module: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float
    
    status: str = Field(regex="^(completed|failed|timeout|partial)$")
    success: bool
    
    target: TargetInfo
    findings: List[Finding] = Field(default_factory=list)
    statistics: StatisticsData
    performance: PerformanceMetrics
    errors: List[ErrorInfo] = Field(default_factory=list)
    
    # 元數據
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationData(BaseModel):
    """內循環優化數據"""
    task_id: str
    feature_module: str
    
    # Payload 效率分析
    payload_efficiency: Dict[str, float] = Field(default_factory=dict)
    successful_patterns: List[str] = Field(default_factory=list)
    failed_patterns: List[str] = Field(default_factory=list)
    
    # 性能建議
    recommended_concurrency: Optional[int] = None
    recommended_timeout_ms: Optional[int] = None
    recommended_rate_limit: Optional[int] = None
    
    # 策略調整
    strategy_adjustments: Dict[str, Any] = Field(default_factory=dict)
    priority_adjustments: Dict[str, float] = Field(default_factory=dict)


class ReportData(BaseModel):
    """外循環報告數據"""
    task_id: str
    feature_module: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # 漏洞摘要
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    
    # 驗證狀態
    verified_findings: int = 0
    unverified_findings: int = 0
    false_positives: int = 0
    
    # Bug Bounty 相關
    bounty_eligible_count: int = 0
    estimated_total_value: Optional[str] = None
    
    # 詳細發現
    findings: List[Finding]
    
    # 合規性
    owasp_coverage: Dict[str, int] = Field(default_factory=dict)
    cwe_distribution: Dict[str, int] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """漏洞驗證結果"""
    finding_id: str
    verified: bool
    confidence: float = Field(ge=0.0, le=1.0)
    verification_method: str
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoreFeedback(BaseModel):
    """給 Core 的反饋"""
    task_id: str
    feature_module: str
    
    # 執行結果
    execution_success: bool
    findings_count: int
    high_value_findings: int
    
    # 優化建議
    optimization_suggestions: OptimizationData
    
    # 下一步建議
    recommended_next_actions: List[str] = Field(default_factory=list)
    continue_testing: bool = True
    
    # 學習數據
    learning_data: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Base Coordinator
# ============================================================================

class BaseCoordinator(ABC):
    """基礎協調器 - 實現雙閉環數據收集與處理
    
    職責：
    1. 收集 Features 執行結果
    2. 驗證數據格式和完整性
    3. 提取內循環優化數據
    4. 提取外循環報告數據
    5. 驗證漏洞真實性
    6. 生成回饋給 Core
    """
    
    def __init__(
        self,
        mq_client: Optional[Any] = None,
        db_client: Optional[Any] = None,
        cache_client: Optional[Any] = None,
        feature_module: str = "unknown"
    ):
        """初始化協調器
        
        Args:
            mq_client: 消息隊列客戶端（RabbitMQ/Redis）
            db_client: 數據庫客戶端（PostgreSQL/MongoDB）
            cache_client: 緩存客戶端（Redis）
            feature_module: 對應的 Feature 模組名稱
        """
        self.mq_client = mq_client
        self.db_client = db_client
        self.cache_client = cache_client
        self.feature_module = feature_module
        
        # 去重緩存（避免重複處理）
        self._processed_tasks: Set[str] = set()
        
        logger.info(f"Initialized {self.__class__.__name__} for {feature_module}")
    
    async def collect_result(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """收集並處理 Features 返回的結果
        
        Args:
            result_dict: Features 返回的原始結果字典
            
        Returns:
            處理後的完整數據，包含：
            - internal_loop: 內循環優化數據
            - external_loop: 外循環報告數據
            - verification: 漏洞驗證結果
            - feedback: 給 Core 的反饋
        """
        try:
            # 1. 驗證並解析結果
            result = await self._validate_result(result_dict)
            
            # 2. 檢查是否已處理（去重）
            if result.task_id in self._processed_tasks:
                logger.warning(f"Task {result.task_id} already processed, skipping")
                return {"status": "duplicate", "task_id": result.task_id}
            
            # 3. 存儲原始結果
            await self._store_raw_result(result)
            
            # 4. 提取內循環優化數據
            optimization_data = await self._extract_optimization_data(result)
            
            # 5. 提取外循環報告數據
            report_data = await self._extract_report_data(result)
            
            # 6. 驗證漏洞真實性
            verification_results = await self._verify_findings(result)
            
            # 7. 生成給 Core 的反饋
            feedback = await self._generate_feedback(
                result,
                optimization_data,
                verification_results
            )
            
            # 8. 發送回饋給 Core（通過 MQ）
            if self.mq_client:
                await self._send_feedback_to_core(feedback)
            
            # 9. 標記為已處理
            self._processed_tasks.add(result.task_id)
            
            logger.info(
                f"Successfully processed result for task {result.task_id}: "
                f"{len(result.findings)} findings, "
                f"{len(verification_results)} verified"
            )
            
            return {
                "status": "success",
                "task_id": result.task_id,
                "internal_loop": optimization_data.dict(),
                "external_loop": report_data.dict(),
                "verification": [v.dict() for v in verification_results],
                "feedback": feedback.dict(),
            }
            
        except Exception as e:
            logger.error(f"Error processing result: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "task_id": result_dict.get("task_id", "unknown"),
            }
    
    async def _validate_result(self, result_dict: Dict[str, Any]) -> FeatureResult:
        """驗證結果格式（使用 Pydantic）
        
        Args:
            result_dict: 原始結果字典
            
        Returns:
            驗證後的 FeatureResult 對象
            
        Raises:
            ValidationError: 如果格式不正確
        """
        try:
            return FeatureResult(**result_dict)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise
    
    async def _store_raw_result(self, result: FeatureResult) -> None:
        """存儲原始結果到數據庫
        
        Args:
            result: 驗證後的結果對象
        """
        if not self.db_client:
            logger.warning("No database client, skipping storage")
            return
        
        try:
            # 存儲到時序數據庫（性能指標）
            await self._store_performance_metrics(result)
            
            # 存儲到文檔數據庫（完整結果）
            await self._store_full_result(result)
            
            # 更新緩存（最近結果）
            if self.cache_client:
                await self._update_cache(result)
                
        except Exception as e:
            logger.error(f"Error storing result: {e}")
    
    @abstractmethod
    async def _extract_optimization_data(
        self, result: FeatureResult
    ) -> OptimizationData:
        """提取內循環優化數據（子類實現）
        
        分析：
        - Payload 效率
        - 性能瓶頸
        - 策略調整建議
        
        Args:
            result: Features 執行結果
            
        Returns:
            優化數據對象
        """
        pass
    
    @abstractmethod
    async def _extract_report_data(self, result: FeatureResult) -> ReportData:
        """提取外循環報告數據（子類實現）
        
        整理：
        - 漏洞摘要
        - 分類統計
        - Bug Bounty 信息
        
        Args:
            result: Features 執行結果
            
        Returns:
            報告數據對象
        """
        pass
    
    @abstractmethod
    async def _verify_findings(
        self, result: FeatureResult
    ) -> List[VerificationResult]:
        """驗證漏洞真實性（子類實現）
        
        檢查：
        - 證據完整性
        - 可重現性
        - 誤報過濾
        
        Args:
            result: Features 執行結果
            
        Returns:
            驗證結果列表
        """
        pass
    
    async def _generate_feedback(
        self,
        result: FeatureResult,
        optimization_data: OptimizationData,
        verification_results: List[VerificationResult],
    ) -> CoreFeedback:
        """生成給 Core 的反饋
        
        Args:
            result: 原始結果
            optimization_data: 優化數據
            verification_results: 驗證結果
            
        Returns:
            Core 反饋對象
        """
        verified_count = sum(1 for v in verification_results if v.verified)
        high_value_count = sum(
            1
            for f in result.findings
            if f.severity in ["critical", "high"] and f.verified
        )
        
        # 決定是否繼續測試
        continue_testing = (
            result.success
            and verified_count > 0
            and result.performance.rate_limit_hits == 0
        )
        
        # 生成下一步建議
        recommended_actions = await self._generate_next_actions(
            result, verified_count, high_value_count
        )
        
        return CoreFeedback(
            task_id=result.task_id,
            feature_module=result.feature_module,
            execution_success=result.success,
            findings_count=len(result.findings),
            high_value_findings=high_value_count,
            optimization_suggestions=optimization_data,
            recommended_next_actions=recommended_actions,
            continue_testing=continue_testing,
            learning_data=self._extract_learning_data(result),
        )
    
    async def _generate_next_actions(
        self, result: FeatureResult, verified_count: int, high_value_count: int
    ) -> List[str]:
        """生成下一步行動建議
        
        Args:
            result: 執行結果
            verified_count: 驗證通過的漏洞數
            high_value_count: 高價值漏洞數
            
        Returns:
            建議行動列表
        """
        actions = []
        
        if high_value_count > 0:
            actions.append("生成 Bug Bounty 報告")
            actions.append("進行深度利用鏈分析")
        
        if verified_count < len(result.findings):
            actions.append("人工驗證未確認的漏洞")
        
        if result.performance.rate_limit_hits > 0:
            actions.append("降低測試速率")
        
        if result.statistics.false_positives_filtered > 5:
            actions.append("調整檢測閾值")
        
        return actions
    
    def _extract_learning_data(self, result: FeatureResult) -> Dict[str, Any]:
        """提取機器學習訓練數據
        
        Args:
            result: 執行結果
            
        Returns:
            學習數據字典
        """
        return {
            "successful_payloads": [
                f.evidence.payload
                for f in result.findings
                if f.verified
            ],
            "failed_payloads": [],  # TODO: 從統計數據中提取
            "target_characteristics": {
                "response_patterns": [],
                "waf_detected": False,
                "framework_detected": result.metadata.get("framework", "unknown"),
            },
        }
    
    async def _send_feedback_to_core(self, feedback: CoreFeedback) -> None:
        """通過 MQ 發送反饋給 Core
        
        Args:
            feedback: Core 反饋對象
        """
        if not self.mq_client:
            return
        
        try:
            await self.mq_client.publish(
                topic=f"feedback.core.{self.feature_module}",
                payload=feedback.dict(),
            )
            logger.info(f"Sent feedback to Core for task {feedback.task_id}")
        except Exception as e:
            logger.error(f"Error sending feedback: {e}")
    
    # Placeholder methods for storage (to be implemented with actual clients)
    async def _store_performance_metrics(self, result: FeatureResult) -> None:
        """存儲性能指標到時序數據庫"""
        pass
    
    async def _store_full_result(self, result: FeatureResult) -> None:
        """存儲完整結果到文檔數據庫"""
        pass
    
    async def _update_cache(self, result: FeatureResult) -> None:
        """更新緩存"""
        pass
