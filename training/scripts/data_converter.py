"""數據轉換器 - 確保訓練數據與學習系統的格式一致性

架構原則：
- 外部訓練數據（SecurityScenario） → 標準化轉換 → 學習系統（TrainingDataSample）
- 在進入學習系統之前進行驗證和轉換
- 確保標準一致，符合 aiva_common 規範

數據流：
    training/data/distillation_train.json (SecurityScenario)
        → DataConverter.to_training_sample()
        → services/aiva_common/schemas/dual_loop.py (TrainingDataSample)
        → AI 學習系統
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, UTC
from dataclasses import dataclass

# aiva_common Schema 標準
from services.aiva_common.schemas.dual_loop import (
    TrainingDataSample,
    ExecutionPlan,
    ExecutionTrace,
    DeviationRecord,
    ExecutionStep
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityScenario:
    """訓練數據格式（來自 training/data/distillation_train.json）
    
    對應 generate_distillation_dataset.py 的 SecurityScenario
    """
    scenario_text: str              # 場景描述（給 Embedding 編碼）
    raw_context: str                # 原始上下文（HTTP 請求/響應等）
    teacher_vulnerability_type: str # 漏洞類型
    teacher_severity: float         # 嚴重性 (0-1)
    teacher_confidence: float       # 置信度 (0-1)
    teacher_reasoning: str          # 推理過程
    source_doc: str                 # 來源文檔
    scenario_id: str                # 唯一ID
    difficulty_level: str           # 難度：easy/medium/hard


class DataConverter:
    """數據轉換器
    
    功能：
    1. 驗證訓練數據格式（SecurityScenario）
    2. 轉換為學習系統標準格式（TrainingDataSample）
    3. 確保符合 aiva_common Pydantic 規範
    4. 提供雙向轉換和驗證
    """
    
    # 漏洞類型映射（統一命名）
    VULNERABILITY_TYPE_MAPPING = {
        "sql_injection": "sql_injection",
        "xss": "xss",
        "ssrf": "ssrf",
        "rce": "rce",
        "idor": "idor",
        "jwt_attack": "jwt_attack",
        "graphql_introspection": "graphql",
        "authentication": "authentication",
        "business_logic": "business_logic",
    }
    
    # 難度級別映射
    DIFFICULTY_MAPPING = {
        "easy": 1,
        "medium": 2,
        "hard": 3
    }
    
    @classmethod
    def validate_security_scenario(cls, data: Dict[str, Any]) -> bool:
        """驗證 SecurityScenario 格式
        
        Args:
            data: SecurityScenario 字典
            
        Returns:
            是否有效
        """
        required_fields = [
            "scenario_text",
            "raw_context",
            "teacher_vulnerability_type",
            "teacher_severity",
            "teacher_confidence",
            "teacher_reasoning",
            "source_doc",
            "scenario_id",
            "difficulty_level"
        ]
        
        # 檢查必填字段
        for field in required_fields:
            if field not in data:
                logger.error(f"❌ 缺少必填字段: {field}")
                return False
        
        # 檢查數值範圍
        if not (0.0 <= data["teacher_severity"] <= 1.0):
            logger.error(f"❌ teacher_severity 超出範圍 [0, 1]: {data['teacher_severity']}")
            return False
        
        if not (0.0 <= data["teacher_confidence"] <= 1.0):
            logger.error(f"❌ teacher_confidence 超出範圍 [0, 1]: {data['teacher_confidence']}")
            return False
        
        # 檢查漏洞類型
        vuln_type = data["teacher_vulnerability_type"]
        if vuln_type not in cls.VULNERABILITY_TYPE_MAPPING:
            logger.warning(f"⚠️  未知漏洞類型: {vuln_type}（將保持原值）")
        
        # 檢查難度級別
        difficulty = data["difficulty_level"]
        if difficulty not in cls.DIFFICULTY_MAPPING:
            logger.warning(f"⚠️  未知難度級別: {difficulty}（將保持原值）")
        
        return True
    
    @classmethod
    def to_training_sample(
        cls, 
        scenario: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> TrainingDataSample:
        """將 SecurityScenario 轉換為 TrainingDataSample
        
        轉換邏輯：
        1. 驗證輸入格式
        2. 構建 ExecutionPlan（從 scenario_text 生成）
        3. 構建 ExecutionTrace（從 raw_context 生成）
        4. 構建 DeviationRecord（從 teacher_* 字段生成）
        5. 返回標準 TrainingDataSample
        
        Args:
            scenario: SecurityScenario 字典
            execution_id: 可選的執行ID（用於追蹤）
            
        Returns:
            TrainingDataSample 實例
            
        Raises:
            ValueError: 如果數據格式無效
        """
        # 1. 驗證輸入
        if not cls.validate_security_scenario(scenario):
            raise ValueError(f"無效的 SecurityScenario 格式: {scenario.get('scenario_id', 'unknown')}")
        
        scenario_id = scenario["scenario_id"]
        execution_id = execution_id or f"train_{scenario_id}"
        
        logger.debug(f"🔄 轉換 SecurityScenario → TrainingDataSample: {scenario_id}")
        
        # 2. 構建 ExecutionPlan（模擬計劃）
        plan = ExecutionPlan(
            plan_id=execution_id,
            objective=scenario["scenario_text"],  # 場景描述作為目標
            steps=[
                ExecutionStep(
                    step_id=f"{execution_id}_step1",
                    action="analyze_vulnerability",
                    parameters={
                        "vulnerability_type": scenario["teacher_vulnerability_type"],
                        "context": scenario["raw_context"],
                        "source": scenario["source_doc"]
                    },
                    expected_result="vulnerability_identified"
                )
            ],
            created_at=datetime.now(UTC)
        )
        
        # 3. 構建 ExecutionTrace（模擬執行軌跡）
        trace = [
            ExecutionTrace(
                trace_id=f"{execution_id}_trace1",
                step_id=f"{execution_id}_step1",
                action="analyze_vulnerability",
                parameters={
                    "vulnerability_type": scenario["teacher_vulnerability_type"],
                    "severity": scenario["teacher_severity"],
                    "confidence": scenario["teacher_confidence"]
                },
                result={
                    "vulnerability_detected": True,
                    "vulnerability_type": scenario["teacher_vulnerability_type"],
                    "reasoning": scenario["teacher_reasoning"]
                },
                status="completed",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC)
            )
        ]
        
        # 4. 構建 DeviationRecord（Teacher Model 的軟標籤）
        deviations = [
            DeviationRecord(
                deviation_id=f"{scenario_id}_dev1",
                planned_action="detect_vulnerability",
                actual_action="vulnerability_analyzed",
                deviation_type="teacher_label",
                severity_impact=scenario["teacher_severity"],
                description=scenario["teacher_reasoning"],
                cause={
                    "vulnerability_type": scenario["teacher_vulnerability_type"],
                    "confidence": scenario["teacher_confidence"],
                    "difficulty": scenario["difficulty_level"]
                },
                timestamp=datetime.now(UTC)
            )
        ]
        
        # 5. 確定結果類型（基於 Teacher 的置信度）
        confidence = scenario["teacher_confidence"]
        if confidence >= 0.8:
            outcome = "success"
        elif confidence >= 0.5:
            outcome = "partial_success"
        else:
            outcome = "failure"
        
        # 6. 創建 TrainingDataSample（符合 aiva_common 規範）
        training_sample = TrainingDataSample(
            sample_id=scenario_id,
            plan=plan,
            trace=trace,
            deviations=deviations,
            outcome=outcome,
            created_at=datetime.now(UTC)
        )
        
        logger.debug(f"✅ 轉換成功: {scenario_id} → {training_sample.sample_id}")
        
        return training_sample
    
    @classmethod
    def from_training_sample(
        cls,
        sample: TrainingDataSample
    ) -> Dict[str, Any]:
        """將 TrainingDataSample 反向轉換為 SecurityScenario 格式
        
        用途：
        - 驗證轉換的正確性
        - 導出訓練數據為原始格式
        
        Args:
            sample: TrainingDataSample 實例
            
        Returns:
            SecurityScenario 字典
        """
        # 從 ExecutionPlan 提取信息
        scenario_text = sample.plan.objective
        
        # 從 ExecutionTrace 提取信息
        first_trace = sample.trace[0] if sample.trace else None
        raw_context = str(first_trace.parameters) if first_trace else ""
        
        # 從 DeviationRecord 提取 Teacher 標籤
        first_deviation = sample.deviations[0] if sample.deviations else None
        
        if first_deviation and first_deviation.cause:
            teacher_vulnerability_type = first_deviation.cause.get("vulnerability_type", "unknown")
            teacher_confidence = first_deviation.cause.get("confidence", 0.5)
            difficulty_level = first_deviation.cause.get("difficulty", "medium")
        else:
            teacher_vulnerability_type = "unknown"
            teacher_confidence = 0.5
            difficulty_level = "medium"
        
        teacher_severity = first_deviation.severity_impact if first_deviation else 0.5
        teacher_reasoning = first_deviation.description if first_deviation else ""
        
        # 構建 SecurityScenario
        scenario = {
            "scenario_text": scenario_text,
            "raw_context": raw_context,
            "teacher_vulnerability_type": teacher_vulnerability_type,
            "teacher_severity": teacher_severity,
            "teacher_confidence": teacher_confidence,
            "teacher_reasoning": teacher_reasoning,
            "source_doc": "converted_from_training_sample",
            "scenario_id": sample.sample_id,
            "difficulty_level": difficulty_level
        }
        
        return scenario
    
    @classmethod
    def batch_convert(
        cls,
        scenarios: list[Dict[str, Any]]
    ) -> tuple[list[TrainingDataSample], list[str]]:
        """批量轉換 SecurityScenario
        
        Args:
            scenarios: SecurityScenario 字典列表
            
        Returns:
            (成功轉換的樣本列表, 失敗的 scenario_id 列表)
        """
        converted = []
        failed_ids = []
        
        for i, scenario in enumerate(scenarios):
            try:
                sample = cls.to_training_sample(scenario)
                converted.append(sample)
            except Exception as e:
                scenario_id = scenario.get("scenario_id", f"unknown_{i}")
                logger.error(f"❌ 轉換失敗 [{scenario_id}]: {e}")
                failed_ids.append(scenario_id)
        
        success_rate = len(converted) / len(scenarios) * 100 if scenarios else 0
        logger.info(f"📊 批量轉換完成: {len(converted)}/{len(scenarios)} ({success_rate:.1f}%)")
        
        if failed_ids:
            logger.warning(f"⚠️  失敗的樣本: {', '.join(failed_ids)}")
        
        return converted, failed_ids
    
    @classmethod
    def validate_conversion(
        cls,
        original: Dict[str, Any],
        converted: TrainingDataSample
    ) -> bool:
        """驗證轉換的正確性
        
        Args:
            original: 原始 SecurityScenario
            converted: 轉換後的 TrainingDataSample
            
        Returns:
            轉換是否保持了關鍵信息
        """
        # 檢查 ID 一致性
        if original["scenario_id"] != converted.sample_id:
            logger.error(f"❌ ID 不一致: {original['scenario_id']} vs {converted.sample_id}")
            return False
        
        # 檢查場景描述
        if original["scenario_text"] != converted.plan.objective:
            logger.error(f"❌ scenario_text 不一致")
            return False
        
        # 檢查漏洞類型（從 DeviationRecord 提取）
        if converted.deviations:
            deviation_vuln = converted.deviations[0].cause.get("vulnerability_type")
            if original["teacher_vulnerability_type"] != deviation_vuln:
                logger.error(f"❌ 漏洞類型不一致: {original['teacher_vulnerability_type']} vs {deviation_vuln}")
                return False
        
        logger.debug(f"✅ 轉換驗證通過: {original['scenario_id']}")
        return True
