"""
AIVA 能力評估器模組
實現訓練探索和學習反饋機制
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from pathlib import Path

from services.aiva_common.enums import (
    ModuleName,
    Severity,
    TaskStatus,
    ProgrammingLanguage
)
from services.aiva_common.schemas import (
    CapabilityInfo,
    CapabilityScorecard
)
from services.aiva_common.utils.logging import get_logger
from services.integration.capability import CapabilityRegistry

logger = get_logger(__name__)


@dataclass
class LearningSession:
    """學習會話"""
    session_id: str
    capability_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    feedback_score: Optional[float] = None  # 1-5 分
    user_feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityInsight:
    """能力洞察"""
    capability_id: str
    name: str
    language: ProgrammingLanguage
    
    # 性能指標
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    
    # 使用模式
    peak_usage_hours: List[int] = field(default_factory=list)
    common_inputs: Dict[str, int] = field(default_factory=dict)
    frequent_errors: Dict[str, int] = field(default_factory=dict)
    
    # 學習指標
    improvement_trend: float = 0.0  # 改進趨勢 (-1 到 1)
    learning_velocity: float = 0.0  # 學習速度
    confidence_score: float = 0.0   # 信心分數
    
    # 推薦
    optimization_suggestions: List[str] = field(default_factory=list)
    training_recommendations: List[str] = field(default_factory=list)
    
    last_analysis: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingPlan:
    """訓練計劃"""
    plan_id: str
    capability_id: str
    objectives: List[str]
    training_data: List[Dict[str, Any]]
    validation_criteria: Dict[str, Any]
    expected_improvements: Dict[str, float]
    estimated_duration_hours: float
    priority: int = 1  # 1-5, 5最高
    status: str = "planned"  # planned, running, completed, failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CapabilityPerformanceTracker:
    """能力性能追蹤器"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, LearningSession] = {}
        self.session_history: List[LearningSession] = []
        
        # 初始化標記，稍後異步載入歷史數據
        self._history_loaded = False
    
    async def start_session(
        self, 
        capability_id: str, 
        inputs: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """開始學習會話"""
        session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{capability_id[:8]}"
        
        session = LearningSession(
            session_id=session_id,
            capability_id=capability_id,
            start_time=datetime.utcnow(),
            inputs=inputs.copy()
        )
        
        self.sessions[session_id] = session
        logger.info(f"學習會話開始: {session_id} for {capability_id}")
        
        return session_id
    
    async def end_session(
        self,
        session_id: str,
        outputs: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None,
        execution_time_ms: float = 0.0,
        memory_usage_mb: float = 0.0
    ) -> None:
        """結束學習會話"""
        if session_id not in self.sessions:
            logger.warning(f"找不到學習會話: {session_id}")
            return
        
        session = self.sessions[session_id]
        session.end_time = datetime.utcnow()
        session.outputs = outputs.copy()
        session.success = success
        session.error_message = error_message
        session.execution_time_ms = execution_time_ms
        session.memory_usage_mb = memory_usage_mb
        
        # 移到歷史記錄
        self.session_history.append(session)
        del self.sessions[session_id]
        
        # 保存到磁碟
        await self._save_session(session)
        
        logger.info(f"學習會話結束: {session_id}, 成功: {success}")
    
    async def add_feedback(
        self,
        session_id: str,
        feedback_score: float,
        user_feedback: Optional[str] = None
    ) -> None:
        """添加用戶反饋"""
        # 查找會話（可能在進行中或歷史記錄）
        session = None
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            for hist_session in self.session_history:
                if hist_session.session_id == session_id:
                    session = hist_session
                    break
        
        if not session:
            logger.warning(f"找不到會話: {session_id}")
            return
        
        session.feedback_score = max(1.0, min(5.0, feedback_score))  # 限制在1-5範圍
        session.user_feedback = user_feedback
        
        # 如果是歷史會話，重新保存
        if session_id not in self.sessions:
            await self._save_session(session)
        
        logger.info(f"反饋已添加到會話 {session_id}: {feedback_score}/5")
    
    async def get_capability_sessions(
        self,
        capability_id: str,
        limit: int = 100,
        days: int = 30
    ) -> List[LearningSession]:
        """獲取能力的學習會話"""
        # 確保歷史數據已載入
        if not self._history_loaded:
            await self._load_session_history()
            self._history_loaded = True
            
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        sessions = [
            session for session in self.session_history
            if session.capability_id == capability_id 
            and session.start_time >= cutoff_date
        ]
        
        # 按時間倒序排列
        sessions.sort(key=lambda s: s.start_time, reverse=True)
        return sessions[:limit]
    
    async def _load_session_history(self) -> None:
        """載入會話歷史"""
        try:
            history_file = self.data_dir / "session_history.jsonl"
            if not history_file.exists():
                return
            
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        session = self._dict_to_session(data)
                        self.session_history.append(session)
                    except Exception as e:
                        logger.warning(f"載入會話記錄失敗: {e}")
            
            logger.info(f"載入了 {len(self.session_history)} 個學習會話")
            
        except Exception as e:
            logger.error(f"載入會話歷史失敗: {e}")
    
    async def _save_session(self, session: LearningSession) -> None:
        """保存會話到磁碟"""
        try:
            history_file = self.data_dir / "session_history.jsonl"
            
            with open(history_file, 'a', encoding='utf-8') as f:
                data = self._session_to_dict(session)
                f.write(json.dumps(data, ensure_ascii=False, default=str) + '\n')
        
        except Exception as e:
            logger.error(f"保存會話失敗: {e}")
    
    def _session_to_dict(self, session: LearningSession) -> Dict[str, Any]:
        """轉換會話為字典"""
        return {
            "session_id": session.session_id,
            "capability_id": session.capability_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "inputs": session.inputs,
            "outputs": session.outputs,
            "success": session.success,
            "error_message": session.error_message,
            "execution_time_ms": session.execution_time_ms,
            "memory_usage_mb": session.memory_usage_mb,
            "feedback_score": session.feedback_score,
            "user_feedback": session.user_feedback,
            "metadata": session.metadata
        }
    
    def _dict_to_session(self, data: Dict[str, Any]) -> LearningSession:
        """從字典創建會話"""
        return LearningSession(
            session_id=data["session_id"],
            capability_id=data["capability_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            success=data.get("success", False),
            error_message=data.get("error_message"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            memory_usage_mb=data.get("memory_usage_mb", 0.0),
            feedback_score=data.get("feedback_score"),
            user_feedback=data.get("user_feedback"),
            metadata=data.get("metadata", {})
        )


class CapabilityInsightAnalyzer:
    """能力洞察分析器"""
    
    def __init__(self, performance_tracker: CapabilityPerformanceTracker):
        self.performance_tracker = performance_tracker
    
    async def analyze_capability(
        self,
        capability_id: str,
        capability_info: CapabilityInfo,
        days: int = 30
    ) -> CapabilityInsight:
        """分析能力並生成洞察"""
        sessions = await self.performance_tracker.get_capability_sessions(
            capability_id, limit=1000, days=days
        )
        
        if not sessions:
            return CapabilityInsight(
                capability_id=capability_id,
                name=capability_info.name,
                language=capability_info.language
            )
        
        insight = CapabilityInsight(
            capability_id=capability_id,
            name=capability_info.name,
            language=capability_info.language
        )
        
        # 基本性能指標
        await self._analyze_performance_metrics(insight, sessions)
        
        # 使用模式分析
        await self._analyze_usage_patterns(insight, sessions)
        
        # 學習指標分析
        await self._analyze_learning_metrics(insight, sessions)
        
        # 生成建議
        await self._generate_recommendations(insight, sessions)
        
        insight.last_analysis = datetime.utcnow()
        
        return insight
    
    async def _analyze_performance_metrics(
        self,
        insight: CapabilityInsight,
        sessions: List[LearningSession]
    ) -> None:
        """分析性能指標"""
        if not sessions:
            return
        
        # 成功率
        successful_sessions = [s for s in sessions if s.success]
        insight.success_rate = len(successful_sessions) / len(sessions)
        insight.error_rate = 1.0 - insight.success_rate
        
        # 執行時間
        execution_times = [s.execution_time_ms for s in sessions if s.execution_time_ms > 0]
        if execution_times:
            insight.avg_execution_time = statistics.mean(execution_times)
    
    async def _analyze_usage_patterns(
        self,
        insight: CapabilityInsight,
        sessions: List[LearningSession]
    ) -> None:
        """分析使用模式"""
        if not sessions:
            return
        
        # 高峰使用時間
        usage_hours = defaultdict(int)
        for session in sessions:
            hour = session.start_time.hour
            usage_hours[hour] += 1
        
        if usage_hours:
            # 找出使用量前3的小時
            sorted_hours = sorted(usage_hours.items(), key=lambda x: x[1], reverse=True)
            insight.peak_usage_hours = [hour for hour, _ in sorted_hours[:3]]
        
        # 常見輸入
        input_patterns = defaultdict(int)
        for session in sessions:
            for key, value in session.inputs.items():
                if isinstance(value, (str, int, float, bool)):
                    pattern = f"{key}:{str(value)[:50]}"  # 限制長度
                    input_patterns[pattern] += 1
        
        # 保留前10個最常見的輸入模式
        insight.common_inputs = dict(
            sorted(input_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # 頻繁錯誤
        error_patterns = defaultdict(int)
        for session in sessions:
            if session.error_message:
                # 提取錯誤類型（取錯誤消息的前50字符）
                error_type = session.error_message[:50]
                error_patterns[error_type] += 1
        
        insight.frequent_errors = dict(
            sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        )
    
    async def _analyze_learning_metrics(
        self,
        insight: CapabilityInsight,
        sessions: List[LearningSession]
    ) -> None:
        """分析學習指標"""
        if len(sessions) < 5:  # 需要足夠的數據
            return
        
        # 改進趨勢分析（最近與早期的成功率比較）
        sessions_sorted = sorted(sessions, key=lambda s: s.start_time)
        total_sessions = len(sessions_sorted)
        
        # 取前25%和後25%進行比較
        early_count = max(1, total_sessions // 4)
        recent_count = max(1, total_sessions // 4)
        
        early_sessions = sessions_sorted[:early_count]
        recent_sessions = sessions_sorted[-recent_count:]
        
        early_success_rate = sum(1 for s in early_sessions if s.success) / len(early_sessions)
        recent_success_rate = sum(1 for s in recent_sessions if s.success) / len(recent_sessions)
        
        insight.improvement_trend = recent_success_rate - early_success_rate
        
        # 學習速度（基於執行時間的改進）
        early_times = [s.execution_time_ms for s in early_sessions if s.execution_time_ms > 0]
        recent_times = [s.execution_time_ms for s in recent_sessions if s.execution_time_ms > 0]
        
        if early_times and recent_times:
            early_avg_time = statistics.mean(early_times)
            recent_avg_time = statistics.mean(recent_times)
            time_improvement = (early_avg_time - recent_avg_time) / early_avg_time
            insight.learning_velocity = max(0, time_improvement)
        
        # 信心分數（基於成功率和用戶反饋）
        base_confidence = insight.success_rate
        
        # 加入用戶反饋因子
        feedback_sessions = [s for s in sessions if s.feedback_score is not None]
        if feedback_sessions:
            avg_feedback = statistics.mean([s.feedback_score for s in feedback_sessions]) / 5.0
            insight.confidence_score = (base_confidence * 0.7) + (avg_feedback * 0.3)
        else:
            insight.confidence_score = base_confidence
    
    async def _generate_recommendations(
        self,
        insight: CapabilityInsight,
        sessions: List[LearningSession]
    ) -> None:
        """生成優化建議"""
        # 性能優化建議
        if insight.avg_execution_time > 10000:  # 超過10秒
            insight.optimization_suggestions.append("考慮優化執行時間，目前平均執行時間較長")
        
        if insight.error_rate > 0.2:  # 錯誤率超過20%
            insight.optimization_suggestions.append("錯誤率較高，建議檢查錯誤處理機制")
        
        if insight.improvement_trend < -0.1:  # 性能下降超過10%
            insight.optimization_suggestions.append("性能呈下降趨勢，建議進行深度分析")
        
        # 訓練建議
        if len(sessions) < 20:
            insight.training_recommendations.append("使用數據不足，建議增加測試案例")
        
        if insight.confidence_score < 0.7:
            insight.training_recommendations.append("信心分數偏低，建議收集更多用戶反饋")
        
        if insight.frequent_errors:
            most_common_error = max(insight.frequent_errors.items(), key=lambda x: x[1])
            insight.training_recommendations.append(
                f"針對常見錯誤進行訓練：{most_common_error[0][:30]}..."
            )
        
        if not insight.optimization_suggestions:
            insight.optimization_suggestions.append("性能表現良好，繼續保持")
        
        if not insight.training_recommendations:
            insight.training_recommendations.append("當前訓練數據充足，可考慮探索新的使用場景")


class TrainingPlanGenerator:
    """訓練計劃生成器"""
    
    def __init__(
        self,
        insight_analyzer: CapabilityInsightAnalyzer,
        data_dir: Optional[Path] = None
    ):
        self.insight_analyzer = insight_analyzer
        self.data_dir = data_dir or Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_plans: Dict[str, TrainingPlan] = {}
    
    async def generate_training_plan(
        self,
        capability_id: str,
        capability_info: CapabilityInfo,
        insight: CapabilityInsight
    ) -> TrainingPlan:
        """生成訓練計劃"""
        plan_id = f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{capability_id[:8]}"
        
        # 根據洞察生成訓練目標
        objectives = self._generate_objectives(insight)
        
        # 生成訓練數據
        training_data = await self._generate_training_data(capability_info, insight)
        
        # 設定驗證標準
        validation_criteria = self._generate_validation_criteria(insight)
        
        # 預期改進
        expected_improvements = self._generate_expected_improvements(insight)
        
        # 估算訓練時間
        estimated_duration = self._estimate_training_duration(capability_info, insight)
        
        # 設定優先級
        priority = self._calculate_priority(insight)
        
        plan = TrainingPlan(
            plan_id=plan_id,
            capability_id=capability_id,
            objectives=objectives,
            training_data=training_data,
            validation_criteria=validation_criteria,
            expected_improvements=expected_improvements,
            estimated_duration_hours=estimated_duration,
            priority=priority
        )
        
        self.training_plans[plan_id] = plan
        await self._save_training_plan(plan)
        
        logger.info(f"訓練計劃已生成: {plan_id} for {capability_id}")
        
        return plan
    
    def _generate_objectives(self, insight: CapabilityInsight) -> List[str]:
        """生成訓練目標"""
        objectives = []
        
        if insight.success_rate < 0.8:
            objectives.append(f"提高成功率至80%以上（當前：{insight.success_rate:.1%}）")
        
        if insight.avg_execution_time > 5000:
            objectives.append(f"優化執行時間至5秒以內（當前：{insight.avg_execution_time:.0f}ms）")
        
        if insight.improvement_trend < 0:
            objectives.append("扭轉性能下降趨勢，恢復穩定表現")
        
        if insight.confidence_score < 0.7:
            objectives.append(f"提升信心分數至70%以上（當前：{insight.confidence_score:.1%}）")
        
        if insight.frequent_errors:
            objectives.append("減少常見錯誤的發生頻率")
        
        if not objectives:
            objectives.append("保持當前性能水準並探索優化空間")
        
        return objectives[:5]  # 限制最多5個目標
    
    async def _generate_training_data(
        self,
        capability_info: CapabilityInfo,
        insight: CapabilityInsight
    ) -> List[Dict[str, Any]]:
        """生成訓練數據"""
        training_data = []
        
        # 基於常見輸入模式生成測試數據
        for input_pattern, count in insight.common_inputs.items():
            if ":" in input_pattern:
                key, value = input_pattern.split(":", 1)
                
                # 生成變化的測試數據
                for i in range(min(3, count)):  # 每個模式最多3個變體
                    test_case = {
                        "input": {key: self._generate_variant(value)},
                        "expected_success": True,
                        "weight": count / sum(insight.common_inputs.values()),
                        "description": f"基於常見模式的測試案例: {key}"
                    }
                    training_data.append(test_case)
        
        # 基於錯誤模式生成修復測試
        for error_msg, count in insight.frequent_errors.items():
            test_case = {
                "input": {"error_scenario": error_msg[:30]},
                "expected_success": True,
                "weight": 0.8,  # 錯誤修復很重要
                "description": f"錯誤修復測試: {error_msg[:50]}..."
            }
            training_data.append(test_case)
        
        # 如果沒有足夠的數據，生成基本測試案例
        if len(training_data) < 5:
            basic_cases = self._generate_basic_test_cases(capability_info)
            training_data.extend(basic_cases)
        
        return training_data[:20]  # 限制最多20個訓練案例
    
    def _generate_variant(self, base_value: str) -> str:
        """生成輸入值的變體"""
        # 簡單的變體生成邏輯
        if base_value.startswith("http"):
            # URL 變體
            return base_value.replace("http://", "https://") if "http://" in base_value else base_value
        elif base_value.isdigit():
            # 數字變體
            return str(int(base_value) + 1)
        else:
            # 字符串變體
            return base_value + "_variant"
    
    def _generate_basic_test_cases(self, capability_info: CapabilityInfo) -> List[Dict[str, Any]]:
        """生成基本測試案例"""
        test_cases = []
        
        # 基於能力的輸入參數生成測試
        if capability_info.inputs:
            for inp in capability_info.inputs[:3]:  # 最多3個輸入參數
                test_value = self._generate_test_value(inp.type, inp.name)
                
                test_case = {
                    "input": {inp.name: test_value},
                    "expected_success": True,
                    "weight": 0.5,
                    "description": f"基本測試: {inp.name} ({inp.type})"
                }
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_test_value(self, param_type: str, param_name: str) -> Any:
        """根據參數類型生成測試值"""
        param_type = param_type.lower()
        param_name = param_name.lower()
        
        if "url" in param_name or param_type == "url":
            return "https://example.com"
        elif "string" in param_type or "str" in param_type:
            return "test_value"
        elif "int" in param_type or "number" in param_type:
            return 123
        elif "bool" in param_type:
            return True
        elif "list" in param_type or "array" in param_type:
            return ["item1", "item2"]
        elif "dict" in param_type or "object" in param_type:
            return {"key": "value"}
        else:
            return "default_test_value"
    
    def _generate_validation_criteria(self, insight: CapabilityInsight) -> Dict[str, Any]:
        """生成驗證標準"""
        return {
            "min_success_rate": max(0.8, insight.success_rate + 0.1),
            "max_execution_time_ms": insight.avg_execution_time * 0.9 if insight.avg_execution_time > 0 else 5000,
            "min_confidence_score": max(0.7, insight.confidence_score + 0.1),
            "max_error_rate": max(0.1, insight.error_rate - 0.05),
            "required_test_cases": 10,
            "validation_period_days": 7
        }
    
    def _generate_expected_improvements(self, insight: CapabilityInsight) -> Dict[str, float]:
        """生成預期改進"""
        improvements = {}
        
        if insight.success_rate < 0.8:
            improvements["success_rate"] = min(0.9, insight.success_rate + 0.15)
        
        if insight.avg_execution_time > 5000:
            improvements["execution_time_reduction"] = 0.2  # 減少20%
        
        if insight.confidence_score < 0.7:
            improvements["confidence_score"] = min(0.85, insight.confidence_score + 0.15)
        
        if insight.error_rate > 0.1:
            improvements["error_rate_reduction"] = 0.5  # 減少50%
        
        return improvements
    
    def _estimate_training_duration(
        self,
        capability_info: CapabilityInfo,
        insight: CapabilityInsight
    ) -> float:
        """估算訓練時間（小時）"""
        base_hours = 2.0  # 基礎時間
        
        # 根據能力複雜度調整
        if capability_info.inputs and len(capability_info.inputs) > 3:
            base_hours += 1.0
        
        # 根據當前性能調整
        if insight.success_rate < 0.5:
            base_hours += 2.0  # 性能較差需要更多時間
        
        if insight.frequent_errors:
            base_hours += len(insight.frequent_errors) * 0.5
        
        # 根據語言調整
        if capability_info.language == ProgrammingLanguage.PYTHON:
            base_hours *= 1.0  # Python 基準
        elif capability_info.language == ProgrammingLanguage.GO:
            base_hours *= 1.2  # Go 稍微複雜
        else:
            base_hours *= 1.3  # 其他語言
        
        return min(24.0, base_hours)  # 最多24小時
    
    def _calculate_priority(self, insight: CapabilityInsight) -> int:
        """計算訓練優先級（1-5，5最高）"""
        score = 3  # 基礎分數
        
        # 成功率影響
        if insight.success_rate < 0.5:
            score += 2
        elif insight.success_rate < 0.7:
            score += 1
        
        # 改進趨勢影響
        if insight.improvement_trend < -0.1:
            score += 1
        
        # 錯誤率影響
        if insight.error_rate > 0.3:
            score += 1
        
        # 信心分數影響
        if insight.confidence_score < 0.5:
            score += 1
        
        return min(5, max(1, score))
    
    async def _save_training_plan(self, plan: TrainingPlan) -> None:
        """保存訓練計劃"""
        try:
            plan_file = self.data_dir / f"{plan.plan_id}.json"
            
            plan_data = {
                "plan_id": plan.plan_id,
                "capability_id": plan.capability_id,
                "objectives": plan.objectives,
                "training_data": plan.training_data,
                "validation_criteria": plan.validation_criteria,
                "expected_improvements": plan.expected_improvements,
                "estimated_duration_hours": plan.estimated_duration_hours,
                "priority": plan.priority,
                "status": plan.status,
                "created_at": plan.created_at.isoformat(),
                "started_at": plan.started_at.isoformat() if plan.started_at else None,
                "completed_at": plan.completed_at.isoformat() if plan.completed_at else None
            }
            
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logger.error(f"無法保存訓練計劃: {e}")


class AIVACapabilityEvaluator:
    """
    AIVA 能力評估器主類
    
    功能:
    - 性能追蹤和分析
    - 能力洞察生成
    - 訓練計劃制定
    - 學習反饋集成
    """
    
    def __init__(
        self,
        capability_registry: Optional[CapabilityRegistry] = None,
        data_dir: Optional[Path] = None
    ):
        self.capability_registry = capability_registry or CapabilityRegistry()
        self.data_dir = data_dir or Path("data/learning")
        
        self.performance_tracker = CapabilityPerformanceTracker(self.data_dir / "performance")
        self.insight_analyzer = CapabilityInsightAnalyzer(self.performance_tracker)
        self.training_plan_generator = TrainingPlanGenerator(
            self.insight_analyzer,
            self.data_dir / "training"
        )
        
        logger.info("AIVA 能力評估器已初始化")
    
    async def start_evaluation_session(
        self,
        capability_id: str,
        inputs: Dict[str, Any]
    ) -> str:
        """開始評估會話"""
        return await self.performance_tracker.start_session(capability_id, inputs)
    
    async def end_evaluation_session(
        self,
        session_id: str,
        outputs: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None,
        execution_time_ms: float = 0.0,
        memory_usage_mb: float = 0.0
    ) -> None:
        """結束評估會話"""
        await self.performance_tracker.end_session(
            session_id, outputs, success, error_message, execution_time_ms, memory_usage_mb
        )
    
    async def add_user_feedback(
        self,
        session_id: str,
        feedback_score: float,
        user_feedback: Optional[str] = None
    ) -> None:
        """添加用戶反饋"""
        await self.performance_tracker.add_feedback(session_id, feedback_score, user_feedback)
    
    async def analyze_capability(
        self,
        capability_id: str,
        days: int = 30
    ) -> CapabilityInsight:
        """分析能力並生成洞察"""
        capabilities = await self.capability_registry.search_capabilities(capability_id)
        
        if not capabilities:
            raise ValueError(f"找不到能力: {capability_id}")
        
        capability_info = capabilities[0]
        
        return await self.insight_analyzer.analyze_capability(
            capability_id, capability_info, days
        )
    
    async def generate_training_plan(
        self,
        capability_id: str
    ) -> TrainingPlan:
        """生成訓練計劃"""
        # 獲取能力信息
        capabilities = await self.capability_registry.search_capabilities(capability_id)
        
        if not capabilities:
            raise ValueError(f"找不到能力: {capability_id}")
        
        capability_info = capabilities[0]
        
        # 分析能力
        insight = await self.analyze_capability(capability_id)
        
        # 生成訓練計劃
        return await self.training_plan_generator.generate_training_plan(
            capability_id, capability_info, insight
        )
    
    async def get_capability_insights(
        self,
        limit: int = 10,
        days: int = 30
    ) -> List[CapabilityInsight]:
        """獲取多個能力的洞察"""
        insights = []
        
        try:
            # 獲取所有能力
            capabilities = await self.capability_registry.list_capabilities(limit=limit)
            
            # 並行分析
            tasks = [
                self.analyze_capability(cap.id, days)
                for cap in capabilities
            ]
            
            insights = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 過濾出成功的結果
            valid_insights = [
                insight for insight in insights
                if isinstance(insight, CapabilityInsight)
            ]
            
            # 按信心分數排序
            valid_insights.sort(key=lambda i: i.confidence_score, reverse=True)
            
            return valid_insights
        
        except Exception as e:
            logger.error(f"獲取能力洞察失敗: {e}")
            return insights
    
    async def get_top_training_priorities(self, limit: int = 5) -> List[TrainingPlan]:
        """獲取最高優先級的訓練計劃"""
        plans = list(self.training_plan_generator.training_plans.values())
        
        # 按優先級和創建時間排序
        plans.sort(key=lambda p: (p.priority, p.created_at), reverse=True)
        
        return plans[:limit]
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """獲取評估統計信息"""
        total_sessions = len(self.performance_tracker.session_history)
        successful_sessions = sum(1 for s in self.performance_tracker.session_history if s.success)
        
        with_feedback = sum(
            1 for s in self.performance_tracker.session_history
            if s.feedback_score is not None
        )
        
        avg_feedback = 0.0
        if with_feedback > 0:
            avg_feedback = statistics.mean([
                s.feedback_score for s in self.performance_tracker.session_history
                if s.feedback_score is not None
            ])
        
        return {
            "total_sessions": total_sessions,
            "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0.0,
            "sessions_with_feedback": with_feedback,
            "average_feedback_score": avg_feedback,
            "active_training_plans": len(self.training_plan_generator.training_plans),
            "data_directory": str(self.data_dir)
        }


# 創建全域能力評估器實例
capability_evaluator = AIVACapabilityEvaluator()