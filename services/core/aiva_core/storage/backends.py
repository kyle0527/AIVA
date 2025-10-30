"""
數據存儲後端實現

支持多種存儲後端：
- SQLite: 輕量級，適合開發和小規模部署
- PostgreSQL: 生產級，適合大規模部署
- JSONL: 文件格式，適合導出和分析
- Hybrid: 混合存儲，數據庫 + 文件
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, List, Dict, Union


from sqlalchemy import create_engine, desc, func
# Removed unused declarative_base import
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    ExperienceSampleModel,
    KnowledgeEntryModel,
    ModelCheckpointModel,
    TraceRecordModel,
    TrainingSessionModel,
)

# 匯入資料結構 - 使用動態匯入避免循環依賴
# Removed unused TYPE_CHECKING import

# 動態匯入 - 添加路徑到 sys.path
import sys
from pathlib import Path

# 初始化 logger
logger = logging.getLogger(__name__)

# 使用現實路徑導入 - 遵循CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md規範
try:
    # 使用現實路徑導入 (遵循AIVA Common指南)
    import sys
    import os
    from pathlib import Path
    
    # 添加services路徑到Python path (現實路徑優於虛擬環境)
    services_path = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(services_path))
    
    # 設置必要環境變數（不自創，使用現有配置）
    if 'AIVA_RABBITMQ_URL' not in os.environ:
        os.environ['AIVA_RABBITMQ_URL'] = 'amqp://guest:guest@localhost:5672/'
    
    from aiva_common.schemas import ExperienceSample, TraceRecord
    SCHEMAS_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Schema 導入失敗，使用Mock類型: {e}")
    
    # 使用Mock類型 (緊急備案)
    class MockExperienceSample:
            def __init__(self, **kwargs):
                # ExperienceSample required fields
                self.sample_id: str = kwargs.get('sample_id', '')
                self.session_id: str = kwargs.get('session_id', '')
                self.plan_id: str = kwargs.get('plan_id', '')
                self.state_before: dict = kwargs.get('state_before', {})
                self.action_taken: dict = kwargs.get('action_taken', {})
                self.state_after: dict = kwargs.get('state_after', {})
                self.reward: float = kwargs.get('reward', 0.0)
                self.reward_breakdown: dict = kwargs.get('reward_breakdown', {})
                self.context: dict = kwargs.get('context', {})
                self.target_info: dict = kwargs.get('target_info', {})
                self.timestamp = kwargs.get('timestamp', datetime.now())
                self.duration_ms: int = kwargs.get('duration_ms', 0)
                self.quality_score: float = kwargs.get('quality_score', 1.0)
                self.is_positive: bool = kwargs.get('is_positive', True)
                self.confidence: float = kwargs.get('confidence', 1.0)
                self.learning_tags: list = kwargs.get('learning_tags', [])
                self.difficulty_level: int = kwargs.get('difficulty_level', 1)
                
                # Set any additional attributes
                for key, value in kwargs.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
            
            def model_dump(self):
                return {key: value for key, value in self.__dict__.items()}
    
    class MockTraceRecord:
        def __init__(self, **kwargs):
            # TraceRecord required fields
            self.trace_id: str = kwargs.get('trace_id', '')
            self.plan_id: str = kwargs.get('plan_id', '')
            self.step_id: str = kwargs.get('step_id', '')
            self.session_id: str = kwargs.get('session_id', '')
            self.tool_name: str = kwargs.get('tool_name', '')
            self.input_data: dict = kwargs.get('input_data', {})
            self.output_data: dict = kwargs.get('output_data', {})
            self.status: str = kwargs.get('status', 'success')
            self.error_message: str | None = kwargs.get('error_message', None)
            self.execution_time_seconds: float = kwargs.get('execution_time_seconds', 0.0)
            self.timestamp = kwargs.get('timestamp', datetime.now())
            self.environment_response: dict = kwargs.get('environment_response', {})
            self.metadata: dict = kwargs.get('metadata', {})
            
            # Set any additional attributes
            for key, value in kwargs.items():
                if not hasattr(self, key):
                    setattr(self, key, value)
    
    ExperienceSample = MockExperienceSample
    TraceRecord = MockTraceRecord
    SCHEMAS_AVAILABLE = False

# 為類型註解定義別名
ExperienceSampleType = Union["ExperienceSample", Any]
TraceRecordType = Union["TraceRecord", Any]


class StorageBackend(ABC):
    """存儲後端抽象基類"""

    @abstractmethod
    async def save_experience_sample(self, sample: ExperienceSampleType) -> bool:
        """保存經驗樣本"""

    @abstractmethod
    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: Union[str, None] = None,
    ) -> List[ExperienceSampleType]:
        """獲取經驗樣本"""

    @abstractmethod
    async def save_trace(self, trace: TraceRecordType) -> bool:
        """保存追蹤記錄"""

    @abstractmethod
    async def get_traces_by_session(self, session_id: str) -> List[TraceRecordType]:
        """獲取會話的所有追蹤記錄"""

    @abstractmethod
    async def save_training_session(self, session_data: Dict[str, Any]) -> bool:
        """保存訓練會話"""

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """獲取存儲統計"""


class SQLiteBackend(StorageBackend):
    """SQLite 存儲後端"""

    def __init__(self, db_path: str = "./data/database/aiva.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 創建引擎
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # 設為 True 可看到 SQL 日誌
        )

        # 創建表
        Base.metadata.create_all(self.engine)

        # 創建會話工廠
        self.SessionLocal = sessionmaker(bind=self.engine)

    async def save_experience_sample(self, sample: ExperienceSampleType) -> bool:
        """保存經驗樣本"""
        with self.SessionLocal() as session:
            model = ExperienceSampleModel(
                sample_id=sample.sample_id,
                created_at=sample.timestamp,
                plan_data={
                    "plan_id": sample.plan_id,
                    "session_id": sample.session_id,
                    "state_before": sample.state_before,
                    "action_taken": sample.action_taken,
                    "state_after": sample.state_after,
                    "context": sample.context,
                    "target_info": sample.target_info,
                },
                trace_id=None,  # ExperienceSample 不包含 trace_id
                actual_result={
                    "reward": sample.reward,
                    "reward_breakdown": sample.reward_breakdown,
                    "is_positive": sample.is_positive,
                    "confidence": sample.confidence,
                },
                expected_result=None,  # ExperienceSample 不包含 expected_result
                quality_score=sample.quality_score,
                success=sample.is_positive,  # 用 is_positive 代替 success
                vulnerability_type=sample.learning_tags[0] if sample.learning_tags else "unknown",
                scenario_id=sample.context.get("scenario_id"),
                metadata={
                    "duration_ms": sample.duration_ms,
                    "difficulty_level": sample.difficulty_level,
                    "learning_tags": sample.learning_tags,
                },
            )
            session.add(model)
            session.commit()
        return True

    async def save_unified_experience_sample(self, sample: Any) -> bool:
        """保存統一格式的經驗樣本 (來自 aiva_common.schemas.ai.ExperienceSample)"""
        # 移除有問題的匯入，直接使用 Any 類型
        
        with self.SessionLocal() as session:
            # 由於表結構期望的是較複雜的經驗樣本格式，我們需要將統一格式適配到現有表結構
            # 這是一個臨時解決方案，未來可能需要重新設計表結構
            
            model = ExperienceSampleModel(
                sample_id=sample.sample_id,
                created_at=sample.timestamp,
                plan_data={
                    "plan_id": sample.plan_id,
                    "vulnerability_type": sample.learning_tags[0] if sample.learning_tags else "unknown",
                    "metadata": {"source": "unified_storage_adapter"},
                },
                trace_id=None,  # 設為 NULL 避開外鍵約束
                actual_result={
                    "success": sample.is_positive,
                    "state_after": sample.state_after,
                    "action_taken": sample.action_taken,
                },
                expected_result=None,  # 不可用於此格式
                quality_score=sample.quality_score or 0.5,
                success=sample.is_positive,
                vulnerability_type=sample.learning_tags[0] if sample.learning_tags else "unknown",
                scenario_id=sample.context.get("scan_id"),
                metadata={
                    "context": sample.context,
                    "target_info": sample.target_info,
                    "reward": sample.reward,
                    "reward_breakdown": sample.reward_breakdown,
                    "learning_tags": sample.learning_tags,
                },
            )
            session.add(model)
            session.commit()
        return True

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: Union[str, None] = None,
    ) -> List[ExperienceSampleType]:
        """獲取經驗樣本"""
        with self.SessionLocal() as session:
            query = session.query(ExperienceSampleModel)

            # 過濾條件
            query = query.filter(ExperienceSampleModel.quality_score >= min_quality)
            if vulnerability_type:
                query = query.filter(
                    ExperienceSampleModel.vulnerability_type == vulnerability_type
                )

            # 排序和限制
            query = query.order_by(desc(ExperienceSampleModel.quality_score)).limit(
                limit
            )

            # 轉換為 Pydantic 模型
            results: List[ExperienceSampleType] = []
            for model in query.all():
                # 從數據庫模型重建 ExperienceSample
                # 顯式轉換類型以避免 SQLAlchemy Column 類型錯誤
                sample = ExperienceSample(
                    sample_id=str(model.sample_id),  # type: ignore
                    session_id=str(model.plan_data.get("session_id", "unknown")),
                    plan_id=str(model.plan_data.get("plan_id", "unknown")),
                    state_before=dict(model.plan_data.get("state_before", {})),
                    action_taken=dict(model.plan_data.get("action_taken", {})),
                    state_after=dict(model.plan_data.get("state_after", {})),
                    reward=float(model.actual_result.get("reward", 0.0)),
                    reward_breakdown=dict(model.actual_result.get("reward_breakdown", {})),
                    context=dict(model.plan_data.get("context", {})),
                    target_info=dict(model.plan_data.get("target_info", {})),
                    timestamp=model.created_at,  # type: ignore
                    duration_ms=model.metadata.get("duration_ms") if model.metadata else None,
                    quality_score=float(model.quality_score) if model.quality_score is not None else None,  # type: ignore[arg-type]
                    is_positive=bool(model.success),  # type: ignore
                    confidence=float(model.actual_result.get("confidence", 1.0)),
                    learning_tags=list(model.metadata.get("learning_tags", [])) if model.metadata else [],
                    difficulty_level=int(model.metadata.get("difficulty_level", 1)) if model.metadata else 1,
                )
                results.append(sample)  # type: ignore

            return results

    async def save_trace(self, trace: TraceRecordType) -> bool:
        """保存追蹤記錄"""
        with self.SessionLocal() as session:
            model = TraceRecordModel(
                trace_id=trace.trace_id,
                session_id=trace.session_id,
                created_at=trace.timestamp,
                plan_data={
                    "plan_id": trace.plan_id,
                    "step_id": trace.step_id,
                    "tool_name": trace.tool_name,
                },
                steps=[{
                    "trace_id": trace.trace_id,
                    "plan_id": trace.plan_id,
                    "step_id": trace.step_id,
                    "tool_name": trace.tool_name,
                    "input_data": trace.input_data,
                    "output_data": trace.output_data,
                    "status": trace.status,
                    "error_message": trace.error_message,
                    "execution_time_seconds": trace.execution_time_seconds,
                    "environment_response": trace.environment_response,
                }],
                total_steps=1,  # 這是單個步驟記錄
                successful_steps=1 if trace.status == "success" else 0,
                failed_steps=1 if trace.status in ["failed", "error"] else 0,
                duration_seconds=trace.execution_time_seconds,
                final_result=trace.output_data,
                error_message=trace.error_message,
                metadata=trace.metadata,
            )
            session.add(model)
            session.commit()
        return True

    async def get_traces_by_session(self, session_id: str) -> List[TraceRecordType]:
        """獲取會話的所有追蹤記錄

        注意：由於 TraceRecord schema 已更新為單個步驟記錄，
        此方法返回資料庫中存儲的所有步驟記錄
        """
        with self.SessionLocal() as session:
            query = (
                session.query(TraceRecordModel)
                .filter(TraceRecordModel.session_id == session_id)
                .order_by(desc(TraceRecordModel.created_at))
            )

            results: List[TraceRecordType] = []
            for model in query.all():
                # 由於 TraceRecordModel.steps 儲存的是步驟列表，
                # 我們需要為每個步驟創建一個 TraceRecord
                # 直接檢查模型實例的 steps 屬性
                # 在 SQLAlchemy ORM 中，model.steps 應該是實際的數據而不是 Column
                try:
                    steps_data = getattr(model, 'steps', None)
                    if steps_data and isinstance(steps_data, list):
                        for step_data in steps_data:
                            trace = TraceRecord(
                            trace_id=step_data.get("trace_id", model.trace_id),
                            plan_id=step_data.get("plan_id", model.trace_id),
                            step_id=step_data.get("step_id", ""),
                            session_id=str(model.session_id),  # type: ignore
                            tool_name=step_data.get("tool_name", ""),
                            input_data=step_data.get("input_data", {}),
                            output_data=step_data.get("output_data", {}),
                            status=step_data.get("status", "unknown"),
                            error_message=step_data.get("error_message"),
                            execution_time_seconds=step_data.get(
                                "execution_time_seconds", 0.0
                            ),
                            timestamp=step_data.get("timestamp", model.created_at),
                            environment_response=step_data.get(
                                "environment_response", {}
                            ),
                            metadata=step_data.get("metadata", {}),
                        )
                            results.append(trace)  # type: ignore
                except (AttributeError, TypeError) as e:
                    # 如果步驟數據格式不正確，跳過此記錄
                    print(f"警告：跳過格式不正確的追蹤記錄: {e}")
                    continue

            return results

    async def save_training_session(self, session_data: Dict[str, Any]) -> bool:
        """保存訓練會話"""
        with self.SessionLocal() as session:
            model = TrainingSessionModel(
                session_id=session_data["session_id"],
                created_at=session_data.get("created_at", datetime.utcnow()),
                completed_at=session_data.get("completed_at"),
                session_type=session_data.get("session_type", "single"),
                scenario_id=session_data.get("scenario_id"),
                config=session_data.get("config", {}),
                total_episodes=session_data.get("total_episodes", 0),
                successful_episodes=session_data.get("successful_episodes", 0),
                total_samples=session_data.get("total_samples", 0),
                high_quality_samples=session_data.get("high_quality_samples", 0),
                avg_reward=session_data.get("avg_reward"),
                avg_quality=session_data.get("avg_quality"),
                best_reward=session_data.get("best_reward"),
                status=session_data.get("status", "running"),
                metadata=session_data.get("metadata", {}),
            )
            session.add(model)
            session.commit()
        return True

    async def get_statistics(self) -> Dict[str, Any]:
        """獲取存儲統計"""
        with self.SessionLocal() as session:
            stats = {
                "total_experiences": session.query(
                    func.count(ExperienceSampleModel.id)
                ).scalar(),
                "high_quality_experiences": session.query(
                    func.count(ExperienceSampleModel.id)
                )
                .filter(ExperienceSampleModel.quality_score >= 0.7)
                .scalar(),
                "total_traces": session.query(func.count(TraceRecordModel.id)).scalar(),
                "total_sessions": session.query(
                    func.count(TrainingSessionModel.id)
                ).scalar(),
                "total_checkpoints": session.query(
                    func.count(ModelCheckpointModel.id)
                ).scalar(),
                "total_knowledge": session.query(
                    func.count(KnowledgeEntryModel.id)
                ).scalar(),
                "database_size": (
                    0  # PostgreSQL database size not tracked via file system
                ),
            }

            # 按類型統計經驗樣本
            vuln_types = session.query(
                ExperienceSampleModel.vulnerability_type,
                func.count(ExperienceSampleModel.id),
            ).group_by(ExperienceSampleModel.vulnerability_type).all()
            
            # 將 SQLAlchemy Row 對象轉換為字典
            stats["experiences_by_type"] = {row[0]: row[1] for row in vuln_types}

            return stats


class PostgreSQLBackend(SQLiteBackend):
    """PostgreSQL 存儲後端（繼承 SQLite 的實現）"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "aiva",
        user: str = "aiva",
        password: str = "aiva",
    ):
        # 不調用父類 __init__
        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{database}",
            echo=False,
            pool_size=10,
            max_overflow=20,
        )

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)


class JSONLBackend(StorageBackend):
    """JSONL 文件存儲後端"""

    def __init__(self, data_dir: str = "./data/training/experiences"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 文件路徑
        self.experiences_file = self.data_dir / "experiences.jsonl"
        self.high_quality_file = self.data_dir / "high_quality.jsonl"

    async def save_experience_sample(self, sample: ExperienceSampleType) -> bool:
        """保存經驗樣本到 JSONL"""
        data = sample.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()

        # 保存到主文件
        with open(self.experiences_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # 高質量樣本額外保存
        if sample.quality_score is not None and sample.quality_score >= 0.7:
            with open(self.high_quality_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return True

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: Union[str, None] = None,
    ) -> List[ExperienceSampleType]:
        """從 JSONL 讀取經驗樣本"""
        if not self.experiences_file.exists():
            return []

        samples = []
        with open(self.experiences_file, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # 過濾
                if data.get("quality_score", 0) < min_quality:
                    continue
                if (
                    vulnerability_type
                    and data.get("plan", {}).get("vulnerability_type")
                    != vulnerability_type
                ):
                    continue

                # 轉換時間戳
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                sample = ExperienceSample(**data)
                samples.append(sample)  # type: ignore

                if len(samples) >= limit:
                    break

        # 按質量排序
        samples.sort(key=lambda x: x.quality_score, reverse=True)
        return samples[:limit]

    async def save_trace(self, trace: TraceRecordType) -> bool:
        """保存追蹤記錄（JSONL 不實現，返回 True）"""
        return True

    async def get_traces_by_session(self, session_id: str) -> List[TraceRecordType]:
        """獲取追蹤記錄（JSONL 不實現）"""
        return []

    async def save_training_session(self, session_data: Dict[str, Any]) -> bool:
        """保存訓練會話（JSONL 不實現，返回 True）"""
        return True

    async def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息"""
        total = 0
        high_quality = 0

        if self.experiences_file.exists():
            with open(self.experiences_file, encoding="utf-8") as f:
                for line in f:
                    total += 1
                    data = json.loads(line)
                    if data.get("quality_score", 0) >= 0.7:
                        high_quality += 1

        return {
            "total_experiences": total,
            "high_quality_experiences": high_quality,
            "file_size": (
                self.experiences_file.stat().st_size
                if self.experiences_file.exists()
                else 0
            ),
        }


class HybridBackend(StorageBackend):
    """混合存儲後端（數據庫 + JSONL）"""

    def __init__(
        self,
        db_backend: StorageBackend,
        jsonl_backend: JSONLBackend | None = None,
    ):
        self.db_backend = db_backend
        self.jsonl_backend = jsonl_backend or JSONLBackend()

    async def save_experience_sample(self, sample: ExperienceSampleType) -> bool:
        """雙寫：數據庫 + JSONL"""
        await self.db_backend.save_experience_sample(sample)
        await self.jsonl_backend.save_experience_sample(sample)
        return True

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: Union[str, None] = None,
    ) -> List[ExperienceSampleType]:
        """從數據庫讀取（更快）"""
        return await self.db_backend.get_experience_samples(
            limit, min_quality, vulnerability_type
        )

    async def save_trace(self, trace: TraceRecordType) -> bool:
        """保存到數據庫"""
        return await self.db_backend.save_trace(trace)

    async def get_traces_by_session(self, session_id: str) -> List[TraceRecordType]:
        """從數據庫讀取"""
        return await self.db_backend.get_traces_by_session(session_id)

    async def save_training_session(self, session_data: Dict[str, Any]) -> bool:
        """保存到數據庫"""
        return await self.db_backend.save_training_session(session_data)

    async def get_statistics(self) -> Dict[str, Any]:
        """合併統計"""
        db_stats = await self.db_backend.get_statistics()
        jsonl_stats = await self.jsonl_backend.get_statistics()

        return {
            **db_stats,
            "jsonl_file_size": jsonl_stats.get("file_size", 0),
        }
