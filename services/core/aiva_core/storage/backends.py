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
from pathlib import Path
from typing import Any

from services.aiva_common.schemas import AttackPlan, AttackResult, ExperienceSample, TraceRecord
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    ExperienceSampleModel,
    KnowledgeEntryModel,
    ModelCheckpointModel,
    TraceRecordModel,
    TrainingSessionModel,
)


class StorageBackend(ABC):
    """存儲後端抽象基類"""

    @abstractmethod
    async def save_experience_sample(self, sample: ExperienceSample) -> bool:
        """保存經驗樣本"""

    @abstractmethod
    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: str | None = None,
    ) -> list[ExperienceSample]:
        """獲取經驗樣本"""

    @abstractmethod
    async def save_trace(self, trace: TraceRecord) -> bool:
        """保存追蹤記錄"""

    @abstractmethod
    async def get_traces_by_session(self, session_id: str) -> list[TraceRecord]:
        """獲取會話的所有追蹤記錄"""

    @abstractmethod
    async def save_training_session(self, session_data: dict[str, Any]) -> bool:
        """保存訓練會話"""

    @abstractmethod
    async def get_statistics(self) -> dict[str, Any]:
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

    async def save_experience_sample(self, sample: ExperienceSample) -> bool:
        """保存經驗樣本"""
        with self.SessionLocal() as session:
            model = ExperienceSampleModel(
                sample_id=sample.sample_id,
                created_at=sample.timestamp,
                plan_data=sample.plan.model_dump(),
                trace_id=sample.trace_id,
                actual_result=sample.actual_result.model_dump(),
                expected_result=(
                    sample.expected_result.model_dump()
                    if sample.expected_result
                    else None
                ),
                quality_score=sample.quality_score,
                success=sample.actual_result.success,
                vulnerability_type=sample.plan.vulnerability_type,
                scenario_id=sample.plan.metadata.get("scenario_id"),
                metadata=sample.metadata,
            )
            session.add(model)
            session.commit()
        return True

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: str | None = None,
    ) -> list[ExperienceSample]:
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
            results = []
            for model in query.all():
                sample = ExperienceSample(
                    sample_id=model.sample_id,
                    timestamp=model.created_at,
                    plan=AttackPlan(**model.plan_data),
                    trace_id=model.trace_id,
                    actual_result=AttackResult(**model.actual_result),
                    expected_result=(
                        AttackResult(**model.expected_result)
                        if model.expected_result
                        else None
                    ),
                    quality_score=model.quality_score,
                    metadata=model.metadata or {},
                )
                results.append(sample)

            return results

    async def save_trace(self, trace: TraceRecord) -> bool:
        """保存追蹤記錄"""
        with self.SessionLocal() as session:
            model = TraceRecordModel(
                trace_id=trace.trace_id,
                session_id=trace.session_id,
                created_at=trace.timestamp,
                plan_data=trace.plan.model_dump(),
                steps=[step.model_dump() for step in trace.steps],
                total_steps=trace.total_steps,
                successful_steps=trace.successful_steps,
                failed_steps=trace.failed_steps,
                duration_seconds=trace.duration_seconds,
                final_result=(
                    trace.final_result.model_dump() if trace.final_result else None
                ),
                error_message=trace.error_message,
                metadata=trace.metadata,
            )
            session.add(model)
            session.commit()
        return True

    async def get_traces_by_session(self, session_id: str) -> list[TraceRecord]:
        """獲取會話的所有追蹤記錄"""
        with self.SessionLocal() as session:
            query = (
                session.query(TraceRecordModel)
                .filter(TraceRecordModel.session_id == session_id)
                .order_by(TraceRecordModel.created_at)
            )

            results = []
            for model in query.all():
                trace = TraceRecord(
                    trace_id=model.trace_id,
                    session_id=model.session_id,
                    timestamp=model.created_at,
                    plan=AttackPlan(**model.plan_data),
                    steps=model.steps,
                    total_steps=model.total_steps,
                    successful_steps=model.successful_steps,
                    failed_steps=model.failed_steps,
                    duration_seconds=model.duration_seconds,
                    final_result=(
                        AttackResult(**model.final_result)
                        if model.final_result
                        else None
                    ),
                    error_message=model.error_message,
                    metadata=model.metadata or {},
                )
                results.append(trace)

            return results

    async def save_training_session(self, session_data: dict[str, Any]) -> bool:
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

    async def get_statistics(self) -> dict[str, Any]:
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
                    self.db_path.stat().st_size if self.db_path.exists() else 0
                ),
            }

            # 按類型統計經驗樣本
            vuln_types = (
                session.query(
                    ExperienceSampleModel.vulnerability_type,
                    func.count(ExperienceSampleModel.id),
                )
                .group_by(ExperienceSampleModel.vulnerability_type)
                .all()
            )
            stats["experiences_by_type"] = {vtype: count for vtype, count in vuln_types}

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

    async def save_experience_sample(self, sample: ExperienceSample) -> bool:
        """保存經驗樣本到 JSONL"""
        data = sample.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()

        # 保存到主文件
        with open(self.experiences_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # 高質量樣本額外保存
        if sample.quality_score >= 0.7:
            with open(self.high_quality_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return True

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: str | None = None,
    ) -> list[ExperienceSample]:
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
                samples.append(sample)

                if len(samples) >= limit:
                    break

        # 按質量排序
        samples.sort(key=lambda x: x.quality_score, reverse=True)
        return samples[:limit]

    async def save_trace(self, trace: TraceRecord) -> bool:
        """保存追蹤記錄（JSONL 不實現，返回 True）"""
        return True

    async def get_traces_by_session(self, session_id: str) -> list[TraceRecord]:
        """獲取追蹤記錄（JSONL 不實現）"""
        return []

    async def save_training_session(self, session_data: dict[str, Any]) -> bool:
        """保存訓練會話（JSONL 不實現，返回 True）"""
        return True

    async def get_statistics(self) -> dict[str, Any]:
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

    async def save_experience_sample(self, sample: ExperienceSample) -> bool:
        """雙寫：數據庫 + JSONL"""
        await self.db_backend.save_experience_sample(sample)
        await self.jsonl_backend.save_experience_sample(sample)
        return True

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: str | None = None,
    ) -> list[ExperienceSample]:
        """從數據庫讀取（更快）"""
        return await self.db_backend.get_experience_samples(
            limit, min_quality, vulnerability_type
        )

    async def save_trace(self, trace: TraceRecord) -> bool:
        """保存到數據庫"""
        return await self.db_backend.save_trace(trace)

    async def get_traces_by_session(self, session_id: str) -> list[TraceRecord]:
        """從數據庫讀取"""
        return await self.db_backend.get_traces_by_session(session_id)

    async def save_training_session(self, session_data: dict[str, Any]) -> bool:
        """保存到數據庫"""
        return await self.db_backend.save_training_session(session_data)

    async def get_statistics(self) -> dict[str, Any]:
        """合併統計"""
        db_stats = await self.db_backend.get_statistics()
        jsonl_stats = await self.jsonl_backend.get_statistics()

        return {
            **db_stats,
            "jsonl_file_size": jsonl_stats.get("file_size", 0),
        }
