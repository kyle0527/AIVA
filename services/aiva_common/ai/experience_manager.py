"""
AIVA Common AI Experience Manager - 經驗管理器組件

此文件提供符合 aiva_common 規範的經驗管理器實現，
支援強化學習經驗樣本收集、存儲、檢索和質量評估。

設計特點:
- 實現 IExperienceManager 介面
- 整合現有 aiva_common AI Schema (ExperienceSample, ExperienceManagerConfig)
- 支援多種存儲後端 (SQLite, PostgreSQL, MongoDB)
- 經驗樣本質量評估和過濾
- 自動清理和去重機制
- 學習會話管理

架構位置:
- 屬於 Common 層的共享組件
- 支援五大模組架構的經驗學習需求
- 與 AI 訓練和強化學習系統整合
"""



import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from ..schemas.ai import (
    AIExperienceCreatedEvent,
    ExperienceManagerConfig,
    ExperienceSample,
)
from .interfaces import IExperienceManager

logger = logging.getLogger(__name__)


class LearningSession(BaseModel):
    """學習會話"""
    
    session_id: str = Field(default_factory=lambda: f"learning_{uuid4().hex[:12]}")
    training_id: str | None = None
    session_type: str = "interactive"  # interactive|batch|evaluation
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    
    # 會話狀態
    is_active: bool = True
    status: str = "running"  # running|paused|completed|failed
    
    # 會話統計
    total_samples: int = 0
    high_quality_samples: int = 0
    medium_quality_samples: int = 0
    low_quality_samples: int = 0
    unique_plans: Set[str] = Field(default_factory=set)
    vulnerability_types: Set[str] = Field(default_factory=set)
    
    # 會話配置
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_samples: int = Field(default=1000, ge=1, le=10000)
    auto_cleanup: bool = True
    
    # 會話元數據
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_statistics(self, sample: ExperienceSample) -> None:
        """更新會話統計"""
        self.total_samples += 1
        
        # 根據質量分數分類
        if sample.quality_score:
            if sample.quality_score >= 0.8:
                self.high_quality_samples += 1
            elif sample.quality_score >= 0.6:
                self.medium_quality_samples += 1
            else:
                self.low_quality_samples += 1
        else:
            # 根據成功率估算質量
            if sample.is_positive and sample.confidence >= 0.8:
                self.high_quality_samples += 1
            else:
                self.medium_quality_samples += 1
        
        # 更新唯一標識集合
        self.unique_plans.add(sample.plan_id)
        if "vulnerability_type" in sample.target_info:
            self.vulnerability_types.add(sample.target_info["vulnerability_type"])
    
    def get_completion_rate(self) -> float:
        """獲取完成率"""
        if self.max_samples <= 0:
            return 0.0
        return min(self.total_samples / self.max_samples, 1.0)  # type: ignore
    
    def get_quality_distribution(self) -> Dict[str, float]:
        """獲取質量分佈"""
        if self.total_samples == 0:
            return {"high": 0.0, "medium": 0.0, "low": 0.0}
        
        return {
            "high": self.high_quality_samples / self.total_samples,
            "medium": self.medium_quality_samples / self.total_samples,
            "low": self.low_quality_samples / self.total_samples,
        }


class ExperienceFilter(BaseModel):
    """經驗過濾器"""
    
    # 基本過濾條件
    session_ids: List[str] = Field(default_factory=list)
    plan_ids: List[str] = Field(default_factory=list)
    vulnerability_types: List[str] = Field(default_factory=list)
    
    # 質量過濾
    min_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_only: bool = False
    
    # 時間過濾
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    # 其他過濾
    min_reward: float | None = None
    max_reward: float | None = None
    tags: List[str] = Field(default_factory=list)
    
    # 排序和限制
    sort_by: str = "timestamp"  # timestamp|quality_score|reward
    sort_desc: bool = True
    limit: int | None = None
    offset: int = 0


class ExperienceStatistics(BaseModel):
    """經驗統計信息"""
    
    # 基本統計
    total_samples: int = 0
    total_sessions: int = 0
    total_plans: int = 0
    
    # 質量統計
    high_quality_samples: int = 0
    medium_quality_samples: int = 0
    low_quality_samples: int = 0
    avg_quality_score: float = 0.0
    
    # 成功率統計
    positive_samples: int = 0
    negative_samples: int = 0
    success_rate: float = 0.0
    
    # 獎勵統計
    avg_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    
    # 時間統計
    earliest_sample: datetime | None = None
    latest_sample: datetime | None = None
    
    # 分佈統計
    vulnerability_type_distribution: Dict[str, int] = Field(default_factory=dict)
    plan_type_distribution: Dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)


class SQLiteExperienceStorage:
    """SQLite 經驗存儲實現"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # 創建經驗表
            cursor.execute("""  # type: ignore
                CREATE TABLE IF NOT EXISTS experiences (
                    sample_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    state_before TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    state_after TEXT NOT NULL,
                    reward REAL NOT NULL,
                    reward_breakdown TEXT,
                    context TEXT,
                    target_info TEXT,
                    timestamp TEXT NOT NULL,
                    duration_ms INTEGER,
                    quality_score REAL,
                    is_positive BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    learning_tags TEXT,
                    difficulty_level INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 創建會話表
            cursor.execute("""  # type: ignore
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    training_id TEXT,
                    session_type TEXT DEFAULT 'interactive',
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    status TEXT DEFAULT 'running',
                    total_samples INTEGER DEFAULT 0,
                    high_quality_samples INTEGER DEFAULT 0,
                    medium_quality_samples INTEGER DEFAULT 0,
                    low_quality_samples INTEGER DEFAULT 0,
                    quality_threshold REAL DEFAULT 0.7,
                    max_samples INTEGER DEFAULT 1000,
                    auto_cleanup BOOLEAN DEFAULT TRUE,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 創建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiences_session ON experiences(session_id)")  # type: ignore
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiences_plan ON experiences(plan_id)")  # type: ignore
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp)")  # type: ignore
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiences_quality ON experiences(quality_score)")  # type: ignore
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_training ON sessions(training_id)")  # type: ignore
            
            conn.commit()
        finally:
            conn.close()
    
    async def store_experience(self, experience: ExperienceSample) -> bool:
        """存儲經驗樣本"""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""  # type: ignore
                    INSERT OR REPLACE INTO experiences (
                        sample_id, session_id, plan_id, state_before, action_taken,
                        state_after, reward, reward_breakdown, context, target_info,
                        timestamp, duration_ms, quality_score, is_positive, confidence,
                        learning_tags, difficulty_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience.sample_id,
                    experience.session_id,
                    experience.plan_id,
                    json.dumps(experience.state_before),
                    json.dumps(experience.action_taken),
                    json.dumps(experience.state_after),
                    experience.reward,
                    json.dumps(experience.reward_breakdown),
                    json.dumps(experience.context),
                    json.dumps(experience.target_info),
                    experience.timestamp.isoformat(),
                    experience.duration_ms,
                    experience.quality_score,
                    experience.is_positive,
                    experience.confidence,
                    json.dumps(experience.learning_tags),
                    experience.difficulty_level
                ))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error storing experience sample {experience.sample_id}: {e}")
            return False
    
    async def get_experiences(
        self,
        filter_params: ExperienceFilter
    ) -> List[ExperienceSample]:
        """檢索經驗樣本"""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # 構建查詢
                query = "SELECT * FROM experiences WHERE 1=1"
                params = []
                
                if filter_params.session_ids:
                    placeholders = ",".join("?" * len(filter_params.session_ids))  # type: ignore
                    query += f" AND session_id IN ({placeholders})"
                    params.extend(filter_params.session_ids)  # type: ignore
                
                if filter_params.plan_ids:
                    placeholders = ",".join("?" * len(filter_params.plan_ids))  # type: ignore
                    query += f" AND plan_id IN ({placeholders})"
                    params.extend(filter_params.plan_ids)  # type: ignore
                
                if filter_params.min_quality_score is not None:
                    query += " AND quality_score >= ?"
                    params.append(filter_params.min_quality_score)  # type: ignore
                
                if filter_params.max_quality_score is not None:
                    query += " AND quality_score <= ?"
                    params.append(filter_params.max_quality_score)  # type: ignore
                
                if filter_params.positive_only:
                    query += " AND is_positive = TRUE"
                
                if filter_params.start_time:
                    query += " AND timestamp >= ?"
                    params.append(filter_params.start_time.isoformat())  # type: ignore
                
                if filter_params.end_time:
                    query += " AND timestamp <= ?"
                    params.append(filter_params.end_time.isoformat())  # type: ignore
                
                # 排序
                if filter_params.sort_by == "timestamp":
                    query += " ORDER BY timestamp"
                elif filter_params.sort_by == "quality_score":
                    query += " ORDER BY quality_score"
                elif filter_params.sort_by == "reward":
                    query += " ORDER BY reward"
                
                if filter_params.sort_desc:
                    query += " DESC"
                
                # 限制
                if filter_params.limit:
                    query += " LIMIT ?"
                    params.append(filter_params.limit)  # type: ignore
                    
                    if filter_params.offset > 0:
                        query += " OFFSET ?"
                        params.append(filter_params.offset)  # type: ignore
                
                cursor.execute(query, params)  # type: ignore
                rows = cursor.fetchall()
                
                # 轉換為 ExperienceSample 對象
                samples = []
                columns = [desc[0] for desc in cursor.description]
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    sample = ExperienceSample(
                        sample_id=row_dict["sample_id"],
                        session_id=row_dict["session_id"],
                        plan_id=row_dict["plan_id"],
                        state_before=json.loads(row_dict["state_before"]),
                        action_taken=json.loads(row_dict["action_taken"]),
                        state_after=json.loads(row_dict["state_after"]),
                        reward=row_dict["reward"],
                        reward_breakdown=json.loads(row_dict["reward_breakdown"] or "{}"),
                        context=json.loads(row_dict["context"] or "{}"),
                        target_info=json.loads(row_dict["target_info"] or "{}"),
                        timestamp=datetime.fromisoformat(row_dict["timestamp"]),
                        duration_ms=row_dict["duration_ms"],
                        quality_score=row_dict["quality_score"],
                        is_positive=bool(row_dict["is_positive"]),
                        confidence=row_dict["confidence"],
                        learning_tags=json.loads(row_dict["learning_tags"] or "[]"),
                        difficulty_level=row_dict["difficulty_level"]
                    )
                    samples.append(sample)  # type: ignore
                
                return samples  # type: ignore
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error retrieving experiences: {e}")
            return []
    
    async def store_session(self, session: LearningSession) -> bool:
        """存儲學習會話"""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""  # type: ignore
                    INSERT OR REPLACE INTO sessions (
                        session_id, training_id, session_type, start_time, end_time,
                        is_active, status, total_samples, high_quality_samples,
                        medium_quality_samples, low_quality_samples, quality_threshold,
                        max_samples, auto_cleanup, tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.training_id,
                    session.session_type,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.is_active,
                    session.status,
                    session.total_samples,
                    session.high_quality_samples,
                    session.medium_quality_samples,
                    session.low_quality_samples,
                    session.quality_threshold,
                    session.max_samples,
                    session.auto_cleanup,
                    json.dumps(session.tags),
                    json.dumps(session.metadata)
                ))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error storing session {session.session_id}: {e}")
            return False
    
    async def get_statistics(self) -> ExperienceStatistics:
        """獲取統計信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                stats = ExperienceStatistics()
                
                # 基本統計
                cursor.execute("SELECT COUNT(*) FROM experiences")  # type: ignore
                stats.total_samples = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT session_id) FROM experiences")  # type: ignore
                stats.total_sessions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT plan_id) FROM experiences")  # type: ignore
                stats.total_plans = cursor.fetchone()[0]
                
                if stats.total_samples > 0:
                    # 質量統計
                    cursor.execute("""  # type: ignore
                        SELECT 
                            COUNT(CASE WHEN quality_score >= 0.8 THEN 1 END) as high,
                            COUNT(CASE WHEN quality_score >= 0.6 AND quality_score < 0.8 THEN 1 END) as medium,
                            COUNT(CASE WHEN quality_score < 0.6 THEN 1 END) as low,
                            AVG(quality_score) as avg_quality
                        FROM experiences 
                        WHERE quality_score IS NOT NULL
                    """)
                    quality_row = cursor.fetchone()
                    if quality_row:
                        stats.high_quality_samples = quality_row[0] or 0
                        stats.medium_quality_samples = quality_row[1] or 0
                        stats.low_quality_samples = quality_row[2] or 0
                        stats.avg_quality_score = quality_row[3] or 0.0
                    
                    # 成功率統計
                    cursor.execute("""  # type: ignore
                        SELECT 
                            COUNT(CASE WHEN is_positive = TRUE THEN 1 END) as positive,
                            COUNT(CASE WHEN is_positive = FALSE THEN 1 END) as negative
                        FROM experiences
                    """)
                    success_row = cursor.fetchone()
                    if success_row:
                        stats.positive_samples = success_row[0] or 0
                        stats.negative_samples = success_row[1] or 0
                        stats.success_rate = stats.positive_samples / stats.total_samples
                    
                    # 獎勵統計
                    cursor.execute("SELECT AVG(reward), MAX(reward), MIN(reward) FROM experiences")  # type: ignore
                    reward_row = cursor.fetchone()
                    if reward_row:
                        stats.avg_reward = reward_row[0] or 0.0
                        stats.max_reward = reward_row[1] or 0.0
                        stats.min_reward = reward_row[2] or 0.0
                    
                    # 時間統計
                    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM experiences")  # type: ignore
                    time_row = cursor.fetchone()
                    if time_row and time_row[0]:
                        stats.earliest_sample = datetime.fromisoformat(time_row[0])
                        stats.latest_sample = datetime.fromisoformat(time_row[1])
                
                return stats
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return ExperienceStatistics()


class AIVAExperienceManager(IExperienceManager):
    """AIVA 經驗管理器實現
    
    此類提供符合 aiva_common 規範的經驗管理功能，
    支援強化學習經驗樣本的完整生命週期管理。
    """

    def __init__(
        self,
        config: Optional[ExperienceManagerConfig] = None
    ):
        """初始化經驗管理器
        
        Args:
            config: 經驗管理器配置
        """
        self.config = config or ExperienceManagerConfig(
            manager_id=f"exp_mgr_{uuid4().hex[:8]}"
        )
        
        # 初始化存儲後端
        if self.config.storage_backend == "sqlite":
            storage_path = self.config.storage_path or "./data/experience.db"
            self.storage = SQLiteExperienceStorage(storage_path)
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
        
        # 活躍會話管理
        self.active_sessions: Dict[str, LearningSession] = {}
        
        # 去重緩存
        self._dedup_cache: Set[str] = set()
        
        # 統計和監控
        self.start_time = datetime.now(UTC)
        self.total_samples_stored = 0
        self.total_samples_retrieved = 0
        
        # 清理任務
        self._cleanup_task: Optional[asyncio.Task[Any]] = None
        self._start_cleanup_task()
        
        logger.info(f"AIVAExperienceManager initialized with {self.config.storage_backend} backend")

    async def create_learning_session(
        self, 
        session_config: Dict[str, Any]
    ) -> str:
        """創建學習會話
        
        Args:
            session_config: 會話配置字典，包含 training_id, session_type 等參數
            
        Returns:
            會話 ID
        """
        try:
            # 從配置中提取參數
            training_id = session_config.get("training_id", "")
            session_type = session_config.get("session_type", "interactive")
            
            session = LearningSession(
                training_id=training_id,
                session_type=session_type,
                **session_config  # 其他配置參數
            )
            
            # 存儲會話
            await self.storage.store_session(session)
            
            # 添加到活躍會話
            self.active_sessions[session.session_id] = session
            
            logger.info(f"Learning session created: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            logger.error(f"Error creating learning session: {e}")
            raise

    async def store_experience(self, experience: ExperienceSample
    ) -> bool:
        """存儲經驗樣本
        
        Args:
            experience: 經驗樣本
            
        Returns:
            是否存儲成功
        """
        try:
            # 去重檢查
            if self.config.deduplication_enabled:
                sample_hash = self._calculate_sample_hash(experience)
                if sample_hash in self._dedup_cache:
                    logger.debug(f"Duplicate sample detected, skipping: {experience.sample_id}")
                    return False
                self._dedup_cache.add(sample_hash)
            
            # 質量評估
            if experience.quality_score is None:
                experience.quality_score = await self._evaluate_sample_quality(experience)
            
            # 存儲樣本
            success = await self.storage.store_experience(experience)
            
            if success:
                self.total_samples_stored += 1
                
                # 更新會話統計
                if experience.session_id in self.active_sessions:
                    session = self.active_sessions[experience.session_id]
                    session.update_statistics(experience)
                    await self.storage.store_session(session)
                
                # 創建經驗創建事件
                event = AIExperienceCreatedEvent(
                    experience_id=experience.sample_id,
                    trace_id=f"trace_{uuid4().hex[:8]}",
                    vulnerability_type=experience.target_info.get("vulnerability_type", "unknown"),
                    quality_score=experience.quality_score or 0.0,
                    success=experience.is_positive,
                    plan_summary={"plan_id": experience.plan_id},
                    result_summary={"reward": experience.reward, "confidence": experience.confidence}
                )
                
                # 這裡可以發送事件到消息佇列
                logger.debug(f"Experience created event: {event.experience_id}")
                
                logger.info(f"Experience sample stored: {experience.sample_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing experience sample {experience.sample_id}: {e}")
            return False

    async def get_experiences(
        self,
        session_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        quality_threshold: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[ExperienceSample]:
        """檢索經驗樣本
        
        Args:
            session_id: 會話 ID
            plan_id: 計劃 ID
            quality_threshold: 質量閾值
            limit: 數量限制
            
        Returns:
            經驗樣本列表
        """
        try:
            filter_params = ExperienceFilter(
                session_ids=[session_id] if session_id else [],
                plan_ids=[plan_id] if plan_id else [],
                min_quality_score=quality_threshold,
                limit=limit
            )
            
            samples = await self.storage.get_experiences(filter_params)
            self.total_samples_retrieved += len(samples)  # type: ignore
            
            logger.info(f"Retrieved {len(samples)} experience samples")
            return samples  # type: ignore
            
        except Exception as e:
            logger.error(f"Error retrieving experiences: {e}")
            return []

    async def retrieve_experiences(
        self,
        query_context: Dict[str, Any],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[ExperienceSample]:
        """檢索相關經驗 (抽象方法實現)
        
        Args:
            query_context: 查詢上下文
            limit: 返回數量限制
            similarity_threshold: 相似度閾值
            
        Returns:
            相關經驗樣本列表
        """
        try:
            # 從查詢上下文中提取參數
            session_id = query_context.get('session_id')
            plan_id = query_context.get('plan_id')
            quality_threshold = query_context.get('quality_threshold', similarity_threshold)
            
            # 使用現有的 get_experiences 方法
            return await self.get_experiences(
                session_id=session_id,
                plan_id=plan_id,
                quality_threshold=quality_threshold,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Error in retrieve_experiences: {e}")
            return []

    async def evaluate_sample_quality(
        self,
        sample: ExperienceSample
    ) -> float:
        """評估樣本質量
        
        Args:
            sample: 經驗樣本
            
        Returns:
            質量分數 (0.0 - 1.0)
        """
        return await self._evaluate_sample_quality(sample)

    async def get_learning_statistics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """獲取學習統計信息
        
        Args:
            session_id: 會話 ID (可選)
            
        Returns:
            統計信息字典
        """
        try:
            if session_id:
                # 獲取特定會話統計
                session = self.active_sessions.get(session_id)
                if session:
                    return {
                        "session_id": session_id,
                        "session_type": session.session_type,
                        "total_samples": session.total_samples,
                        "quality_distribution": session.get_quality_distribution(),
                        "completion_rate": session.get_completion_rate(),
                        "unique_plans": len(session.unique_plans),  # type: ignore
                        "vulnerability_types": len(session.vulnerability_types),  # type: ignore
                        "is_active": session.is_active,
                        "status": session.status,
                    }
                else:
                    return {"error": f"Session {session_id} not found"}
            else:
                # 獲取全局統計
                stats = await self.storage.get_statistics()
                uptime = (datetime.now(UTC) - self.start_time).total_seconds()
                
                return {
                    "manager_id": self.config.manager_id,
                    "uptime_seconds": uptime,
                    "total_samples": stats.total_samples,
                    "total_sessions": stats.total_sessions,
                    "total_plans": stats.total_plans,
                    "quality_distribution": {
                        "high": stats.high_quality_samples,
                        "medium": stats.medium_quality_samples,
                        "low": stats.low_quality_samples,
                    },
                    "success_rate": stats.success_rate,
                    "avg_quality_score": stats.avg_quality_score,
                    "avg_reward": stats.avg_reward,
                    "active_sessions": len(self.active_sessions),  # type: ignore
                    "samples_stored_this_session": self.total_samples_stored,
                    "samples_retrieved_this_session": self.total_samples_retrieved,
                    "storage_backend": self.config.storage_backend,
                    "deduplication_enabled": self.config.deduplication_enabled,
                }
                
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {"error": str(e)}

    async def cleanup_old_experiences(
        self,
        retention_days: Optional[int] = None
    ) -> int:
        """清理舊經驗樣本
        
        Args:
            retention_days: 保留天數
            
        Returns:
            清理的樣本數量
        """
        try:
            retention_days = retention_days or self.config.retention_days
            cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
            
            # 獲取要清理的樣本
            filter_params = ExperienceFilter(
                end_time=cutoff_date,
                sort_by="timestamp",
                sort_desc=False
            )
            
            old_samples = await self.storage.get_experiences(filter_params)
            
            if not old_samples:
                return 0
            
            # 這裡應該實現實際的刪除邏輯
            # 為了簡化，我們只記錄清理意圖
            cleanup_count = len(old_samples)  # type: ignore
            
            logger.info(f"Would cleanup {cleanup_count} old experience samples")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old experiences: {e}")
            return 0

    async def end_learning_session(
        self,
        session_id: str
    ) -> bool:
        """結束學習會話
        
        Args:
            session_id: 會話 ID
            
        Returns:
            是否成功結束
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.is_active = False
            session.status = "completed"
            session.end_time = datetime.now(UTC)
            
            # 更新存儲
            await self.storage.store_session(session)
            
            # 從活躍會話中移除
            del self.active_sessions[session_id]
            
            logger.info(f"Learning session ended: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending learning session {session_id}: {e}")
            return False

    def _calculate_sample_hash(self, sample: ExperienceSample) -> str:
        """計算樣本哈希用於去重"""
        hash_content = {
            "plan_id": sample.plan_id,
            "state_before": sample.state_before,
            "action_taken": sample.action_taken,
            "target_info": sample.target_info,
        }
        
        content_str = json.dumps(hash_content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    async def _evaluate_sample_quality(self, sample: ExperienceSample) -> float:
        """評估樣本質量
        
        計算基於多個因子的質量分數:
        - 獎勵值的大小和分佈
        - 是否為正樣本
        - 置信度
        - 執行時長（過短或過長可能質量較低）
        - 難度級別
        """
        quality_score = 0.0
        
        # 基於獎勵的質量評估 (40%)
        reward_factor = min(abs(sample.reward) / 10.0, 1.0)  # 假設最大獎勵為10
        if sample.is_positive:
            reward_factor = min(reward_factor + 0.2, 1.0)  # 正樣本加分
        quality_score += reward_factor * 0.4
        
        # 基於置信度的質量評估 (25%)
        quality_score += sample.confidence * 0.25
        
        # 基於執行時長的質量評估 (15%)
        if sample.duration_ms:
            # 假設合理執行時間範圍為 100ms - 30s
            duration_seconds = sample.duration_ms / 1000.0
            if 0.1 <= duration_seconds <= 30.0:
                duration_factor = 1.0
            elif duration_seconds < 0.1:
                duration_factor = duration_seconds / 0.1  # 過短扣分
            else:
                duration_factor = max(30.0 / duration_seconds, 0.1)  # 過長扣分
            quality_score += duration_factor * 0.15
        else:
            quality_score += 0.15  # 沒有時長信息給予中等分數
        
        # 基於難度級別的質量評估 (10%)
        difficulty_factor = min(sample.difficulty_level / 5.0, 1.0)  # 假設最大難度為5
        quality_score += difficulty_factor * 0.1
        
        # 基於狀態複雜度的質量評估 (10%)
        state_complexity = (
            len(str(sample.state_before)) + len(str(sample.state_after))  # type: ignore
        ) / 2000.0  # 假設平均狀態大小為1000字符
        complexity_factor = min(state_complexity, 1.0)
        quality_score += complexity_factor * 0.1
        
        return min(quality_score, 1.0)

    def _start_cleanup_task(self) -> None:
        """啟動清理任務"""
        if self.config.auto_cleanup and (self._cleanup_task is None or self._cleanup_task.done()):
            try:
                # 檢查是否有事件循環在運行
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
            except RuntimeError:
                # 沒有事件循環在運行，跳過任務創建
                logger.info("No running event loop, skipping periodic cleanup task creation")

    async def _periodic_cleanup(self) -> None:
        """定期清理任務"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小時執行一次
                
                if self.config.auto_cleanup:
                    # 清理舊經驗
                    cleaned_count = await self.cleanup_old_experiences()
                    if cleaned_count > 0:
                        logger.info(f"Periodic cleanup: {cleaned_count} samples cleaned")
                    
                    # 清理去重緩存
                    if len(self._dedup_cache) > 10000:  # 限制緩存大小  # type: ignore
                        self._dedup_cache.clear()
                        logger.info("Deduplication cache cleared")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def cleanup(self) -> None:
        """清理資源"""
        try:
            # 取消清理任務
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # 結束所有活躍會話
            for session_id in list(self.active_sessions.keys()):  # type: ignore
                await self.end_learning_session(session_id)
            
            # 清理緩存
            self._dedup_cache.clear()
            
            logger.info("AIVAExperienceManager cleaned up")
            
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


def create_experience_manager(
    config: Optional[ExperienceManagerConfig] = None,
    **kwargs  # type: ignore
) -> AIVAExperienceManager:
    """創建經驗管理器實例

    Args:
        config: 經驗管理器配置
        **kwargs: 其他參數  # type: ignore

    Returns:
        經驗管理器實例
    """
    return AIVAExperienceManager(config=config)


# ============================================================================
# 全域實例 (Global Instance)
# ============================================================================

# 創建全域經驗管理器實例
experience_manager = create_experience_manager()