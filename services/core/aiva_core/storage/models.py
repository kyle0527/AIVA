"""
SQLAlchemy ORM 模型定義

所有訓練數據的數據庫模型
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ExperienceSampleModel(Base):
    """經驗樣本模型"""

    __tablename__ = "experience_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 計畫相關
    plan_data = Column(JSON, nullable=False)  # 完整的 AttackPlan
    trace_id = Column(String(64), ForeignKey("trace_records.trace_id"), nullable=True)

    # 結果
    actual_result = Column(JSON, nullable=False)  # AttackResult
    expected_result = Column(JSON, nullable=True)  # 預期結果（如果有）

    # 質量評分
    quality_score = Column(Float, nullable=False, index=True)
    success = Column(Boolean, nullable=False, index=True)

    # 分類
    vulnerability_type = Column(
        String(32), nullable=False, index=True
    )  # sqli, xss, etc
    scenario_id = Column(String(64), nullable=True, index=True)

    # 元數據
    metadata = Column(JSON, nullable=True)

    # 關聯
    trace = relationship("TraceRecordModel", back_populates="experiences")

    __table_args__ = (
        Index("idx_quality_type", "quality_score", "vulnerability_type"),
        Index("idx_created_success", "created_at", "success"),
    )


class TraceRecordModel(Base):
    """執行追蹤記錄模型"""

    __tablename__ = "trace_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(64), unique=True, nullable=False, index=True)
    session_id = Column(
        String(64),
        ForeignKey("training_sessions.session_id"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 追蹤數據
    plan_data = Column(JSON, nullable=False)  # AttackPlan
    steps = Column(JSON, nullable=False)  # List[TraceStep]

    # 統計
    total_steps = Column(Integer, nullable=False)
    successful_steps = Column(Integer, nullable=False)
    failed_steps = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)

    # 結果
    final_result = Column(JSON, nullable=True)  # AttackResult
    error_message = Column(Text, nullable=True)

    # 元數據
    metadata = Column(JSON, nullable=True)

    # 關聯
    session = relationship("TrainingSessionModel", back_populates="traces")
    experiences = relationship("ExperienceSampleModel", back_populates="trace")

    __table_args__ = (Index("idx_session_created", "session_id", "created_at"),)


class TrainingSessionModel(Base):
    """訓練會話模型"""

    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # 會話信息
    session_type = Column(String(32), nullable=False)  # single / batch / continuous
    scenario_id = Column(String(64), nullable=True, index=True)

    # 配置
    config = Column(JSON, nullable=False)  # TrainingConfig

    # 統計
    total_episodes = Column(Integer, default=0)
    successful_episodes = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    high_quality_samples = Column(Integer, default=0)

    # 指標
    avg_reward = Column(Float, nullable=True)
    avg_quality = Column(Float, nullable=True)
    best_reward = Column(Float, nullable=True)

    # 狀態
    status = Column(
        String(16), nullable=False, default="running"
    )  # running / completed / failed

    # 元數據
    metadata = Column(JSON, nullable=True)

    # 關聯
    traces = relationship("TraceRecordModel", back_populates="session")

    __table_args__ = (Index("idx_status_created", "status", "created_at"),)


class ModelCheckpointModel(Base):
    """模型檢查點模型"""

    __tablename__ = "model_checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 模型信息
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False, index=True)
    model_type = Column(String(32), nullable=False)  # bio_neuron / decision_core / etc

    # 文件路徑
    checkpoint_path = Column(String(512), nullable=False)
    config_path = Column(String(512), nullable=True)

    # 訓練信息
    epoch = Column(Integer, nullable=True)
    training_session_id = Column(String(64), nullable=True, index=True)

    # 性能指標
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    reward = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)

    # 狀態
    is_best = Column(Boolean, default=False, index=True)
    is_deployed = Column(Boolean, default=False, index=True)

    # 元數據
    metadata = Column(JSON, nullable=True)

    __table_args__ = (Index("idx_version_deployed", "model_version", "is_deployed"),)


class KnowledgeEntryModel(Base):
    """知識條目模型（RAG）"""

    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 知識類型
    knowledge_type = Column(
        String(32), nullable=False, index=True
    )  # vulnerability / payload / technique / etc

    # 內容
    content = Column(Text, nullable=False)
    embedding_vector = Column(JSON, nullable=True)  # 可選：存儲嵌入向量的引用

    # 分類
    category = Column(String(64), nullable=True, index=True)
    tags = Column(JSON, nullable=True)  # List[str]

    # 關聯
    source_sample_id = Column(String(64), nullable=True)  # 來自哪個經驗樣本
    related_cve = Column(String(32), nullable=True, index=True)
    related_cwe = Column(String(32), nullable=True, index=True)

    # 質量
    confidence = Column(Float, default=1.0, nullable=False)
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, nullable=True)

    # 元數據
    metadata = Column(JSON, nullable=True)

    __table_args__ = (Index("idx_type_category", "knowledge_type", "category"),)


class ScenarioModel(Base):
    """靶場場景模型"""

    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 場景信息
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    vulnerability_type = Column(String(32), nullable=False, index=True)
    difficulty = Column(String(16), nullable=False)  # easy / medium / hard

    # 配置
    config = Column(JSON, nullable=False)  # 完整的場景配置

    # OWASP 範圍
    owasp_range = Column(String(16), nullable=True, index=True)  # SQLI-1, XSS-2, etc

    # 狀態
    is_active = Column(Boolean, default=True, index=True)
    is_custom = Column(Boolean, default=False)

    # 統計
    total_attempts = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    avg_success_rate = Column(Float, nullable=True)

    # 元數據
    metadata = Column(JSON, nullable=True)

    __table_args__ = (Index("idx_type_difficulty", "vulnerability_type", "difficulty"),)
