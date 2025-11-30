"""SQLAlchemy 數據模型 - 能力元數據數據庫

遵循 AIVA v2.0 規範
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class CapabilityRecord(Base):
    """能力記錄主表"""
    __tablename__ = "capability_records"
    
    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(500), unique=True, nullable=False, index=True)
    
    # 基本信息
    name = Column(String(200), nullable=False, index=True)
    module = Column(String(200), nullable=False, index=True)
    language = Column(String(50), nullable=False)
    file_path = Column(Text)
    
    # 版本控制
    version = Column(Integer, nullable=False, default=1)
    content_hash = Column(String(64), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    
    # 完整元數據 JSON
    metadata_json = Column(Text, nullable=False)
    
    # 時間戳
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # 關係
    versions = relationship("CapabilityVersion", back_populates="capability", cascade="all, delete-orphan")
    stats = relationship("CapabilityInvocationStats", back_populates="capability", uselist=False, cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        Index('idx_cap_name', 'name'),
        Index('idx_cap_module', 'module'),
        Index('idx_cap_hash', 'content_hash'),
        Index('idx_cap_active', 'is_active'),
    )


class CapabilityVersion(Base):
    """能力版本歷史表"""
    __tablename__ = "capability_versions"
    
    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 外鍵
    capability_key = Column(String(500), ForeignKey('capability_records.key'), nullable=False)
    
    # 版本信息
    version = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    metadata_json = Column(Text, nullable=False)
    
    # 變更信息
    change_type = Column(String(20), nullable=False, index=True)  # added, modified, deleted
    change_summary = Column(Text)
    
    # 時間戳
    created_at = Column(DateTime, nullable=False, index=True)
    
    # 關係
    capability = relationship("CapabilityRecord", back_populates="versions")
    
    # 唯一約束
    __table_args__ = (
        Index('idx_ver_change_type', 'change_type'),
        Index('idx_ver_created_at', 'created_at'),
    )


class CapabilityChangeLog(Base):
    """能力變更日誌表（每次掃描記錄一條）"""
    __tablename__ = "capability_change_logs"
    
    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 掃描信息
    scan_id = Column(String(36), nullable=False)
    scan_timestamp = Column(DateTime, nullable=False, index=True)
    
    # 統計信息
    added_count = Column(Integer, nullable=False, default=0)
    modified_count = Column(Integer, nullable=False, default=0)
    deleted_count = Column(Integer, nullable=False, default=0)
    unchanged_count = Column(Integer, nullable=False, default=0)
    total_capabilities = Column(Integer, nullable=False)
    
    # 詳細信息 JSON
    details_json = Column(Text)
    
    # 索引
    __table_args__ = (
        Index('idx_log_timestamp', 'scan_timestamp'),
    )


class CapabilityInvocationStats(Base):
    """能力調用統計表"""
    __tablename__ = "capability_invocation_stats"
    
    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 外鍵
    capability_key = Column(String(500), ForeignKey('capability_records.key'), nullable=False, index=True)
    
    # 統計信息
    invocation_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    
    # 性能信息
    avg_duration_ms = Column(Float)
    last_invoked_at = Column(DateTime)
    
    # 時間戳
    updated_at = Column(DateTime, nullable=False)
    
    # 關係
    capability = relationship("CapabilityRecord", back_populates="stats")
    
    # 索引
    __table_args__ = (
        Index('idx_stats_key', 'capability_key'),
    )
