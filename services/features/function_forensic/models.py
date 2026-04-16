"""
Forensic Module Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ==================== Enums ====================

class ForensicAnalysisType(str, Enum):
    """取證分析類型"""
    DISK_IMAGE = "disk_image"
    MEMORY_DUMP = "memory_dump" 
    NETWORK_CAPTURE = "network_capture"
    FILE_CARVING = "file_carving"
    TIMELINE_ANALYSIS = "timeline_analysis"
    REGISTRY_ANALYSIS = "registry_analysis"
    LOG_ANALYSIS = "log_analysis"


class EvidenceType(str, Enum):
    """證據類型"""
    DISK = "disk"
    MEMORY = "memory"
    NETWORK = "network"
    FILE = "file"
    LOG = "log"
    DATABASE = "database"


class FileSystemType(str, Enum):
    """檔案系統類型"""
    NTFS = "ntfs"
    FAT32 = "fat32"
    EXT4 = "ext4"
    HFS_PLUS = "hfs+"
    APFS = "apfs"
    EXFAT = "exfat"


class ArtifactCategory(str, Enum):
    """痕跡分類"""
    BROWSER_HISTORY = "browser_history"
    EMAIL = "email"
    DOCUMENTS = "documents"
    IMAGES = "images"
    EXECUTABLES = "executables"
    DELETED_FILES = "deleted_files"
    ENCRYPTED_FILES = "encrypted_files"


# ==================== Data Models ====================

@dataclass
class CaseInfo:
    """案件資訊"""
    case_id: str
    case_name: str
    investigator: str
    
    # Case Details
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, in_progress, closed
    
    # Evidence Tracking
    evidence_items: list[str] = field(default_factory=list)
    evidence_location: str = ""
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """證據項"""
    evidence_id: str
    case_id: str
    evidence_type: EvidenceType
    
    # File Info
    file_path: str
    file_size: int
    file_hash: str  # MD5, SHA1, SHA256
    
    # Acquisition
    acquired_at: datetime = field(default_factory=datetime.now)
    acquired_by: str = ""
    acquisition_tool: str = ""
    
    # Verification
    verified: bool = False
    chain_of_custody: list[dict[str, Any]] = field(default_factory=list)
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """分析配置"""
    analysis_type: ForensicAnalysisType
    evidence_id: str
    
    # Analysis Options
    deep_scan: bool = False
    include_deleted: bool = True
    file_carving: bool = False
    
    # Output
    output_dir: str = "analysis_output"
    report_format: str = "html"  # html, pdf, json
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """分析結果"""
    success: bool
    analysis_type: ForensicAnalysisType
    evidence_id: str
    
    # Findings
    artifacts_found: int = 0
    deleted_files_recovered: int = 0
    suspicious_items: int = 0
    
    # Artifacts
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    timeline_events: list[dict[str, Any]] = field(default_factory=list)
    
    # Report
    report_file: str = ""
    log_file: str = ""
    
    # Timing
    analysis_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    error: str | None = None


@dataclass
class TimelineEvent:
    """時間軸事件"""
    timestamp: datetime
    event_type: str
    source: str
    description: str
    
    # Additional Info
    file_path: str | None = None
    user: str | None = None
    process: str | None = None
    
    metadata: dict[str, Any] = field(default_factory=dict)
