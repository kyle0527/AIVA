"""
Reverse Engineering Module Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

# 使用 aiva_common 標準枚舉
from services.aiva_common.enums import ThreatLevel


# ==================== Enums ====================

class BinaryType(str, Enum):
    """二進制類型"""
    EXE = "exe"
    DLL = "dll"
    SO = "so"
    APK = "apk"
    IPA = "ipa"
    ELF = "elf"
    MACH_O = "macho"


class AnalysisMode(str, Enum):
    """分析模式"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


class DecompilerType(str, Enum):
    """反編譯器類型"""
    JADX = "jadx"
    JD_GUI = "jd-gui"
    GHIDRA = "ghidra"
    IDA_PRO = "ida_pro"
    RADARE2 = "radare2"
    BINARY_NINJA = "binary_ninja"


# ==================== Data Models ====================

@dataclass
class BinaryInfo:
    """二進制檔案資訊"""
    file_path: str
    file_name: str
    file_size: int
    file_hash: str  # SHA256
    
    # Binary Metadata
    binary_type: BinaryType
    architecture: str  # x86, x64, ARM, etc.
    entry_point: Optional[str] = None
    
    # Additional Info
    compiler: Optional[str] = None
    packer: Optional[str] = None
    is_stripped: bool = False
    is_packed: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APKInfo:
    """APK 檔案資訊"""
    file_path: str
    package_name: str
    version_name: str
    version_code: int
    
    # App Info
    app_name: str = ""
    min_sdk: int = 0
    target_sdk: int = 0
    
    # Certificates
    certificates: List[Dict[str, Any]] = field(default_factory=list)
    is_signed: bool = False
    
    # Permissions
    permissions: List[str] = field(default_factory=list)
    dangerous_permissions: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompileConfig:
    """反編譯配置"""
    file_path: str
    decompiler: DecompilerType
    output_dir: str
    
    # Options
    include_resources: bool = True
    export_gradle: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompileResult:
    """反編譯結果"""
    success: bool
    output_dir: str
    
    # Statistics
    classes_decompiled: int = 0
    methods_decompiled: int = 0
    total_files: int = 0
    
    # Quality
    decompile_rate: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MalwareAnalysisResult:
    """惡意軟體分析結果"""
    file_hash: str
    threat_level: ThreatLevel
    
    # Detection
    is_malicious: bool = False
    confidence: float = 0.0
    
    # Indicators
    suspicious_apis: List[str] = field(default_factory=list)
    suspicious_strings: List[str] = field(default_factory=list)
    network_indicators: List[str] = field(default_factory=list)
    
    # Behavior
    behaviors: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    # Report
    report_file: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CodeAnalysisResult:
    """程式碼分析結果"""
    file_path: str
    
    # Code Metrics
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    
    # Complexity
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    
    # Issues
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    code_smells: List[Dict[str, Any]] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
