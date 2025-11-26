"""
Steganography Module Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


# ==================== Enums ====================

class SteganographyMethod(str, Enum):
    """隱寫方法"""
    LSB = "lsb"
    DCT = "dct"
    DWT = "dwt"
    SPREAD_SPECTRUM = "spread_spectrum"
    WHITESPACE = "whitespace"
    EOF = "eof"


class CarrierType(str, Enum):
    """載體類型"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    DOCUMENT = "document"


class ImageFormat(str, Enum):
    """圖片格式"""
    PNG = "png"
    BMP = "bmp"
    JPEG = "jpeg"
    GIF = "gif"
    TIFF = "tiff"


class EncryptionAlgorithm(str, Enum):
    """加密算法"""
    NONE = "none"
    AES_128 = "aes_128"
    AES_256 = "aes_256"
    DES = "des"
    TRIPLE_DES = "triple_des"


# ==================== Data Models ====================

@dataclass
class EmbedConfig:
    """嵌入配置"""
    method: SteganographyMethod
    carrier_file: str
    secret_data: str  # File path or text
    output_file: str
    
    # Security
    password: Optional[str] = None
    encryption: EncryptionAlgorithm = EncryptionAlgorithm.NONE
    
    # Advanced Options
    compression: bool = False
    noise_level: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedResult:
    """嵌入結果"""
    success: bool
    output_file: str
    
    # Statistics
    carrier_size: int = 0
    secret_size: int = 0
    output_size: int = 0
    capacity_used_percent: float = 0.0
    
    # Quality Metrics
    psnr: Optional[float] = None  # Peak Signal-to-Noise Ratio
    ssim: Optional[float] = None  # Structural Similarity Index
    
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractConfig:
    """提取配置"""
    method: SteganographyMethod
    stego_file: str
    output_file: str
    
    # Security
    password: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractResult:
    """提取結果"""
    success: bool
    output_file: str
    
    # Statistics
    extracted_size: int = 0
    
    # Verification
    verified: bool = False
    checksum: Optional[str] = None
    
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DetectionResult:
    """檢測結果"""
    has_hidden_data: bool
    confidence: float  # 0.0 - 1.0
    
    # Detection Details
    method_detected: Optional[SteganographyMethod] = None
    anomaly_score: float = 0.0
    
    # Analysis
    statistics: Dict[str, Any] = field(default_factory=dict)
    suspicious_regions: list = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
