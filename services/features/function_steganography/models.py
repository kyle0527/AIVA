"""
Steganography Module Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

from services.aiva_common.enums import VulnerabilityType, Severity, Confidence
from services.aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


# ==================== Enums ====================

class SteganographyType(str, Enum):
    """隱寫術類型"""
    LSB_IMAGE = "lsb_image"
    DCT_IMAGE = "dct_image"
    AUDIO_LSB = "audio_lsb"
    TEXT_WHITESPACE = "text_whitespace"
    FILE_METADATA = "file_metadata"
    FREQUENCY_DOMAIN = "frequency_domain"


class DetectionMethod(str, Enum):
    """檢測方法"""
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUAL_ANALYSIS = "visual_analysis"
    HISTOGRAM_ANALYSIS = "histogram_analysis"
    CHI_SQUARE_TEST = "chi_square_test"
    AI_DETECTION = "ai_detection"
    ENTROPY_ANALYSIS = "entropy_analysis"


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
    
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedSteganographyDetector:
    """增強型隱寫術檢測器"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    async def comprehensive_detection(self, file_path: Path) -> Dict[str, Any]:
        """全面隱寫術檢測"""
        try:
            logger.info(f"檢測隱寫術: {file_path}")
            
            if not file_path.exists():
                return {
                    'file_path': str(file_path),
                    'status': 'file_not_found',
                    'error': '文件不存在',
                    'analysis_time': datetime.now()
                }
            
            if not self._is_supported_format(file_path):
                return {
                    'file_path': str(file_path),
                    'status': 'unsupported_format',
                    'error': f'不支持的文件格式: {file_path.suffix}',
                    'analysis_time': datetime.now()
                }
            
            detection_results = []
            file_size = file_path.stat().st_size
            
            # 實際文件頭檢測
            with open(file_path, 'rb') as f:
                header = f.read(32)
                
                # 檢測圖片格式
                detected_format = None
                if header.startswith(b'\xFF\xD8\xFF'):
                    detected_format = 'JPEG'
                elif header.startswith(b'\x89PNG\r\n\x1a\n'):
                    detected_format = 'PNG'
                elif header.startswith(b'GIF8'):
                    detected_format = 'GIF'
                elif header.startswith(b'BM'):
                    detected_format = 'BMP'
                
                if detected_format:
                    detection_results.append({
                        'method': DetectionMethod.STATISTICAL_ANALYSIS,
                        'type': SteganographyType.LSB_IMAGE,
                        'detected_format': detected_format,
                        'confidence': 0.8,
                        'suspicious': False
                    })
                
                # LSB統計分析（簡化版）
                if detected_format and file_size < 10 * 1024 * 1024:  # 小於10MB才進行分析
                    try:
                        # 讀取文件末尾檢查異常數據
                        f.seek(-min(1024, file_size), 2)
                        tail_data = f.read()
                        
                        # 檢測尾部是否有異常數據模式
                        zero_ratio = tail_data.count(0) / len(tail_data)
                        
                        if zero_ratio < 0.1:  # 尾部幾乎沒有零字節，可能有隱藏數據
                            detection_results.append({
                                'method': DetectionMethod.ENTROPY_ANALYSIS,
                                'type': SteganographyType.FILE_METADATA,
                                'anomaly': 'high_entropy_tail',
                                'confidence': 0.6,
                                'suspicious': True
                            })
                    except Exception as analysis_error:
                        logger.debug(f"統計分析失敗: {analysis_error}")
            
            return {
                'file_path': str(file_path),
                'file_size': file_size,
                'detection_methods': [m.value for m in DetectionMethod],
                'steganography_types': [t.value for t in SteganographyType],
                'analysis_time': datetime.now(),
                'results': detection_results,
                'status': 'completed',
                'suspicious_findings': len([r for r in detection_results if r.get('suspicious', False)])
            }
        except Exception as e:
            logger.error(f"隱寫術檢測失敗: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """檢查是否為支援的格式"""
        return file_path.suffix.lower() in self.supported_formats
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
