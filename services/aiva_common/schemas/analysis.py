"""
AIVA 分析結果統一標準模型

提供所有代碼分析結果的統一基礎架構，
確保跨服務的一致性和可維護性。
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..enums.common import Severity


class AnalysisType(str, Enum):
    """分析類型枚舉"""
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    JAVA = "java"
    CPP = "cpp"
    GENERIC = "generic"


class DataLeak(BaseModel):
    """資料洩漏詳情結構化模型"""
    
    leak_type: str = Field(description="洩漏類型 (e.g., 'api_key', 'password', 'token')")
    description: str = Field(description="洩漏描述")
    severity: Severity = Field(default=Severity.MEDIUM, description="洩漏嚴重程度")
    location: str | None = Field(default=None, description="洩漏位置 (行號或函數名)")
    pattern_matched: str | None = Field(default=None, description="匹配的模式")
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True


class BaseAnalysisResult(BaseModel):
    """所有分析結果的基礎模型
    
    提供通用欄位和標準化結構，支援組合式繼承模式。
    遵循 Pydantic v2 最佳實踐。
    """
    
    # 核心識別欄位
    analysis_id: str = Field(description="分析唯一識別ID")
    url: str = Field(description="分析目標URL或文件路徑")
    analysis_type: AnalysisType = Field(description="分析類型")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="分析完成時間戳 (UTC)"
    )
    
    # 基礎評分系統
    risk_score: float = Field(
        ge=0.0, le=10.0, default=0.0,
        description="風險評分 (0.0-10.0, 0為最低風險)"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="分析置信度 (0.0-1.0, 1為最高置信度)"
    )
    
    # 通用發現
    findings: list[str] = Field(
        default_factory=list,
        description="通用發現列表"
    )
    
    # 元數據和擴展欄位
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="擴展元數據，支援自定義欄位"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True
        validate_assignment = True


class JavaScriptAnalysisResult(BaseAnalysisResult):
    """JavaScript 代碼分析結果統一標準
    
    整合所有現有 JavaScriptAnalysisResult 模型的功能，
    提供完整的 JavaScript 安全分析結果結構。
    """
    
    # 預設分析類型
    analysis_type: AnalysisType = Field(default=AnalysisType.JAVASCRIPT)
    
    # 源碼基礎信息
    source_size_bytes: int = Field(
        ge=0, default=0,
        description="源代碼大小 (字節)"
    )
    has_sensitive_data: bool = Field(
        default=False,
        description="是否包含敏感數據"
    )
    
    # 安全分析欄位
    dangerous_functions: list[str] = Field(
        default_factory=list,
        description="檢測到的危險函數 (e.g., eval, Function, setTimeout with string)"
    )
    sensitive_patterns: list[str] = Field(
        default_factory=list,
        description="敏感模式匹配結果"
    )
    
    # API與網路端點
    api_endpoints: list[str] = Field(
        default_factory=list,
        description="發現的API端點"
    )
    ajax_endpoints: list[str] = Field(
        default_factory=list,
        description="AJAX調用端點"
    )
    external_resources: list[str] = Field(
        default_factory=list,
        description="外部資源URL"
    )
    
    # 前端特定分析
    dom_sinks: list[str] = Field(
        default_factory=list,
        description="DOM接收器 (潛在XSS風險點)"
    )
    cookies_accessed: list[str] = Field(
        default_factory=list,
        description="存取的Cookie名稱"
    )
    
    # 結構化資料洩漏
    data_leaks: list[DataLeak] = Field(
        default_factory=list,
        description="結構化資料洩漏詳情"
    )
    
    # 可疑行為模式
    suspicious_patterns: list[str] = Field(
        default_factory=list,
        description="可疑行為模式"
    )
    
    # 完整評分系統
    security_score: int = Field(
        ge=0, le=100, default=0,
        description="整體安全評分 (0-100, 100為最安全)"
    )
    
    # 向後兼容欄位 (保留但已棄用)
    apis_called: list[str] = Field(
        default_factory=list,
        description="[已棄用] 請使用 api_endpoints"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True
        validate_assignment = True


class LegacyJavaScriptAnalysisResultAdapter:
    """向後兼容適配器
    
    支援從舊版 JavaScriptAnalysisResult 模型轉換到新的統一標準。
    確保無縫升級和現有代碼的兼容性。
    """
    
    @staticmethod
    def from_scan_models(legacy_data: dict[str, Any]) -> JavaScriptAnalysisResult:
        """從 services/scan/models.py 的 JavaScriptAnalysisResult 轉換"""
        return JavaScriptAnalysisResult(
            analysis_id=legacy_data.get("analysis_id", ""),
            url=legacy_data.get("url", ""),
            source_size_bytes=legacy_data.get("source_size_bytes", 0),
            dangerous_functions=legacy_data.get("dangerous_functions", []),
            external_resources=legacy_data.get("external_resources", []),
            data_leaks=[
                DataLeak(
                    leak_type=leak.get("type", "unknown"),
                    description=leak.get("description", ""),
                    location=leak.get("location")
                )
                for leak in legacy_data.get("data_leaks", [])
            ],
            findings=legacy_data.get("findings", []),
            api_endpoints=legacy_data.get("apis_called", []),
            ajax_endpoints=legacy_data.get("ajax_endpoints", []),
            suspicious_patterns=legacy_data.get("suspicious_patterns", []),
            risk_score=legacy_data.get("risk_score", 0.0),
            security_score=legacy_data.get("security_score", 100),
        )
    
    @staticmethod
    def from_aiva_scan_schemas(legacy_data: dict[str, Any]) -> JavaScriptAnalysisResult:
        """從 services/scan/aiva_scan/schemas.py 的 JavaScriptAnalysisResult 轉換"""
        return JavaScriptAnalysisResult(
            analysis_id=f"scan_{legacy_data.get('url', '').replace('/', '_')}",
            url=legacy_data.get("url", ""),
            has_sensitive_data=legacy_data.get("has_sensitive_data", False),
            api_endpoints=legacy_data.get("api_endpoints", []),
            dom_sinks=legacy_data.get("dom_sinks", []),
            dangerous_functions=legacy_data.get("sensitive_functions", []),
            external_resources=legacy_data.get("external_requests", []),
            cookies_accessed=legacy_data.get("cookies_accessed", []),
        )


# 導出統一接口
__all__ = [
    "BaseAnalysisResult",
    "JavaScriptAnalysisResult", 
    "DataLeak",
    "AnalysisType",
    "LegacyJavaScriptAnalysisResultAdapter",
]