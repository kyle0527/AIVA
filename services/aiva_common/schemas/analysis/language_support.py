"""
程式語言分析和多語言支援的 Schema
支援 Go, Rust, Java, C#, JavaScript, TypeScript 等多種程式語言
"""

from pydantic import BaseModel, Field

from ...enums import (
    CodeQualityMetric,
    LanguageFramework,
    ProgrammingLanguage,
    SecurityPattern,
    Severity,
    VulnerabilityByLanguage,
)


class LanguageDetectionResult(BaseModel):
    """程式語言檢測結果"""

    primary_language: ProgrammingLanguage = Field(description="主要程式語言")
    confidence: float = Field(ge=0.0, le=1.0, description="檢測信心度")
    secondary_languages: list[ProgrammingLanguage] = Field(
        default_factory=list, description="次要程式語言"
    )
    frameworks: list[LanguageFramework] = Field(
        default_factory=list, description="檢測到的框架"
    )
    file_extensions: list[str] = Field(default_factory=list, description="檔案副檔名")
    lines_of_code: int = Field(ge=0, description="程式碼行數")


class LanguageSpecificVulnerability(BaseModel):
    """特定程式語言的漏洞"""

    language: ProgrammingLanguage = Field(description="程式語言")
    vulnerability_type: VulnerabilityByLanguage = Field(description="漏洞類型")
    severity: Severity = Field(description="嚴重程度")
    description: str = Field(description="漏洞描述")
    code_snippet: str | None = Field(default=None, description="問題程式碼片段")
    line_number: int | None = Field(default=None, ge=1, description="行號")
    file_path: str | None = Field(default=None, description="檔案路徑")
    function_name: str | None = Field(default=None, description="函數名稱")
    remediation: str | None = Field(default=None, description="修復建議")
    cwe_id: str | None = Field(default=None, description="CWE ID")
    owasp_category: str | None = Field(default=None, description="OWASP 分類")


class MultiLanguageCodebase(BaseModel):
    """多語言程式碼庫分析"""

    project_name: str = Field(description="專案名稱")
    languages: dict[ProgrammingLanguage, int] = Field(
        description="語言分布（語言：程式碼行數）"
    )
    primary_language: ProgrammingLanguage = Field(description="主要程式語言")
    frameworks: list[LanguageFramework] = Field(
        default_factory=list, description="使用的框架"
    )
    total_files: int = Field(ge=0, description="總檔案數")
    total_lines: int = Field(ge=0, description="總程式碼行數")
    vulnerability_distribution: dict[ProgrammingLanguage, int] = Field(
        default_factory=dict, description="各語言漏洞分布"
    )
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict, description="依賴套件（語言：套件列表）"
    )


class LanguageSpecificScanConfig(BaseModel):
    """特定程式語言的掃描配置"""

    language: ProgrammingLanguage = Field(description="目標程式語言")
    scan_patterns: list[SecurityPattern] = Field(description="要檢查的安全模式")
    quality_metrics: list[CodeQualityMetric] = Field(
        default_factory=list, description="程式碼品質指標"
    )
    exclude_paths: list[str] = Field(default_factory=list, description="排除路徑")
    include_patterns: list[str] = Field(
        default_factory=list, description="包含檔案模式"
    )
    custom_rules: list[str] = Field(default_factory=list, description="自訂規則")
    max_file_size: int = Field(default=1048576, description="最大檔案大小（bytes）")
    timeout_seconds: int = Field(default=300, description="掃描超時時間（秒）")


class CrossLanguageAnalysis(BaseModel):
    """跨語言分析結果"""

    analysis_id: str = Field(description="分析ID")
    project_name: str = Field(description="專案名稱")
    languages_analyzed: list[ProgrammingLanguage] = Field(description="分析的語言")
    cross_language_issues: list[str] = Field(
        default_factory=list, description="跨語言問題"
    )
    integration_points: list[str] = Field(
        default_factory=list, description="語言整合點"
    )
    security_boundaries: list[str] = Field(default_factory=list, description="安全邊界")
    data_flow_risks: list[str] = Field(default_factory=list, description="資料流風險")
    recommendations: list[str] = Field(default_factory=list, description="建議")
    risk_score: float = Field(ge=0.0, le=10.0, description="風險評分")


class LanguageSpecificPayload(BaseModel):
    """特定程式語言的測試載荷"""

    language: ProgrammingLanguage = Field(description="目標程式語言")
    payload_type: str = Field(description="載荷類型")
    payload_content: str = Field(description="載荷內容")
    encoding: str = Field(default="utf-8", description="編碼方式")
    expected_behavior: str | None = Field(default=None, description="預期行為")
    bypass_techniques: list[str] = Field(default_factory=list, description="繞過技術")
    target_functions: list[str] = Field(default_factory=list, description="目標函數")
    success_indicators: list[str] = Field(default_factory=list, description="成功指標")


class AILanguageModel(BaseModel):
    """AI 程式語言模型配置"""

    model_config = {"protected_namespaces": ()}

    model_name: str = Field(description="模型名稱")
    supported_languages: list[ProgrammingLanguage] = Field(description="支援的語言")
    model_type: str = Field(
        description="模型類型"
    )  # code_generation, vulnerability_detection, etc.
    version: str = Field(description="模型版本")
    capabilities: list[str] = Field(description="能力列表")
    training_data_size: int | None = Field(default=None, description="訓練資料大小")
    accuracy_metrics: dict[str, float] = Field(
        default_factory=dict, description="精確度指標"
    )
    api_endpoint: str | None = Field(default=None, description="API 端點")
    authentication_required: bool = Field(default=True, description="是否需要認證")


class CodeQualityReport(BaseModel):
    """程式碼品質報告"""

    language: ProgrammingLanguage = Field(description="程式語言")
    file_path: str = Field(description="檔案路徑")
    metrics: dict[CodeQualityMetric, float] = Field(description="品質指標")
    issues: list[str] = Field(default_factory=list, description="發現的問題")
    suggestions: list[str] = Field(default_factory=list, description="改進建議")
    overall_score: float = Field(ge=0.0, le=100.0, description="整體評分")
    timestamp: str = Field(description="分析時間")


class LanguageInteroperability(BaseModel):
    """語言互操作性分析"""

    source_language: ProgrammingLanguage = Field(description="來源語言")
    target_language: ProgrammingLanguage = Field(description="目標語言")
    interop_method: str = Field(description="互操作方法")  # FFI, JNI, WebAssembly, etc.
    security_considerations: list[str] = Field(
        default_factory=list, description="安全考量"
    )
    performance_impact: str | None = Field(default=None, description="效能影響")
    compatibility_issues: list[str] = Field(
        default_factory=list, description="相容性問題"
    )
    recommendations: list[str] = Field(default_factory=list, description="建議")
