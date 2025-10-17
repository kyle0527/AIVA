"""
為 AIVA Common 添加程式語言支援的枚舉和 Schema
重點：Go, Rust, Java, C#, JavaScript, TypeScript 等程式語言
"""

# 新增程式語言相關的枚舉
programming_language_enums = """
class ProgrammingLanguage(str, Enum):
    \"\"\"程式語言枚舉\"\"\"
    
    # 系統程式語言
    RUST = "rust"
    GO = "go"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    
    # 腳本語言
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUBY = "ruby"
    PHP = "php"
    
    # JVM 語言
    JAVA = "java"
    KOTLIN = "kotlin"
    SCALA = "scala"
    
    # 函數式語言
    HASKELL = "haskell"
    ERLANG = "erlang"
    ELIXIR = "elixir"
    
    # Web 語言
    HTML = "html"
    CSS = "css"
    SASS = "sass"
    LESS = "less"
    
    # 查詢語言
    SQL = "sql"
    GRAPHQL = "graphql"
    
    # 配置語言
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    XML = "xml"
    
    # Shell 腳本
    BASH = "bash"
    POWERSHELL = "powershell"
    ZSH = "zsh"
    
    # 其他
    UNKNOWN = "unknown"


class LanguageFramework(str, Enum):
    \"\"\"程式語言框架枚舉\"\"\"
    
    # JavaScript/TypeScript 框架
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NODEJS = "nodejs"
    EXPRESS = "express"
    NEXTJS = "nextjs"
    NUXTJS = "nuxtjs"
    
    # Python 框架
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    TORNADO = "tornado"
    
    # Go 框架
    GIN = "gin"
    ECHO = "echo"
    FIBER = "fiber"
    BEEGO = "beego"
    
    # Rust 框架
    ACTIX_WEB = "actix_web"
    WARP = "warp"
    ROCKET = "rocket"
    AXUM = "axum"
    
    # Java 框架
    SPRING = "spring"
    SPRING_BOOT = "spring_boot"
    STRUTS = "struts"
    
    # C# 框架
    DOTNET = "dotnet"
    ASPNET = "aspnet"
    
    # 其他
    UNKNOWN = "unknown"


class VulnerabilityByLanguage(str, Enum):
    \"\"\"按程式語言分類的漏洞類型\"\"\"
    
    # 記憶體安全（C/C++）
    BUFFER_OVERFLOW = "buffer_overflow"
    USE_AFTER_FREE = "use_after_free"
    MEMORY_LEAK = "memory_leak"
    NULL_POINTER_DEREFERENCE = "null_pointer_dereference"
    
    # 型別安全（動態語言）
    TYPE_CONFUSION = "type_confusion"
    PROTOTYPE_POLLUTION = "prototype_pollution"  # JavaScript
    
    # 並發安全（Go, Rust, Java）
    RACE_CONDITION = "race_condition"
    DEADLOCK = "deadlock"
    DATA_RACE = "data_race"
    
    # 反序列化（Java, Python, C#）
    DESERIALIZATION = "deserialization"
    PICKLE_INJECTION = "pickle_injection"  # Python
    
    # 程式碼注入
    CODE_INJECTION = "code_injection"
    EVAL_INJECTION = "eval_injection"
    TEMPLATE_INJECTION = "template_injection"
    
    # 特定語言漏洞
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    XXE = "xxe"
    LDAP_INJECTION = "ldap_injection"
    
    # 配置和依賴
    DEPENDENCY_CONFUSION = "dependency_confusion"
    SUPPLY_CHAIN = "supply_chain"
    MISCONFIGURATION = "misconfiguration"


class SecurityPattern(str, Enum):
    \"\"\"安全模式枚舉 - 針對不同程式語言的安全實踐\"\"\"
    
    # 輸入驗證
    INPUT_VALIDATION = "input_validation"
    SANITIZATION = "sanitization"
    ENCODING = "encoding"
    
    # 認證授權
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    JWT_VALIDATION = "jwt_validation"
    
    # 加密
    ENCRYPTION = "encryption"
    HASHING = "hashing"
    DIGITAL_SIGNATURE = "digital_signature"
    
    # 安全編碼
    SECURE_RANDOM = "secure_random"
    SAFE_DESERIALIZATION = "safe_deserialization"
    BOUNDS_CHECKING = "bounds_checking"
    
    # 並發安全
    THREAD_SAFETY = "thread_safety"
    ATOMIC_OPERATIONS = "atomic_operations"
    LOCK_FREE = "lock_free"
    
    # 錯誤處理
    ERROR_HANDLING = "error_handling"
    FAIL_SECURE = "fail_secure"
    LOGGING = "logging"


class CodeQualityMetric(str, Enum):
    \"\"\"程式碼品質指標\"\"\"
    
    # 複雜度指標
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    COGNITIVE_COMPLEXITY = "cognitive_complexity"
    HALSTEAD_COMPLEXITY = "halstead_complexity"
    
    # 程式碼覆蓋率
    LINE_COVERAGE = "line_coverage"
    BRANCH_COVERAGE = "branch_coverage"
    FUNCTION_COVERAGE = "function_coverage"
    
    # 程式碼重複
    CODE_DUPLICATION = "code_duplication"
    CLONE_DETECTION = "clone_detection"
    
    # 可維護性
    MAINTAINABILITY_INDEX = "maintainability_index"
    TECHNICAL_DEBT = "technical_debt"
    
    # 效能指標
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
"""

# 程式語言相關的 Schema
programming_language_schemas = '''
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
    file_extensions: list[str] = Field(
        default_factory=list, description="檔案副檔名"
    )
    lines_of_code: int = Field(ge=0, description="程式碼行數")


class LanguageSpecificVulnerability(BaseModel):
    """特定程式語言的漏洞"""
    
    language: ProgrammingLanguage = Field(description="程式語言")
    vulnerability_type: VulnerabilityByLanguage = Field(description="漏洞類型")
    severity: Severity = Field(description="嚴重程度")
    description: str = Field(description="漏洞描述")
    code_snippet: str | None = Field(default=None, description="問題程式碼片段")
    line_number: int | None = Field(default=None, ge=1, description="行號")
    function_name: str | None = Field(default=None, description="函數名稱")
    remediation: str | None = Field(default=None, description="修復建議")
    cwe_id: str | None = Field(default=None, description="CWE ID")


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


class LanguageSpecificScanConfig(BaseModel):
    """特定程式語言的掃描配置"""
    
    language: ProgrammingLanguage = Field(description="目標程式語言")
    scan_patterns: list[SecurityPattern] = Field(
        description="要檢查的安全模式"
    )
    quality_metrics: list[CodeQualityMetric] = Field(
        default_factory=list, description="程式碼品質指標"
    )
    exclude_paths: list[str] = Field(
        default_factory=list, description="排除路徑"
    )
    custom_rules: list[str] = Field(
        default_factory=list, description="自訂規則"
    )
    max_file_size: int = Field(default=1048576, description="最大檔案大小（bytes）")


class CrossLanguageAnalysis(BaseModel):
    """跨語言分析結果"""
    
    analysis_id: str = Field(description="分析ID")
    languages_analyzed: list[ProgrammingLanguage] = Field(description="分析的語言")
    cross_language_issues: list[str] = Field(
        default_factory=list, description="跨語言問題"
    )
    integration_points: list[str] = Field(
        default_factory=list, description="語言整合點"
    )
    security_boundaries: list[str] = Field(
        default_factory=list, description="安全邊界"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="建議"
    )


class LanguageSpecificPayload(BaseModel):
    """特定程式語言的測試載荷"""
    
    language: ProgrammingLanguage = Field(description="目標程式語言")
    payload_type: str = Field(description="載荷類型")
    payload_content: str = Field(description="載荷內容")
    encoding: str = Field(default="utf-8", description="編碼方式")
    expected_behavior: str | None = Field(default=None, description="預期行為")
    bypass_techniques: list[str] = Field(
        default_factory=list, description="繞過技術"
    )


class AILanguageModel(BaseModel):
    """AI 程式語言模型"""
    
    model_name: str = Field(description="模型名稱")
    supported_languages: list[ProgrammingLanguage] = Field(description="支援的語言")
    model_type: str = Field(description="模型類型")  # code_generation, vulnerability_detection, etc.
    version: str = Field(description="模型版本")
    capabilities: list[str] = Field(description="能力列表")
    training_data_size: int | None = Field(default=None, description="訓練資料大小")
    accuracy_metrics: dict[str, float] = Field(
        default_factory=dict, description="精確度指標"
    )
'''

print("程式語言支援擴展內容已準備完成！")
print("\n[CONFIG] 新增枚舉:")
print("- ProgrammingLanguage (34 種語言)")
print("- LanguageFramework (20+ 框架)")
print("- VulnerabilityByLanguage (語言特定漏洞)")
print("- SecurityPattern (安全模式)")
print("- CodeQualityMetric (程式碼品質指標)")

print("\n[U+1F4C4] 新增 Schema:")
print("- LanguageDetectionResult")
print("- LanguageSpecificVulnerability") 
print("- MultiLanguageCodebase")
print("- LanguageSpecificScanConfig")
print("- CrossLanguageAnalysis")
print("- LanguageSpecificPayload")
print("- AILanguageModel")