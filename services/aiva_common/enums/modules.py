"""
模組相關枚舉 - 模組名稱、主題等
"""

from enum import Enum


class ProgrammingLanguage(str, Enum):
    """程式語言枚舉"""

    # 系統程式語言 - 使用官方正式名稱
    RUST = "rust"  # 官方名稱: Rust
    GO = "go"  # 官方名稱: Go (不是 Golang)
    C = "c"  # 官方名稱: C
    CPP = "c++"  # 官方名稱: C++ (不是 cpp) - ISO/IEC 14882 標準
    CSHARP = "c#"  # 官方名稱: C# (不是 csharp) - Microsoft 官方標準

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


class ECMAScriptVersion(str, Enum):
    """ECMAScript 版本枚舉（包含最新標準，基於 ECMA-262 官方標準）"""

    ES3 = "ES3"  # 1999
    ES5 = "ES5"  # 2009
    ES5_1 = "ES5.1"  # 2011
    ES2015 = "ES2015"  # ES6, 2015
    ES2016 = "ES2016"  # ES7, 2016
    ES2017 = "ES2017"  # ES8, 2017
    ES2018 = "ES2018"  # ES9, 2018
    ES2019 = "ES2019"  # ES10, 2019
    ES2020 = "ES2020"  # ES11, 2020
    ES2021 = "ES2021"  # ES12, 2021
    ES2022 = "ES2022"  # ES13, 2022
    ES2023 = "ES2023"  # ES14, 2023
    ES2024 = "ES2024"  # ES15, 2024
    ES2025 = "ES2025"  # ES16, 2025
    ES2026 = "ES2026"  # ES17, 2026 (預期)


class JavaScriptFeature(str, Enum):
    """JavaScript 特性枚舉（基於 ECMA-262 官方規範）"""

    # ES2015+ Features
    ARROW_FUNCTIONS = "arrow_functions"
    CLASSES = "classes"
    TEMPLATE_LITERALS = "template_literals"
    DESTRUCTURING = "destructuring"
    DEFAULT_PARAMETERS = "default_parameters"
    REST_PARAMETERS = "rest_parameters"
    SPREAD_OPERATOR = "spread_operator"
    LET_CONST = "let_const"
    FOR_OF = "for_of"
    PROMISES = "promises"
    MODULES = "modules"
    MAP_SET = "map_set"
    SYMBOLS = "symbols"
    ITERATORS = "iterators"
    GENERATORS = "generators"

    # ES2017+
    ASYNC_AWAIT = "async_await"
    OBJECT_VALUES_ENTRIES = "object_values_entries"
    STRING_PADDING = "string_padding"
    TRAILING_COMMAS = "trailing_commas"

    # ES2018+
    REST_SPREAD_PROPERTIES = "rest_spread_properties"
    ASYNC_ITERATION = "async_iteration"
    PROMISE_FINALLY = "promise_finally"
    REGEXP_FEATURES = "regexp_features"

    # ES2019+
    ARRAY_FLAT = "array_flat"
    OBJECT_FROM_ENTRIES = "object_from_entries"
    STRING_TRIM_START_END = "string_trim_start_end"
    OPTIONAL_CATCH_BINDING = "optional_catch_binding"
    JSON_SUPERSET = "json_superset"

    # ES2020+
    BIGINT = "bigint"
    DYNAMIC_IMPORT = "dynamic_import"
    NULLISH_COALESCING = "nullish_coalescing"
    OPTIONAL_CHAINING = "optional_chaining"
    PROMISE_ALL_SETTLED = "promise_all_settled"
    GLOBAL_THIS = "global_this"

    # ES2021+
    LOGICAL_ASSIGNMENT = "logical_assignment"
    NUMERIC_SEPARATORS = "numeric_separators"
    PROMISE_ANY = "promise_any"
    STRING_REPLACE_ALL = "string_replace_all"
    WEAK_REFS = "weak_refs"

    # ES2022+
    CLASS_FIELDS = "class_fields"
    PRIVATE_METHODS = "private_methods"
    STATIC_CLASS_FIELDS = "static_class_fields"
    REGEXP_MATCH_INDICES = "regexp_match_indices"
    TOP_LEVEL_AWAIT = "top_level_await"
    ARRAY_AT = "array_at"
    ERROR_CAUSE = "error_cause"

    # ES2023+
    ARRAY_FIND_LAST = "array_find_last"
    HASHBANG_GRAMMAR = "hashbang_grammar"
    SYMBOLS_AS_WEAK_MAP_KEYS = "symbols_as_weak_map_keys"

    # ES2024+
    ARRAY_BUFFER_RESIZE = "array_buffer_resize"
    REGEXP_V_FLAG = "regexp_v_flag"
    PROMISE_WITH_RESOLVERS = "promise_with_resolvers"

    # ES2025+ (预期特性)
    TEMPORAL = "temporal"
    RECORDS_TUPLES = "records_tuples"
    PATTERN_MATCHING = "pattern_matching"
    DECIMAL = "decimal"

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
    """程式語言框架枚舉"""

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


class CodeQualityMetric(str, Enum):
    """程式碼品質指標"""

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


class ModuleName(str, Enum):
    API_GATEWAY = "ApiGateway"
    CORE = "CoreModule"
    SCAN = "ScanModule"
    INTEGRATION = "IntegrationModule"
    FUNCTION = "FunctionModule"
    FUNC_XSS = "FunctionXSS"
    FUNC_SQLI = "FunctionSQLI"
    FUNC_SSRF = "FunctionSSRF"
    FUNC_IDOR = "FunctionIDOR"
    FUNC_CLIENT_AUTH_BYPASS = "FunctionClientAuthBypass"
    OAST = "OASTService"
    THREAT_INTEL = "ThreatIntelModule"
    AUTHZ = "AuthZModule"
    POSTEX = "PostExModule"
    REMEDIATION = "RemediationModule"
    BIZLOGIC = "BizLogicModule"
    COMMON = "CommonModule"


class Topic(str, Enum):
    # Scan Topics
    SCAN_START = "tasks.scan.start"  # 別名，向後兼容
    TASK_SCAN_START = "tasks.scan.start"
    RESULTS_SCAN_COMPLETED = "results.scan.completed"

    # Function Topics
    TASK_FUNCTION_START = "tasks.function.start"
    TASK_FUNCTION_XSS = "tasks.function.xss"
    TASK_FUNCTION_SQLI = "tasks.function.sqli"
    TASK_FUNCTION_SSRF = "tasks.function.ssrf"
    FUNCTION_IDOR_TASK = "tasks.function.idor"
    TASK_FUNCTION_CLIENT_AUTH_BYPASS = "tasks.function.client_auth_bypass"
    RESULTS_FUNCTION_COMPLETED = "results.function.completed"

    # AI Training Topics
    TASK_AI_TRAINING_START = "tasks.ai.training.start"
    TASK_AI_TRAINING_EPISODE = "tasks.ai.training.episode"
    TASK_AI_TRAINING_STOP = "tasks.ai.training.stop"
    RESULTS_AI_TRAINING_PROGRESS = "results.ai.training.progress"
    RESULTS_AI_TRAINING_COMPLETED = "results.ai.training.completed"
    RESULTS_AI_TRAINING_FAILED = "results.ai.training.failed"

    # AI Experience & Learning Topics
    EVENT_AI_EXPERIENCE_CREATED = "events.ai.experience.created"
    EVENT_AI_TRACE_COMPLETED = "events.ai.trace.completed"
    EVENT_AI_MODEL_UPDATED = "events.ai.model.updated"
    COMMAND_AI_MODEL_DEPLOY = "commands.ai.model.deploy"

    # RAG Knowledge Topics
    TASK_RAG_KNOWLEDGE_UPDATE = "tasks.rag.knowledge.update"
    TASK_RAG_QUERY = "tasks.rag.query"
    RESULTS_RAG_RESPONSE = "results.rag.response"

    # General Topics
    FINDING_DETECTED = "findings.detected"
    LOG_RESULTS_ALL = "log.results.all"
    STATUS_TASK_UPDATE = "status.task.update"
    FEEDBACK_CORE_STRATEGY = "feedback.core.strategy"
    MODULE_HEARTBEAT = "module.heartbeat"
    COMMAND_TASK_CANCEL = "command.task.cancel"
    CONFIG_GLOBAL_UPDATE = "config.global.update"

    # ThreatIntel Topics
    TASK_THREAT_INTEL_LOOKUP = "tasks.threat_intel.lookup"
    TASK_IOC_ENRICHMENT = "tasks.threat_intel.ioc_enrichment"
    TASK_MITRE_MAPPING = "tasks.threat_intel.mitre_mapping"
    RESULTS_THREAT_INTEL = "results.threat_intel"

    # Scan Progress & Failure Topics (新增)
    RESULTS_SCAN_PROGRESS = "results.scan.progress"
    RESULTS_SCAN_FAILED = "results.scan.failed"
    EVENT_SCAN_ASSET_DISCOVERED = "events.scan.asset.discovered"

    # Function Progress & Failure Topics (新增)
    RESULTS_FUNCTION_PROGRESS = "results.function.progress"
    RESULTS_FUNCTION_FAILED = "results.function.failed"
    EVENT_FUNCTION_VULN_FOUND = "events.function.vuln.found"

    # Integration Analysis Topics (新增)
    TASK_INTEGRATION_ANALYSIS_START = "tasks.integration.analysis.start"
    RESULTS_INTEGRATION_ANALYSIS_PROGRESS = "results.integration.analysis.progress"
    RESULTS_INTEGRATION_ANALYSIS_COMPLETED = "results.integration.analysis.completed"
    COMMAND_INTEGRATION_REPORT_GENERATE = "commands.integration.report.generate"
    EVENT_INTEGRATION_REPORT_GENERATED = "events.integration.report.generated"

    # AI Scenario Topics (新增)
    EVENT_AI_SCENARIO_LOADED = "events.ai.scenario.loaded"

    # AuthZ Topics
    TASK_AUTHZ_CHECK = "tasks.authz.check"
    TASK_AUTHZ_ANALYZE = "tasks.authz.analyze"
    RESULTS_AUTHZ = "results.authz"

    # PostEx Topics (僅用於授權測試環境)
    TASK_POSTEX_TEST = "tasks.postex.test"
    TASK_POSTEX_PRIVILEGE_ESCALATION = "tasks.postex.privilege_escalation"
    TASK_POSTEX_LATERAL_MOVEMENT = "tasks.postex.lateral_movement"
    TASK_POSTEX_DATA_EXFILTRATION = "tasks.postex.data_exfiltration"
    TASK_POSTEX_PERSISTENCE = "tasks.postex.persistence"
    RESULTS_POSTEX = "results.postex"

    # Remediation Topics
    TASK_REMEDIATION_GENERATE = "tasks.remediation.generate"
    RESULTS_REMEDIATION = "results.remediation"
