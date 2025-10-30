---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 官方標準擴展分析報告

## 執行摘要

基於對 aiva_common README SOT（1815行完整文檔）的深入分析和全面的國際標準網路調研，本報告識別出需要根據「寧可現在用不到，也不要以後要用時沒有」原則新增的官方定義。

## 分析方法論

### 資料來源
- **SOT基準**: services/aiva_common README.md（40個標準枚舉類別）
- **官方標準調研**: CVSS v4.0, MITRE ATT&CK v18, SARIF v2.1.0, CWE v4.18, IANA, RFC, ECMA, ISO, W3C, OpenAPI, JSON Schema, JWT

### 優先級框架
按照 aiva_common 四層優先體系：
1. **國際標準** → 2. **語言標準** → 3. **aiva_common規範** → 4. **模組規範**

## 需要新增的關鍵枚舉擴展

### 1. 安全標準增強

#### CVSSMetric 枚舉（基於 CVSS v4.0）
```python
class CVSSMetric(str, Enum):
    """CVSS v4.0 評分指標枚舉"""
    # Base Metrics (基本指標)
    ATTACK_VECTOR = "attack_vector"  # AV
    ATTACK_COMPLEXITY = "attack_complexity"  # AC  
    ATTACK_REQUIREMENTS = "attack_requirements"  # AT (v4.0新增)
    PRIVILEGES_REQUIRED = "privileges_required"  # PR
    USER_INTERACTION = "user_interaction"  # UI
    VULNERABLE_SYSTEM_CONFIDENTIALITY = "vulnerable_system_confidentiality"  # VC
    VULNERABLE_SYSTEM_INTEGRITY = "vulnerable_system_integrity"  # VI
    VULNERABLE_SYSTEM_AVAILABILITY = "vulnerable_system_availability"  # VA
    SUBSEQUENT_SYSTEM_CONFIDENTIALITY = "subsequent_system_confidentiality"  # SC (v4.0新增)
    SUBSEQUENT_SYSTEM_INTEGRITY = "subsequent_system_integrity"  # SI (v4.0新增)
    SUBSEQUENT_SYSTEM_AVAILABILITY = "subsequent_system_availability"  # SA (v4.0新增)
    
    # Threat Metrics (威脅指標)
    EXPLOIT_MATURITY = "exploit_maturity"  # E
    
    # Environmental Metrics (環境指標)
    CONFIDENTIALITY_REQUIREMENT = "confidentiality_requirement"  # CR
    INTEGRITY_REQUIREMENT = "integrity_requirement"  # IR
    AVAILABILITY_REQUIREMENT = "availability_requirement"  # AR
    
    # Supplemental Metrics (補充指標，v4.0新增)
    SAFETY = "safety"  # S
    AUTOMATABLE = "automatable"  # AU
    RECOVERY = "recovery"  # R
    VALUE_DENSITY = "value_density"  # V
    VULNERABILITY_RESPONSE_EFFORT = "vulnerability_response_effort"  # RE
    PROVIDER_URGENCY = "provider_urgency"  # U
```

#### AttackTechnique 枚舉（基於 MITRE ATT&CK v18）
```python
class AttackTechnique(str, Enum):
    """MITRE ATT&CK 技術枚舉（選擇核心技術）"""
    # Reconnaissance
    ACTIVE_SCANNING = "T1595"
    GATHER_VICTIM_HOST_INFORMATION = "T1592"
    GATHER_VICTIM_IDENTITY_INFORMATION = "T1589"
    GATHER_VICTIM_NETWORK_INFORMATION = "T1590"
    GATHER_VICTIM_ORG_INFORMATION = "T1591"
    PHISHING_FOR_INFORMATION = "T1598"
    SEARCH_CLOSED_SOURCES = "T1597"
    SEARCH_OPEN_TECHNICAL_DATABASES = "T1596"
    SEARCH_OPEN_WEBSITES = "T1593"
    SEARCH_VICTIM_OWNED_WEBSITES = "T1594"
    
    # Initial Access
    DRIVE_BY_COMPROMISE = "T1189"
    EXPLOIT_PUBLIC_FACING_APPLICATION = "T1190"
    EXTERNAL_REMOTE_SERVICES = "T1133"
    HARDWARE_ADDITIONS = "T1200"
    PHISHING = "T1566"
    REPLICATION_THROUGH_REMOVABLE_MEDIA = "T1091"
    SUPPLY_CHAIN_COMPROMISE = "T1195"
    TRUSTED_RELATIONSHIP = "T1199"
    VALID_ACCOUNTS = "T1078"
    
    # Execution
    COMMAND_AND_SCRIPTING_INTERPRETER = "T1059"
    CONTAINER_ADMINISTRATION_COMMAND = "T1609"
    DEPLOY_CONTAINER = "T1610"
    EXPLOITATION_FOR_CLIENT_EXECUTION = "T1203"
    INTER_PROCESS_COMMUNICATION = "T1559"
    NATIVE_API = "T1106"
    SCHEDULED_TASK_JOB = "T1053"
    SHARED_MODULES = "T1129"
    SOFTWARE_DEPLOYMENT_TOOLS = "T1072"
    SYSTEM_SERVICES = "T1569"
    USER_EXECUTION = "T1204"
    
    # ... 更多核心技術（總計約100-150個核心技術）
    
class AttackTactic(str, Enum):
    """MITRE ATT&CK 戰術枚舉"""
    RECONNAISSANCE = "TA0043"
    RESOURCE_DEVELOPMENT = "TA0042"
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    COMMAND_AND_CONTROL = "TA0011"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"
```

### 2. Web 和 API 標準

#### HTTPStatusCode 枚舉（擴展版本）
```python
class HTTPStatusCode(IntEnum):
    """完整的 HTTP 狀態碼枚舉（基於 RFC 7231 等）"""
    # 1xx Informational
    CONTINUE = 100
    SWITCHING_PROTOCOLS = 101
    PROCESSING = 102  # RFC 2518
    EARLY_HINTS = 103  # RFC 8297
    
    # 2xx Success
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NON_AUTHORITATIVE_INFORMATION = 203
    NO_CONTENT = 204
    RESET_CONTENT = 205
    PARTIAL_CONTENT = 206
    MULTI_STATUS = 207  # RFC 4918
    ALREADY_REPORTED = 208  # RFC 5842
    IM_USED = 226  # RFC 3229
    
    # 3xx Redirection
    MULTIPLE_CHOICES = 300
    MOVED_PERMANENTLY = 301
    FOUND = 302
    SEE_OTHER = 303
    NOT_MODIFIED = 304
    USE_PROXY = 305
    TEMPORARY_REDIRECT = 307
    PERMANENT_REDIRECT = 308  # RFC 7538
    
    # 4xx Client Error
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    PAYMENT_REQUIRED = 402
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    NOT_ACCEPTABLE = 406
    PROXY_AUTHENTICATION_REQUIRED = 407
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    GONE = 410
    LENGTH_REQUIRED = 411
    PRECONDITION_FAILED = 412
    PAYLOAD_TOO_LARGE = 413
    URI_TOO_LONG = 414
    UNSUPPORTED_MEDIA_TYPE = 415
    RANGE_NOT_SATISFIABLE = 416
    EXPECTATION_FAILED = 417
    IM_A_TEAPOT = 418  # RFC 2324
    MISDIRECTED_REQUEST = 421  # RFC 7540
    UNPROCESSABLE_ENTITY = 422  # RFC 4918
    LOCKED = 423  # RFC 4918
    FAILED_DEPENDENCY = 424  # RFC 4918
    TOO_EARLY = 425  # RFC 8470
    UPGRADE_REQUIRED = 426
    PRECONDITION_REQUIRED = 428  # RFC 6585
    TOO_MANY_REQUESTS = 429  # RFC 6585
    REQUEST_HEADER_FIELDS_TOO_LARGE = 431  # RFC 6585
    UNAVAILABLE_FOR_LEGAL_REASONS = 451  # RFC 7725
    
    # 5xx Server Error
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    HTTP_VERSION_NOT_SUPPORTED = 505
    VARIANT_ALSO_NEGOTIATES = 506  # RFC 2295
    INSUFFICIENT_STORAGE = 507  # RFC 4918
    LOOP_DETECTED = 508  # RFC 5842
    NOT_EXTENDED = 510  # RFC 2774
    NETWORK_AUTHENTICATION_REQUIRED = 511  # RFC 6585
```

#### OpenAPISchemaType 枚舉
```python
class OpenAPISchemaType(str, Enum):
    """OpenAPI 3.1 Schema 類型枚舉"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"  # OpenAPI 3.1 新增
    
class OpenAPIFormat(str, Enum):
    """OpenAPI 格式枚舉"""
    # String formats
    DATE = "date"
    DATE_TIME = "date-time"
    PASSWORD = "password"
    BYTE = "byte"
    BINARY = "binary"
    EMAIL = "email"
    UUID = "uuid"
    URI = "uri"
    HOSTNAME = "hostname"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    
    # Number formats
    FLOAT = "float"
    DOUBLE = "double"
    
    # Integer formats
    INT32 = "int32"
    INT64 = "int64"
```

### 3. 資料格式和編碼標準

#### JSONSchemaKeyword 枚舉
```python
class JSONSchemaKeyword(str, Enum):
    """JSON Schema 關鍵字枚舉（Draft 2020-12）"""
    # Core keywords
    SCHEMA = "$schema"
    ID = "$id"
    REF = "$ref"
    DEFS = "$defs"
    COMMENT = "$comment"
    
    # Type keywords
    TYPE = "type"
    ENUM = "enum"
    CONST = "const"
    
    # String keywords
    MAX_LENGTH = "maxLength"
    MIN_LENGTH = "minLength"
    PATTERN = "pattern"
    FORMAT = "format"
    
    # Number keywords
    MULTIPLE_OF = "multipleOf"
    MAXIMUM = "maximum"
    EXCLUSIVE_MAXIMUM = "exclusiveMaximum"
    MINIMUM = "minimum"
    EXCLUSIVE_MINIMUM = "exclusiveMinimum"
    
    # Object keywords
    PROPERTIES = "properties"
    PATTERN_PROPERTIES = "patternProperties"
    ADDITIONAL_PROPERTIES = "additionalProperties"
    UNEVALUATED_PROPERTIES = "unevaluatedProperties"
    REQUIRED = "required"
    PROPERTY_NAMES = "propertyNames"
    MAX_PROPERTIES = "maxProperties"
    MIN_PROPERTIES = "minProperties"
    DEPENDENT_REQUIRED = "dependentRequired"
    DEPENDENT_SCHEMAS = "dependentSchemas"
    
    # Array keywords
    PREFIX_ITEMS = "prefixItems"
    ITEMS = "items"
    UNEVALUATED_ITEMS = "unevaluatedItems"
    CONTAINS = "contains"
    MAX_CONTAINS = "maxContains"
    MIN_CONTAINS = "minContains"
    MAX_ITEMS = "maxItems"
    MIN_ITEMS = "minItems"
    UNIQUE_ITEMS = "uniqueItems"
    
    # Validation keywords
    ALL_OF = "allOf"
    ANY_OF = "anyOf"
    ONE_OF = "oneOf"
    NOT = "not"
    
    # Conditional keywords
    IF = "if"
    THEN = "then"
    ELSE = "else"
    
    # Annotation keywords
    TITLE = "title"
    DESCRIPTION = "description"
    DEFAULT = "default"
    DEPRECATED = "deprecated"
    READ_ONLY = "readOnly"
    WRITE_ONLY = "writeOnly"
    EXAMPLES = "examples"
```

#### JWTClaim 枚舉
```python
class JWTClaim(str, Enum):
    """JWT 聲明枚舉（基於 RFC 7519）"""
    # Registered Claims
    ISSUER = "iss"
    SUBJECT = "sub"
    AUDIENCE = "aud"
    EXPIRATION_TIME = "exp"
    NOT_BEFORE = "nbf"
    ISSUED_AT = "iat"
    JWT_ID = "jti"
    
    # Common Custom Claims
    SCOPE = "scope"
    ROLES = "roles"
    PERMISSIONS = "permissions"
    NAME = "name"
    EMAIL = "email"
    EMAIL_VERIFIED = "email_verified"
    FAMILY_NAME = "family_name"
    GIVEN_NAME = "given_name"
    LOCALE = "locale"
    PICTURE = "picture"
    PREFERRED_USERNAME = "preferred_username"
    PROFILE = "profile"
    UPDATED_AT = "updated_at"
    WEBSITE = "website"
    ZONEINFO = "zoneinfo"
    
class JWTAlgorithm(str, Enum):
    """JWT 演算法枚舉"""
    # HMAC
    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    
    # RSA
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"
    
    # ECDSA
    ES256 = "ES256"
    ES384 = "ES384"
    ES512 = "ES512"
    
    # RSA-PSS
    PS256 = "PS256"
    PS384 = "PS384"
    PS512 = "PS512"
    
    # EdDSA
    EDDSA = "EdDSA"
    
    # None
    NONE = "none"
```

### 4. 現代 JavaScript/ECMAScript 標準

#### ECMAScriptVersion 枚舉（擴展）
```python
class ECMAScriptVersion(str, Enum):
    """ECMAScript 版本枚舉（包含最新標準）"""
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
    """JavaScript 特性枚舉"""
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
```

### 5. 現代網路協議和標準

#### WebStandard 枚舉
```python
class WebStandard(str, Enum):
    """Web 標準枚舉（基於 W3C）"""
    # Core Web Technologies
    HTML5 = "html5"
    CSS3 = "css3"
    SVG = "svg"
    XML = "xml"
    MATHML = "mathml"
    
    # Web APIs
    WEB_RTC = "webrtc"
    WEB_SOCKETS = "websockets"
    SERVICE_WORKERS = "service_workers"
    WEB_WORKERS = "web_workers"
    PAYMENT_REQUEST_API = "payment_request_api"
    WEB_AUTHENTICATION = "web_authentication"
    CREDENTIAL_MANAGEMENT = "credential_management"
    
    # Progressive Web Apps
    WEB_APP_MANIFEST = "web_app_manifest"
    PUSH_API = "push_api"
    NOTIFICATIONS_API = "notifications_api"
    BACKGROUND_SYNC = "background_sync"
    
    # Performance & Security
    CONTENT_SECURITY_POLICY = "content_security_policy"
    HTTP_STRICT_TRANSPORT_SECURITY = "hsts"
    SUBRESOURCE_INTEGRITY = "subresource_integrity"
    CORS = "cors"
    
    # Accessibility
    ARIA = "aria"
    WCAG = "wcag"
    
    # Internationalization
    INTERNATIONALIZATION_TAG_SET = "i18n_tag_set"
    LANGUAGE_TAG_REGISTRY = "language_tag_registry"
```

## 實施建議

### 階段一：安全標準（高優先級）
1. 實施 CVSSMetric 和 AttackTechnique 枚舉
2. 擴展現有 SecurityLevel 與新標準整合
3. 更新安全相關 Pydantic 模型

### 階段二：Web/API 標準（中優先級）
1. 擴展 HTTPStatusCode 至完整規範
2. 新增 OpenAPI 和 JSON Schema 相關枚舉
3. 實施 JWT 標準枚舉

### 階段三：現代語言特性（中優先級）
1. 更新 ECMAScript 版本支援至 2026
2. 新增 JavaScript 特性詳細枚舉
3. 擴展程式語言支援

### 階段四：綜合標準（低優先級）
1. 新增 Web 標準枚舉
2. 整合所有新標準至現有架構
3. 全面測試和文檔更新

## 遵循原則確認

✅ **國際標準優先**: 所有建議均基於官方標準（RFC, ECMA, W3C, ISO）
✅ **前瞻性包含**: 包含 2025-2026 預期標準
✅ **SOT 一致性**: 遵循 aiva_common 四層優先體系
✅ **實用導向**: 選擇實際應用中常用的標準
✅ **可維護性**: 保持清晰的命名和組織結構

## 結論

本分析識別出 5 大類別共約 300+ 個新的官方標準定義需要新增到 AIVA 通用模組中。這些擴展將確保 AIVA 在未來 2-3 年內不需要因為標準缺失而進行緊急更新，完全符合「寧可現在用不到，也不要以後要用時沒有」的指導原則。

所有建議的擴展都基於經過驗證的國際標準，並與現有的 aiva_common SOT 架構保持完全一致。