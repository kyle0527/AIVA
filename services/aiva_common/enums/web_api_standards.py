"""
Web API 標準枚舉 - OpenAPI, JSON Schema, JWT 等官方標準
"""



from enum import Enum, IntEnum


class HTTPStatusCode(IntEnum):
    """完整的 HTTP 狀態碼枚舉（基於 RFC 7231 等官方標準）"""
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


class OpenAPISchemaType(str, Enum):
    """OpenAPI 3.1 Schema 類型枚舉（基於 OpenAPI 官方規範）"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"  # OpenAPI 3.1 新增


class OpenAPIFormat(str, Enum):
    """OpenAPI 格式枚舉（基於 OpenAPI 官方規範）"""
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


class OpenAPIParameterLocation(str, Enum):
    """OpenAPI 參數位置枚舉"""
    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"


class OpenAPISecuritySchemeType(str, Enum):
    """OpenAPI 安全方案類型枚舉"""
    API_KEY = "apiKey"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPEN_ID_CONNECT = "openIdConnect"
    MUTUAL_TLS = "mutualTLS"


class JSONSchemaKeyword(str, Enum):
    """JSON Schema 關鍵字枚舉（Draft 2020-12 官方標準）"""
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


class JWTClaim(str, Enum):
    """JWT 聲明枚舉（基於 RFC 7519 官方標準）"""
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
    """JWT 演算法枚舉（基於 RFC 7518 官方標準）"""
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


class WebStandard(str, Enum):
    """Web 標準枚舉（基於 W3C 官方標準）"""
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


class SARIFLevel(str, Enum):
    """SARIF（Static Analysis Results Interchange Format）層級枚舉（基於 SARIF v2.1.0 官方標準）"""
    ERROR = "error"
    WARNING = "warning" 
    INFO = "info"
    NOTE = "note"


class SARIFResultKind(str, Enum):
    """SARIF 結果類別枚舉"""
    FAIL = "fail"
    PASS = "pass"
    REVIEW = "review"
    OPEN = "open"
    NOT_APPLICABLE = "notApplicable"
    INFORMATIONAL = "informational"


class SARIFArtifactRoles(str, Enum):
    """SARIF 工件角色枚舉"""
    ANALYSIS_TARGET = "analysisTarget"
    ATTACHMENT = "attachment"
    RESPONSE_FILE = "responseFile"
    RESULT_FILE = "resultFile"
    STANDARD_STREAM = "standardStream"
    TRACED_FILE = "tracedFile"