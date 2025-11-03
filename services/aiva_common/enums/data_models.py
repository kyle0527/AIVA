"""
數據模型和格式相關枚舉

此模組定義了 AIVA 系統中使用的各種數據模型、格式、
結構和標準相關的枚舉類型。

符合標準:
- JSON Schema Draft 2020-12
- OpenAPI 3.1.0
- XML Schema (XSD) 1.1
- YAML 1.2
- CSV RFC 4180
- Avro 1.11.0
- Parquet Format
- Protocol Buffers v3
"""

from enum import Enum

# ============================================================================
# 數據格式和序列化
# ============================================================================

# DataFormat 已移除重複定義，統一使用 services.aiva_common.enums.common.DataFormat
# 原 data_models.py 中的 DataFormat 於 2024-12-19 移除，避免重複定義
# 如需檔案格式名稱，請使用 common.DataFormat 或創建 FileExtension 枚舉


class SerializationFormat(str, Enum):
    """序列化格式"""

    BINARY = "binary"
    TEXT = "text"
    BASE64 = "base64"
    HEX = "hex"
    URL_ENCODED = "url_encoded"
    GZIP_COMPRESSED = "gzip_compressed"
    DEFLATE_COMPRESSED = "deflate_compressed"
    BROTLI_COMPRESSED = "brotli_compressed"
    LZ4_COMPRESSED = "lz4_compressed"
    SNAPPY_COMPRESSED = "snappy_compressed"


# EncodingType 已移除重複定義，統一使用 services.aiva_common.enums.common.EncodingType
# 原 data_models.py 中的 EncodingType 於 2024-12-19 移除，避免與 common.EncodingType 衝突
# 如需字符編碼相關功能，請使用 common.EncodingType (包含官方標準註釋)
    SHIFT_JIS = "shift_jis"
    EUC_JP = "euc-jp"
    KOI8_R = "koi8-r"


# ============================================================================
# JSON Schema 相關
# ============================================================================


class JSONSchemaType(str, Enum):
    """JSON Schema 數據類型 (Draft 2020-12)"""

    NULL = "null"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NUMBER = "number"
    STRING = "string"
    INTEGER = "integer"


class JSONSchemaFormat(str, Enum):
    """JSON Schema 格式 (Draft 2020-12)"""

    # String formats
    DATE_TIME = "date-time"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    EMAIL = "email"
    IDN_EMAIL = "idn-email"
    HOSTNAME = "hostname"
    IDN_HOSTNAME = "idn-hostname"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    URI = "uri"
    URI_REFERENCE = "uri-reference"
    IRI = "iri"
    IRI_REFERENCE = "iri-reference"
    UUID = "uuid"
    URI_TEMPLATE = "uri-template"
    JSON_POINTER = "json-pointer"
    RELATIVE_JSON_POINTER = "relative-json-pointer"
    REGEX = "regex"

    # Binary formats
    BASE64 = "base64"
    BASE64URL = "base64url"

    # Custom formats (commonly used)
    PASSWORD = "password"
    BINARY = "binary"
    BYTE = "byte"


class JSONSchemaValidation(str, Enum):
    """JSON Schema 驗證關鍵字"""

    # Type validation
    TYPE = "type"
    ENUM = "enum"
    CONST = "const"

    # Numeric validation
    MULTIPLE_OF = "multipleOf"
    MAXIMUM = "maximum"
    EXCLUSIVE_MAXIMUM = "exclusiveMaximum"
    MINIMUM = "minimum"
    EXCLUSIVE_MINIMUM = "exclusiveMinimum"

    # String validation
    MAX_LENGTH = "maxLength"
    MIN_LENGTH = "minLength"
    PATTERN = "pattern"
    FORMAT = "format"

    # Array validation
    MAX_ITEMS = "maxItems"
    MIN_ITEMS = "minItems"
    UNIQUE_ITEMS = "uniqueItems"
    MAX_CONTAINS = "maxContains"
    MIN_CONTAINS = "minContains"

    # Object validation
    MAX_PROPERTIES = "maxProperties"
    MIN_PROPERTIES = "minProperties"
    REQUIRED = "required"
    DEPENDENT_REQUIRED = "dependentRequired"

    # Conditional validation
    IF = "if"
    THEN = "then"
    ELSE = "else"
    ALL_OF = "allOf"
    ANY_OF = "anyOf"
    ONE_OF = "oneOf"
    NOT = "not"


# ============================================================================
# 數據庫模型相關
# ============================================================================


class RelationshipType(str, Enum):
    """數據庫關係類型"""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class IndexType(str, Enum):
    """數據庫索引類型"""

    PRIMARY = "primary"
    UNIQUE = "unique"
    COMPOSITE = "composite"
    PARTIAL = "partial"
    EXPRESSION = "expression"
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"
    FULLTEXT = "fulltext"
    SPATIAL = "spatial"


class ConstraintType(str, Enum):
    """數據庫約束類型"""

    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    CHECK = "check"
    DEFAULT = "default"
    EXCLUSION = "exclusion"


class DataType(str, Enum):
    """通用數據類型"""

    # Primitive types
    BOOLEAN = "boolean"
    INTEGER = "integer"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    DECIMAL = "decimal"
    STRING = "string"
    TEXT = "text"
    BINARY = "binary"

    # Date/Time types
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    INTERVAL = "interval"

    # Complex types
    ARRAY = "array"
    OBJECT = "object"
    MAP = "map"
    SET = "set"
    LIST = "list"
    TUPLE = "tuple"

    # Special types
    UUID = "uuid"
    JSON = "json"
    JSONB = "jsonb"
    XML = "xml"
    GEOMETRY = "geometry"
    GEOGRAPHY = "geography"
    POINT = "point"
    POLYGON = "polygon"

    # Network types
    INET = "inet"
    CIDR = "cidr"
    MACADDR = "macaddr"

    # System types
    OID = "oid"
    MONEY = "money"
    ENUM = "enum"
    RANGE = "range"


# ============================================================================
# API 數據模型
# ============================================================================


class APIDataType(str, Enum):
    """API 數據類型"""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    METADATA = "metadata"
    PAGINATION = "pagination"
    FILTER = "filter"
    SORT = "sort"
    SEARCH = "search"
    AGGREGATE = "aggregate"
    BATCH = "batch"


class ResponseStructure(str, Enum):
    """響應結構類型"""

    SIMPLE = "simple"
    ENVELOPE = "envelope"
    HAL = "hal"  # Hypertext Application Language
    JSON_API = "json_api"
    ODATA = "odata"
    GRAPHQL = "graphql"
    COLLECTION_JSON = "collection_json"
    SIREN = "siren"


class PaginationType(str, Enum):
    """分頁類型"""

    OFFSET_LIMIT = "offset_limit"
    PAGE_SIZE = "page_size"
    CURSOR = "cursor"
    TOKEN = "token"
    KEYSET = "keyset"
    SEEK = "seek"


class SortOrder(str, Enum):
    """排序順序"""

    ASC = "asc"
    DESC = "desc"
    ASCENDING = "ascending"
    DESCENDING = "descending"


# ============================================================================
# 文件格式和結構
# ============================================================================


class FileFormat(str, Enum):
    """文件格式"""

    # Text formats
    TXT = "txt"
    RTF = "rtf"
    MD = "md"  # Markdown

    # Document formats
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    ODT = "odt"

    # Spreadsheet formats
    XLS = "xls"
    XLSX = "xlsx"
    ODS = "ods"

    # Presentation formats
    PPT = "ppt"
    PPTX = "pptx"
    ODP = "odp"

    # Image formats
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    SVG = "svg"
    WEBP = "webp"

    # Audio formats
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"

    # Video formats
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WMV = "wmv"
    FLV = "flv"
    MKV = "mkv"
    WEBM = "webm"

    # Archive formats
    ZIP = "zip"
    RAR = "rar"
    TAR = "tar"
    GZ = "gz"
    BZ2 = "bz2"
    XZ = "xz"
    SEVEN_Z = "7z"

    # Code formats
    HTML = "html"
    CSS = "css"
    JS = "js"
    TS = "ts"
    PY = "py"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CS = "cs"
    PHP = "php"
    RB = "rb"
    GO = "go"
    RS = "rs"

    # Configuration formats
    CONF = "conf"
    CFG = "cfg"
    ENV = "env"

    # Log formats
    LOG = "log"


class CompressionType(str, Enum):
    """壓縮類型"""

    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"
    LZ4 = "lz4"
    LZMA = "lzma"
    SNAPPY = "snappy"
    ZSTD = "zstd"
    BZ2 = "bz2"


# ============================================================================
# 消息和事件格式
# ============================================================================


class MessageFormat(str, Enum):
    """消息格式"""

    JSON = "json"
    XML = "xml"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    MESSAGEPACK = "messagepack"
    THRIFT = "thrift"
    CAPNPROTO = "capnproto"
    FLATBUFFERS = "flatbuffers"


class EventType(str, Enum):
    """事件類型"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    EXECUTE = "execute"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    AUDIT = "audit"
    METRIC = "metric"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"


class MessagePriority(str, Enum):
    """消息優先級"""

    EMERGENCY = "emergency"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BULK = "bulk"


class DeliveryMode(str, Enum):
    """傳遞模式"""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    BEST_EFFORT = "best_effort"


# ============================================================================
# 數據質量和驗證
# ============================================================================


class ValidationLevel(str, Enum):
    """驗證級別"""

    STRICT = "strict"
    LENIENT = "lenient"
    PERMISSIVE = "permissive"
    IGNORE = "ignore"


class DataQualityDimension(str, Enum):
    """數據質量維度"""

    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"
    ACCESSIBILITY = "accessibility"


class ValidationRule(str, Enum):
    """驗證規則類型"""

    REQUIRED = "required"
    FORMAT = "format"
    RANGE = "range"
    LENGTH = "length"
    PATTERN = "pattern"
    CUSTOM = "custom"
    REFERENCE = "reference"
    BUSINESS_RULE = "business_rule"


class ErrorSeverity(str, Enum):
    """錯誤嚴重程度"""

    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    TRACE = "trace"


# ============================================================================
# 數據處理和轉換
# ============================================================================


class TransformationType(str, Enum):
    """轉換類型"""

    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    CLEAN = "clean"
    NORMALIZE = "normalize"
    DENORMALIZE = "denormalize"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    SORT = "sort"
    JOIN = "join"
    SPLIT = "split"
    MERGE = "merge"
    PIVOT = "pivot"
    UNPIVOT = "unpivot"
    WINDOW = "window"
    RANK = "rank"


class AggregationFunction(str, Enum):
    """聚合函數"""

    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MODE = "mode"
    STDDEV = "stddev"
    VARIANCE = "variance"
    FIRST = "first"
    LAST = "last"
    DISTINCT_COUNT = "distinct_count"
    PERCENTILE = "percentile"


class JoinType(str, Enum):
    """連接類型"""

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"
    SELF = "self"
    NATURAL = "natural"
    LEFT_ANTI = "left_anti"
    LEFT_SEMI = "left_semi"


# ============================================================================
# 數據存儲和訪問模式
# ============================================================================


class ConsistencyLevel(str, Enum):
    """一致性級別"""

    STRONG = "strong"
    EVENTUAL = "eventual"
    WEAK = "weak"
    SESSION = "session"
    BOUNDED_STALENESS = "bounded_staleness"
    CONSISTENT_PREFIX = "consistent_prefix"


class IsolationLevel(str, Enum):
    """隔離級別"""

    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"
    SNAPSHOT = "snapshot"


class CachingStrategy(str, Enum):
    """緩存策略"""

    CACHE_ASIDE = "cache_aside"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    REFRESH_AHEAD = "refresh_ahead"
    READ_THROUGH = "read_through"


class PartitioningStrategy(str, Enum):
    """分區策略"""

    HASH = "hash"
    RANGE = "range"
    LIST = "list"
    ROUND_ROBIN = "round_robin"
    CONSISTENT_HASH = "consistent_hash"
    DIRECTORY = "directory"
