"""
AI 相關枚舉 - 機器學習、自然語言處理、知識管理等

此模組定義了 AIVA AI 系統中使用的各種枚舉類型，
包括對話系統、RAG、技能圖分析等功能相關的標準枚舉。

符合標準:
- OpenAI API 標準
- LangChain 框架標準
- Hugging Face 模型標準
- ONNX 模型標準
- MLOps 最佳實踐
"""

from enum import Enum

# ============================================================================
# 機器學習模型相關
# ============================================================================


class ModelType(str, Enum):
    """機器學習模型類型"""

    # 語言模型
    LANGUAGE_MODEL = "language_model"
    CHAT_MODEL = "chat_model"
    COMPLETION_MODEL = "completion_model"
    EMBEDDING_MODEL = "embedding_model"

    # 專用模型
    CLASSIFICATION_MODEL = "classification_model"
    REGRESSION_MODEL = "regression_model"
    CLUSTERING_MODEL = "clustering_model"
    RECOMMENDATION_MODEL = "recommendation_model"

    # 多模態模型
    VISION_LANGUAGE_MODEL = "vision_language_model"
    AUDIO_MODEL = "audio_model"
    MULTIMODAL_MODEL = "multimodal_model"

    # 強化學習
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    Q_LEARNING = "q_learning"
    POLICY_GRADIENT = "policy_gradient"


class ModelProvider(str, Enum):
    """模型提供商"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    LOCAL = "local"
    OLLAMA = "ollama"
    TOGETHER = "together"
    REPLICATE = "replicate"


class ModelCapability(str, Enum):
    """模型能力類型"""

    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_EMBEDDING = "text_embedding"
    TEXT_SUMMARIZATION = "text_summarization"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    SPEECH_SYNTHESIS = "speech_synthesis"


class ModelStatus(str, Enum):
    """模型狀態"""

    INITIALIZING = "initializing"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    UNLOADED = "unloaded"
    DEPRECATED = "deprecated"


# ============================================================================
# 對話系統相關
# ============================================================================


class ConversationRole(str, Enum):
    """對話角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ConversationState(str, Enum):
    """對話狀態"""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    ARCHIVED = "archived"


class IntentCategory(str, Enum):
    """意圖分類"""

    # 基本對話
    GREETING = "greeting"
    FAREWELL = "farewell"
    SMALL_TALK = "small_talk"
    CLARIFICATION = "clarification"

    # 信息類
    INFORMATION_REQUEST = "information_request"
    STATUS_INQUIRY = "status_inquiry"
    HELP_REQUEST = "help_request"
    EXPLANATION_REQUEST = "explanation_request"

    # 操作類
    COMMAND_EXECUTION = "command_execution"
    TASK_CREATION = "task_creation"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_MANIPULATION = "data_manipulation"

    # 安全相關
    SECURITY_QUERY = "security_query"
    VULNERABILITY_REPORT = "vulnerability_report"
    THREAT_ANALYSIS = "threat_analysis"
    COMPLIANCE_CHECK = "compliance_check"

    # 分析類
    DATA_ANALYSIS = "data_analysis"
    REPORT_GENERATION = "report_generation"
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"

    # 特殊意圖
    UNCLEAR = "unclear"
    ERROR_HANDLING = "error_handling"
    ESCALATION = "escalation"
    OUT_OF_SCOPE = "out_of_scope"


class ResponseType(str, Enum):
    """回應類型"""

    TEXT = "text"
    STRUCTURED_DATA = "structured_data"
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    INTERACTIVE = "interactive"


class ConversationTurn(str, Enum):
    """對話輪次類型"""

    USER_INPUT = "user_input"
    ASSISTANT_RESPONSE = "assistant_response"
    SYSTEM_MESSAGE = "system_message"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESPONSE = "function_response"
    ERROR_MESSAGE = "error_message"


# ============================================================================
# RAG (檢索增強生成) 相關
# ============================================================================


class DocumentType(str, Enum):
    """文檔類型"""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    EXCEL = "excel"
    WORD = "word"
    POWERPOINT = "powerpoint"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class KnowledgeSource(str, Enum):
    """知識來源"""

    DOCUMENTATION = "documentation"
    CODEBASE = "codebase"
    WIKI = "wiki"
    FAQ = "faq"
    MANUAL = "manual"
    TUTORIAL = "tutorial"
    BLOG_POST = "blog_post"
    RESEARCH_PAPER = "research_paper"
    BOOK = "book"
    COURSE = "course"
    VIDEO_TRANSCRIPT = "video_transcript"
    CONVERSATION_HISTORY = "conversation_history"
    DATABASE = "database"
    API_DOCUMENTATION = "api_documentation"
    USER_GENERATED = "user_generated"


class RetrievalStrategy(str, Enum):
    """檢索策略"""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH_BASED = "graph_based"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    MULTI_MODAL = "multi_modal"


class EmbeddingModel(str, Enum):
    """嵌入模型"""

    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMERS_ALL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMERS_ALL_MPNET = "sentence-transformers/all-mpnet-base-v2"
    COHERE_EMBED_ENGLISH = "embed-english-v2.0"
    COHERE_EMBED_MULTILINGUAL = "embed-multilingual-v2.0"
    HUGGINGFACE_BGE_BASE = "BAAI/bge-base-en-v1.5"
    HUGGINGFACE_BGE_LARGE = "BAAI/bge-large-en-v1.5"


class VectorStore(str, Enum):
    """向量存儲"""

    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    VESPA = "vespa"
    ELASTICSEARCH = "elasticsearch"
    OPENSEARCH = "opensearch"
    MEMORY = "memory"


class ChunkingStrategy(str, Enum):
    """文檔分塊策略"""

    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    MARKDOWN_HEADER = "markdown_header"
    CODE_AWARE = "code_aware"
    OVERLAP = "overlap"


# ============================================================================
# 技能圖分析相關
# ============================================================================


class SkillLevel(str, Enum):
    """技能水平"""

    BEGINNER = "beginner"
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class SkillCategory(str, Enum):
    """技能類別"""

    PROGRAMMING = "programming"
    SECURITY = "security"
    DATA_SCIENCE = "data_science"
    DEVOPS = "devops"
    WEB_DEVELOPMENT = "web_development"
    MOBILE_DEVELOPMENT = "mobile_development"
    CLOUD_COMPUTING = "cloud_computing"
    MACHINE_LEARNING = "machine_learning"
    DATABASE = "database"
    NETWORKING = "networking"
    SYSTEM_ADMINISTRATION = "system_administration"
    PROJECT_MANAGEMENT = "project_management"
    SOFT_SKILLS = "soft_skills"
    DOMAIN_SPECIFIC = "domain_specific"


class SkillRelationType(str, Enum):
    """技能關係類型"""

    PREREQUISITE = "prerequisite"
    DEPENDENCY = "dependency"
    COMPLEMENT = "complement"
    ALTERNATIVE = "alternative"
    ENHANCEMENT = "enhancement"
    SIMILAR = "similar"
    RELATED = "related"
    ADVANCED_VERSION = "advanced_version"


class LearningPathType(str, Enum):
    """學習路徑類型"""

    LINEAR = "linear"
    BRANCHING = "branching"
    ADAPTIVE = "adaptive"
    SELF_PACED = "self_paced"
    GUIDED = "guided"
    PROJECT_BASED = "project_based"
    COMPETENCY_BASED = "competency_based"


class AssessmentType(str, Enum):
    """評估類型"""

    SELF_ASSESSMENT = "self_assessment"
    PEER_ASSESSMENT = "peer_assessment"
    AUTOMATED_TEST = "automated_test"
    PROJECT_REVIEW = "project_review"
    INTERVIEW = "interview"
    CERTIFICATION = "certification"
    PORTFOLIO = "portfolio"


# ============================================================================
# 計劃執行相關
# ============================================================================


class ExecutionStrategy(str, Enum):
    """執行策略"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    ADAPTIVE = "adaptive"
    ROLLBACK = "rollback"
    RETRY = "retry"


class ExecutionStatus(str, Enum):
    """執行狀態"""

    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ROLLBACK = "rollback"


class ActionType(str, Enum):
    """動作類型"""

    SCAN = "scan"
    TEST = "test"
    ANALYZE = "analyze"
    REPORT = "report"
    NOTIFY = "notify"
    BACKUP = "backup"
    RESTORE = "restore"
    VALIDATE = "validate"
    CONFIGURE = "configure"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    CLEAN = "clean"


class PriorityLevel(str, Enum):
    """優先級級別"""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


# ============================================================================
# 經驗學習相關
# ============================================================================


class LearningType(str, Enum):
    """學習類型"""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    SEMI_SUPERVISED = "semi_supervised"
    SELF_SUPERVISED = "self_supervised"
    TRANSFER_LEARNING = "transfer_learning"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"


class FeedbackType(str, Enum):
    """反饋類型"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    VALIDATION = "validation"
    REWARD = "reward"
    PENALTY = "penalty"


class ExperienceSource(str, Enum):
    """經驗來源"""

    USER_INTERACTION = "user_interaction"
    AUTOMATED_TEST = "automated_test"
    SYSTEM_MONITORING = "system_monitoring"
    EXPERT_FEEDBACK = "expert_feedback"
    PEER_LEARNING = "peer_learning"
    SIMULATION = "simulation"
    HISTORICAL_DATA = "historical_data"
    REAL_WORLD = "real_world"


class MemoryType(str, Enum):
    """記憶類型"""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# ============================================================================
# 能力評估相關
# ============================================================================


class CapabilityType(str, Enum):
    """能力類型"""

    TECHNICAL = "technical"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    COMMUNICATION = "communication"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    LEADERSHIP = "leadership"
    COLLABORATION = "collaboration"
    ADAPTATION = "adaptation"
    LEARNING = "learning"


class EvaluationMethod(str, Enum):
    """評估方法"""

    BENCHMARK = "benchmark"
    PEER_COMPARISON = "peer_comparison"
    HISTORICAL_COMPARISON = "historical_comparison"
    CRITERIA_BASED = "criteria_based"
    PERFORMANCE_METRICS = "performance_metrics"
    USER_FEEDBACK = "user_feedback"
    EXPERT_REVIEW = "expert_review"
    AUTOMATED_SCORING = "automated_scoring"


class MetricType(str, Enum):
    """指標類型"""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"


class PerformanceLevel(str, Enum):
    """性能水平"""

    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"
    CRITICAL = "critical"


# ============================================================================
# 跨語言橋接相關
# ============================================================================


class LanguageType(str, Enum):
    """語言類型"""

    PROGRAMMING = "programming"
    NATURAL = "natural"
    MARKUP = "markup"
    QUERY = "query"
    CONFIGURATION = "configuration"
    PROTOCOL = "protocol"


class TranslationType(str, Enum):
    """翻譯類型"""

    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    BEHAVIORAL = "behavioral"
    INTERFACE = "interface"


class InteroperabilityLevel(str, Enum):
    """互操作性級別"""

    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    PRAGMATIC = "pragmatic"
    DYNAMIC = "dynamic"
    CONCEPTUAL = "conceptual"


# ============================================================================
# AI 工作流程相關
# ============================================================================


class WorkflowType(str, Enum):
    """工作流程類型"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    DECISION_TREE = "decision_tree"
    STATE_MACHINE = "state_machine"
    EVENT_DRIVEN = "event_driven"
    PIPELINE = "pipeline"


class WorkflowStatus(str, Enum):
    """工作流程狀態"""

    CREATED = "created"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TriggerType(str, Enum):
    """觸發器類型"""

    MANUAL = "manual"
    SCHEDULE = "schedule"
    EVENT = "event"
    CONDITION = "condition"
    WEBHOOK = "webhook"
    API_CALL = "api_call"
    FILE_CHANGE = "file_change"
    THRESHOLD = "threshold"


# ============================================================================
# 智能代理相關
# ============================================================================


class AgentType(str, Enum):
    """代理類型"""

    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"
    LEARNING = "learning"
    COLLABORATIVE = "collaborative"
    AUTONOMOUS = "autonomous"
    HIERARCHICAL = "hierarchical"


class AgentRole(str, Enum):
    """代理角色"""

    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    MONITOR = "monitor"
    ADVISOR = "advisor"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"


class CommunicationProtocol(str, Enum):
    """通信協議"""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PEER_TO_PEER = "peer_to_peer"


# ============================================================================
# 數據處理相關
# ============================================================================


class DataQuality(str, Enum):
    """數據質量"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    INCONSISTENT = "inconsistent"


class ProcessingStage(str, Enum):
    """處理階段"""

    RAW = "raw"
    PREPROCESSED = "preprocessed"
    CLEANED = "cleaned"
    TRANSFORMED = "transformed"
    VALIDATED = "validated"
    ENRICHED = "enriched"
    ANALYZED = "analyzed"
    READY = "ready"


class DataFlow(str, Enum):
    """數據流向"""

    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"
    FEEDBACK = "feedback"
    CACHED = "cached"
    ARCHIVED = "archived"
    STREAMING = "streaming"
    BATCH = "batch"
