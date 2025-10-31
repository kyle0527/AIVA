"""
學術研究和知識管理枚舉

遵循以下標準：
- IEEE Standards for Academic Research
- ACM Computing Classification System
- Dublin Core Metadata Initiative
- Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH)
- Digital Object Identifier (DOI) System
- Creative Commons Licensing
- FAIR Data Principles (Findable, Accessible, Interoperable, Reusable)
- Research Data Alliance (RDA) Metadata Standards
"""

from enum import Enum

# ==================== 學術研究領域 ====================


class ResearchDiscipline(Enum):
    """研究學科 - 基於 ACM Computing Classification System"""

    COMPUTER_SCIENCE = "COMPUTER_SCIENCE"
    ARTIFICIAL_INTELLIGENCE = "ARTIFICIAL_INTELLIGENCE"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    DATA_SCIENCE = "DATA_SCIENCE"
    SOFTWARE_ENGINEERING = "SOFTWARE_ENGINEERING"
    HUMAN_COMPUTER_INTERACTION = "HUMAN_COMPUTER_INTERACTION"
    COMPUTER_GRAPHICS = "COMPUTER_GRAPHICS"
    COMPUTER_VISION = "COMPUTER_VISION"
    NATURAL_LANGUAGE_PROCESSING = "NATURAL_LANGUAGE_PROCESSING"
    ROBOTICS = "ROBOTICS"
    CYBERSECURITY = "CYBERSECURITY"
    DATABASE_SYSTEMS = "DATABASE_SYSTEMS"
    DISTRIBUTED_SYSTEMS = "DISTRIBUTED_SYSTEMS"
    NETWORKS = "NETWORKS"
    ALGORITHMS = "ALGORITHMS"
    COMPUTATIONAL_THEORY = "COMPUTATIONAL_THEORY"
    BIOINFORMATICS = "BIOINFORMATICS"
    QUANTUM_COMPUTING = "QUANTUM_COMPUTING"
    BLOCKCHAIN = "BLOCKCHAIN"
    IOT = "IOT"


class ResearchType(Enum):
    """研究類型"""

    BASIC_RESEARCH = "BASIC_RESEARCH"  # 基礎研究
    APPLIED_RESEARCH = "APPLIED_RESEARCH"  # 應用研究
    DEVELOPMENT_RESEARCH = "DEVELOPMENT_RESEARCH"  # 開發研究
    EXPERIMENTAL = "EXPERIMENTAL"  # 實驗研究
    THEORETICAL = "THEORETICAL"  # 理論研究
    EMPIRICAL = "EMPIRICAL"  # 實證研究
    QUALITATIVE = "QUALITATIVE"  # 定性研究
    QUANTITATIVE = "QUANTITATIVE"  # 定量研究
    MIXED_METHODS = "MIXED_METHODS"  # 混合方法
    LONGITUDINAL = "LONGITUDINAL"  # 縱向研究
    CROSS_SECTIONAL = "CROSS_SECTIONAL"  # 橫斷面研究
    CASE_STUDY = "CASE_STUDY"  # 案例研究
    SURVEY = "SURVEY"  # 調查研究
    META_ANALYSIS = "META_ANALYSIS"  # 元分析
    SYSTEMATIC_REVIEW = "SYSTEMATIC_REVIEW"  # 系統性回顧


class ResearchMethodology(Enum):
    """研究方法論"""

    SCIENTIFIC_METHOD = "SCIENTIFIC_METHOD"
    DESIGN_SCIENCE = "DESIGN_SCIENCE"
    ACTION_RESEARCH = "ACTION_RESEARCH"
    ETHNOGRAPHY = "ETHNOGRAPHY"
    GROUNDED_THEORY = "GROUNDED_THEORY"
    PHENOMENOLOGY = "PHENOMENOLOGY"
    HERMENEUTICS = "HERMENEUTICS"
    CONTENT_ANALYSIS = "CONTENT_ANALYSIS"
    DISCOURSE_ANALYSIS = "DISCOURSE_ANALYSIS"
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"
    SIMULATION = "SIMULATION"
    MODELING = "MODELING"
    PROTOTYPING = "PROTOTYPING"


# ==================== 學術出版和同行評議 ====================


class PublicationType(Enum):
    """出版物類型 - 基於學術出版標準"""

    JOURNAL_ARTICLE = "JOURNAL_ARTICLE"
    CONFERENCE_PAPER = "CONFERENCE_PAPER"
    BOOK = "BOOK"
    BOOK_CHAPTER = "BOOK_CHAPTER"
    THESIS = "THESIS"
    DISSERTATION = "DISSERTATION"
    TECHNICAL_REPORT = "TECHNICAL_REPORT"
    WHITE_PAPER = "WHITE_PAPER"
    WORKING_PAPER = "WORKING_PAPER"
    PREPRINT = "PREPRINT"
    POSTER = "POSTER"
    PRESENTATION = "PRESENTATION"
    PATENT = "PATENT"
    DATASET = "DATASET"
    SOFTWARE = "SOFTWARE"
    WEBPAGE = "WEBPAGE"
    BLOG_POST = "BLOG_POST"


class PeerReviewType(Enum):
    """同行評議類型"""

    SINGLE_BLIND = "SINGLE_BLIND"  # 單盲評議
    DOUBLE_BLIND = "DOUBLE_BLIND"  # 雙盲評議
    TRIPLE_BLIND = "TRIPLE_BLIND"  # 三盲評議
    OPEN_REVIEW = "OPEN_REVIEW"  # 開放評議
    POST_PUBLICATION = "POST_PUBLICATION"  # 發表後評議
    COLLABORATIVE = "COLLABORATIVE"  # 協作評議


class ReviewStatus(Enum):
    """評議狀態"""

    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    REVIEWER_ASSIGNED = "REVIEWER_ASSIGNED"
    REVIEW_COMPLETED = "REVIEW_COMPLETED"
    ACCEPTED = "ACCEPTED"
    ACCEPTED_WITH_MINOR_REVISIONS = "ACCEPTED_WITH_MINOR_REVISIONS"
    ACCEPTED_WITH_MAJOR_REVISIONS = "ACCEPTED_WITH_MAJOR_REVISIONS"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"
    DESK_REJECTED = "DESK_REJECTED"
    REVISION_SUBMITTED = "REVISION_SUBMITTED"
    IN_PRESS = "IN_PRESS"
    PUBLISHED = "PUBLISHED"


class JournalRanking(Enum):
    """期刊排名"""

    Q1 = "Q1"  # 第一四分位 (前25%)
    Q2 = "Q2"  # 第二四分位
    Q3 = "Q3"  # 第三四分位
    Q4 = "Q4"  # 第四四分位
    UNRANKED = "UNRANKED"  # 未排名
    PREDATORY = "PREDATORY"  # 掠奪性期刊


class ConferenceRank(Enum):
    """會議排名 - 基於 CORE 排名系統"""

    A_STAR = "A_STAR"  # A* 頂級會議
    A = "A"  # A 級會議
    B = "B"  # B 級會議
    C = "C"  # C 級會議
    UNRANKED = "UNRANKED"  # 未排名


# ==================== 知識管理和本體論 ====================


class KnowledgeType(Enum):
    """知識類型"""

    EXPLICIT = "EXPLICIT"  # 明確知識
    TACIT = "TACIT"  # 隱性知識
    PROCEDURAL = "PROCEDURAL"  # 程序性知識
    DECLARATIVE = "DECLARATIVE"  # 陳述性知識
    CONDITIONAL = "CONDITIONAL"  # 條件性知識
    METACOGNITIVE = "METACOGNITIVE"  # 元認知知識
    DOMAIN_SPECIFIC = "DOMAIN_SPECIFIC"  # 領域特定知識
    GENERAL = "GENERAL"  # 通用知識


class OntologyType(Enum):
    """本體論類型"""

    DOMAIN_ONTOLOGY = "DOMAIN_ONTOLOGY"  # 領域本體
    UPPER_ONTOLOGY = "UPPER_ONTOLOGY"  # 上層本體
    TASK_ONTOLOGY = "TASK_ONTOLOGY"  # 任務本體
    APPLICATION_ONTOLOGY = "APPLICATION_ONTOLOGY"  # 應用本體
    FOUNDATIONAL_ONTOLOGY = "FOUNDATIONAL_ONTOLOGY"  # 基礎本體
    CORE_ONTOLOGY = "CORE_ONTOLOGY"  # 核心本體
    LIGHTWEIGHT_ONTOLOGY = "LIGHTWEIGHT_ONTOLOGY"  # 輕量級本體
    HEAVYWEIGHT_ONTOLOGY = "HEAVYWEIGHT_ONTOLOGY"  # 重量級本體


class SemanticRelation(Enum):
    """語義關係"""

    IS_A = "IS_A"  # 是一個
    PART_OF = "PART_OF"  # 是...的一部分
    HAS_PART = "HAS_PART"  # 包含部分
    INSTANCE_OF = "INSTANCE_OF"  # 是...的實例
    SUBCLASS_OF = "SUBCLASS_OF"  # 是...的子類
    SIMILAR_TO = "SIMILAR_TO"  # 類似於
    OPPOSITE_OF = "OPPOSITE_OF"  # 與...相對
    CAUSES = "CAUSES"  # 導致
    DEPENDS_ON = "DEPENDS_ON"  # 依賴於
    PRECEDES = "PRECEDES"  # 先於
    FOLLOWS = "FOLLOWS"  # 跟隨
    CONTAINS = "CONTAINS"  # 包含
    LOCATED_IN = "LOCATED_IN"  # 位於


# ==================== 數據管理和元數據 ====================


class MetadataStandard(Enum):
    """元數據標準 - 基於國際標準"""

    DUBLIN_CORE = "DUBLIN_CORE"  # Dublin Core
    MARC = "MARC"  # Machine-Readable Cataloging
    MODS = "MODS"  # Metadata Object Description Schema
    METS = "METS"  # Metadata Encoding and Transmission Standard
    PREMIS = "PREMIS"  # Preservation Metadata
    EAD = "EAD"  # Encoded Archival Description
    TEI = "TEI"  # Text Encoding Initiative
    DATACITE = "DATACITE"  # DataCite Metadata Schema
    SCHEMA_ORG = "SCHEMA_ORG"  # Schema.org
    RDF = "RDF"  # Resource Description Framework
    OWL = "OWL"  # Web Ontology Language
    SKOS = "SKOS"  # Simple Knowledge Organization System


class DataType(Enum):
    """數據類型 - 研究數據分類"""

    OBSERVATIONAL = "OBSERVATIONAL"  # 觀察數據
    EXPERIMENTAL = "EXPERIMENTAL"  # 實驗數據
    SIMULATION = "SIMULATION"  # 模擬數據
    COMPUTATIONAL = "COMPUTATIONAL"  # 計算數據
    SURVEY = "SURVEY"  # 調查數據
    INTERVIEW = "INTERVIEW"  # 訪談數據
    TEXTUAL = "TEXTUAL"  # 文本數據
    NUMERICAL = "NUMERICAL"  # 數值數據
    MULTIMEDIA = "MULTIMEDIA"  # 多媒體數據
    GEOSPATIAL = "GEOSPATIAL"  # 地理空間數據
    TEMPORAL = "TEMPORAL"  # 時間序列數據
    NETWORK = "NETWORK"  # 網絡數據
    SENSOR = "SENSOR"  # 傳感器數據
    ADMINISTRATIVE = "ADMINISTRATIVE"  # 管理數據
    REFERENCE = "REFERENCE"  # 參考數據


class DataFormat(Enum):
    """研究數據格式"""

    CSV = "CSV"
    JSON = "JSON"
    XML = "XML"
    RDF_XML = "RDF_XML"
    TURTLE = "TURTLE"
    N3 = "N3"
    HDF5 = "HDF5"
    NETCDF = "NETCDF"
    FITS = "FITS"  # Flexible Image Transport System
    DICOM = "DICOM"  # Digital Imaging and Communications in Medicine
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    PDF = "PDF"
    DOC = "DOC"
    DOCX = "DOCX"
    TEX = "TEX"
    BIBTEX = "BIBTEX"
    ENDNOTE = "ENDNOTE"
    RIS = "RIS"


class AccessLevel(Enum):
    """訪問級別 - 基於 FAIR 原則"""

    OPEN = "OPEN"  # 開放訪問
    RESTRICTED = "RESTRICTED"  # 限制訪問
    EMBARGOED = "EMBARGOED"  # 禁發期
    CONFIDENTIAL = "CONFIDENTIAL"  # 機密
    CLASSIFIED = "CLASSIFIED"  # 分類信息
    PROPRIETARY = "PROPRIETARY"  # 專有
    LICENSED = "LICENSED"  # 許可


# ==================== 研究倫理和合規 ====================


class EthicsApprovalStatus(Enum):
    """倫理審查狀態"""

    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    APPROVED_WITH_CONDITIONS = "APPROVED_WITH_CONDITIONS"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"
    EXPIRED = "EXPIRED"
    SUSPENDED = "SUSPENDED"


class ResearchEthicsCategory(Enum):
    """研究倫理類別"""

    HUMAN_SUBJECTS = "HUMAN_SUBJECTS"  # 人體受試者
    ANIMAL_SUBJECTS = "ANIMAL_SUBJECTS"  # 動物受試者
    BIOSAFETY = "BIOSAFETY"  # 生物安全
    DATA_PRIVACY = "DATA_PRIVACY"  # 數據隱私
    INFORMED_CONSENT = "INFORMED_CONSENT"  # 知情同意
    RISK_BENEFIT = "RISK_BENEFIT"  # 風險效益
    CONFIDENTIALITY = "CONFIDENTIALITY"  # 保密性
    VULNERABLE_POPULATIONS = "VULNERABLE_POPULATIONS"  # 脆弱人群


class ConsentType(Enum):
    """同意類型"""

    INFORMED_CONSENT = "INFORMED_CONSENT"
    BROAD_CONSENT = "BROAD_CONSENT"
    DYNAMIC_CONSENT = "DYNAMIC_CONSENT"
    OPT_IN = "OPT_IN"
    OPT_OUT = "OPT_OUT"
    IMPLICIT_CONSENT = "IMPLICIT_CONSENT"
    EXPLICIT_CONSENT = "EXPLICIT_CONSENT"


# ==================== 學術評估和指標 ====================


class CitationMetric(Enum):
    """引用指標"""

    CITATION_COUNT = "CITATION_COUNT"
    H_INDEX = "H_INDEX"
    I10_INDEX = "I10_INDEX"
    G_INDEX = "G_INDEX"
    IMPACT_FACTOR = "IMPACT_FACTOR"
    EIGEN_FACTOR = "EIGEN_FACTOR"
    SCIMAGO_SJR = "SCIMAGO_SJR"  # SCImago Journal Rank
    SNIP = "SNIP"  # Source Normalized Impact per Paper
    ALTMETRICS = "ALTMETRICS"  # Alternative metrics


class ImpactMeasure(Enum):
    """影響力測量"""

    ACADEMIC_CITATIONS = "ACADEMIC_CITATIONS"
    SOCIAL_MEDIA_MENTIONS = "SOCIAL_MEDIA_MENTIONS"
    NEWS_MENTIONS = "NEWS_MENTIONS"
    POLICY_CITATIONS = "POLICY_CITATIONS"
    PATENT_CITATIONS = "PATENT_CITATIONS"
    DOWNLOAD_COUNT = "DOWNLOAD_COUNT"
    VIEW_COUNT = "VIEW_COUNT"
    BLOG_MENTIONS = "BLOG_MENTIONS"
    WIKIPEDIA_MENTIONS = "WIKIPEDIA_MENTIONS"


class ResearcherRole(Enum):
    """研究人員角色"""

    PRINCIPAL_INVESTIGATOR = "PRINCIPAL_INVESTIGATOR"  # 主要研究員
    CO_INVESTIGATOR = "CO_INVESTIGATOR"  # 共同研究員
    RESEARCH_ASSISTANT = "RESEARCH_ASSISTANT"  # 研究助理
    POSTDOC = "POSTDOC"  # 博士後研究員
    PHD_STUDENT = "PHD_STUDENT"  # 博士生
    MASTER_STUDENT = "MASTER_STUDENT"  # 碩士生
    VISITING_RESEARCHER = "VISITING_RESEARCHER"  # 訪問學者
    COLLABORATOR = "COLLABORATOR"  # 合作者
    CONSULTANT = "CONSULTANT"  # 顧問
    SUPERVISOR = "SUPERVISOR"  # 指導教授


# ==================== 學術會議和活動 ====================


class ConferenceType(Enum):
    """會議類型"""

    ACADEMIC_CONFERENCE = "ACADEMIC_CONFERENCE"
    WORKSHOP = "WORKSHOP"
    SYMPOSIUM = "SYMPOSIUM"
    SEMINAR = "SEMINAR"
    COLLOQUIUM = "COLLOQUIUM"
    SUMMIT = "SUMMIT"
    FORUM = "FORUM"
    PANEL_DISCUSSION = "PANEL_DISCUSSION"
    ROUNDTABLE = "ROUNDTABLE"
    KEYNOTE_SPEECH = "KEYNOTE_SPEECH"
    POSTER_SESSION = "POSTER_SESSION"
    DEMO_SESSION = "DEMO_SESSION"
    TUTORIAL = "TUTORIAL"
    MASTERCLASS = "MASTERCLASS"


class ParticipationType(Enum):
    """參與類型"""

    PRESENTER = "PRESENTER"
    KEYNOTE_SPEAKER = "KEYNOTE_SPEAKER"
    INVITED_SPEAKER = "INVITED_SPEAKER"
    PANELIST = "PANELIST"
    MODERATOR = "MODERATOR"
    CHAIR = "CHAIR"
    REVIEWER = "REVIEWER"
    ORGANIZER = "ORGANIZER"
    ATTENDEE = "ATTENDEE"
    STUDENT_VOLUNTEER = "STUDENT_VOLUNTEER"
    SPONSOR = "SPONSOR"


class PresentationFormat(Enum):
    """展示格式"""

    ORAL_PRESENTATION = "ORAL_PRESENTATION"
    POSTER_PRESENTATION = "POSTER_PRESENTATION"
    DEMO = "DEMO"
    LIGHTNING_TALK = "LIGHTNING_TALK"
    PANEL_DISCUSSION = "PANEL_DISCUSSION"
    ROUNDTABLE = "ROUNDTABLE"
    WORKSHOP = "WORKSHOP"
    TUTORIAL = "TUTORIAL"
    KEYNOTE = "KEYNOTE"
    INVITED_TALK = "INVITED_TALK"
    VIRTUAL_PRESENTATION = "VIRTUAL_PRESENTATION"
    HYBRID_PRESENTATION = "HYBRID_PRESENTATION"


# ==================== 知識產權和授權 ====================


class LicenseType(Enum):
    """授權類型 - 基於 Creative Commons"""

    CC0 = "CC0"  # 公共領域
    CC_BY = "CC_BY"  # 署名
    CC_BY_SA = "CC_BY_SA"  # 署名-相同方式共享
    CC_BY_NC = "CC_BY_NC"  # 署名-非商業性使用
    CC_BY_NC_SA = "CC_BY_NC_SA"  # 署名-非商業性使用-相同方式共享
    CC_BY_ND = "CC_BY_ND"  # 署名-禁止演繹
    CC_BY_NC_ND = "CC_BY_NC_ND"  # 署名-非商業性使用-禁止演繹
    MIT = "MIT"  # MIT 授權
    GPL = "GPL"  # GNU General Public License
    APACHE = "APACHE"  # Apache License
    BSD = "BSD"  # BSD License
    PROPRIETARY = "PROPRIETARY"  # 專有授權
    COPYRIGHT = "COPYRIGHT"  # 版權保護


class IntellectualPropertyType(Enum):
    """知識產權類型"""

    COPYRIGHT = "COPYRIGHT"  # 著作權
    PATENT = "PATENT"  # 專利
    TRADEMARK = "TRADEMARK"  # 商標
    TRADE_SECRET = "TRADE_SECRET"  # 商業秘密
    DESIGN_PATENT = "DESIGN_PATENT"  # 設計專利
    UTILITY_MODEL = "UTILITY_MODEL"  # 實用新型
    GEOGRAPHICAL_INDICATION = "GEOGRAPHICAL_INDICATION"  # 地理標示


# ==================== 研究合作和網絡 ====================


class CollaborationType(Enum):
    """合作類型"""

    INSTITUTIONAL = "INSTITUTIONAL"  # 機構合作
    INTERNATIONAL = "INTERNATIONAL"  # 國際合作
    INDUSTRY_ACADEMIA = "INDUSTRY_ACADEMIA"  # 產學合作
    INTERDISCIPLINARY = "INTERDISCIPLINARY"  # 跨學科合作
    MULTI_SITE = "MULTI_SITE"  # 多站點合作
    JOINT_VENTURE = "JOINT_VENTURE"  # 合資企業
    CONSORTIUM = "CONSORTIUM"  # 聯盟
    NETWORK = "NETWORK"  # 網絡
    PARTNERSHIP = "PARTNERSHIP"  # 夥伴關係


class FundingType(Enum):
    """資助類型"""

    GOVERNMENT_GRANT = "GOVERNMENT_GRANT"  # 政府資助
    PRIVATE_FOUNDATION = "PRIVATE_FOUNDATION"  # 私人基金會
    INDUSTRY_FUNDING = "INDUSTRY_FUNDING"  # 產業資助
    INTERNAL_FUNDING = "INTERNAL_FUNDING"  # 內部資助
    CROWDFUNDING = "CROWDFUNDING"  # 眾籌
    FELLOWSHIP = "FELLOWSHIP"  # 獎學金
    SCHOLARSHIP = "SCHOLARSHIP"  # 學術獎學金
    SEED_FUNDING = "SEED_FUNDING"  # 種子資金
    VENTURE_CAPITAL = "VENTURE_CAPITAL"  # 風險投資


class ResearchPhase(Enum):
    """研究階段"""

    PROPOSAL = "PROPOSAL"  # 提案階段
    PLANNING = "PLANNING"  # 規劃階段
    DATA_COLLECTION = "DATA_COLLECTION"  # 數據收集
    ANALYSIS = "ANALYSIS"  # 分析階段
    WRITING = "WRITING"  # 寫作階段
    REVIEW = "REVIEW"  # 審查階段
    REVISION = "REVISION"  # 修訂階段
    SUBMISSION = "SUBMISSION"  # 提交階段
    PUBLICATION = "PUBLICATION"  # 發表階段
    DISSEMINATION = "DISSEMINATION"  # 傳播階段
    COMPLETED = "COMPLETED"  # 完成
    SUSPENDED = "SUSPENDED"  # 暫停
    CANCELLED = "CANCELLED"  # 取消
