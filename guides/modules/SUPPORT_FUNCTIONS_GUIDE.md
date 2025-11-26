# AIVA Features - 支援功能架構 🔧

## 📋 目錄

- [📑 目錄](#目錄)
- [🎯 **支援功能在 AIVA 中的角色**](#支援功能在-aiva-中的角色)
  - [**🚀 基礎設施定位**](#基礎設施定位)
  - [**⚡ 支援組件統計分析**](#支援組件統計分析)
- [🚨 **支援層架構問題分析**](#支援層架構問題分析)
  - [**⚠️ 發現的重複與不一致問題**](#發現的重複與不一致問題)
- [🏗️ **支援功能架構模式**](#支援功能架構模式)
  - [**⚙️ 統一配置管理系統**](#統一配置管理系統)
  - [**📋 統一 Schema 系統**](#統一-schema-系統)
  - [**👷 模組化 Worker 系統**](#模組化-worker-系統)
- [🔧 **支援層重構建議**](#支援層重構建議)
  - [**✅ 推薦的支援層架構**](#推薦的支援層架構)
- [📚 **相關文件**](#相關文件)
  - [**🔗 多層架構導航**](#多層架構導航)
  - [**📊 架構分析報告**](#架構分析報告)
- [🔗 相關資源](#相關資源)
  - [模組開發指南](#模組開發指南)
  - [架構指南](#架構指南)
  - [開發指南](#開發指南)
  - [服務文檔](#服務文檔)

> **定位**: 基礎設施層、系統支援、工具服務  
> **規模**: 346 個支援組件 (12.9%)  
> **職責**: 配置管理、Schema 定義、Worker 系統、工具鏈、測試框架

## 📑 目錄

- [🎯 支援功能在 AIVA 中的角色](#-支援功能在-aiva-中的角色)
- [⚙️ 配置管理系統](#-配置管理系統)
- [📋 Schema 定義架構](#-schema-定義架構)
- [🔄 Worker 執行系統](#-worker-執行系統)
- [🛠️ 工具鏈與實用程式](#-工具鏈與實用程式)
- [🧪 測試框架](#-測試框架)
- [📊 性能監控](#-性能監控)
- [🔧 開發工具](#-開發工具)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

**🔙 導航**: [← 返回主文件](../README.md) | **相關文件**: [🔴 核心功能](README_CORE.md) | [🛡️ 安全功能](README_SECURITY.md) | [🏢 業務功能](README_BUSINESS.md)

---

## 🎯 **支援功能在 AIVA 中的角色**

### **🚀 基礎設施定位**
支援功能層是 AIVA Features 的「**基礎設施骨架**」，為所有上層功能提供穩固的技術支撐：

```
🔧 支援功能基礎設施架構
├── ⚙️ 配置管理系統 (62組件)
│   ├── 🐍 Configuration_Models (22組件) - 配置結構定義
│   ├── 🐍 example_config (2組件) - 配置範例
│   ├── 🐍 advanced_detection_config (9組件) - 高級檢測配置
│   └── 🐹 config (1組件) - Go 配置服務
├── 📋 Schema 與模型系統 (74組件)
│   ├── 🐍 schemas (30組件) - 資料結構定義
│   ├── 🐍 models (16組件) - 資料模型
│   ├── 🐍 test_schemas (8組件) - 測試結構
│   └── 🐍 result_schema (5組件) - 結果結構
├── 👷 Worker 執行系統 (67組件)
│   ├── 🐍 worker (31組件) - 核心執行器
│   ├── 🐍 worker_statistics (20組件) - 執行統計
│   ├── 🐍 enhanced_worker (5組件) - 增強執行器
│   ├── 🐍 feature_step_executor (8組件) - 步驟執行器
│   └── 🐍 cross_user_tester (3組件) - 跨用戶測試
├── 🔐 認證與安全支援 (29組件)
│   └── 🐍 Authentication_Security (29組件) - 認證基礎設施
├── 🛠️ 功能管理系統 (26組件)
│   ├── 🐍 Feature_Management (12組件) - 功能管理
│   ├── 🐍 Smart_Detection (11組件) - 智能檢測支援
│   ├── 🐍 feature_registry (2組件) - 功能註冊
│   └── 🐍 feature_base (1組件) - 功能基礎
├── 🌐 網路與客戶端 (24組件)
│   ├── 🐹 client (9組件) - Go 客戶端
│   ├── 🐍 http_client (6組件) - HTTP 客戶端
│   ├── 🐹 client_test (3組件) - 客戶端測試
│   └── 🐹 logger (3組件) - 日誌服務
└── 🎯 專業檢測工具 (64組件)
    ├── 🐍 resource_id_extractor (6組件) - 資源 ID 提取
    ├── 🐍 smart_idor_detector (2組件) - IDOR 檢測器
    ├── 🐍 lateral_movement (6組件) - 橫向移動檢測
    ├── 🐍 persistence_checker (7組件) - 持久化檢查
    └── 其他專業工具...
```

### **⚡ 支援組件統計分析**
- **配置管理**: 62 個組件 (17.9% - 系統配置基礎)
- **Schema 模型**: 74 個組件 (21.4% - 資料結構核心)
- **Worker 系統**: 67 個組件 (19.4% - 執行引擎)
- **認證支援**: 29 個組件 (8.4% - 安全基礎設施)
- **功能管理**: 26 個組件 (7.5% - 功能協調)
- **網路服務**: 24 個組件 (6.9% - 通信基礎)
- **專業工具**: 64 個組件 (18.5% - 檢測工具鏈)

---

## 🚨 **支援層架構問題分析**

### **⚠️ 發現的重複與不一致問題**

#### **問題 1: 功能檢測模組支援重複**
```
❌ 當前問題:
- SQL_Injection_Detection: 在 Support Layer 有 1 組件
- SSRF_Detection: 在 Support Layer 有 2 組件  
- XSS_Detection: 在 Support Layer 有 2 組件
- 同時這些功能在 Security/Business Layer 都有大量組件

🔍 根本原因:
- 缺乏清晰的支援/實現邊界定義
- Schema 與業務邏輯混在支援層
- 配置管理與檢測邏輯耦合

✅ 改進方案:
Support Layer 應該只包含:
- 配置 Schema 定義
- 結果資料結構  
- 執行器框架
- 不應包含檢測邏輯實現
```

#### **問題 2: 認證服務架構分散**
```
❌ 當前狀況:
- Business Layer: Authentication_Security (15組件 Go)
- Support Layer: Authentication_Security (29組件 Python)
- 功能重疊但實現分離

🔍 問題分析:
- Go 組件負責高效能認證服務
- Python 組件負責認證配置管理
- 缺乏統一的認證抽象層

✅ 建議重構:
- Support Layer: 認證 Schema + 配置管理
- Business Layer: 認證服务實現
- 建立統一認證介面標準
```

#### **問題 3: Worker 系統複雜度過高**
```
❌ 複雜度問題:
- worker (31組件) - 核心過於複雜
- enhanced_worker (5組件) - 功能重疊
- feature_step_executor (8組件) - 職責不清
- cross_user_tester (3組件) - 專業功能混入通用層

🔍 架構問題:
- 單一 Worker 承擔過多職責
- 專業檢測邏輯混入基礎設施
- 缺乏模組化的執行器設計

✅ 重構建議:
1. 拆分 Worker 為多個專責組件
2. 將專業檢測邏輯上移到業務層
3. 建立可插拔的執行器架構
```

---

## 🏗️ **支援功能架構模式**

### **⚙️ 統一配置管理系統**

```python
"""
AIVA 統一配置管理系統
提供跨語言、跨模組的配置管理能力
"""

from typing import Dict, Any, Optional, Type, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml
import json
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

T = TypeVar('T')

class ConfigScope(Enum):
    """配置作用域"""
    GLOBAL = "global"           # 全域配置
    SERVICE = "service"         # 服務級配置  
    FEATURE = "feature"         # 功能級配置
    RUNTIME = "runtime"         # 運行時配置

class ConfigFormat(Enum):
    """配置格式"""
    YAML = "yaml"
    JSON = "json" 
    ENV = "env"
    TOML = "toml"

@dataclass
class ConfigMetadata:
    """配置元資料"""
    name: str
    version: str
    scope: ConfigScope
    format: ConfigFormat
    description: str
    schema_version: str = "1.0"
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

class ConfigValidator(ABC):
    """配置驗證器抽象基類"""
    
    @abstractmethod
    def validate(self, config_data: Dict[str, Any]) -> bool:
        """驗證配置資料"""
        pass
    
    @abstractmethod 
    def get_errors(self) -> list[str]:
        """獲取驗證錯誤"""
        pass

class SchemaValidator(ConfigValidator):
    """基於 Schema 的配置驗證器"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.errors = []
    
    def validate(self, config_data: Dict[str, Any]) -> bool:
        """驗證配置是否符合 Schema"""
        self.errors.clear()
        return self._validate_recursive(config_data, self.schema, "")
    
    def _validate_recursive(self, data: Any, schema: Any, path: str) -> bool:
        """遞迴驗證"""
        if isinstance(schema, dict):
            if "type" in schema:
                return self._validate_type(data, schema, path)
            else:
                # 物件驗證
                if not isinstance(data, dict):
                    self.errors.append(f"{path}: Expected object, got {type(data).__name__}")
                    return False
                
                valid = True
                for key, sub_schema in schema.items():
                    sub_path = f"{path}.{key}" if path else key
                    if key in data:
                        valid &= self._validate_recursive(data[key], sub_schema, sub_path)
                    elif self._is_required(sub_schema):
                        self.errors.append(f"{sub_path}: Required field missing")
                        valid = False
                
                return valid
        else:
            return data == schema
    
    def _validate_type(self, data: Any, schema: Dict[str, Any], path: str) -> bool:
        """驗證資料類型"""
        expected_type = schema["type"]
        
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        python_type = type_mapping.get(expected_type)
        if python_type and not isinstance(data, python_type):
            self.errors.append(f"{path}: Expected {expected_type}, got {type(data).__name__}")
            return False
            
        # 額外驗證
        if expected_type == "array" and "items" in schema:
            for i, item in enumerate(data):
                if not self._validate_recursive(item, schema["items"], f"{path}[{i}]"):
                    return False
        
        return True
    
    def _is_required(self, schema: Any) -> bool:
        """檢查欄位是否必需"""
        return isinstance(schema, dict) and schema.get("required", False)
    
    def get_errors(self) -> list[str]:
        """獲取驗證錯誤"""
        return self.errors.copy()

@dataclass 
class SASTConfig:
    """SAST 配置"""
    enabled: bool = True
    engines: list[str] = field(default_factory=lambda: ["rust-analyzer", "semgrep"])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    timeout_seconds: int = 300
    exclude_patterns: list[str] = field(default_factory=lambda: ["**/node_modules/**", "**/.git/**"])
    language_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_language_config(self, language: str) -> Dict[str, Any]:
        """獲取特定語言配置"""
        return self.language_configs.get(language, {})

@dataclass
class WorkerConfig:
    """Worker 配置"""
    max_workers: int = 10
    queue_size: int = 1000
    timeout_seconds: int = 600
    retry_attempts: int = 3
    retry_delay: float = 1.0
    heartbeat_interval: int = 30
    metrics_enabled: bool = True
    
@dataclass
class DatabaseConfig:
    """資料庫配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "aiva"
    username: str = "aiva_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    ssl_mode: str = "prefer"
    connection_timeout: int = 30

@dataclass
class SecurityConfig:
    """安全配置"""
    jwt_secret_key: str = ""
    jwt_expiration: int = 3600
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    allowed_origins: list[str] = field(default_factory=list)
    api_key_header: str = "X-API-Key"
    encryption_algorithm: str = "AES-256-GCM"

@dataclass
class AIVAConfig:
    """AIVA 主配置"""
    # 基礎配置
    service_name: str = "aiva-features"
    version: str = "2.0.0"
    environment: str = "development"
    debug: bool = False
    
    # 子系統配置
    sast: SASTConfig = field(default_factory=SASTConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # 功能開關
    features: Dict[str, bool] = field(default_factory=lambda: {
        "sca_analysis": True,
        "cspm_scanning": True,
        "vulnerability_detection": True,
        "metrics_collection": True
    })
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """檢查功能是否啟用"""
        return self.features.get(feature_name, False)

class ConfigurationManager(Generic[T]):
    """通用配置管理器"""
    
    def __init__(self, config_class: Type[T]):
        self.config_class = config_class
        self.config: Optional[T] = None
        self.validators: list[ConfigValidator] = []
        self.metadata: Optional[ConfigMetadata] = None
        
    def add_validator(self, validator: ConfigValidator) -> None:
        """添加配置驗證器"""
        self.validators.append(validator)
    
    async def load_from_file(self, file_path: str) -> T:
        """從檔案載入配置"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # 根據副檔名判斷格式
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        return await self.load_from_dict(data)
    
    async def load_from_dict(self, data: Dict[str, Any]) -> T:
        """從字典載入配置"""
        # 驗證配置
        await self._validate_config(data)
        
        # 轉換為配置物件
        try:
            if hasattr(self.config_class, '__dataclass_fields__'):
                # dataclass
                self.config = self.config_class(**data)
            else:
                # 普通類
                self.config = self.config_class()
                for key, value in data.items():
                    setattr(self.config, key, value)
                    
            return self.config
        except Exception as e:
            raise ValueError(f"Failed to create configuration object: {str(e)}")
    
    async def _validate_config(self, data: Dict[str, Any]) -> None:
        """驗證配置資料"""
        for validator in self.validators:
            if not validator.validate(data):
                errors = validator.get_errors()
                raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_config(self) -> T:
        """獲取當前配置"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        return self.config
    
    async def save_to_file(self, file_path: str) -> None:
        """保存配置到檔案"""
        if self.config is None:
            raise RuntimeError("No configuration to save")
        
        path = Path(file_path)
        
        # 轉換為字典
        if hasattr(self.config, '__dataclass_fields__'):
            data = asdict(self.config)
        else:
            data = vars(self.config)
        
        # 根據副檔名保存
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")

# 配置管理器工廠
class ConfigManagerFactory:
    """配置管理器工廠"""
    
    _managers: Dict[str, ConfigurationManager] = {}
    
    @classmethod
    def get_manager(cls, config_name: str, config_class: Type[T]) -> ConfigurationManager[T]:
        """獲取配置管理器"""
        if config_name not in cls._managers:
            cls._managers[config_name] = ConfigurationManager(config_class)
        return cls._managers[config_name]
    
    @classmethod
    def create_aiva_manager(cls) -> ConfigurationManager[AIVAConfig]:
        """創建 AIVA 配置管理器"""
        manager = cls.get_manager("aiva", AIVAConfig)
        
        # 添加 AIVA 配置的 Schema 驗證器
        schema = {
            "service_name": {"type": "string", "required": True},
            "version": {"type": "string", "required": True},
            "environment": {"type": "string", "required": True},
            "sast": {
                "enabled": {"type": "boolean"},
                "engines": {"type": "array", "items": {"type": "string"}},
                "max_file_size": {"type": "integer"},
            }
        }
        
        manager.add_validator(SchemaValidator(schema))
        return manager
```

### **📋 統一 Schema 系統**

```python
"""
AIVA 統一 Schema 系統  
提供跨模組的資料結構定義和驗證
"""

from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

class Severity(Enum):
    """嚴重程度枚舉"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class Status(Enum):
    """狀態枚舉"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BaseSchema:
    """基礎 Schema"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_timestamp(self):
        """更新時間戳"""
        self.updated_at = datetime.utcnow()

@dataclass
class Location(BaseSchema):
    """程式碼位置"""
    file_path: str
    line_number: int
    column_number: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    code_snippet: Optional[str] = None

@dataclass
class Finding(BaseSchema):
    """安全發現基礎結構"""
    title: str
    description: str
    severity: Severity
    category: str
    location: Location
    confidence: float = 1.0  # 0.0 - 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_tag(self, tag: str):
        """添加標籤"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_metadata(self, key: str, value: Any):
        """設定元資料"""
        self.metadata[key] = value
        self.update_timestamp()

@dataclass
class SASTFinding(Finding):
    """SAST 檢測發現"""
    rule_id: str
    rule_name: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    fix_suggestion: Optional[str] = None

@dataclass 
class SCAFinding(Finding):
    """SCA 檢測發現"""
    package_name: str
    package_version: str
    vulnerability_id: str  # CVE ID
    cvss_score: Optional[float] = None
    affected_versions: List[str] = field(default_factory=list)
    fixed_versions: List[str] = field(default_factory=list)

@dataclass
class CSPMFinding(Finding):
    """CSPM 檢測發現"""
    resource_id: str
    resource_type: str
    cloud_provider: str
    region: str
    compliance_framework: str
    rule_id: str
    remediation_steps: List[str] = field(default_factory=list)

@dataclass
class ScanRequest(BaseSchema):
    """掃描請求基礎結構"""
    scan_type: str
    target: str  # 掃描目標
    options: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 優先級 0-10
    
@dataclass
class ScanResult(BaseSchema):
    """掃描結果基礎結構"""
    request_id: str
    status: Status
    scan_type: str
    target: str
    findings: List[Finding] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    
    def add_finding(self, finding: Finding):
        """添加發現"""
        self.findings.append(finding)
        self.update_statistics()
    
    def update_statistics(self):
        """更新統計資訊"""
        severity_counts = {}
        for finding in self.findings:
            severity = finding.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        self.statistics.update({
            "total_findings": len(self.findings),
            "severity_breakdown": severity_counts,
            "last_updated": datetime.utcnow().isoformat()
        })
        self.update_timestamp()

@dataclass 
class WorkerTask(BaseSchema):
    """Worker 任務結構"""
    task_type: str
    payload: Dict[str, Any]
    status: Status = Status.PENDING
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_details: Optional[str] = None
    
    def start_task(self, worker_id: str):
        """開始任務"""
        self.status = Status.RUNNING
        self.assigned_worker = worker_id
        self.started_at = datetime.utcnow()
        self.update_timestamp()
    
    def complete_task(self):
        """完成任務"""
        self.status = Status.COMPLETED
        self.completed_at = datetime.utcnow()
        self.update_timestamp()
    
    def fail_task(self, error: str):
        """任務失敗"""
        self.current_retry += 1
        self.error_details = error
        
        if self.current_retry >= self.max_retries:
            self.status = Status.FAILED
        else:
            self.status = Status.PENDING  # 重試
        
        self.update_timestamp()

# Schema 註冊表
class SchemaRegistry:
    """Schema 註冊表"""
    
    _schemas: Dict[str, Type[BaseSchema]] = {}
    
    @classmethod
    def register(cls, name: str, schema_class: Type[BaseSchema]):
        """註冊 Schema"""
        cls._schemas[name] = schema_class
    
    @classmethod
    def get_schema(cls, name: str) -> Type[BaseSchema]:
        """獲取 Schema"""
        if name not in cls._schemas:
            raise ValueError(f"Schema not found: {name}")
        return cls._schemas[name]
    
    @classmethod
    def list_schemas(cls) -> List[str]:
        """列出所有 Schema"""
        return list(cls._schemas.keys())

# 註冊內建 Schema
SchemaRegistry.register("finding", Finding)
SchemaRegistry.register("sast_finding", SASTFinding)
SchemaRegistry.register("sca_finding", SCAFinding)
SchemaRegistry.register("cspm_finding", CSPMFinding)
SchemaRegistry.register("scan_request", ScanRequest)
SchemaRegistry.register("scan_result", ScanResult)
SchemaRegistry.register("worker_task", WorkerTask)
```

### **👷 模組化 Worker 系統**

```python
"""
AIVA 模組化 Worker 系統
提供可插拔的任務執行框架
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

class WorkerType(Enum):
    """Worker 類型"""
    ASYNC = "async"      # 異步 Worker
    SYNC = "sync"        # 同步 Worker  
    THREAD = "thread"    # 多線程 Worker
    PROCESS = "process"  # 多進程 Worker

@dataclass
class WorkerMetrics:
    """Worker 指標"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_duration: float = 0.0
    last_activity: Optional[datetime] = None
    
    def update_completion(self, duration: float):
        """更新完成指標"""
        self.completed_tasks += 1
        self.total_tasks += 1
        
        # 計算平均持續時間
        if self.completed_tasks == 1:
            self.average_duration = duration
        else:
            self.average_duration = (
                (self.average_duration * (self.completed_tasks - 1) + duration) / 
                self.completed_tasks
            )
        
        self.last_activity = datetime.utcnow()
    
    def update_failure(self):
        """更新失敗指標"""
        self.failed_tasks += 1
        self.total_tasks += 1
        self.last_activity = datetime.utcnow()

class BaseWorker(ABC):
    """Worker 基礎類"""
    
    def __init__(self, worker_id: str, worker_type: WorkerType):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.metrics = WorkerMetrics()
        self.is_running = False
        self.logger = logging.getLogger(f"worker.{worker_id}")
    
    @abstractmethod
    async def execute_task(self, task: WorkerTask) -> Any:
        """執行任務"""
        pass
    
    @abstractmethod
    async def start(self):
        """啟動 Worker"""
        pass
    
    @abstractmethod
    async def stop(self):
        """停止 Worker"""
        pass
    
    def get_metrics(self) -> WorkerMetrics:
        """獲取 Worker 指標"""
        return self.metrics

class AsyncWorker(BaseWorker):
    """異步 Worker"""
    
    def __init__(self, worker_id: str, task_handler: Callable[[WorkerTask], Any]):
        super().__init__(worker_id, WorkerType.ASYNC)
        self.task_handler = task_handler
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
    
    async def execute_task(self, task: WorkerTask) -> Any:
        """執行異步任務"""
        start_time = time.time()
        
        try:
            task.start_task(self.worker_id)
            result = await self.task_handler(task)
            
            duration = time.time() - start_time
            task.complete_task()
            self.metrics.update_completion(duration)
            
            self.logger.info(f"Task completed: {task.id}, duration: {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            task.fail_task(error_msg)
            self.metrics.update_failure()
            
            self.logger.error(f"Task failed: {task.id}, error: {error_msg}, duration: {duration:.2f}s")
            raise
    
    async def add_task(self, task: WorkerTask):
        """添加任務到隊列"""
        await self.task_queue.put(task)
    
    async def start(self):
        """啟動異步 Worker"""
        if self.is_running:
            return
        
        self.is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        self.logger.info(f"Async worker started: {self.worker_id}")
    
    async def stop(self):
        """停止異步 Worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Async worker stopped: {self.worker_id}")
    
    async def _worker_loop(self):
        """Worker 主循環"""
        while self.is_running:
            try:
                # 等待任務，設定超時避免無限等待
                task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                await self.execute_task(task)
                
            except asyncio.TimeoutError:
                # 超時是正常的，繼續循環
                continue
            except Exception as e:
                self.logger.error(f"Worker loop error: {e}")
                # 發生錯誤時短暫休息
                await asyncio.sleep(1.0)

class ThreadWorker(BaseWorker):
    """多線程 Worker"""
    
    def __init__(self, worker_id: str, task_handler: Callable[[WorkerTask], Any], max_threads: int = 5):
        super().__init__(worker_id, WorkerType.THREAD)
        self.task_handler = task_handler
        self.max_threads = max_threads
        self.task_queue: queue.Queue = queue.Queue()
        self.executor: Optional[ThreadPoolExecutor] = None
        self._stop_event = threading.Event()
    
    async def execute_task(self, task: WorkerTask) -> Any:
        """執行多線程任務"""
        def sync_execute():
            start_time = time.time()
            
            try:
                task.start_task(self.worker_id)
                result = self.task_handler(task)
                
                duration = time.time() - start_time
                task.complete_task()
                self.metrics.update_completion(duration)
                
                self.logger.info(f"Task completed: {task.id}, duration: {duration:.2f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                
                task.fail_task(error_msg)
                self.metrics.update_failure()
                
                self.logger.error(f"Task failed: {task.id}, error: {error_msg}")
                raise
        
        # 在線程池中執行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, sync_execute)
    
    def add_task(self, task: WorkerTask):
        """添加任務到隊列"""
        self.task_queue.put(task)
    
    async def start(self):
        """啟動多線程 Worker"""
        if self.is_running:
            return
        
        self.is_running = True
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)
        
        # 啟動工作線程
        for i in range(self.max_threads):
            threading.Thread(
                target=self._worker_thread,
                name=f"{self.worker_id}-thread-{i}",
                daemon=True
            ).start()
        
        self.logger.info(f"Thread worker started: {self.worker_id} with {self.max_threads} threads")
    
    async def stop(self):
        """停止多線程 Worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.logger.info(f"Thread worker stopped: {self.worker_id}")
    
    def _worker_thread(self):
        """工作線程主函數"""
        while self.is_running and not self._stop_event.is_set():
            try:
                # 從隊列取任務，設定超時
                task = self.task_queue.get(timeout=1.0)
                
                # 執行任務（同步）
                start_time = time.time()
                
                try:
                    task.start_task(f"{self.worker_id}-{threading.current_thread().name}")
                    result = self.task_handler(task)
                    
                    duration = time.time() - start_time
                    task.complete_task()
                    self.metrics.update_completion(duration)
                    
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    
                    task.fail_task(error_msg)
                    self.metrics.update_failure()
                    
                    self.logger.error(f"Task failed in thread: {task.id}, error: {error_msg}")
                
            except queue.Empty:
                # 隊列為空，繼續等待
                continue
            except Exception as e:
                self.logger.error(f"Worker thread error: {e}")
                time.sleep(1.0)

class WorkerManager:
    """Worker 管理器"""
    
    def __init__(self):
        self.workers: Dict[str, BaseWorker] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.dispatcher_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("worker_manager")
    
    def register_worker(self, worker: BaseWorker):
        """註冊 Worker"""
        self.workers[worker.worker_id] = worker
        self.logger.info(f"Worker registered: {worker.worker_id} ({worker.worker_type.value})")
    
    def unregister_worker(self, worker_id: str):
        """註銷 Worker"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            self.logger.info(f"Worker unregistered: {worker_id}")
    
    async def submit_task(self, task: WorkerTask):
        """提交任務"""
        await self.task_queue.put(task)
        self.logger.info(f"Task submitted: {task.id} ({task.task_type})")
    
    async def start(self):
        """啟動 Worker 管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 啟動所有 Worker
        for worker in self.workers.values():
            await worker.start()
        
        # 啟動任務分發器
        self.dispatcher_task = asyncio.create_task(self._task_dispatcher())
        
        self.logger.info("Worker manager started")
    
    async def stop(self):
        """停止 Worker 管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止任務分發器
        if self.dispatcher_task:
            self.dispatcher_task.cancel()
            try:
                await self.dispatcher_task
            except asyncio.CancelledError:
                pass
        
        # 停止所有 Worker
        for worker in self.workers.values():
            await worker.stop()
        
        self.logger.info("Worker manager stopped")
    
    async def _task_dispatcher(self):
        """任務分發器"""
        while self.is_running:
            try:
                # 等待任務
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # 選擇合適的 Worker
                worker = self._select_worker(task)
                if worker:
                    if worker.worker_type == WorkerType.ASYNC:
                        await worker.add_task(task)
                    elif worker.worker_type == WorkerType.THREAD:
                        worker.add_task(task)
                    else:
                        self.logger.warning(f"Unsupported worker type: {worker.worker_type}")
                else:
                    self.logger.warning(f"No suitable worker found for task: {task.id}")
                    # 將任務放回隊列稍後重試
                    await asyncio.sleep(1.0)
                    await self.task_queue.put(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(1.0)
    
    def _select_worker(self, task: WorkerTask) -> Optional[BaseWorker]:
        """選擇合適的 Worker"""
        # 簡單的負載均衡策略：選擇任務數最少的 Worker
        available_workers = [w for w in self.workers.values() if w.is_running]
        
        if not available_workers:
            return None
        
        # 按完成的任務數排序，選擇最少的
        return min(available_workers, key=lambda w: w.metrics.total_tasks)
    
    def get_worker_metrics(self) -> Dict[str, WorkerMetrics]:
        """獲取所有 Worker 指標"""
        return {
            worker_id: worker.get_metrics()
            for worker_id, worker in self.workers.items()
        }
```

---

## 🔧 **支援層重構建議**

### **✅ 推薦的支援層架構**

```python
"""
推薦的支援層重構架構
清晰分離基礎設施與業務邏輯
"""

class SupportLayerArchitecture:
    """支援層架構重構方案"""
    
    RECOMMENDED_STRUCTURE = {
        "configuration_management": {
            "components": [
                "ConfigurationManager",
                "SchemaValidator", 
                "EnvironmentHandler",
                "ConfigurationTemplates"
            ],
            "languages": ["Python"],
            "responsibility": "統一配置管理"
        },
        
        "data_schemas": {
            "components": [
                "BaseSchema",
                "FindingSchemas",
                "RequestResponseSchemas", 
                "ValidationSchemas"
            ],
            "languages": ["Python"],
            "responsibility": "資料結構定義"
        },
        
        "worker_framework": {
            "components": [
                "BaseWorker",
                "AsyncWorker",
                "ThreadWorker",
                "WorkerManager"
            ],
            "languages": ["Python", "Go"],
            "responsibility": "任務執行框架"
        },
        
        "infrastructure_tools": {
            "components": [
                "DatabaseConnector",
                "CacheManager",
                "MessageQueue",
                "MetricsCollector"
            ],
            "languages": ["Python", "Go"],
            "responsibility": "基礎設施工具"
        },
        
        "testing_framework": {
            "components": [
                "TestHarness",
                "MockServices", 
                "TestDataGenerator",
                "IntegrationTestSuite"
            ],
            "languages": ["Python"],
            "responsibility": "測試支援框架"
        }
    }
    
    ELIMINATION_TARGETS = [
        # 這些應該移到業務層
        "SQL_Injection_Detection",
        "SSRF_Detection", 
        "XSS_Detection",
        
        # 這些應該移到安全層
        "smart_idor_detector",
        "lateral_movement",
        "persistence_checker"
    ]
```

---

---

## 📚 **相關文件**

### **🔗 多層架構導航**
- 🏠 [**主導航文件**](../README.md) - 總體架構與快速導航
- 📋 **功能層文件**:
  - 🔴 [核心功能架構](README_CORE.md) - 智能管理與協調
  - 🛡️ [安全功能架構](README_SECURITY.md) - 漏洞檢測與安全掃描
  - 🏢 [業務功能架構](README_BUSINESS.md) - SCA/CSPM 服務實現
- 💻 **語言層文件**:
  - 🐍 [Python 開發指南](README_PYTHON.md) - 配置管理與業務協調
  - � [Go 開發指南](README_GO.md) - 高效能基礎設施服務
  - 🦀 [Rust 開發指南](README_RUST.md) - 安全關鍵組件開發

### **📊 架構分析報告**
- 📈 [Features 模組架構分析](../../../_out/FEATURES_MODULE_ARCHITECTURE_ANALYSIS.md)
- 🔍 [支援功能架構圖](../../../_out/architecture_diagrams/functional/FEATURES_SUPPORT_FUNCTIONS.mmd)
- 📋 [功能分類資料](../../../_out/architecture_diagrams/features_diagram_classification.json)

---

**�📝 版本**: v2.0 - Support Functions Architecture Guide  
**🔄 最後更新**: 2024-10-24  
**🔧 主要語言**: Python (基礎設施) + Go (高效能工具)  
**👥 維護團隊**: AIVA Infrastructure Team

**🚨 緊急重構建議**:
1. **立即移除**: 支援層中的檢測邏輯組件
2. **短期重構**: 拆分過於複雜的 Worker 系統  
3. **中期目標**: 建立清晰的基礎設施邊界
4. **長期規劃**: 實現完全插拔式的支援架構

*這是 AIVA Features 模組支援功能組件的完整架構指南，重點關注基礎設施清理和模組邊界重新定義。*

---

## 🔗 相關資源

### 模組開發指南
- 📖 [Python 開發指南](./PYTHON_DEVELOPMENT_GUIDE.md)
- 📖 [Go 開發指南](./GO_DEVELOPMENT_GUIDE.md)
- 📖 [Rust 開發指南](./RUST_DEVELOPMENT_GUIDE.md)
- 📖 [AI 引擎指南](./AI_ENGINE_GUIDE.md)
- 📖 [功能模組開發指南](./FEATURE_MODULES_DEVELOPMENT_GUIDE.md)
- 📖 [模組遷移指南](./MODULE_MIGRATION_GUIDE.md)

### 架構指南
- 📖 [跨語言 Schema 指南](../architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md)
- 📖 [兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 開發指南
- 📖 [開發快速指南](../development/DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

### 服務文檔
- 🔧 [Features 模組](../../services/features/README.md)
- 🔧 [Scan 引擎文檔](../../services/scan/README.md)

