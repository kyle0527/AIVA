# AIVA Features - 核心功能架構 🔧

> **定位**: 系統基礎服務、配置管理、效能監控  
> **規模**: 42 個核心組件 (2.3%)  
> **職責**: Configuration、Logging、Metrics、Health Checks、Error Handling

---

## 🎯 **核心功能在 AIVA 中的角色**

### **🚀 基礎設施定位**
核心功能模組是 AIVA Features 的「**基礎設施層**」，為所有其他功能模組提供基本服務：

```
🔧 核心基礎設施架構
├── ⚙️ 配置管理系統 (Configuration)
│   ├── 環境變數處理 (8組件)
│   ├── 配置檔案解析 (6組件)
│   └── 動態配置更新 (4組件)
├── 📝 日誌記錄系統 (Logging) 
│   ├── 結構化日誌 (5組件)
│   ├── 日誌輪轉 (3組件)
│   └── 遠端日誌傳送 (2組件)
├── 📊 效能指標收集 (Metrics)
│   ├── 系統指標 (4組件)
│   ├── 應用指標 (3組件)
│   └── 自訂指標 (2組件)
└── 🏥 健康檢查系統 (Health)
    ├── 服務狀態檢查 (3組件)
    ├── 依賴項檢查 (2組件)
    └── 資源使用監控 (2組件)
```

### **⚡ 核心組件統計**
- **配置管理**: 18 個組件 (42.9% - 系統配置基礎)
- **日誌系統**: 10 個組件 (23.8% - 可觀測性核心)
- **效能指標**: 9 個組件 (21.4% - 監控與警報)
- **健康檢查**: 5 個組件 (11.9% - 系統狀態管理)

---

## 🏗️ **核心架構模式**

### **⚙️ 配置管理系統**

```python
"""
AIVA 核心配置管理系統
提供統一的配置載入、驗證和動態更新機制
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, Type, TypeVar
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

T = TypeVar('T')

class ConfigFormat(Enum):
    """配置檔案格式"""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"
    TOML = "toml"

@dataclass
class ConfigSource:
    """配置來源定義"""
    source_type: str
    path: Optional[str] = None
    env_prefix: Optional[str] = None
    priority: int = 0
    format: ConfigFormat = ConfigFormat.YAML
    watch: bool = False

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

@dataclass
class SecurityConfig:
    """安全配置"""
    jwt_secret_key: str = ""
    jwt_expiration: int = 3600  # seconds
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    allowed_origins: list[str] = field(default_factory=list)
    api_key_header: str = "X-API-Key"

@dataclass
class LoggingConfig:
    """日誌配置"""
    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    enable_console: bool = True
    enable_remote: bool = False
    remote_endpoint: Optional[str] = None

@dataclass
class MetricsConfig:
    """指標配置"""
    enabled: bool = True
    endpoint: str = "/metrics"
    port: int = 9090
    namespace: str = "aiva"
    push_gateway: Optional[str] = None
    collection_interval: int = 15  # seconds

@dataclass
class AIVAConfig:
    """AIVA 主配置類"""
    environment: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 子配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # 功能模組配置
    sast: Dict[str, Any] = field(default_factory=dict)
    dag: Dict[str, Any] = field(default_factory=dict)
    cspm: Dict[str, Any] = field(default_factory=dict)

class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self):
        self._config: Optional[AIVAConfig] = None
        self._sources: list[ConfigSource] = []
        self._observers: list[Observer] = []
        self._callbacks: list[callable] = []
        
    def add_source(self, source: ConfigSource) -> None:
        """添加配置來源"""
        self._sources.append(source)
        # 按優先級排序
        self._sources.sort(key=lambda s: s.priority, reverse=True)
        
    def add_file_source(
        self, 
        path: str, 
        format: ConfigFormat = ConfigFormat.YAML,
        priority: int = 0,
        watch: bool = False
    ) -> None:
        """添加檔案配置來源"""
        source = ConfigSource(
            source_type="file",
            path=path,
            format=format,
            priority=priority,
            watch=watch
        )
        self.add_source(source)
        
    def add_env_source(
        self, 
        prefix: str = "AIVA_",
        priority: int = 100  # 環境變數優先級較高
    ) -> None:
        """添加環境變數配置來源"""
        source = ConfigSource(
            source_type="env",
            env_prefix=prefix,
            priority=priority,
            format=ConfigFormat.ENV
        )
        self.add_source(source)
    
    async def load_config(self) -> AIVAConfig:
        """載入配置"""
        config_data = {}
        
        # 按優先級順序載入配置
        for source in self._sources:
            try:
                source_data = await self._load_source(source)
                # 合併配置，低優先級的先載入，高優先級的覆蓋
                config_data = self._merge_config(config_data, source_data)
            except Exception as e:
                print(f"Failed to load config from {source}: {e}")
                
        # 轉換為配置物件
        self._config = self._dict_to_config(config_data)
        
        # 設定檔案監控
        await self._setup_file_watching()
        
        return self._config
    
    async def _load_source(self, source: ConfigSource) -> Dict[str, Any]:
        """載入單一配置來源"""
        if source.source_type == "file":
            return await self._load_file_config(source)
        elif source.source_type == "env":
            return self._load_env_config(source)
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
    
    async def _load_file_config(self, source: ConfigSource) -> Dict[str, Any]:
        """載入檔案配置"""
        if not source.path or not os.path.exists(source.path):
            return {}
            
        with open(source.path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if source.format == ConfigFormat.YAML:
            return yaml.safe_load(content) or {}
        elif source.format == ConfigFormat.JSON:
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported file format: {source.format}")
    
    def _load_env_config(self, source: ConfigSource) -> Dict[str, Any]:
        """載入環境變數配置"""
        config = {}
        prefix = source.env_prefix or ""
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前綴並轉換為配置路徑
                config_key = key[len(prefix):].lower()
                config_path = config_key.split('_')
                
                # 設定巢狀配置值
                current = config
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                    
                # 嘗試轉換類型
                current[config_path[-1]] = self._convert_value(value)
                
        return config
    
    def _convert_value(self, value: str) -> Any:
        """轉換配置值類型"""
        # 布林值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # 數字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
            
        # 列表 (逗號分隔)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
            
        return value
    
    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """合併配置字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AIVAConfig:
        """將字典轉換為配置物件"""
        # 提取子配置
        database_data = data.pop('database', {})
        security_data = data.pop('security', {})
        logging_data = data.pop('logging', {})
        metrics_data = data.pop('metrics', {})
        sast_data = data.pop('sast', {})
        dag_data = data.pop('dag', {})
        cspm_data = data.pop('cspm', {})
        
        return AIVAConfig(
            **data,
            database=DatabaseConfig(**database_data),
            security=SecurityConfig(**security_data),
            logging=LoggingConfig(**logging_data),
            metrics=MetricsConfig(**metrics_data),
            sast=sast_data,
            dag=dag_data,
            cspm=cspm_data
        )
    
    async def _setup_file_watching(self) -> None:
        """設定檔案監控"""
        for source in self._sources:
            if source.source_type == "file" and source.watch and source.path:
                handler = ConfigFileHandler(self._on_config_changed)
                observer = Observer()
                observer.schedule(handler, str(Path(source.path).parent), recursive=False)
                observer.start()
                self._observers.append(observer)
    
    def _on_config_changed(self, file_path: str) -> None:
        """配置檔案變更回調"""
        asyncio.create_task(self._reload_config())
    
    async def _reload_config(self) -> None:
        """重新載入配置"""
        try:
            old_config = self._config
            new_config = await self.load_config()
            
            # 通知配置變更
            for callback in self._callbacks:
                await callback(old_config, new_config)
                
        except Exception as e:
            print(f"Failed to reload config: {e}")
    
    def on_config_changed(self, callback: callable) -> None:
        """註冊配置變更回調"""
        self._callbacks.append(callback)
    
    def get_config(self) -> AIVAConfig:
        """獲取當前配置"""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config
    
    def cleanup(self) -> None:
        """清理資源"""
        for observer in self._observers:
            observer.stop()
            observer.join()

class ConfigFileHandler(FileSystemEventHandler):
    """配置檔案監控處理器"""
    
    def __init__(self, callback: callable):
        self.callback = callback
        
    def on_modified(self, event):
        if not event.is_directory:
            self.callback(event.src_path)

# 全域配置管理器實例
config_manager = ConfigurationManager()

async def setup_configuration() -> AIVAConfig:
    """設定配置管理"""
    # 添加配置來源
    config_manager.add_file_source("config/default.yaml", priority=0, watch=True)
    config_manager.add_file_source("config/local.yaml", priority=50, watch=True)
    config_manager.add_env_source("AIVA_", priority=100)
    
    # 載入配置
    config = await config_manager.load_config()
    return config

def get_config() -> AIVAConfig:
    """獲取當前配置"""
    return config_manager.get_config()
```

### **📝 結構化日誌系統**

```python
"""
AIVA 結構化日誌系統
提供統一的日誌記錄、格式化和分發機制
"""

import logging
import json
import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import contextvars
import traceback
import aiofiles
from logging.handlers import RotatingFileHandler

class LogLevel(Enum):
    """日誌等級"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogContext:
    """日誌上下文"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service: str = "aiva-features"
    component: Optional[str] = None
    operation: Optional[str] = None

@dataclass
class LogRecord:
    """結構化日誌記錄"""
    timestamp: str
    level: str
    message: str
    service: str
    component: Optional[str] = None
    operation: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {k: v for k, v in asdict(self).items() if v is not None}

# 上下文變數
log_context: contextvars.ContextVar[LogContext] = contextvars.ContextVar('log_context')

class StructuredLogger:
    """結構化日誌記錄器"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = logging.getLogger("aiva")
        self.logger.setLevel(getattr(logging, config.level.upper()))
        
        # 清除預設處理器
        self.logger.handlers.clear()
        
        # 設定處理器
        self._setup_handlers()
        
    def _setup_handlers(self) -> None:
        """設定日誌處理器"""
        # 控制台處理器
        if self.config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)
            
        # 檔案處理器
        if self.config.file_path:
            file_handler = RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)
    
    def _get_formatter(self) -> logging.Formatter:
        """獲取日誌格式器"""
        if self.config.format == "json":
            return JSONFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _create_log_record(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> LogRecord:
        """創建日誌記錄"""
        context = self._get_context()
        
        return LogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            message=message,
            service=context.service,
            component=context.component,
            operation=context.operation,
            request_id=context.request_id,
            user_id=context.user_id,
            session_id=context.session_id,
            **kwargs
        )
    
    def _get_context(self) -> LogContext:
        """獲取當前日誌上下文"""
        try:
            return log_context.get()
        except LookupError:
            return LogContext()
    
    def debug(self, message: str, **kwargs) -> None:
        """記錄 DEBUG 日誌"""
        record = self._create_log_record(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(json.dumps(record.to_dict()))
    
    def info(self, message: str, **kwargs) -> None:
        """記錄 INFO 日誌"""
        record = self._create_log_record(LogLevel.INFO, message, **kwargs)
        self.logger.info(json.dumps(record.to_dict()))
    
    def warning(self, message: str, **kwargs) -> None:
        """記錄 WARNING 日誌"""
        record = self._create_log_record(LogLevel.WARNING, message, **kwargs)
        self.logger.warning(json.dumps(record.to_dict()))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """記錄 ERROR 日誌"""
        error_data = None
        if exception:
            error_data = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        record = self._create_log_record(
            LogLevel.ERROR, 
            message, 
            error=error_data,
            **kwargs
        )
        self.logger.error(json.dumps(record.to_dict()))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """記錄 CRITICAL 日誌"""
        error_data = None
        if exception:
            error_data = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        record = self._create_log_record(
            LogLevel.CRITICAL, 
            message, 
            error=error_data,
            **kwargs
        )
        self.logger.critical(json.dumps(record.to_dict()))

class JSONFormatter(logging.Formatter):
    """JSON 格式器"""
    
    def format(self, record):
        # 如果 record.msg 已經是 JSON，直接返回
        if isinstance(record.msg, str) and record.msg.startswith('{'):
            return record.msg
        
        # 否則包裝為標準格式
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": "aiva-features",
            "logger": record.name,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)

class LogContextManager:
    """日誌上下文管理器"""
    
    def __init__(self, **context_data):
        self.context_data = context_data
        self.token = None
    
    def __enter__(self):
        current_context = self._get_current_context()
        
        # 合併上下文
        new_context = LogContext(
            **{**asdict(current_context), **self.context_data}
        )
        
        self.token = log_context.set(new_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            log_context.reset(self.token)
    
    def _get_current_context(self) -> LogContext:
        """獲取當前上下文"""
        try:
            return log_context.get()
        except LookupError:
            return LogContext()

class OperationLogger:
    """操作日誌記錄器"""
    
    def __init__(self, logger: StructuredLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        
        with LogContextManager(operation=self.operation):
            self.logger.info(f"Operation started: {self.operation}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        
        with LogContextManager(operation=self.operation):
            if exc_type is None:
                self.logger.info(
                    f"Operation completed: {self.operation}",
                    duration_ms=duration
                )
            else:
                self.logger.error(
                    f"Operation failed: {self.operation}",
                    exception=exc_val,
                    duration_ms=duration
                )

# 全域日誌記錄器
logger: Optional[StructuredLogger] = None

def setup_logging(config: LoggingConfig) -> StructuredLogger:
    """設定日誌系統"""
    global logger
    logger = StructuredLogger(config)
    return logger

def get_logger() -> StructuredLogger:
    """獲取日誌記錄器"""
    if logger is None:
        raise RuntimeError("Logger not initialized")
    return logger

def log_operation(operation: str):
    """操作日誌裝飾器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with OperationLogger(get_logger(), operation):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with OperationLogger(get_logger(), operation):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
```

---

## 📊 **效能指標與監控**

### **⚡ Prometheus 整合**
```python
"""
AIVA 效能指標收集系統
集成 Prometheus 進行指標收集和監控
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from prometheus_client.exposition import generate_latest
import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import psutil

@dataclass
class MetricDefinition:
    """指標定義"""
    name: str
    description: str
    labels: list[str] = None
    
class AIVAMetrics:
    """AIVA 指標收集器"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # 初始化指標
        self._init_metrics()
        
    def _init_metrics(self) -> None:
        """初始化指標"""
        namespace = self.config.namespace
        
        # HTTP 請求指標
        self.http_requests_total = Counter(
            f"{namespace}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            f"{namespace}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        # SAST 分析指標
        self.sast_files_analyzed = Counter(
            f"{namespace}_sast_files_analyzed_total",
            "Total files analyzed by SAST",
            ["language", "status"],
            registry=self.registry
        )
        
        self.sast_analysis_duration = Histogram(
            f"{namespace}_sast_analysis_duration_seconds",
            "SAST analysis duration in seconds",
            ["language"],
            registry=self.registry
        )
        
        self.sast_vulnerabilities_found = Counter(
            f"{namespace}_sast_vulnerabilities_found_total",
            "Total vulnerabilities found by SAST",
            ["severity", "category"],
            registry=self.registry
        )
        
        # DAG 分析指標
        self.dag_nodes_processed = Counter(
            f"{namespace}_dag_nodes_processed_total",
            "Total DAG nodes processed",
            ["node_type"],
            registry=self.registry
        )
        
        self.dag_analysis_duration = Histogram(
            f"{namespace}_dag_analysis_duration_seconds",
            "DAG analysis duration in seconds",
            registry=self.registry
        )
        
        # 系統指標
        self.system_cpu_usage = Gauge(
            f"{namespace}_system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            f"{namespace}_system_memory_usage_bytes",
            "System memory usage in bytes",
            ["type"],  # total, available, used
            registry=self.registry
        )
        
        # 應用指標
        self.active_connections = Gauge(
            f"{namespace}_active_connections",
            "Number of active connections",
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            f"{namespace}_cache_operations_total",
            "Total cache operations",
            ["operation", "status"],  # get/set/delete, hit/miss/error
            registry=self.registry
        )
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """記錄 HTTP 請求指標"""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_sast_analysis(
        self,
        language: str,
        status: str,
        duration: float,
        vulnerabilities: Dict[str, int] = None
    ) -> None:
        """記錄 SAST 分析指標"""
        self.sast_files_analyzed.labels(
            language=language,
            status=status
        ).inc()
        
        self.sast_analysis_duration.labels(
            language=language
        ).observe(duration)
        
        if vulnerabilities:
            for severity, count in vulnerabilities.items():
                self.sast_vulnerabilities_found.labels(
                    severity=severity,
                    category="general"
                ).inc(count)
    
    def record_dag_analysis(
        self,
        nodes_count: int,
        duration: float,
        node_types: Dict[str, int] = None
    ) -> None:
        """記錄 DAG 分析指標"""
        if node_types:
            for node_type, count in node_types.items():
                self.dag_nodes_processed.labels(
                    node_type=node_type
                ).inc(count)
        
        self.dag_analysis_duration.observe(duration)
    
    def update_system_metrics(self) -> None:
        """更新系統指標"""
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu_usage.set(cpu_percent)
        
        # 記憶體使用情況
        memory = psutil.virtual_memory()
        self.system_memory_usage.labels(type="total").set(memory.total)
        self.system_memory_usage.labels(type="available").set(memory.available)
        self.system_memory_usage.labels(type="used").set(memory.used)
    
    def record_cache_operation(
        self,
        operation: str,
        status: str
    ) -> None:
        """記錄快取操作指標"""
        self.cache_operations.labels(
            operation=operation,
            status=status
        ).inc()
    
    def set_active_connections(self, count: int) -> None:
        """設定活躍連接數"""
        self.active_connections.set(count)
    
    def get_metrics_output(self) -> bytes:
        """獲取指標輸出（Prometheus 格式）"""
        return generate_latest(self.registry)

class MetricsCollector:
    """指標收集器"""
    
    def __init__(self, metrics: AIVAMetrics, config: MetricsConfig):
        self.metrics = metrics
        self.config = config
        self._running = False
        self._task = None
    
    async def start(self) -> None:
        """開始指標收集"""
        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
    
    async def stop(self) -> None:
        """停止指標收集"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _collection_loop(self) -> None:
        """指標收集循環"""
        while self._running:
            try:
                # 更新系統指標
                self.metrics.update_system_metrics()
                
                # 等待下次收集
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # 錯誤時等待較短時間

class MetricsMiddleware:
    """HTTP 指標中間件"""
    
    def __init__(self, metrics: AIVAMetrics):
        self.metrics = metrics
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        self.metrics.record_http_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response

# 全域指標收集器
metrics: Optional[AIVAMetrics] = None
collector: Optional[MetricsCollector] = None

def setup_metrics(config: MetricsConfig) -> AIVAMetrics:
    """設定指標系統"""
    global metrics, collector
    
    metrics = AIVAMetrics(config)
    collector = MetricsCollector(metrics, config)
    
    return metrics

def get_metrics() -> AIVAMetrics:
    """獲取指標收集器"""
    if metrics is None:
        raise RuntimeError("Metrics not initialized")
    return metrics

async def start_metrics_collection():
    """開始指標收集"""
    if collector:
        await collector.start()

async def stop_metrics_collection():
    """停止指標收集"""
    if collector:
        await collector.stop()
```

---

## 🏥 **健康檢查系統**

### **🔍 服務狀態監控**
```python
"""
AIVA 健康檢查系統
提供服務健康狀態檢查和依賴項監控
"""

import asyncio
import aiohttp
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import psutil
from datetime import datetime, timedelta

class HealthStatus(Enum):
    """健康狀態"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """健康檢查結果"""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any] = None

@dataclass
class SystemHealth:
    """系統健康狀況"""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details or {}
                }
                for check in self.checks
            ]
        }

class HealthChecker:
    """健康檢查器基類"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def check(self) -> HealthCheckResult:
        """執行健康檢查"""
        raise NotImplementedError

class DatabaseHealthChecker(HealthChecker):
    """資料庫健康檢查器"""
    
    def __init__(self, database_config: DatabaseConfig):
        super().__init__("database")
        self.config = database_config
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            # 這裡應該實際連接資料庫
            # 暫時模擬檢查
            await asyncio.sleep(0.01)  # 模擬延遲
            
            # 檢查連接池狀態
            pool_status = await self._check_connection_pool()
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection is healthy",
                duration_ms=duration,
                timestamp=datetime.now(),
                details=pool_status
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                duration_ms=duration,
                timestamp=datetime.now()
            )
    
    async def _check_connection_pool(self) -> Dict[str, Any]:
        """檢查連接池狀態"""
        return {
            "max_connections": self.config.pool_size,
            "active_connections": 5,  # 模擬值
            "idle_connections": 3,    # 模擬值
        }

class RedisHealthChecker(HealthChecker):
    """Redis 健康檢查器"""
    
    def __init__(self, redis_url: str):
        super().__init__("redis")
        self.redis_url = redis_url
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            # 這裡應該實際連接 Redis
            # 暫時模擬檢查
            await asyncio.sleep(0.005)  # 模擬延遲
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Redis connection is healthy",
                duration_ms=duration,
                timestamp=datetime.now(),
                details={
                    "url": self.redis_url,
                    "ping_response": "PONG"
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                duration_ms=duration,
                timestamp=datetime.now()
            )

class ExternalServiceHealthChecker(HealthChecker):
    """外部服務健康檢查器"""
    
    def __init__(self, name: str, url: str, timeout: float = 5.0):
        super().__init__(name)
        self.url = url
        self.timeout = timeout
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.url) as response:
                    duration = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"Service {self.name} is healthy",
                            duration_ms=duration,
                            timestamp=datetime.now(),
                            details={
                                "url": self.url,
                                "status_code": response.status,
                                "response_time_ms": duration
                            }
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.DEGRADED,
                            message=f"Service {self.name} returned status {response.status}",
                            duration_ms=duration,
                            timestamp=datetime.now(),
                            details={
                                "url": self.url,
                                "status_code": response.status
                            }
                        )
                        
        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service {self.name} timed out",
                duration_ms=duration,
                timestamp=datetime.now(),
                details={"url": self.url, "timeout": self.timeout}
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service {self.name} failed: {str(e)}",
                duration_ms=duration,
                timestamp=datetime.now(),
                details={"url": self.url, "error": str(e)}
            )

class SystemResourceChecker(HealthChecker):
    """系統資源檢查器"""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0):
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            # 檢查 CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 檢查記憶體使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 檢查磁碟使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            duration = (time.time() - start_time) * 1000
            
            # 判斷健康狀況
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.DEGRADED
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else HealthStatus.UNHEALTHY
                messages.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                messages.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status
                messages.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources are healthy"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                duration_ms=duration,
                timestamp=datetime.now(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2)
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                duration_ms=duration,
                timestamp=datetime.now()
            )

class HealthManager:
    """健康管理器"""
    
    def __init__(self):
        self.checkers: List[HealthChecker] = []
        self.start_time = datetime.now()
    
    def add_checker(self, checker: HealthChecker) -> None:
        """添加健康檢查器"""
        self.checkers.append(checker)
    
    async def check_health(self) -> SystemHealth:
        """執行所有健康檢查"""
        tasks = [checker.check() for checker in self.checkers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 處理異常情況
                check_results.append(HealthCheckResult(
                    name=self.checkers[i].name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(result)}",
                    duration_ms=0,
                    timestamp=datetime.now()
                ))
            else:
                check_results.append(result)
        
        # 計算整體狀態
        overall_status = self._calculate_overall_status(check_results)
        
        # 計算運行時間
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return SystemHealth(
            status=overall_status,
            checks=check_results,
            timestamp=datetime.now(),
            uptime_seconds=uptime
        )
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """計算整體健康狀況"""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        
        # 如果有任何檢查失敗，系統就不健康
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # 如果有任何檢查降級，系統就降級
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # 如果有未知狀態，系統狀態未知
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        
        # 所有檢查都通過
        return HealthStatus.HEALTHY

# 全域健康管理器
health_manager = HealthManager()

def setup_health_checks(config: AIVAConfig) -> None:
    """設定健康檢查"""
    # 添加資料庫檢查
    health_manager.add_checker(DatabaseHealthChecker(config.database))
    
    # 添加系統資源檢查
    health_manager.add_checker(SystemResourceChecker())
    
    # 可以添加更多檢查器...

async def get_health() -> SystemHealth:
    """獲取系統健康狀況"""
    return await health_manager.check_health()
```

---

## 🛠️ **核心模組整合範例**

### **🚀 應用程式啟動**
```python
"""
AIVA Features 應用程式啟動模組
整合所有核心功能組件
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# 導入核心模組
from .config import setup_configuration, get_config
from .logging import setup_logging, get_logger, LogContextManager
from .metrics import setup_metrics, get_metrics, start_metrics_collection, stop_metrics_collection
from .health import setup_health_checks, get_health

class AIVAApplication:
    """AIVA 應用程式"""
    
    def __init__(self):
        self.app = None
        self.config = None
        self.logger = None
        self.metrics = None
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self) -> FastAPI:
        """初始化應用程式"""
        # 1. 載入配置
        self.config = await setup_configuration()
        print(f"Configuration loaded: environment={self.config.environment}")
        
        # 2. 設定日誌
        self.logger = setup_logging(self.config.logging)
        self.logger.info("Logging system initialized")
        
        # 3. 設定指標
        if self.config.metrics.enabled:
            self.metrics = setup_metrics(self.config.metrics)
            await start_metrics_collection()
            self.logger.info("Metrics collection started")
        
        # 4. 設定健康檢查
        setup_health_checks(self.config)
        self.logger.info("Health checks configured")
        
        # 5. 創建 FastAPI 應用
        self.app = FastAPI(
            title="AIVA Features API",
            description="AIVA Security Features Module",
            version="1.0.0",
            lifespan=self.lifespan
        )
        
        # 6. 配置中間件
        self._setup_middleware()
        
        # 7. 配置路由
        self._setup_routes()
        
        # 8. 設定信號處理
        self._setup_signal_handlers()
        
        self.logger.info("Application initialized successfully")
        return self.app
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """應用程式生命週期管理"""
        # 啟動
        self.logger.info("Application starting up...")
        yield
        
        # 關閉
        self.logger.info("Application shutting down...")
        await self.cleanup()
    
    def _setup_middleware(self):
        """設定中間件"""
        # CORS 中間件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.security.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 指標中間件
        if self.metrics:
            from .metrics import MetricsMiddleware
            self.app.add_middleware(MetricsMiddleware, metrics=self.metrics)
        
        # 日誌中間件
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            # 生成請求 ID
            import uuid
            request_id = str(uuid.uuid4())
            
            # 設定日誌上下文
            with LogContextManager(
                request_id=request_id,
                component="api",
                operation=f"{request.method} {request.url.path}"
            ):
                self.logger.info(
                    f"Request started: {request.method} {request.url.path}",
                    metadata={
                        "client_ip": request.client.host,
                        "user_agent": request.headers.get("user-agent"),
                        "request_size": request.headers.get("content-length", 0)
                    }
                )
                
                start_time = asyncio.get_event_loop().time()
                response = await call_next(request)
                duration = (asyncio.get_event_loop().time() - start_time) * 1000
                
                self.logger.info(
                    f"Request completed: {request.method} {request.url.path}",
                    metadata={
                        "status_code": response.status_code,
                        "response_time_ms": duration
                    }
                )
                
                return response
    
    def _setup_routes(self):
        """設定路由"""
        @self.app.get("/")
        async def root():
            return {"message": "AIVA Features API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            """健康檢查端點"""
            health = await get_health()
            
            if health.status.value == "healthy":
                status_code = 200
            elif health.status.value == "degraded":
                status_code = 200  # 降級但仍可服務
            else:
                status_code = 503  # 服務不可用
                
            return Response(
                content=health.to_dict(),
                status_code=status_code,
                media_type="application/json"
            )
        
        @self.app.get("/ready")
        async def readiness_check():
            """就緒檢查端點"""
            # 簡單的就緒檢查
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
        if self.config.metrics.enabled:
            @self.app.get("/metrics")
            async def metrics():
                """Prometheus 指標端點"""
                metrics_output = self.metrics.get_metrics_output()
                return Response(
                    content=metrics_output,
                    media_type="text/plain; version=0.0.4; charset=utf-8"
                )
    
    def _setup_signal_handlers(self):
        """設定信號處理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def cleanup(self):
        """清理資源"""
        self.logger.info("Cleaning up resources...")
        
        # 停止指標收集
        if self.config.metrics.enabled:
            await stop_metrics_collection()
            self.logger.info("Metrics collection stopped")
        
        # 清理配置管理器
        from .config import config_manager
        config_manager.cleanup()
        
        self.logger.info("Cleanup completed")
    
    async def run(self, host: str = None, port: int = None):
        """運行應用程式"""
        import uvicorn
        
        host = host or self.config.host
        port = port or self.config.port
        
        # 配置 uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_config=None,  # 使用我們自己的日誌系統
            access_log=False,  # 禁用 uvicorn 的訪問日誌
        )
        
        server = uvicorn.Server(config)
        
        self.logger.info(f"Starting server on {host}:{port}")
        
        # 在背景任務中運行服務器
        server_task = asyncio.create_task(server.serve())
        
        # 等待關閉信號
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        
        # 等待任一任務完成
        done, pending = await asyncio.wait(
            [server_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # 關閉服務器
        if server_task in pending:
            server.should_exit = True
            await server_task
        
        # 取消待處理的任務
        for task in pending:
            task.cancel()

# 應用程式工廠函數
async def create_app() -> FastAPI:
    """創建 AIVA 應用程式"""
    app_instance = AIVAApplication()
    return await app_instance.initialize()

# 主函數
async def main():
    """主函數"""
    app_instance = AIVAApplication()
    await app_instance.initialize()
    await app_instance.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

**📝 版本**: v2.0 - Core Functions Architecture Guide  
**🔄 最後更新**: 2024-10-24  
**🔧 Python 版本**: 3.11+  
**👥 維護團隊**: AIVA Core Infrastructure Team

*這是 AIVA Features 模組核心功能組件的完整架構指南，專注於配置管理、日誌記錄、效能監控和健康檢查等基礎設施功能的實現。*