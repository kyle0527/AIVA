# AIVA Features - Python 開發指南 🐍 (2025-11-06更新)

## 📑 目錄

- [🎯 **Python 在 AIVA 中的角色**](#python-在-aiva-中的角色)
  - [**🧠 核心定位**](#核心定位)
  - [**⚡ Python 組件統計**](#python-組件統計)
- [🏗️ **Python 架構模式**](#python-架構模式)
  - [**🎯 核心模式: 智能檢測管理器**](#核心模式-智能檢測管理器)
  - [**🛡️ 安全檢測模式: SQL 注入檢測器**](#安全檢測模式-sql-注入檢測器)
  - [**🔄 跨語言整合模式**](#跨語言整合模式)
- [🛠️ **Python 開發環境設定**](#python-開發環境設定)
  - [**📦 依賴管理**](#依賴管理)
  - [**🚀 快速開始**](#快速開始)
- [🧪 **測試策略**](#測試策略)
  - [**🔍 單元測試範例**](#單元測試範例)
  - [**📊 效能測試**](#效能測試)
- [📈 **效能優化指南**](#效能優化指南)
  - [**⚡ 異步最佳實踐**](#異步最佳實踐)
  - [**🧠 記憶體最佳化**](#記憶體最佳化)
- [🚨 **錯誤處理與日誌**](#錯誤處理與日誌)
  - [**🛡️ 統一錯誤處理**](#統一錯誤處理)
  - [**📊 結構化日誌**](#結構化日誌)
- [🔧 **部署與維運**](#部署與維運)
  - [**🐳 Docker 配置**](#docker-配置)
  - [**📊 監控與指標**](#監控與指標)
- [🔗 相關資源](#相關資源)
  - [模組開發指南](#模組開發指南)
  - [架構指南](#架構指南)
  - [開發指南](#開發指南)
  - [服務文檔](#服務文檔)

---

## 🎯 **Python 在 AIVA 中的角色**

### **🧠 核心定位**
Python 在 AIVA Features 模組中扮演「**智能協調者**」的角色：

```
🐍 Python 核心職責圖
├── 🎯 智能協調層 (核心功能)
│   ├── 統一智能檢測管理器 (20組件)
│   ├── 高價值目標識別系統 (14組件)  
│   └── 功能管理器 (多組件)
├── 🛡️ 安全檢測層 (安全功能)
│   ├── SQL 注入檢測引擎 (59組件)
│   ├── XSS 跨站腳本檢測 (63組件)
│   └── SSRF 請求偽造檢測 (58組件)
├── 🏢 業務整合層 (業務功能)  
│   ├── API 介面與整合
│   ├── 資料模型與配置
│   └── 結果彙整與報告
└── 🔧 基礎支援層 (支援功能)
    ├── Worker 系統 (31組件)
    ├── Schema 定義 (30組件) 
    ├── 配置管理 (22組件)
    └── 工具與輔助功能
```

### **⚡ Python 組件統計**
- **核心功能**: 46 個組件 (智能管理與協調)
- **安全功能**: 180 個組件 (漏洞檢測實現)  
- **業務功能**: 53 個組件 (API 與整合)
- **支援功能**: 444 個組件 (基礎設施)

---

## 🏗️ **Python 架構模式**

### **🎯 核心模式: 智能檢測管理器**

```python
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio
from aiva.core.detector import BaseDetector
from aiva.core.manager import DetectionManager

@dataclass  
class DetectionConfig:
    """檢測配置"""
    target_url: str
    detection_types: List[str]
    timeout: int = 30
    max_concurrent: int = 10
    
class UnifiedSmartDetectionManager:
    """統一智能檢測管理器 - Python 核心協調類"""
    
    def __init__(self):
        self.detectors: Dict[str, BaseDetector] = {}
        self.active_scans: Dict[str, asyncio.Task] = {}
        
    async def register_detector(self, name: str, detector: BaseDetector):
        """註冊檢測器"""
        self.detectors[name] = detector
        await detector.initialize()
        
    async def coordinate_detection(self, config: DetectionConfig) -> AsyncGenerator[Dict, None]:
        """協調多種檢測器執行智能檢測"""
        
        # 1. 智能任務分派
        tasks = self._create_detection_tasks(config)
        
        # 2. 並發執行控制  
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # 3. 即時結果流式返回
        async for result in self._execute_with_coordination(tasks, semaphore):
            yield self._enrich_result(result)
    
    async def _create_detection_tasks(self, config: DetectionConfig) -> List[asyncio.Task]:
        """創建檢測任務"""
        tasks = []
        for detection_type in config.detection_types:
            if detection_type in self.detectors:
                detector = self.detectors[detection_type]
                task = asyncio.create_task(
                    detector.detect(config.target_url)
                )
                tasks.append(task)
        return tasks
        
    async def _execute_with_coordination(self, tasks, semaphore):
        """協調執行任務"""
        for task in asyncio.as_completed(tasks):
            async with semaphore:
                try:
                    result = await task
                    yield result
                except Exception as e:
                    yield {"error": str(e), "task": task}
```

### **🛡️ 安全檢測模式: SQL 注入檢測器**

```python
import aiohttp
import asyncio
from typing import List, Dict, Optional
from enum import Enum

class InjectionType(Enum):
    BOOLEAN_BASED = "boolean_based"
    TIME_BASED = "time_based"  
    ERROR_BASED = "error_based"
    UNION_BASED = "union_based"
    STACKED_QUERIES = "stacked_queries"

class SQLiDetector(BaseDetector):
    """SQL 注入檢測器 - 多引擎檢測實現"""
    
    def __init__(self):
        self.payloads = self._load_payloads()
        self.engines = {
            InjectionType.BOOLEAN_BASED: BooleanBasedEngine(),
            InjectionType.TIME_BASED: TimeBasedEngine(),
            InjectionType.ERROR_BASED: ErrorBasedEngine(),
            InjectionType.UNION_BASED: UnionBasedEngine(),
            InjectionType.STACKED_QUERIES: StackedQueriesEngine()
        }
    
    async def detect(self, target_url: str, parameters: Dict[str, str] = None) -> Dict:
        """執行 SQL 注入檢測"""
        
        results = {
            "vulnerable": False,
            "injection_types": [],
            "payloads": [],
            "confidence": 0.0
        }
        
        # 並發測試所有引擎
        tasks = []
        for injection_type, engine in self.engines.items():
            task = asyncio.create_task(
                self._test_injection_type(engine, target_url, parameters, injection_type)
            )
            tasks.append(task)
        
        # 收集結果
        engine_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析和整合結果
        for result in engine_results:
            if isinstance(result, dict) and result.get("vulnerable"):
                results["vulnerable"] = True
                results["injection_types"].append(result["type"])
                results["payloads"].extend(result["successful_payloads"])
        
        # 計算整體置信度
        results["confidence"] = self._calculate_confidence(results)
        
        return results
    
    async def _test_injection_type(self, engine, target_url, parameters, injection_type):
        """測試特定類型的注入"""
        try:
            return await engine.test(target_url, parameters, self.payloads[injection_type])
        except Exception as e:
            return {"error": str(e), "type": injection_type}
```

### **🔄 跨語言整合模式**

```python
import ctypes
import json
from pathlib import Path

class RustSastBridge:
    """Python ↔ Rust SAST 引擎橋接"""
    
    def __init__(self):
        # 載入 Rust 編譯的動態庫
        lib_path = Path(__file__).parent / "target/release/libsast_engine.so"
        self.rust_lib = ctypes.CDLL(str(lib_path))
        
        # 定義 C 介面
        self.rust_lib.sast_scan.argtypes = [ctypes.c_char_p]
        self.rust_lib.sast_scan.restype = ctypes.c_char_p
        
    async def scan_with_rust_sast(self, code_path: str) -> Dict:
        """使用 Rust SAST 引擎進行掃描"""
        
        # 準備參數
        scan_config = {
            "target_path": code_path,
            "rules": "all",
            "output_format": "json"
        }
        
        config_json = json.dumps(scan_config).encode('utf-8')
        
        # 調用 Rust 函數
        result_ptr = self.rust_lib.sast_scan(config_json)
        result_json = ctypes.string_at(result_ptr).decode('utf-8')
        
        # 解析結果
        rust_result = json.loads(result_json)
        
        # 轉換為 Python 格式
        return self._convert_rust_result(rust_result)
    
    def _convert_rust_result(self, rust_result: Dict) -> Dict:
        """轉換 Rust 結果為 Python 標準格式"""
        return {
            "scan_id": rust_result.get("scan_id"),
            "vulnerabilities": [
                {
                    "type": vuln["vulnerability_type"],
                    "severity": vuln["severity"].lower(),
                    "file": vuln["location"]["file"],
                    "line": vuln["location"]["line"],
                    "description": vuln["message"]
                }
                for vuln in rust_result.get("vulnerabilities", [])
            ],
            "statistics": rust_result.get("stats", {})
        }

class GoServiceClient:
    """Python ↔ Go 服務客戶端"""
    
    def __init__(self, service_url: str = "http://localhost:8080"):
        self.service_url = service_url
        
    async def call_go_sca_service(self, project_path: str) -> Dict:
        """調用 Go SCA 服務"""
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "project_path": project_path,
                "scan_type": "dependency_check",
                "include_dev_deps": True
            }
            
            async with session.post(
                f"{self.service_url}/api/sca/scan", 
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Go SCA service error: {response.status}")
```

---

## 🛠️ **Python 開發環境設定**

### **📦 依賴管理**
```toml
# pyproject.toml
[tool.poetry]
name = "aiva-features-python"
version = "2.0.0"
description = "AIVA Features Python Components"

[tool.poetry.dependencies]
python = "^3.11"
asyncio = "*"
aiohttp = "^3.9.0"
pydantic = "^2.0.0"  
fastapi = "^0.104.0"
sqlalchemy = "^2.0.0"
redis = "^5.0.0"
celery = "^5.3.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"  
pytest-cov = "^4.1.0"
black = "^23.0.0"
isort = "^5.12.0"
mypy = "^1.5.0"
ruff = "^0.1.0"

[tool.poetry.group.security.dependencies]
bandit = "^1.7.0"
safety = "^2.3.0"
```

### **🚀 快速開始**
```bash
# 1. 環境設定
cd services/features/
# 確認全域 Python 環境
python --version  # Windows: venv\Scripts\activate

# 2. 安裝依賴
pip install poetry
poetry install

# 3. 開發工具設定
poetry run pre-commit install

# 4. 執行測試
poetry run pytest tests/ -v --cov

# 5. 程式碼品質檢查
poetry run black .
poetry run isort .  
poetry run mypy .
poetry run ruff check .
```

---

## 🧪 **測試策略**

### **🔍 單元測試範例**
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from aiva.features.sqli import SQLiDetector

@pytest.mark.asyncio
class TestSQLiDetector:
    
    async def test_boolean_based_injection(self):
        """測試布林型 SQL 注入檢測"""
        detector = SQLiDetector()
        
        # 模擬易受攻擊的目標
        with patch('aiohttp.ClientSession.request') as mock_request:
            # 設定不同回應來模擬布林型注入
            mock_request.side_effect = [
                AsyncMock(text=lambda: "Welcome user123"),  # 正常回應
                AsyncMock(text=lambda: "Welcome user123"),  # True 條件
                AsyncMock(text=lambda: "Invalid credentials")  # False 條件  
            ]
            
            result = await detector.detect(
                target_url="http://test.com/login",
                parameters={"username": "test", "password": "test"}
            )
            
            assert result["vulnerable"] == True
            assert InjectionType.BOOLEAN_BASED.value in result["injection_types"]
            assert result["confidence"] > 0.8
    
    async def test_time_based_injection(self):
        """測試時間型 SQL 注入檢測"""
        detector = SQLiDetector()
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            # 模擬時間延遲回應
            async def slow_response():
                await asyncio.sleep(5)  # 模擬 SQL WAITFOR DELAY
                return AsyncMock(text=lambda: "Login failed")
            
            mock_request.side_effect = [
                AsyncMock(text=lambda: "Login failed"),  # 正常回應 (<1s)
                slow_response()  # 延遲回應 (~5s)
            ]
            
            result = await detector.detect("http://test.com/search?q=test")
            
            assert result["vulnerable"] == True
            assert InjectionType.TIME_BASED.value in result["injection_types"]

@pytest.mark.integration 
class TestCrossLanguageIntegration:
    
    async def test_python_rust_sast_integration(self):
        """測試 Python ↔ Rust SAST 整合"""
        bridge = RustSastBridge()
        
        # 準備測試程式碼
        test_code_path = "/tmp/test_code/"
        self._create_vulnerable_code(test_code_path)
        
        # 執行 Rust SAST 掃描
        result = await bridge.scan_with_rust_sast(test_code_path)
        
        # 驗證結果格式和內容
        assert "vulnerabilities" in result
        assert len(result["vulnerabilities"]) > 0
        assert result["vulnerabilities"][0]["type"] in ["sql_injection", "xss", "path_traversal"]
```

### **📊 效能測試**
```python
import time
import asyncio
from aiva.features.manager import UnifiedSmartDetectionManager

@pytest.mark.performance
class TestPerformance:
    
    async def test_concurrent_detection_performance(self):
        """測試並發檢測效能"""
        manager = UnifiedSmartDetectionManager()
        
        # 註冊檢測器
        await manager.register_detector("sqli", SQLiDetector())
        await manager.register_detector("xss", XSSDetector())
        await manager.register_detector("ssrf", SSRFDetector())
        
        # 準備測試目標
        targets = [f"http://test{i}.com" for i in range(100)]
        
        start_time = time.time()
        
        # 並發檢測
        tasks = []
        for target in targets:
            config = DetectionConfig(
                target_url=target,
                detection_types=["sqli", "xss", "ssrf"],
                max_concurrent=10
            )
            task = asyncio.create_task(
                list(manager.coordinate_detection(config))
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 效能斷言
        assert duration < 60  # 100個目標應該在60秒內完成
        assert len(results) == 100
        
        # 輸出效能統計
        print(f"處理 {len(targets)} 個目標耗時: {duration:.2f}s")
        print(f"平均每個目標: {duration/len(targets):.2f}s")
```

---

## 📈 **效能優化指南**

### **⚡ 異步最佳實踐**
```python
# ✅ 良好實踐: 使用 asyncio 和 aiohttp
import asyncio
import aiohttp

async def efficient_batch_scanning(urls: List[str], max_concurrent: int = 10):
    """高效批次掃描"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scan_single_url(session: aiohttp.ClientSession, url: str):
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    return await process_response(response)
            except asyncio.TimeoutError:
                return {"url": url, "error": "timeout"}
    
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, limit_per_host=10)
    ) as session:
        tasks = [scan_single_url(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# ❌ 避免: 同步 HTTP 請求和阻塞操作
import requests  # 不推薦用於高併發

def slow_batch_scanning(urls: List[str]):  # 避免
    results = []
    for url in urls:  # 順序執行，效率低
        response = requests.get(url, timeout=30)  # 阻塞操作
        results.append(process_response(response))
    return results
```

### **🧠 記憶體最佳化**
```python
# ✅ 使用生成器和流式處理
async def stream_large_dataset(data_source: str) -> AsyncGenerator[Dict, None]:
    """流式處理大型資料集"""
    async with aiofiles.open(data_source, 'r') as f:
        async for line in f:
            if line.strip():
                yield json.loads(line)

# ✅ 適當的快取策略
from functools import lru_cache
import redis.asyncio as redis

class CachedDetector:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
    
    @lru_cache(maxsize=1000)  # 記憶體快取
    def get_payload_templates(self, attack_type: str) -> List[str]:
        return self._load_templates(attack_type)
    
    async def get_scan_result(self, target_hash: str) -> Optional[Dict]:
        """從 Redis 快取獲取掃描結果"""
        cached = await self.redis.get(f"scan_result:{target_hash}")
        return json.loads(cached) if cached else None
    
    async def cache_scan_result(self, target_hash: str, result: Dict, ttl: int = 3600):
        """快取掃描結果"""
        await self.redis.setex(
            f"scan_result:{target_hash}", 
            ttl, 
            json.dumps(result)
        )
```

---

## 🚨 **錯誤處理與日誌**

### **🛡️ 統一錯誤處理**
```python
import logging
from typing import Optional
from enum import Enum

class AivaErrorType(Enum):
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"  
    VALIDATION_ERROR = "validation_error"
    DETECTION_ERROR = "detection_error"
    INTEGRATION_ERROR = "integration_error"

class AivaException(Exception):
    """AIVA 統一異常類"""
    
    def __init__(self, error_type: AivaErrorType, message: str, details: Optional[Dict] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "timestamp": datetime.utcnow().isoformat()
        }

# 統一錯誤處理裝飾器
def handle_aiva_errors(func):
    """AIVA 錯誤處理裝飾器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except aiohttp.ClientTimeout:
            raise AivaException(
                AivaErrorType.TIMEOUT_ERROR,
                f"Request timeout in {func.__name__}",
                {"function": func.__name__, "args": str(args)[:100]}
            )
        except aiohttp.ClientError as e:
            raise AivaException(
                AivaErrorType.NETWORK_ERROR,
                f"Network error in {func.__name__}: {str(e)}",
                {"function": func.__name__, "original_error": str(e)}
            )
        except Exception as e:
            logging.exception(f"Unexpected error in {func.__name__}")
            raise AivaException(
                AivaErrorType.DETECTION_ERROR,
                f"Detection error in {func.__name__}: {str(e)}",
                {"function": func.__name__, "original_error": str(e)}
            )
    return wrapper
```

### **📊 結構化日誌**
```python
import structlog

# 配置結構化日誌
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class LoggingDetector(BaseDetector):
    """帶有詳細日誌的檢測器"""
    
    async def detect(self, target_url: str) -> Dict:
        scan_id = self._generate_scan_id()
        
        logger.info(
            "detection_started",
            scan_id=scan_id,
            target_url=target_url,
            detector_type=self.__class__.__name__
        )
        
        try:
            result = await self._perform_detection(target_url)
            
            logger.info(
                "detection_completed", 
                scan_id=scan_id,
                vulnerable=result.get("vulnerable", False),
                vulnerabilities_found=len(result.get("vulnerabilities", [])),
                duration=result.get("duration", 0)
            )
            
            return result
            
        except AivaException as e:
            logger.error(
                "detection_failed",
                scan_id=scan_id, 
                error_type=e.error_type.value,
                error_message=e.message,
                error_details=e.details
            )
            raise
```

---

## 🔧 **部署與維運**

### **🐳 Docker 配置**
```dockerfile
# Dockerfile.python
FROM python:3.11-slim

WORKDIR /app

# 系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 依賴
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# 應用程式碼
COPY . .

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import aiva.features; print('OK')" || exit 1

# 執行
CMD ["python", "-m", "aiva.features.main"]
```

### **📊 監控與指標**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Prometheus 指標
DETECTION_COUNTER = Counter('aiva_detections_total', 'Total detections', ['detector_type', 'status'])
DETECTION_DURATION = Histogram('aiva_detection_duration_seconds', 'Detection duration', ['detector_type'])
ACTIVE_SCANS = Gauge('aiva_active_scans', 'Number of active scans')

class MonitoredDetector(BaseDetector):
    """帶有監控的檢測器"""
    
    async def detect(self, target_url: str) -> Dict:
        ACTIVE_SCANS.inc()
        start_time = time.time()
        
        try:
            result = await self._perform_detection(target_url)
            DETECTION_COUNTER.labels(
                detector_type=self.__class__.__name__,
                status='success'
            ).inc()
            return result
            
        except Exception as e:
            DETECTION_COUNTER.labels(
                detector_type=self.__class__.__name__,
                status='error'  
            ).inc()
            raise
            
        finally:
            duration = time.time() - start_time
            DETECTION_DURATION.labels(
                detector_type=self.__class__.__name__
            ).observe(duration)
            ACTIVE_SCANS.dec()

# 啟動指標服務
start_http_server(8000)  # Prometheus metrics on :8000
```

---

**📝 版本**: v2.0 - Python Development Guide  
**🔄 最後更新**: 2025-10-24  
**🐍 Python 版本**: 3.11+  
**👥 維護團隊**: AIVA Python Development Team

*這是 AIVA Features 模組 Python 組件的完整開發指南，涵蓋了架構設計、開發模式、測試策略和部署運維的所有方面。*


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

