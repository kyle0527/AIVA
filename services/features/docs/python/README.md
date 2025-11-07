# 🐍 Python開發模組指南

**導航**: [← 返回文檔中心](../README.md) | [← 返回主模組](../../README.md)

---

## 📑 目錄

- [Python模組架構](#python模組架構)
- [開發環境配置](#開發環境配置)
- [編碼規範與最佳實踐](#編碼規範與最佳實踐)
- [模組實現模式](#模組實現模式)
- [測試框架](#測試框架)
- [部署與打包](#部署與打包)
- [效能優化](#效能優化)

---

## 🏗️ Python模組架構

AIVA Features的Python模組採用統一架構，確保代碼一致性和可維護性。

### 📊 **Python代碼統計**
- **總檔案數**: 75個Python檔案
- **總代碼行數**: 12,002行 (占87%)
- **平均檔案大小**: 160行/檔案
- **模組覆蓋**: 6個主要功能模組

### **標準目錄結構**
```
function_*/                 # 功能模組根目錄
├── __init__.py            # 模組初始化
├── main.py               # 主要執行檔案
├── requirements.txt      # Python依賴
├── README.md            # 模組文檔
├── core/                # 核心實現
│   ├── __init__.py
│   ├── detector.py      # 檢測器基類
│   ├── engine.py        # 檢測引擎
│   └── analyzer.py      # 分析器
├── utils/               # 工具函數
│   ├── __init__.py
│   ├── helpers.py       # 輔助函數
│   ├── validators.py    # 驗證器
│   └── formatters.py    # 格式化器
├── tests/               # 測試檔案
│   ├── __init__.py
│   ├── test_detector.py
│   ├── test_engine.py
│   └── test_analyzer.py
└── config/              # 配置檔案
    ├── settings.py      # 設定檔案
    └── rules.json       # 檢測規則
```

---

## ⚙️ 開發環境配置

### **Python版本要求**
- **最低版本**: Python 3.8+
- **推薦版本**: Python 3.10+
- **支援平台**: Windows, Linux, macOS

### **虛擬環境設置**
```bash
# 創建虛擬環境
python -m venv aiva_env

# 啟動虛擬環境
# Windows
aiva_env\Scripts\activate
# Linux/macOS  
source aiva_env/bin/activate

# 安裝基礎依賴
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 開發依賴
```

### **必要開發工具**
```bash
# 代碼格式化
pip install black isort

# 靜態分析
pip install pylint flake8 mypy

# 測試框架
pip install pytest pytest-cov pytest-mock

# 文檔生成
pip install sphinx sphinx-rtd-theme

# 開發輔助
pip install pre-commit ipython jupyter
```

### **VS Code配置**
```json
{
    "python.defaultInterpreterPath": "./aiva_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

---

## 📝 編碼規範與最佳實踐

### **代碼風格**

#### **命名規範**
```python
# 類名 - PascalCase
class VulnerabilityDetector:
    pass

# 函數和變數 - snake_case
def detect_sql_injection():
    vulnerability_count = 0

# 常數 - UPPER_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30

# 私有成員 - 單下劃線前綴
class Detector:
    def _internal_method(self):
        self._private_var = "internal"
```

#### **型別提示**
```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class DetectionResult:
    vulnerability_type: str
    severity: str
    location: str
    confidence: float

def analyze_code(
    code: str, 
    rules: List[str],
    timeout: Optional[int] = None
) -> List[DetectionResult]:
    """分析代碼漏洞
    
    Args:
        code: 要分析的代碼
        rules: 檢測規則列表
        timeout: 超時時間(秒)
        
    Returns:
        檢測結果列表
        
    Raises:
        TimeoutError: 檢測超時
        ValueError: 無效的輸入參數
    """
    pass
```

### **文檔字符串規範**
```python
def detect_vulnerability(target_url: str, payload: str) -> Dict[str, any]:
    """檢測目標URL的漏洞
    
    這個函數會發送特定的payload到目標URL，
    並分析回應來判斷是否存在漏洞。
    
    Args:
        target_url (str): 目標URL
        payload (str): 檢測載荷
        
    Returns:
        Dict[str, any]: 檢測結果，包含:
            - found (bool): 是否發現漏洞
            - severity (str): 漏洞嚴重程度
            - details (str): 詳細描述
            
    Raises:
        requests.RequestException: 網路請求異常
        ValueError: URL格式錯誤
        
    Example:
        >>> result = detect_vulnerability("http://example.com", "' OR 1=1--")
        >>> print(result['found'])
        True
    """
    pass
```

### **異常處理模式**
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AIVAError(Exception):
    """AIVA基礎異常類"""
    pass

class DetectionError(AIVAError):
    """檢測異常"""
    pass

class NetworkError(AIVAError):  
    """網路異常"""
    pass

def safe_detection_wrapper(func):
    """安全檢測裝飾器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            logger.error(f"網路請求失敗: {e}")
            raise NetworkError(f"無法連接到目標: {e}")
        except Exception as e:
            logger.error(f"檢測過程異常: {e}")
            raise DetectionError(f"檢測失敗: {e}")
    return wrapper
```

---

## 🏭 模組實現模式

### **檢測器基類模式**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass 
class VulnerabilityResult:
    """漏洞檢測結果"""
    type: str
    severity: str
    confidence: float
    location: str
    description: str
    remediation: str

class BaseDetector(ABC):
    """檢測器基類"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def detect(self, target: str, **kwargs) -> List[VulnerabilityResult]:
        """執行檢測
        
        Args:
            target: 檢測目標
            **kwargs: 額外參數
            
        Returns:
            檢測結果列表
        """
        pass
        
    @abstractmethod
    def validate_target(self, target: str) -> bool:
        """驗證檢測目標"""
        pass
        
    def preprocess(self, target: str) -> str:
        """預處理檢測目標"""
        return target.strip()
        
    def postprocess(self, results: List[VulnerabilityResult]) -> List[VulnerabilityResult]:
        """後處理檢測結果"""
        return sorted(results, key=lambda x: x.confidence, reverse=True)
```

### **檢測引擎模式**
```python
from typing import List, Dict, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

class DetectionEngine:
    """統一檢測引擎"""
    
    def __init__(self):
        self.detectors: Dict[str, BaseDetector] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def register_detector(self, name: str, detector: BaseDetector):
        """註冊檢測器"""
        self.detectors[name] = detector
        self.logger.info(f"註冊檢測器: {name}")
        
    def run_detection(
        self, 
        target: str, 
        detector_names: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, List[VulnerabilityResult]]:
        """執行檢測"""
        
        detectors_to_run = detector_names or list(self.detectors.keys())
        results = {}
        
        if parallel:
            results = self._run_parallel_detection(target, detectors_to_run)
        else:
            results = self._run_sequential_detection(target, detectors_to_run)
            
        return results
        
    def _run_parallel_detection(self, target: str, detector_names: List[str]) -> Dict[str, List[VulnerabilityResult]]:
        """並行檢測"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_detector = {
                executor.submit(self.detectors[name].detect, target): name 
                for name in detector_names if name in self.detectors
            }
            
            for future in as_completed(future_to_detector):
                detector_name = future_to_detector[future]
                try:
                    results[detector_name] = future.result()
                except Exception as e:
                    self.logger.error(f"檢測器 {detector_name} 執行失敗: {e}")
                    results[detector_name] = []
                    
        return results
```

### **配置管理模式**
```python
import json
import os
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_cache: Dict[str, Any] = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """載入配置檔案"""
        if config_name in self.config_cache:
            return self.config_cache[config_name]
            
        config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        self.config_cache[config_name] = config
        return config
        
    def get_env_config(self, key: str, default: Any = None) -> Any:
        """獲取環境變數配置"""
        return os.environ.get(f"AIVA_{key.upper()}", default)
        
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """合併多個配置"""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged
```

---

## 🧪 測試框架

### **測試結構**
```
tests/
├── unit/                 # 單元測試
│   ├── test_detector.py
│   ├── test_engine.py
│   └── test_utils.py
├── integration/          # 整合測試
│   ├── test_full_scan.py
│   └── test_api.py
├── fixtures/            # 測試數據
│   ├── sample_code.py
│   └── test_payloads.json
└── conftest.py         # pytest配置
```

### **測試模式範例**
```python
import pytest
from unittest.mock import Mock, patch
from mymodule import SQLInjectionDetector, VulnerabilityResult

class TestSQLInjectionDetector:
    """SQL注入檢測器測試"""
    
    @pytest.fixture
    def detector(self):
        """檢測器實例"""
        config = {"timeout": 10, "max_payloads": 100}
        return SQLInjectionDetector(config)
        
    @pytest.fixture
    def mock_response(self):
        """模擬HTTP回應"""
        response = Mock()
        response.status_code = 200
        response.text = "Error: SQL syntax error"
        response.headers = {"Content-Type": "text/html"}
        return response
        
    def test_detect_basic_sql_injection(self, detector):
        """測試基本SQL注入檢測"""
        target = "http://example.com/login?id=1"
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = "SQL syntax error"
            mock_get.return_value.status_code = 200
            
            results = detector.detect(target)
            
            assert len(results) > 0
            assert results[0].type == "SQL_INJECTION"
            assert results[0].confidence > 0.8
            
    def test_validate_target_valid_url(self, detector):
        """測試有效URL驗證"""
        valid_url = "http://example.com"
        assert detector.validate_target(valid_url) is True
        
    def test_validate_target_invalid_url(self, detector):
        """測試無效URL驗證"""
        invalid_url = "not-a-url"
        assert detector.validate_target(invalid_url) is False
        
    @pytest.mark.parametrize("payload,expected", [
        ("' OR 1=1--", True),
        ("'; DROP TABLE users--", True),
        ("normal input", False),
    ])
    def test_payload_detection(self, detector, payload, expected):
        """測試不同payload的檢測"""
        # 實現測試邏輯
        pass
        
    @pytest.mark.asyncio
    async def test_async_detection(self, detector):
        """測試異步檢測"""
        # 異步測試實現
        pass
```

### **測試執行與覆蓋率**
```bash
# 執行所有測試
pytest

# 執行特定測試檔案
pytest tests/unit/test_detector.py

# 執行測試並生成覆蓋率報告
pytest --cov=mymodule --cov-report=html

# 執行測試並顯示詳細輸出
pytest -v

# 執行標記的測試
pytest -m "slow"  # 執行標記為slow的測試

# 並行執行測試
pytest -n 4  # 需要 pytest-xdist
```

---

## 📦 部署與打包

### **requirements.txt管理**
```
# requirements.txt - 核心依賴
requests>=2.25.0
urllib3>=1.26.0
beautifulsoup4>=4.9.0
lxml>=4.6.0
pyyaml>=5.4.0
click>=8.0.0
colorama>=0.4.4

# requirements-dev.txt - 開發依賴
pytest>=6.0.0
pytest-cov>=2.12.0
pytest-mock>=3.6.0
black>=21.0.0
isort>=5.9.0
pylint>=2.8.0
mypy>=0.910
```

### **setup.py配置**
```python
from setuptools import setup, find_packages

setup(
    name="aiva-features-module",
    version="1.0.0",
    description="AIVA Features Security Module",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AIVA Team",
    author_email="dev@aiva.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "urllib3>=1.26.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
        "colorama>=0.4.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "pylint>=2.8.0",
            "mypy>=0.910",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "aiva-detector=mymodule.main:main",
        ]
    },
)
```

### **Docker化**
```dockerfile
# Dockerfile
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔案
COPY requirements.txt .

# 安裝Python依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 設置環境變數
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 啟動命令
CMD ["python", "main.py"]
```

---

## 🚀 效能優化

### **記憶體優化**
```python
import gc
from typing import Iterator, Generator

class MemoryEfficientDetector:
    """記憶體效率檢測器"""
    
    def process_large_dataset(self, data_source: str) -> Generator[VulnerabilityResult, None, None]:
        """處理大數據集 - 使用生成器節省記憶體"""
        
        with open(data_source, 'r') as file:
            for line_num, line in enumerate(file):
                # 處理單行
                result = self._process_line(line)
                
                if result:
                    yield result
                    
                # 定期清理記憶體
                if line_num % 1000 == 0:
                    gc.collect()
                    
    def batch_process(self, items: List[str], batch_size: int = 100) -> Iterator[List[VulnerabilityResult]]:
        """批次處理 - 控制記憶體使用"""
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            results = []
            
            for item in batch:
                result = self._process_item(item)
                if result:
                    results.append(result)
                    
            yield results
            
            # 清理批次記憶體
            del batch, results
            gc.collect()
```

### **併發優化**
```python
import asyncio
import aiohttp
from typing import List, Coroutine

class AsyncDetector:
    """異步檢測器"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def detect_multiple_targets(self, targets: List[str]) -> List[VulnerabilityResult]:
        """並行檢測多個目標"""
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._detect_single_target(session, target) for target in targets]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 過濾異常結果
            valid_results = [r for r in results if not isinstance(r, Exception)]
            return valid_results
            
    async def _detect_single_target(self, session: aiohttp.ClientSession, target: str) -> VulnerabilityResult:
        """檢測單個目標"""
        
        async with self.semaphore:  # 控制併發數
            try:
                async with session.get(target, timeout=10) as response:
                    content = await response.text()
                    return self._analyze_response(content)
                    
            except asyncio.TimeoutError:
                raise TimeoutError(f"檢測超時: {target}")
            except Exception as e:
                raise DetectionError(f"檢測失敗: {e}")
```

### **快取優化**
```python
from functools import lru_cache
import hashlib
import pickle
from typing import Optional

class CachedDetector:
    """帶快取的檢測器"""
    
    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        
    @lru_cache(maxsize=128)
    def _cached_analysis(self, content_hash: str) -> Optional[VulnerabilityResult]:
        """快取分析結果"""
        # 實際的分析邏輯
        return self._perform_analysis(content_hash)
        
    def detect_with_cache(self, content: str) -> Optional[VulnerabilityResult]:
        """帶快取的檢測"""
        
        # 生成內容雜湊
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # 檢查快取
        result = self._cached_analysis(content_hash)
        
        return result
        
    def clear_cache(self):
        """清理快取"""
        self._cached_analysis.cache_clear()
        
    def cache_info(self):
        """快取統計資訊"""
        return self._cached_analysis.cache_info()
```

---

## 🔗 相關連結

### **開發指南**
- [🐹 Go開發指南](../go/README.md) - Go模組開發
- [🛡️ 安全模組](../security/README.md) - 完整實現參考
- [🔧 開發中模組](../development/README.md) - 進行中的開發

### **工具與資源**
- [Python官方文檔](https://docs.python.org/3/) - Python語言參考
- [pytest文檔](https://docs.pytest.org/) - 測試框架
- [Black代碼格式化](https://black.readthedocs.io/) - 代碼風格
- [MyPy型別檢查](https://mypy.readthedocs.io/) - 靜態型別檢查

### **最佳實踐資源**
- [PEP 8](https://www.python.org/dev/peps/pep-0008/) - Python風格指南
- [Google Python風格指南](https://google.github.io/styleguide/pyguide.html)
- [Real Python](https://realpython.com/) - Python學習資源

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Python Development Team*