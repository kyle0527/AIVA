"""
進階可選依賴處理工具模組
基於 Real Python, PEP 8, 和 importlib 最佳實踐

提供統一的可選依賴管理，包括：
- importlib.util.find_spec() 檢查
- try-except 導入模式  
- Mock 物件替代
- 清晰的錯誤消息
"""

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import sys

# 類型檢查時可用的導入
if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)

class OptionalDependency:
    """管理單個可選依賴的類別"""
    
    def __init__(self, name: str, install_name: Optional[str] = None):
        self.name = name
        self.install_name = install_name or name
        self.is_available = False
        self.module: Optional["ModuleType"] = None
        self.error_msg: Optional[str] = None
        self._check_availability()
    
    def _check_availability(self) -> None:
        """使用 importlib.util.find_spec 檢查依賴可用性"""
        try:
            spec = importlib.util.find_spec(self.name)
            if spec is not None:
                # 嘗試實際導入以確認可用性
                self.module = importlib.import_module(self.name)
                self.is_available = True
                logger.debug(f"Optional dependency '{self.name}' is available")
            else:
                self.error_msg = f"Package '{self.name}' not found"
                logger.debug(f"Optional dependency '{self.name}' not available: not found")
        except ImportError as e:
            self.error_msg = f"Import error for '{self.name}': {e}"
            logger.debug(f"Optional dependency '{self.name}' not available: {e}")
        except Exception as e:
            self.error_msg = f"Unexpected error loading '{self.name}': {e}"
            logger.warning(f"Unexpected error checking '{self.name}': {e}")
    
    def require(self) -> "ModuleType":
        """要求依賴必須可用，否則拋出有意義的錯誤"""
        if not self.is_available:
            install_cmd = f"pip install {self.install_name}"
            raise ImportError(
                f"Required optional dependency '{self.name}' is not available. "
                f"Install with: {install_cmd}. "
                f"Error: {self.error_msg}"
            )
        return self.module
    
    def get_or_none(self) -> Optional["ModuleType"]:
        """安全獲取模組，不可用時返回 None"""
        return self.module if self.is_available else None


class MockObject:
    """提供有用錯誤信息的 Mock 物件"""
    
    def __init__(self, name: str, install_name: str):
        self.name = name
        self.install_name = install_name
    
    def __getattr__(self, item: str) -> "MockObject":
        return MockObject(f"{self.name}.{item}", self.install_name)
    
    def __call__(self, *args, **kwargs) -> None:
        raise ImportError(
            f"Cannot use '{self.name}' because the package is not installed. "
            f"Install with: pip install {self.install_name}"
        )
    
    def __bool__(self) -> bool:
        return False
    
    def __str__(self) -> str:
        return f"<MockObject for unavailable {self.name}>"
    
    def __repr__(self) -> str:
        return self.__str__()


class OptionalDependencyManager:
    """管理多個可選依賴的管理器"""
    
    def __init__(self):
        self.dependencies: Dict[str, OptionalDependency] = {}
        self.mocks: Dict[str, MockObject] = {}
    
    def register(self, name: str, install_name: Optional[str] = None, 
                create_mock: bool = True) -> OptionalDependency:
        """註冊一個可選依賴"""
        dep = OptionalDependency(name, install_name)
        self.dependencies[name] = dep
        
        if create_mock and not dep.is_available:
            self.mocks[name] = MockObject(name, dep.install_name)
        
        return dep
    
    def is_available(self, name: str) -> bool:
        """檢查依賴是否可用"""
        return self.dependencies.get(name, OptionalDependency("")).is_available
    
    def require(self, name: str) -> "ModuleType":
        """要求依賴必須可用"""
        if name not in self.dependencies:
            raise ValueError(f"Dependency '{name}' not registered")
        return self.dependencies[name].require()
    
    def get_or_mock(self, name: str) -> Any:
        """獲取模組或其 Mock 替代"""
        if name not in self.dependencies:
            raise ValueError(f"Dependency '{name}' not registered")
        
        dep = self.dependencies[name]
        if dep.is_available:
            return dep.module
        else:
            return self.mocks.get(name, MockObject(name, dep.install_name))
    
    def get_or_none(self, name: str) -> Optional["ModuleType"]:
        """安全獲取模組"""
        if name not in self.dependencies:
            return None
        return self.dependencies[name].get_or_none()
    
    def check_all(self) -> Dict[str, bool]:
        """檢查所有已註冊依賴的狀態"""
        return {name: dep.is_available for name, dep in self.dependencies.items()}
    
    def get_missing(self) -> List[str]:
        """獲取缺失的依賴列表"""
        return [name for name, dep in self.dependencies.items() if not dep.is_available]


# --- 專用 Mock 實現 ---

class MockNumPy:
    """NumPy 的 Mock 實現，提供核心功能的無操作版本"""
    
    def __init__(self):
        self.float32 = float
        self.float64 = float
        self.int32 = int
        self.int64 = int
        self.ndarray = list  # 回退到 Python list
    
    def array(self, data, dtype=None):
        """模擬 numpy.array()"""
        # dtype 參數保留用於 API 相容性
        _ = dtype  # 標記為有意未使用
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]
    
    def zeros(self, shape, dtype=None):
        """模擬 numpy.zeros()"""
        # dtype 參數保留用於 API 相容性，可用於未來擴展
        _ = dtype  # 標記為有意未使用，恢復時可移除此行
        if isinstance(shape, int):
            return [0.0] * shape
        elif isinstance(shape, (list, tuple)) and len(shape) == 1:
            return [0.0] * shape[0]
        else:
            return []
    
    def ones(self, shape, dtype=None):
        """模擬 numpy.ones()"""
        # dtype 參數保留用於 API 相容性，可用於未來擴展
        _ = dtype  # 標記為有意未使用，恢復時可移除此行
        if isinstance(shape, int):
            return [1.0] * shape
        elif isinstance(shape, (list, tuple)) and len(shape) == 1:
            return [1.0] * shape[0]
        else:
            return []
    
    def random(self):
        """模擬 numpy.random 模組"""
        return MockNumPyRandom()
    
    def sqrt(self, x):
        """模擬 numpy.sqrt()"""
        import math
        if isinstance(x, (list, tuple)):
            return [math.sqrt(abs(val)) for val in x]
        return math.sqrt(abs(x))
    
    def dot(self, a, b):
        """簡化的矩陣乘法模擬"""
        # 對於標量
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a * b
        # 對於向量 (list)，回傳 0
        return 0.0
    
    def max(self, arr):
        """模擬 numpy.max()"""
        if isinstance(arr, (list, tuple)):
            return max(arr) if arr else 0
        return arr
    
    def min(self, arr):
        """模擬 numpy.min()"""
        if isinstance(arr, (list, tuple)):
            return min(arr) if arr else 0
        return arr
    
    def mean(self, arr):
        """模擬 numpy.mean()"""
        if isinstance(arr, (list, tuple)):
            return sum(arr) / len(arr) if arr else 0
        return arr
    
    def std(self, arr):
        """模擬 numpy.std()"""
        if isinstance(arr, (list, tuple)) and len(arr) > 1:
            mean_val = sum(arr) / len(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
            return variance ** 0.5
        return 0.0
    
    def sum(self, arr):
        """模擬 numpy.sum()"""
        if isinstance(arr, (list, tuple)):
            return sum(arr)
        return arr
    
    def any(self, arr, axis=None):
        """模擬 numpy.any()"""
        # axis 參數保留用於 API 相容性，可用於多維數組處理
        _ = axis  # 標記為有意未使用，恢復時可移除此行並實現軸向操作
        if isinstance(arr, (list, tuple)):
            return any(arr)
        return bool(arr)


class MockNumPyRandom:
    """NumPy random 模組的 Mock"""
    
    def randn(self, *shape):
        """模擬 numpy.random.randn()"""
        import random
        if len(shape) == 0:
            return random.gauss(0, 1)
        elif len(shape) == 1:
            return [random.gauss(0, 1) for _ in range(shape[0])]
        else:
            # 多維數組回傳嵌套列表
            total_size = 1
            for dim in shape:
                total_size *= dim
            return [random.gauss(0, 1) for _ in range(total_size)]


# --- 專用 Mock 註冊 ---

def register_ml_mocks(deps_manager: OptionalDependencyManager):
    """註冊機器學習相關的 Mock 實現"""
    
    # 註冊 numpy 並設置專用 Mock
    numpy_dep = deps_manager.register('numpy', 'numpy', create_mock=False)
    if not numpy_dep.is_available:
        deps_manager.mocks['numpy'] = MockNumPy()
    
    # 註冊其他 ML 庫保持原有 MockObject
    ml_deps = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'scipy': 'scipy'
    }
    
    for pkg_name, install_name in ml_deps.items():
        deps_manager.register(pkg_name, install_name)


# 全域依賴管理器實例
deps = OptionalDependencyManager()

# 預註冊常用依賴
COMMON_DEPENDENCIES = {
    'requests': 'requests', 
    'aiohttp': 'aiohttp',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'pydantic': 'pydantic',
    'sqlalchemy': 'sqlalchemy',
    'redis': 'redis',
    'celery': 'celery'
}

# 自動註冊常用依賴
for pkg_name, install_name in COMMON_DEPENDENCIES.items():
    deps.register(pkg_name, install_name)

# 註冊機器學習 Mock 實現
register_ml_mocks(deps)

# 便利函數
def has_sklearn() -> bool:
    """檢查 scikit-learn 是否可用"""
    return deps.is_available('sklearn') and deps.is_available('joblib')

def has_visualization() -> bool:
    """檢查可視化庫是否可用"""
    return deps.is_available('matplotlib') or deps.is_available('plotly')

def has_async_http() -> bool:
    """檢查異步 HTTP 庫是否可用"""
    return deps.is_available('aiohttp')

def has_web_framework() -> bool:
    """檢查 Web 框架是否可用"""
    return deps.is_available('fastapi') or deps.is_available('uvicorn')

def require_sklearn():
    """要求 scikit-learn 可用，進行組合檢查"""
    if not has_sklearn():
        missing = []
        if not deps.is_available('sklearn'):
            missing.append('scikit-learn')
        if not deps.is_available('joblib'):
            missing.append('joblib')
        
        raise ImportError(
            f"Machine learning functionality requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )
    
    return deps.require('sklearn'), deps.require('joblib')

# 導出主要接口
__all__ = [
    'OptionalDependency',
    'OptionalDependencyManager', 
    'MockObject',
    'deps',
    'has_sklearn',
    'has_visualization',
    'has_async_http',
    'has_web_framework',
    'require_sklearn'
]