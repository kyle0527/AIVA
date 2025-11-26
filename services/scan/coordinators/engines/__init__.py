"""
掃描引擎適配器包

導出所有引擎適配器，方便統一管理和使用。

使用範例：
```python
from services.scan.coordinators.engines import (
    BaseScannerAdapter,
    PythonAdapter,
    TypeScriptAdapter,
    RustAdapter,
    GoAdapter
)

# 創建適配器實例
python_adapter = PythonAdapter()
if await python_adapter.is_available():
    result = await python_adapter.scan(targets, options)
```
"""

from .base import BaseScannerAdapter
from .python_adapter import PythonAdapter
from .typescript_adapter import TypeScriptAdapter
from .rust_adapter import RustAdapter
from .go_adapter import GoAdapter

__all__ = [
    "BaseScannerAdapter",
    "PythonAdapter",
    "TypeScriptAdapter",
    "RustAdapter",
    "GoAdapter"
]
