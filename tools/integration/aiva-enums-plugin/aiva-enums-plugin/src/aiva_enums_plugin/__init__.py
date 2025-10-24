from __future__ import annotations

from importlib import import_module

# 轉接模組化的 aiva_common.enums
enums_package = import_module("aiva_common.enums")

# 使用 enums 包的 __all__ 列表來獲取所有導出的枚舉
_all_names = getattr(enums_package, "__all__", [])
__all__ = _all_names

# 重新導出所有公開枚舉
for name in _all_names:
    if hasattr(enums_package, name):
        globals()[name] = getattr(enums_package, name)
