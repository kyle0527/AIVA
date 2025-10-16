"""
aiva_schemas_plugin

轉接層（adapter）：將 `aiva_common.schemas` 的公開 API 集中再輸出。
這讓專案可以統一： `from aiva_schemas_plugin import ...`
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = []  # 會在模組載入後根據 aiva_common.schemas 內容填入


def _load_and_reexport() -> None:
    # 動態載入模組化的 schemas 結構
    # 現在 aiva_common.schemas 是一個包含多個子模組的包
    schemas_package = import_module("aiva_common.schemas")

    # 獲取 schemas 包的 __all__ 列表，這包含所有導出的類別
    public = getattr(schemas_package, "__all__", [])

    # 將所有公開類別 re-export 到當前命名空間
    for name in public:
        if hasattr(schemas_package, name):
            globals()[name] = getattr(schemas_package, name)

    # 允許 `import aiva_schemas_plugin as schemas` 與原行為相容
    globals()["schemas"] = globals()

    # 註冊公開 API
    globals()["__all__"] = sorted(public)


try:
    _load_and_reexport()
except Exception as exc:  # pragma: no cover - 只在環境未就緒時觸發
    # 在安裝或 CI 階段若找不到 aiva_common，給出清楚訊息
    raise ImportError(
        "aiva_schemas_plugin 需要可匯入的 'aiva_common.schemas'。"
        "請將 aiva_common 放入 PYTHONPATH 或在同一個環境中可用。"
    ) from exc
