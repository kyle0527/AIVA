"""
Dynamic Engine Module

用於處理動態內容的模組，包括：
- 無頭瀏覽器池管理
- JavaScript 互動模擬
- 動態內容提取
- AJAX/API 端點處理
"""

from __future__ import annotations

from .ajax_api_handler import AjaxApiHandler
from .dynamic_content_extractor import (
    ContentType,
    DynamicContent,
    DynamicContentExtractor,
    ExtractionConfig,
    ExtractionStrategy,
    NetworkRequest,
)
from .headless_browser_pool import (
    BrowserInstance,
    BrowserStatus,
    BrowserType,
    HeadlessBrowserPool,
    PageInstance,
    PoolConfig,
)
from .js_interaction_simulator import (
    InteractionResult,
    InteractionType,
    JsEvent,
    JsInteractionSimulator,
)

__all__ = [
    # AJAX/API Handler
    "AjaxApiHandler",
    # Browser Pool
    "BrowserInstance",
    "BrowserStatus",
    "BrowserType",
    "HeadlessBrowserPool",
    "PageInstance",
    "PoolConfig",
    # Content Extractor
    "ContentType",
    "DynamicContent",
    "DynamicContentExtractor",
    "ExtractionConfig",
    "ExtractionStrategy",
    "NetworkRequest",
    # JS Simulator
    "InteractionResult",
    "InteractionType",
    "JsEvent",
    "JsInteractionSimulator",
]
