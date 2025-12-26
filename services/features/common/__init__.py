"""
AIVA Features Common - 共享基礎設施與工具

此模組提供 Features 模組的共享組件，避免重複實現。

組件架構：
1. 網絡層（來自 aiva_common）:
   - RetryingAsyncClient: 智能重試的 HTTP 客戶端
   - RateLimiter: Token Bucket 速率限制器

2. 瀏覽器引擎:
   - BrowserEngine: Playwright 真實瀏覽器引擎

3. 檢測器 (detectors/):
   - PassiveFingerprinter: 被動指紋識別
   - SensitiveInfoDetector: 敏感信息檢測
   - JavaScriptAnalyzer: JavaScript 分析器

4. 提取器 (extractors/):
   - DynamicContentExtractor: 動態內容提取
   - AjaxApiHandler: AJAX/API 處理
   - JsInteractionSimulator: JS 交互模擬

5. 測試器 (testers/):
   - CrossUserTester: 跨用戶測試
   - VerticalEscalationTester: 垂直權限提升測試

6. 統計 (worker_statistics.py):
   - StatisticsCollector: 統計信息收集

使用範例：
    from aiva_common.utils.network import RetryingAsyncClient, RateLimiter
    from services.features.common.browser_engine import BrowserEngine
    from services.features.common.detectors.passive_fingerprinter import PassiveFingerprinter

    # 創建 HTTP 客戶端（帶重試和速率限制）
    client = RetryingAsyncClient(retries=3, timeout=20.0)
    
    # 創建瀏覽器引擎
    browser = BrowserEngine(headless=True, max_contexts=5)
    
    # 使用被動指紋識別
    fingerprinter = PassiveFingerprinter()
    fingerprints = fingerprinter.from_headers(response.headers)

架構重組 (2025-12-20):
- 從 scan/python_engine 遷移核心能力
- 統一使用 aiva_common 網絡工具
- 提供完整的共享基礎設施
"""

from .worker_statistics import StatisticsCollector, ErrorCategory

__all__ = [
    # 統計收集
    "StatisticsCollector",
    "ErrorCategory",
    # 網絡工具從 aiva_common 導出（便於使用）
    "RetryingAsyncClient",
    "RateLimiter",
    # 瀏覽器引擎
    "BrowserEngine",
]

# 便捷導出（避免 Features 模組需要記住完整路徑）
try:
    from services.aiva_common.utils.network import RetryingAsyncClient, RateLimiter
except ImportError:
    # 向後兼容
    RetryingAsyncClient = None
    RateLimiter = None

try:
    from .browser_engine import BrowserEngine
except ImportError:
    BrowserEngine = None

__version__ = "1.0.0"  # 共享基礎設施版本