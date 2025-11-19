"""
掃描上下文 - 管理掃描過程中的狀態和數據收集
"""



import time
from typing import TYPE_CHECKING

from services.aiva_common.schemas import (
    Asset,
    Fingerprints,
    JavaScriptAnalysisResult,
    ScanStartPayload,
    SensitiveMatch,
)
from services.aiva_common.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ScanContext:
    """
    掃描上下文管理器

    管理掃描過程中的所有狀態、統計信息和收集的數據。
    提供集中式的數據訪問和更新接口。

    特性:
    - 掃描狀態追蹤
    - 資產收集和管理
    - 統計信息實時更新
    - 指紋信息存儲
    - 掃描時長計算
    """

    def __init__(self, request: ScanStartPayload) -> None:
        """
        初始化掃描上下文

        Args:
            request: 掃描請求數據
        """
        self.request = request
        self._start_time = time.time()

        # 收集的資產
        self.assets: list[Asset] = []
        
        # Asset 去重 (參考 Rust A4 優化 - HashSet 去重)
        self._asset_keys: set[str] = set()

        # 統計信息
        self.urls_found = 0
        self.forms_found = 0
        self.apis_found = 0
        self.pages_crawled = 0

        # 指紋信息
        self.fingerprints: Fingerprints | None = None

        # 錯誤追蹤
        self.errors: list[dict[str, str]] = []

        logger.debug(f"Scan context initialized for scan_id: {request.scan_id}")

    @property
    def scan_duration(self) -> int:
        """
        獲取掃描持續時間（秒）

        Returns:
            掃描已經運行的秒數
        """
        return int(time.time() - self._start_time)

    def add_asset(self, asset: Asset) -> None:
        """
        添加資產到收集列表 (自動去重)

        Args:
            asset: 要添加的資產對象
        """
        # 生成唯一鍵 (type + value + method) - 參考 Rust A4 優化
        asset_key = f"{asset.type}:{asset.value}"
        if hasattr(asset, 'method') and asset.method:
            asset_key += f":{asset.method}"
        
        # 檢查是否已存在
        if asset_key in self._asset_keys:
            logger.debug(f"Asset skipped (duplicate): {asset.type} - {asset.value}")
            return
        
        # 添加新資產
        self._asset_keys.add(asset_key)
        self.assets.append(asset)
        logger.debug(f"Asset added: {asset.type} - {asset.value}")

    def increment_urls_found(self, count: int = 1) -> None:
        """
        增加發現的 URL 計數

        Args:
            count: 要增加的數量，默認為 1
        """
        self.urls_found += count

    def add_forms_found(self, count: int) -> None:
        """
        增加發現的表單計數

        Args:
            count: 發現的表單數量
        """
        self.forms_found += count

    def increment_apis_found(self, count: int = 1) -> None:
        """
        增加發現的 API 計數

        Args:
            count: 要增加的數量，默認為 1
        """
        self.apis_found += count

    def add_sensitive_match(self, match: SensitiveMatch) -> None:
        """
        記錄發現的敏感資料匹配

        Args:
            match: 敏感資料匹配對象
        """
        if not hasattr(self, "sensitive_matches"):
            self.sensitive_matches: list[SensitiveMatch] = []
        self.sensitive_matches.append(match)
        logger.info(f"Sensitive match recorded: {match.pattern_name} at {match.url or 'unknown'}")

    def add_js_analysis_result(self, result: JavaScriptAnalysisResult) -> None:
        """
        記錄 JavaScript 分析結果

        Args:
            result: JavaScript 分析結果對象
        """
        if not hasattr(self, "js_analysis_results"):
            self.js_analysis_results: list[JavaScriptAnalysisResult] = []
        self.js_analysis_results.append(result)
        logger.debug(f"JS analysis result recorded for {result.url}")

    def increment_pages_crawled(self, count: int = 1) -> None:
        """
        增加已爬取頁面計數

        Args:
            count: 要增加的數量，默認為 1
        """
        self.pages_crawled += count

    def set_fingerprints(self, fingerprints: Fingerprints) -> None:
        """
        設置指紋信息

        Args:
            fingerprints: 指紋信息對象
        """
        self.fingerprints = fingerprints
        logger.debug("Fingerprints set")

    def add_error(self, error_type: str, message: str, url: str | None = None) -> None:
        """
        記錄錯誤信息

        Args:
            error_type: 錯誤類型
            message: 錯誤消息
            url: 相關的 URL（可選）
        """
        error_info = {
            "type": error_type,
            "message": message,
            "timestamp": str(int(time.time())),
        }
        if url:
            error_info["url"] = url

        self.errors.append(error_info)
        logger.warning(f"Error recorded: {error_type} - {message}")

    def get_statistics(self) -> dict[str, int]:
        """
        獲取當前統計信息

        Returns:
            包含所有統計數據的字典
        """
        return {
            "urls_found": self.urls_found,
            "forms_found": self.forms_found,
            "apis_found": self.apis_found,
            "pages_crawled": self.pages_crawled,
            "assets_collected": len(self.assets),
            "scan_duration_seconds": self.scan_duration,
            "errors_count": len(self.errors),
        }

    def reset_statistics(self) -> None:
        """重置所有統計計數器"""
        self.urls_found = 0
        self.forms_found = 0
        self.apis_found = 0
        self.pages_crawled = 0
        logger.debug("Statistics reset")

    def __repr__(self) -> str:
        """返回上下文的字符串表示"""
        stats = self.get_statistics()
        return (
            f"ScanContext(scan_id={self.request.scan_id}, "
            f"urls={stats['urls_found']}, "
            f"forms={stats['forms_found']}, "
            f"duration={stats['scan_duration_seconds']}s)"
        )
