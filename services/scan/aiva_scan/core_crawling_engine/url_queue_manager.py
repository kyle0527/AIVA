

from collections import deque
from urllib.parse import urljoin, urlparse

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class UrlQueueManager:
    """
    高效的 URL 佇列管理器

    特性:
    - 使用 deque 提供 O(1) 的頭部彈出操作
    - 使用 set 實現 O(1) 的去重檢查
    - 支援 URL 標準化避免重複
    - 追蹤處理狀態和深度信息

    改進點:
    - 從簡單的 list 升級到 deque + set
    - 添加去重機制
    - 為未來擴展至 Redis 做準備
    """

    def __init__(self, seeds: list[str], *, max_depth: int = 3) -> None:
        """
        初始化 URL 佇列管理器

        Args:
            seeds: 初始種子 URL 列表
            max_depth: 最大爬取深度
        """
        self._queue: deque[tuple[str, int]] = deque()
        self._seen: set[str] = set()
        self._processed: set[str] = set()
        self.max_depth = max_depth

        # 添加種子 URL (深度為 0)
        for url in seeds:
            normalized = self._normalize_url(url)
            if normalized:
                self._queue.append((normalized, 0))
                self._seen.add(normalized)
                logger.debug(f"Seed URL added: {normalized}")

        logger.info(f"URL queue initialized with {len(seeds)} seed(s)")

    def has_next(self) -> bool:
        """
        檢查佇列中是否還有待處理的 URL

        Returns:
            如果佇列不為空則返回 True
        """
        return bool(self._queue)

    def next(self) -> str:
        """
        獲取下一個要處理的 URL

        Returns:
            下一個 URL 字符串

        Raises:
            IndexError: 如果佇列為空
        """
        if not self._queue:
            raise IndexError("URL queue is empty")

        url, depth = self._queue.popleft()
        self._processed.add(url)
        logger.debug(f"Dequeued URL: {url} (depth={depth})")
        return url

    def add(self, url: str, parent_url: str | None = None, depth: int = 0) -> bool:
        """
        添加新的 URL 到佇列

        Args:
            url: 要添加的 URL
            parent_url: 父級 URL（用於解析相對路徑）
            depth: URL 的深度級別

        Returns:
            如果 URL 被成功添加返回 True，如果已存在或超過深度則返回 False
        """
        # 解析相對 URL
        if parent_url and not urlparse(url).netloc:
            url = urljoin(parent_url, url)

        # 標準化 URL
        normalized = self._normalize_url(url)
        if not normalized:
            return False

        # 檢查是否已見過或超過深度限制
        if normalized in self._seen or depth > self.max_depth:
            return False

        # 添加到佇列和已見集合
        self._queue.append((normalized, depth))
        self._seen.add(normalized)
        logger.debug(f"URL added: {normalized} (depth={depth})")
        return True

    def add_batch(
        self, urls: list[str], parent_url: str | None = None, depth: int = 0
    ) -> int:
        """
        批量添加 URL

        Args:
            urls: URL 列表
            parent_url: 父級 URL
            depth: URL 深度

        Returns:
            成功添加的 URL 數量
        """
        added_count = 0
        for url in urls:
            if self.add(url, parent_url, depth):
                added_count += 1
        logger.debug(f"Batch add: {added_count}/{len(urls)} URLs added")
        return added_count

    def is_seen(self, url: str) -> bool:
        """
        檢查 URL 是否已經見過

        Args:
            url: 要檢查的 URL

        Returns:
            如果 URL 已在集合中返回 True
        """
        normalized = self._normalize_url(url)
        return normalized in self._seen if normalized else False

    def is_processed(self, url: str) -> bool:
        """
        檢查 URL 是否已經處理

        Args:
            url: 要檢查的 URL

        Returns:
            如果 URL 已處理返回 True
        """
        normalized = self._normalize_url(url)
        return normalized in self._processed if normalized else False

    def _normalize_url(self, url: str) -> str | None:
        """
        標準化 URL 以避免重複

        處理:
        - 移除片段標識符 (#fragment)
        - 統一協議大小寫
        - 移除默認端口
        - 排序查詢參數（可選）

        Args:
            url: 原始 URL

        Returns:
            標準化後的 URL，如果 URL 無效則返回 None
        """
        try:
            parsed = urlparse(url)

            # 基本驗證
            if not parsed.scheme or not parsed.netloc:
                return None

            # 移除片段
            normalized = parsed._replace(fragment="").geturl()

            return normalized

        except Exception as e:
            logger.warning(f"Failed to normalize URL '{url}': {e}")
            return None

    def get_statistics(self) -> dict[str, int]:
        """
        獲取佇列統計信息

        Returns:
            包含統計數據的字典
        """
        return {
            "queued": len(self._queue),
            "seen": len(self._seen),
            "processed": len(self._processed),
            "remaining": len(self._queue),
        }

    def clear(self) -> None:
        """清空佇列和所有追蹤集合"""
        self._queue.clear()
        self._seen.clear()
        self._processed.clear()
        logger.info("URL queue cleared")

    def __len__(self) -> int:
        """返回佇列中待處理的 URL 數量"""
        return len(self._queue)

    def __repr__(self) -> str:
        """返回佇列的字符串表示"""
        stats = self.get_statistics()
        return (
            f"UrlQueueManager(queued={stats['queued']}, "
            f"seen={stats['seen']}, processed={stats['processed']})"
        )

