

import re
from urllib.parse import urlparse

from services.aiva_common.schemas import ScanScope
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScopeManager:
    """
    維護和管理掃描範圍。

    主要功能:
    - 控制允許掃描的主機和域名
    - 處理子域名包含規則
    - 管理排除模式（URL 路徑、文件類型等）
    - 驗證 URL 是否在掃描範圍內
    - 支持通配符和正則表達式模式

    使用範例:
        scope = ScanScope(
            allowed_hosts=["example.com"],
            include_subdomains=True,
            exclusions=["*/admin/*", "*.pdf"]
        )
        manager = ScopeManager(scope)

        # 檢查 URL 是否在範圍內
        if manager.is_in_scope("https://api.example.com/users"):
            # 處理這個 URL
            pass
    """

    def __init__(self, scope: ScanScope) -> None:
        """
        初始化範圍管理器

        Args:
            scope: 掃描範圍配置
        """
        self.scope = scope
        self._allowed_hosts_set = set(scope.allowed_hosts or [])
        self._exclusion_patterns = self._compile_exclusion_patterns(scope.exclusions)
        self._stats = {"total_checked": 0, "in_scope": 0, "out_of_scope": 0}

        logger.info(
            f"ScopeManager initialized: {len(self._allowed_hosts_set)} allowed hosts, "
            f"{len(self._exclusion_patterns)} exclusion patterns, "
            f"subdomains: {scope.include_subdomains}"
        )

    def is_in_scope(self, url: str) -> bool:
        """
        檢查 URL 是否在掃描範圍內

        Args:
            url: 要檢查的 URL

        Returns:
            bool: True 如果在範圍內，否則 False
        """
        self._stats["total_checked"] += 1

        try:
            parsed = urlparse(url)

            # 檢查主機是否允許
            if not self._is_host_allowed(parsed.hostname or ""):
                logger.debug(f"Host not allowed: {parsed.hostname}")
                self._stats["out_of_scope"] += 1
                return False

            # 檢查是否匹配排除模式
            if self._matches_exclusion(url):
                logger.debug(f"Matches exclusion pattern: {url}")
                self._stats["out_of_scope"] += 1
                return False

            self._stats["in_scope"] += 1
            return True

        except Exception as e:
            logger.warning(f"Failed to parse URL {url}: {e}")
            self._stats["out_of_scope"] += 1
            return False

    def is_host_allowed(self, hostname: str) -> bool:
        """
        檢查主機名是否被允許

        Args:
            hostname: 主機名

        Returns:
            bool: True 如果允許
        """
        return self._is_host_allowed(hostname)

    def add_allowed_host(self, hostname: str) -> None:
        """
        添加允許的主機

        Args:
            hostname: 主機名
        """
        if hostname and hostname not in self._allowed_hosts_set:
            self._allowed_hosts_set.add(hostname)
            self.scope.allowed_hosts.append(hostname)
            logger.info(f"Added allowed host: {hostname}")

    def remove_allowed_host(self, hostname: str) -> None:
        """
        移除允許的主機

        Args:
            hostname: 主機名
        """
        if hostname in self._allowed_hosts_set:
            self._allowed_hosts_set.remove(hostname)
            if hostname in self.scope.allowed_hosts:
                self.scope.allowed_hosts.remove(hostname)
            logger.info(f"Removed allowed host: {hostname}")

    def add_exclusion(self, pattern: str) -> None:
        """
        添加排除模式

        Args:
            pattern: 排除模式（支持通配符）
        """
        if pattern and pattern not in self.scope.exclusions:
            self.scope.exclusions.append(pattern)
            self._exclusion_patterns.append(self._compile_pattern(pattern))
            logger.info(f"Added exclusion pattern: {pattern}")

    def remove_exclusion(self, pattern: str) -> None:
        """
        移除排除模式

        Args:
            pattern: 排除模式
        """
        if pattern in self.scope.exclusions:
            self.scope.exclusions.remove(pattern)
            # 重新編譯所有模式
            self._exclusion_patterns = self._compile_exclusion_patterns(
                self.scope.exclusions
            )
            logger.info(f"Removed exclusion pattern: {pattern}")

    def get_stats(self) -> dict[str, int]:
        """
        獲取統計信息

        Returns:
            dict: 統計數據
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """重置統計信息"""
        self._stats = {"total_checked": 0, "in_scope": 0, "out_of_scope": 0}

    def get_allowed_hosts(self) -> list[str]:
        """獲取允許的主機列表"""
        return list(self._allowed_hosts_set)

    def get_exclusions(self) -> list[str]:
        """獲取排除模式列表"""
        return self.scope.exclusions.copy()

    def is_subdomain_included(self) -> bool:
        """是否包含子域名"""
        return self.scope.include_subdomains

    def set_include_subdomains(self, include: bool) -> None:
        """
        設置是否包含子域名

        Args:
            include: 是否包含
        """
        self.scope.include_subdomains = include
        logger.info(f"Set include_subdomains to: {include}")

    def _is_host_allowed(self, hostname: str) -> bool:
        """
        檢查主機名是否被允許（內部方法）

        Args:
            hostname: 主機名

        Returns:
            bool: True 如果允許
        """
        if not hostname:
            return False

        # 如果沒有指定允許的主機，則允許所有主機
        if not self._allowed_hosts_set:
            return True

        # 精確匹配
        if hostname in self._allowed_hosts_set:
            return True

        # 如果啟用子域名，檢查是否為允許主機的子域名
        if self.scope.include_subdomains:
            for allowed_host in self._allowed_hosts_set:
                if hostname.endswith(f".{allowed_host}"):
                    return True

        return False

    def _matches_exclusion(self, url: str) -> bool:
        """
        檢查 URL 是否匹配任何排除模式

        Args:
            url: URL

        Returns:
            bool: True 如果匹配排除模式
        """
        return any(pattern.search(url) for pattern in self._exclusion_patterns)

    def _compile_exclusion_patterns(self, exclusions: list[str]) -> list[re.Pattern]:
        """
        編譯排除模式為正則表達式

        Args:
            exclusions: 排除模式列表

        Returns:
            list: 編譯後的正則表達式列表
        """
        patterns = []
        for exclusion in exclusions:
            try:
                pattern = self._compile_pattern(exclusion)
                patterns.append(pattern)
            except re.error as e:
                logger.warning(f"Invalid exclusion pattern '{exclusion}': {e}")
        return patterns

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """
        將通配符模式編譯為正則表達式

        支持的模式：
        - * : 匹配任意字符（除了 /）
        - ** : 匹配任意字符（包括 /）
        - ? : 匹配單個字符
        - [abc] : 匹配字符集合中的任意字符

        Args:
            pattern: 通配符模式

        Returns:
            re.Pattern: 編譯後的正則表達式
        """
        # 轉義特殊字符（除了 *, ?, [, ]）
        regex = re.escape(pattern)

        # 替換通配符為正則表達式
        regex = regex.replace(r"\*\*", "DOUBLE_STAR")  # 暫時標記 **
        regex = regex.replace(r"\*", "[^/]*")  # * 匹配除 / 外的任意字符
        regex = regex.replace("DOUBLE_STAR", ".*")  # ** 匹配任意字符
        regex = regex.replace(r"\?", ".")  # ? 匹配單個字符

        # 添加錨點
        regex = f"^{regex}$"

        return re.compile(regex, re.IGNORECASE)

    def filter_urls(self, urls: list[str]) -> list[str]:
        """
        過濾 URL 列表，只返回在範圍內的 URL

        Args:
            urls: URL 列表

        Returns:
            list: 範圍內的 URL 列表
        """
        return [url for url in urls if self.is_in_scope(url)]

    def get_scope_summary(self) -> str:
        """
        獲取範圍配置摘要

        Returns:
            str: 摘要文本
        """
        allowed_hosts_display = (
            ", ".join(self._allowed_hosts_set) if self._allowed_hosts_set else "All"
        )

        lines = [
            "Scope Configuration:",
            f"  Allowed Hosts: {allowed_hosts_display}",
            f"  Include Subdomains: {self.scope.include_subdomains}",
            f"  Exclusion Patterns: {len(self.scope.exclusions)}",
        ]

        if self.scope.exclusions:
            lines.append("  Exclusions:")
            for exclusion in self.scope.exclusions[:5]:  # 只顯示前 5 個
                lines.append(f"    - {exclusion}")
            if len(self.scope.exclusions) > 5:
                lines.append(f"    ... and {len(self.scope.exclusions) - 5} more")

        stats = self.get_stats()
        if stats["total_checked"] > 0:
            lines.append("\n  Statistics:")
            lines.append(f"    Total Checked: {stats['total_checked']}")
            lines.append(f"    In Scope: {stats['in_scope']}")
            lines.append(f"    Out of Scope: {stats['out_of_scope']}")
            if stats["total_checked"] > 0:
                in_scope_percent = (stats["in_scope"] / stats["total_checked"]) * 100
                lines.append(f"    In Scope %: {in_scope_percent:.1f}%")

        return "\n".join(lines)

    def validate_scope(self) -> tuple[bool, list[str]]:
        """
        驗證範圍配置的有效性

        Returns:
            tuple: (是否有效, 錯誤消息列表)
        """
        errors = []

        # 檢查排除模式的有效性
        for pattern in self.scope.exclusions:
            try:
                self._compile_pattern(pattern)
            except re.error as e:
                errors.append(f"Invalid exclusion pattern '{pattern}': {e}")

        # 檢查允許的主機是否為有效的主機名
        for host in self._allowed_hosts_set:
            if not self._is_valid_hostname(host):
                errors.append(f"Invalid hostname: '{host}'")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _is_valid_hostname(self, hostname: str) -> bool:
        """
        檢查主機名是否有效

        Args:
            hostname: 主機名

        Returns:
            bool: True 如果有效
        """
        if not hostname:
            return False

        # 簡單的主機名驗證
        # 允許字母、數字、連字符和點
        pattern = (
            r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
            r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        return bool(re.match(pattern, hostname))

    def clone(self) -> ScopeManager:
        """
        克隆當前範圍管理器

        Returns:
            ScopeManager: 新的範圍管理器實例
        """
        import copy

        new_scope = copy.deepcopy(self.scope)
        return ScopeManager(new_scope)

    def merge_with(self, other: ScopeManager) -> None:
        """
        合併另一個範圍管理器的配置

        Args:
            other: 另一個 ScopeManager 實例
        """
        # 合併允許的主機
        for host in other.get_allowed_hosts():
            self.add_allowed_host(host)

        # 合併排除模式
        for exclusion in other.get_exclusions():
            if exclusion not in self.scope.exclusions:
                self.add_exclusion(exclusion)

        # 使用更寬鬆的子域名設置
        if other.is_subdomain_included() and not self.is_subdomain_included():
            self.set_include_subdomains(True)

        logger.info("Merged scope configurations")
