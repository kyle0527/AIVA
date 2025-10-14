"""
Intel Aggregator - 威脅情報聚合器

整合多個威脅情報源（VirusTotal, AbuseIPDB, Shodan 等）的主要聚合器。
支援異步查詢、結果快取、錯誤重試機制。
"""

import asyncio
from datetime import datetime, timedelta
import hashlib
import os
from typing import Any

from dotenv import load_dotenv
import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import vt

from services.aiva_common.enums import IntelSource, ThreatLevel

load_dotenv()

logger = structlog.get_logger(__name__)


class IntelAggregator:
    """
    威脅情報聚合器

    整合多個威脅情報源，提供統一的查詢介面。
    """

    def __init__(
        self,
        vt_api_key: str | None = None,
        abuseipdb_api_key: str | None = None,
        shodan_api_key: str | None = None,
        cache_ttl: int = 3600,
        max_concurrent: int = 5,
    ):
        """
        初始化威脅情報聚合器

        Args:
            vt_api_key: VirusTotal API Key
            abuseipdb_api_key: AbuseIPDB API Key
            shodan_api_key: Shodan API Key
            cache_ttl: 快取過期時間（秒）
            max_concurrent: 最大並發請求數
        """
        self.vt_api_key = vt_api_key or os.getenv("VIRUSTOTAL_API_KEY")
        self.abuseipdb_api_key = abuseipdb_api_key or os.getenv("ABUSEIPDB_API_KEY")
        self.shodan_api_key = shodan_api_key or os.getenv("SHODAN_API_KEY")

        self.cache_ttl = cache_ttl
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 快取存儲
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # HTTP 客戶端
        self.http_client: httpx.AsyncClient | None = None

        logger.info(
            "intel_aggregator_initialized",
            cache_ttl=cache_ttl,
            max_concurrent=max_concurrent,
            sources_available=self._get_available_sources(),
        )

    def _get_available_sources(self) -> list[str]:
        """獲取可用的情報源列表"""
        sources = []
        if self.vt_api_key:
            sources.append(IntelSource.VIRUSTOTAL)
        if self.abuseipdb_api_key:
            sources.append(IntelSource.ABUSEIPDB)
        if self.shodan_api_key:
            sources.append(IntelSource.SHODAN)
        return sources

    async def __aenter__(self):
        """異步上下文管理器進入"""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        if self.http_client:
            await self.http_client.aclose()

    def _generate_cache_key(self, indicator: str, source: str) -> str:
        """生成快取鍵"""
        return hashlib.sha256(f"{source}:{indicator}".encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> dict[str, Any] | None:
        """從快取獲取數據"""
        if cache_key not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key)
        if timestamp and datetime.now() - timestamp > timedelta(seconds=self.cache_ttl):
            # 快取過期
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache.get(cache_key)

    def _set_to_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        """設置快取數據"""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def query_virustotal(self, indicator: str, indicator_type: str) -> dict[str, Any]:
        """
        查詢 VirusTotal

        Args:
            indicator: 指標（IP/Domain/Hash）
            indicator_type: 指標類型 (ip_address, domain, file)

        Returns:
            VirusTotal 查詢結果
        """
        if not self.vt_api_key:
            return {"error": "VirusTotal API key not configured"}

        cache_key = self._generate_cache_key(indicator, IntelSource.VIRUSTOTAL)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug("virustotal_cache_hit", indicator=indicator)
            return cached_data

        async with self.semaphore:
            try:
                async with vt.Client(self.vt_api_key) as client:
                    if indicator_type == "ip_address":
                        obj = await client.get_object_async(f"/ip_addresses/{indicator}")
                    elif indicator_type == "domain":
                        obj = await client.get_object_async(f"/domains/{indicator}")
                    elif indicator_type == "file":
                        obj = await client.get_object_async(f"/files/{indicator}")
                    else:
                        return {"error": f"Unsupported indicator type: {indicator_type}"}

                    result = {
                        "source": IntelSource.VIRUSTOTAL,
                        "indicator": indicator,
                        "indicator_type": indicator_type,
                        "last_analysis_stats": obj.last_analysis_stats,
                        "reputation": getattr(obj, "reputation", None),
                        "last_analysis_date": getattr(obj, "last_analysis_date", None),
                        "categories": getattr(obj, "categories", {}),
                        "threat_level": self._calculate_vt_threat_level(
                            obj.last_analysis_stats
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }

                    self._set_to_cache(cache_key, result)
                    logger.info(
                        "virustotal_query_success",
                        indicator=indicator,
                        threat_level=result["threat_level"],
                    )
                    return result

            except vt.APIError as e:
                logger.error("virustotal_api_error", indicator=indicator, error=str(e))
                return {"error": str(e), "source": IntelSource.VIRUSTOTAL}
            except Exception as e:
                logger.error("virustotal_query_failed", indicator=indicator, error=str(e))
                return {"error": str(e), "source": IntelSource.VIRUSTOTAL}

    def _calculate_vt_threat_level(self, stats: dict[str, int]) -> str:
        """根據 VirusTotal 統計計算威脅等級"""
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        total = sum(stats.values())

        if total == 0:
            return ThreatLevel.UNKNOWN

        malicious_ratio = (malicious + suspicious) / total

        if malicious_ratio >= 0.5:
            return ThreatLevel.CRITICAL
        elif malicious_ratio >= 0.3:
            return ThreatLevel.HIGH
        elif malicious_ratio >= 0.1:
            return ThreatLevel.MEDIUM
        elif malicious_ratio > 0:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def query_abuseipdb(self, ip_address: str, max_age_days: int = 90) -> dict[str, Any]:
        """
        查詢 AbuseIPDB

        Args:
            ip_address: IP 地址
            max_age_days: 查詢最近幾天的記錄

        Returns:
            AbuseIPDB 查詢結果
        """
        if not self.abuseipdb_api_key:
            return {"error": "AbuseIPDB API key not configured"}

        cache_key = self._generate_cache_key(ip_address, IntelSource.ABUSEIPDB)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug("abuseipdb_cache_hit", ip_address=ip_address)
            return cached_data

        async with self.semaphore:
            try:
                url = "https://api.abuseipdb.com/api/v2/check"
                headers = {
                    "Key": self.abuseipdb_api_key,
                    "Accept": "application/json",
                }
                params = {
                    "ipAddress": ip_address,
                    "maxAgeInDays": max_age_days,
                    "verbose": True,
                }

                if not self.http_client:
                    self.http_client = httpx.AsyncClient(timeout=30.0)

                response = await self.http_client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                if "data" in data:
                    abuse_data = data["data"]
                    result = {
                        "source": IntelSource.ABUSEIPDB,
                        "indicator": ip_address,
                        "indicator_type": "ip_address",
                        "abuse_confidence_score": abuse_data.get("abuseConfidenceScore", 0),
                        "total_reports": abuse_data.get("totalReports", 0),
                        "num_distinct_users": abuse_data.get("numDistinctUsers", 0),
                        "is_whitelisted": abuse_data.get("isWhitelisted", False),
                        "country_code": abuse_data.get("countryCode"),
                        "usage_type": abuse_data.get("usageType"),
                        "threat_level": self._calculate_abuseipdb_threat_level(
                            abuse_data.get("abuseConfidenceScore", 0)
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }

                    self._set_to_cache(cache_key, result)
                    logger.info(
                        "abuseipdb_query_success",
                        ip_address=ip_address,
                        threat_level=result["threat_level"],
                        confidence=result["abuse_confidence_score"],
                    )
                    return result

                return {"error": "No data returned", "source": IntelSource.ABUSEIPDB}

            except httpx.HTTPStatusError as e:
                logger.error("abuseipdb_http_error", ip_address=ip_address, status=e.response.status_code)
                return {"error": str(e), "source": IntelSource.ABUSEIPDB}
            except Exception as e:
                logger.error("abuseipdb_query_failed", ip_address=ip_address, error=str(e))
                return {"error": str(e), "source": IntelSource.ABUSEIPDB}

    def _calculate_abuseipdb_threat_level(self, confidence_score: int) -> str:
        """根據 AbuseIPDB 信心分數計算威脅等級"""
        if confidence_score >= 90:
            return ThreatLevel.CRITICAL
        elif confidence_score >= 75:
            return ThreatLevel.HIGH
        elif confidence_score >= 50:
            return ThreatLevel.MEDIUM
        elif confidence_score >= 25:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO

    async def aggregate_intelligence(
        self,
        indicator: str,
        indicator_type: str,
        sources: list[IntelSource] | None = None,
    ) -> dict[str, Any]:
        """
        聚合多個情報源的結果

        Args:
            indicator: 指標
            indicator_type: 指標類型
            sources: 要查詢的情報源列表，None 則查詢所有可用源

        Returns:
            聚合後的情報結果
        """
        if sources is None:
            sources = [IntelSource(s) for s in self._get_available_sources()]

        tasks = []
        source_names = []

        for source in sources:
            if source == IntelSource.VIRUSTOTAL and self.vt_api_key:
                tasks.append(self.query_virustotal(indicator, indicator_type))
                source_names.append(source)
            elif source == IntelSource.ABUSEIPDB and self.abuseipdb_api_key and indicator_type == "ip_address":
                tasks.append(self.query_abuseipdb(indicator))
                source_names.append(source)

        if not tasks:
            return {
                "indicator": indicator,
                "indicator_type": indicator_type,
                "error": "No intelligence sources available",
                "timestamp": datetime.now().isoformat(),
            }

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 處理結果
        aggregated = {
            "indicator": indicator,
            "indicator_type": indicator_type,
            "sources": {},
            "overall_threat_level": ThreatLevel.UNKNOWN,
            "timestamp": datetime.now().isoformat(),
        }

        threat_levels = []
        for source_name, result in zip(source_names, results, strict=True):
            if isinstance(result, Exception):
                aggregated["sources"][source_name] = {"error": str(result)}
            elif isinstance(result, dict):
                aggregated["sources"][source_name] = result
                if "threat_level" in result:
                    threat_levels.append(result["threat_level"])

        # 計算整體威脅等級（取最高）
        if threat_levels:
            level_priority = {
                ThreatLevel.CRITICAL: 5,
                ThreatLevel.HIGH: 4,
                ThreatLevel.MEDIUM: 3,
                ThreatLevel.LOW: 2,
                ThreatLevel.INFO: 1,
                ThreatLevel.UNKNOWN: 0,
            }
            highest_level = max(threat_levels, key=lambda x: level_priority.get(x, 0))
            aggregated["overall_threat_level"] = highest_level

        logger.info(
            "intelligence_aggregated",
            indicator=indicator,
            sources_count=len(source_names),
            threat_level=aggregated["overall_threat_level"],
        )

        return aggregated

    async def batch_query(
        self,
        indicators: list[tuple[str, str]],
        sources: list[IntelSource] | None = None,
    ) -> list[dict[str, Any]]:
        """
        批量查詢多個指標

        Args:
            indicators: 指標列表，格式為 [(indicator, indicator_type), ...]
            sources: 要查詢的情報源

        Returns:
            批量查詢結果列表
        """
        tasks = [
            self.aggregate_intelligence(indicator, indicator_type, sources)
            for indicator, indicator_type in indicators
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        logger.info("batch_query_completed", total=len(indicators), success=len(processed_results))
        return processed_results

    def clear_cache(self) -> None:
        """清除所有快取"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("cache_cleared")


async def main():
    """測試範例"""
    aggregator = IntelAggregator()

    async with aggregator:
        # 單一查詢
        result = await aggregator.aggregate_intelligence("8.8.8.8", "ip_address")
        print("Single Query Result:")
        print(result)

        # 批量查詢
        indicators = [
            ("8.8.8.8", "ip_address"),
            ("google.com", "domain"),
        ]
        batch_results = await aggregator.batch_query(indicators)
        print("\nBatch Query Results:")
        for r in batch_results:
            print(r)


if __name__ == "__main__":
    asyncio.run(main())
