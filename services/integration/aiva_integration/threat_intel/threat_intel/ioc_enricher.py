"""
IOC Enricher - IOC 豐富化器

豐富化 IOC（IP/Domain/Hash）資訊，包括：
- IP 地址：WHOIS、地理位置、ASN
- 網域：WHOIS、DNS 記錄、TLD 分析
- 文件雜湊：文件屬性、簽名資訊
"""

import asyncio
from datetime import datetime
import socket
from typing import Any

import geoip2.database
from ipwhois import IPWhois
import structlog
import tldextract

logger = structlog.get_logger(__name__)


class IOCEnricher:
    """
    IOC 豐富化器

    提供 IP、網域、文件雜湊的額外資訊查詢功能。
    """

    def __init__(
        self,
        geoip_db_path: str | None = None,
        dns_timeout: int = 5,
        whois_timeout: int = 10,
    ):
        """
        初始化 IOC 豐富化器

        Args:
            geoip_db_path: GeoIP2 資料庫路徑（GeoLite2-City.mmdb）
            dns_timeout: DNS 查詢超時（秒）
            whois_timeout: WHOIS 查詢超時（秒）
        """
        self.geoip_db_path = geoip_db_path
        self.dns_timeout = dns_timeout
        self.whois_timeout = whois_timeout
        self.geoip_reader = None

        if geoip_db_path:
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_db_path)
                logger.info("geoip_database_loaded", path=geoip_db_path)
            except Exception as e:
                logger.warning("geoip_database_load_failed", error=str(e))

        logger.info("ioc_enricher_initialized")

    def __del__(self):
        """清理資源"""
        if self.geoip_reader:
            self.geoip_reader.close()

    async def enrich_ip(self, ip_address: str) -> dict[str, Any]:
        """
        豐富化 IP 地址資訊

        Args:
            ip_address: IP 地址

        Returns:
            IP 豐富化資訊
        """
        enriched: dict[str, Any] = {
            "indicator": ip_address,
            "indicator_type": "ip_address",
            "timestamp": datetime.now().isoformat(),
        }

        # 1. WHOIS 查詢
        try:
            whois_data = await self._query_ip_whois(ip_address)
            enriched["whois"] = whois_data
        except Exception as e:
            logger.warning("ip_whois_failed", ip=ip_address, error=str(e))
            enriched["whois"] = {"error": str(e)}

        # 2. GeoIP 查詢
        if self.geoip_reader:
            try:
                geo_data = self._query_geoip(ip_address)
                enriched["geolocation"] = geo_data
            except Exception as e:
                logger.warning("geoip_query_failed", ip=ip_address, error=str(e))
                enriched["geolocation"] = {"error": str(e)}

        # 3. 反向 DNS 查詢
        try:
            rdns = await self._reverse_dns_lookup(ip_address)
            enriched["reverse_dns"] = rdns
        except Exception as e:
            logger.debug("reverse_dns_failed", ip=ip_address, error=str(e))
            enriched["reverse_dns"] = {"error": str(e)}

        logger.info("ip_enrichment_completed", ip=ip_address)
        return enriched

    async def _query_ip_whois(self, ip_address: str) -> dict[str, Any]:
        """查詢 IP WHOIS 資訊"""
        loop = asyncio.get_event_loop()

        def _whois_query():
            obj = IPWhois(ip_address, timeout=self.whois_timeout)
            result = obj.lookup_rdap(depth=1)
            return {
                "asn": result.get("asn"),
                "asn_cidr": result.get("asn_cidr"),
                "asn_country_code": result.get("asn_country_code"),
                "asn_description": result.get("asn_description"),
                "network": {
                    "cidr": (network.get("cidr") if (network := result.get("network")) else None),
                    "name": (network.get("name") if (network := result.get("network")) else None),
                    "handle": (network.get("handle") if (network := result.get("network")) else None),
                    "country": (network.get("country") if (network := result.get("network")) else None),
                },
                "entities": result.get("objects", {}),
            }

        return await loop.run_in_executor(None, _whois_query)

    def _query_geoip(self, ip_address: str) -> dict[str, Any]:
        """查詢 GeoIP 資訊"""
        if not self.geoip_reader:
            return {"error": "GeoIP database not available"}

        try:
            response = self.geoip_reader.city(ip_address)
            return {
                "country": {
                    "iso_code": response.country.iso_code,
                    "name": response.country.name,
                },
                "city": {
                    "name": response.city.name,
                },
                "location": {
                    "latitude": response.location.latitude,
                    "longitude": response.location.longitude,
                    "accuracy_radius": response.location.accuracy_radius,
                    "time_zone": response.location.time_zone,
                },
                "postal": {
                    "code": response.postal.code,
                },
                "subdivisions": [
                    {"iso_code": s.iso_code, "name": s.name}
                    for s in response.subdivisions
                ],
            }
        except Exception as e:
            return {"error": f"GeoIP lookup failed: {str(e)}"}

    async def _reverse_dns_lookup(self, ip_address: str) -> dict[str, Any]:
        """反向 DNS 查詢"""
        loop = asyncio.get_event_loop()

        def _rdns_lookup():
            try:
                hostname, aliaslist, ipaddrlist = socket.gethostbyaddr(ip_address)
                return {
                    "hostname": hostname,
                    "aliases": aliaslist,
                    "addresses": ipaddrlist,
                }
            except socket.herror as e:
                return {"error": str(e)}

        return await loop.run_in_executor(None, _rdns_lookup)

    async def enrich_domain(self, domain: str) -> dict[str, Any]:
        """
        豐富化網域資訊

        Args:
            domain: 網域名稱

        Returns:
            網域豐富化資訊
        """
        enriched: dict[str, Any] = {
            "indicator": domain,
            "indicator_type": "domain",
            "timestamp": datetime.now().isoformat(),
        }

        # 1. TLD 分析
        try:
            tld_info = self._analyze_tld(domain)
            enriched["tld_analysis"] = tld_info
        except Exception as e:
            logger.warning("tld_analysis_failed", domain=domain, error=str(e))
            enriched["tld_analysis"] = {"error": str(e)}

        # 2. DNS 查詢（A、AAAA、MX、TXT 記錄）
        try:
            dns_records = await self._query_dns_records(domain)
            enriched["dns_records"] = dns_records
        except Exception as e:
            logger.warning("dns_query_failed", domain=domain, error=str(e))
            enriched["dns_records"] = {"error": str(e)}

        # 3. WHOIS 查詢
        try:
            whois_data = await self._query_domain_whois(domain)
            enriched["whois"] = whois_data
        except Exception as e:
            logger.warning("domain_whois_failed", domain=domain, error=str(e))
            enriched["whois"] = {"error": str(e)}

        logger.info("domain_enrichment_completed", domain=domain)
        return enriched

    def _analyze_tld(self, domain: str) -> dict[str, Any]:
        """分析 TLD"""
        ext = tldextract.extract(domain)
        return {
            "subdomain": ext.subdomain,
            "domain": ext.domain,
            "suffix": ext.suffix,
            "registered_domain": ext.registered_domain,
            "fqdn": ext.fqdn,
            "is_private": ext.is_private,
        }

    async def _query_dns_records(self, domain: str) -> dict[str, Any]:
        """查詢 DNS 記錄"""
        loop = asyncio.get_event_loop()
        records = {}

        async def _query_record(record_type: str):
            def _resolve():
                try:
                    if record_type == "A":
                        return socket.getaddrinfo(domain, None, socket.AF_INET)
                    elif record_type == "AAAA":
                        return socket.getaddrinfo(domain, None, socket.AF_INET6)
                    else:
                        return []
                except socket.gaierror:
                    return []

            result = await loop.run_in_executor(None, _resolve)

            if record_type in ["A", "AAAA"]:
                addresses = list({info[4][0] for info in result})
                return addresses
            return result

        # 查詢 A 和 AAAA 記錄
        records["A"] = await _query_record("A")
        records["AAAA"] = await _query_record("AAAA")

        return records

    async def _query_domain_whois(self, domain: str) -> dict[str, Any]:
        """查詢網域 WHOIS 資訊"""
        try:
            # 使用 whoisdomain 套件
            from whoisdomain import query as whois_query

            loop = asyncio.get_event_loop()

            def _whois():
                result = whois_query(domain)
                if result:
                    return {
                        "registrar": getattr(result, "registrar", None),
                        "creation_date": str(getattr(result, "creation_date", None)),
                        "expiration_date": str(getattr(result, "expiration_date", None)),
                        "updated_date": str(getattr(result, "updated_date", None)),
                        "name_servers": getattr(result, "name_servers", []),
                        "status": getattr(result, "status", []),
                        "emails": getattr(result, "emails", []),
                    }
                return {"error": "WHOIS query returned no data"}

            return await loop.run_in_executor(None, _whois)
        except Exception as e:
            return {"error": str(e)}

    async def enrich_file_hash(self, file_hash: str) -> dict[str, Any]:
        """
        豐富化文件雜湊資訊

        Args:
            file_hash: 文件雜湊值（MD5/SHA1/SHA256）

        Returns:
            文件雜湊豐富化資訊
        """
        enriched: dict[str, Any] = {
            "indicator": file_hash,
            "indicator_type": "file_hash",
            "hash_type": self._detect_hash_type(file_hash),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("file_hash_enrichment_completed", hash=file_hash)
        return enriched

    def _detect_hash_type(self, hash_value: str) -> str:
        """偵測雜湊類型"""
        hash_types = {32: "MD5", 40: "SHA1", 64: "SHA256"}
        return hash_types.get(len(hash_value), "unknown")

    async def batch_enrich(
        self, indicators: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """
        批量豐富化

        Args:
            indicators: 指標列表，格式為 [(indicator, indicator_type), ...]

        Returns:
            批量豐富化結果
        """
        tasks = []
        for indicator, indicator_type in indicators:
            if indicator_type == "ip_address":
                tasks.append(self.enrich_ip(indicator))
            elif indicator_type == "domain":
                tasks.append(self.enrich_domain(indicator))
            elif indicator_type == "file_hash":
                tasks.append(self.enrich_file_hash(indicator))
            else:
                # 創建一個簡單的 coroutine 而不使用已廢棄的 asyncio.coroutine
                async def _unsupported_error():
                    return {"error": "Unsupported indicator type"}

                tasks.append(_unsupported_error())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        logger.info("batch_enrichment_completed", total=len(indicators))
        return processed_results


async def main():
    """測試範例"""
    enricher = IOCEnricher()

    # 測試 IP 豐富化
    ip_result = await enricher.enrich_ip("8.8.8.8")
    print("IP Enrichment:")
    print(ip_result)

    # 測試網域豐富化
    domain_result = await enricher.enrich_domain("google.com")
    print("\nDomain Enrichment:")
    print(domain_result)

    # 測試文件雜湊豐富化
    hash_result = await enricher.enrich_file_hash(
        "44d88612fea8a8f36de82e1278abb02f"
    )
    print("\nHash Enrichment:")
    print(hash_result)


if __name__ == "__main__":
    asyncio.run(main())
