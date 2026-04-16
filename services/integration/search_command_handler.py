"""
網路搜索命令處理器

處理各種網路搜索相關的命令，支持兩種調用方式：
1. **CLI 模式**（推薦）：通過命令行直接調用
2. **AICommand 模式**（向後兼容）：通過 AICommand 接口調用

支持的搜索類型：
- 漏洞數據庫搜索（ExploitDB, CVE, CWE）
- 威脅情報搜索（VirusTotal, AbuseIPDB, AlienVault OTX）
- 開源情報搜索（Google, DuckDuckGo, GitHub, Shodan）
- 社交工程搜索（WHOIS, 域名信息）

CLI 使用示例：
    python -m services.integration.search_command_handler \\
        --search-type google \\
        --query "CVE-2024-1234" \\
        --max-results 10

架構更新（2026-01-08）：
- ✅ 優先使用 CLI 模式
- ✅ 保留 AICommand 接口作為向後兼容層
"""

import asyncio
from datetime import UTC, datetime
import os
from typing import Any

import aiohttp
from aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandCallback,
    CommandContext,
    CommandStatus,
    CommandType,
)
from aiva_common.utils import get_logger

logger = get_logger(__name__)


class SearchCommandHandler:
    """網路搜索命令處理器
    
    支持的搜索類型：
    1. 漏洞數據庫：ExploitDB, CVE Details, CWE, CAPEC
    2. 威脅情報：VirusTotal, AbuseIPDB, AlienVault OTX
    3. OSINT：Google, DuckDuckGo, GitHub, Shodan, Censys
    4. 社交工程：Email breach, WHOIS, Domain info
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """初始化搜索命令處理器
        
        Args:
            config: 配置字典，包含各個 API 的密鑰
        """
        self.config = config or {}
        self.logger = logger
        
        # API 配置
        self.api_keys = {
            "google": os.getenv("GOOGLE_API_KEY") or self.config.get("google_api_key"),
            "google_cx": os.getenv("GOOGLE_SEARCH_ENGINE_ID") or self.config.get("google_search_engine_id"),
            "github": os.getenv("GITHUB_TOKEN") or self.config.get("github_token"),
            "shodan": os.getenv("SHODAN_API_KEY") or self.config.get("shodan_api_key"),
            "nvd": os.getenv("NVD_API_KEY") or self.config.get("nvd_api_key"),
            "virustotal": os.getenv("VIRUSTOTAL_API_KEY") or self.config.get("virustotal_api_key"),
            "abuseipdb": os.getenv("ABUSEIPDB_API_KEY") or self.config.get("abuseipdb_api_key"),
        }
        
        # 搜索引擎映射
        self.search_handlers = {
            CommandType.SEARCH_GOOGLE: self._search_google,
            CommandType.SEARCH_DUCKDUCKGO: self._search_duckduckgo,
            CommandType.SEARCH_GITHUB: self._search_github,
            CommandType.SEARCH_EXPLOIT_DB: self._search_exploitdb,
            CommandType.SEARCH_CVE_DETAILS: self._search_cve_details,
            CommandType.SEARCH_SHODAN: self._search_shodan,
            CommandType.SEARCH_THREAT_INTEL: self._search_threat_intel,
            CommandType.SEARCH_WHOIS: self._search_whois,
            CommandType.SEARCH_DOMAIN_INFO: self._search_domain_info,
        }
        
        # 回調處理器
        self.callback_handler: CommandCallback | None = None
        
        self.logger.info("✅ SearchCommandHandler 已初始化")
    
    def set_callback(self, callback: CommandCallback) -> None:
        """設置回調處理器
        
        Args:
            callback: 回調處理器實例
        """
        self.callback_handler = callback
        self.logger.debug("✅ 已設置回調處理器")
    
    async def handle_command(
        self,
        command: AICommand,
        context: CommandContext | None = None
    ) -> AICommandResult:
        """處理搜索命令
        
        Args:
            command: AI 命令
            context: 執行上下文
            
        Returns:
            命令執行結果
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"🔍 開始搜索: {command.command_type.value}")
        
        try:
            # 通知開始
            if self.callback_handler:
                await self.callback_handler.on_progress(
                    command.command_id,
                    0.1,
                    f"開始 {command.command_type.value} 搜索..."
                )
            
            # 路由到對應的搜索處理器
            handler = self.search_handlers.get(command.command_type)
            
            if not handler:
                raise ValueError(f"不支持的搜索類型: {command.command_type.value}")
            
            # 執行搜索（處理同步和異步函數）
            import inspect
            if inspect.iscoroutinefunction(handler):
                results = await handler(command.payload)
            else:
                results = handler(command.payload)
            
            # 通知完成
            if self.callback_handler:
                await self.callback_handler.on_progress(
                    command.command_id,
                    1.0,
                    "搜索完成"
                )
            
            execution_time = time.time() - start_time
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result=results,
                error="",
                error_code="",
                error_details={},
                execution_time=execution_time,
                started_at=datetime.fromtimestamp(start_time, tz=UTC),
                completed_at=datetime.now(UTC),
                metrics={
                    "search_type": command.command_type.value,
                    "results_count": self._count_results(results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ 搜索失敗: {e}", exc_info=True)
            
            execution_time = time.time() - start_time
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},
                error=str(e),
                error_code="SEARCH_EXECUTION_ERROR",
                error_details={"exception_type": type(e).__name__},
                execution_time=execution_time,
                started_at=datetime.fromtimestamp(start_time, tz=UTC),
                completed_at=datetime.now(UTC)
            )
    
    # ===== 搜索引擎實現 =====
    
    async def _search_google(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Google Custom Search API 搜索
        
        Args:
            payload: {
                "query": str,
                "max_results": int (default: 10),
                "language": str (default: "zh-TW")
            }
        """
        query = payload.get("query", "")
        max_results = min(payload.get("max_results", 10), 100)
        language = payload.get("language", "zh-TW")
        
        if not self.api_keys["google"] or not self.api_keys["google_cx"]:
            self.logger.warning("⚠️  Google API Key 或 Search Engine ID 未配置，使用模擬數據")
            return {
                "query": query,
                "source": "google",
                "warning": "API Key 未配置，返回模擬數據",
                "items": []
            }
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_keys["google"],
            "cx": self.api_keys["google_cx"],
            "q": query,
            "num": min(max_results, 10),  # Google API 單次最多 10 個
            "lr": f"lang_{language.split('-')[0]}"
        }
        
        headers = {"User-Agent": "AIVA/6.3"}
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=timeout) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"Google API 返回錯誤: {response.status}")
                    
                    data = await response.json()
                    
                    # 通知中間結果
                    if self.callback_handler:
                        await self.callback_handler.on_partial_result(
                            payload.get("command_id", "unknown"),
                            "search_results",
                            {
                                "source": "google",
                                "count": len(data.get("items", []))
                            }
                        )
                    
                    return {
                        "query": query,
                        "source": "google",
                        "total_results": int(data.get("searchInformation", {}).get("totalResults", 0)),
                        "search_time": float(data.get("searchInformation", {}).get("searchTime", 0)),
                        "items": [
                            {
                                "title": item.get("title"),
                                "link": item.get("link"),
                                "snippet": item.get("snippet"),
                                "displayLink": item.get("displayLink")
                            }
                            for item in data.get("items", [])
                        ]
                    }
        
        except (TimeoutError, aiohttp.ClientError) as e:
            self.logger.error(f"Google 搜索失敗: {e}")
            raise
    
    async def _search_duckduckgo(self, payload: dict[str, Any]) -> dict[str, Any]:
        """DuckDuckGo Instant Answer API 搜索
        
        Args:
            payload: {
                "query": str,
                "max_results": int (default: 10)
            }
        """
        query = payload.get("query", "")
        max_results = payload.get("max_results", 10)
        
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    # DuckDuckGo API 有時返回 202（接受請求），需要特殊處理
                    if response.status == 202:
                        self.logger.warning("⚠️  DuckDuckGo API 返回 202，重試中...")
                        await asyncio.sleep(1)
                        
                        # 重試一次
                        async with session.get(url, params=params, timeout=timeout) as retry_response:
                            if retry_response.status != 200:
                                # 如果仍然失敗，返回空結果而不是拋出異常
                                self.logger.warning(f"⚠️  DuckDuckGo API 重試後仍返回 {retry_response.status}，返回空結果")
                                return {
                                    "query": query,
                                    "source": "duckduckgo",
                                    "warning": f"API 返回狀態碼 {retry_response.status}，可能暫時不可用",
                                    "abstract": "",
                                    "related_topics": []
                                }
                            
                            data = await retry_response.json()
                    
                    elif response.status != 200:
                        self.logger.warning(f"⚠️  DuckDuckGo API 返回 {response.status}，返回空結果")
                        return {
                            "query": query,
                            "source": "duckduckgo",
                            "warning": f"API 返回狀態碼 {response.status}",
                            "abstract": "",
                            "related_topics": []
                        }
                    else:
                        data = await response.json()
                    
                    # 提取相關主題
                    related_topics = []
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if "Text" in topic:
                            related_topics.append({
                                "text": topic.get("Text", ""),
                                "url": topic.get("FirstURL", "")
                            })
                    
                    return {
                        "query": query,
                        "source": "duckduckgo",
                        "abstract": data.get("Abstract", ""),
                        "abstract_text": data.get("AbstractText", ""),
                        "abstract_url": data.get("AbstractURL", ""),
                        "related_topics": related_topics
                    }
        
        except TimeoutError:
            self.logger.error("DuckDuckGo 搜索超時")
            return {
                "query": query,
                "source": "duckduckgo",
                "error": "搜索超時",
                "abstract": "",
                "related_topics": []
            }
        except Exception as e:
            self.logger.error(f"DuckDuckGo 搜索失敗: {e}")
            return {
                "query": query,
                "source": "duckduckgo",
                "error": str(e),
                "abstract": "",
                "related_topics": []
            }
    
    async def _search_github(self, payload: dict[str, Any]) -> dict[str, Any]:
        """GitHub Search API 搜索
        
        Args:
            payload: {
                "query": str,
                "search_type": str (repositories/code/issues, default: repositories),
                "max_results": int (default: 30)
            }
        """
        query = payload.get("query", "")
        search_type = payload.get("search_type", "repositories")
        max_results = min(payload.get("max_results", 30), 100)
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.api_keys["github"]:
            headers["Authorization"] = f"token {self.api_keys['github']}"
        
        url = f"https://api.github.com/search/{search_type}"
        params = {
            "q": query,
            "per_page": max_results,
            "sort": "stars" if search_type == "repositories" else "indexed"
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"GitHub API 返回錯誤: {response.status}")
                    
                    data = await response.json()
                    
                    return {
                        "query": query,
                        "source": "github",
                        "search_type": search_type,
                        "total_count": data.get("total_count", 0),
                        "items": [
                            {
                                "name": item.get("name", item.get("title", "")),
                                "full_name": item.get("full_name", ""),
                                "url": item.get("html_url", ""),
                                "description": item.get("description", ""),
                                "stars": item.get("stargazers_count", 0),
                                "language": item.get("language", ""),
                                "updated_at": item.get("updated_at", "")
                            }
                            for item in data.get("items", [])
                        ]
                    }
        
        except (TimeoutError, aiohttp.ClientError) as e:
            self.logger.error(f"GitHub 搜索失敗: {e}")
            raise
    
    def _search_exploitdb(self, payload: dict[str, Any]) -> dict[str, Any]:
        """ExploitDB 搜索（使用公開 API 或爬蟲）
        
        Args:
            payload: {
                "query": str,
                "platform": str (optional)
            }
        """
        query = payload.get("query", "")
        platform = payload.get("platform", "")
        
        # ExploitDB 沒有官方公開 API，這裡返回搜索 URL 和模擬數據
        search_url = f"https://www.exploit-db.com/search?q={query}"
        if platform:
            search_url += f"&platform={platform}"
        
        self.logger.warning("⚠️  ExploitDB 沒有公開 API，返回搜索 URL")
        
        return {
            "query": query,
            "source": "exploitdb",
            "platform": platform,
            "search_url": search_url,
            "note": "ExploitDB 沒有公開 API，請訪問搜索 URL 進行手動查詢",
            "exploits": []
        }
    
    async def _search_cve_details(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """CVE Details 搜索（使用 NVD API）
        
        Args:
            payload: {
                "cve_id": str (optional),
                "keyword": str (optional),
                "max_results": int (default: 20)
            }
        """
        cve_id = payload.get("cve_id", "")
        keyword = payload.get("keyword", "")
        max_results = min(payload.get("max_results", 20), 100)
        
        # 使用 NVD API
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        
        headers = {}
        if self.api_keys["nvd"]:
            headers["apiKey"] = self.api_keys["nvd"]
        
        params = {
            "resultsPerPage": max_results
        }
        
        if cve_id:
            params["cveId"] = cve_id
        elif keyword:
            params["keywordSearch"] = keyword
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"NVD API 返回錯誤: {response.status}")
                    
                    data = await response.json()
                    
                    vulnerabilities = []
                    for vuln in data.get("vulnerabilities", []):
                        cve = vuln.get("cve", {})
                        vulnerabilities.append({
                            "id": cve.get("id", ""),
                            "published": cve.get("published", ""),
                            "modified": cve.get("lastModified", ""),
                            "description": cve.get("descriptions", [{}])[0].get("value", ""),
                            "severity": cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseSeverity", "Unknown"),
                            "score": cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseScore", 0.0)
                        })
                    
                    return {
                        "query": cve_id or keyword,
                        "source": "nvd",
                        "total_results": data.get("totalResults", 0),
                        "vulnerabilities": vulnerabilities
                    }
        
        except (TimeoutError, aiohttp.ClientError) as e:
            self.logger.error(f"CVE 搜索失敗: {e}")
            raise
    
    async def _search_shodan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Shodan 搜索
        
        Args:
            payload: {
                "query": str,
                "max_results": int (default: 100)
            }
        """
        query = payload.get("query", "")
        max_results = payload.get("max_results", 100)
        
        if not self.api_keys["shodan"]:
            self.logger.warning("⚠️  Shodan API Key 未配置")
            return {
                "query": query,
                "source": "shodan",
                "error": "API Key 未配置",
                "matches": []
            }
        
        url = "https://api.shodan.io/shodan/host/search"
        params = {
            "key": self.api_keys["shodan"],
            "query": query
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"Shodan API 返回錯誤: {response.status}")
                    
                    data = await response.json()
                    
                    return {
                        "query": query,
                        "source": "shodan",
                        "total": data.get("total", 0),
                        "matches": [
                            {
                                "ip": match.get("ip_str"),
                                "port": match.get("port"),
                                "org": match.get("org"),
                                "location": match.get("location", {}).get("country_name", ""),
                                "data": match.get("data", "")[:200]
                            }
                            for match in data.get("matches", [])[:max_results]
                        ]
                    }
        
        except (TimeoutError, aiohttp.ClientError) as e:
            self.logger.error(f"Shodan 搜索失敗: {e}")
            raise
    
    def _search_threat_intel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """威脅情報搜索（整合多個來源）
        
        Args:
            payload: {
                "indicator": str,  # IP, domain, hash
                "type": str  # ip, domain, hash
            }
        """
        indicator = payload.get("indicator", "")
        indicator_type = payload.get("type", "ip")
        
        # 查詢多個威脅情報源
        results = {}
        
        if self.api_keys["virustotal"]:
            try:
                results["virustotal"] = self._query_virustotal(indicator)
            except Exception as e:
                results["virustotal"] = {"error": str(e)}
        
        if self.api_keys["abuseipdb"] and indicator_type == "ip":
            try:
                results["abuseipdb"] = self._query_abuseipdb(indicator)
            except Exception as e:
                results["abuseipdb"] = {"error": str(e)}
        
        return {
            "indicator": indicator,
            "type": indicator_type,
            "source": "threat_intel",
            "sources": results
        }
    
    def _query_virustotal(self, indicator: str) -> Dict:
        """查詢 VirusTotal"""
        # 實現 VirusTotal API 查詢
        return {"status": "not_implemented", "indicator": indicator}
    
    def _query_abuseipdb(self, ip: str) -> Dict:
        """查詢 AbuseIPDB"""
        # 實現 AbuseIPDB API 查詢
        return {"status": "not_implemented", "ip": ip}
    
    def _search_whois(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """WHOIS 查詢
        
        Args:
            payload: {
                "domain": str
            }
        """
        domain = payload.get("domain", "")
        
        # 使用 WHOIS API 或命令行工具
        self.logger.info(f"執行 WHOIS 查詢: {domain}")
        
        return {
            "domain": domain,
            "source": "whois",
            "note": "WHOIS 查詢需要額外實現",
            "data": {}
        }
    
    def _search_domain_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """域名信息查詢
        
        Args:
            payload: {
                "domain": str
            }
        """
        domain = payload.get("domain", "")
        
        self.logger.info(f"查詢域名信息: {domain}")
        
        return {
            "domain": domain,
            "source": "domain_info",
            "note": "域名信息查詢需要額外實現",
            "data": {}
        }
    
    # ===== 輔助方法 =====
    
    def _count_results(self, results: Dict[str, Any]) -> int:
        """計算搜索結果數量"""
        if "items" in results:
            return len(results["items"])
        elif "matches" in results:
            return len(results["matches"])
        elif "vulnerabilities" in results:
            return len(results["vulnerabilities"])
        elif "exploits" in results:
            return len(results["exploits"])
        else:
            return 0


# ==================== CLI 入口點 ====================

async def main_cli():
    """CLI 主函數
    
    提供命令行接口來執行搜索任務，符合 aiva_common 規範
    
    使用示例：
        # Google 搜索
        python -m services.integration.search_command_handler \\
            --search-type google --query "CVE-2024-1234"
        
        # GitHub 搜索
        python -m services.integration.search_command_handler \\
            --search-type github --query "xss vulnerability"
        
        # ExploitDB 搜索
        python -m services.integration.search_command_handler \\
            --search-type exploitdb --query "linux kernel"
    """
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="AIVA 搜索命令處理器 - CLI 模式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--search-type",
        required=True,
        choices=["google", "duckduckgo", "github", "exploitdb", "cve", "shodan", "threat_intel", "whois", "domain"],
        help="搜索類型"
    )
    
    parser.add_argument(
        "--query",
        required=True,
        help="搜索查詢字符串"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="最大結果數量（默認：10）"
    )
    
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="json",
        help="輸出格式（默認：json）"
    )
    
    parser.add_argument(
        "--config-file",
        help="配置文件路徑（JSON 格式）"
    )
    
    args = parser.parse_args()
    
    # 加載配置
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, encoding='utf-8') as f:
            config = json.load(f)
    
    # 創建處理器
    handler = SearchCommandHandler(config)
    
    # 構建 payload
    payload = {
        "query": args.query,
        "max_results": args.max_results
    }
    
    # 映射搜索類型到處理方法
    search_mapping = {
        "google": handler._search_google,
        "duckduckgo": handler._search_duckduckgo,
        "github": handler._search_github,
        "exploitdb": handler._search_exploitdb,
        "cve": handler._search_cve_details,
        "shodan": handler._search_shodan,
        "threat_intel": handler._search_threat_intel,
        "whois": handler._search_whois,
        "domain": handler._search_domain_info,
    }
    
    try:
        # 執行搜索
        logger.info(f"🔍 執行 {args.search_type} 搜索: {args.query}")
        
        search_func = search_mapping[args.search_type]
        results = await search_func(payload)
        
        # 輸出結果
        if args.output == "json":
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"\n搜索結果（{args.search_type}）:")
            print(f"查詢: {args.query}")
            print(f"結果數: {handler._count_results(results)}")
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ 搜索失敗: {e}", exc_info=True)
        print(json.dumps({
            "error": str(e),
            "search_type": args.search_type,
            "query": args.query
        }, indent=2), file=__import__('sys').stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main_cli()))
