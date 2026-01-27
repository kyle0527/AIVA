"""
function_web_scanner.scanners.subdomain_scanner
子域名掃描器

執行被動和主動子域名發現。
"""

import dns.resolver
import requests
from typing import List, Dict, Set
from dataclasses import dataclass

from aiva_common.utils import get_logger
from aiva_common.enums.common import Confidence

logger = get_logger(__name__)


@dataclass
class Subdomain:
    """子域名結果"""
    subdomain: str
    ip_addresses: List[str]
    source: str  # passive, bruteforce, dns_zone_transfer, etc.
    confidence: Confidence
    additional_info: Dict


class SubdomainScanner:
    """子域名掃描器"""
    
    def __init__(self, wordlist_path: str = None):
        """
        初始化子域名掃描器
        
        Args:
            wordlist_path: 子域名字典文件路徑
        """
        self.wordlist_path = wordlist_path or self._get_default_wordlist()
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = 2
        self.resolver.lifetime = 2
        logger.info(f"子域名掃描器初始化 (wordlist: {self.wordlist_path})")
    
    def scan(self, domain: str, mode: str = "all") -> List[Subdomain]:
        """
        掃描子域名
        
        Args:
            domain: 目標域名
            mode: 掃描模式 (passive, active, all)
        
        Returns:
            List[Subdomain]: 發現的子域名列表
        """
        subdomains = set()
        
        if mode in ["passive", "all"]:
            subdomains.update(self._passive_discovery(domain))
        
        if mode in ["active", "all"]:
            subdomains.update(self._bruteforce_discovery(domain))
            subdomains.update(self._dns_zone_transfer(domain))
        
        logger.info(f"發現 {len(subdomains)} 個子域名")
        return list(subdomains)
    
    def _passive_discovery(self, domain: str) -> Set[Subdomain]:
        """被動子域名發現"""
        subdomains = set()
        
        # 1. Certificate Transparency Logs
        subdomains.update(self._search_crtsh(domain))
        
        # 2. DNS Aggregators (VirusTotal, SecurityTrails, etc.)
        # Note: Requires API keys
        
        return subdomains
    
    def _search_crtsh(self, domain: str) -> Set[Subdomain]:
        """從 crt.sh 搜索證書透明度日誌"""
        subdomains = set()
        
        try:
            url = f"https://crt.sh/?q=%.{domain}&output=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                certs = response.json()
                
                for cert in certs:
                    name_value = cert.get("name_value", "")
                    for subdomain in name_value.split("\n"):
                        subdomain = subdomain.strip()
                        if subdomain and subdomain.endswith(domain):
                            ips = self._resolve_domain(subdomain)
                            if ips:
                                subdomains.add(Subdomain(
                                    subdomain=subdomain,
                                    ip_addresses=ips,
                                    source="crtsh",
                                    confidence=Confidence.HIGH,
                                    additional_info={"cert_id": cert.get("id")}
                                ))
        
        except Exception as e:
            logger.error(f"crt.sh 搜索錯誤: {e}")
        
        return subdomains
    
    def _bruteforce_discovery(self, domain: str) -> Set[Subdomain]:
        """暴力破解子域名"""
        subdomains = set()
        
        try:
            with open(self.wordlist_path, 'r') as f:
                wordlist = [line.strip() for line in f if line.strip()]
            
            for word in wordlist[:1000]:  # Limit for performance
                subdomain = f"{word}.{domain}"
                ips = self._resolve_domain(subdomain)
                
                if ips:
                    subdomains.add(Subdomain(
                        subdomain=subdomain,
                        ip_addresses=ips,
                        source="bruteforce",
                        confidence=Confidence.HIGH,
                        additional_info={}
                    ))
        
        except FileNotFoundError:
            logger.warning(f"Wordlist not found: {self.wordlist_path}")
        except Exception as e:
            logger.error(f"暴力破解錯誤: {e}")
        
        return subdomains
    
    def _dns_zone_transfer(self, domain: str) -> Set[Subdomain]:
        """嘗試 DNS 區域傳輸"""
        subdomains = set()
        
        try:
            # Get nameservers
            ns_records = dns.resolver.resolve(domain, 'NS')
            
            for ns in ns_records:
                nameserver = str(ns.target).rstrip('.')
                
                try:
                    # Attempt zone transfer
                    zone = dns.zone.from_xfr(dns.query.xfr(nameserver, domain))
                    
                    for name in zone.nodes.keys():
                        subdomain = f"{name}.{domain}"
                        ips = self._resolve_domain(subdomain)
                        
                        if ips:
                            subdomains.add(Subdomain(
                                subdomain=subdomain,
                                ip_addresses=ips,
                                source="zone_transfer",
                                confidence=Confidence.HIGH,
                                additional_info={"nameserver": nameserver}
                            ))
                
                except Exception as e:
                    logger.debug(f"Zone transfer failed for {nameserver}: {e}")
        
        except Exception as e:
            logger.debug(f"DNS zone transfer error: {e}")
        
        return subdomains
    
    def _resolve_domain(self, domain: str) -> List[str]:
        """解析域名獲取 IP 地址"""
        try:
            answers = self.resolver.resolve(domain, 'A')
            return [str(rdata) for rdata in answers]
        except:
            return []
    
    def _get_default_wordlist(self) -> str:
        """獲取默認字典路徑"""
        return "services/features/function_web_scanner/wordlists/subdomains.txt"
