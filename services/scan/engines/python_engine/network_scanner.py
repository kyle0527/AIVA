"""
AIVA 網路掃描器
負責執行端口掃描、服務發現和網路枚舉

整合 python-nmap 進行真實掃描 (P0-1 Full 修復)
遵循 aiva_common v2.0 規範
"""

import asyncio
import logging
import socket
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, UTC
import ipaddress

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False
    logging.warning("python-nmap not installed, using heuristic mode only")

# ✅ 使用 aiva_common 統一日誌
from aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


class NetworkScanner:
    """網路掃描器主類別 (v2.0 - 整合 python-nmap)
    
    支援兩種模式:
    1. Nmap 模式: 使用 python-nmap 進行真實掃描 (準確)
    2. Heuristic 模式: 使用 Python socket 進行基礎掃描 (降級)
    """
    
    def __init__(self):
        self.session_id = None
        self.scan_config = {}
        self.results = []
        self.common_ports = [
            21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995, 1723, 3306, 3389, 5432, 5900, 8080
        ]
        
        # ✅ 初始化 nmap 掃描器
        self.nm = None
        if NMAP_AVAILABLE:
            try:
                self.nm = nmap.PortScanner()
                logger.info("✅ NetworkScanner initialized with python-nmap")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize nmap scanner: {e}")
                self.nm = None
        else:
            logger.info("⚠️  NetworkScanner initialized in heuristic mode (nmap not available)")
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化網路掃描器"""
        try:
            self.scan_config = config or {}
            self.session_id = f"net_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"網路掃描器已初始化，會話ID: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"網路掃描器初始化失敗: {e}")
            return False
    
    async def scan_target(self, target: str, scan_type: str = "port_scan") -> Dict[str, Any]:
        """掃描指定目標"""
        try:
            results = {
                "target": target,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "scan_type": scan_type,
                "results": {},
                "summary": {}
            }
            
            logger.info(f"開始網路掃描: {target} ({scan_type})")
            
            if scan_type == "port_scan":
                results["results"] = await self._port_scan(target)
            elif scan_type == "service_discovery":
                results["results"] = await self._service_discovery(target)
            elif scan_type == "network_enum":
                results["results"] = await self._network_enumeration(target)
            else:
                results["results"] = await self._comprehensive_scan(target)
            
            # 生成掃描摘要
            results["summary"] = self._generate_summary(results["results"])
            
            self.results.append(results)
            logger.info(f"網路掃描完成: {target}")
            
            return results
            
        except Exception as e:
            logger.error(f"網路掃描失敗 {target}: {e}")
            return {"error": str(e), "target": target}
    
    async def _port_scan(self, target: str) -> Dict[str, Any]:
        """執行端口掃描 (v2.0 - 整合 python-nmap)"""
        results = {
            "open_ports": [],
            "closed_ports": [],
            "filtered_ports": [],
            "total_scanned": 0
        }
        
        # 解析目標
        host = self._extract_host(target)
        if not host:
            return {"error": "無效的目標地址"}
        
        logger.info(f"🔍 掃描端口: {host}")
        
        # ✅ 優先使用 nmap 進行真實掃描
        if self.nm is not None:
            return await self._nmap_port_scan(host)
        
        # 降級到基礎掃描
        logger.info("⚠️  Using heuristic port scan (nmap not available)")
        
        # 掃描常見端口
        for port in self.common_ports:
            try:
                status = await self._check_port(host, port)
                results["total_scanned"] += 1
                
                if status == "open":
                    results["open_ports"].append({
                        "port": port,
                        "protocol": "tcp",
                        "service": self._get_service_name(port),
                        "banner": await self._get_banner(host, port)
                    })
                    logger.debug(f"發現開放端口: {host}:{port}")
                elif status == "closed":
                    results["closed_ports"].append(port)
                else:
                    results["filtered_ports"].append(port)
                    
            except Exception as e:
                logger.debug(f"端口掃描錯誤 {host}:{port}: {e}")
                results["filtered_ports"].append(port)
        
        return results
    
    async def _service_discovery(self, target: str) -> Dict[str, Any]:
        """服務發現掃描"""
        results = {
            "discovered_services": [],
            "web_services": [],
            "database_services": [],
            "other_services": []
        }
        
        host = self._extract_host(target)
        logger.debug(f"服務發現目標: {host}")
        
        # 先進行端口掃描
        port_results = await self._port_scan(target)
        
        for port_info in port_results.get("open_ports", []):
            port = port_info["port"]
            service_info = {
                "port": port,
                "service": port_info["service"],
                "protocol": port_info["protocol"],
                "banner": port_info["banner"],
                "version": "unknown",
                "category": self._categorize_service(port)
            }
            
            # 版本檢測已整合到 nmap 掃描中，不需要單獨調用
            
            results["discovered_services"].append(service_info)
            
            # 按類別分組
            category = service_info["category"]
            if category == "web":
                results["web_services"].append(service_info)
            elif category == "database":
                results["database_services"].append(service_info)
            else:
                results["other_services"].append(service_info)
        
        return results
    
    async def _network_enumeration(self, target: str) -> Dict[str, Any]:
        """網路枚舉掃描"""
        results = {
            "host_info": {},
            "network_info": {},
            "reachability": {},
            "dns_info": {}
        }
        
        host = self._extract_host(target)
        
        # 主機資訊
        results["host_info"] = await self._get_host_info(host)
        
        # 網路資訊
        results["network_info"] = self._get_network_info(host)
        
        # 可達性測試
        results["reachability"] = await self._test_reachability(host)
        
        # DNS資訊
        results["dns_info"] = self._get_dns_info(host)
        
        return results
    
    async def _comprehensive_scan(self, target: str) -> Dict[str, Any]:
        """綜合掃描"""
        results = {
            "port_scan": await self._port_scan(target),
            "service_discovery": await self._service_discovery(target),
            "network_enum": await self._network_enumeration(target)
        }
        
        return results
    
    async def _check_port(self, host: str, port: int) -> str:
        """檢查端口狀態"""
        try:
            async with asyncio.timeout(1.0):
                _, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
            return "open"
        except ConnectionRefusedError:
            return "closed"
        except (TimeoutError, asyncio.TimeoutError):
            return "filtered"
        except OSError:
            return "closed"
        except Exception:
            return "unknown"
    
    async def _get_banner(self, host: str, port: int) -> str:
        """獲取服務banner"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=2.0)
            
            # 嘗試讀取banner
            banner_data = await asyncio.wait_for(reader.read(1024), timeout=1.0)
            banner = banner_data.decode('utf-8', errors='ignore').strip()
            
            writer.close()
            await writer.wait_closed()
            
            return banner if banner else "No banner"
            
        except Exception:
            return "Unable to retrieve banner"
    
    def _get_service_name(self, port: int) -> str:
        """根據端口號獲取服務名稱"""
        service_map = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            80: "HTTP", 110: "POP3", 111: "RPC", 135: "RPC", 139: "NetBIOS",
            143: "IMAP", 443: "HTTPS", 993: "IMAPS", 995: "POP3S",
            1723: "PPTP", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
            5900: "VNC", 8080: "HTTP-Alt"
        }
        return service_map.get(port, f"Unknown ({port})")
    
    def _categorize_service(self, port: int) -> str:
        """服務分類"""
        web_ports = [80, 443, 8080, 8443, 8000, 8888]
        db_ports = [3306, 5432, 1433, 1521, 27017]
        
        if port in web_ports:
            return "web"
        elif port in db_ports:
            return "database"
        else:
            return "other"
    
    async def _nmap_port_scan(self, host: str) -> Dict[str, Any]:
        """使用 nmap 進行真實端口掃描 (P0-1 Full 修復)
        
        Args:
            host: 目標主機
            
        Returns:
            包含真實掃描結果的字典
        """
        results = {
            "open_ports": [],
            "closed_ports": [],
            "filtered_ports": [],
            "total_scanned": 0
        }
        
        try:
            # 當在非 Nmap 環境或 nmap 初始化失敗時，nm 為 None
            if not self.nm:
                logger.warning("⚠️ Nmap not available, using heuristic mode")
                return await self._heuristic_port_scan(host)
            
            # 構建端口列表字串
            ports_str = ','.join(map(str, self.common_ports))
            
            # 執行 nmap 掃描 (在線程池中執行以避免阻塞)
            loop = asyncio.get_event_loop()
            if self.nm is not None:  # 明確檢查 nm 不為 None
                await loop.run_in_executor(
                    None,
                    lambda: self.nm.scan(
                        hosts=host,
                        ports=ports_str,
                        arguments='-sV --version-intensity 5'  # 服務版本檢測
                    )
                )
            else:
                logger.error("Nmap instance is None, cannot perform scan")
                return {"error": "Nmap not available"}
            
            # 解析結果
            if host in self.nm.all_hosts():
                for proto in self.nm[host].all_protocols():
                    ports = self.nm[host][proto].keys()
                    results["total_scanned"] += len(ports)
                    
                    for port in ports:
                        port_info = self.nm[host][proto][port]
                        state = port_info['state']
                        
                        port_data = {
                            "port": port,
                            "protocol": proto,
                            "service": port_info.get('name', 'unknown'),
                            "version": port_info.get('version', ''),
                            "product": port_info.get('product', ''),
                            "extrainfo": port_info.get('extrainfo', ''),
                            "state": state,
                            "confidence": 0.9  # nmap 掃描高可信度
                        }
                        
                        if state == 'open':
                            results["open_ports"].append(port_data)
                        elif state == 'closed':
                            results["closed_ports"].append(port_data)
                        elif state == 'filtered':
                            results["filtered_ports"].append(port_data)
            
            logger.info(f"✅ Nmap scan completed: {len(results['open_ports'])} open ports found")
            
        except Exception as e:
            logger.error(f"❌ Nmap scan failed: {e}")
            # 返回錯誤但不中斷
            results["error"] = str(e)
        
        return results
    
    def _detect_service_version(self) -> Optional[Dict[str, str]]:
        """檢測服務版本 - 已整合到 nmap 掃描中"""  
        # ✅ P0-1 修復完成: 服務版本檢測已整合到 _nmap_port_scan
        # 此方法保留用於兼容性
        return None
    
    async def _heuristic_port_scan(self, host: str) -> Dict[str, Any]:
        """當 nmap 不可用時的啟發式端口掃描"""
        logger.info(f"🔍 Running heuristic port scan for {host}")
        
        results = {
            "open_ports": [],
            "closed_ports": [],
            "scan_method": "heuristic",
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        try:
            # 掃描常見端口
            tasks = []
            for port in self.common_ports:
                task = self._check_port_simple(host, port)
                tasks.append(task)
            
            # 並行檢查端口
            port_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(port_results):
                port = self.common_ports[i]
                if isinstance(result, Exception):
                    logger.debug(f"Port {port} check failed: {result}")
                    continue
                
                if result == "open":
                    service_name = self._guess_service_by_port(port)
                    results["open_ports"].append({
                        "port": port,
                        "protocol": "tcp",
                        "state": "open",
                        "service": service_name,
                        "method": "heuristic"
                    })
                else:
                    results["closed_ports"].append(port)
            
            logger.info(f"✅ Heuristic scan completed. Found {len(results['open_ports'])} open ports")
            
        except Exception as e:
            logger.error(f"❌ Heuristic port scan failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _check_port_simple(self, host: str, port: int) -> str:
        """簡單的端口檢查"""
        try:
            async with asyncio.timeout(3.0):
                _, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                return "open"
        except ConnectionRefusedError:
            return "closed"
    
    def _guess_service_by_port(self, port: int) -> str:
        """根據端口號猜測服務類型"""
        common_services = {
            21: "ftp",
            22: "ssh", 
            23: "telnet",
            25: "smtp",
            53: "dns",
            80: "http",
            110: "pop3",
            135: "rpc",
            139: "netbios",
            443: "https",
            445: "smb",
            993: "imaps",
            995: "pop3s",
            3389: "rdp",
            5432: "postgresql",
            3306: "mysql",
            1433: "mssql",
            6379: "redis",
            27017: "mongodb"
        }
        return common_services.get(port, "unknown")
    
    async def _get_host_info(self, host: str) -> Dict[str, Any]:
        """獲取主機資訊"""
        info = {
            "hostname": host,
            "ip_address": "",
            "os_guess": "Unknown",
            "uptime": "Unknown"
        }
        
        try:
            # 解析IP地址
            ip = socket.gethostbyname(host)
            info["ip_address"] = ip
            
            # 簡單的OS指紋識別（基於端口檢查）
            windows_port = await self._check_port(host, 135)  # Windows RPC
            ssh_port = await self._check_port(host, 22)  # SSH
            
            if windows_port == "open":
                info["os_guess"] = "Windows"
            elif ssh_port == "open":
                info["os_guess"] = "Linux/Unix"
            
        except Exception as e:
            logger.debug(f"獲取主機資訊失敗: {e}")
        
        return info
    
    def _get_network_info(self, host: str) -> Dict[str, Any]:
        """獲取網路資訊"""
        info = {
            "network": "Unknown",
            "subnet_mask": "Unknown",
            "gateway": "Unknown"
        }
        
        try:
            ip = socket.gethostbyname(host)
            # 簡單的網路推測
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                info["network"] = "Private Network"
                if ip.startswith("192.168"):
                    info["subnet_mask"] = "255.255.255.0"
                elif ip.startswith("10."):
                    info["subnet_mask"] = "255.0.0.0"
                elif ip.startswith("172."):
                    info["subnet_mask"] = "255.255.0.0"
            else:
                info["network"] = "Public Network"
        except Exception:
            pass
        
        return info
    
    async def _test_reachability(self, host: str) -> Dict[str, Any]:
        """測試可達性"""
        reachability = {
            "ping_response": False,
            "tcp_connect": False,
            "response_time": 0
        }
        
        try:
            start_time = datetime.now()
            
            # 測試TCP連接（使用端口80）
            status = await self._check_port(host, 80)
            if status == "open":
                reachability["tcp_connect"] = True
                reachability["ping_response"] = True
            
            end_time = datetime.now()
            reachability["response_time"] = (end_time - start_time).total_seconds() * 1000
            
        except Exception:
            pass
        
        return reachability
    
    def _get_dns_info(self, host: str) -> Dict[str, Any]:
        """獲取DNS資訊"""
        dns_info = {
            "hostname": host,
            "ip_addresses": [],
            "reverse_dns": "Unknown"
        }
        
        try:
            # 正向DNS查詢
            ip = socket.gethostbyname(host)
            dns_info["ip_addresses"].append(ip)
            
            # 反向DNS查詢
            try:
                reverse_hostname = socket.gethostbyaddr(ip)[0]
                dns_info["reverse_dns"] = reverse_hostname
            except Exception:
                pass
                
        except Exception:
            pass
        
        return dns_info
    
    def _extract_host(self, target: str) -> str:
        """從目標字串中提取主機名稱或IP"""
        # 移除協議前綴
        if "://" in target:
            target = target.split("://")[1]
        
        # 移除端口和路徑
        if "/" in target:
            target = target.split("/")[0]
        if ":" in target and target.count(":") <= 1:  # 不是IPv6
            target = target.split(":")[0]
        
        return target
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成掃描摘要"""
        summary = {
            "total_open_ports": 0,
            "critical_services": [],
            "security_concerns": [],
            "recommendations": []
        }
        
        # 如果是端口掃描結果
        if "open_ports" in results:
            summary["total_open_ports"] = len(results["open_ports"])
            
            for port_info in results["open_ports"]:
                port = port_info["port"]
                service = port_info["service"]
                
                # 識別關鍵服務
                if port in [21, 23, 135, 139, 445, 3389]:  # 常見風險端口
                    summary["critical_services"].append(f"{service} ({port})")
                    summary["security_concerns"].append(f"發現潛在風險服務: {service}")
        
        # 生成建議
        if summary["critical_services"]:
            summary["recommendations"].append("關閉不必要的風險服務")
            summary["recommendations"].append("對重要服務實施存取控制")
        
        return summary
    
    def get_scan_results(self) -> List[Dict[str, Any]]:
        """獲取所有掃描結果"""
        return self.results
    
    def cleanup(self) -> None:
        """清理掃描器資源"""
        self.results.clear()
        logger.info(f"網路掃描器已清理，會話: {self.session_id}")


def demo_network_scanner():
    """網路掃描器演示函數"""
    async def run_demo():
        scanner = NetworkScanner()
        
        # 初始化
        scanner.initialize()
        
        # 掃描測試目標
        results = await scanner.scan_target("localhost:3000", "port_scan")
        
        print("🔍 網路掃描結果:")
        print(f"目標: {results.get('target')}")
        print(f"掃描類型: {results.get('scan_type')}")
        
        scan_results = results.get('results', {})
        if 'open_ports' in scan_results:
            print(f"開放端口數量: {len(scan_results['open_ports'])}")
            for port_info in scan_results['open_ports'][:5]:  # 顯示前5個
                print(f"- 端口 {port_info['port']}: {port_info['service']}")
        
        summary = results.get('summary', {})
        if summary.get('security_concerns'):
            print("安全關注點:")
            for concern in summary['security_concerns']:
                print(f"- {concern}")
        
        scanner.cleanup()
    
    # 執行演示
    asyncio.run(run_demo())


if __name__ == "__main__":
    demo_network_scanner()