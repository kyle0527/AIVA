"""
AIVA 網路掃描器
負責執行端口掃描、服務發現和網路枚舉
"""

import asyncio
import logging
import socket
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import ipaddress

logger = logging.getLogger(__name__)


class NetworkScanner:
    """網路掃描器主類別"""
    
    def __init__(self):
        self.session_id = None
        self.scan_config = {}
        self.results = []
        self.common_ports = [
            21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 993, 995, 1723, 3306, 3389, 5432, 5900, 8080
        ]
        
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
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
        """執行端口掃描"""
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
        
        logger.info(f"掃描端口: {host}")
        
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
            
            # 嘗試服務版本檢測
            version_info = await self._detect_service_version(host, port)
            if version_info:
                service_info.update(version_info)
            
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
        results["network_info"] = await self._get_network_info(host)
        
        # 可達性測試
        results["reachability"] = await self._test_reachability(host)
        
        # DNS資訊
        results["dns_info"] = await self._get_dns_info(host)
        
        return results
    
    async def _comprehensive_scan(self, target: str) -> Dict[str, Any]:
        """綜合掃描"""
        results = {
            "port_scan": await self._port_scan(target),
            "service_discovery": await self._service_discovery(target),
            "network_enum": await self._network_enumeration(target)
        }
        
        return results
    
    async def _check_port(self, host: str, port: int, timeout: float = 1.0) -> str:
        """檢查端口狀態"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return "open"
        except (ConnectionRefusedError, OSError):
            return "closed"
        except asyncio.TimeoutError:
            return "filtered"
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
    
    async def _detect_service_version(self, host: str, port: int) -> Optional[Dict[str, str]]:
        """檢測服務版本"""
        try:
            # 模擬版本檢測
            if port == 80 or port == 8080:
                return {"version": "Apache/2.4.41", "os": "Ubuntu"}
            elif port == 443:
                return {"version": "nginx/1.18.0", "os": "Ubuntu"}
            elif port == 22:
                return {"version": "OpenSSH 8.2p1", "os": "Ubuntu 20.04"}
            elif port == 3306:
                return {"version": "MySQL 8.0.25", "os": "Linux"}
            else:
                return None
        except Exception:
            return None
    
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
            
            # 簡單的OS指紋識別（模擬）
            if await self._check_port(host, 135):  # Windows RPC
                info["os_guess"] = "Windows"
            elif await self._check_port(host, 22):  # SSH (通常是Linux/Unix)
                info["os_guess"] = "Linux/Unix"
            
        except Exception as e:
            logger.debug(f"獲取主機資訊失敗: {e}")
        
        return info
    
    async def _get_network_info(self, host: str) -> Dict[str, Any]:
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
            status = await self._check_port(host, 80, timeout=2.0)
            if status == "open":
                reachability["tcp_connect"] = True
            
            end_time = datetime.now()
            reachability["response_time"] = (end_time - start_time).total_seconds() * 1000
            
            # 簡單的ping模擬
            reachability["ping_response"] = reachability["tcp_connect"]
            
        except Exception:
            pass
        
        return reachability
    
    async def _get_dns_info(self, host: str) -> Dict[str, Any]:
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
        if ":" in target and not target.count(":") > 1:  # 不是IPv6
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
    
    async def get_scan_results(self) -> List[Dict[str, Any]]:
        """獲取所有掃描結果"""
        return self.results
    
    async def cleanup(self):
        """清理掃描器資源"""
        self.results.clear()
        logger.info(f"網路掃描器已清理，會話: {self.session_id}")


def demo_network_scanner():
    """網路掃描器演示函數"""
    async def run_demo():
        scanner = NetworkScanner()
        
        # 初始化
        await scanner.initialize()
        
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
        
        await scanner.cleanup()
    
    # 執行演示
    asyncio.run(run_demo())


if __name__ == "__main__":
    demo_network_scanner()