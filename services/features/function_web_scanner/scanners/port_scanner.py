"""
function_web_scanner.scanners.port_scanner
端口掃描器

快速掃描常見服務端口。
"""

import concurrent.futures
from dataclasses import dataclass
import socket

from aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PortResult:
    """端口掃描結果"""
    host: str
    port: int
    state: str  # open, closed, filtered
    service: str = None
    banner: str = None


class PortScanner:
    """端口掃描器"""
    
    # 常見端口和服務映射
    COMMON_PORTS = {
        21: "FTP",
        22: "SSH",
        23: "Telnet",
        25: "SMTP",
        53: "DNS",
        80: "HTTP",
        110: "POP3",
        143: "IMAP",
        443: "HTTPS",
        445: "SMB",
        1433: "MSSQL",
        3306: "MySQL",
        3389: "RDP",
        5432: "PostgreSQL",
        5900: "VNC",
        6379: "Redis",
        8080: "HTTP-Proxy",
        8443: "HTTPS-Alt",
        27017: "MongoDB"
    }
    
    def __init__(self, timeout: int = 1, threads: int = 50):
        """
        初始化端口掃描器
        
        Args:
            timeout: 連接超時（秒）
            threads: 並發線程數
        """
        self.timeout = timeout
        self.threads = threads
        logger.info(f"端口掃描器初始化 (timeout: {timeout}s, threads: {threads})")
    
    def scan(self, host: str, ports: list[int] = None) -> list[PortResult]:
        """
        掃描端口
        
        Args:
            host: 目標主機
            ports: 端口列表（默認掃描常見端口）
        
        Returns:
            List[PortResult]: 開放的端口列表
        """
        if ports is None:
            ports = list(self.COMMON_PORTS.keys())
        
        results = []
        
        logger.info(f"掃描 {host} 的 {len(ports)} 個端口...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_to_port = {executor.submit(self._scan_port, host, port): port for port in ports}
            
            for future in concurrent.futures.as_completed(future_to_port):
                result = future.result()
                if result and result.state == "open":
                    results.append(result)
        
        logger.info(f"發現 {len(results)} 個開放端口")
        return results
    
    def _scan_port(self, host: str, port: int) -> PortResult:
        """掃描單個端口"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, port))
            
            if result == 0:
                # 端口開放，嘗試獲取 banner
                banner = self._grab_banner(sock)
                service = self.COMMON_PORTS.get(port, "Unknown")
                
                sock.close()
                
                return PortResult(
                    host=host,
                    port=port,
                    state="open",
                    service=service,
                    banner=banner
                )
            else:
                sock.close()
                return None
        
        except TimeoutError:
            return None
        except Exception as e:
            logger.debug(f"端口掃描錯誤 {host}:{port} - {e}")
            return None
    
    def _grab_banner(self, sock: socket.socket) -> str:
        """獲取服務 banner"""
        try:
            sock.settimeout(0.5)
            banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            return banner if banner else None
        except OSError:
            return None
