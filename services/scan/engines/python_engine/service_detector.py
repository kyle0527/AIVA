"""
AIVA 服務檢測器
負責識別和分析目標系統上運行的服務
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class ServiceDetector:
    """服務檢測器主類別"""
    
    def __init__(self):
        self.session_id = None
        self.detection_config = {}
        self.results = []
        self.service_signatures = self._load_service_signatures()
        
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化服務檢測器"""
        try:
            self.detection_config = config or {}
            self.session_id = f"svc_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"服務檢測器已初始化，會話ID: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"服務檢測器初始化失敗: {e}")
            return False
    
    async def detect_services(self, target: str, ports: List[int] = None) -> Dict[str, Any]:
        """檢測目標系統上的服務"""
        try:
            results = {
                "target": target,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "detected_services": [],
                "service_summary": {
                    "total_services": 0,
                    "web_servers": 0,
                    "database_servers": 0,
                    "mail_servers": 0,
                    "other_services": 0
                }
            }
            
            host = self._extract_host(target)
            logger.info(f"開始服務檢測: {host}")
            
            # 如果沒有提供端口，使用常見端口
            if ports is None:
                ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1080, 3306, 3389, 5432, 8080]
            
            # 檢測每個端口上的服務
            for port in ports:
                service_info = await self._detect_service_on_port(host, port)
                if service_info:
                    results["detected_services"].append(service_info)
                    
                    # 更新統計
                    service_type = service_info.get("category", "other")
                    results["service_summary"][f"{service_type}_servers"] = \
                        results["service_summary"].get(f"{service_type}_servers", 0) + 1
            
            # 更新總計
            results["service_summary"]["total_services"] = len(results["detected_services"])
            
            # 進行深度服務分析
            if results["detected_services"]:
                self._perform_deep_analysis(results)
            
            self.results.append(results)
            logger.info(f"服務檢測完成，發現 {len(results['detected_services'])} 個服務")
            
            return results
            
        except Exception as e:
            logger.error(f"服務檢測失敗 {target}: {e}")
            return {"error": str(e), "target": target}
    
    async def _detect_service_on_port(self, host: str, port: int) -> Optional[Dict[str, Any]]:
        """檢測特定端口上的服務"""
        try:
            # 檢查端口是否開放
            if not await self._is_port_open(host, port):
                return None
            
            service_info = {
                "port": port,
                "protocol": "tcp",
                "service_name": "unknown",
                "version": "unknown",
                "category": "other",
                "banner": "",
                "fingerprint": {},
                "confidence": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # 獲取服務banner
            banner = await self._get_service_banner(host, port)
            service_info["banner"] = banner
            
            # 基於端口號的初步識別
            service_info.update(self._identify_by_port(port))
            
            # 基於banner的服務識別
            if banner:
                banner_analysis = self._analyze_banner(banner, port)
                service_info.update(banner_analysis)
            
            # 執行服務特定的檢測
            specific_info = await self._perform_service_specific_detection(host, port, service_info["service_name"])
            if specific_info:
                service_info.update(specific_info)
            
            logger.debug(f"檢測到服務: {host}:{port} - {service_info['service_name']}")
            return service_info
            
        except Exception as e:
            logger.debug(f"服務檢測錯誤 {host}:{port}: {e}")
            return None
    
    async def _is_port_open(self, host: str, port: int) -> bool:
        """檢查端口是否開放"""
        try:
            async with asyncio.timeout(2.0):  # 使用 timeout context manager
                _, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                return True
        except Exception:
            return False
    
    async def _get_service_banner(self, host: str, port: int) -> str:
        """獲取服務banner"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=3.0)
            
            # 對於HTTP服務，發送HTTP請求
            if port in [80, 8080, 443, 8443]:
                request = b"GET / HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n" % host.encode()
                writer.write(request)
                await writer.drain()
            
            # 讀取響應
            banner_data = await asyncio.wait_for(reader.read(2048), timeout=2.0)
            banner = banner_data.decode('utf-8', errors='ignore').strip()
            
            writer.close()
            await writer.wait_closed()
            
            return banner
            
        except Exception as e:
            logger.debug(f"獲取banner失敗 {host}:{port}: {e}")
            return ""
    
    def _identify_by_port(self, port: int) -> Dict[str, Any]:
        """基於端口號識別服務"""
        port_services = {
            21: {"service_name": "FTP", "category": "file_transfer", "confidence": 80},
            22: {"service_name": "SSH", "category": "remote_access", "confidence": 90},
            23: {"service_name": "Telnet", "category": "remote_access", "confidence": 80},
            25: {"service_name": "SMTP", "category": "mail", "confidence": 85},
            53: {"service_name": "DNS", "category": "dns", "confidence": 90},
            80: {"service_name": "HTTP", "category": "web", "confidence": 90},
            110: {"service_name": "POP3", "category": "mail", "confidence": 85},
            143: {"service_name": "IMAP", "category": "mail", "confidence": 85},
            443: {"service_name": "HTTPS", "category": "web", "confidence": 90},
            993: {"service_name": "IMAPS", "category": "mail", "confidence": 85},
            995: {"service_name": "POP3S", "category": "mail", "confidence": 85},
            3306: {"service_name": "MySQL", "category": "database", "confidence": 90},
            3389: {"service_name": "RDP", "category": "remote_access", "confidence": 90},
            5432: {"service_name": "PostgreSQL", "category": "database", "confidence": 90},
            8080: {"service_name": "HTTP-Alt", "category": "web", "confidence": 80}
        }
        
        return port_services.get(port, {"service_name": f"Unknown-{port}", "category": "other", "confidence": 20})
    
    def _analyze_banner(self, banner: str, port: int) -> Dict[str, Any]:
        """分析服務banner"""
        analysis = {"confidence": 50, "port": port}  # 保留端口信息用於分析
        banner_lower = banner.lower()
        
        # 使用專門的檢測方法
        if self._is_http_service(banner_lower):
            return self._analyze_http_banner(banner, banner_lower, analysis)
        elif self._is_ssh_service(banner_lower):
            return self._analyze_ssh_banner(banner, banner_lower, analysis)
        elif self._is_ftp_service(banner, banner_lower):
            return self._analyze_ftp_banner(banner, banner_lower, analysis)
        elif self._is_smtp_service(banner, banner_lower):
            return self._analyze_smtp_banner(analysis)
        elif self._is_database_service(banner_lower):
            return self._analyze_database_banner(banner, banner_lower, analysis)
        
        return analysis
    
    def _is_http_service(self, banner_lower: str) -> bool:
        """檢查是否為HTTP服務"""
        return "http/" in banner_lower
    
    def _is_ssh_service(self, banner_lower: str) -> bool:
        """檢查是否為SSH服務"""
        return "ssh" in banner_lower
    
    def _is_ftp_service(self, banner: str, banner_lower: str) -> bool:
        """檢查是否為FTP服務"""
        return "ftp" in banner_lower or banner.startswith("220 ")
    
    def _is_smtp_service(self, banner: str, banner_lower: str) -> bool:
        """檢查是否為SMTP服務"""
        return banner.startswith("220 ") and ("mail" in banner_lower or "smtp" in banner_lower)
    
    def _is_database_service(self, banner_lower: str) -> bool:
        """檢查是否為數據庫服務"""
        return "mysql" in banner_lower or "postgresql" in banner_lower
    
    def _analyze_http_banner(self, banner: str, banner_lower: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析HTTP banner"""
        analysis.update({
            "service_name": "HTTP",
            "category": "web", 
            "confidence": 95
        })
        
        # 檢測Web服務器類型
        server_info = self._detect_web_server_type(banner, banner_lower)
        analysis.update(server_info)
        
        return analysis
    
    def _analyze_ssh_banner(self, banner: str, banner_lower: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析SSH banner"""
        analysis.update({
            "service_name": "SSH",
            "category": "remote_access",
            "confidence": 95
        })
        
        if "openssh" in banner_lower:
            analysis["version"] = self._extract_version(banner, "openssh")
            analysis["ssh_implementation"] = "OpenSSH"
        
        return analysis
    
    def _analyze_ftp_banner(self, banner: str, banner_lower: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析FTP banner"""
        analysis.update({
            "service_name": "FTP",
            "category": "file_transfer",
            "confidence": 90
        })
        
        # 檢測FTP實現類型
        ftp_impl = self._detect_ftp_implementation(banner_lower)
        if ftp_impl:
            analysis["ftp_implementation"] = ftp_impl
        
        return analysis
    
    def _analyze_smtp_banner(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析SMTP banner"""
        analysis.update({
            "service_name": "SMTP",
            "category": "mail",
            "confidence": 90
        })
        return analysis
    
    def _analyze_database_banner(self, banner: str, banner_lower: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析數據庫banner"""
        if "mysql" in banner_lower:
            analysis.update({
                "service_name": "MySQL",
                "category": "database",
                "confidence": 95,
                "version": self._extract_version(banner, "mysql")
            })
        elif "postgresql" in banner_lower:
            analysis.update({
                "service_name": "PostgreSQL",
                "category": "database", 
                "confidence": 95
            })
        
        return analysis
    
    def _detect_web_server_type(self, banner: str, banner_lower: str) -> Dict[str, str]:
        """檢測Web服務器類型"""
        if "apache" in banner_lower:
            return {
                "version": self._extract_version(banner, "apache"),
                "server_software": "Apache"
            }
        elif "nginx" in banner_lower:
            return {
                "version": self._extract_version(banner, "nginx"),
                "server_software": "Nginx"
            }
        elif "iis" in banner_lower or "microsoft" in banner_lower:
            return {
                "server_software": "IIS",
                "version": self._extract_version(banner, "iis")
            }
        return {}
    
    def _detect_ftp_implementation(self, banner_lower: str) -> str:
        """檢測FTP實現類型"""
        if "vsftpd" in banner_lower:
            return "vsftpd"
        elif "proftpd" in banner_lower:
            return "ProFTPD"
        return ""
    
    def _extract_version(self, banner: str, service: str) -> str:
        """從banner中提取版本信息"""
        import re
        
        patterns = {
            "apache": r"Apache/([0-9\.]+)",
            "nginx": r"nginx/([0-9\.]+)",
            "openssh": r"OpenSSH_([0-9\.]+)",
            "mysql": r"([0-9\.]+)",
            "iis": r"IIS/([0-9\.]+)"
        }
        
        pattern = patterns.get(service.lower())
        if pattern:
            match = re.search(pattern, banner, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    async def _perform_service_specific_detection(self, host: str, port: int, service_name: str) -> Optional[Dict[str, Any]]:
        """執行服務特定的檢測"""
        try:
            if service_name.upper() == "HTTP" or service_name.upper() == "HTTPS":
                return await self._detect_web_service_details(host, port)
            elif service_name.upper() == "SSH":
                return await self._detect_ssh_details(host, port)
            elif service_name.upper() == "FTP":
                return await self._detect_ftp_details(host, port)
            elif "MYSQL" in service_name.upper():
                return await self._detect_mysql_details(host, port)
            else:
                return None
        except Exception as e:
            logger.debug(f"服務特定檢測失敗 {service_name}: {e}")
            return None
    
    async def _detect_web_service_details(self, host: str, port: int) -> Dict[str, Any]:
        """檢測Web服務詳細信息"""
        details = {
            "web_technologies": [],
            "server_headers": {},
            "supported_methods": [],
            "security_headers": {}
        }
        
        try:
            # 發送HTTP請求獲取更多信息
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=3.0)
            
            request = f"HEAD / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
            writer.write(request.encode())
            await writer.drain()
            
            response = await asyncio.wait_for(reader.read(2048), timeout=2.0)
            response_text = response.decode('utf-8', errors='ignore')
            
            writer.close()
            await writer.wait_closed()
            
            # 解析HTTP頭
            lines = response_text.split('\r\n')
            for line in lines[1:]:  # 跳過狀態行
                if ':' in line:
                    key, value = line.split(':', 1)
                    details["server_headers"][key.strip().lower()] = value.strip()
            
            # 檢測安全頭
            security_headers = [
                'x-frame-options', 'x-xss-protection', 'x-content-type-options',
                'strict-transport-security', 'content-security-policy'
            ]
            
            for header in security_headers:
                if header in details["server_headers"]:
                    details["security_headers"][header] = details["server_headers"][header]
            
            # 檢測Web技術
            server_header = details["server_headers"].get("server", "")
            if "apache" in server_header.lower():
                details["web_technologies"].append("Apache")
            if "nginx" in server_header.lower():
                details["web_technologies"].append("Nginx")
            if "php" in server_header.lower():
                details["web_technologies"].append("PHP")
            
        except Exception as e:
            logger.debug(f"Web服務詳細檢測失敗: {e}")
        
        return details
    
    async def _detect_ssh_details(self, host: str, port: int) -> Dict[str, Any]:
        """檢測SSH服務詳細信息"""
        details = {
            "ssh_version": "unknown",
            "supported_algorithms": [],
            "host_key_types": []
        }
        
        # SSH檢測邏輯（簡化版）
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=3.0)
            
            # 讀取SSH版本字符串
            ssh_banner = await asyncio.wait_for(reader.readline(), timeout=2.0)
            details["ssh_version"] = ssh_banner.decode('utf-8', errors='ignore').strip()
            
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            logger.debug(f"SSH詳細檢測失敗: {e}")
        
        return details
    
    async def _detect_ftp_details(self, host: str, port: int) -> Dict[str, Any]:
        """檢測FTP服務詳細信息"""
        details = {
            "ftp_banner": "",
            "anonymous_login": False,
            "features": []
        }
        
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=3.0)
            
            # 讀取FTP歡迎消息
            welcome = await asyncio.wait_for(reader.readline(), timeout=2.0)
            details["ftp_banner"] = welcome.decode('utf-8', errors='ignore').strip()
            
            # 測試匿名登錄
            writer.write(b"USER anonymous\r\n")
            await writer.drain()
            
            response = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if b"331" in response:  # 需要密碼
                writer.write(b"PASS anonymous@example.com\r\n")
                await writer.drain()
                
                async with asyncio.timeout(2.0):
                    login_response = await reader.readline()
                if b"230" in login_response:  # 登錄成功
                    details["anonymous_login"] = True
            
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            logger.debug(f"FTP詳細檢測失敗: {e}")
        
        return details
    
    async def _detect_mysql_details(self, host: str, port: int) -> Dict[str, Any]:
        """檢測MySQL服務詳細信息"""
        details = {
            "mysql_version": "unknown",
            "server_capabilities": [],
            "authentication_methods": []
        }
        
        # MySQL檢測邏輯（簡化版，實際需要MySQL協議解析）
        try:
            async with asyncio.timeout(3.0):
                reader, writer = await asyncio.open_connection(host, port)
            
            # 讀取MySQL握手包
            async with asyncio.timeout(2.0):
                handshake = await reader.read(1024)
            
            # 簡單解析版本信息（實際需要更複雜的協議解析）
            if len(handshake) >= 5:
                # MySQL握手包格式解析（簡化）
                details["mysql_version"] = "detected"
            
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            logger.debug(f"MySQL詳細檢測失敗: {e}")
        
        return details
    
    def _perform_deep_analysis(self, results: Dict[str, Any]):
        """執行深度服務分析"""
        try:
            analysis = {
                "service_relationships": [],
                "security_assessment": {},
                "performance_indicators": {},
                "compliance_status": {}
            }
            
            detected_services = results["detected_services"]
            
            # 分析服務關係
            web_services = [s for s in detected_services if s["category"] == "web"]
            db_services = [s for s in detected_services if s["category"] == "database"]
            
            if web_services and db_services:
                analysis["service_relationships"].append({
                    "type": "web_database",
                    "description": "檢測到Web服務和數據庫服務，可能存在Web應用"
                })
            
            # 安全評估
            security_issues = []
            for service in detected_services:
                # 檢查過時服務
                if service["service_name"] == "Telnet":
                    security_issues.append("發現不安全的Telnet服務")
                
                # 檢查匿名訪問
                if service.get("anonymous_login"):
                    security_issues.append("FTP服務允許匿名登錄")
                
                # 檢查缺少安全頭
                if service["category"] == "web" and not service.get("security_headers"):
                    security_issues.append("Web服務缺少安全頭")
            
            analysis["security_assessment"]["issues"] = security_issues
            analysis["security_assessment"]["risk_level"] = "HIGH" if security_issues else "LOW"
            
            results["deep_analysis"] = analysis
            
        except Exception as e:
            logger.error(f"深度分析失敗: {e}")
    
    def _extract_host(self, target: str) -> str:
        """從目標字串中提取主機名"""
        if "://" in target:
            target = target.split("://")[1]
        if "/" in target:
            target = target.split("/")[0]
        if ":" in target and target.count(":") <= 1:
            target = target.split(":")[0]
        return target
    
    def _load_service_signatures(self) -> Dict[str, Any]:
        """載入服務特徵庫"""
        return {
            "web_signatures": [
                {"pattern": r"Server: Apache/(\d+\.\d+)", "service": "Apache", "type": "web"},
                {"pattern": r"Server: nginx/(\d+\.\d+)", "service": "Nginx", "type": "web"},
                {"pattern": r"Server: Microsoft-IIS/(\d+\.\d+)", "service": "IIS", "type": "web"}
            ],
            "database_signatures": [
                {"pattern": r"MySQL", "service": "MySQL", "type": "database"},
                {"pattern": r"PostgreSQL", "service": "PostgreSQL", "type": "database"}
            ]
        }
    
    def get_detection_results(self) -> List[Dict[str, Any]]:
        """獲取所有檢測結果"""
        return self.results
    
    def cleanup(self):
        """清理檢測器資源"""
        self.results.clear()
        logger.info(f"服務檢測器已清理，會話: {self.session_id}")


def demo_service_detector():
    """服務檢測器演示函數"""
    async def run_demo():
        detector = ServiceDetector()
        
        # 初始化
        detector.initialize()
        
        # 檢測測試目標
        results = await detector.detect_services("localhost:3000")
        
        print("🔍 服務檢測結果:")
        print(f"目標: {results.get('target')}")
        
        services = results.get('detected_services', [])
        print(f"檢測到 {len(services)} 個服務:")
        
        for service in services:
            print(f"- 端口 {service['port']}: {service['service_name']}")
            print(f"  類別: {service['category']}")
            print(f"  置信度: {service['confidence']}%")
            if service.get('version') != 'unknown':
                print(f"  版本: {service['version']}")
        
        # 顯示安全評估
        if 'deep_analysis' in results:
            analysis = results['deep_analysis']
            security = analysis.get('security_assessment', {})
            
            if security.get('issues'):
                print("\n⚠️ 安全問題:")
                for issue in security['issues']:
                    print(f"- {issue}")
        
        await detector.cleanup()
    
    # 執行演示
    asyncio.run(run_demo())


if __name__ == "__main__":
    demo_service_detector()