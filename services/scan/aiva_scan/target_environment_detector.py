#!/usr/bin/env python3
"""
AIVA 靶場環境檢測器
用途: 自動檢測靶場環境狀態，包括端口掃描、服務檢測和連通性驗證
基於: 系統現況分析中的手動觸發框架需求
"""

import asyncio
import socket
import requests
import subprocess
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

class TargetEnvironmentDetector:
    """靶場環境檢測器"""
    
    def __init__(self):
        self.scan_results = {}
        self.last_scan_time = None
        self.timeout = 5  # 連接超時 (秒)
        
        # 常見靶場配置
        self.common_targets = {
            "DVWA": {"ports": [80, 443], "path": "/dvwa", "keywords": ["DVWA"]},
            "WebGoat": {"ports": [8080], "path": "/WebGoat", "keywords": ["WebGoat"]},
            "Metasploitable": {"ports": [21, 22, 23, 25, 53, 80, 111, 139, 445], "keywords": ["ubuntu", "metasploitable"]},
            "VulnHub": {"ports": [80, 22, 443], "keywords": ["vulnerable", "vulnhub"]},
            "HackTheBox": {"ports": [80, 443, 22], "keywords": ["hackthebox", "htb"]},
            "TryHackMe": {"ports": [80, 443, 22], "keywords": ["tryhackme", "thm"]},
            "本地開發": {"ports": [3000, 8000, 8080, 9000], "keywords": ["localhost", "development"]}
        }
    
    async def detect_environment(self, target_ips: List[str] = None) -> Dict[str, Any]:
        """
        全面檢測靶場環境
        
        Args:
            target_ips: 指定要掃描的 IP 列表，為空則掃描本地網段
            
        Returns:
            檢測結果字典
        """
        print("🎯 開始靶場環境檢測...")
        detection_start = time.time()
        
        if not target_ips:
            target_ips = await self._discover_local_targets()
        
        results = {
            "scan_time": datetime.now().isoformat(),
            "targets_scanned": len(target_ips),
            "discovered_services": [],
            "identified_platforms": [],
            "recommendations": [],
            "scan_duration": 0
        }
        
        # 並行掃描所有目標
        scan_tasks = []
        for ip in target_ips:
            task = asyncio.create_task(self._scan_target(ip))
            scan_tasks.append(task)
        
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # 處理掃描結果
        for i, result in enumerate(scan_results):
            if isinstance(result, Exception):
                print(f"⚠️  掃描 {target_ips[i]} 時發生錯誤: {result}")
                continue
                
            if result["services"]:
                results["discovered_services"].extend(result["services"])
                
                # 識別靶場平台
                platform = self._identify_platform(result)
                if platform:
                    results["identified_platforms"].append({
                        "ip": target_ips[i],
                        "platform": platform,
                        "confidence": result.get("confidence", 0.5)
                    })
        
        # 生成建議
        results["recommendations"] = self._generate_recommendations(results)
        results["scan_duration"] = round(time.time() - detection_start, 2)
        
        self.scan_results = results
        self.last_scan_time = datetime.now()
        
        print(f"✅ 環境檢測完成 (耗時 {results['scan_duration']}s)")
        print(f"   發現 {len(results['discovered_services'])} 個服務")
        print(f"   識別 {len(results['identified_platforms'])} 個靶場平台")
        
        return results
    
    async def _discover_local_targets(self) -> List[str]:
        """發現本地網段可能的目標"""
        print("🔍 發現本地網段目標...")
        
        targets = ["127.0.0.1", "localhost"]  # 基本本地目標
        
        try:
            # 獲取本機 IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # 掃描同網段常見 IP
            base_ip = ".".join(local_ip.split(".")[:-1]) + "."
            common_ips = [f"{base_ip}{i}" for i in [1, 10, 100, 200, 254]]
            
            targets.extend(common_ips)
            
        except Exception as e:
            print(f"⚠️  本地網段發現異常: {e}")
        
        print(f"   將掃描 {len(targets)} 個目標")
        return targets
    
    async def _scan_target(self, ip: str) -> Dict[str, Any]:
        """掃描單個目標"""
        result = {
            "ip": ip,
            "services": [],
            "web_services": [],
            "ssh_available": False,
            "confidence": 0.0
        }
        
        # 掃描常見端口
        all_ports = set()
        for platform_info in self.common_targets.values():
            all_ports.update(platform_info["ports"])
        
        # 限制掃描端口數量以提高速度
        priority_ports = [21, 22, 23, 25, 53, 80, 135, 139, 443, 445, 993, 995, 3000, 8000, 8080, 8443, 9000]
        scan_ports = list(all_ports.intersection(priority_ports))[:15]  # 最多掃描 15 個端口
        
        port_tasks = []
        for port in scan_ports:
            task = asyncio.create_task(self._check_port(ip, port))
            port_tasks.append(task)
        
        port_results = await asyncio.gather(*port_tasks, return_exceptions=True)
        
        # 處理端口掃描結果
        for i, is_open in enumerate(port_results):
            if isinstance(is_open, Exception):
                continue
                
            if is_open:
                port = scan_ports[i]
                service_info = {
                    "port": port,
                    "service": self._identify_service(port),
                    "ip": ip
                }
                result["services"].append(service_info)
                
                # 檢查 Web 服務
                if port in [80, 443, 8080, 8000, 3000, 9000]:
                    web_info = await self._check_web_service(ip, port)
                    if web_info:
                        result["web_services"].append(web_info)
                
                # 檢查 SSH
                if port == 22:
                    result["ssh_available"] = True
        
        return result
    
    async def _check_port(self, ip: str, port: int) -> bool:
        """檢查指定端口是否開放"""
        try:
            # 使用 asyncio 進行非同步連接測試
            future = asyncio.open_connection(ip, port)
            reader, writer = await asyncio.wait_for(future, timeout=self.timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return False
        except Exception:
            return False
    
    async def _check_web_service(self, ip: str, port: int) -> Optional[Dict[str, Any]]:
        """檢查 Web 服務並獲取基本資訊"""
        protocols = ["http"] if port != 443 else ["https"]
        if port in [80, 443]:
            protocols = ["http", "https"]
        
        for protocol in protocols:
            try:
                url = f"{protocol}://{ip}:{port}"
                
                # 使用 requests 檢查 Web 服務
                response = requests.get(url, timeout=self.timeout, verify=False)
                
                return {
                    "url": url,
                    "status_code": response.status_code,
                    "title": self._extract_title(response.text),
                    "server": response.headers.get("Server", "Unknown"),
                    "content_length": len(response.text)
                }
                
            except requests.exceptions.RequestException:
                continue  # 嘗試下一個協議
            except Exception:
                continue
        
        return None
    
    def _extract_title(self, html: str) -> str:
        """從 HTML 中提取標題"""
        try:
            import re
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            if title_match:
                return title_match.group(1).strip()[:100]  # 限制長度
        except Exception:
            pass
        return "Unknown"
    
    def _identify_service(self, port: int) -> str:
        """根據端口識別服務類型"""
        service_map = {
            21: "FTP",
            22: "SSH", 
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            135: "RPC",
            139: "NetBIOS",
            443: "HTTPS",
            445: "SMB",
            993: "IMAPS",
            995: "POP3S",
            3000: "Development Server",
            8000: "HTTP Alt",
            8080: "HTTP Proxy",
            8443: "HTTPS Alt",
            9000: "Web Service"
        }
        return service_map.get(port, f"Unknown ({port})")
    
    def _identify_platform(self, scan_result: Dict[str, Any]) -> Optional[str]:
        """根據掃描結果識別靶場平台"""
        ip = scan_result["ip"]
        services = scan_result["services"]
        web_services = scan_result.get("web_services", [])
        
        # 根據開放端口和 Web 內容匹配平台
        for platform, config in self.common_targets.items():
            score = 0
            
            # 端口匹配評分
            open_ports = [s["port"] for s in services]
            matching_ports = set(open_ports).intersection(set(config["ports"]))
            if matching_ports:
                score += len(matching_ports) / len(config["ports"]) * 50
            
            # Web 內容關鍵字匹配
            for web_service in web_services:
                title = web_service.get("title", "").lower()
                server = web_service.get("server", "").lower()
                
                for keyword in config["keywords"]:
                    if keyword.lower() in title or keyword.lower() in server:
                        score += 30
            
            # 設置信心度
            scan_result["confidence"] = score / 100
            
            if score > 40:  # 信心度閾值
                return platform
        
        return None
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """根據檢測結果生成建議"""
        recommendations = []
        
        if not results["discovered_services"]:
            recommendations.append("❌ 未發現任何服務，請確認靶場是否已啟動")
            recommendations.append("💡 建議檢查 Docker 容器或虛擬機狀態")
        
        if not results["identified_platforms"]:
            recommendations.append("⚠️  未識別出已知靶場平台")
            recommendations.append("💡 可能需要手動配置目標資訊")
        else:
            for platform_info in results["identified_platforms"]:
                platform = platform_info["platform"]
                confidence = platform_info["confidence"]
                recommendations.append(
                    f"✅ 發現 {platform} 靶場 (信心度: {confidence:.1%})"
                )
        
        # Web 服務建議
        web_count = len([s for s in results["discovered_services"] if s["port"] in [80, 443, 8080]])
        if web_count > 0:
            recommendations.append(f"🌐 發現 {web_count} 個 Web 服務，適合 Web 應用滲透測試")
        
        # SSH 服務建議
        ssh_count = len([s for s in results["discovered_services"] if s["port"] == 22])
        if ssh_count > 0:
            recommendations.append(f"🔐 發現 {ssh_count} 個 SSH 服務，可進行登入嘗試")
        
        return recommendations
    
    def export_results(self, output_path: str = None) -> str:
        """匯出檢測結果"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"target_detection_report_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.scan_results, f, ensure_ascii=False, indent=2)
            
            print(f"📊 檢測報告已輸出至: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 報告輸出失敗: {e}")
            return ""
    
    def get_available_targets(self) -> List[Dict[str, Any]]:
        """獲取可用的攻擊目標列表"""
        if not self.scan_results:
            return []
        
        targets = []
        for platform_info in self.scan_results.get("identified_platforms", []):
            ip = platform_info["ip"]
            platform = platform_info["platform"]
            
            # 查找該 IP 的服務
            services = [s for s in self.scan_results.get("discovered_services", []) 
                       if s["ip"] == ip]
            
            targets.append({
                "ip": ip,
                "platform": platform,
                "confidence": platform_info["confidence"],
                "services": services,
                "recommended": platform_info["confidence"] > 0.6
            })
        
        return sorted(targets, key=lambda x: x["confidence"], reverse=True)

# 使用範例和測試
async def demo_target_detection():
    """示範靶場環境檢測功能"""
    print("🎯 AIVA 靶場環境檢測器示範")
    print("=" * 50)
    
    detector = TargetEnvironmentDetector()
    
    # 執行環境檢測
    results = await detector.detect_environment()
    
    # 顯示結果摘要
    print(f"\n📊 檢測摘要:")
    print(f"   掃描目標數: {results['targets_scanned']}")
    print(f"   發現服務數: {len(results['discovered_services'])}")
    print(f"   識別平台數: {len(results['identified_platforms'])}")
    print(f"   掃描耗時: {results['scan_duration']}s")
    
    # 顯示發現的服務
    if results['discovered_services']:
        print(f"\n🔍 發現的服務:")
        for service in results['discovered_services'][:10]:  # 最多顯示 10 個
            print(f"   {service['ip']}:{service['port']} - {service['service']}")
    
    # 顯示識別的平台
    if results['identified_platforms']:
        print(f"\n🎯 識別的靶場平台:")
        for platform in results['identified_platforms']:
            print(f"   {platform['ip']} - {platform['platform']} (信心度: {platform['confidence']:.1%})")
    
    # 顯示建議
    if results['recommendations']:
        print(f"\n💡 建議:")
        for rec in results['recommendations']:
            print(f"   {rec}")
    
    # 獲取可用目標
    targets = detector.get_available_targets()
    if targets:
        print(f"\n✅ 推薦的攻擊目標:")
        for target in targets[:3]:  # 顯示前 3 個
            print(f"   {target['ip']} - {target['platform']} (推薦: {'是' if target['recommended'] else '否'})")
    
    # 匯出報告
    report_path = detector.export_results()
    if report_path:
        print(f"\n📄 詳細報告: {report_path}")

if __name__ == "__main__":
    asyncio.run(demo_target_detection())