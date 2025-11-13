"""
測試 AIVA 功能偵察模組
基於 HackingTool 設計的測試用例
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock

from services.integration.capability.function_recon import (
    ReconTarget,
    ReconTargetType,
    FunctionReconManager,
    NetworkScanner,
    DNSRecon,
    WebRecon,
    OSINTRecon,
    ReconCLI
)


class TestReconTarget:
    """測試偵察目標"""
    
    def test_valid_ip_target(self):
        """測試有效IP目標"""
        target = ReconTarget(
            target="192.168.1.1",
            target_type=ReconTargetType.IP_ADDRESS,
            description="測試IP"
        )
        assert target.target == "192.168.1.1"
        assert target.target_type == ReconTargetType.IP_ADDRESS
    
    def test_invalid_ip_target(self):
        """測試無效IP目標"""
        with pytest.raises(ValueError):
            ReconTarget(
                target="invalid_ip",
                target_type=ReconTargetType.IP_ADDRESS
            )
    
    def test_email_target(self):
        """測試電子郵件目標"""
        target = ReconTarget(
            target="test@example.com",
            target_type=ReconTargetType.EMAIL
        )
        assert target.target == "test@example.com"
    
    def test_invalid_email_target(self):
        """測試無效電子郵件目標"""
        with pytest.raises(ValueError):
            ReconTarget(
                target="invalid_email",
                target_type=ReconTargetType.EMAIL
            )


class TestNetworkScanner:
    """測試網絡掃描器"""
    
    def setup_method(self):
        """設置測試環境"""
        self.scanner = NetworkScanner()
    
    @pytest.mark.asyncio
    async def test_nmap_scan_success(self):
        """測試成功的nmap掃描"""
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # 模擬成功的nmap輸出
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"Nmap scan report for 8.8.8.8\nPORT    STATE SERVICE\n53/tcp  open  domain",
                b""
            )
            mock_exec.return_value = mock_process
            
            result = await self.scanner.nmap_scan("8.8.8.8", "basic")
            
            assert result["success"] is True
            assert "8.8.8.8" in result["output"]
            assert result["target"] == "8.8.8.8"
    
    @pytest.mark.asyncio
    async def test_nmap_scan_failure(self):
        """測試失敗的nmap掃描"""
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # 模擬失敗的nmap執行
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (
                b"",
                b"Error: Invalid target"
            )
            mock_exec.return_value = mock_process
            
            result = await self.scanner.nmap_scan("invalid_target", "basic")
            
            assert result["success"] is False
            assert "Error" in result["error"]


class TestDNSRecon:
    """測試DNS偵察"""
    
    def setup_method(self):
        """設置測試環境"""
        self.dns_recon = DNSRecon()
    
    def test_host_to_ip_success(self):
        """測試成功的主機名解析"""
        with patch('socket.gethostbyname') as mock_resolve:
            mock_resolve.return_value = "8.8.8.8"
            
            result = self.dns_recon.host_to_ip("google.com")
            
            assert result["success"] is True
            assert result["ip"] == "8.8.8.8"
            assert result["hostname"] == "google.com"
    
    def test_host_to_ip_failure(self):
        """測試失敗的主機名解析"""
        with patch('socket.gethostbyname') as mock_resolve:
            mock_resolve.side_effect = Exception("Name resolution failed")
            
            result = self.dns_recon.host_to_ip("nonexistent.domain")
            
            assert result["success"] is False
            assert "Name resolution failed" in result["error"]
    
    def test_reverse_dns_success(self):
        """測試成功的反向DNS查詢"""
        with patch('socket.gethostbyaddr') as mock_reverse:
            mock_reverse.return_value = ("google-dns.google.com", [], ["8.8.8.8"])
            
            result = self.dns_recon.reverse_dns("8.8.8.8")
            
            assert result["success"] is True
            assert result["hostname"] == "google-dns.google.com"
            assert result["ip"] == "8.8.8.8"


class TestWebRecon:
    """測試Web偵察"""
    
    def setup_method(self):
        """設置測試環境"""
        self.web_recon = WebRecon()
    
    def test_website_info_success(self):
        """測試成功的網站信息收集"""
        with patch.object(self.web_recon.session, 'get') as mock_get:
            # 模擬HTTP回應
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'Server': 'nginx/1.18.0',
                'Content-Type': 'text/html; charset=utf-8'
            }
            mock_response.content = b"<html><head><title>Test</title></head></html>"
            mock_get.return_value = mock_response
            
            result = self.web_recon.website_info("example.com")
            
            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["server"] == "nginx/1.18.0"
    
    def test_check_admin_panels(self):
        """測試管理面板檢查"""
        with patch.object(self.web_recon.session, 'get') as mock_get:
            # 模擬找到一個管理面板
            def side_effect(url, **kwargs):
                mock_response = Mock()
                if "admin" in url:
                    mock_response.status_code = 200
                    mock_response.text = "<html><head><title>Admin Panel</title></head></html>"
                else:
                    mock_response.status_code = 404
                    mock_response.text = "Not Found"
                return mock_response
            
            mock_get.side_effect = side_effect
            
            result = self.web_recon.check_admin_panels("example.com")
            
            assert result["success"] is True
            assert result["total_found"] >= 1
            assert len(result["found_panels"]) >= 1


class TestFunctionReconManager:
    """測試功能偵察管理器"""
    
    def setup_method(self):
        """設置測試環境"""
        self.manager = FunctionReconManager()
    
    def test_create_target(self):
        """測試創建偵察目標"""
        target = self.manager.create_target(
            "192.168.1.1",
            ReconTargetType.IP_ADDRESS,
            "測試目標"
        )
        
        assert target.target == "192.168.1.1"
        assert target.target_type == ReconTargetType.IP_ADDRESS
        assert target.description == "測試目標"
    
    @pytest.mark.asyncio
    async def test_comprehensive_scan_ip(self):
        """測試IP地址綜合掃描"""
        target = ReconTarget(
            target="8.8.8.8",
            target_type=ReconTargetType.IP_ADDRESS
        )
        
        # 模擬各種掃描方法
        with patch.object(self.manager, '_scan_network') as mock_network, \
             patch.object(self.manager, '_scan_ports') as mock_ports, \
             patch.object(self.manager, '_scan_reverse_dns') as mock_dns:
            
            # 設置模擬結果
            from services.integration.capability.function_recon import ReconResult, ReconStatus
            
            mock_network.return_value = ReconResult(
                target=target,
                scan_type="network_scan",
                status=ReconStatus.COMPLETED,
                data={"success": True, "output": "Network scan completed"}
            )
            
            mock_ports.return_value = ReconResult(
                target=target,
                scan_type="port_scan",
                status=ReconStatus.COMPLETED,
                data={"success": True, "output": "Port scan completed"}
            )
            
            mock_dns.return_value = ReconResult(
                target=target,
                scan_type="reverse_dns",
                status=ReconStatus.COMPLETED,
                data={"success": True, "hostname": "dns.google"}
            )
            
            results = await self.manager.comprehensive_scan(target)
            
            assert len(results) == 3
            assert all(r.status == ReconStatus.COMPLETED for r in results)
    
    def test_get_scan_summary(self):
        """測試掃描摘要"""
        # 添加一些測試結果
        from services.integration.capability.function_recon import ReconResult, ReconStatus
        
        target = ReconTarget("test.com", ReconTargetType.DOMAIN)
        
        self.manager.results = [
            ReconResult(target, "dns_scan", ReconStatus.COMPLETED),
            ReconResult(target, "web_scan", ReconStatus.COMPLETED),
            ReconResult(target, "osint_scan", ReconStatus.FAILED)
        ]
        
        summary = self.manager.get_scan_summary()
        
        assert summary["total_scans"] == 3
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == pytest.approx(66.67, rel=1e-2)


class TestReconCLI:
    """測試偵察CLI"""
    
    def setup_method(self):
        """設置測試環境"""
        self.cli = ReconCLI()
    
    def test_detect_target_type_ip(self):
        """測試IP地址檢測"""
        target_type = self.cli._detect_target_type("192.168.1.1")
        assert target_type == ReconTargetType.IP_ADDRESS
    
    def test_detect_target_type_email(self):
        """測試電子郵件檢測"""
        target_type = self.cli._detect_target_type("user@example.com")
        assert target_type == ReconTargetType.EMAIL
    
    def test_detect_target_type_url(self):
        """測試URL檢測"""
        target_type = self.cli._detect_target_type("https://example.com")
        assert target_type == ReconTargetType.URL
    
    def test_detect_target_type_domain(self):
        """測試域名檢測"""
        target_type = self.cli._detect_target_type("example.com")
        assert target_type == ReconTargetType.DOMAIN


@pytest.mark.integration
class TestReconIntegration:
    """集成測試"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """測試完整工作流程"""
        manager = FunctionReconManager()
        
        # 創建測試目標
        target = manager.create_target(
            "8.8.8.8",
            ReconTargetType.IP_ADDRESS,
            "Google DNS 集成測試"
        )
        
        # 執行單項掃描測試（不依賴外部工具）
        dns_result = manager._scan_reverse_dns(target)
        assert dns_result.target == target
        assert dns_result.scan_type == "reverse_dns"
        
        # 檢查結果記錄
        assert len(manager.results) >= 1
        
        # 測試摘要生成
        summary = manager.get_scan_summary()
        assert summary["total_scans"] >= 1


if __name__ == "__main__":
    # 運行基本測試
    asyncio.run(pytest.main([__file__, "-v"]))