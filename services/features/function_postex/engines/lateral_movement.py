"""
function_postex.engines.lateral_movement
橫向移動檢測引擎

檢測和執行網路內部橫向移動技術。
"""

import socket
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv4Address

from aiva_common.utils import get_logger
from aiva_common.enums.common import Severity, Confidence

logger = get_logger(__name__)


@dataclass
class LateralMovementVector:
    """橫向移動向量"""
    id: str
    severity: Severity
    confidence: Confidence
    title: str
    description: str
    evidence: Dict[str, Any]
    recommendation: str
    technique: str  # smb, wmi, psexec, rdp, ssh, etc.
    target_hosts: List[str]


class LateralMovementEngine:
    """橫向移動檢測引擎"""
    
    def __init__(self, safe_mode: bool = False):
        """
        初始化橫向移動引擎
        
        Args:
            safe_mode: 安全模式（僅檢測，不執行）
        """
        self.safe_mode = safe_mode
        logger.info(f"橫向移動引擎初始化 (SafeMode: {safe_mode})")
    
    def scan_network(self, network: str = "192.168.1.0/24") -> List[LateralMovementVector]:
        """
        掃描網路尋找橫向移動機會
        
        Args:
            network: 目標網路 CIDR
        
        Returns:
            List[LateralMovementVector]: 發現的橫向移動向量
        """
        vectors = []
        
        try:
            net = IPv4Network(network, strict=False)
            alive_hosts = self._discover_hosts(net)
            
            logger.info(f"發現 {len(alive_hosts)} 個活躍主機")
            
            for host in alive_hosts:
                vectors.extend(self._check_smb_access(host))
                vectors.extend(self._check_ssh_access(host))
                vectors.extend(self._check_rdp_access(host))
                vectors.extend(self._check_winrm_access(host))
        
        except Exception as e:
            logger.error(f"網路掃描錯誤: {e}")
        
        return vectors
    
    def _discover_hosts(self, network: IPv4Network) -> List[str]:
        """發現活躍主機"""
        alive_hosts = []
        
        for ip in network.hosts():
            if self._is_host_alive(str(ip)):
                alive_hosts.append(str(ip))
        
        return alive_hosts
    
    def _is_host_alive(self, host: str, timeout: int = 1) -> bool:
        """檢查主機是否存活（ICMP/TCP）"""
        try:
            # Try TCP connect to common ports
            for port in [80, 443, 445, 22, 3389]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    return True
            return False
        except:
            return False
    
    def _check_smb_access(self, host: str) -> List[LateralMovementVector]:
        """檢查 SMB 訪問"""
        vectors = []
        
        try:
            # Check if SMB port is open (445)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, 445))
            sock.close()
            
            if result == 0:
                vectors.append(LateralMovementVector(
                    id=f"LATERAL-SMB-{host.replace('.', '_')}",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.MEDIUM,
                    title=f"SMB Service Accessible: {host}",
                    description=f"SMB port 445 is open on {host}, potential for lateral movement",
                    evidence={
                        "host": host,
                        "port": 445,
                        "service": "SMB"
                    },
                    recommendation="Test for SMB relay attacks, null sessions, or credential reuse",
                    technique="smb",
                    target_hosts=[host]
                ))
        
        except Exception as e:
            logger.debug(f"SMB check failed for {host}: {e}")
        
        return vectors
    
    def _check_ssh_access(self, host: str) -> List[LateralMovementVector]:
        """檢查 SSH 訪問"""
        vectors = []
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, 22))
            sock.close()
            
            if result == 0:
                vectors.append(LateralMovementVector(
                    id=f"LATERAL-SSH-{host.replace('.', '_')}",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.MEDIUM,
                    title=f"SSH Service Accessible: {host}",
                    description=f"SSH port 22 is open on {host}",
                    evidence={
                        "host": host,
                        "port": 22,
                        "service": "SSH"
                    },
                    recommendation="Test for weak credentials, SSH key reuse, or authorized_keys persistence",
                    technique="ssh",
                    target_hosts=[host]
                ))
        
        except Exception as e:
            logger.debug(f"SSH check failed for {host}: {e}")
        
        return vectors
    
    def _check_rdp_access(self, host: str) -> List[LateralMovementVector]:
        """檢查 RDP 訪問"""
        vectors = []
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, 3389))
            sock.close()
            
            if result == 0:
                vectors.append(LateralMovementVector(
                    id=f"LATERAL-RDP-{host.replace('.', '_')}",
                    severity=Severity.HIGH,
                    confidence=Confidence.MEDIUM,
                    title=f"RDP Service Accessible: {host}",
                    description=f"RDP port 3389 is open on {host}",
                    evidence={
                        "host": host,
                        "port": 3389,
                        "service": "RDP"
                    },
                    recommendation="Test for weak RDP credentials or RDP vulnerabilities (BlueKeep)",
                    technique="rdp",
                    target_hosts=[host]
                ))
        
        except Exception as e:
            logger.debug(f"RDP check failed for {host}: {e}")
        
        return vectors
    
    def _check_winrm_access(self, host: str) -> List[LateralMovementVector]:
        """檢查 WinRM 訪問"""
        vectors = []
        
        try:
            for port in [5985, 5986]:  # HTTP, HTTPS
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    vectors.append(LateralMovementVector(
                        id=f"LATERAL-WINRM-{host.replace('.', '_')}_{port}",
                        severity=Severity.HIGH,
                        confidence=Confidence.MEDIUM,
                        title=f"WinRM Service Accessible: {host}:{port}",
                        description=f"WinRM is enabled on {host}, allowing remote command execution",
                        evidence={
                            "host": host,
                            "port": port,
                            "service": "WinRM"
                        },
                        recommendation="Test for credential reuse or weak WinRM authentication",
                        technique="winrm",
                        target_hosts=[host]
                    ))
        
        except Exception as e:
            logger.debug(f"WinRM check failed for {host}: {e}")
        
        return vectors
