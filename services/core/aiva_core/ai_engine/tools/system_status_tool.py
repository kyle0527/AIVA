# -*- coding: utf-8 -*-
"""
系統狀態檢查工具

提供系統健康檢查功能，包括 CPU、記憶體、磁碟、網路等資源監控。
用於面板顯示和系統診斷。
"""
import shutil
import psutil
import platform
import time
import socket
from typing import Dict, Any, List, Optional
from datetime import datetime
import os


class SystemStatusTool:
    """
    系統健康檢查工具
    
    提供系統資源監控和狀態檢查：
    - CPU 使用率
    - 記憶體使用情況
    - 磁碟空間
    - 網路連接
    - 進程信息
    - 系統負載
    
    無外部依賴（僅需 psutil，已在 requirements 中）
    
    使用範例：
        >>> tool = SystemStatusTool()
        >>> status = tool.snapshot()
        >>> print(status['cpu_percent'], status['mem']['percent'])
    """
    
    def __init__(self, disk_paths: Optional[List[str]] = None):
        """
        初始化系統狀態工具
        
        Args:
            disk_paths: 要監控的磁碟路徑列表，預設為根目錄
        """
        self.disk_paths = disk_paths or ["/"] if os.name != "nt" else ["C:\\"]
        self.start_time = time.time()
        
    def snapshot(self) -> Dict[str, Any]:
        """
        獲取系統快照
        
        Returns:
            完整的系統狀態字典
        """
        return {
            "timestamp": self.get_timestamp(),
            "platform": self.get_platform_info(),
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "disk": self.get_disk_info(),
            "network": self.get_network_info(),
            "process": self.get_process_info(),
            "uptime": self.get_uptime(),
        }
    
    def get_timestamp(self) -> Dict[str, Any]:
        """獲取時間戳"""
        now = datetime.now()
        return {
            "unix": int(time.time()),
            "iso": now.isoformat(),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_platform_info(self) -> Dict[str, str]:
        """獲取平台信息"""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
        }
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """獲取 CPU 信息"""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            
            return {
                "count_physical": psutil.cpu_count(logical=False),
                "count_logical": psutil.cpu_count(logical=True),
                "percent": psutil.cpu_percent(interval=0.1),
                "percent_per_cpu": psutil.cpu_percent(interval=0.1, percpu=True),
                "frequency": {
                    "current": cpu_freq.current if cpu_freq else None,
                    "min": cpu_freq.min if cpu_freq else None,
                    "max": cpu_freq.max if cpu_freq else None,
                },
                "times": {
                    "user": cpu_times.user,
                    "system": cpu_times.system,
                    "idle": cpu_times.idle,
                },
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_memory_info(self) -> Dict[str, Any]:
        """獲取記憶體信息"""
        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "virtual": {
                    "total": vm.total,
                    "available": vm.available,
                    "used": vm.used,
                    "free": vm.free,
                    "percent": vm.percent,
                    "total_gb": round(vm.total / (1024**3), 2),
                    "available_gb": round(vm.available / (1024**3), 2),
                    "used_gb": round(vm.used / (1024**3), 2),
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent,
                    "total_gb": round(swap.total / (1024**3), 2),
                    "used_gb": round(swap.used / (1024**3), 2),
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_disk_info(self) -> Dict[str, Any]:
        """獲取磁碟信息"""
        try:
            disks = {}
            
            for path in self.disk_paths:
                try:
                    usage = shutil.disk_usage(path)
                    io = psutil.disk_io_counters()
                    
                    disks[path] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": (usage.used / usage.total * 100) if usage.total > 0 else 0,
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                    }
                    
                    if io:
                        disks[path]["io"] = {
                            "read_count": io.read_count,
                            "write_count": io.write_count,
                            "read_bytes": io.read_bytes,
                            "write_bytes": io.write_bytes,
                        }
                except Exception:
                    disks[path] = {"error": "無法訪問該路徑"}
            
            return disks
        except Exception as e:
            return {"error": str(e)}
    
    def get_network_info(self) -> Dict[str, Any]:
        """獲取網路信息"""
        try:
            net_io = psutil.net_io_counters()
            connections = psutil.net_connections(kind='inet')
            
            return {
                "io": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errin": net_io.errin,
                    "errout": net_io.errout,
                    "dropin": net_io.dropin,
                    "dropout": net_io.dropout,
                },
                "connections": {
                    "total": len(connections),
                    "established": sum(1 for c in connections if c.status == 'ESTABLISHED'),
                    "listen": sum(1 for c in connections if c.status == 'LISTEN'),
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_process_info(self) -> Dict[str, Any]:
        """獲取進程信息"""
        try:
            current_process = psutil.Process()
            
            return {
                "total_count": len(psutil.pids()),
                "current_pid": current_process.pid,
                "current_name": current_process.name(),
                "current_cpu_percent": current_process.cpu_percent(interval=0.1),
                "current_memory_mb": round(current_process.memory_info().rss / (1024**2), 2),
                "current_threads": current_process.num_threads(),
                "current_status": current_process.status(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_uptime(self) -> Dict[str, Any]:
        """獲取系統和服務運行時間"""
        try:
            boot_time = psutil.boot_time()
            current_time = time.time()
            system_uptime = current_time - boot_time
            service_uptime = current_time - self.start_time
            
            return {
                "system": {
                    "seconds": int(system_uptime),
                    "formatted": self._format_uptime(system_uptime),
                    "boot_time": datetime.fromtimestamp(boot_time).isoformat(),
                },
                "service": {
                    "seconds": int(service_uptime),
                    "formatted": self._format_uptime(service_uptime),
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _format_uptime(self, seconds: float) -> str:
        """格式化運行時間"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}天")
        if hours > 0:
            parts.append(f"{hours}小時")
        if minutes > 0:
            parts.append(f"{minutes}分鐘")
        if secs > 0 or not parts:
            parts.append(f"{secs}秒")
        
        return " ".join(parts)
    
    def check_health(self) -> Dict[str, Any]:
        """
        系統健康檢查
        
        Returns:
            健康狀態和建議
        """
        snapshot = self.snapshot()
        issues = []
        warnings = []
        
        # 檢查 CPU 使用率
        cpu_percent = snapshot['cpu'].get('percent', 0)
        if cpu_percent > 90:
            issues.append(f"CPU 使用率過高: {cpu_percent}%")
        elif cpu_percent > 75:
            warnings.append(f"CPU 使用率偏高: {cpu_percent}%")
        
        # 檢查記憶體使用率
        mem_percent = snapshot['memory']['virtual'].get('percent', 0)
        if mem_percent > 90:
            issues.append(f"記憶體使用率過高: {mem_percent}%")
        elif mem_percent > 80:
            warnings.append(f"記憶體使用率偏高: {mem_percent}%")
        
        # 檢查磁碟空間
        for path, disk in snapshot['disk'].items():
            if 'percent' in disk:
                if disk['percent'] > 95:
                    issues.append(f"磁碟 {path} 空間嚴重不足: {disk['percent']:.1f}%")
                elif disk['percent'] > 85:
                    warnings.append(f"磁碟 {path} 空間不足: {disk['percent']:.1f}%")
        
        # 判斷整體健康狀態
        if issues:
            status = "unhealthy"
        elif warnings:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "snapshot": snapshot,
            "checked_at": datetime.now().isoformat()
        }


# 便捷函數
def get_system_snapshot() -> Dict[str, Any]:
    """快速獲取系統快照的便捷函數"""
    tool = SystemStatusTool()
    return tool.snapshot()


def check_system_health() -> Dict[str, Any]:
    """快速檢查系統健康的便捷函數"""
    tool = SystemStatusTool()
    return tool.check_health()


__all__ = ["SystemStatusTool", "get_system_snapshot", "check_system_health"]
