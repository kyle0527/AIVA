"""
AIVA 統計收集器模組
==================

用於收集和管理 AIVA 系統各種性能指標和統計數據。
支援XSS模組、攻擊引擎、掃描器等組件的統計收集。

功能:
- Worker 性能統計
- 攻擊成功率統計  
- 系統資源使用統計
- 實時監控指標
"""

import asyncio
import time
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque


@dataclass
class WorkerStats:
    """Worker 統計數據"""
    worker_id: str
    start_time: datetime
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_activity: Optional[datetime] = None


@dataclass
class AttackStats:
    """攻擊統計數據"""
    attack_type: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    payload_stats: Dict[str, int] = field(default_factory=dict)


class StatisticsCollector:
    """AIVA 統計收集器"""
    
    def __init__(self, retention_hours: int = 24):
        """
        初始化統計收集器
        
        Args:
            retention_hours: 數據保留時間(小時)
        """
        self.retention_hours = retention_hours
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.attack_stats: Dict[str, AttackStats] = {}
        self.system_metrics: deque = deque(maxlen=1000)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.start_time = datetime.now()
        
    # ==================== Worker 統計 ====================
    
    def register_worker(self, worker_id: str) -> None:
        """註冊新的worker"""
        self.worker_stats[worker_id] = WorkerStats(
            worker_id=worker_id,
            start_time=datetime.now()
        )
    
    def update_worker_task(self, worker_id: str, task_completed: bool, 
                               response_time: float = 0.0) -> None:
        """更新worker任務統計"""
        if worker_id not in self.worker_stats:
            self.register_worker(worker_id)
        
        stats = self.worker_stats[worker_id]
        stats.total_tasks += 1
        stats.last_activity = datetime.now()
        
        if task_completed:
            stats.completed_tasks += 1
        else:
            stats.failed_tasks += 1
        
        # 更新平均響應時間
        if response_time > 0:
            if stats.avg_response_time == 0:
                stats.avg_response_time = response_time
            else:
                stats.avg_response_time = (stats.avg_response_time + response_time) / 2
    
    def update_worker_resources(self, worker_id: str, cpu: float, memory: float) -> None:
        """更新worker資源使用統計"""
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id].cpu_usage = cpu
            self.worker_stats[worker_id].memory_usage = memory
    
    def get_worker_stats(self, worker_id: str) -> Optional[Dict]:
        """獲取指定worker的統計"""
        if worker_id not in self.worker_stats:
            return None
        
        stats = self.worker_stats[worker_id]
        success_rate = (stats.completed_tasks / stats.total_tasks * 100 
                       if stats.total_tasks > 0 else 0)
        
        return {
            "worker_id": worker_id,
            "uptime": str(datetime.now() - stats.start_time),
            "total_tasks": stats.total_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(stats.avg_response_time, 3),
            "cpu_usage": round(stats.cpu_usage, 2),
            "memory_usage": round(stats.memory_usage, 2),
            "last_activity": stats.last_activity.isoformat() if stats.last_activity else None
        }
    
    # ==================== 攻擊統計 ====================
    
    def record_attack_attempt(self, attack_type: str, success: bool, 
                              execution_time: float = 0.0, payload: Optional[str] = None) -> None:
        """記錄攻擊嘗試"""
        if attack_type not in self.attack_stats:
            self.attack_stats[attack_type] = AttackStats(attack_type=attack_type)
        
        stats = self.attack_stats[attack_type]
        stats.total_attempts += 1
        
        if success:
            stats.successful_attempts += 1
        else:
            stats.failed_attempts += 1
        
        # 更新成功率
        stats.success_rate = (stats.successful_attempts / stats.total_attempts * 100)
        
        # 更新執行時間
        if execution_time > 0:
            if stats.avg_execution_time == 0:
                stats.avg_execution_time = execution_time
            else:
                stats.avg_execution_time = (stats.avg_execution_time + execution_time) / 2
        
        # 記錄payload統計
        if payload:
            if payload not in stats.payload_stats:
                stats.payload_stats[payload] = 0
            stats.payload_stats[payload] += 1
    
    def get_attack_stats(self, attack_type: Optional[str] = None) -> Dict:
        """獲取攻擊統計"""
        if attack_type and attack_type in self.attack_stats:
            stats = self.attack_stats[attack_type]
            return {
                "attack_type": attack_type,
                "total_attempts": stats.total_attempts,
                "successful_attempts": stats.successful_attempts,
                "failed_attempts": stats.failed_attempts,
                "success_rate": round(stats.success_rate, 2),
                "avg_execution_time": round(stats.avg_execution_time, 3),
                "top_payloads": sorted(stats.payload_stats.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]
            }
        else:
            # 返回所有攻擊統計
            all_stats = {}
            for attack_type, stats in self.attack_stats.items():
                all_stats[attack_type] = {
                    "total_attempts": stats.total_attempts,
                    "success_rate": round(stats.success_rate, 2),
                    "avg_execution_time": round(stats.avg_execution_time, 3)
                }
            return all_stats
    
    # ==================== 系統性能統計 ====================
    
    def collect_system_metrics(self) -> None:
        """收集系統性能指標"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metric = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
            
            self.system_metrics.append(metric)
            
            # 記錄到歷史數據
            self.performance_history["cpu"].append(cpu_percent)
            self.performance_history["memory"].append(memory.percent)
            self.performance_history["disk"].append(disk.percent)
            
        except Exception as e:
            print(f"收集系統指標失敗: {e}")
    
    def get_system_stats(self) -> Dict:
        """獲取系統統計"""
        if not self.system_metrics:
            self.collect_system_metrics()
        
        if self.system_metrics:
            latest = self.system_metrics[-1]
            
            # 計算平均值
            cpu_history = list(self.performance_history["cpu"])
            memory_history = list(self.performance_history["memory"])
            
            return {
                "current": latest,
                "averages": {
                    "cpu_avg": round(sum(cpu_history) / len(cpu_history), 2) if cpu_history else 0,
                    "memory_avg": round(sum(memory_history) / len(memory_history), 2) if memory_history else 0
                },
                "uptime": str(datetime.now() - self.start_time),
                "total_workers": len(self.worker_stats),
                "active_attacks": len(self.attack_stats)
            }
        else:
            return {"error": "無法獲取系統指標"}
    
    # ==================== 性能指標 ====================
    
    def get_performance_metrics(self) -> Dict:
        """獲取完整性能指標"""
        # 收集當前系統指標
        self.collect_system_metrics()
        
        # 計算worker性能
        total_workers = len(self.worker_stats)
        active_workers = sum(1 for stats in self.worker_stats.values() 
                           if stats.last_activity and 
                           (datetime.now() - stats.last_activity).seconds < 300)  # 5分鐘內活躍
        
        # 計算攻擊效率
        total_attacks = sum(stats.total_attempts for stats in self.attack_stats.values())
        successful_attacks = sum(stats.successful_attempts for stats in self.attack_stats.values())
        overall_success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0
        
        # 計算平均響應時間
        response_times = [stats.avg_response_time for stats in self.worker_stats.values() 
                         if stats.avg_response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_stats(),
            "workers": {
                "total": total_workers,
                "active": active_workers,
                "avg_response_time": round(avg_response_time, 3)
            },
            "attacks": {
                "total_attempts": total_attacks,
                "successful": successful_attacks,
                "overall_success_rate": round(overall_success_rate, 2)
            }
        }
    
    # ==================== 報告生成 ====================
    
    def generate_report(self, include_details: bool = True) -> str:
        """生成完整統計報告"""
        metrics = self.get_performance_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("[TARGET] AIVA 系統統計報告")
        report.append("=" * 60)
        report.append(f"生成時間: {metrics['timestamp']}")
        report.append(f"系統運行時間: {metrics['system']['uptime']}")
        report.append("")
        
        # 系統性能
        if "current" in metrics["system"]:
            current = metrics["system"]["current"]
            report.append("💻 系統性能:")
            report.append(f"  CPU使用率: {current['cpu_percent']}%")
            report.append(f"  記憶體使用率: {current['memory_percent']}%")
            report.append(f"  可用記憶體: {current['memory_available_gb']} GB")
            report.append(f"  磁碟使用率: {current['disk_percent']}%")
            report.append("")
        
        # Worker統計
        report.append("👥 Worker統計:")
        report.append(f"  總Worker數: {metrics['workers']['total']}")
        report.append(f"  活躍Worker數: {metrics['workers']['active']}")
        report.append(f"  平均響應時間: {metrics['workers']['avg_response_time']}秒")
        report.append("")
        
        # 攻擊統計
        report.append("⚔️ 攻擊統計:")
        report.append(f"  總攻擊嘗試: {metrics['attacks']['total_attempts']}")
        report.append(f"  成功攻擊: {metrics['attacks']['successful']}")
        report.append(f"  整體成功率: {metrics['attacks']['overall_success_rate']}%")
        report.append("")
        
        if include_details:
            # Worker詳細信息
            report.append("[STATS] Worker詳細統計:")
            for worker_id, stats in self.worker_stats.items():
                worker_detail = self.get_worker_stats(worker_id)
                if worker_detail:
                    report.append(f"  {worker_id}:")
                    report.append(f"    任務: {worker_detail['completed_tasks']}/{worker_detail['total_tasks']} "
                                f"(成功率: {worker_detail['success_rate']}%)")
                    report.append(f"    響應時間: {worker_detail['avg_response_time']}秒")
                    report.append(f"    CPU: {worker_detail['cpu_usage']}% | 記憶體: {worker_detail['memory_usage']}%")
            report.append("")
            
            # 攻擊類型詳細統計
            report.append("[ATTACK] 攻擊類型詳細統計:")
            for attack_type in self.attack_stats:
                attack_detail = self.get_attack_stats(attack_type)
                report.append(f"  {attack_type}:")
                report.append(f"    嘗試: {attack_detail['total_attempts']} | "
                            f"成功: {attack_detail['successful_attempts']} | "
                            f"成功率: {attack_detail['success_rate']}%")
                report.append(f"    平均執行時間: {attack_detail['avg_execution_time']}秒")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    # ==================== 數據清理 ====================
    
    def cleanup_old_data(self) -> None:
        """清理過期數據"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # 清理過期的worker統計
        expired_workers = [
            worker_id for worker_id, stats in self.worker_stats.items()
            if stats.last_activity and stats.last_activity < cutoff_time
        ]
        
        for worker_id in expired_workers:
            del self.worker_stats[worker_id]
        
        if expired_workers:
            print(f"清理了 {len(expired_workers)} 個過期worker統計")


# ==================== 全局實例 ====================

# 創建全局統計收集器實例
global_stats_collector = StatisticsCollector()


# ==================== 便捷函數 ====================

def collect_worker_stats(worker_id: str, task_completed: bool, response_time: float = 0.0):
    """便捷函數：更新worker統計"""
    global_stats_collector.update_worker_task(worker_id, task_completed, response_time)


def record_attack(attack_type: str, success: bool, execution_time: float = 0.0, payload: Optional[str] = None):
    """便捷函數：記錄攻擊統計"""
    global_stats_collector.record_attack_attempt(attack_type, success, execution_time, payload)


def get_current_stats() -> Dict:
    """便捷函數：獲取當前統計"""
    return global_stats_collector.get_performance_metrics()


def generate_stats_report() -> str:
    """便捷函數：生成統計報告"""
    return global_stats_collector.generate_report()


if __name__ == "__main__":
    # 測試統計收集器
    def test_collector():
        collector = StatisticsCollector()
        
        # 註冊workers
        collector.register_worker("xss_worker_1")
        collector.register_worker("sqli_worker_1")
        
        # 模擬任務
        collector.update_worker_task("xss_worker_1", True, 0.5)
        collector.update_worker_task("xss_worker_1", False, 1.2)
        collector.update_worker_task("sqli_worker_1", True, 0.8)
        
        # 模擬攻擊
        collector.record_attack_attempt("xss", True, 2.1, "alert(1)")
        collector.record_attack_attempt("xss", False, 1.8, "alert(2)")
        collector.record_attack_attempt("sqli", True, 3.5, "' OR 1=1--")
        
        # 生成報告
        print(collector.generate_report())
    
    test_collector()