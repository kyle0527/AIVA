#!/usr/bin/env python3
"""
AIVA JSON 操作記錄器
用途: 結構化記錄 AI 的每個操作步驟，為前端整合準備
基於: 自動啟動並持續執行 AI 攻擊學習的框架設計文件
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from queue import Queue, Empty
import logging

class AIOperationRecorder:
    """AI 操作記錄器 - 結構化記錄 AI 操作供前端使用"""
    
    def __init__(self, output_dir: str = "logs", enable_realtime: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.enable_realtime = enable_realtime
        self.operation_queue = Queue()
        self.operation_history = []
        self.session_id = self._generate_session_id()
        
        # 實時記錄線程
        self.recording_thread = None
        self.is_recording = False
        
        # 統計資料
        self.stats = {
            "total_operations": 0,
            "operations_by_type": {},
            "session_start": datetime.now().isoformat(),
            "last_operation": None
        }
        
        self.logger = self._setup_logger()
        
        if enable_realtime:
            self.start_recording()
    
    def _generate_session_id(self) -> str:
        """生成會話 ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"aiva_session_{timestamp}"
    
    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger("AIOperationRecorder")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 文件處理器
            log_file = self.output_dir / f"{self.session_id}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # 控制台處理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def record_operation(
        self, 
        command: str, 
        description: str, 
        parameters: Dict[str, Any] = None,
        operation_type: str = "command",
        result: Any = None,
        duration: float = None,
        success: bool = True
    ) -> str:
        """
        記錄一個 AI 操作
        
        Args:
            command: 命令或操作名稱
            description: 操作描述
            parameters: 操作參數
            operation_type: 操作類型 (command, decision, analysis, etc.)
            result: 操作結果
            duration: 執行時間 (秒)
            success: 是否成功
            
        Returns:
            操作 ID
        """
        operation_id = f"{self.session_id}_{len(self.operation_history) + 1:04d}"
        
        operation_record = {
            "operation_id": operation_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "description": description,
            "operation_type": operation_type,
            "parameters": parameters or {},
            "result": result,
            "duration": duration,
            "success": success,
            "metadata": {
                "sequence_number": len(self.operation_history) + 1,
                "recorded_at": time.time()
            }
        }
        
        # 加入佇列和歷史
        if self.enable_realtime:
            self.operation_queue.put(operation_record)
        else:
            self.operation_history.append(operation_record)
        
        # 更新統計
        self._update_stats(operation_record)
        
        # 記錄日誌
        status = "✅" if success else "❌"
        self.logger.info(
            f"{status} [{operation_type.upper()}] {command}: {description}"
        )
        
        return operation_id
    
    def record_ai_decision(
        self, 
        decision: str, 
        context: Dict[str, Any] = None,
        confidence: float = None,
        reasoning: str = None
    ) -> str:
        """記錄 AI 決策操作"""
        parameters = {
            "context": context or {},
            "confidence": confidence,
            "reasoning": reasoning
        }
        
        return self.record_operation(
            command="ai_decision",
            description=f"AI 決策: {decision}",
            parameters=parameters,
            operation_type="decision"
        )
    
    def record_attack_step(
        self,
        step_name: str,
        target: str,
        tool: str = None,
        parameters: Dict[str, Any] = None,
        result: Dict[str, Any] = None,
        duration: float = None,
        success: bool = True
    ) -> str:
        """記錄攻擊步驟操作"""
        step_params = {
            "target": target,
            "tool": tool,
            "step_parameters": parameters or {}
        }
        
        return self.record_operation(
            command=step_name,
            description=f"執行攻擊步驟: {step_name} -> {target}",
            parameters=step_params,
            operation_type="attack_step",
            result=result,
            duration=duration,
            success=success
        )
    
    def record_scan_operation(
        self,
        scan_type: str,
        target: str,
        results: Dict[str, Any] = None,
        duration: float = None
    ) -> str:
        """記錄掃描操作"""
        parameters = {
            "scan_type": scan_type,
            "target": target
        }
        
        return self.record_operation(
            command=f"scan_{scan_type}",
            description=f"執行 {scan_type} 掃描: {target}",
            parameters=parameters,
            operation_type="scan",
            result=results,
            duration=duration,
            success=results is not None
        )
    
    def record_training_cycle(
        self,
        cycle_number: int,
        improvements: Dict[str, Any] = None,
        duration: float = None
    ) -> str:
        """記錄訓練週期"""
        parameters = {
            "cycle_number": cycle_number,
            "improvements": improvements or {}
        }
        
        return self.record_operation(
            command="training_cycle",
            description=f"完成訓練週期 #{cycle_number}",
            parameters=parameters,
            operation_type="training",
            result=improvements,
            duration=duration
        )
    
    def start_recording(self):
        """啟動實時記錄"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.logger.info(f"🎬 開始實時記錄 - 會話 ID: {self.session_id}")
    
    def stop_recording(self):
        """停止實時記錄"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
        
        # 處理剩餘的操作
        self._flush_queue()
        
        self.logger.info("⏹️  實時記錄已停止")
    
    def _recording_worker(self):
        """實時記錄工作線程"""
        while self.is_recording:
            try:
                # 從佇列獲取操作記錄
                operation = self.operation_queue.get(timeout=1)
                self.operation_history.append(operation)
                
                # 即時寫入檔案
                self._write_operation_to_file(operation)
                
                self.operation_queue.task_done()
                
            except Empty:
                continue  # 繼續等待
            except Exception as e:
                self.logger.error(f"記錄工作線程錯誤: {e}")
    
    def _flush_queue(self):
        """清空佇列中的剩餘操作"""
        try:
            while True:
                operation = self.operation_queue.get_nowait()
                self.operation_history.append(operation)
                self._write_operation_to_file(operation)
        except Empty:
            pass
    
    def _write_operation_to_file(self, operation: Dict[str, Any]):
        """將操作寫入即時檔案"""
        try:
            realtime_file = self.output_dir / f"{self.session_id}_realtime.jsonl"
            
            with open(realtime_file, 'a', encoding='utf-8') as f:
                json.dump(operation, f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"寫入即時檔案失敗: {e}")
    
    def _update_stats(self, operation: Dict[str, Any]):
        """更新統計資料"""
        self.stats["total_operations"] += 1
        self.stats["last_operation"] = operation["timestamp"]
        
        op_type = operation["operation_type"]
        self.stats["operations_by_type"][op_type] = (
            self.stats["operations_by_type"].get(op_type, 0) + 1
        )
    
    def get_operations_by_type(self, operation_type: str) -> List[Dict[str, Any]]:
        """獲取指定類型的操作記錄"""
        return [op for op in self.operation_history if op["operation_type"] == operation_type]
    
    def get_recent_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取最近的操作記錄"""
        return self.operation_history[-limit:] if self.operation_history else []
    
    def get_operations_in_timerange(
        self, 
        start_time: datetime, 
        end_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """獲取時間範圍內的操作記錄"""
        if end_time is None:
            end_time = datetime.now()
        
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        return [
            op for op in self.operation_history 
            if start_iso <= op["timestamp"] <= end_iso
        ]
    
    def export_session_report(self, output_path: str = None) -> str:
        """匯出會話報告"""
        if not output_path:
            output_path = self.output_dir / f"{self.session_id}_report.json"
        
        report = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": self.stats["session_start"],
                "end_time": datetime.now().isoformat(),
                "total_operations": len(self.operation_history)
            },
            "statistics": self.stats,
            "operations": self.operation_history,
            "summary": self._generate_summary()
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 會話報告已匯出: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"匯出報告失敗: {e}")
            return ""
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成會話摘要"""
        if not self.operation_history:
            return {"message": "無操作記錄"}
        
        # 成功率統計
        successful_ops = sum(1 for op in self.operation_history if op.get("success", True))
        success_rate = successful_ops / len(self.operation_history) * 100
        
        # 平均執行時間
        durations = [op.get("duration", 0) for op in self.operation_history if op.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # 最常用操作
        command_counts = {}
        for op in self.operation_history:
            cmd = op.get("command", "unknown")
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        most_used = max(command_counts.items(), key=lambda x: x[1]) if command_counts else ("無", 0)
        
        return {
            "operation_count": len(self.operation_history),
            "success_rate": f"{success_rate:.1f}%",
            "average_duration": f"{avg_duration:.2f}s",
            "most_used_command": f"{most_used[0]} ({most_used[1]} 次)",
            "operation_types": dict(self.stats["operations_by_type"]),
            "session_duration": self._calculate_session_duration()
        }
    
    def _calculate_session_duration(self) -> str:
        """計算會話持續時間"""
        if not self.operation_history:
            return "0s"
        
        start_time = datetime.fromisoformat(self.stats["session_start"])
        end_time = datetime.now()
        duration = end_time - start_time
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def get_frontend_data(self) -> Dict[str, Any]:
        """獲取前端可用的格式化資料"""
        recent_ops = self.get_recent_operations(20)
        
        return {
            "session_id": self.session_id,
            "current_stats": self.stats,
            "recent_operations": [
                {
                    "id": op["operation_id"],
                    "timestamp": op["timestamp"],
                    "command": op["command"],
                    "description": op["description"],
                    "type": op["operation_type"],
                    "success": op.get("success", True),
                    "duration": op.get("duration")
                }
                for op in recent_ops
            ],
            "real_time_status": {
                "is_recording": self.is_recording,
                "queue_size": self.operation_queue.qsize() if self.enable_realtime else 0,
                "last_update": datetime.now().isoformat()
            }
        }

# 使用範例和測試
def demo_operation_recorder():
    """示範 AI 操作記錄器功能"""
    print("📊 AIVA AI 操作記錄器示範")
    print("=" * 50)
    
    # 創建記錄器
    recorder = AIOperationRecorder(output_dir="demo_logs")
    
    # 模擬 AI 操作序列
    print("🎬 開始記錄 AI 操作...")
    
    # 1. 記錄決策
    recorder.record_ai_decision(
        decision="選擇 Web 應用掃描策略",
        confidence=0.85,
        reasoning="目標開放 80/443 端口，適合 Web 掃描"
    )
    
    # 2. 記錄掃描操作
    import time
    scan_start = time.time()
    time.sleep(1)  # 模擬掃描時間
    
    recorder.record_scan_operation(
        scan_type="port_scan",
        target="192.168.1.100",
        results={"open_ports": [80, 443, 22], "services": ["http", "https", "ssh"]},
        duration=time.time() - scan_start
    )
    
    # 3. 記錄攻擊步驟
    attack_start = time.time()
    time.sleep(0.5)
    
    recorder.record_attack_step(
        step_name="sql_injection_test",
        target="http://192.168.1.100/login",
        tool="sqlmap",
        parameters={"payload": "' OR 1=1--", "method": "POST"},
        result={"vulnerable": True, "injection_type": "boolean"},
        duration=time.time() - attack_start
    )
    
    # 4. 記錄訓練週期
    recorder.record_training_cycle(
        cycle_number=1,
        improvements={"accuracy": 0.02, "speed": 0.15},
        duration=5.2
    )
    
    # 5. 一般操作記錄
    recorder.record_operation(
        command="save_state",
        description="保存 AI 學習狀態",
        operation_type="system",
        result={"saved": True, "file": "ai_state_20251018.pkl"}
    )
    
    print(f"✅ 已記錄 {recorder.stats['total_operations']} 個操作")
    
    # 顯示統計
    print("\n📈 統計資料:")
    for op_type, count in recorder.stats["operations_by_type"].items():
        print(f"   {op_type}: {count} 次")
    
    # 顯示最近操作
    recent = recorder.get_recent_operations(3)
    print(f"\n🕒 最近 3 個操作:")
    for op in recent:
        print(f"   {op['timestamp'][-8:]} - {op['command']}: {op['description']}")
    
    # 獲取前端資料
    frontend_data = recorder.get_frontend_data()
    print(f"\n🌐 前端資料準備就緒:")
    print(f"   會話 ID: {frontend_data['session_id']}")
    print(f"   最近操作數: {len(frontend_data['recent_operations'])}")
    print(f"   記錄狀態: {'進行中' if frontend_data['real_time_status']['is_recording'] else '已停止'}")
    
    # 匯出報告
    time.sleep(2)  # 確保所有操作都被處理
    recorder.stop_recording()
    
    report_path = recorder.export_session_report()
    if report_path:
        print(f"\n📄 會話報告: {report_path}")

if __name__ == "__main__":
    demo_operation_recorder()