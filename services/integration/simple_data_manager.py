"""
简单数据管理器 - 整合模块数据存储
==========================================

设计原则:
1. 简单：JSON 文件存储，不用考虑持久性
2. 时间戳：所有数据都有时间
3. 分类：按能力类型分类（XSS, SQLi, SSRF 等）
4. 追加：不断更新，不删除历史数据

数据流:
- app.py → 存储任务数据
- 学习系统 → 读取完整历史数据
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SimpleDataManager:
    """简单数据管理器 - 按能力分类存储"""
    
    def __init__(self, base_dir: str = "services/integration/data/experiences"):
        """
        初始化数据管理器
        
        Args:
            base_dir: 数据存储基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SimpleDataManager initialized at {self.base_dir}")
    
    def save_task_data(
        self,
        capability: str,  # 能力类型: xss, sqli, ssrf, phase0, phase1 等
        task_id: str,
        target: str,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        result_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        保存任务数据（按能力分类）
        
        Args:
            capability: 能力类型（xss, sqli, ssrf, phase0 等）
            task_id: 任务 ID
            target: 目标 URL
            request_data: 请求数据
            response_data: 响应数据
            result_data: 结果数据
            metadata: 额外元数据
            
        Returns:
            是否成功保存
        """
        try:
            # 按能力类型创建文件
            capability_file = self.base_dir / f"{capability}.jsonl"
            
            # 构建记录（包含时间戳）
            record = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "capability": capability,
                "target": target,
                "request": request_data,
                "response": response_data,
                "result": result_data,
                "metadata": metadata or {}
            }
            
            # 追加到文件（JSONL 格式：每行一个 JSON）
            with open(capability_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            logger.debug(f"Saved {capability} task data: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save task data: {e}")
            return False
    
    def load_capability_data(
        self,
        capability: str,
        limit: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        读取指定能力的历史数据
        
        Args:
            capability: 能力类型
            limit: 限制返回数量
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            
        Returns:
            记录列表
        """
        try:
            capability_file = self.base_dir / f"{capability}.jsonl"
            
            if not capability_file.exists():
                logger.warning(f"No data file for capability: {capability}")
                return []
            
            records = []
            with open(capability_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    record = json.loads(line)
                    
                    # 时间过滤
                    if start_time or end_time:
                        record_time = datetime.fromisoformat(record["timestamp"])
                        if start_time and record_time < start_time:
                            continue
                        if end_time and record_time > end_time:
                            continue
                    
                    records.append(record)
            
            # 限制数量（返回最新的）
            if limit and len(records) > limit:
                records = records[-limit:]
            
            logger.info(f"Loaded {len(records)} records for {capability}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to load capability data: {e}")
            return []
    
    def load_all_data(
        self,
        limit_per_capability: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        读取所有能力的数据
        
        Args:
            limit_per_capability: 每个能力限制数量
            
        Returns:
            按能力分类的数据字典
        """
        all_data = {}
        
        # 遍历所有 .jsonl 文件
        for jsonl_file in self.base_dir.glob("*.jsonl"):
            capability = jsonl_file.stem
            data = self.load_capability_data(capability, limit=limit_per_capability)
            if data:
                all_data[capability] = data
        
        total_records = sum(len(records) for records in all_data.values())
        logger.info(
            f"Loaded {total_records} total records across "
            f"{len(all_data)} capabilities"
        )
        
        return all_data
    
    def get_statistics(self) -> dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_capabilities": 0,
            "total_records": 0,
            "capabilities": {}
        }
        
        for jsonl_file in self.base_dir.glob("*.jsonl"):
            capability = jsonl_file.stem
            
            # 统计行数
            with open(jsonl_file, encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            
            stats["capabilities"][capability] = {
                "count": count,
                "file": str(jsonl_file)
            }
            stats["total_records"] += count
        
        stats["total_capabilities"] = len(stats["capabilities"])
        
        return stats


# 全局单例
_data_manager = None

def get_data_manager() -> SimpleDataManager:
    """获取全局数据管理器实例"""
    global _data_manager
    if _data_manager is None:
        _data_manager = SimpleDataManager()
    return _data_manager
