"""指令儲存庫

管理CLI指令執行歷史的存儲和查詢,支持:
- 指令執行記錄存儲
- 歷史查詢和統計
- 性能分析
- 成功率追蹤
"""

from datetime import UTC, datetime, timedelta
import logging
from typing import Any

from sqlalchemy import and_, desc, func

from .backends import StorageBackend
from .models import Base, CommandExecutionModel

logger = logging.getLogger(__name__)


class CommandRepository:
    """指令儲存庫
    
    基於 StorageBackend 提供CLI指令專用的存儲接口
    """
    
    def __init__(self, backend: StorageBackend):
        """初始化指令儲存庫
        
        Args:
            backend: 存儲後端實例
        """
        self.backend = backend
        
        # 如果是 SQLite 或 PostgreSQL backend,初始化表
        if hasattr(backend, 'engine'):
            Base.metadata.create_all(backend.engine)
            logger.info("CommandRepository initialized with database backend")
        else:
            logger.warning("CommandRepository initialized with non-database backend")
    
    async def save_command_execution(
        self,
        command_id: str,
        command_name: str,
        command_type: str,
        capability_endpoint: str,
        flow_path: list[str],
        flow_length: int,
        primary_module: str,
        modules_involved: list[str],
        status: str,
        success: bool,
        execution_time_ms: float,
        flow_id: str | None = None,
        flow_preference: str = "balanced",
        parameters: dict[str, Any] | None = None,
        result_data: dict[str, Any] | None = None,
        error_message: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        is_ai_generated: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """保存指令執行記錄
        
        Args:
            command_id: 指令 ID
            command_name: 指令名稱
            command_type: 指令類型 (exec, query, flows等)
            capability_endpoint: 能力終點
            flow_path: 執行路徑
            flow_length: 路徑長度
            primary_module: 主要模組
            modules_involved: 涉及的所有模組
            status: 執行狀態
            success: 是否成功
            execution_time_ms: 執行時間(毫秒)
            flow_id: 數據流 ID
            flow_preference: 流程偏好
            parameters: 執行參數
            result_data: 結果數據
            error_message: 錯誤訊息
            user_id: 用戶 ID
            session_id: 會話 ID
            is_ai_generated: 是否AI生成
            metadata: 額外元數據
            
        Returns:
            是否保存成功
        """
        if not hasattr(self.backend, 'SessionLocal'):
            logger.warning("Backend does not support command execution storage")
            return False
        
        session = self.backend.SessionLocal()
        try:
            record = CommandExecutionModel(
                command_id=command_id,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                command_name=command_name,
                command_type=command_type,
                capability_endpoint=capability_endpoint,
                flow_id=flow_id,
                flow_path=flow_path,
                flow_length=flow_length,
                flow_preference=flow_preference,
                primary_module=primary_module,
                modules_involved=modules_involved,
                parameters=parameters,
                status=status,
                success=success,
                result_data=result_data,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                user_id=user_id,
                session_id=session_id,
                is_ai_generated=is_ai_generated,
                extra_metadata=metadata,
            )
            
            session.add(record)
            session.commit()
            
            logger.debug(f"Saved command execution: {command_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save command execution: {e}")
            return False
        finally:
            session.close()
    
    async def get_command_history(
        self,
        limit: int = 50,
        offset: int = 0,
        command_name: str | None = None,
        capability_endpoint: str | None = None,
        status: str | None = None,
        success: bool | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        primary_module: str | None = None,
        flow_preference: str | None = None,
        is_ai_generated: bool | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """查詢指令執行歷史
        
        Args:
            limit: 返回記錄數
            offset: 偏移量
            command_name: 篩選指令名稱
            capability_endpoint: 篩選能力終點
            status: 篩選狀態
            success: 篩選成功/失敗
            user_id: 篩選用戶
            session_id: 篩選會話
            primary_module: 篩選主要模組
            flow_preference: 篩選流程偏好
            is_ai_generated: 篩選AI生成
            start_date: 起始日期
            end_date: 結束日期
            
        Returns:
            指令執行記錄列表
        """
        if not hasattr(self.backend, 'SessionLocal'):
            logger.warning("Backend does not support command history query")
            return []
        
        session = self.backend.SessionLocal()
        try:
            query = session.query(CommandExecutionModel)
            
            # 應用篩選條件
            if command_name:
                query = query.filter(CommandExecutionModel.command_name == command_name)
            if capability_endpoint:
                query = query.filter(CommandExecutionModel.capability_endpoint == capability_endpoint)
            if status:
                query = query.filter(CommandExecutionModel.status == status)
            if success is not None:
                query = query.filter(CommandExecutionModel.success == success)
            if user_id:
                query = query.filter(CommandExecutionModel.user_id == user_id)
            if session_id:
                query = query.filter(CommandExecutionModel.session_id == session_id)
            if primary_module:
                query = query.filter(CommandExecutionModel.primary_module == primary_module)
            if flow_preference:
                query = query.filter(CommandExecutionModel.flow_preference == flow_preference)
            if is_ai_generated is not None:
                query = query.filter(CommandExecutionModel.is_ai_generated == is_ai_generated)
            if start_date:
                query = query.filter(CommandExecutionModel.created_at >= start_date)
            if end_date:
                query = query.filter(CommandExecutionModel.created_at <= end_date)
            
            # 排序和分頁
            query = query.order_by(desc(CommandExecutionModel.created_at))
            query = query.offset(offset).limit(limit)
            
            records = query.all()
            
            return [
                {
                    "command_id": r.command_id,
                    "created_at": r.created_at.isoformat(),
                    "command_name": r.command_name,
                    "command_type": r.command_type,
                    "capability_endpoint": r.capability_endpoint,
                    "flow_id": r.flow_id,
                    "flow_path": r.flow_path,
                    "flow_length": r.flow_length,
                    "flow_preference": r.flow_preference,
                    "primary_module": r.primary_module,
                    "modules_involved": r.modules_involved,
                    "parameters": r.parameters,
                    "status": r.status,
                    "success": r.success,
                    "result_data": r.result_data,
                    "error_message": r.error_message,
                    "execution_time_ms": r.execution_time_ms,
                    "user_id": r.user_id,
                    "session_id": r.session_id,
                    "is_ai_generated": r.is_ai_generated,
                    "metadata": r.extra_metadata,
                }
                for r in records
            ]
            
        except Exception as e:
            logger.error(f"Failed to query command history: {e}")
            return []
        finally:
            session.close()
    
    async def get_command_statistics(
        self,
        capability_endpoint: str | None = None,
        primary_module: str | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        """獲取指令統計信息
        
        Args:
            capability_endpoint: 篩選能力終點
            primary_module: 篩選主要模組
            days: 統計天數
            
        Returns:
            統計信息字典
        """
        if not hasattr(self.backend, 'SessionLocal'):
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
            }
        
        session = self.backend.SessionLocal()
        try:
            # 時間範圍
            start_date = datetime.now(UTC) - timedelta(days=days)
            
            query = session.query(CommandExecutionModel).filter(
                CommandExecutionModel.created_at >= start_date
            )
            
            # 應用篩選
            if capability_endpoint:
                query = query.filter(CommandExecutionModel.capability_endpoint == capability_endpoint)
            if primary_module:
                query = query.filter(CommandExecutionModel.primary_module == primary_module)
            
            # 總數
            total = query.count()
            
            # 成功/失敗
            successful = query.filter(CommandExecutionModel.success == True).count()
            failed = query.filter(CommandExecutionModel.success == False).count()
            
            # 成功率
            success_rate = (successful / total * 100) if total > 0 else 0.0
            
            # 平均執行時間
            avg_time = session.query(
                func.avg(CommandExecutionModel.execution_time_ms)
            ).filter(
                CommandExecutionModel.created_at >= start_date
            )
            
            if capability_endpoint:
                avg_time = avg_time.filter(CommandExecutionModel.capability_endpoint == capability_endpoint)
            if primary_module:
                avg_time = avg_time.filter(CommandExecutionModel.primary_module == primary_module)
            
            avg_execution_time = avg_time.scalar() or 0.0
            
            # 按流程偏好統計
            preference_stats = {}
            for pref in ["fastest", "balanced", "complete"]:
                count = query.filter(CommandExecutionModel.flow_preference == pref).count()
                preference_stats[pref] = count
            
            # 按模組統計
            module_stats = session.query(
                CommandExecutionModel.primary_module,
                func.count(CommandExecutionModel.id).label("count")
            ).filter(
                CommandExecutionModel.created_at >= start_date
            ).group_by(
                CommandExecutionModel.primary_module
            ).all()
            
            return {
                "total_executions": total,
                "successful_executions": successful,
                "failed_executions": failed,
                "success_rate": round(success_rate, 2),
                "avg_execution_time_ms": round(avg_execution_time, 2),
                "preference_distribution": preference_stats,
                "module_distribution": {m: c for m, c in module_stats},
                "period_days": days,
            }
            
        except Exception as e:
            logger.error(f"Failed to get command statistics: {e}")
            return {
                "error": str(e),
                "total_executions": 0,
            }
        finally:
            session.close()
    
    async def get_popular_capabilities(
        self,
        limit: int = 10,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """獲取最常用的能力
        
        Args:
            limit: 返回數量
            days: 統計天數
            
        Returns:
            能力使用統計列表
        """
        if not hasattr(self.backend, 'SessionLocal'):
            return []
        
        session = self.backend.SessionLocal()
        try:
            start_date = datetime.now(UTC) - timedelta(days=days)
            
            results = session.query(
                CommandExecutionModel.capability_endpoint,
                CommandExecutionModel.primary_module,
                func.count(CommandExecutionModel.id).label("usage_count"),
                func.avg(CommandExecutionModel.execution_time_ms).label("avg_time"),
                func.sum(
                    func.cast(CommandExecutionModel.success, func.INTEGER())
                ).label("success_count"),
            ).filter(
                CommandExecutionModel.created_at >= start_date
            ).group_by(
                CommandExecutionModel.capability_endpoint,
                CommandExecutionModel.primary_module
            ).order_by(
                desc("usage_count")
            ).limit(limit).all()
            
            return [
                {
                    "capability_endpoint": r.capability_endpoint,
                    "primary_module": r.primary_module,
                    "usage_count": r.usage_count,
                    "avg_execution_time_ms": round(r.avg_time, 2),
                    "success_rate": round((r.success_count / r.usage_count * 100), 2),
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get popular capabilities: {e}")
            return []
        finally:
            session.close()
    
    async def get_slow_executions(
        self,
        threshold_ms: float = 1000.0,
        limit: int = 20,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """獲取執行緩慢的指令
        
        Args:
            threshold_ms: 時間閾值(毫秒)
            limit: 返回數量
            days: 統計天數
            
        Returns:
            緩慢執行記錄列表
        """
        if not hasattr(self.backend, 'SessionLocal'):
            return []
        
        session = self.backend.SessionLocal()
        try:
            start_date = datetime.now(UTC) - timedelta(days=days)
            
            records = session.query(CommandExecutionModel).filter(
                and_(
                    CommandExecutionModel.created_at >= start_date,
                    CommandExecutionModel.execution_time_ms >= threshold_ms
                )
            ).order_by(
                desc(CommandExecutionModel.execution_time_ms)
            ).limit(limit).all()
            
            return [
                {
                    "command_id": r.command_id,
                    "capability_endpoint": r.capability_endpoint,
                    "flow_path": r.flow_path,
                    "flow_length": r.flow_length,
                    "execution_time_ms": r.execution_time_ms,
                    "created_at": r.created_at.isoformat(),
                    "success": r.success,
                }
                for r in records
            ]
            
        except Exception as e:
            logger.error(f"Failed to get slow executions: {e}")
            return []
        finally:
            session.close()
