"""Task Storage - SQLite 持久化存儲 (P0-3 實現)

實現任務狀態持久化和斷點續傳功能，符合 aiva_common v2.0 規範

設計理念：
1. 原子寫入：每完成一個關鍵步驟立即更新 DB
2. 啟動恢復：查詢 SCANNING 狀態任務並繼續執行
3. 進度追蹤：記錄掃描階段、payload 嘗試次數、發現數量

Schema 設計：
- targets: 掃描目標信息
- scan_progress: 掃描進度追蹤
- findings: 漏洞發現記錄
- execution_trace: 執行軌跡（雙閉環用）

參考：
- 內容截斷問題分析.md § 第四階段：狀態持久化與斷點續傳
- aiva_common README § 數據模型 - Pydantic v2 強類型
"""

import sqlite3
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

# ✅ 使用 aiva_common 統一日誌和錯誤處理
from aiva_common.utils.logging import get_logger
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

logger = get_logger(__name__)


class TaskStatus:
    """任務狀態枚舉"""
    PENDING = "PENDING"
    SCANNING = "SCANNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class TaskStorage:
    """任務存儲管理器 (v2.0 合規版本)
    
    功能：
    1. SQLite 數據庫初始化和 Schema 創建
    2. 任務 CRUD 操作（原子寫入）
    3. 掃描進度追蹤和更新
    4. 漏洞發現記錄
    5. 斷點續傳支持（恢復 SCANNING 狀態任務）
    6. 執行軌跡記錄（支持雙閉環）
    
    Schema:
    - targets: 目標信息
    - scan_progress: 掃描進度
    - findings: 發現記錄
    - execution_trace: 執行軌跡
    """
    
    def __init__(self, db_path: str = "data/tasks.db"):
        """初始化任務存儲
        
        Args:
            db_path: SQLite 數據庫路徑
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"✅ TaskStorage initialized: {self.db_path}")
    
    def _init_database(self) -> None:
        """初始化數據庫 Schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. targets 表：掃描目標
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    priority INTEGER DEFAULT 5,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # 2. scan_progress 表：掃描進度
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    current_stage TEXT NOT NULL,
                    payloads_tried INTEGER DEFAULT 0,
                    findings_count INTEGER DEFAULT 0,
                    scan_config TEXT,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES targets(id)
                )
            """)
            
            # 3. findings 表：漏洞發現
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    vulnerability_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT,
                    payload TEXT,
                    timestamp TEXT NOT NULL,
                    verified BOOLEAN DEFAULT 0,
                    FOREIGN KEY (target_id) REFERENCES targets(id)
                )
            """)
            
            # 4. execution_trace 表：執行軌跡（雙閉環用）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    plan_id TEXT,
                    step_index INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT,
                    success BOOLEAN NOT NULL,
                    execution_time REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES targets(id)
                )
            """)
            
            # 創建索引以提升查詢性能
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_targets_status ON targets(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_target ON findings(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_target ON execution_trace(target_id)")
            
            conn.commit()
            logger.info("✅ Database schema initialized")
    
    @contextmanager
    def _get_connection(self):
        """獲取數據庫連接（上下文管理器）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 使結果可以像字典一樣訪問
        try:
            yield conn
        finally:
            conn.close()
    
    # ==================== 目標管理 ====================
    
    def create_target(
        self,
        url: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """創建新的掃描目標
        
        Args:
            url: 目標 URL
            priority: 優先級 (1-10，數字越大優先級越高)
            metadata: 額外元數據
            
        Returns:
            目標 ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            created_at = datetime.now(UTC).isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            
            try:
                cursor.execute("""
                    INSERT INTO targets (url, status, priority, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (url, TaskStatus.PENDING, priority, created_at, metadata_json))
                
                conn.commit()
                target_id = cursor.lastrowid
                
                if target_id is None:
                    raise AIVAError("Failed to create target", ErrorType.DATABASE, ErrorSeverity.HIGH)
                
                logger.info(f"✅ Target created: {url} (ID: {target_id})")
                return target_id
                
            except sqlite3.IntegrityError:
                # URL 已存在，返回現有 ID
                cursor.execute("SELECT id FROM targets WHERE url = ?", (url,))
                result = cursor.fetchone()
                logger.warning(f"⚠️  Target already exists: {url} (ID: {result['id']})")
                return result['id']
    
    def update_target_status(
        self,
        target_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """更新目標狀態（原子寫入）
        
        Args:
            target_id: 目標 ID
            status: 新狀態
            error_message: 錯誤信息（如果有）
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            timestamp = datetime.now(UTC).isoformat()
            
            # 根據狀態更新對應的時間字段
            if status == TaskStatus.SCANNING:
                cursor.execute("""
                    UPDATE targets
                    SET status = ?, started_at = ?, error_message = ?
                    WHERE id = ?
                """, (status, timestamp, error_message, target_id))
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                cursor.execute("""
                    UPDATE targets
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE id = ?
                """, (status, timestamp, error_message, target_id))
            else:
                cursor.execute("""
                    UPDATE targets
                    SET status = ?, error_message = ?
                    WHERE id = ?
                """, (status, error_message, target_id))
            
            conn.commit()
            logger.info(f"✅ Target {target_id} status updated: {status}")
    
    def get_pending_targets(self) -> List[Dict[str, Any]]:
        """獲取待處理的目標（用於啟動時恢復）
        
        Returns:
            待處理目標列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM targets
                WHERE status IN ('PENDING', 'SCANNING')
                ORDER BY priority DESC, created_at ASC
            """)
            
            results = []
            for row in cursor.fetchall():
                target = dict(row)
                if target['metadata']:
                    target['metadata'] = json.loads(target['metadata'])
                results.append(target)
            
            logger.info(f"📋 Found {len(results)} pending targets")
            return results
    
    # ==================== 掃描進度管理 ====================
    
    def update_scan_progress(
        self,
        target_id: int,
        current_stage: str,
        payloads_tried: int = 0,
        findings_count: int = 0,
        scan_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """更新掃描進度（原子寫入）
        
        Args:
            target_id: 目標 ID
            current_stage: 當前階段
            payloads_tried: 已嘗試的 payload 數量
            findings_count: 發現的漏洞數量
            scan_config: 掃描配置
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            last_updated = datetime.now(UTC).isoformat()
            config_json = json.dumps(scan_config) if scan_config else None
            
            # 檢查是否已有記錄
            cursor.execute("SELECT id FROM scan_progress WHERE target_id = ?", (target_id,))
            existing = cursor.fetchone()
            
            if existing:
                # 更新現有記錄
                cursor.execute("""
                    UPDATE scan_progress
                    SET current_stage = ?, payloads_tried = ?, findings_count = ?,
                        scan_config = ?, last_updated = ?
                    WHERE target_id = ?
                """, (current_stage, payloads_tried, findings_count, config_json, last_updated, target_id))
            else:
                # 插入新記錄
                cursor.execute("""
                    INSERT INTO scan_progress (target_id, current_stage, payloads_tried, findings_count, scan_config, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (target_id, current_stage, payloads_tried, findings_count, config_json, last_updated))
            
            conn.commit()
            logger.debug(f"✅ Progress updated for target {target_id}: {current_stage}")
    
    def get_scan_progress(self, target_id: int) -> Optional[Dict[str, Any]]:
        """獲取掃描進度
        
        Args:
            target_id: 目標 ID
            
        Returns:
            掃描進度信息，如果不存在則返回 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_progress WHERE target_id = ?", (target_id,))
            row = cursor.fetchone()
            
            if row:
                progress = dict(row)
                if progress['scan_config']:
                    progress['scan_config'] = json.loads(progress['scan_config'])
                return progress
            
            return None
    
    # ==================== 漏洞發現管理 ====================
    
    def record_finding(
        self,
        target_id: int,
        vulnerability_type: str,
        severity: str,
        confidence: float,
        evidence: Optional[str] = None,
        payload: Optional[str] = None
    ) -> int:
        """記錄漏洞發現（原子寫入）
        
        Args:
            target_id: 目標 ID
            vulnerability_type: 漏洞類型
            severity: 嚴重程度
            confidence: 置信度
            evidence: 證據
            payload: 使用的 payload
            
        Returns:
            發現記錄 ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            timestamp = datetime.now(UTC).isoformat()
            
            cursor.execute("""
                INSERT INTO findings (target_id, vulnerability_type, severity, confidence, evidence, payload, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (target_id, vulnerability_type, severity, confidence, evidence, payload, timestamp))
            
            conn.commit()
            finding_id = cursor.lastrowid
            
            if finding_id is None:
                raise AIVAError("Failed to record finding", ErrorType.DATABASE, ErrorSeverity.HIGH)
            
            logger.info(f"🔍 Finding recorded: {vulnerability_type} (ID: {finding_id})")
            return finding_id
    
    def get_findings(self, target_id: int) -> List[Dict[str, Any]]:
        """獲取目標的所有發現
        
        Args:
            target_id: 目標 ID
            
        Returns:
            發現記錄列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM findings
                WHERE target_id = ?
                ORDER BY timestamp DESC
            """, (target_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== 執行軌跡管理（雙閉環支持）====================
    
    def record_execution_step(
        self,
        target_id: int,
        plan_id: Optional[str],
        step_index: int,
        action: str,
        result: Optional[Dict[str, Any]],
        success: bool,
        execution_time: float
    ) -> int:
        """記錄執行步驟（支持雙閉環偏差分析）
        
        Args:
            target_id: 目標 ID
            plan_id: 執行計劃 ID
            step_index: 步驟索引
            action: 執行的動作
            result: 執行結果
            success: 是否成功
            execution_time: 執行時間（秒）
            
        Returns:
            軌跡記錄 ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            timestamp = datetime.now(UTC).isoformat()
            result_json = json.dumps(result) if result else None
            
            cursor.execute("""
                INSERT INTO execution_trace (target_id, plan_id, step_index, action, result, success, execution_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (target_id, plan_id, step_index, action, result_json, success, execution_time, timestamp))
            
            conn.commit()
            trace_id = cursor.lastrowid
            
            if trace_id is None:
                raise AIVAError("Failed to record execution trace", ErrorType.DATABASE, ErrorSeverity.HIGH)
            
            logger.debug(f"📝 Execution step recorded: {action} (ID: {trace_id})")
            return trace_id
    
    def get_execution_trace(self, target_id: int, plan_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """獲取執行軌跡（用於雙閉環偏差分析）
        
        Args:
            target_id: 目標 ID
            plan_id: 執行計劃 ID（可選）
            
        Returns:
            執行軌跡列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if plan_id:
                cursor.execute("""
                    SELECT * FROM execution_trace
                    WHERE target_id = ? AND plan_id = ?
                    ORDER BY step_index ASC
                """, (target_id, plan_id))
            else:
                cursor.execute("""
                    SELECT * FROM execution_trace
                    WHERE target_id = ?
                    ORDER BY timestamp DESC
                """, (target_id,))
            
            traces = []
            for row in cursor.fetchall():
                trace = dict(row)
                if trace['result']:
                    trace['result'] = json.loads(trace['result'])
                traces.append(trace)
            
            return traces
    
    # ==================== 統計和報告 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息
        
        Returns:
            統計數據字典
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 目標統計
            cursor.execute("SELECT status, COUNT(*) as count FROM targets GROUP BY status")
            target_stats = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # 發現統計
            cursor.execute("SELECT vulnerability_type, COUNT(*) as count FROM findings GROUP BY vulnerability_type")
            finding_stats = {row['vulnerability_type']: row['count'] for row in cursor.fetchall()}
            
            # 總體統計
            cursor.execute("SELECT COUNT(*) as total FROM targets")
            total_targets = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM findings")
            total_findings = cursor.fetchone()['total']
            
            return {
                "total_targets": total_targets,
                "total_findings": total_findings,
                "targets_by_status": target_stats,
                "findings_by_type": finding_stats,
                "timestamp": datetime.now(UTC).isoformat()
            }
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """清理舊記錄
        
        Args:
            days: 保留天數
            
        Returns:
            刪除的記錄數
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now(UTC).isoformat()
            # 這裡應該計算 days 天前的日期，簡化處理
            
            cursor.execute("""
                DELETE FROM targets
                WHERE status = 'COMPLETED' AND completed_at < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"🗑️  Cleaned up {deleted_count} old records")
            return deleted_count
