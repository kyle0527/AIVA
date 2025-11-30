"""Capability Registry - 能力註冊中心

統一管理能力元數據，支持雙寫機制 (PostgreSQL + ChromaDB)

Architecture: AIVA v2.0 Capability Metadata Database Design
Created: 2025-01-XX (Day 3 Implementation)
Purpose: 
  - 接收內循環掃描結果
  - 計算內容哈希識別變化
  - 執行增量更新 (新增/修改/刪除)
  - 同步到 PostgreSQL 和 ChromaDB
"""

import hashlib
import json
import logging
from datetime import datetime, UTC
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from services.aiva_common.schemas.dual_loop import (
    ModuleCapability,
    ChangeType,
)
from .models import (
    CapabilityRecord,
    CapabilityVersion,
    CapabilityChangeLog,
    CapabilityInvocationStats,
)

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """能力註冊中心 - 統一管理能力元數據
    
    核心職責:
    1. 接收內循環掃描的能力列表
    2. 基於內容哈希檢測變更 (added/modified/deleted/unchanged)
    3. 執行增量更新到 PostgreSQL
    4. 同步到 ChromaDB 向量數據庫
    5. 記錄變更日誌和統計信息
    """
    
    def __init__(self, pg_session: Session, chroma_collection=None):
        """初始化註冊中心
        
        Args:
            pg_session: SQLAlchemy Session (PostgreSQL)
            chroma_collection: ChromaDB Collection (可選，用於 RAG)
        """
        self.pg_session = pg_session
        self.chroma_collection = chroma_collection
        logger.info("CapabilityRegistry initialized")
    
    def register_capabilities(
        self, 
        capabilities: list[ModuleCapability]
    ) -> dict[str, Any]:
        """註冊/更新能力列表
        
        這是核心方法，內循環掃描完成後調用此方法進行批量註冊。
        
        Args:
            capabilities: 內循環掃描到的能力列表
            
        Returns:
            變更統計: {
                "added": int,
                "modified": int, 
                "deleted": int,
                "unchanged": int,
                "scan_id": str,
                "total_capabilities": int
            }
        """
        scan_id = str(uuid4())
        stats = {
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "unchanged": 0,
            "scan_id": scan_id,
        }
        
        logger.info(f"🔄 Starting capability registration (scan_id: {scan_id})")
        
        try:
            # 1. 獲取現有能力列表
            existing_caps = self._get_all_existing_capabilities()
            existing_keys = {self._make_key(cap): cap for cap in existing_caps}
            
            # 2. 處理新掃描的能力
            scanned_keys = set()
            
            for cap in capabilities:
                key = self._make_key(cap)
                scanned_keys.add(key)
                
                content_hash = self._compute_hash(cap)
                
                if key not in existing_keys:
                    # 新增能力
                    self._add_capability(cap, content_hash)
                    stats["added"] += 1
                    logger.debug(f"✅ Added: {key}")
                    
                elif existing_keys[key]["content_hash"] != content_hash:
                    # 能力已變更
                    self._update_capability(cap, content_hash)
                    stats["modified"] += 1
                    logger.debug(f"🔄 Modified: {key}")
                    
                else:
                    # 無變化
                    stats["unchanged"] += 1
            
            # 3. 檢測已刪除的能力
            deleted_keys = set(existing_keys.keys()) - scanned_keys
            for key in deleted_keys:
                self._mark_deleted(existing_keys[key])
                stats["deleted"] += 1
                logger.debug(f"❌ Deleted: {key}")
            
            # 4. 記錄變更日誌
            stats["total_capabilities"] = len(scanned_keys)
            self._log_scan_result(scan_id, stats)
            
            logger.info(
                f"✅ Registration complete: +{stats['added']} "
                f"~{stats['modified']} -{stats['deleted']} "
                f"={stats['unchanged']} (total: {stats['total_capabilities']})"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}", exc_info=True)
            self.pg_session.rollback()
            raise
    
    def _make_key(self, cap: ModuleCapability | dict) -> str:
        """生成能力唯一標識: module::name::file_path"""
        if isinstance(cap, dict):
            return f"{cap['module']}::{cap['name']}::{cap.get('file_path', '')}"
        return f"{cap.module}::{cap.name}::{cap.file_path or ''}"
    
    def _compute_hash(self, cap: ModuleCapability) -> str:
        """計算能力內容哈希 (排除時間戳等易變欄位)
        
        穩定欄位：
        - name, module, language
        - parameters (name, type, description)
        - return_definition
        - invocation (protocol, endpoint, mapping)
        
        排除：
        - created_at, updated_at
        - version
        - usage_examples (可能經常變動)
        """
        stable_content = {
            "name": cap.name,
            "module": cap.module,
            "language": cap.language,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                }
                for p in cap.parameters
            ],
            "return_info": cap.return_info.model_dump() if cap.return_info else None,
            "invocation": cap.invocation.model_dump() if cap.invocation else None,
        }
        content_str = json.dumps(stable_content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _add_capability(self, cap: ModuleCapability, content_hash: str):
        """新增能力到數據庫"""
        key = self._make_key(cap)
        now = datetime.now(UTC)
        
        # PostgreSQL: 主記錄
        record = CapabilityRecord(
            key=key,
            name=cap.name,
            module=cap.module,
            language=cap.language,
            file_path=cap.file_path or "",
            version=1,
            content_hash=content_hash,
            is_active=True,
            metadata_json=cap.model_dump_json(),
            created_at=now,
            updated_at=now,
        )
        self.pg_session.add(record)
        
        # PostgreSQL: 版本歷史
        version = CapabilityVersion(
            capability_key=key,
            version=1,
            content_hash=content_hash,
            metadata_json=cap.model_dump_json(),
            change_type=ChangeType.ADDED.value,
            created_at=now,
        )
        self.pg_session.add(version)
        
        # 初始化調用統計
        stats = CapabilityInvocationStats(
            capability_key=key,
            invocation_count=0,
            success_count=0,
            failure_count=0,
            updated_at=now,
        )
        self.pg_session.add(stats)
        
        self.pg_session.commit()
        
        # ChromaDB: 向量存儲 (用於 RAG 查詢)
        if self.chroma_collection:
            self._sync_to_chromadb(cap, "add", content_hash)
    
    def _update_capability(
        self, 
        cap: ModuleCapability, 
        content_hash: str
    ):
        """更新能力"""
        key = self._make_key(cap)
        now = datetime.now(UTC)
        
        # 獲取當前版本號
        stmt = select(CapabilityRecord).where(CapabilityRecord.key == key)
        record = self.pg_session.execute(stmt).scalar_one()
        # 使用 getattr 獲取 ORM 對象的屬性值
        current_version = getattr(record, 'version', 1)
        new_version = current_version + 1
        
        # 更新主記錄 - 使用 update 語句
        update_stmt = (
            update(CapabilityRecord)
            .where(CapabilityRecord.key == key)
            .values(
                version=new_version,
                content_hash=content_hash,
                metadata_json=cap.model_dump_json(),
                updated_at=now
            )
        )
        self.pg_session.execute(update_stmt)
        
        # 新增版本記錄
        version = CapabilityVersion(
            capability_key=key,
            version=new_version,
            content_hash=content_hash,
            metadata_json=cap.model_dump_json(),
            change_type=ChangeType.MODIFIED.value,
            created_at=now,
        )
        self.pg_session.add(version)
        self.pg_session.commit()
        
        # 同步到 ChromaDB
        if self.chroma_collection:
            self._sync_to_chromadb(cap, "update", content_hash)
    
    def _mark_deleted(self, existing_record: dict):
        """標記能力為已刪除 (軟刪除)"""
        key = existing_record["key"]
        now = datetime.now(UTC)
        
        # 更新主記錄
        stmt = (
            update(CapabilityRecord)
            .where(CapabilityRecord.key == key)
            .values(is_active=False, updated_at=now)
        )
        self.pg_session.execute(stmt)
        
        # 記錄刪除版本
        version = CapabilityVersion(
            capability_key=key,
            version=existing_record["version"] + 1,
            content_hash="",
            metadata_json="",
            change_type=ChangeType.DELETED.value,
            created_at=now,
        )
        self.pg_session.add(version)
        self.pg_session.commit()
        
        # ChromaDB 刪除
        if self.chroma_collection:
            self._sync_to_chromadb(
                {"key": key, "content_hash": existing_record["content_hash"]},
                "delete",
                existing_record["content_hash"]
            )
    
    def _sync_to_chromadb(
        self, 
        cap: ModuleCapability | dict, 
        operation: str,
        content_hash: str
    ):
        """同步到 ChromaDB 向量數據庫
        
        Args:
            cap: 能力對象或字典
            operation: "add" | "update" | "delete"
            content_hash: 內容哈希 (用於生成 doc_id)
        """
        if not self.chroma_collection:
            return
        
        doc_id = f"cap_{content_hash}"
        
        try:
            if operation in ["add", "update"]:
                # 構建文檔內容
                if isinstance(cap, dict):
                    return  # 刪除操作不需要完整對象
                
                content = self._build_chromadb_document(cap)
                metadata = self._build_chromadb_metadata(cap)
                
                self.chroma_collection.upsert(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[metadata]
                )
                logger.debug(f"✅ ChromaDB synced: {operation} {doc_id}")
                
            elif operation == "delete":
                self.chroma_collection.delete(ids=[doc_id])
                logger.debug(f"❌ ChromaDB deleted: {doc_id}")
                
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB sync failed: {e}")
            # ChromaDB 同步失敗不應阻塞主流程
    
    def _build_chromadb_document(self, cap: ModuleCapability) -> str:
        """構建 ChromaDB 文檔內容"""
        param_lines = "\n".join([
            f"  - {p.name} ({p.type}): {p.description or 'No description'}"
            for p in cap.parameters
        ])
        
        invocation_info = "unknown"
        if cap.invocation:
            invocation_info = (
                f"Protocol: {cap.invocation.protocol}, "
                f"Module: {cap.invocation.module_arg or 'N/A'}, "
                f"Function: {cap.invocation.function_arg or 'N/A'}"
            )
        
        return f"""
Capability: {cap.name}
Module: {cap.module}
Language: {cap.language}
Description: {cap.description or 'No description'}

Parameters:
{param_lines}

Invocation:
  {invocation_info}

File: {cap.file_path or 'unknown'}
Function: {cap.function}
"""
    def _build_chromadb_metadata(self, cap: ModuleCapability) -> dict:
        """構建 ChromaDB 元數據"""
        return {
            "capability_name": cap.name,
            "module": cap.module,
            "language": cap.language,
            "namespace": "self_awareness",
            "type": "capability",
            "invocation_protocol": cap.invocation.protocol if cap.invocation else None,
            "invocation_module": cap.invocation.module_arg if cap.invocation else None,
            "invocation_function": cap.invocation.function_arg if cap.invocation else None,
        }
    
    def _get_all_existing_capabilities(self) -> list[dict]:
        """獲取所有現有能力 (僅活躍的)"""
        stmt = select(
            CapabilityRecord.key,
            CapabilityRecord.name,
            CapabilityRecord.module,
            CapabilityRecord.version,
            CapabilityRecord.content_hash,
        ).where(CapabilityRecord.is_active == True)
        
        results = self.pg_session.execute(stmt).all()
        
        return [
            {
                "key": row.key,
                "name": row.name,
                "module": row.module,
                "version": row.version,
                "content_hash": row.content_hash,
            }
            for row in results
        ]
    
    def _log_scan_result(self, scan_id: str, stats: dict):
        """記錄掃描結果到變更日誌表"""
        log = CapabilityChangeLog(
            scan_id=scan_id,
            scan_timestamp=datetime.now(UTC),
            added_count=stats["added"],
            modified_count=stats["modified"],
            deleted_count=stats["deleted"],
            unchanged_count=stats["unchanged"],
            total_capabilities=stats["total_capabilities"],
            details_json=json.dumps(stats, ensure_ascii=False),
        )
        self.pg_session.add(log)
        self.pg_session.commit()
    
    # ==================== 查詢方法 ====================
    
    def get_capability_by_name(self, name: str) -> ModuleCapability | None:
        """根據名稱查詢能力 (返回 Pydantic 模型)"""
        stmt = select(CapabilityRecord).where(
            CapabilityRecord.name == name,
            CapabilityRecord.is_active == True
        )
        record = self.pg_session.execute(stmt).scalar_one_or_none()
        
        if record:
            # 明確訪問 SQLAlchemy Column 屬性的值
            metadata_str = getattr(record, 'metadata_json', None)
            if metadata_str:
                return ModuleCapability.model_validate_json(metadata_str)
        return None
    
    def get_capability_by_key(self, key: str) -> ModuleCapability | None:
        """根據完整 key 查詢能力"""
        stmt = select(CapabilityRecord).where(
            CapabilityRecord.key == key,
            CapabilityRecord.is_active == True
        )
        record = self.pg_session.execute(stmt).scalar_one_or_none()
        
        if record:
            return ModuleCapability.model_validate_json(str(record.metadata_json))
        return None
    
    def get_capability_history(self, name: str) -> list[CapabilityVersion]:
        """查詢能力變更歷史"""
        key_pattern = f"%::{name}::%"
        stmt = (
            select(CapabilityVersion)
            .where(CapabilityVersion.capability_key.like(key_pattern))
            .order_by(CapabilityVersion.version.desc())
        )
        return list(self.pg_session.execute(stmt).scalars().all())
    
    def list_all_capabilities(
        self, 
        module: str | None = None,
        language: str | None = None
    ) -> list[ModuleCapability]:
        """列出所有活躍能力"""
        stmt = select(CapabilityRecord).where(CapabilityRecord.is_active == True)
        
        if module:
            stmt = stmt.where(CapabilityRecord.module == module)
        if language:
            stmt = stmt.where(CapabilityRecord.language == language)
        
        records = self.pg_session.execute(stmt).scalars().all()
        return [
            ModuleCapability.model_validate_json(str(r.metadata_json))
            for r in records
        ]
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息"""
        from sqlalchemy import func
        
        # 總能力數
        total = self.pg_session.execute(
            select(func.count(CapabilityRecord.id))
            .where(CapabilityRecord.is_active == True)
        ).scalar()
        
        # 按模組統計
        by_module = self.pg_session.execute(
            select(
                CapabilityRecord.module,
                func.count(CapabilityRecord.id)
            )
            .where(CapabilityRecord.is_active == True)
            .group_by(CapabilityRecord.module)
        ).all()
        
        # 按語言統計
        by_language = self.pg_session.execute(
            select(
                CapabilityRecord.language,
                func.count(CapabilityRecord.id)
            )
            .where(CapabilityRecord.is_active == True)
            .group_by(CapabilityRecord.language)
        ).all()
        
        return {
            "total_capabilities": total,
            "by_module": {row[0]: row[1] for row in by_module},
            "by_language": {row[0]: row[1] for row in by_language},
        }
