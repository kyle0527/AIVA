# AIVA 能力元數據數據庫設計方案

**設計目標**: 避免每次內循環分析後產生大量檔案,使用數據庫管理能力元數據,支持增量更新和版本演化

**日期**: 2025-11-28  
**參考標準**: Martin Fowler 分散式系統模式、Schema Migration 最佳實踐、Data Contract 設計

---

## 📋 目錄

- [核心問題分析](#核心問題分析)
- [業界最佳實踐參考](#業界最佳實踐參考)
- [AIVA 解決方案設計](#aiva-解決方案設計)
- [數據庫 Schema 設計](#數據庫-schema-設計)
- [增量更新機制](#增量更新機制)
- [數據合約通信設計](#數據合約通信設計)
- [實施步驟](#實施步驟)

---

## 核心問題分析

### 當前痛點

1. **文件膨脹問題**: 內循環每次分析生成大量 JSON/Python 檔案
2. **無法識別變化**: 不知道哪些能力是新增、修改還是刪除
3. **歷史追溯困難**: 無法查看能力演化歷史
4. **調用信息缺失**: AI 查詢 RAG 後不知如何調用能力
5. **數據合約未定義**: 內外循環通信缺乏標準化接口

### 用戶需求

> "不希望每次分析探索完就多一堆檔案或資料，因為基本上不太會大幅更新，希望利用資料庫，分析完後建立目前操作方式紀錄，後續更新優化時能夠識別，就有變化的部分更新或是擴張用法"

**關鍵需求**:
- ✅ 使用數據庫而非文件存儲
- ✅ 支持增量更新 (只更新變化部分)
- ✅ 自動識別新增/修改/刪除
- ✅ 保留操作方式記錄 (invocation metadata)
- ✅ 數據合約定義清晰的通信接口

---

## 業界最佳實踐參考

### 1. Versioned Value Pattern (Martin Fowler)

**核心思想**: 每次更新都保存新版本,不覆蓋舊數據

```python
# 每個能力都有版本號
capability_v1 = {
    "name": "detect_sqli",
    "version": 1,
    "parameters": [{"name": "url", "type": "str"}]
}

capability_v2 = {
    "name": "detect_sqli",
    "version": 2,
    "parameters": [
        {"name": "url", "type": "str"},
        {"name": "timeout", "type": "int"}  # 新增參數
    ]
}
```

**優點**: 可追溯歷史、回滾、對比版本差異

### 2. Hash-Based Change Detection

**核心思想**: 使用內容哈希識別變化

```python
import hashlib
import json

def compute_capability_hash(cap: dict) -> str:
    """計算能力內容的 SHA256 哈希"""
    # 排除 metadata 中會變動的欄位 (如 timestamp)
    stable_content = {
        "name": cap["name"],
        "module": cap["module"],
        "parameters": cap["parameters"],
        "return_type": cap["return_type"]
    }
    content_str = json.dumps(stable_content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()
```

**檢測邏輯**:
```python
old_hash = db.get_capability_hash("detect_sqli")
new_hash = compute_capability_hash(current_capability)

if old_hash != new_hash:
    # 能力已變更,更新數據庫
    db.update_capability(current_capability, new_version)
```

### 3. Schema Migration (零停機更新)

**Dual Writing 策略**:
```python
# 階段 1: 同時寫入新舊格式
def save_capability(cap):
    # 舊格式: ChromaDB
    chroma_client.add(cap)
    
    # 新格式: PostgreSQL
    pg_client.insert(cap)

# 階段 2: 數據回填
def backfill():
    old_caps = chroma_client.get_all()
    for cap in old_caps:
        pg_client.insert(cap)

# 階段 3: 切換讀取
def get_capability(name):
    # return chroma_client.get(name)  # 舊
    return pg_client.get(name)  # 新

# 階段 4: 停止寫入 ChromaDB,移除舊數據
```

### 4. Data Contract (數據合約)

**Protobuf 定義** (業界標準):
```protobuf
// capability_contract.proto
syntax = "proto3";

message CapabilityMetadata {
  string name = 1;
  string module = 2;
  string language = 3;
  int32 version = 4;
  
  InvocationInfo invocation = 5;
  repeated Parameter parameters = 6;
  ReturnInfo return_info = 7;
}

message InvocationInfo {
  string protocol = 1;  // "http", "grpc", "direct"
  string endpoint = 2;
  string module_arg = 3;
  string function_arg = 4;
}

message Parameter {
  string name = 1;
  string type = 2;
  bool required = 3;
  string default_value = 4;
  string description = 5;
}
```

**或使用 Pydantic** (Python 原生):
```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class InvocationInfo(BaseModel):
    protocol: str = Field(description="http/grpc/direct")
    endpoint: Optional[str] = None
    module_arg: str
    function_arg: str

class Parameter(BaseModel):
    name: str
    type: str
    required: bool = True
    default_value: Optional[str] = None
    description: str

class CapabilityMetadata(BaseModel):
    name: str
    module: str
    language: str
    version: int
    content_hash: str = Field(description="內容哈希,用於檢測變化")
    
    invocation: InvocationInfo
    parameters: List[Parameter]
    return_type: Optional[str] = None
    
    created_at: datetime
    updated_at: datetime
```

---

## AIVA 解決方案設計

### 整體架構

```
┌─────────────────────────────────────────────────────────────┐
│                    內循環 (Internal Loop)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │CapabilityAnalyzer│→│HashComputer│→│ChangeDetector│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            ↓                                 │
│                  ┌──────────────────┐                        │
│                  │CapabilityRegistry│                        │
│                  │  (統一注冊中心)   │                        │
│                  └──────────────────┘                        │
│                            ↓                                 │
│         ┌──────────────────┴──────────────────┐             │
│         ↓                                      ↓             │
│  ┌─────────────┐                        ┌─────────────┐     │
│  │ PostgreSQL  │                        │  ChromaDB   │     │
│  │(關係型DB)   │                        │ (向量搜索)  │     │
│  │- 能力元數據 │                        │- 語義搜索   │     │
│  │- 版本歷史   │                        │- RAG 查詢   │     │
│  │- 變更記錄   │                        └─────────────┘     │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                            ↑ ↓
                   ┌──────────────────┐
                   │  Data Contract   │
                   │  (Pydantic Model)│
                   └──────────────────┘
                            ↑ ↓
┌─────────────────────────────────────────────────────────────┐
│                  AI 決策層 (Cognitive Core)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ RAG 查詢     │→│CapabilityInvoker│→│UnifiedCaller │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 核心組件

#### 1. CapabilityRegistry (能力註冊中心)

**職責**:
- 接收內循環掃描結果
- 計算內容哈希識別變化
- 執行增量更新 (新增/修改/刪除)
- 同步到 PostgreSQL 和 ChromaDB

```python
# services/core/aiva_core/internal_exploration/capability_registry.py
from typing import List, Dict, Any
import hashlib
import json
from datetime import datetime, UTC
from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session

from aiva_common.schemas.capability_contract import CapabilityMetadata, ChangeType
from .models import CapabilityRecord, CapabilityVersion, CapabilityChangeLog


class CapabilityRegistry:
    """能力註冊中心 - 統一管理能力元數據"""
    
    def __init__(self, pg_session: Session, chroma_collection):
        self.pg_session = pg_session
        self.chroma_collection = chroma_collection
    
    def register_capabilities(
        self, 
        capabilities: List[CapabilityMetadata]
    ) -> Dict[str, Any]:
        """註冊/更新能力列表
        
        Args:
            capabilities: 內循環掃描到的能力列表
            
        Returns:
            變更統計: {"added": 10, "modified": 5, "deleted": 2, "unchanged": 765}
        """
        stats = {"added": 0, "modified": 0, "deleted": 0, "unchanged": 0}
        
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
                
            elif existing_keys[key].content_hash != content_hash:
                # 能力已變更
                self._update_capability(cap, content_hash)
                stats["modified"] += 1
                
            else:
                # 無變化
                stats["unchanged"] += 1
        
        # 3. 檢測已刪除的能力
        deleted_keys = set(existing_keys.keys()) - scanned_keys
        for key in deleted_keys:
            self._mark_deleted(existing_keys[key])
            stats["deleted"] += 1
        
        # 4. 記錄變更日誌
        self._log_scan_result(stats)
        
        return stats
    
    def _make_key(self, cap: CapabilityMetadata) -> str:
        """生成能力唯一標識: module::name::file_path"""
        return f"{cap.module}::{cap.name}::{cap.file_path}"
    
    def _compute_hash(self, cap: CapabilityMetadata) -> str:
        """計算能力內容哈希 (排除時間戳等易變欄位)"""
        stable_content = {
            "name": cap.name,
            "module": cap.module,
            "parameters": [p.dict() for p in cap.parameters],
            "return_type": cap.return_type,
            "invocation": cap.invocation.dict() if cap.invocation else None
        }
        content_str = json.dumps(stable_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _add_capability(self, cap: CapabilityMetadata, content_hash: str):
        """新增能力到數據庫"""
        # PostgreSQL: 主記錄
        record = CapabilityRecord(
            key=self._make_key(cap),
            name=cap.name,
            module=cap.module,
            language=cap.language,
            file_path=cap.file_path,
            version=1,
            content_hash=content_hash,
            is_active=True,
            metadata_json=cap.json(),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        self.pg_session.add(record)
        
        # PostgreSQL: 版本歷史
        version = CapabilityVersion(
            capability_key=record.key,
            version=1,
            content_hash=content_hash,
            metadata_json=cap.json(),
            change_type=ChangeType.ADDED,
            created_at=datetime.now(UTC)
        )
        self.pg_session.add(version)
        self.pg_session.commit()
        
        # ChromaDB: 向量存儲 (用於 RAG 查詢)
        self._sync_to_chromadb(cap, "add")
    
    def _update_capability(self, cap: CapabilityMetadata, content_hash: str):
        """更新能力"""
        key = self._make_key(cap)
        
        # 獲取當前版本號
        stmt = select(CapabilityRecord).where(CapabilityRecord.key == key)
        record = self.pg_session.execute(stmt).scalar_one()
        new_version = record.version + 1
        
        # 更新主記錄
        record.version = new_version
        record.content_hash = content_hash
        record.metadata_json = cap.json()
        record.updated_at = datetime.now(UTC)
        
        # 新增版本記錄
        version = CapabilityVersion(
            capability_key=key,
            version=new_version,
            content_hash=content_hash,
            metadata_json=cap.json(),
            change_type=ChangeType.MODIFIED,
            created_at=datetime.now(UTC)
        )
        self.pg_session.add(version)
        self.pg_session.commit()
        
        # 同步到 ChromaDB
        self._sync_to_chromadb(cap, "update")
    
    def _mark_deleted(self, cap: CapabilityMetadata):
        """標記能力為已刪除 (軟刪除)"""
        key = self._make_key(cap)
        
        stmt = (
            update(CapabilityRecord)
            .where(CapabilityRecord.key == key)
            .values(is_active=False, updated_at=datetime.now(UTC))
        )
        self.pg_session.execute(stmt)
        
        # 記錄刪除版本
        version = CapabilityVersion(
            capability_key=key,
            version=cap.version + 1,
            content_hash="",
            metadata_json="",
            change_type=ChangeType.DELETED,
            created_at=datetime.now(UTC)
        )
        self.pg_session.add(version)
        self.pg_session.commit()
        
        # ChromaDB 刪除
        self._sync_to_chromadb(cap, "delete")
    
    def _sync_to_chromadb(self, cap: CapabilityMetadata, operation: str):
        """同步到 ChromaDB 向量數據庫"""
        doc_id = f"cap_{self._compute_hash(cap)}"
        
        if operation == "add" or operation == "update":
            # 構建文檔內容
            content = f"""
Capability: {cap.name}
Module: {cap.module}
Language: {cap.language}
Description: {cap.description or 'No description'}

Parameters:
{chr(10).join([f"  - {p.name} ({p.type}): {p.description}" for p in cap.parameters])}

Invocation:
  Protocol: {cap.invocation.protocol if cap.invocation else 'unknown'}
  Module: {cap.invocation.module_arg if cap.invocation else 'unknown'}
  Function: {cap.invocation.function_arg if cap.invocation else 'unknown'}
"""
            
            metadata = {
                "capability_name": cap.name,
                "module": cap.module,
                "language": cap.language,
                "version": cap.version,
                "namespace": "self_awareness",
                "type": "capability",
                "invocation_protocol": cap.invocation.protocol if cap.invocation else None,
                "invocation_module": cap.invocation.module_arg if cap.invocation else None,
                "invocation_function": cap.invocation.function_arg if cap.invocation else None,
            }
            
            self.chroma_collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata]
            )
        
        elif operation == "delete":
            try:
                self.chroma_collection.delete(ids=[doc_id])
            except Exception as e:
                # ChromaDB 可能不存在該文檔
                pass
    
    def get_capability_by_name(self, name: str) -> CapabilityMetadata | None:
        """根據名稱查詢能力"""
        stmt = select(CapabilityRecord).where(
            CapabilityRecord.name == name,
            CapabilityRecord.is_active == True
        )
        record = self.pg_session.execute(stmt).scalar_one_or_none()
        
        if record:
            return CapabilityMetadata.parse_raw(record.metadata_json)
        return None
    
    def get_capability_history(self, name: str) -> List[CapabilityVersion]:
        """查詢能力變更歷史"""
        key_pattern = f"%::{name}::%"
        stmt = (
            select(CapabilityVersion)
            .where(CapabilityVersion.capability_key.like(key_pattern))
            .order_by(CapabilityVersion.version.desc())
        )
        return self.pg_session.execute(stmt).scalars().all()
```

---

## 數據庫 Schema 設計

### PostgreSQL Tables

```sql
-- 1. 能力主表
CREATE TABLE capability_records (
    id SERIAL PRIMARY KEY,
    key VARCHAR(500) UNIQUE NOT NULL,  -- module::name::file_path
    name VARCHAR(200) NOT NULL,
    module VARCHAR(200) NOT NULL,
    language VARCHAR(50) NOT NULL,
    file_path TEXT,
    
    version INTEGER NOT NULL DEFAULT 1,
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 哈希
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    metadata_json TEXT NOT NULL,  -- 完整元數據 JSON
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    INDEX idx_name (name),
    INDEX idx_module (module),
    INDEX idx_hash (content_hash),
    INDEX idx_active (is_active)
);

-- 2. 版本歷史表
CREATE TABLE capability_versions (
    id SERIAL PRIMARY KEY,
    capability_key VARCHAR(500) NOT NULL REFERENCES capability_records(key),
    version INTEGER NOT NULL,
    
    content_hash VARCHAR(64) NOT NULL,
    metadata_json TEXT NOT NULL,
    
    change_type VARCHAR(20) NOT NULL,  -- 'ADDED', 'MODIFIED', 'DELETED'
    change_summary TEXT,  -- 變更摘要 (可選)
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    UNIQUE(capability_key, version),
    INDEX idx_change_type (change_type),
    INDEX idx_created_at (created_at)
);

-- 3. 變更日誌表
CREATE TABLE capability_change_logs (
    id SERIAL PRIMARY KEY,
    scan_id UUID NOT NULL,
    scan_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    added_count INTEGER NOT NULL DEFAULT 0,
    modified_count INTEGER NOT NULL DEFAULT 0,
    deleted_count INTEGER NOT NULL DEFAULT 0,
    unchanged_count INTEGER NOT NULL DEFAULT 0,
    
    total_capabilities INTEGER NOT NULL,
    
    details_json TEXT,  -- 詳細變更列表
    
    INDEX idx_scan_timestamp (scan_timestamp)
);

-- 4. 調用統計表 (用於監控)
CREATE TABLE capability_invocation_stats (
    id SERIAL PRIMARY KEY,
    capability_key VARCHAR(500) NOT NULL REFERENCES capability_records(key),
    
    invocation_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    
    avg_duration_ms FLOAT,
    last_invoked_at TIMESTAMP WITH TIME ZONE,
    
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    INDEX idx_capability_key (capability_key)
);
```

### SQLAlchemy Models

```python
# services/core/aiva_core/internal_exploration/models.py
from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()

class ChangeType(str, Enum):
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


class CapabilityRecord(Base):
    __tablename__ = "capability_records"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(500), unique=True, nullable=False)
    name = Column(String(200), nullable=False, index=True)
    module = Column(String(200), nullable=False, index=True)
    language = Column(String(50), nullable=False)
    file_path = Column(Text)
    
    version = Column(Integer, nullable=False, default=1)
    content_hash = Column(String(64), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    
    metadata_json = Column(Text, nullable=False)
    
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # 關聯
    versions = relationship("CapabilityVersion", back_populates="capability")
    stats = relationship("CapabilityInvocationStats", back_populates="capability")


class CapabilityVersion(Base):
    __tablename__ = "capability_versions"
    
    id = Column(Integer, primary_key=True)
    capability_key = Column(String(500), ForeignKey("capability_records.key"), nullable=False)
    version = Column(Integer, nullable=False)
    
    content_hash = Column(String(64), nullable=False)
    metadata_json = Column(Text, nullable=False)
    
    change_type = Column(String(20), nullable=False, index=True)
    change_summary = Column(Text)
    
    created_at = Column(DateTime, nullable=False, index=True)
    
    capability = relationship("CapabilityRecord", back_populates="versions")


class CapabilityChangeLog(Base):
    __tablename__ = "capability_change_logs"
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(String(36), nullable=False)
    scan_timestamp = Column(DateTime, nullable=False, index=True)
    
    added_count = Column(Integer, nullable=False, default=0)
    modified_count = Column(Integer, nullable=False, default=0)
    deleted_count = Column(Integer, nullable=False, default=0)
    unchanged_count = Column(Integer, nullable=False, default=0)
    
    total_capabilities = Column(Integer, nullable=False)
    
    details_json = Column(Text)


class CapabilityInvocationStats(Base):
    __tablename__ = "capability_invocation_stats"
    
    id = Column(Integer, primary_key=True)
    capability_key = Column(String(500), ForeignKey("capability_records.key"), nullable=False, index=True)
    
    invocation_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    
    avg_duration_ms = Column(Float)
    last_invoked_at = Column(DateTime)
    
    updated_at = Column(DateTime, nullable=False)
    
    capability = relationship("CapabilityRecord", back_populates="stats")
```

---

## 增量更新機制

### 變更檢測算法

```python
def detect_changes(
    old_capabilities: List[CapabilityMetadata],
    new_capabilities: List[CapabilityMetadata]
) -> Dict[str, List[CapabilityMetadata]]:
    """檢測能力變更
    
    Returns:
        {
            "added": [...],
            "modified": [...],
            "deleted": [...],
            "unchanged": [...]
        }
    """
    old_dict = {cap.key: cap for cap in old_capabilities}
    new_dict = {cap.key: cap for cap in new_capabilities}
    
    changes = {
        "added": [],
        "modified": [],
        "deleted": [],
        "unchanged": []
    }
    
    # 檢測新增和修改
    for key, new_cap in new_dict.items():
        if key not in old_dict:
            changes["added"].append(new_cap)
        elif old_dict[key].content_hash != new_cap.content_hash:
            changes["modified"].append(new_cap)
        else:
            changes["unchanged"].append(new_cap)
    
    # 檢測刪除
    for key, old_cap in old_dict.items():
        if key not in new_dict:
            changes["deleted"].append(old_cap)
    
    return changes
```

### 差異摘要生成

```python
def generate_change_summary(
    old_cap: CapabilityMetadata,
    new_cap: CapabilityMetadata
) -> str:
    """生成變更摘要"""
    summary_parts = []
    
    # 檢測參數變化
    old_params = {p.name: p for p in old_cap.parameters}
    new_params = {p.name: p for p in new_cap.parameters}
    
    added_params = set(new_params.keys()) - set(old_params.keys())
    removed_params = set(old_params.keys()) - set(new_params.keys())
    
    if added_params:
        summary_parts.append(f"Added parameters: {', '.join(added_params)}")
    if removed_params:
        summary_parts.append(f"Removed parameters: {', '.join(removed_params)}")
    
    # 檢測返回類型變化
    if old_cap.return_type != new_cap.return_type:
        summary_parts.append(
            f"Return type changed: {old_cap.return_type} → {new_cap.return_type}"
        )
    
    # 檢測調用信息變化
    if old_cap.invocation != new_cap.invocation:
        summary_parts.append("Invocation metadata changed")
    
    return "; ".join(summary_parts) if summary_parts else "No significant changes"
```

---

## 數據合約通信設計

### Pydantic Schema 定義

```python
# services/aiva_common/schemas/capability_contract.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class Protocol(str, Enum):
    """通信協議"""
    HTTP = "http"
    GRPC = "grpc"
    DIRECT = "direct"
    UNIFIED_CALLER = "unified_caller"


class InvocationInfo(BaseModel):
    """調用信息 - 定義如何調用能力"""
    protocol: Protocol = Field(description="通信協議")
    endpoint: Optional[str] = Field(None, description="HTTP/gRPC 端點 URL")
    module_arg: str = Field(description="模組名稱參數")
    function_arg: str = Field(description="函數名稱參數")
    parameter_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="參數映射關係,如 {'target_url': 'url'}"
    )
    
    # 額外配置
    timeout: Optional[int] = Field(None, description="超時時間(秒)")
    retry_count: Optional[int] = Field(None, description="重試次數")
    async_mode: bool = Field(False, description="是否異步調用")


class Parameter(BaseModel):
    """參數定義"""
    name: str
    type: str  # "str", "int", "float", "bool", "dict", "list"
    required: bool = True
    default_value: Optional[Any] = None
    description: str = ""
    example: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None  # {"min": 0, "max": 100}


class ReturnInfo(BaseModel):
    """返回值定義"""
    type: str
    description: str = ""
    example: Optional[Any] = None
    structure: Optional[Dict[str, Any]] = None  # 複雜類型的結構定義


class UsageExample(BaseModel):
    """使用範例"""
    title: str
    description: str
    input: Dict[str, Any]
    expected_output: Optional[Any] = None
    code_snippet: str  # Python 代碼範例


class CapabilityMetadata(BaseModel):
    """能力元數據 - 完整數據合約"""
    # 基本信息
    name: str = Field(description="能力函數名稱")
    module: str = Field(description="所屬模組")
    language: str = Field(description="編程語言: Python/Go/Rust/TypeScript")
    file_path: str = Field(description="源碼文件路徑")
    
    # 版本信息
    version: int = Field(1, description="版本號")
    content_hash: str = Field(description="內容哈希,用於檢測變化")
    
    # 分類
    category: str = Field("utility", description="能力分類")
    sub_category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # 描述
    description: str = Field("", description="能力描述")
    complexity: int = Field(1, description="複雜度 1-5")
    
    # 調用信息 (核心!)
    invocation: Optional[InvocationInfo] = Field(
        None,
        description="調用元數據 - AI 需要這個來知道如何調用"
    )
    
    # 參數和返回值
    parameters: List[Parameter] = Field(default_factory=list)
    return_info: Optional[ReturnInfo] = None
    
    # 使用範例
    usage_examples: List[UsageExample] = Field(default_factory=list)
    
    # Python 代碼範例 (快速參考)
    call_example_python: Optional[str] = None
    call_example_http: Optional[str] = None
    
    # 元數據
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "detect_sqli",
                "module": "function_sqli",
                "language": "Python",
                "file_path": "services/function/function_sqli/aiva_func_sqli/smart_sqli_detector.py",
                "version": 1,
                "content_hash": "a1b2c3d4e5f6g7h8",
                "category": "scanning",
                "sub_category": "vulnerability_scan",
                "description": "Detect SQL injection vulnerabilities",
                "invocation": {
                    "protocol": "unified_caller",
                    "module_arg": "function_sqli",
                    "function_arg": "detect_sqli",
                    "async_mode": True
                },
                "parameters": [
                    {
                        "name": "target_url",
                        "type": "str",
                        "required": True,
                        "description": "目標 URL"
                    }
                ],
                "call_example_python": "caller.call_function('function_sqli', 'detect_sqli', {'target_url': 'http://example.com'})"
            }
        }


class CapabilityQueryRequest(BaseModel):
    """能力查詢請求"""
    query: str = Field(description="自然語言查詢,如 'SQL injection testing'")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="過濾條件: {'language': 'Python', 'category': 'scanning'}"
    )
    top_k: int = Field(5, description="返回結果數量")
    include_inactive: bool = Field(False, description="是否包含已刪除的能力")


class CapabilityQueryResponse(BaseModel):
    """能力查詢響應"""
    query: str
    results: List[CapabilityMetadata]
    total_found: int
    relevance_scores: List[float]
    timestamp: datetime


class CapabilityInvocationRequest(BaseModel):
    """能力調用請求"""
    capability_name: str = Field(description="能力名稱")
    parameters: Dict[str, Any] = Field(description="調用參數")
    timeout: Optional[int] = None
    async_mode: bool = False


class CapabilityInvocationResponse(BaseModel):
    """能力調用響應"""
    success: bool
    capability_name: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float
    timestamp: datetime
```

### 內外循環通信接口

```python
# services/core/aiva_core/internal_exploration/internal_loop_api.py
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime, UTC

from aiva_common.schemas.capability_contract import (
    CapabilityQueryRequest,
    CapabilityQueryResponse,
    CapabilityMetadata
)
from .capability_registry import CapabilityRegistry

router = APIRouter(prefix="/internal-loop", tags=["internal-loop"])


@router.post("/capabilities/query", response_model=CapabilityQueryResponse)
async def query_capabilities(request: CapabilityQueryRequest):
    """查詢能力 (供 AI 決策層使用)"""
    registry = get_capability_registry()
    
    # RAG 向量搜索
    results = registry.search_capabilities(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
        include_inactive=request.include_inactive
    )
    
    return CapabilityQueryResponse(
        query=request.query,
        results=results,
        total_found=len(results),
        relevance_scores=[r.relevance_score for r in results],
        timestamp=datetime.now(UTC)
    )


@router.get("/capabilities/{capability_name}", response_model=CapabilityMetadata)
async def get_capability(capability_name: str):
    """獲取能力詳細信息"""
    registry = get_capability_registry()
    cap = registry.get_capability_by_name(capability_name)
    
    if not cap:
        raise HTTPException(status_code=404, detail="Capability not found")
    
    return cap


@router.get("/capabilities/{capability_name}/history")
async def get_capability_history(capability_name: str):
    """獲取能力變更歷史"""
    registry = get_capability_registry()
    history = registry.get_capability_history(capability_name)
    
    return {
        "capability_name": capability_name,
        "total_versions": len(history),
        "versions": [
            {
                "version": v.version,
                "change_type": v.change_type,
                "change_summary": v.change_summary,
                "created_at": v.created_at
            }
            for v in history
        ]
    }


@router.post("/scan/trigger")
async def trigger_internal_scan():
    """觸發內循環掃描 (手動或定時任務)"""
    from .module_explorer import ModuleExplorer
    
    explorer = ModuleExplorer()
    capabilities = await explorer.scan_all_modules()
    
    registry = get_capability_registry()
    stats = registry.register_capabilities(capabilities)
    
    return {
        "scan_completed": True,
        "timestamp": datetime.now(UTC),
        "statistics": stats
    }
```

---

## 實施步驟

### Phase 1: 數據庫遷移 (1-2 天)

```python
# 1. 創建 PostgreSQL tables
# scripts/migrations/001_create_capability_tables.sql

# 2. Dual Writing: 同時寫入 PostgreSQL 和 ChromaDB
# 修改 internal_loop_connector.py

# 3. 數據回填: 將現有 ChromaDB 數據導入 PostgreSQL
# scripts/backfill_capabilities.py

# 4. 測試雙寫模式
```

### Phase 2: Registry 實現 (2-3 天)

```python
# 1. 實現 CapabilityRegistry 類
# services/core/aiva_core/internal_exploration/capability_registry.py

# 2. 實現哈希計算和變更檢測
# 3. 實現增量更新邏輯
# 4. 單元測試
```

### Phase 3: Data Contract 整合 (1-2 天)

```python
# 1. 定義 Pydantic schemas
# services/aiva_common/schemas/capability_contract.py

# 2. 更新 internal_loop_connector 使用新 schema
# 3. 實現 FastAPI 接口
# 4. API 測試
```

### Phase 4: AI 決策層集成 (2-3 天)

```python
# 1. 實現 CapabilityInvoker
# services/core/aiva_core/task_planning/capability_invoker.py

# 2. 集成到 execution_planner
# 3. 端到端測試: RAG 查詢 → AI 決策 → 實際調用
```

### Phase 5: 切換讀取路徑 (1 天)

```python
# 1. AI 查詢切換到 PostgreSQL + ChromaDB
# 2. 停止寫入舊格式
# 3. 移除舊代碼
```

---

## 總結

### 核心優勢

1. **✅ 無文件膨脹**: 所有元數據存儲在數據庫,不再生成大量 JSON 文件
2. **✅ 增量更新**: 哈希檢測自動識別變化,只更新修改部分
3. **✅ 版本追溯**: 保留完整變更歷史,可查看任意版本
4. **✅ 調用清晰**: `InvocationInfo` 明確告訴 AI 如何調用能力
5. **✅ 數據合約**: Pydantic schema 定義標準化通信接口
6. **✅ 零停機遷移**: Dual Writing 策略確保平滑過渡

### 下一步行動

```bash
# 1. 創建數據庫
cd C:\D\fold7\AIVA-git
python scripts/migrations/create_capability_db.py

# 2. 運行回填
python scripts/backfill_capabilities.py

# 3. 觸發內循環掃描
python -m aiva_cli internal-loop scan --update-db

# 4. 測試 API
curl http://localhost:8000/internal-loop/capabilities/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SQL injection testing", "top_k": 5}'
```

---

**技術參考**:
- Martin Fowler Patterns of Distributed Systems
- Schema Migration Best Practices (Wikipedia)
- Pydantic Data Validation
- PostgreSQL Versioning Patterns
- ChromaDB Vector Store Integration
