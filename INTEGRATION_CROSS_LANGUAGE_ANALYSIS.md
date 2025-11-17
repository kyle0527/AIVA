# Integration Coordinators - 現有數據合約使用分析與多語言支持方案

## 📊 現有數據合約使用情況分析

### ✅ 已使用的 aiva_common 合約

#### 1. **基礎響應模型** （部分使用）

**當前使用**:
```python
# base_coordinator.py 中
from services.aiva_common.schemas import APIResponse  # ✅ 已引用但未使用
```

**aiva_common 提供**:
```python
# services/aiva_common/schemas/base.py
class APIResponse(BaseModel):
    success: bool
    message: str
    data: dict | list | None
    timestamp: datetime
    trace_id: str | None
    errors: list[str] | None
    metadata: dict | None
```

**問題**: 
- ✅ 已導入但未實際使用
- ❌ Coordinator 自定義了完整的 Pydantic models
- ❌ 未遵循「使用 aiva_common 統一定義」原則

---

#### 2. **漏洞發現模型** （完全未使用）

**當前狀態**: ❌ 完全重複定義

**Coordinator 自定義**:
```python
# base_coordinator.py
class Finding(BaseModel):
    id: str
    vulnerability_type: str
    severity: str
    cvss_score: float
    cwe_id: str
    # ... 30+ 字段
```

**aiva_common 已有**:
```python
# services/aiva_common/schemas/vulnerability_finding.py
class UnifiedVulnerabilityFinding(BaseModel):
    finding_id: str  
    vulnerability_type: VulnerabilityType  # 枚舉類型
    severity: Severity  # 枚舉類型
    confidence: Confidence  # 枚舉類型
    target: Target  # 複雜目標對象
    evidence: List[FindingEvidence]  # 標準證據格式
    # ... 完整的標準字段
```

**優勢**:
- ✅ 已集成 OWASP、CWE、CVSS 標準
- ✅ 包含 Bug Bounty 相關字段
- ✅ 完整的枚舉類型定義
- ✅ 符合 SOT (Single Source of Truth) 原則

---

#### 3. **枚舉類型** （完全未使用）

**當前狀態**: ❌ 使用字符串常量

**Coordinator 使用**:
```python
severity: str = Field(regex="^(critical|high|medium|low|info)$")
status: str = Field(regex="^(completed|failed|timeout|partial)$")
```

**aiva_common 提供**:
```python
# services/aiva_common/enums/security.py
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class Confidence(str, Enum):
    CONFIRMED = "confirmed"
    FIRM = "firm"
    TENTATIVE = "tentative"
```

**優勢**:
- ✅ 類型安全
- ✅ IDE 自動完成
- ✅ 避免字符串錯誤

---

#### 4. **任務模型** （部分可用）

**aiva_common 提供**:
```python
# services/aiva_common/schemas/base.py
class Task(BaseModel):
    task_id: str
    task_type: str
    status: str
    priority: int
    target_url: str | None
    target: Target | None  # 支持完整目標對象
```

**Coordinator 使用**: ❌ 部分重複定義

---

### ❌ 完全重複定義的模型

| Coordinator 模型 | aiva_common 對應 | 重複程度 |
|-----------------|------------------|---------|
| `FeatureResult` | `Task` + 擴展 | 70% |
| `Finding` | `UnifiedVulnerabilityFinding` | 90% |
| `TargetInfo` | `Target` | 80% |
| `EvidenceData` | `FindingEvidence` | 75% |
| `ImpactAssessment` | 內建於 `UnifiedVulnerabilityFinding` | 100% |
| `RemediationAdvice` | 內建於 `UnifiedVulnerabilityFinding` | 100% |

---

## 🔄 重構建議：使用 aiva_common 合約

### 方案 1：完全遵循 aiva_common（推薦）

#### 優點
- ✅ 符合 SOT 原則
- ✅ 自動獲得跨語言支持
- ✅ 減少維護成本
- ✅ 統一數據格式

#### 重構步驟

```python
# services/integration/coordinators/base_coordinator.py

# ============ 使用 aiva_common 標準合約 ============
from aiva_common.schemas import (
    APIResponse,
    Task,
)
from aiva_common.schemas.vulnerability_finding import (
    UnifiedVulnerabilityFinding as Finding,
    VulnerabilityCategory,
)
from aiva_common.schemas.security.findings import (
    Target,
    FindingEvidence,
)
from aiva_common.enums import (
    Severity,
    Confidence,
    VulnerabilityType,
    ModuleName,
)

# ============ 僅定義 Coordinator 特有的模型 ============

class OptimizationData(BaseModel):
    """內循環優化數據（Coordinator 特有）"""
    task_id: str
    feature_module: ModuleName  # 使用統一枚舉
    payload_efficiency: Dict[str, float]
    successful_patterns: List[str]
    # ... Coordinator 特有字段

class ReportData(BaseModel):
    """外循環報告數據（基於標準 Finding）"""
    task_id: str
    feature_module: ModuleName
    findings: List[Finding]  # 使用標準 Finding
    # ... 統計數據

class FeatureResult(BaseModel):
    """Features 返回結果（擴展自 Task）"""
    # 繼承 Task 的基礎字段
    task_id: str
    task_type: str
    status: str
    
    # Features 特有擴展
    findings: List[Finding]  # 使用標準 Finding
    statistics: StatisticsData
    performance: PerformanceMetrics
```

---

### 方案 2：漸進式遷移（次優）

#### 階段 1：使用基礎類型
```python
from aiva_common.schemas import APIResponse, Task
from aiva_common.enums import Severity, Confidence, ModuleName
```

#### 階段 2：使用目標和證據
```python
from aiva_common.schemas.security.findings import Target, FindingEvidence
```

#### 階段 3：完全遷移到標準 Finding
```python
from aiva_common.schemas.vulnerability_finding import UnifiedVulnerabilityFinding
```

---

## 🌐 多語言支持方案

### 現有 aiva_common 跨語言框架

#### 1. **Protocol Buffers 支持**

**已實現**:
```python
# services/aiva_common/cross_language/core.py
class CrossLanguageService:
    """跨語言服務核心"""
    - gRPC 通訊
    - Protocol Buffers 序列化
    - 統一錯誤映射
    - 連接池管理
```

**已支持的語言適配器**:
```python
# services/aiva_common/cross_language/adapters/
├── go_adapter.py       # Go 語言適配器
├── rust_adapter.py     # Rust 語言適配器
└── __init__.py
```

---

### Integration Coordinators 多語言集成方案

#### 架構設計

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Layer                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Python Coordinator (BaseCoordinator)        │   │
│  │  - 使用 aiva_common 標準合約                        │   │
│  │  - Pydantic models                                  │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                    │
│                        ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Cross-Language Adapter Layer               │   │
│  │  - Protocol Buffers 轉換                           │   │
│  │  - JSON 序列化/反序列化                            │   │
│  │  - gRPC 通訊                                       │   │
│  └──────┬──────────────┬──────────────┬───────────────┘   │
│         │              │              │                    │
└─────────┼──────────────┼──────────────┼────────────────────┘
          │              │              │
          ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Go Feature │  │ Rust Feature│  │Python Feature│
│   Service   │  │   Service   │  │   Service   │
│             │  │             │  │             │
│ - 接收 Proto│  │ - 接收 Proto│  │ - 接收 JSON │
│ - 返回 Proto│  │ - 返回 Proto│  │ - 返回 Dict │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

#### 實現方案

##### 1. **定義跨語言消息合約（.proto）**

```protobuf
// services/aiva_common/protos/integration.proto
syntax = "proto3";

package aiva.integration;

// 基於 aiva_common 標準定義
message Finding {
    string finding_id = 1;
    string vulnerability_type = 2;
    string severity = 3;
    double cvss_score = 4;
    string cwe_id = 5;
    string owasp_category = 6;
    
    message Evidence {
        string payload = 1;
        string request = 2;
        string response = 3;
        double confidence = 4;
    }
    Evidence evidence = 7;
    
    message PoC {
        repeated string steps = 1;
        string curl_command = 2;
    }
    PoC poc = 8;
}

message FeatureResult {
    string task_id = 1;
    string feature_module = 2;
    string status = 3;
    bool success = 4;
    
    repeated Finding findings = 5;
    
    message Statistics {
        int32 payloads_tested = 1;
        int32 requests_sent = 2;
        double success_rate = 3;
    }
    Statistics statistics = 6;
    
    message Performance {
        double avg_response_time_ms = 1;
        int32 rate_limit_hits = 2;
    }
    Performance performance = 7;
}

// gRPC 服務定義
service CoordinatorService {
    rpc CollectResult(FeatureResult) returns (CoordinationResponse);
}

message CoordinationResponse {
    bool success = 1;
    string task_id = 2;
    
    message OptimizationData {
        map<string, double> payload_efficiency = 1;
        int32 recommended_concurrency = 2;
    }
    OptimizationData internal_loop = 3;
    
    message ReportData {
        int32 total_findings = 1;
        int32 high_count = 2;
        string estimated_total_value = 3;
    }
    ReportData external_loop = 4;
}
```

##### 2. **Python Coordinator 適配層**

```python
# services/integration/coordinators/cross_language_adapter.py
from aiva_common.cross_language import CrossLanguageService, PythonAdapter
from aiva_common.schemas.vulnerability_finding import UnifiedVulnerabilityFinding
from google.protobuf.json_format import MessageToDict, ParseDict

# 自動生成的 Proto 類
from aiva_common.protos import integration_pb2

class CoordinatorCrossLanguageAdapter:
    """Coordinator 跨語言適配器"""
    
    def __init__(self):
        self.service = CrossLanguageService(
            config=CrossLanguageConfig(),
            adapter=PythonAdapter()
        )
    
    async def convert_to_proto(
        self, 
        result: Dict[str, Any]
    ) -> integration_pb2.FeatureResult:
        """Python Dict → Protocol Buffers"""
        # 使用 aiva_common 標準驗證
        validated = FeatureResult(**result)
        
        # 轉換為 Proto
        proto_message = integration_pb2.FeatureResult()
        ParseDict(validated.dict(), proto_message)
        return proto_message
    
    async def convert_from_proto(
        self, 
        proto_message: integration_pb2.FeatureResult
    ) -> Dict[str, Any]:
        """Protocol Buffers → Python Dict"""
        # Proto → Dict
        result_dict = MessageToDict(proto_message)
        
        # 使用 aiva_common 標準驗證
        validated = FeatureResult(**result_dict)
        return validated.dict()
    
    async def handle_go_result(self, proto_bytes: bytes) -> Dict[str, Any]:
        """處理 Go Feature 返回的 Proto 結果"""
        # 反序列化 Proto
        proto_message = integration_pb2.FeatureResult()
        proto_message.ParseFromString(proto_bytes)
        
        # 轉換為 Python 標準格式
        return await self.convert_from_proto(proto_message)
    
    async def handle_rust_result(self, proto_bytes: bytes) -> Dict[str, Any]:
        """處理 Rust Feature 返回的 Proto 結果"""
        return await self.handle_go_result(proto_bytes)
    
    async def handle_python_result(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """處理 Python Feature 返回的 JSON/Dict 結果"""
        # 直接使用，但先驗證
        validated = FeatureResult(**result_dict)
        return validated.dict()
```

##### 3. **更新 BaseCoordinator 支持多語言**

```python
# services/integration/coordinators/base_coordinator.py
class BaseCoordinator(ABC):
    def __init__(self, **kwargs):
        self.cross_lang_adapter = CoordinatorCrossLanguageAdapter()
        # ... 其他初始化
    
    async def collect_result(
        self, 
        result_data: Union[Dict[str, Any], bytes],
        source_language: str = "python"
    ) -> Dict[str, Any]:
        """收集結果（支持多語言）
        
        Args:
            result_data: 
                - Python: Dict[str, Any]
                - Go/Rust: bytes (Proto serialized)
            source_language: "python" | "go" | "rust"
        """
        # 1. 根據來源語言轉換為統一格式
        if source_language == "python":
            result_dict = await self.cross_lang_adapter.handle_python_result(result_data)
        elif source_language == "go":
            result_dict = await self.cross_lang_adapter.handle_go_result(result_data)
        elif source_language == "rust":
            result_dict = await self.cross_lang_adapter.handle_rust_result(result_data)
        else:
            raise ValueError(f"Unsupported language: {source_language}")
        
        # 2. 驗證並解析結果（使用 aiva_common 標準）
        result = await self._validate_result(result_dict)
        
        # 3. 後續處理（與之前相同）
        # ...
```

---

#### 4. **Go Feature Service 示例**

```go
// services/features/function_xss/main.go
package main

import (
    "context"
    pb "aiva/protos/integration"
    "google.golang.org/grpc"
)

type XSSFeatureService struct {
    pb.UnimplementedFeatureServiceServer
}

func (s *XSSFeatureService) ExecuteTest(
    ctx context.Context, 
    req *pb.TestRequest,
) (*pb.FeatureResult, error) {
    // 1. 執行 XSS 測試
    findings := performXSSTests(req.Target)
    
    // 2. 構建 Proto 響應
    result := &pb.FeatureResult{
        TaskId: req.TaskId,
        FeatureModule: "function_xss",
        Status: "completed",
        Success: true,
        Findings: findings,
        Statistics: &pb.FeatureResult_Statistics{
            PayloadsTested: 50,
            RequestsSent: 55,
            SuccessRate: 0.85,
        },
    }
    
    return result, nil
}

func main() {
    // 啟動 gRPC 服務
    lis, _ := net.Listen("tcp", ":50051")
    grpcServer := grpc.NewServer()
    pb.RegisterFeatureServiceServer(grpcServer, &XSSFeatureService{})
    grpcServer.Serve(lis)
}
```

---

#### 5. **Rust Feature Service 示例**

```rust
// services/features/function_sqli/src/main.rs
use tonic::{transport::Server, Request, Response, Status};
use aiva_protos::integration::{FeatureResult, Finding};

pub struct SqliFeatureService {}

#[tonic::async_trait]
impl feature_service_server::FeatureService for SqliFeatureService {
    async fn execute_test(
        &self,
        request: Request<TestRequest>,
    ) -> Result<Response<FeatureResult>, Status> {
        let req = request.into_inner();
        
        // 執行 SQL 注入測試
        let findings = perform_sqli_tests(&req.target);
        
        // 構建 Proto 響應
        let result = FeatureResult {
            task_id: req.task_id,
            feature_module: "function_sqli".to_string(),
            status: "completed".to_string(),
            success: true,
            findings,
            statistics: Some(Statistics {
                payloads_tested: 100,
                requests_sent: 120,
                success_rate: 0.9,
            }),
            ..Default::default()
        };
        
        Ok(Response::new(result))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50052".parse()?;
    let service = SqliFeatureService::default();
    
    Server::builder()
        .add_service(FeatureServiceServer::new(service))
        .serve(addr)
        .await?;
    
    Ok(())
}
```

---

## 📋 實施計劃

### Phase 1：重構使用 aiva_common 合約 ✅

**任務**:
1. 修改 `base_coordinator.py` 使用標準合約
2. 移除重複的模型定義
3. 更新 `xss_coordinator.py` 使用標準枚舉
4. 更新測試和文檔

**預期收益**:
- 減少 500+ 行重複代碼
- 自動獲得類型安全
- 符合 SOT 原則

---

### Phase 2：Protocol Buffers 定義 ⬜

**任務**:
1. 創建 `integration.proto` 定義
2. 生成 Python/Go/Rust 代碼
3. 更新 CI/CD 自動生成流程

**檔案**:
```
services/aiva_common/protos/
├── integration.proto      # Coordinator 專用合約
├── feature.proto          # Feature 通用合約
├── BUILD                  # Bazel 構建配置
└── generated/
    ├── python/
    │   └── integration_pb2.py
    ├── go/
    │   └── integration.pb.go
    └── rust/
        └── integration.rs
```

---

### Phase 3：跨語言適配層 ⬜

**任務**:
1. 實現 `CoordinatorCrossLanguageAdapter`
2. 更新 `BaseCoordinator` 支持多語言
3. 創建語言檢測邏輯

---

### Phase 4：多語言 Features 示例 ⬜

**任務**:
1. Go XSS Feature 示例
2. Rust SQLi Feature 示例
3. 整合測試和文檔

---

## 🎯 總結

### 當前問題
1. ❌ **完全未使用** aiva_common 標準合約
2. ❌ **重複定義** 90% 的數據模型
3. ❌ **缺少跨語言** 支持機制
4. ❌ **不符合 SOT** 原則

### 解決方案
1. ✅ **完全遷移**到 aiva_common 標準合約
2. ✅ **使用現有**的跨語言框架
3. ✅ **Protocol Buffers** 實現多語言通訊
4. ✅ **統一數據格式**，無需轉換器

### 預期收益
- 📉 減少 **70% 代碼重複**
- 🚀 自動獲得 **Go/Rust** 支持
- 🔒 提升 **類型安全**
- 📊 符合 **SOT 原則**
- 🌐 實現 **真正的跨語言**協作
