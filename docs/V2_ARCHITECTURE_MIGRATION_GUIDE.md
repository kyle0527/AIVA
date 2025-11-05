# AIVA V2 Architecture Migration Guide

## Overview

This document guides developers through migrating from V1 to V2 unified architecture in AIVA.

**Last Updated**: 2025-11-05  
**Status**: Phase 1 & 2 Complete, Phase 3 & 4 In Progress

## Table of Contents

1. [Cross-Language Communication](#cross-language-communication)
2. [Schema Definitions](#schema-definitions)
3. [Feature Modules](#feature-modules)
4. [Data Storage](#data-storage)

---

## Cross-Language Communication

### V1 (Deprecated)
- **Transport**: HTTP requests (`requests.get/post`) and subprocess calls
- **Issues**: No type safety, unreliable error handling, no connection pooling
- **Files**: Direct calls in `multilang_coordinator.py`

### V2 (Recommended)
- **Transport**: gRPC with Protocol Buffers
- **Benefits**: Type-safe, bidirectional streaming, connection pooling, health checks
- **Files**: 
  - `services/aiva_common/cross_language/core.py` - Core service
  - `services/aiva_common/protocols/aiva_services.proto` - Service definitions

### Migration Path

#### For Python Clients

**Old Way (V1)**:
```python
import requests

# HTTP call to Go service
response = requests.post('http://localhost:8081/api/ai/process', 
                        json={'task': 'scan', 'params': {}})
```

**New Way (V2)**:
```python
from services.aiva_common.cross_language import get_cross_language_service

# gRPC call with auto-retry and connection pooling
service = get_cross_language_service()
response = await service.call_service(
    stub_class=AIServiceStub,
    method_name='ProcessTask',
    request=task_request,
    target='localhost:50051'
)
```

#### For Go/Rust/TypeScript Services

Services need to implement gRPC servers using `aiva_services.proto`:

```go
// Example Go gRPC service implementation
import (
    pb "github.com/kyle0527/aiva/services/aiva_common/protocols"
    "google.golang.org/grpc"
)

type aiServiceServer struct {
    pb.UnimplementedAIServiceServer
}

func (s *aiServiceServer) ProcessTask(ctx context.Context, req *pb.AIVARequest) (*pb.AIVAResponse, error) {
    // Process the task
    return &pb.AIVAResponse{
        Header: req.Header,
        Status: &pb.ResponseStatus{Code: 0, Message: "Success"},
        Payload: result,
    }, nil
}
```

### Current Status

✅ **Complete**:
- `CrossLanguageService` infrastructure ready
- Health check with graceful fallback
- `multilang_coordinator.py` updated to use V2 with V1 fallback

⚠️ **TODO**:
- Implement gRPC servers in Go services (services/features/function_*_go)
- Implement gRPC servers in Rust services 
- Implement gRPC servers in TypeScript services (services/scan/aiva_scan_node)
- Generate language-specific code from `.proto` files

---

## Schema Definitions

### V1 (Deprecated)

**Problem**: 7 scattered `schemas.py` files with duplicate definitions (1527 total lines)

Files:
- `services/core/aiva_core/schemas.py`
- `services/scan/aiva_scan/schemas.py`
- `services/features/function_idor/schemas.py`
- `services/features/function_postex/schemas.py`
- `services/features/function_sqli/schemas.py`
- `services/features/function_ssrf/schemas.py`
- `services/features/function_xss/schemas.py`

### V2 (Recommended)

**Solution**: Single Source of Truth (SOT) in YAML

**File**: `services/aiva_common/core_schema_sot.yaml` (2305 lines)

**Benefits**:
- Single definition eliminates duplication
- Language-agnostic (can generate Python, Go, Rust, TypeScript)
- Centralized validation rules
- Version control for schema evolution

### Migration Path

#### Step 1: Check for Deprecation Warnings

All V1 schema files now emit `DeprecationWarning`:

```python
# When you import
from services.core.aiva_core.schemas import AssetAnalysis

# You'll see:
# DeprecationWarning: services/core/aiva_core/schemas.py is deprecated (V1 architecture). 
# Please migrate to V2 unified schema: services/aiva_common/core_schema_sot.yaml
```

#### Step 2: Define Schema in V2 SOT

Add your schema to `core_schema_sot.yaml`:

```yaml
base_types:
  YourNewSchema:
    description: Your schema description
    fields:
      field_name:
        type: str
        required: true
        description: Field description
      another_field:
        type: int
        required: false
        default: 0
```

#### Step 3: Generate Code (Future)

```bash
# Generate Python/Go/Rust/TypeScript from YAML
python tools/schema_codegen_tool.py --input services/aiva_common/core_schema_sot.yaml \
                                     --output services/aiva_common/schemas/generated/
```

#### Step 4: Update Imports

```python
# Old (V1)
from services.core.aiva_core.schemas import AssetAnalysis

# New (V2) - once codegen is ready
from services.aiva_common.schemas.generated import AssetAnalysis
```

### Current Status

✅ **Complete**:
- V2 SOT defined in `core_schema_sot.yaml`
- Deprecation warnings added to all V1 schema files
- Migration documentation

⚠️ **TODO**:
- Implement `schema_codegen_tool.py` to generate code from YAML
- Generate Python Pydantic models
- Generate Go structs
- Generate Rust structs  
- Generate TypeScript interfaces
- Migrate all usages to generated schemas
- Remove V1 schema files

---

## Feature Modules

### V1 (Deprecated)

**Location**: `services/features/`

**Problem**: Features scattered across multiple directories with inconsistent patterns

### V2 (Recommended)

**Location**: `services/integration/capability/`

**Registry**: `services/integration/capability/capability_registry.yaml`

**Benefits**:
- Centralized feature registration
- Consistent lifecycle management
- Unified configuration
- Better discoverability

### Migration Path

#### Step 1: Review Current Feature

Check if your feature in `services/features/` has unique logic or duplicates existing capability functionality.

#### Step 2: Register in Capability Registry

Add to `capability_registry.yaml`:

```yaml
capabilities:
  your_feature:
    name: "Your Feature Name"
    description: "Feature description"
    version: "1.0.0"
    module_path: "services.integration.capability.your_feature"
    enabled: true
    dependencies: []
    configuration:
      param1: value1
```

#### Step 3: Implement Capability Interface

```python
# services/integration/capability/your_feature.py
from .base import BaseCapability

class YourFeatureCapability(BaseCapability):
    def __init__(self, config):
        super().__init__(config)
    
    async def execute(self, task):
        # Feature implementation
        pass
```

#### Step 4: Test Integration

```python
from services.integration.capability.registry import CapabilityRegistry

registry = CapabilityRegistry()
capability = registry.get_capability('your_feature')
result = await capability.execute(task)
```

### Current Status

⚠️ **TODO**:
- Complete capability registry implementation
- Migrate features from `services/features/` to `services/integration/capability/`
- Remove `services/features/` directory after migration

---

## Data Storage

### V1 (Deprecated)

**File**: `services/integration/aiva_integration/ai_operation_recorder.py` (519 lines)

**Issues**:
- Local JSON file storage
- No database integration
- Limited query capabilities
- No transaction support

### V2 (Recommended)

**File**: `services/integration/aiva_integration/reception/experience_repository.py` (434 lines)

**Benefits**:
- SQLAlchemy ORM for type-safe database access
- Transaction support
- Rich query capabilities
- Centralized data access

### Migration Path

#### Old Way (V1):
```python
from services.integration.aiva_integration.ai_operation_recorder import AIOperationRecorder

recorder = AIOperationRecorder(output_dir="logs")
recorder.log_operation(
    operation_type="scan",
    details={"target": "example.com"},
    status="success"
)
```

#### New Way (V2):
```python
from services.integration.aiva_integration.reception import ExperienceRepository

repo = ExperienceRepository(database_url="sqlite:///aiva_experience.db")
repo.save_experience(
    plan_id="plan_001",
    attack_type="scan",
    ast_graph={},
    execution_trace={"target": "example.com"},
    metrics={"completion_rate": 1.0},
    feedback={"reward": 10}
)
```

### Migration Strategy

**Option 1: Adapter Pattern (Recommended for gradual migration)**

```python
# Create adapter that wraps both
class UnifiedRecorder:
    def __init__(self):
        self.experience_repo = ExperienceRepository(...)
        self.legacy_recorder = AIOperationRecorder(...)  # Fallback
    
    def record(self, operation):
        try:
            # Try V2 first
            self.experience_repo.save_experience(...)
        except Exception:
            # Fallback to V1
            self.legacy_recorder.log_operation(...)
```

**Option 2: Direct Migration (Breaking change)**

Replace all `AIOperationRecorder` imports with `ExperienceRepository`.

### Current Status

⚠️ **TODO**:
- Implement adapter pattern
- Migrate all `AIOperationRecorder` usages
- Deprecate `ai_operation_recorder.py`
- Document data migration from JSON to SQLite

---

## Testing V2 Architecture

### Run Tests

```bash
# Test cross-language service
python -m pytest tests/test_cross_language_service.py

# Test schema validation
python -m pytest tests/test_schemas.py

# Test capability registry
python -m pytest tests/test_capability_registry.py
```

### Verify Health

```bash
# Check gRPC service health
python -c "
from services.aiva_common.cross_language import get_cross_language_service
import asyncio

service = get_cross_language_service()
is_healthy = asyncio.run(service.health_check('localhost:50051'))
print(f'Service healthy: {is_healthy}')
"
```

---

## FAQ

### Q: Can I use both V1 and V2 simultaneously?

**A**: Yes! The migration is designed for gradual adoption. V2 code falls back to V1 when V2 infrastructure isn't available.

### Q: When will V1 be removed?

**A**: V1 will be deprecated after:
1. All gRPC services are implemented
2. All schemas are generated from SOT
3. All features are migrated to capability registry
4. All data access goes through ExperienceRepository

Expected: Q2 2026

### Q: What if gRPC server is not available?

**A**: The system automatically falls back to V1 (HTTP/subprocess) with a warning log.

### Q: How do I report issues?

**A**: Create an issue with label `v2-migration` in the GitHub repository.

---

## References

- V2 Architecture Design: `docs/ARCHITECTURE.md`
- Protocol Buffers Guide: `services/aiva_common/protocols/README.md`
- gRPC Best Practices: `docs/GRPC_BEST_PRACTICES.md`
- Schema SOT Documentation: `services/aiva_common/README_SCHEMA.md`

---

**Contributors**: AI Development Team  
**Contact**: [Create GitHub Issue](https://github.com/kyle0527/AIVA/issues)
