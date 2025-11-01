# Python to TypeScript Conversion Guide

## 📋 Overview

This guide demonstrates how to convert Python code and schemas to TypeScript using the AIVA Converters Plugin.

## 🚀 Quick Examples

### Schema Conversion

#### Python (Pydantic Model)
```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

class VulnerabilityFinding(BaseModel):
    id: str = Field(description="Unique finding identifier")
    title: str = Field(description="Vulnerability title")
    severity: Severity = Field(description="Severity level")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    affected_urls: List[str] = Field(default_factory=list)
    description: Optional[str] = None
```

#### Generated TypeScript (Interface)
```typescript
export enum Severity {
  CRITICAL = "critical",
  HIGH = "high", 
  MEDIUM = "medium",
  LOW = "low"
}

export interface VulnerabilityFinding {
  /** Unique finding identifier */
  id: string;
  
  /** Vulnerability title */
  title: string;
  
  /** Severity level */
  severity: Severity;
  
  /** Confidence score */
  confidence: number;
  
  /** URLs affected by this vulnerability */
  affected_urls: string[];
  
  /** Detailed description */
  description?: string | null;
}
```

## 🔧 Using the Plugin

### Automatic Schema Generation
```bash
# Generate TypeScript from Python schemas
python plugins/aiva_converters/core/schema_codegen_tool.py --lang typescript

# Generate from specific source
python plugins/aiva_converters/core/typescript_generator.py --input schemas/aiva_schemas.json
```

### Manual Conversion Process
```bash
# Step 1: Get conversion guidance
.\plugins\aiva_converters\scripts\language_converter.ps1 -SourceLang python -TargetLang typescript

# Step 2: Use schema generator
python plugins/aiva_converters/core/schema_codegen_tool.py --lang typescript --validate
```

## 📊 Type Mappings

| Python Type | TypeScript Type | Notes |
|------------|-----------------|--------|
| `str` | `string` | Direct mapping |
| `int`, `float` | `number` | Unified numeric type |
| `bool` | `boolean` | Direct mapping |
| `List[T]` | `T[]` | Array type |
| `Dict[str, T]` | `Record<string, T>` | Object type |
| `Optional[T]` | `T \| null` | Union with null |
| `datetime` | `string` | ISO 8601 format |
| `Enum` | `enum` | TypeScript enum |

## 🎯 Best Practices

### 1. **Maintain Schema Consistency**
```python
# Python - Use descriptive field names and types
class APIResponse(BaseModel):
    success: bool = Field(description="Operation success status")
    data: Optional[Any] = Field(None, description="Response data")
    error_message: Optional[str] = Field(None, description="Error details")
```

```typescript
// TypeScript - Generated interface maintains consistency
export interface APIResponse {
  /** Operation success status */
  success: boolean;
  
  /** Response data */
  data?: any | null;
  
  /** Error details */
  error_message?: string | null;
}
```

### 2. **Handle Async Operations**
```python
# Python async function
async def fetch_vulnerability_data(target_url: str) -> VulnerabilityFinding:
    """Fetch vulnerability data from target URL"""
    # Implementation
    pass
```

```typescript
// TypeScript equivalent
async function fetchVulnerabilityData(targetUrl: string): Promise<VulnerabilityFinding> {
  // Implementation
}
```

### 3. **Error Handling Patterns**
```python
# Python with exceptions
try:
    result = await scan_target(url)
    return APIResponse(success=True, data=result)
except ValidationError as e:
    return APIResponse(success=False, error_message=str(e))
```

```typescript
// TypeScript with Result pattern
async function scanTarget(url: string): Promise<APIResponse> {
  try {
    const result = await scanTargetImpl(url);
    return { success: true, data: result };
  } catch (error) {
    return { success: false, error_message: error.message };
  }
}
```

## 🔍 Advanced Conversion

### Complex Nested Types
```python
# Python nested model
class ScanResult(BaseModel):
    findings: List[VulnerabilityFinding] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    statistics: Optional["ScanStatistics"] = None  # Forward reference

class ScanStatistics(BaseModel):
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
```

```typescript
// TypeScript equivalent with proper references
export interface ScanResult {
  findings: VulnerabilityFinding[];
  metadata: Record<string, any>;
  statistics?: ScanStatistics | null;
}

export interface ScanStatistics {
  total_findings: number;
  critical_count: number; 
  high_count: number;
}
```

### Generic Types
```python
# Python generic model
from typing import TypeVar, Generic

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total_count: int
    page: int
    page_size: int
```

```typescript
// TypeScript generic interface
export interface PaginatedResponse<T> {
  items: T[];
  total_count: number;
  page: number;
  page_size: number;
}

// Usage
type VulnerabilityPage = PaginatedResponse<VulnerabilityFinding>;
```

## 🧪 Testing Conversions

### Validation Tests
```python
# Test conversion accuracy
from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator

validator = CrossLanguageValidator()
result = validator.validate_conversion(
    python_schema="VulnerabilityFinding",
    typescript_interface="VulnerabilityFinding"
)

assert result.is_valid
assert result.type_compatibility == "full"
```

### Runtime Validation
```typescript
// TypeScript runtime validation
function validateVulnerabilityFinding(data: any): data is VulnerabilityFinding {
  return (
    typeof data.id === 'string' &&
    typeof data.title === 'string' &&
    Object.values(Severity).includes(data.severity) &&
    typeof data.confidence === 'number' &&
    data.confidence >= 0 && data.confidence <= 1
  );
}
```

## 📈 Performance Considerations

### Serialization Performance
```python
# Python - Pydantic serialization
finding = VulnerabilityFinding(...)
json_data = finding.model_dump_json()  # Fast JSON serialization
```

```typescript
// TypeScript - JSON handling
const finding: VulnerabilityFinding = { ... };
const jsonData = JSON.stringify(finding);  // Native JSON serialization
```

### Bundle Size Optimization
```typescript
// Use tree shaking for imports
import type { VulnerabilityFinding } from './schemas';  // Type-only import
import { validateFinding } from './validators';  // Runtime import only when needed
```

## 🔧 Troubleshooting

### Common Issues

1. **Optional Field Handling**
   - Python `Optional[str]` → TypeScript `string | null`
   - Remember to handle null checks in TypeScript

2. **Date/Time Formats**
   - Python `datetime` → TypeScript `string` (ISO 8601)
   - Use proper date parsing in TypeScript

3. **Enum Values**
   - Ensure enum values match exactly between languages
   - Use string enums in TypeScript for JSON compatibility

### Debug Mode
```bash
# Enable debug logging during conversion
python plugins/aiva_converters/core/schema_codegen_tool.py --debug --lang typescript
```

---

**Example Updated**: November 2, 2025  
**Plugin Version**: 1.0.0