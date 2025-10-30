# AIVA 跨語言函數處理最佳實踐指南

本文檔基於官方標準和 AIVA 專案分析，制定了跨語言函數處理的標準化規範。

## 1. 官方標準參考

### Python 函數設計標準
- **PEP 8**: 程式碼風格指南
- **PEP 484**: 型別提示規範
- **Pydantic**: 資料驗證和序列化
- **asyncio**: 非同步程式設計最佳實踐

### TypeScript 聲明檔案標準
- **Declaration Files**: `.d.ts` 檔案規範
- **Modules**: 模組型別聲明
- **Namespaces**: 命名空間組織
- **Compatibility**: 向前相容性

### Protocol Buffers 標準
- **Proto3 Language Guide**: 語法和最佳實踐
- **Style Guide**: 程式碼風格
- **Versioning**: 版本控制策略
- **Cross-Language Compatibility**: 跨語言相容性

## 2. 當前問題分析

### schemas/ 目錄問題

#### 2.1 代碼生成不一致
```
❌ 問題: 使用多種生成工具
   - TypeScript: "AIVA Official Tools"  
   - Go: "quicktype"
   
✅ 解決方案: 統一使用自定義生成工具
   - 建立 tools/schema_generator.py
   - 從 Python Pydantic 模型作為單一真實來源
   - 確保所有語言同步更新
```

#### 2.2 型別映射複雜
```
❌ 問題: Go 檔案中重複型別名稱
   type Defs struct {
       ModuleName     AssetType    // 錯誤映射
       ThreatLevel    AssetType    // 重複映射
   }

✅ 解決方案: 重構型別系統
   - 明確區分基礎型別和複合型別
   - 使用嚴格的命名規範
   - 建立型別映射表
```

#### 2.3 Protocol Buffers 改進
```
當前狀態: ✅ 基本結構正確
需要改進:
   - 添加版本控制欄位
   - 建立向前相容性策略
   - 完善文檔註釋
```

### services/aiva_common/ 目錄問題

#### 2.4 配置衝突
```python
❌ 問題: 多處重複配置
model_config = {"protected_namespaces": ()}

✅ 解決方案: 統一配置管理
# services/aiva_common/config/model_config.py
STANDARD_MODEL_CONFIG = {
    "protected_namespaces": (),
    "str_strip_whitespace": True,
    "validate_assignment": True,
    "arbitrary_types_allowed": False,
}
```

#### 2.5 跨語言同步
```
❌ 問題: 手動維護多語言定義
✅ 解決方案: 自動化同步流程
   1. Python Pydantic 模型作為主要來源
   2. 自動生成其他語言定義
   3. CI/CD 整合自動更新
```

## 3. 標準化規範

### 3.1 函數命名規範

#### Python 函數命名
```python
# ✅ 正確示例
def validate_vulnerability_data(payload: VulnerabilityPayload) -> ValidationResult:
    """驗證漏洞資料的完整性和格式"""
    pass

def process_scan_results(scan_id: str, results: List[ScanResult]) -> ProcessedResults:
    """處理掃描結果並生成報告"""
    pass

# ❌ 避免的命名
async def do_stuff():  # 不明確的命名
    pass
    
def ProcessData():  # 不符合 snake_case
    pass
```

#### TypeScript 型別命名
```typescript
// ✅ 正確示例
export interface VulnerabilityPayload {
  vulnerability_id: string;
  title: string;
  severity: "low" | "medium" | "high" | "critical";
}

export type ScanStatus = "pending" | "running" | "completed" | "failed";

// ❌ 避免的命名
interface Data {}  // 過於通用
interface vulnerability_payload {}  // 不符合 PascalCase
```

#### Protocol Buffers 命名
```protobuf
// ✅ 正確示例
message VulnerabilityReport {
  string vulnerability_id = 1;
  string title = 2;
  Severity severity = 3;
  repeated string tags = 4;
}

enum Severity {
  SEVERITY_UNSPECIFIED = 0;
  SEVERITY_LOW = 1;
  SEVERITY_MEDIUM = 2; 
  SEVERITY_HIGH = 3;
  SEVERITY_CRITICAL = 4;
}
```

### 3.2 錯誤處理標準

#### Python 錯誤處理
```python
from typing import Result, Ok, Err  # 使用 Result 型別

def process_security_data(data: dict) -> Result[ProcessedData, SecurityError]:
    """處理安全資料，返回結果或錯誤"""
    try:
        validated_data = SecurityDataModel.model_validate(data)
        processed = SecurityProcessor.process(validated_data)
        return Ok(processed)
    except ValidationError as e:
        return Err(SecurityError.ValidationFailed(str(e)))
    except ProcessingError as e:
        return Err(SecurityError.ProcessingFailed(str(e)))
```

#### TypeScript 錯誤處理
```typescript
// 定義統一的錯誤型別
export interface AIVAError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export type AIVAResult<T> = {
  success: true;
  data: T;
} | {
  success: false;
  error: AIVAError;
};
```

### 3.3 資料驗證標準

#### Pydantic 驗證器
```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Optional

class SecurityFinding(BaseModel):
    """安全發現標準模型"""
    
    model_config = STANDARD_MODEL_CONFIG
    
    finding_id: str = Field(min_length=1, description="發現唯一標識")
    title: str = Field(min_length=1, max_length=200, description="發現標題")
    severity: Literal["low", "medium", "high", "critical"] = Field(description="嚴重程度")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('finding_id')
    @classmethod
    def validate_finding_id(cls, v: str) -> str:
        if not v.startswith('finding_'):
            raise ValueError('finding_id must start with "finding_"')
        return v
```

## 4. 實施計劃

### 階段一：基礎重構 (Week 1)
1. **統一生成工具**
   - 完成 `tools/schema_generator.py` 開發
   - 建立從 Python 到其他語言的轉換邏輯
   - 測試生成結果的正確性

2. **配置標準化**
   - 建立 `services/aiva_common/config/model_config.py`
   - 更新所有 Python 模型使用統一配置
   - 驗證配置一致性

### 階段二：型別系統優化 (Week 2)
1. **重構複雜型別**
   - 簡化 Go 語言型別映射
   - 優化 TypeScript 型別定義
   - 確保跨語言型別一致性

2. **Protocol Buffers 改進**
   - 添加版本控制欄位
   - 完善服務定義
   - 建立向前相容性文檔

### 階段三：自動化流程 (Week 3)
1. **CI/CD 整合**
   - GitHub Actions 自動生成
   - 型別檢查和驗證
   - 自動同步更新

2. **文檔和測試**
   - 完善 API 文檔
   - 建立型別測試套件
   - 效能基準測試

## 5. 品質保證

### 5.1 型別安全檢查
```bash
# TypeScript 型別檢查
npm run type-check

# Python 型別檢查
mypy services/aiva_common/

# Protocol Buffers 編譯檢查
protoc --proto_path=schemas/crosslang schemas/crosslang/*.proto
```

### 5.2 相容性測試
```python
def test_cross_language_compatibility():
    """測試跨語言資料序列化相容性"""
    # Python -> JSON -> TypeScript
    python_data = SecurityFinding(...)
    json_data = python_data.model_dump_json()
    
    # 驗證 TypeScript 可以正確解析
    assert validate_typescript_parsing(json_data)
    
    # Python -> Protobuf -> Go
    proto_data = python_data.to_protobuf()
    assert validate_go_parsing(proto_data)
```

### 5.3 效能監控
```python
# 序列化效能測試
@benchmark
def test_serialization_performance():
    large_dataset = generate_test_data(10000)
    
    # 測試各種序列化方式的效能
    json_time = time_json_serialization(large_dataset)
    protobuf_time = time_protobuf_serialization(large_dataset)
    
    assert protobuf_time < json_time * 2  # Protobuf 應該更快
```

## 6. 監控和維護

### 6.1 型別版本控制
- 使用語義版本控制 (Semantic Versioning)
- 向前相容性保證
- 棄用政策和遷移路徑

### 6.2 效能監控
- 序列化/反序列化效能
- 記憶體使用情況
- 跨語言呼叫延遲

### 6.3 錯誤追蹤
- 型別轉換錯誤統計
- 相容性問題報告
- 自動修復建議

---

## 結論

本指南提供了完整的跨語言函數處理標準化方案。遵循這些最佳實踐將確保：

1. **型別安全**: 強型別檢查和驗證
2. **跨語言一致性**: 統一的資料結構和 API
3. **維護性**: 自動化生成和同步
4. **效能**: 優化的序列化和通訊
5. **可擴展性**: 易於添加新語言支援

建議按照實施計劃分階段執行，確保每個階段都有充分的測試和驗證。