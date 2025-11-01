# AIVA 數據合約架構文檔

## 📋 概覽

AIVA (AI Vulnerability Assessment) 採用完整的數據合約架構，確保微服務系統各組件間的數據一致性和互操作性。本文檔詳細說明 AIVA 系統中所有數據合約的設計、實現和使用方式。

## 🏗️ 架構設計原則

### 單一事實來源 (Single Source of Truth)
- **主要位置**: `services/aiva_common/schemas/`
- **統一管理**: 所有數據合約集中定義，避免重複和不一致
- **版本控制**: 統一的版本管理和向後相容策略

### 跨語言一致性
- **Python**: 使用 Pydantic v2 定義標準模型
- **TypeScript**: 自動生成 `.d.ts` 型別定義
- **Go**: 生成對應的 struct 定義
- **Rust**: 生成 serde 相容的結構體

## 📊 合約分類體系

### 1. 基礎設施合約 (Infrastructure Contracts)

#### 📦 基礎模型 (`services/aiva_common/schemas/base.py`)
```python
# 核心基礎類型
- MessageHeader        # 統一訊息標頭
- Authentication       # 認證資訊
- ScanScope           # 掃描範圍
- Asset               # 資產定義
- RateLimit           # 速率限制
- ExecutionError      # 執行錯誤
```

#### 🔄 訊息系統 (`services/aiva_common/schemas/messaging.py`)
```python
# 統一訊息格式
- AivaMessage         # AIVA 統一訊息格式
- AIVARequest         # 請求消息
- AIVAResponse        # 響應消息
- AIVAEvent          # 事件消息
- AIVACommand        # 命令消息
```

### 2. 業務領域合約 (Domain Contracts)

#### 🔍 漏洞發現 (`services/aiva_common/schemas/findings.py`)
```python
# 漏洞相關模型
- FindingPayload      # 漏洞發現數據
- Vulnerability       # 漏洞詳情
- VulnerabilityCorrelation  # 漏洞關聯
- FindingEvidence     # 漏洞證據
- FindingImpact       # 影響評估
```

#### 🤖 AI 系統 (`services/aiva_common/schemas/ai.py`)
```python
# AI 相關模型
- AITrainingPayload   # AI 訓練數據
- AIVerificationRequest  # AI 驗證請求
- AttackPlan         # 攻擊規劃
- ExperienceSample   # 經驗樣本
- ModelTrainingConfig # 模型訓練配置
```

#### 🛡️ 威脅情報 (`services/aiva_common/schemas/threat_intelligence.py`)
```python
# STIX/TAXII 相容模型
- Indicator          # 威脅指標
- ThreatActor       # 威脅行為者
- Campaign          # 攻擊活動
- IOCEnrichment     # IOC 豐富化
- ThreatIntelligenceReport # 威脅情報報告
```

### 3. API 標準合約 (API Standards)

#### 🌐 API 規範 (`services/aiva_common/schemas/api_standards.py`)
```python
# 多協議支援
- OpenAPIDocument    # OpenAPI 3.1 規範
- AsyncAPIDocument   # AsyncAPI 3.0 規範  
- GraphQLSchema     # GraphQL Schema 定義
- APISecurityTest   # API 安全測試
```

### 4. 任務編排合約 (Task Orchestration)

#### ⚙️ 任務系統 (`services/aiva_common/schemas/tasks.py`)
```python
# 任務相關模型
- ScanStartPayload   # 掃描啟動
- FunctionTaskPayload # 功能任務
- ExploitPayload    # 漏洞利用
- TestExecution     # 測試執行
```

## 🛠️ 工具鏈生態

### 自動化生成工具

#### 1. AIVA Contracts Tooling
```bash
# 安裝位置: tools/integration/aiva-contracts-tooling/
aiva-contracts list-models                    # 列出所有模型
aiva-contracts export-jsonschema --out schema.json  # 生成 JSON Schema
aiva-contracts gen-ts --json schema.json --out types.d.ts  # 生成 TypeScript
```

#### 2. PowerShell 自動化腳本
```powershell
# 位置: tools/common/automation/
.\generate-contracts.ps1 -GenerateAll        # 生成所有合約文件
.\generate-official-contracts.ps1            # 生成官方合約
```

### 驗證和合規工具

#### Schema 驗證器
```bash
# 位置: tools/common/schema/schema_validator.py
python schema_validator.py --validate-all    # 驗證所有 schema
```

#### 跨語言合規檢查
```bash
# 位置: tools/schema_compliance_validator.py  
python schema_compliance_validator.py --check-all  # 檢查所有語言
python schema_compliance_validator.py --language go # 檢查特定語言
```

## 📈 版本管理策略

### 版本號系統
- **Schema 版本**: 1.1.0 (當前)
- **API 版本**: v1, v2 (向後相容)
- **語義版本**: MAJOR.MINOR.PATCH

### 向後相容性
```python
# 字段棄用策略
class ExampleModel(BaseModel):
    new_field: str = Field(description="新字段")
    old_field: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="已棄用，請使用 new_field"
    )
```

### 遷移指南
1. **新增字段**: 使用 Optional 類型，提供預設值
2. **棄用字段**: 標記 `deprecated=True`，保留一個版本週期  
3. **破壞性變更**: 增加 MAJOR 版本號

## 🔄 CI/CD 集成

### 自動化檢查
```yaml
# .github/workflows/contracts-check.yml
- name: Schema Compliance Check
  run: python tools/schema_compliance_validator.py --ci-mode
  
- name: Generate Updated Contracts  
  run: |
    aiva-contracts export-jsonschema --out schemas/aiva_schemas.json
    aiva-contracts gen-ts --json schemas/aiva_schemas.json --out schemas/aiva_schemas.d.ts
```

### 合規閾值
- **合規分數**: >= 95%
- **強制檢查**: 所有 PR 必須通過合規檢查
- **自動修復**: 部分問題支援自動修復

## 🎯 使用最佳實踐

### 開發者指南

#### 1. 創建新合約
```python
# 1. 在適當的 schemas 模組中定義
class NewContract(BaseModel):
    """新合約說明"""
    field1: str = Field(description="字段說明")
    field2: Optional[int] = Field(default=None, description="可選字段")

# 2. 更新 __init__.py 導出
__all__ = [..., "NewContract"]

# 3. 運行生成工具
# aiva-contracts export-jsonschema --out schemas/aiva_schemas.json
```

#### 2. 修改現有合約
```python
# 向前相容的修改
class ExistingContract(BaseModel):
    existing_field: str
    new_optional_field: Optional[str] = Field(default=None)  # 新增可選字段
    
    @field_validator('existing_field')
    @classmethod
    def validate_existing_field(cls, v):
        # 新增驗證邏輯
        return v
```

#### 3. 跨服務使用
```python
# Python 服務
from aiva_common.schemas import FindingPayload, ScanStartPayload

# TypeScript 服務
import { FindingPayload, ScanStartPayload } from '@aiva/contracts';

# Go 服務  
import "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
```

### 命名規範
- **類名**: PascalCase (例: `FindingPayload`)
- **字段名**: snake_case (例: `finding_id`)
- **枚舉**: UPPER_CASE (例: `CRITICAL`, `HIGH`)

## 📚 相關資源

### 文檔連結
- [Pydantic 官方文檔](https://docs.pydantic.dev/)
- [OpenAPI 規範](https://spec.openapis.org/oas/v3.1.0)
- [AsyncAPI 規範](https://www.asyncapi.com/docs/reference)
- [STIX 2.1 規範](https://docs.oasis-open.org/cti/stix/v2.1/)

### 專案內部文檔
- `services/aiva_common/schemas/__init__.py` - 所有可用合約列表
- `tools/integration/aiva-contracts-tooling/README.md` - 工具使用說明
- `_archive/deprecated_schema_tools/CLEANUP_RECORD.md` - 歷史清理記錄

## 🔧 故障排除

### 常見問題

#### 1. 合約導入錯誤
```python
# ❌ 錯誤方式
from services.aiva_common.schemas.findings import FindingPayload

# ✅ 正確方式  
from aiva_common.schemas import FindingPayload
```

#### 2. 版本不匹配
```bash
# 檢查當前版本
python -c "from aiva_common.schemas import __version__; print(__version__)"

# 更新合約文件
aiva-contracts export-jsonschema --out schemas/aiva_schemas.json
```

#### 3. 跨語言類型錯誤
```bash
# 檢查合規性
python tools/schema_compliance_validator.py --language typescript

# 重新生成語言綁定
aiva-contracts gen-ts --json schemas/aiva_schemas.json --out schemas/aiva_schemas.d.ts
```

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025-11-01  
**文檔版本**: 1.0.0