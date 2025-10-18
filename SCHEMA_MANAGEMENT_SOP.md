# AIVA Schema 定義管理標準作業程序 (SOP)

## 📋 概述

本文件定義了 AIVA 專案中 Schema 定義的標準化管理流程，包括新增、修改、刪除定義的最佳實踐，以及多語言轉換的自動化流程。

## 🏗️ 架構原則

### 單一真實來源 (Single Source of Truth)
- **權威定義**: `services/aiva_common/schemas/` 和 `services/aiva_common/enums/`
- **生成檔案**: `schemas/` (多語言輸出)
- **原則**: 只修改權威定義，生成檔案通過工具自動更新

### 分層責任
```
[開發者] → [Python 定義] → [官方工具] → [多語言檔案] → [各語言專案]
```

## 📝 Schema 定義作業標準

### 🔸 新增 Schema 定義

#### 1. 確定分類
根據功能將新 Schema 歸類到適當檔案：

| 檔案 | 用途 | 範例 |
|------|------|------|
| `base.py` | 基礎模型 | MessageHeader, Authentication |
| `messaging.py` | 訊息系統 | AivaMessage, AIVARequest |
| `tasks.py` | 任務相關 | ScanTask, FunctionTask |
| `findings.py` | 漏洞發現 | FindingPayload, Vulnerability |
| `ai.py` | AI 相關 | BioNeuronConfig, TrainingData |
| `api_testing.py` | API 測試 | APITestCase, AuthZPayload |
| `assets.py` | 資產管理 | Asset, AssetInventory |
| `risk.py` | 風險評估 | RiskAssessment, AttackPath |
| `telemetry.py` | 遙測監控 | MetricsPayload, HealthCheck |

#### 2. 編寫 Schema 類別
```python
# 範例：在 findings.py 中新增
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class NewVulnerabilityType(BaseModel):
    """新的漏洞類型定義"""
    
    # 必填欄位
    vulnerability_id: str = Field(..., description="漏洞唯一識別碼")
    title: str = Field(..., description="漏洞標題")
    severity: Severity = Field(..., description="嚴重程度")
    
    # 選填欄位  
    description: Optional[str] = Field(None, description="詳細描述")
    cve_id: Optional[str] = Field(None, description="CVE 編號")
    discovered_at: datetime = Field(default_factory=datetime.now, description="發現時間")
    
    # 驗證規則
    @field_validator('vulnerability_id')
    @classmethod
    def validate_vuln_id(cls, v: str) -> str:
        if not v.startswith('AIVA-'):
            raise ValueError('漏洞 ID 必須以 AIVA- 開頭')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "vulnerability_id": "AIVA-2024-001",
                "title": "SQL 注入漏洞",
                "severity": "High",
                "description": "發現 SQL 注入攻擊向量"
            }
        }
```

#### 3. 更新 `__init__.py` 導出
```python
# 在相應的 __init__.py 中添加
from .findings import (
    # ... 現有匯出
    NewVulnerabilityType,  # 新增這行
)

# 更新 __all__ 列表
__all__ = [
    # ... 現有項目
    "NewVulnerabilityType",  # 新增這行
]
```

#### 4. 更新根目錄 `aiva_common/__init__.py`
```python
# 添加到相應的匯出區塊
from .schemas import (
    # ... 現有匯出
    NewVulnerabilityType,  # 新增
)
```

### 🔹 新增 Enum 定義

#### 1. 確定分類
| 檔案 | 用途 | 範例 |
|------|------|------|
| `common.py` | 通用枚舉 | Severity, Confidence, TaskStatus |
| `modules.py` | 模組相關 | ModuleName, Topic, ProgrammingLanguage |
| `security.py` | 安全測試 | VulnerabilityType, AttackVector |
| `assets.py` | 資產管理 | AssetType, Environment, BusinessCriticality |

#### 2. 編寫 Enum 類別
```python
# 範例：在 security.py 中新增
from enum import Enum

class NewAttackCategory(str, Enum):
    """新的攻擊分類枚舉"""
    
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XML_EXTERNAL = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    KNOWN_VULNERABILITIES = "known_vulnerabilities"
    INSUFFICIENT_LOGGING = "insufficient_logging"
```

### 🔻 刪除/棄用 Schema 定義

#### 1. 棄用標記（推薦）
```python
import warnings
from typing_extensions import deprecated

@deprecated("此類別將在 v2.0.0 中移除，請使用 NewVulnerabilityType")
class OldVulnerabilityType(BaseModel):
    """舊的漏洞類型定義（已棄用）"""
    
    def __init__(self, **data):
        warnings.warn(
            "OldVulnerabilityType 已棄用，請使用 NewVulnerabilityType",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**data)
```

#### 2. 段階性移除
```python
# 階段 1: 標記棄用 (v1.1.0)
@deprecated("...")
class OldType(BaseModel): ...

# 階段 2: 移除匯出 (v1.2.0) 
# 從 __init__.py 中移除，但保留定義

# 階段 3: 完全移除 (v2.0.0)
# 刪除類別定義
```

### 🔄 修改現有 Schema

#### 1. 向後兼容的修改
```python
class ExistingSchema(BaseModel):
    # 原有欄位
    existing_field: str
    
    # 新增選填欄位（向後兼容）
    new_optional_field: Optional[str] = Field(None, description="新增欄位")
    
    # 擴展現有欄位的值域（向後兼容）
    status: Union[OldStatusEnum, NewStatusEnum] = Field(..., description="狀態")
```

#### 2. 破壞性修改（需要版本控制）
```python
# 創建新版本
class ExistingSchemaV2(BaseModel):
    # 重新設計的欄位
    new_field_name: str  # 原 existing_field 重新命名
    enhanced_data: Dict[str, Any]  # 新的資料結構

# 保留舊版本並標記棄用
@deprecated("請使用 ExistingSchemaV2")
class ExistingSchema(BaseModel): ...
```

## 🛠️ 多語言轉換自動化

### 工具鏈配置

#### 1. 官方工具鏈
```powershell
# 完整生成（推薦用於發佈）
.\tools\generate-official-contracts.ps1 -GenerateAll

# 單一語言生成（開發時使用）
.\tools\generate-official-contracts.ps1 -GenerateJsonSchema
.\tools\generate-official-contracts.ps1 -GenerateTypeScript  
.\tools\generate-official-contracts.ps1 -GenerateGo -GenerateRust
```

#### 2. 支援的目標語言

| 語言 | 工具 | 輸出檔案 | 用途 |
|------|------|----------|------|
| JSON Schema | Pydantic API | `aiva_schemas.json` | OpenAPI, 驗證 |
| TypeScript | datamodel-code-generator | `aiva_schemas.d.ts` | 前端/Node.js |
| TypeScript Enums | 自訂生成器 | `enums.ts` | 前端枚舉 |
| Go | quicktype | `aiva_schemas.go` | Go 服務 |
| Rust | quicktype | `aiva_schemas.rs` | Rust 服務 |

### 新增語言支援

#### 1. 評估新語言需求
```powershell
# 檢查 quicktype 支援的語言
quicktype --help | Select-String "language"

# 支援的語言包括：
# - Java, C#, Swift, Kotlin
# - Dart, Objective-C, C++
# - 等更多語言
```

#### 2. 擴展生成腳本
```powershell
# 在 generate-official-contracts.ps1 中添加新語言支援
param(
    # ... 現有參數
    [switch]$GenerateJava,
    [switch]$GenerateCSharp
)

# 添加對應的函數
function Generate-JavaSchemas {
    Write-StepHeader "生成 Java Schema"
    $outputFile = "$OutputDir/aiva_schemas.java"
    
    & quicktype "$OutputDir/aiva_schemas.json" `
        --lang java `
        --package com.aiva.schemas `
        --class-map AivaSchemas `
        --out $outputFile
        
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Java Schema 生成完成: $outputFile"
    } else {
        Write-Error "Java Schema 生成失敗"
    }
}
```

#### 3. 自訂語言轉換器
```python
# tools/custom_language_generator.py
from typing import Dict, Any
import json

class CustomLanguageGenerator:
    """自訂語言 Schema 生成器"""
    
    def __init__(self, schema_file: str):
        with open(schema_file, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
    
    def generate_kotlin(self, output_file: str):
        """生成 Kotlin 數據類別"""
        kotlin_code = self._generate_kotlin_classes(self.schema['$defs'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(kotlin_code)
    
    def _generate_kotlin_classes(self, definitions: Dict[str, Any]) -> str:
        classes = []
        
        for name, definition in definitions.items():
            if definition.get('type') == 'object':
                kotlin_class = self._create_kotlin_data_class(name, definition)
                classes.append(kotlin_class)
        
        return '\n\n'.join([
            "package com.aiva.schemas",
            "",
            "import kotlinx.serialization.Serializable",
            "",
            *classes
        ])
    
    def _create_kotlin_data_class(self, name: str, definition: Dict[str, Any]) -> str:
        properties = definition.get('properties', {})
        required = definition.get('required', [])
        
        fields = []
        for field_name, field_def in properties.items():
            kotlin_type = self._map_json_type_to_kotlin(field_def)
            is_optional = field_name not in required
            
            if is_optional:
                kotlin_type = f"{kotlin_type}?"
            
            fields.append(f"    val {field_name}: {kotlin_type}")
        
        return f"""@Serializable
data class {name}(
{',\n'.join(fields)}
)"""
```

## 📋 開發工作流程

### 日常開發流程

#### 1. Schema 修改流程
```bash
# 1. 修改 Python 定義
# 編輯 services/aiva_common/schemas/*.py 或 enums/*.py

# 2. 驗證語法
python -c "from aiva_common.schemas import NewSchema; print('OK')"

# 3. 重新生成多語言檔案
.\tools\generate-official-contracts.ps1 -GenerateAll

# 4. 驗證生成結果
Get-ChildItem schemas | Select-Object Name, Length

# 5. 測試整合
python -m pytest tests/test_schemas.py
```

#### 2. 版本發佈流程
```bash
# 1. 完整回歸測試
python -m pytest

# 2. 生成所有語言定義
.\tools\generate-official-contracts.ps1 -GenerateAll

# 3. 驗證檔案完整性
.\tools\validate_generated_schemas.ps1

# 4. 更新版本號
# 編輯 services/aiva_common/__init__.py 中的 __version__

# 5. 提交變更
git add .
git commit -m "feat: update schemas v1.x.x"
git tag v1.x.x
```

### 品質保證檢查

#### 1. 自動化驗證腳本
```powershell
# tools/validate_schemas.ps1
Write-Host "🔍 驗證 Schema 定義..." -ForegroundColor Cyan

# 檢查 Python 語法
python -c "import aiva_common; print('Python schemas OK')"

# 檢查生成檔案是否最新
$lastPyEdit = (Get-ChildItem "services\aiva_common\schemas\*.py" | Sort-Object LastWriteTime -Descending)[0].LastWriteTime
$lastGenerated = (Get-ChildItem "schemas\aiva_schemas.json").LastWriteTime

if ($lastPyEdit -gt $lastGenerated) {
    Write-Warning "Python 定義比生成檔案新，建議重新生成"
    exit 1
}

Write-Host "✅ Schema 驗證通過" -ForegroundColor Green
```

#### 2. 單元測試範本
```python
# tests/test_schemas.py
import pytest
from aiva_common.schemas import NewVulnerabilityType
from aiva_common.enums import Severity

def test_new_vulnerability_type_creation():
    """測試新漏洞類型的建立"""
    vuln = NewVulnerabilityType(
        vulnerability_id="AIVA-2024-001",
        title="測試漏洞",
        severity=Severity.HIGH
    )
    
    assert vuln.vulnerability_id == "AIVA-2024-001"
    assert vuln.severity == Severity.HIGH

def test_vulnerability_id_validation():
    """測試漏洞 ID 驗證規則"""
    with pytest.raises(ValueError, match="必須以 AIVA- 開頭"):
        NewVulnerabilityType(
            vulnerability_id="INVALID-001",
            title="測試",
            severity=Severity.HIGH
        )

def test_schema_serialization():
    """測試 Schema 序列化"""
    vuln = NewVulnerabilityType(
        vulnerability_id="AIVA-2024-001",
        title="測試漏洞",
        severity=Severity.HIGH
    )
    
    json_data = vuln.model_dump()
    assert isinstance(json_data, dict)
    assert json_data['vulnerability_id'] == "AIVA-2024-001"
```

## 🔧 自動化腳本範本

### Schema 管理助手腳本
```python
#!/usr/bin/env python3
# tools/schema_manager.py
"""
AIVA Schema 管理助手
用於新增、修改、驗證 Schema 定義
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

class SchemaManager:
    def __init__(self, aiva_root: Path):
        self.aiva_root = aiva_root
        self.schemas_dir = aiva_root / "services" / "aiva_common" / "schemas"
        self.enums_dir = aiva_root / "services" / "aiva_common" / "enums"
    
    def create_new_schema(self, name: str, category: str, fields: Dict[str, str]):
        """創建新的 Schema 定義"""
        template = self._generate_schema_template(name, fields)
        
        target_file = self.schemas_dir / f"{category}.py"
        
        # 插入新 Schema 到適當位置
        content = target_file.read_text(encoding='utf-8')
        new_content = self._insert_schema(content, template)
        target_file.write_text(new_content, encoding='utf-8')
        
        print(f"✅ 新增 Schema: {name} 到 {category}.py")
    
    def validate_all_schemas(self):
        """驗證所有 Schema 定義"""
        try:
            # 動態導入檢查
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "aiva_common", 
                self.aiva_root / "services" / "aiva_common" / "__init__.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print("✅ 所有 Schema 定義語法正確")
            return True
            
        except Exception as e:
            print(f"❌ Schema 驗證失敗: {e}")
            return False
    
    def generate_multilang_schemas(self):
        """生成多語言 Schema 檔案"""
        import subprocess
        
        cmd = [
            "pwsh", "-File", 
            str(self.aiva_root / "tools" / "generate-official-contracts.ps1"),
            "-GenerateAll"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 多語言 Schema 生成完成")
        else:
            print(f"❌ 生成失敗: {result.stderr}")

def main():
    parser = argparse.ArgumentParser(description="AIVA Schema 管理工具")
    parser.add_argument("action", choices=["create", "validate", "generate"])
    parser.add_argument("--name", help="Schema 名稱")
    parser.add_argument("--category", help="Schema 分類")
    parser.add_argument("--fields", help="欄位定義 (JSON 格式)")
    
    args = parser.parse_args()
    
    aiva_root = Path(__file__).parent.parent
    manager = SchemaManager(aiva_root)
    
    if args.action == "create":
        if not all([args.name, args.category, args.fields]):
            print("創建 Schema 需要 --name, --category, --fields 參數")
            sys.exit(1)
        
        fields = json.loads(args.fields)
        manager.create_new_schema(args.name, args.category, fields)
        
    elif args.action == "validate":
        if not manager.validate_all_schemas():
            sys.exit(1)
            
    elif args.action == "generate":
        manager.generate_multilang_schemas()

if __name__ == "__main__":
    main()
```

### 使用範例
```bash
# 創建新 Schema
python tools/schema_manager.py create \
    --name "SecurityTestResult" \
    --category "findings" \
    --fields '{"test_id": "str", "result": "bool", "details": "Optional[str]"}'

# 驗證所有 Schema
python tools/schema_manager.py validate

# 生成多語言檔案
python tools/schema_manager.py generate
```

## 📚 最佳實踐總結

### ✅ 推薦做法
1. **統一命名規範**: 使用 PascalCase 類別名，snake_case 欄位名
2. **完整文檔**: 每個欄位都要有 `description`
3. **類型安全**: 使用具體的型別註解，避免 `Any`
4. **驗證規則**: 添加必要的 `field_validator`
5. **範例數據**: 在 `Config.json_schema_extra` 中提供範例
6. **版本控制**: 重大變更使用版本化類別名
7. **自動化**: 使用腳本自動生成和驗證

### ❌ 避免做法
1. **直接修改生成檔案**: 修改 `schemas/` 下的檔案
2. **破壞性變更**: 刪除必填欄位或改變型別
3. **循環依賴**: Schema 之間的循環引用
4. **過於複雜**: 單一 Schema 包含過多職責
5. **缺乏測試**: 新增 Schema 不寫對應測試

## 🎯 未來擴展計劃

### 短期目標 (1-3 個月)
- [ ] 完善自動化驗證腳本
- [ ] 增加更多語言支援 (Java, C#)
- [ ] 建立 Schema 變更影響分析工具
- [ ] 實作 Schema 版本相容性檢查

### 中期目標 (3-6 個月)
- [ ] 整合 CI/CD 自動化流程
- [ ] 建立 Schema 文檔自動生成
- [ ] 實作 Schema 遷移工具
- [ ] 增加效能最佳化和快取機制

### 長期目標 (6-12 個月)
- [ ] 建立 Schema 註冊中心和版本管理
- [ ] 實作跨服務 Schema 相容性監控
- [ ] 開發 Schema 視覺化編輯工具
- [ ] 建立企業級 Schema 治理框架

---

**維護團隊**: AIVA Development Team  
**文檔版本**: 1.0  
**最後更新**: 2025年10月18日