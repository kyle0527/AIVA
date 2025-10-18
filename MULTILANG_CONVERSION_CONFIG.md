# AIVA 多語言 Schema 轉換配置

## 🎯 支援的語言和工具

### 目前支援
| 語言 | 工具 | 配置檔案 | 輸出檔案 | 狀態 |
|------|------|----------|----------|------|
| JSON Schema | Pydantic API | - | `aiva_schemas.json` | ✅ 完成 |
| TypeScript | datamodel-code-generator | `pyproject.toml` | `aiva_schemas.d.ts` | ✅ 完成 |
| TypeScript Enums | 自訂生成器 | - | `enums.ts` | ✅ 完成 |
| Go | quicktype | - | `aiva_schemas.go` | ✅ 完成 |
| Rust | quicktype | - | `aiva_schemas.rs` | ✅ 完成 |

### 計劃支援
| 語言 | 工具 | 預期輸出 | 優先級 |
|------|------|----------|--------|
| Java | quicktype | `AivaSchemas.java` | 🔥 高 |
| C# | quicktype | `AivaSchemas.cs` | 🔥 高 |
| Swift | quicktype | `AivaSchemas.swift` | 🟡 中 |
| Kotlin | quicktype | `AivaSchemas.kt` | 🟡 中 |
| Dart | quicktype | `aiva_schemas.dart` | 🟢 低 |
| C++ | quicktype | `aiva_schemas.hpp` | 🟢 低 |

## 🔧 工具配置

### Pydantic 配置 (JSON Schema)
```python
# 在 Python Schema 類別中
class Config:
    json_schema_extra = {
        "example": {...},
        "$schema": "https://json-schema.org/draft/2020-12/schema"
    }
```

### datamodel-code-generator 配置 (TypeScript)
```toml
# pyproject.toml
[tool.datamodel-codegen]
input_file_type = "jsonschema"
output_model_type = "typing.TypedDict"
use_generic_container_types = true
use_union_operator = true
```

### quicktype 配置
```json
// quicktype.json (未來配置檔案)
{
  "go": {
    "package": "schemas",
    "just-types": true,
    "acronym-style": "pascal"
  },
  "rust": {
    "derive-debug": true,
    "derive-clone": true,
    "derive-partial-eq": true
  },
  "java": {
    "package": "com.aiva.schemas",
    "just-types": true
  },
  "csharp": {
    "namespace": "Aiva.Schemas",
    "just-types": true
  }
}
```

## 📋 新增語言支援流程

### 1. 評估階段
- [ ] 確認目標語言的生態系統需求
- [ ] 評估現有工具支援程度
- [ ] 確定輸出檔案格式和命名規範

### 2. 工具整合
- [ ] 測試 quicktype 對該語言的支援
- [ ] 如需要，開發自訂轉換器
- [ ] 整合到 `generate-official-contracts.ps1`

### 3. 品質保證
- [ ] 建立該語言的驗證測試
- [ ] 確保型別安全和序列化正確性
- [ ] 更新文檔和使用範例

### 4. 維護計劃
- [ ] 加入 CI/CD 自動化流程
- [ ] 建立錯誤監控和通知
- [ ] 定期更新工具版本

## 🚀 快速新增語言範例

### Java 支援
```powershell
# 在 generate-official-contracts.ps1 中新增
function Generate-JavaSchemas {
    Write-StepHeader "生成 Java Schema"
    $outputFile = "$OutputDir/AivaSchemas.java"
    
    & quicktype "$OutputDir/aiva_schemas.json" `
        --lang java `
        --package com.aiva.schemas `
        --class-map AivaSchemas `
        --just-types `
        --out $outputFile
        
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Java Schema 已生成: $outputFile"
        $size = (Get-Item $outputFile).Length
        Write-Host "   檔案大小: $([math]::Round($size/1024, 1)) KB"
    } else {
        Write-Error "Java Schema 生成失敗"
    }
}

# 新增參數
param(
    # ... 現有參數
    [switch]$GenerateJava
)

# 在主邏輯中新增
if ($GenerateJava -or $GenerateAll) {
    Generate-JavaSchemas
}
```

### C# 支援
```powershell
function Generate-CSharpSchemas {
    Write-StepHeader "生成 C# Schema"
    $outputFile = "$OutputDir/AivaSchemas.cs"
    
    & quicktype "$OutputDir/aiva_schemas.json" `
        --lang csharp `
        --namespace Aiva.Schemas `
        --class-map AivaSchemas `
        --features just-types `
        --out $outputFile
        
    if ($LASTEXITCODE -eq 0) {
        Write-Success "C# Schema 已生成: $outputFile"
    } else {
        Write-Error "C# Schema 生成失敗"
    }
}
```

## 🔄 轉換品質標準

### 必須支援的功能
- [x] 基本資料型別 (string, number, boolean)
- [x] 複合型別 (object, array)
- [x] 可選欄位 (nullable/optional)
- [x] 枚舉型別 (enum)
- [x] 時間型別 (datetime/timestamp)
- [ ] 泛型型別 (generics) - 計劃中
- [ ] 繼承關係 (inheritance) - 計劃中

### 程式碼品質要求
- [x] 符合目標語言的命名慣例
- [x] 包含完整的型別註解
- [x] 生成的程式碼可編譯
- [x] 支援序列化/反序列化
- [ ] 包含 JSDoc/註解 - 改進中
- [ ] 通過 Linter 檢查 - 改進中

### 效能要求
- [x] 生成時間 < 30 秒 (所有語言)
- [x] 檔案大小合理 (< 1MB 單一檔案)
- [x] 記憶體使用可控
- [ ] 支援增量生成 - 未來功能

## 📊 使用統計和監控

### 生成檔案統計
```powershell
# 統計腳本範例
$stats = @{}
Get-ChildItem "schemas" -File | ForEach-Object {
    $ext = $_.Extension
    $size = $_.Length
    
    if (-not $stats.ContainsKey($ext)) {
        $stats[$ext] = @{ Count = 0; TotalSize = 0 }
    }
    
    $stats[$ext].Count++
    $stats[$ext].TotalSize += $size
}

Write-Host "📊 生成檔案統計:" -ForegroundColor Cyan
$stats.GetEnumerator() | ForEach-Object {
    $avg = [math]::Round($_.Value.TotalSize / $_.Value.Count / 1024, 1)
    Write-Host "   $($_.Key): $($_.Value.Count) 個檔案, 平均 ${avg} KB"
}
```

### 錯誤監控
- 生成失敗率追蹤
- 檔案大小異常檢測
- 語法錯誤自動回報
- 效能指標監控

## 🛠️ 開發者工具

### 快速測試指令
```bash
# 測試單一語言生成
.\tools\generate-official-contracts.ps1 -GenerateTypeScript

# 驗證所有生成檔案
python tools\schema_manager.py validate

# 分析 Schema 影響
.\tools\analyze_schema_impact.ps1 -SchemaName "FindingPayload" -Action analyze
```

### 除錯工具
```powershell
# 詳細輸出模式
$DebugPreference = "Continue"
.\tools\generate-official-contracts.ps1 -GenerateAll -Verbose

# 檢查工具版本
quicktype --version
python -c "import pydantic; print(pydantic.__version__)"
datamodel-codegen --version
```

---

**維護**: 此配置檔案隨著新語言支援的加入而更新  
**版本**: 1.0  
**最後更新**: 2025年10月18日