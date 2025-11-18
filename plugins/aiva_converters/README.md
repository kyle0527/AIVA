````markdown
# AIVA 轉換器插件 🔄

## 📚 目錄

- [🎯 插件概述](#-插件概述)
- [🔧 核心腳本功能簡述](#-核心腳本功能簡述)
- [📦 插件組件](#-插件組件)
- [🎯 核心特色](#-核心特色)
- [📁 插件結構](#-插件結構)
- [🚀 快速開始](#-快速開始)
- [🔧 進階使用](#-進階使用)
- [📋 配置](#-配置)
- [🎯 使用案例](#-使用案例)
- [📊 性能基準](#-性能基準)
- [🧪 測試](#-測試)
- [🔗 整合點](#-整合點)
- [📈 發展路線圖](#-發展路線圖)
- [🤝 貢獻](#-貢獻)
- [📚 文檔](#-文檔)
- [🔧 故障排除](#-故障排除)
- [🏆 品質提升里程碑](#-品質提升里程碑-v110)

---

## 🔧 核心腳本功能簡述

### 🚀 **核心生成工具**
| 腳本名稱 | 功能描述 | 主要用途 |
|---------|----------|----------|
| **`schema_codegen_tool.py`** | 🎯 多語言Schema生成器 | 基於YAML配置生成Python/TypeScript/Go/Rust代碼 |
| **`typescript_generator.py`** | 📝 TypeScript介面生成 | JSON Schema轉TypeScript定義，支援類型安全 |
| **`cross_language_validator.py`** | ✅ 跨語言驗證器 | 驗證多語言間類型一致性和兼容性 |
| **`schema_validator.py`** | 🔍 Schema驗證工具 | 驗證Schema合規性和格式正確性 |

### 🔄 **格式轉換器**
| 腳本名稱 | 功能描述 | 輸入格式 → 輸出格式 |
|---------|----------|-------------------|
| **`sarif_converter.py`** | 🛡️ 安全報告轉換器 | AIVA掃描結果 → SARIF 2.1.0標準格式 |
| **`task_converter.py`** | 📋 任務格式轉換器 | 各種任務格式 → 統一任務Schema |
| **`docx_to_md_converter.py`** | 📄 文件格式轉換器 | Word文檔(.docx) → Markdown(.md) |

### ⚙️ **自動化腳本**
| 腳本名稱 | 功能描述 | 執行環境 |
|---------|----------|----------|
| **`generate-contracts.ps1`** | 🏗️ 合約生成腳本 | 自動化生成JSON Schema、TypeScript定義和枚舉 |
| **`generate-official-contracts.ps1`** | 📋 官方合約生成 | 生成官方版本的合約文件和文檔 |

### 🔗 **跨語言支援**
| 腳本名稱 | 功能描述 | 支援特性 |
|---------|----------|----------|
| **`cross_language_interface.py`** | 🌐 跨語言介面 | 維護多語言間的API一致性和類型映射 |

---

## 🎯 插件概述

AIVA 轉換器插件是一個綜合性的轉換工具和生成器集合，用於在不同程式語言和格式之間轉換代碼、結構描述（Schema）和數據。此插件將 AIVA 中所有轉換相關的功能整合為可重複使用、可擴展的工具包。

**插件理念**: 實現程式語言、數據格式和結構描述定義之間的無縫轉換，同時維護 AIVA 基於合約驅動的架構原則。

---

## 📦 插件組件

### 🔄 **結構描述代碼生成**
- **核心工具**: `schema_codegen_tool.py` - 多語言結構描述生成器
- **支援語言**: Python (Pydantic v2)、Go (structs)、Rust (Serde)、TypeScript (interfaces)
- **資料來源**: 來自 `core_schema_sot.yaml` 的單一真相來源

### 🌐 **語言轉換器**
- **互動式轉換器**: `language_converter_final.ps1` - 互動式語言轉換指南
- **TypeScript 生成器**: `generate_typescript_interfaces.py` - JSON Schema 轉 TypeScript
- **跨語言介面**: 維護跨語言兼容性的工具

### 📋 **合約生成**
- **官方合約**: `generate-contracts.ps1` - 生成所有合約文件
- **Schema 驗證**: `schema_compliance_validator.py` - 驗證 Schema 合規性

### 🛠️ **實用工具轉換器**
- **SARIF 轉換器**: `sarif_converter.py` - 安全報告格式轉換
- **任務轉換器**: `task_converter.py` - 任務格式轉換
- **文件轉換器**: `docx_to_md_converter.py` - 文件格式轉換

---

## 🎯 核心特色

### ✅ **多語言支援**
- **Python**: 具備驗證功能的 Pydantic v2 模型
- **TypeScript**: 具備類型安全的介面定義  
- **Go**: 帶有 JSON 標籤的結構體定義
- **Rust**: 具備序列化功能的 Serde 兼容結構

### 🔧 **驗證與合規性**
- **Schema 驗證**: 自動驗證生成的 Schema
- **跨語言兼容性**: 確保語言間的一致性
- **性能基準測試**: 驗證轉換性能

### 🚀 **整合就緒**
- **VS Code 整合**: 兼容開發環境
- **CI/CD 就緒**: 建置管線中的自動生成
- **工具鏈整合**: 與現有 AIVA 工具配合使用

---

## 📁 插件結構

```
plugins/aiva_converters/
├── README.md                           # 本文件
├── core/                               # 核心轉換引擎
│   ├── schema_codegen_tool.py         # 多語言 Schema 生成器
│   ├── typescript_generator.py        # TypeScript 介面生成器  
│   ├── cross_language_validator.py    # 跨語言兼容性驗證
│   └── schema_validator.py            # Schema 驗證工具
├── converters/                         # 特定轉換器
│   ├── sarif_converter.py            # 安全報告轉換器
│   ├── task_converter.py             # 任務格式轉換器
│   └── docx_to_md_converter.py       # 文件轉換器
├── scripts/                           # 自動化腳本
│   ├── generate-contracts.ps1        # 合約生成腳本
│   └── generate-official-contracts.ps1 # 官方合約生成
├── templates/                         # 生成範本
│   ├── python/                       # Python 範本
│   ├── typescript/                   # TypeScript 範本
│   ├── go/                           # Go 範本
│   └── rust/                         # Rust 範本
├── tests/                            # 插件測試
│   ├── test_schema_codegen.py        # Schema 生成測試
│   └── test_conversions.py          # 轉換測試
└── examples/                         # 使用範例
    ├── python_to_typescript.md       # 轉換範例
    ├── schema_generation.md          # Schema 範例
    └── validation_examples.md        # 驗證範例
```

---

## 🚀 快速開始

### 📋 **前置需求**
```bash
# 確保已安裝必要的 Python 包
pip install pydantic>=2.0.0 jinja2>=3.0.0 pyyaml>=6.0.0 python-docx>=0.8.11
```

### 🔧 **環境設置**
```bash
# 導航到 AIVA 根目錄
cd C:\D\fold7\AIVA-git

# 安裝插件依賴項
pip install -r plugins/aiva_converters/requirements.txt

# 設置 Python 路徑（如需要）
$env:PYTHONPATH = "C:\D\fold7\AIVA-git\services"
```

### 🎯 **核心腳本使用範例**

#### 1️⃣ **多語言 Schema 生成**
```bash
# 🔄 生成所有支援的語言 Schema
python plugins/aiva_converters/core/schema_codegen_tool.py --generate-all

# 🐍 生成 Python Pydantic 模型
python plugins/aiva_converters/core/schema_codegen_tool.py --lang python

# 📝 生成 TypeScript 介面
python plugins/aiva_converters/core/schema_codegen_tool.py --lang typescript

# 🦫 生成 Go 結構體
python plugins/aiva_converters/core/schema_codegen_tool.py --lang go

# 🦀 生成 Rust 結構體
python plugins/aiva_converters/core/schema_codegen_tool.py --lang rust

# ✅ 生成並驗證
python plugins/aiva_converters/core/schema_codegen_tool.py --lang python --validate
```

#### 2️⃣ **自動化合約生成**
```powershell
# 🏗️ 生成所有合約文件
.\plugins\aiva_converters\scripts\generate-contracts.ps1 -GenerateAll

# 📋 僅生成 JSON Schema
.\plugins\aiva_converters\scripts\generate-contracts.ps1 -GenerateJsonSchema

# 📝 僅生成 TypeScript 定義
.\plugins\aiva_converters\scripts\generate-contracts.ps1 -GenerateTypeScript

# 🔢 僅生成枚舉
.\plugins\aiva_converters\scripts\generate-contracts.ps1 -GenerateEnums

# 📂 指定輸出目錄
.\plugins\aiva_converters\scripts\generate-contracts.ps1 -GenerateAll -OutputDir ".\custom_output"
```

#### 3️⃣ **格式轉換工具**
```python
# 🛡️ SARIF 安全報告轉換
from plugins.aiva_converters.converters.sarif_converter import SARIFConverter

converter = SARIFConverter()
sarif_report = converter.convert_aiva_results("scan_results.json")

# 📋 任務格式轉換
from plugins.aiva_converters.converters.task_converter import TaskConverter

task_converter = TaskConverter()
converted_task = task_converter.convert("task.json", "target_format")

# 📄 Word 轉 Markdown
from plugins.aiva_converters.converters.docx_to_md_converter import DocxToMdConverter

doc_converter = DocxToMdConverter()
markdown_content = doc_converter.convert("document.docx")
```

#### 4️⃣ **跨語言驗證**
```python
# ✅ 驗證多語言類型一致性
from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator

validator = CrossLanguageValidator()

# 驗證 Python 到 TypeScript 轉換
validation_result = validator.validate_conversion("models.py", "interfaces.ts")

# 驗證所有語言兼容性
compatibility_report = validator.validate_all_languages()
```

---

## 🔧 進階使用

### 自定義 Schema 生成
```python
from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodeGenerator

# 初始化生成器
generator = SchemaCodeGenerator("custom_schema.yaml")

# 生成特定語言
python_files = generator.generate_python_schemas("./output/python")
typescript_files = generator.generate_typescript_schemas("./output/ts")
```

### 轉換驗證
```python
from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator

# 驗證轉換結果
validator = CrossLanguageValidator()
validation_result = validator.validate_conversion("source.py", "target.ts")
```

---

## 📋 配置

### Schema 生成配置
插件使用 `core_schema_sot.yaml` 作為單一真相來源。配置嵌入在 YAML 文件的 `generation_config` 下：

```yaml
generation_config:
  python:
    target_dir: "services/aiva_common/schemas/generated"
    base_imports:
      - "from pydantic import BaseModel, Field"
      - "from typing import Optional, List, Dict, Any"
  
  typescript:
    target_dir: "services/features/common/typescript/aiva_common_ts/schemas/generated"
    
  go:
    target_dir: "services/features/common/go/aiva_common_go/schemas/generated"
    
  rust:
    target_dir: "services/features/common/rust/aiva_common_rust/schemas/generated"
```

---

## 🎯 使用案例

### 1. **多語言專案開發**
- 在 Python、TypeScript、Go 和 Rust 之間生成一致的 Schema
- 維護跨語言邊界的類型安全
- 確保服務間的合約合規性

### 2. **遺留代碼遷移**
- 將現有的 Python 模型轉換為 TypeScript 介面
- 將 Go 結構體轉換為 Rust 結構
- 在不同 Schema 格式之間遷移

### 3. **API 合約生成**
- 從 Schema 定義生成客戶端 SDK
- 從 Schema 元數據創建文檔
- 驗證 API 版本間的兼容性

### 4. **安全報告處理**
- 將 SARIF 報告轉換為各種格式
- 在工具間轉換漏洞數據
- 跨平台標準化安全發現

---

## 📊 性能基準

插件維護 AIVA 的性能標準：

- **JSON 序列化**: 8,536+ 次操作/秒基準
- **Schema 生成**: 典型 Schema 生成時間少於 1 秒
- **跨語言驗證**: 最小開銷
- **記憶體使用**: 針對大型 Schema 集合進行優化

---

## 🧪 測試

### 執行插件測試
```bash
# 執行所有插件測試
python -m pytest plugins/aiva_converters/tests/ -v

# 執行特定測試類別
python -m pytest plugins/aiva_converters/tests/test_schema_codegen.py -v
```

### 驗證測試
```bash
# 驗證生成的 Schema
python plugins/aiva_converters/core/schema_codegen_tool.py --validate

# 跨語言兼容性測試
python plugins/aiva_converters/tests/test_cross_language_compatibility.py
```

---

## 🔗 整合點

### 與 AIVA 核心
- **合約系統**: 使用 `aiva_common.schemas` 作為基礎
- **性能標準**: 維護 6.7 倍性能優勢
- **架構合規性**: 遵循合約驅動的設計原則

### 與開發工作流程
- **VS Code 整合**: 兼容 Pylance、Go 擴充功能、Rust 分析器
- **CI/CD 整合**: 在建置管線中自動生成 Schema
- **文檔生成**: 自動生成 API 文檔

### 與外部工具
- **Schema 驗證器**: JSON Schema、OpenAPI、Protocol Buffers
- **代碼生成器**: 兼容外部代碼生成工具
- **建置系統**: 與 Make、Gradle、Cargo、npm 整合

---

## 📈 發展路線圖

### 第一階段：核心整合 ✅
- [x] 整合現有轉換工具
- [x] 創建統一插件結構
- [x] 建立測試框架

### 第二階段：增強生成 (2025年第一季)
- [ ] 基於範本的生成系統
- [ ] 自定義驗證規則
- [ ] 性能優化

### 第三階段：進階功能 (2025年第二季)  
- [ ] AI 輔助轉換建議
- [ ] 視覺化 Schema 設計器整合
- [ ] 即時驗證回饋

### 第四階段：企業功能 (2025年第三季)
- [ ] 企業級 Schema 註冊中心整合
- [ ] 進階版本控制支援
- [ ] 分散式生成能力

---

## 🤝 貢獻

### 新增轉換器
1. 在 `converters/` 目錄中創建轉換器
2. 遵循現有模式和介面
3. 新增完整測試
4. 更新文檔

### 擴展語言支援
1. 在 `core_schema_sot.yaml` 中新增語言配置
2. 在 `schema_codegen_tool.py` 中實現生成邏輯
3. 創建語言特定範本
4. 新增驗證測試

---

## 📚 文檔

- **[Schema 生成指南](./examples/schema_generation.md)** - 完整的 Schema 生成
- **[語言轉換指南](./examples/python_to_typescript.md)** - 特定語言轉換
- **[驗證範例](./examples/validation_examples.md)** - 驗證和測試模式
- **[性能優化](./docs/performance.md)** - 優化策略

---

## 🔧 故障排除

### 常見問題
1. **Schema 生成失敗**: 檢查 `core_schema_sot.yaml` 語法
2. **類型映射錯誤**: 驗證語言特定的類型映射
3. **性能問題**: 檢查生成批次大小

### 除錯模式
```bash
# 啟用除錯日誌
python plugins/aiva_converters/core/schema_codegen_tool.py --debug --lang python
```

---

## 📖 **腳本詳細參考指南**

### 🔧 **schema_codegen_tool.py 參數詳解**
```bash
python plugins/aiva_converters/core/schema_codegen_tool.py [選項]

# 主要參數
--generate-all          # 生成所有支援的語言
--lang LANGUAGE         # 指定特定語言 (python|typescript|go|rust)
--output-dir DIR        # 自定義輸出目錄
--validate              # 執行驗證檢查
--debug                 # 啟用詳細除錯輸出
--force                 # 強制覆蓋現有文件
--dry-run               # 預覽模式，不實際生成文件

# 範例組合
python schema_codegen_tool.py --lang python --validate --debug
python schema_codegen_tool.py --generate-all --output-dir ./schemas
python schema_codegen_tool.py --lang typescript --dry-run
```

### 🏗️ **generate-contracts.ps1 參數詳解**
```powershell
.\plugins\aiva_converters\scripts\generate-contracts.ps1 [參數]

# 主要參數
-ListModels             # 列出所有可用的模型
-GenerateAll            # 生成所有類型的合約
-GenerateJsonSchema     # 僅生成 JSON Schema
-GenerateTypeScript     # 僅生成 TypeScript 定義
-GenerateEnums          # 僅生成枚舉定義
-OutputDir PATH         # 指定輸出目錄（默認: .\schemas）

# 範例組合
.\generate-contracts.ps1 -GenerateAll -OutputDir "C:\output"
.\generate-contracts.ps1 -GenerateTypeScript -ListModels
.\generate-contracts.ps1 -GenerateJsonSchema
```

### 🔍 **驗證工具使用方式**
```python
# Schema 驗證工具
python plugins/aiva_converters/core/schema_validator.py [檔案路徑]

# 跨語言驗證器
from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator
validator = CrossLanguageValidator()

# 驗證單一轉換
result = validator.validate_conversion("source.py", "target.ts")

# 驗證所有語言兼容性
report = validator.validate_all_languages()
```

### 🔄 **轉換器工具使用**
```python
# SARIF 轉換器
from plugins.aiva_converters.converters.sarif_converter import SARIFConverter
converter = SARIFConverter()
sarif_data = converter.convert_aiva_results("results.json")

# 任務轉換器
from plugins.aiva_converters.converters.task_converter import TaskConverter
converter = TaskConverter()
converted = converter.convert("input.json", "output_format")

# 文件轉換器
from plugins.aiva_converters.converters.docx_to_md_converter import DocxToMdConverter
converter = DocxToMdConverter()
markdown = converter.convert("document.docx")
```

### ⚠️ **常見錯誤與解決方案**
```bash
# 錯誤: ModuleNotFoundError
解決: 確保 PYTHONPATH 正確設置
$env:PYTHONPATH = "C:\D\fold7\AIVA-git\services"

# 錯誤: 權限拒絕
解決: 以管理員權限執行 PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 錯誤: YAML 解析失敗
解決: 檢查 core_schema_sot.yaml 檔案格式
python -c "import yaml; yaml.safe_load(open('core_schema_sot.yaml'))"

# 錯誤: Jinja2 模板錯誤
解決: 確保模板目錄存在且可訪問
ls plugins/aiva_converters/templates/
```

---

## 🏆 品質提升里程碑 (v1.1.0)

> **重大品質提升**: 2025年11月3日完成核心工具認知複雜度重構

### ✅ **schema_codegen_tool.py 品質強化**
- **複雜度優化**: 6 個核心函數從 15+ 複雜度降至 ≤15
- **穩定性提升**: 通過 SonarQube 100% 品質檢查
- **維護性增強**: 45+ 輔助函數提取，職責分離清晰
- **功能保證**: 保持 Python/Go/Rust/TypeScript 完整生成能力

### 🔧 **重構技術應用**
- **Extract Method Pattern**: 大型函數分解為專門化小函數
- **Strategy Pattern**: 複雜條件判斷用策略模式替代
- **Early Return Pattern**: 減少嵌套層級和認知負擔
- **字串常量管理**: 統一常量定義，提升維護性

### 🎯 **品質指標達成**
| 指標 | 重構前 | 重構後 | 改善幅度 |
|------|--------|--------|----------|
| 最高複雜度 | 29 | ≤15 | 48%+ 降低 |
| SonarQube 錯誤 | 7 個 | 0 個 | 100% 修復 |
| 輔助函數 | 12 個 | 45+ 個 | 275% 增加 |
| 代碼可讀性 | 中等 | 優秀 | 顯著提升 |

### 🚀 **對統一通信架構的貢獻**
- **基礎穩固**: 為 AIVA 統一通信架構提供可靠的代碼生成基礎
- **品質保證**: 確保跨語言架構實施的代碼品質標準
- **工具鏈穩定**: 支撐 Schema SoT 和多語言綁定的核心引擎

---

## ⚠️ **故障排除**

### 常見問題
1. **模組無法找到**: 確保 PYTHONPATH 包含項目根目錄
2. **權限錯誤**: 以管理員身份運行 PowerShell
3. **YAML 語法錯誤**: 檢查 `core_schema_sot.yaml` 語法

### 除錯模式
```bash
# 啟用除錯日誌
python plugins/aiva_converters/core/schema_codegen_tool.py --debug --lang python
```

---

## 🎯 **最佳實踐**

### 🔄 **Schema 開發流程**
1. **設計 Schema**: 在 `core_schema_sot.yaml` 中定義數據模型
2. **生成代碼**: 使用 schema_codegen_tool 生成多語言實現
3. **驗證**: 使用 schema_validator 驗證生成的代碼
4. **測試**: 在目標語言中進行整合測試
5. **部署**: 將生成的代碼整合到項目中

### 📝 **Schema 設計指南**
```yaml
# 良好的 Schema 設計範例
ModelName:
  type: object
  required:
    - id
    - name
  properties:
    id:
      type: string
      pattern: "^[a-zA-Z0-9-]+$"
      description: "唯一識別符"
    name:
      type: string
      minLength: 1
      maxLength: 100
      description: "顯示名稱"
    metadata:
      $ref: "#/components/schemas/Metadata"
```

### 🚀 **效能優化建議**
1. **批次處理**: 使用 `--generate-all` 一次生成所有語言
2. **快取**: 啟用模板快取以提升性能
3. **漸進式更新**: 僅重新生成修改的部分

### 🔐 **安全考量**
1. **輸入驗證**: 始終驗證 Schema 輸入
2. **檔案權限**: 適當設置生成檔案的權限
3. **敏感資料**: 避免在 Schema 中硬編碼敏感資訊

### 🔧 **維護提示**
- 定期更新 Jinja2 模板以支持新功能
- 保持 Schema 版本控制和向後兼容性
- 建立自動化測試以驗證跨語言一致性

---

**插件維護者**: AIVA 架構團隊  
**版本**: 1.1.0 (品質提升版)  
**最後更新**: 2025年11月10日  
**兼容性**: AIVA Core 2.x+  
**品質狀態**: ✅ SonarQube 100% 合規 | ✅ 認知複雜度 ≤15

---

**🌟 如果此插件對您有幫助，請給我們一個 Star！**

---

[← 返回 Plugins 主目錄](../README.md)
````