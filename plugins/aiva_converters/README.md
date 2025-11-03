# AIVA Converters Plugin

## 🔧 Plugin Overview

The AIVA Converters Plugin is a comprehensive collection of conversion tools and generators that transform code, schemas, and data between different languages and formats. This plugin consolidates all conversion-related functionality from AIVA into a reusable, extensible toolkit.

**Plugin Philosophy**: Enable seamless transformation between different programming languages, data formats, and schema definitions while maintaining AIVA's contract-driven architecture principles.

## 📦 Plugin Components

### 🔄 **Schema Code Generation**
- **Core Tool**: `schema_codegen_tool.py` - Multi-language schema generator
- **Supports**: Python (Pydantic v2), Go (structs), Rust (Serde), TypeScript (interfaces)
- **Source**: Single source of truth from `core_schema_sot.yaml`

### 🌐 **Language Converters**
- **Interactive Converter**: `language_converter_final.ps1` - Interactive language conversion guide
- **TypeScript Generator**: `generate_typescript_interfaces.py` - JSON Schema to TypeScript
- **Cross-Language Interface**: Tools for maintaining compatibility across languages

### 📋 **Contract Generation**
- **Official Contracts**: `generate-contracts.ps1` - Generate all contract files
- **Schema Validation**: `schema_compliance_validator.py` - Validate schema compliance

### 🛠️ **Utility Converters**
- **SARIF Converter**: `sarif_converter.py` - Security report format conversion
- **Task Converter**: `task_converter.py` - Task format transformation
- **Document Converter**: `docx_to_md_converter.py` - Document format conversion

## 🎯 Key Features

### ✅ **Multi-Language Support**
- **Python**: Pydantic v2 models with validation
- **TypeScript**: Interface definitions with type safety  
- **Go**: Struct definitions with JSON tags
- **Rust**: Serde-compatible structures with serialization

### 🔧 **Validation & Compliance**
- **Schema Validation**: Automatic validation of generated schemas
- **Cross-Language Compatibility**: Ensure consistency across languages
- **Performance Benchmarking**: Validate conversion performance

### 🚀 **Integration Ready**
- **VS Code Integration**: Compatible with development environment
- **CI/CD Ready**: Automated generation in build pipelines
- **Tool Chain Integration**: Works with existing AIVA tooling

## 📁 Plugin Structure

```
plugins/aiva_converters/
├── README.md                           # This file
├── core/                               # Core conversion engines
│   ├── schema_codegen_tool.py         # Multi-language schema generator
│   ├── typescript_generator.py        # TypeScript interface generator  
│   └── cross_language_validator.py    # Cross-language compatibility
├── converters/                         # Specific converters
│   ├── sarif_converter.py            # Security report converter
│   ├── task_converter.py             # Task format converter
│   └── docx_to_md_converter.py       # Document converter
├── scripts/                           # Automation scripts
│   ├── language_converter_final.ps1   # Interactive language converter
│   ├── generate-contracts.ps1        # Contract generation
│   └── generate-official-contracts.ps1 # Official contract generation
├── templates/                         # Generation templates
│   ├── python/                       # Python templates
│   ├── typescript/                   # TypeScript templates
│   ├── go/                           # Go templates
│   └── rust/                         # Rust templates
├── tests/                            # Plugin tests
│   ├── test_schema_codegen.py        # Schema generation tests
│   └── test_conversions.py          # Conversion tests
└── examples/                         # Usage examples
    ├── python_to_typescript.md       # Conversion examples
    ├── schema_generation.md          # Schema examples
    └── validation_examples.md        # Validation examples
```

## 🚀 Quick Start

### Installation
```bash
# Navigate to AIVA root directory
cd C:\D\fold7\AIVA-git

# Install plugin dependencies (if any)
pip install -r plugins/aiva_converters/requirements.txt
```

### Basic Usage

#### Generate All Language Schemas
```bash
# Generate all supported language schemas from SOT
python plugins/aiva_converters/core/schema_codegen_tool.py --generate-all

# Generate specific language
python plugins/aiva_converters/core/schema_codegen_tool.py --lang python
python plugins/aiva_converters/core/schema_codegen_tool.py --lang typescript
```

#### Interactive Language Conversion
```powershell
# Get conversion guidance between languages
.\plugins\aiva_converters\scripts\language_converter_final.ps1 -SourceLang python -TargetLang typescript
```

#### Generate TypeScript Interfaces
```bash
# Generate TypeScript from JSON Schema
python plugins/aiva_converters/core/typescript_generator.py
```

## 🔧 Advanced Usage

### Custom Schema Generation
```python
from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodeGenerator

# Initialize generator
generator = SchemaCodeGenerator("custom_schema.yaml")

# Generate specific language
python_files = generator.generate_python_schemas("./output/python")
typescript_files = generator.generate_typescript_schemas("./output/ts")
```

### Conversion Validation
```python
from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator

# Validate conversion results
validator = CrossLanguageValidator()
validation_result = validator.validate_conversion("source.py", "target.ts")
```

## 📋 Configuration

### Schema Generation Configuration
The plugin uses `core_schema_sot.yaml` as the single source of truth. Configuration is embedded in the YAML file under `generation_config`:

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

## 🎯 Use Cases

### 1. **Multi-Language Project Development**
- Generate consistent schemas across Python, TypeScript, Go, and Rust
- Maintain type safety across language boundaries
- Ensure contract compliance across services

### 2. **Legacy Code Migration**
- Convert existing Python models to TypeScript interfaces
- Transform Go structs to Rust structures
- Migrate between different schema formats

### 3. **API Contract Generation**
- Generate client SDKs from schema definitions
- Create documentation from schema metadata
- Validate API compatibility across versions

### 4. **Security Report Processing**
- Convert SARIF reports to various formats
- Transform vulnerability data between tools
- Standardize security findings across platforms

## 📊 Performance Benchmarks

The plugin maintains AIVA's performance standards:

- **JSON Serialization**: 8,536+ ops/s baseline
- **Schema Generation**: Sub-second for typical schemas
- **Cross-Language Validation**: Minimal overhead
- **Memory Usage**: Optimized for large schema sets

## 🧪 Testing

### Run Plugin Tests
```bash
# Run all plugin tests
python -m pytest plugins/aiva_converters/tests/ -v

# Run specific test categories
python -m pytest plugins/aiva_converters/tests/test_schema_codegen.py -v
```

### Validation Tests
```bash
# Validate generated schemas
python plugins/aiva_converters/core/schema_codegen_tool.py --validate

# Cross-language compatibility tests
python plugins/aiva_converters/tests/test_cross_language_compatibility.py
```

## 🔗 Integration Points

### With AIVA Core
- **Contract System**: Uses `aiva_common.schemas` as foundation
- **Performance Standards**: Maintains 6.7x performance advantage
- **Architecture Compliance**: Follows contract-driven design principles

### With Development Workflow
- **VS Code Integration**: Compatible with Pylance, Go extension, Rust analyzer
- **CI/CD Integration**: Automated schema generation in build pipelines
- **Documentation Generation**: Auto-generates API documentation

### With External Tools
- **Schema Validators**: JSON Schema, OpenAPI, Protocol Buffers
- **Code Generators**: Compatible with external code generation tools
- **Build Systems**: Integrates with Make, Gradle, Cargo, npm

## 📈 Roadmap

### Phase 1: Core Consolidation ✅
- [x] Consolidate existing conversion tools
- [x] Create unified plugin structure
- [x] Establish testing framework

### Phase 2: Enhanced Generation (Q1 2025)
- [ ] Template-based generation system
- [ ] Custom validation rules
- [ ] Performance optimization

### Phase 3: Advanced Features (Q2 2025)  
- [ ] AI-assisted conversion suggestions
- [ ] Visual schema designer integration
- [ ] Real-time validation feedback

### Phase 4: Enterprise Features (Q3 2025)
- [ ] Enterprise schema registry integration
- [ ] Advanced versioning support
- [ ] Distributed generation capabilities

## 🤝 Contributing

### Adding New Converters
1. Create converter in `converters/` directory
2. Follow existing patterns and interfaces
3. Add comprehensive tests
4. Update documentation

### Extending Language Support
1. Add language configuration to `core_schema_sot.yaml`
2. Implement generation logic in `schema_codegen_tool.py`
3. Create language-specific templates
4. Add validation tests

## 📚 Documentation

- **[Schema Generation Guide](./examples/schema_generation.md)** - Comprehensive schema generation
- **[Language Conversion Guide](./examples/python_to_typescript.md)** - Language-specific conversion
- **[Validation Examples](./examples/validation_examples.md)** - Validation and testing patterns
- **[Performance Optimization](./docs/performance.md)** - Optimization strategies

## 🔧 Troubleshooting

### Common Issues
1. **Schema Generation Fails**: Check `core_schema_sot.yaml` syntax
2. **Type Mapping Errors**: Verify language-specific type mappings
3. **Performance Issues**: Review generation batch sizes

### Debug Mode
```bash
# Enable debug logging
python plugins/aiva_converters/core/schema_codegen_tool.py --debug --lang python
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
- **字符串常量管理**: 統一常量定義，提升維護性

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
- **工具鏈穩定**: 支撑 Schema SoT 和多語言綁定的核心引擎

---

**Plugin Maintainer**: AIVA Architecture Team  
**Version**: 1.1.0 (品質提升版)  
**Last Updated**: November 3, 2025  
**Compatibility**: AIVA Core 2.x+  
**品質狀態**: ✅ SonarQube 100% 合規 | ✅ 認知複雜度 ≤15