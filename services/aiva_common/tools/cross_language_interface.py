#!/usr/bin/env python3
"""
AI 組件跨語言轉換接口
====================

此模組提供 AI 組件理解和操作不同程式語言 Schema 的能力，
支援 Python、Go、Rust 三種語言的自動轉換和操作。

功能特色:
- 🔄 自動語言轉換 (Python ↔ Go ↔ Rust)
- 🤖 AI 友好的統一接口
- 📝 智能代碼生成和提示
- 🔍 跨語言一致性驗證
- 🎯 Zero-configuration 使用體驗
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class SchemaField:
    """Schema 字段定義"""
    name: str
    type: str
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    validation: Optional[Dict[str, Any]] = None


@dataclass  
class SchemaDefinition:
    """跨語言 Schema 定義"""
    name: str
    description: str
    category: str
    fields: List[SchemaField]
    language: str = "universal"
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return asdict(self)


class CrossLanguageSchemaInterface:
    """AI 組件跨語言 Schema 操作接口"""
    
    def __init__(self, sot_file: str = "services/aiva_common/core_schema_sot.yaml"):
        """初始化跨語言接口
        
        Args:
            sot_file: Schema SOT YAML 檔案路徑
        """
        self.sot_file = Path(sot_file)
        self.sot_data: Dict[str, Any] = {}
        self.language_mappings = {}
        self._load_schema_data()
        
    def _load_schema_data(self) -> None:
        """載入 Schema 資料和語言映射"""
        with open(self.sot_file, 'r', encoding='utf-8') as f:
            self.sot_data = yaml.safe_load(f)
            
        # 構建語言映射表
        generation_config = self.sot_data.get('generation_config', {})
        for lang, config in generation_config.items():
            self.language_mappings[lang] = config.get('field_mapping', {})
    
    def get_all_schemas(self) -> List[SchemaDefinition]:
        """獲取所有 Schema 定義
        
        Returns:
            所有 Schema 定義的列表
        """
        schemas = []
        
        # 收集所有類別的 Schema
        categories = ['base_types', 'messaging', 'tasks', 'findings', 'async_utils', 'plugins', 'cli']
        
        for category in categories:
            if category in self.sot_data:
                for schema_name, schema_info in self.sot_data[category].items():
                    fields = []
                    
                    for field_name, field_info in schema_info.get('fields', {}).items():
                        field = SchemaField(
                            name=field_name,
                            type=field_info['type'],
                            description=field_info.get('description', ''),
                            required=field_info.get('required', True),
                            default_value=field_info.get('default'),
                            validation=field_info.get('validation')
                        )
                        fields.append(field)  # type: ignore
                    
                    schema = SchemaDefinition(
                        name=schema_name,
                        description=schema_info.get('description', ''),
                        category=category,
                        fields=fields
                    )
                    schemas.append(schema)  # type: ignore
        
        return schemas
    
    def get_schema_by_name(self, schema_name: str) -> Optional[SchemaDefinition]:
        """根據名稱獲取特定 Schema
        
        Args:
            schema_name: Schema 名稱
            
        Returns:
            Schema 定義或 None
        """
        all_schemas = self.get_all_schemas()
        for schema in all_schemas:
            if schema.name == schema_name:
                return schema
        return None
    
    def convert_type_to_language(self, source_type: str, target_language: str) -> str:
        """將類型轉換為指定語言格式
        
        Args:
            source_type: 源類型 (通用格式)
            target_language: 目標語言 (python/go/rust)
            
        Returns:
            目標語言的類型格式
        """
        if target_language not in self.language_mappings:
            return source_type
            
        mapping = self.language_mappings[target_language]
        return mapping.get(source_type, source_type)
    
    def generate_schema_code(self, schema_name: str, target_language: str) -> str:
        """生成指定語言的 Schema 代碼
        
        Args:
            schema_name: Schema 名稱
            target_language: 目標語言 (python/go/rust)
            
        Returns:
            生成的代碼字符串
        """
        schema = self.get_schema_by_name(schema_name)
        if not schema:
            return f"// Schema '{schema_name}' not found"
            
        if target_language == "python":
            return self._generate_python_code(schema)
        elif target_language == "go":
            return self._generate_go_code(schema)
        elif target_language == "rust":
            return self._generate_rust_code(schema)
        else:
            return f"// Unsupported language: {target_language}"
    
    def _generate_python_code(self, schema: SchemaDefinition) -> str:
        """生成 Python Pydantic 代碼"""
        lines = [
            f'class {schema.name}(BaseModel):',
            f'    """',
            f'    {schema.description}',
            f'    """',
            f''
        ]
        
        for field in schema.fields:
            python_type = self.convert_type_to_language(field.type, "python")
            default = f' = Field(default={field.default_value})' if field.default_value is not None else ''
            lines.append(f'    {field.name}: {python_type}{default}  # {field.description}')  # type: ignore
        
        return '\n'.join(lines)
    
    def _generate_go_code(self, schema: SchemaDefinition) -> str:
        """生成 Go struct 代碼"""
        lines = [
            f'// {schema.name} {schema.description}',
            f'type {schema.name} struct {{',
        ]
        
        for field in schema.fields:
            go_type = self.convert_type_to_language(field.type, "go")
            go_name = self._to_go_field_name(field.name)
            json_tag = f',omitempty' if not field.required else ''
            lines.append(f'    {go_name:<20} {go_type:<20} `json:"{field.name}{json_tag}"`  // {field.description}')  # type: ignore
        
        lines.append('}')  # type: ignore
        return '\n'.join(lines)
    
    def _generate_rust_code(self, schema: SchemaDefinition) -> str:
        """生成 Rust struct 代碼"""
        lines = [
            f'/// {schema.description}',
            f'#[derive(Debug, Clone, Serialize, Deserialize)]',
            f'pub struct {schema.name} {{',
        ]
        
        for field in schema.fields:
            rust_type = self.convert_type_to_language(field.type, "rust")
            if not field.required:
                rust_type = f"Option<{rust_type}>"
            lines.append(f'    /// {field.description}')  # type: ignore
            lines.append(f'    pub {field.name}: {rust_type},')  # type: ignore
        
        lines.append('}')  # type: ignore
        return '\n'.join(lines)
    
    def _to_go_field_name(self, snake_case: str) -> str:
        """將 snake_case 轉換為 Go 的 PascalCase"""
        return ''.join(word.capitalize() for word in snake_case.split('_'))
    
    def validate_cross_language_consistency(self) -> Dict[str, List[str]]:
        """驗證跨語言一致性
        
        Returns:
            驗證結果，包含發現的問題
        """
        issues = {
            "missing_mappings": [],
            "inconsistent_types": [],
            "validation_errors": []
        }
        
        all_schemas = self.get_all_schemas()
        
        for schema in all_schemas:
            for field in schema.fields:
                # 檢查是否所有語言都有類型映射
                for lang in ['python', 'go', 'rust']:
                    if field.type not in self.language_mappings.get(lang, {}):
                        issues["missing_mappings"].append(  # type: ignore
                            f"Missing {lang} mapping for type '{field.type}' in {schema.name}.{field.name}"
                        )
        
        return issues
    
    def get_ai_friendly_schema_info(self, schema_name: Optional[str] = None) -> Dict[str, Any]:
        """獲取 AI 友好的 Schema 信息
        
        Args:
            schema_name: 可選的特定 Schema 名稱
            
        Returns:
            AI 可以理解的結構化信息
        """
        if schema_name:
            schema = self.get_schema_by_name(schema_name)
            if not schema:
                return {"error": f"Schema '{schema_name}' not found"}
            schemas = [schema]
        else:
            schemas = self.get_all_schemas()
        
        result = {
            "total_schemas": len(schemas),
            "categories": {},
            "language_support": list(self.language_mappings.keys()),
            "schemas": []
        }
        
        # 按類別統計
        for schema in schemas:
            if schema.category not in result["categories"]:
                result["categories"][schema.category] = 0
            result["categories"][schema.category] += 1
            
            # 詳細 schema 信息
            schema_info = {
                "name": schema.name,
                "description": schema.description,
                "category": schema.category,
                "field_count": len(schema.fields),
                "fields": [],
                "code_examples": {}
            }
            
            # 字段信息
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "type": field.type,
                    "description": field.description,
                    "required": field.required,
                    "language_types": {}
                }
                
                # 各語言類型映射
                for lang in ['python', 'go', 'rust']:
                    field_info["language_types"][lang] = self.convert_type_to_language(field.type, lang)
                
                schema_info["fields"].append(field_info)  # type: ignore
            
            # 代碼示例
            for lang in ['python', 'go', 'rust']:
                schema_info["code_examples"][lang] = self.generate_schema_code(schema.name, lang)
            
            result["schemas"].append(schema_info)  # type: ignore
        
        return result


def create_ai_assistant_prompt() -> str:
    """創建 AI 助手使用提示"""
    return """
# AIVA 跨語言 Schema AI 助手使用指南

## 可用功能

1. **獲取所有 Schema**:
   ```python
   interface = CrossLanguageSchemaInterface()
   all_schemas = interface.get_all_schemas()
   ```

2. **查找特定 Schema**:
   ```python
   schema = interface.get_schema_by_name("AsyncTaskConfig")
   ```

3. **生成語言特定代碼**:
   ```python
   python_code = interface.generate_schema_code("AsyncTaskConfig", "python")
   go_code = interface.generate_schema_code("AsyncTaskConfig", "go")
   rust_code = interface.generate_schema_code("AsyncTaskConfig", "rust")
   ```

4. **類型轉換**:
   ```python
   go_type = interface.convert_type_to_language("Optional[str]", "go")  # *string
   rust_type = interface.convert_type_to_language("List[str]", "rust")  # Vec<String>
   ```

5. **獲取 AI 友好信息**:
   ```python
   info = interface.get_ai_friendly_schema_info("AsyncTaskConfig")
   ```

6. **驗證一致性**:
   ```python
   issues = interface.validate_cross_language_consistency()
   ```

## 支援的 Schema 類別
- base_types: 基礎類型
- messaging: 訊息通訊  
- tasks: 任務管理
- findings: 發現結果
- async_utils: 異步工具
- plugins: 插件管理
- cli: CLI 界面

## 支援的語言
- Python (Pydantic v2)
- Go (structs with JSON tags)
- Rust (Serde with serialization)
"""


if __name__ == "__main__":
    # 示例使用
    interface = CrossLanguageSchemaInterface()
    
    print("🤖 AI 組件跨語言 Schema 接口測試")
    print("=" * 50)
    
    # 獲取 AI 友好信息
    info = interface.get_ai_friendly_schema_info("AsyncTaskConfig")
    print(f"📊 Schema 信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
    
    # 生成代碼示例
    print("\n📝 代碼生成示例:")
    for lang in ['python', 'go', 'rust']:
        code = interface.generate_schema_code("AsyncTaskConfig", lang)
        print(f"\n{lang.upper()}:")
        print(code)
    
    # 驗證一致性
    issues = interface.validate_cross_language_consistency()
    print(f"\n🔍 一致性檢查: {len(issues['missing_mappings'])} 個問題")  # type: ignore
    
    print("\n✅ 跨語言接口測試完成！")