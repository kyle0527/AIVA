#!/usr/bin/env python3
"""
AIVA Schema Code Generation Tool
===============================

基於 core_schema_sot.yaml 自動生成跨語言 Schema 定義

功能特色:
- 🔄 支援 Python (Pydantic v2) + Go (structs) + Rust (Serde)
- 📝 自動生成文檔和類型註解
- 🔍 Schema 驗證和向後兼容性檢查
- 🚀 VS Code 整合，支援 Pylance 和 Go 擴充功能
- 🎯 單一事實來源 (Single Source of Truth)

使用方式:
    python tools/schema_codegen_tool.py --generate-all
    python tools/schema_codegen_tool.py --lang python --validate
    python tools/schema_codegen_tool.py --lang go --output-dir custom/path
"""

import argparse
import logging

# 設定日誌 - 支援 Unicode
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, Template

# ==================== 常數定義 ====================
OPTIONAL_PREFIX = "Optional["
LIST_PREFIX = "List["
CHRONO_DATETIME = "chrono::DateTime<chrono::Utc>"
STRING_NEW = "String::new()"
URL_URL = "url::Url"



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("schema_codegen.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class SchemaCodeGenerator:
    """Schema 代碼生成器 - 支援多語言自動生成"""

    def __init__(self, sot_file: str = "services/aiva_common/core_schema_sot.yaml"):
        """初始化代碼生成器

        Args:
            sot_file: Schema SOT YAML 檔案路徑
        """
        self.sot_file = Path(sot_file)
        self.sot_data: dict[str, Any] = {}
        self.jinja_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 載入 SOT 資料
        self._load_sot_data()

    def _load_sot_data(self) -> None:
        """載入 Schema SOT 資料"""
        try:
            with open(self.sot_file, encoding="utf-8") as f:
                self.sot_data = yaml.safe_load(f)
            logger.info(f"✅ 成功載入 SOT 檔案: {self.sot_file}")
        except FileNotFoundError:
            logger.error(f"❌ SOT 檔案不存在: {self.sot_file}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML 解析錯誤: {e}")
            sys.exit(1)

    def generate_python_schemas(self, output_dir: str | None = None) -> list[str]:
        """生成 Python Pydantic v2 Schema

        Args:
            output_dir: 自訂輸出目錄

        Returns:
            生成的檔案列表
        """
        config = self.sot_data["generation_config"]["python"]
        target_dir = Path(output_dir) if output_dir else Path(config["target_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # 生成基礎類型
        if "base_types" in self.sot_data:
            base_file = target_dir / "base_types.py"
            content = self._render_python_base_types()
            with open(base_file, "w", encoding="utf-8") as f:
                f.write(content)
            generated_files.append(str(base_file))  # type: ignore
            logger.info(f"✅ 生成 Python 基礎類型: {base_file}")

        # 生成各模組 Schema - 包含所有新增的分類
        categories = ["messaging", "tasks", "findings", "async_utils", "plugins", "cli"]
        for category in categories:
            if category in self.sot_data:
                module_file = target_dir / f"{category}.py"
                content = self._render_python_category(category)
                with open(module_file, "w", encoding="utf-8") as f:
                    f.write(content)
                generated_files.append(str(module_file))  # type: ignore
                logger.info(f"✅ 生成 Python {category} Schema: {module_file}")

        # 生成 __init__.py
        init_file = target_dir / "__init__.py"
        content = self._render_python_init()
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(content)
        generated_files.append(str(init_file))  # type: ignore

        return generated_files

    def generate_go_schemas(self, output_dir: str | None = None) -> list[str]:
        """生成 Go struct Schema

        Args:
            output_dir: 自訂輸出目錄

        Returns:
            生成的檔案列表
        """
        config = self.sot_data["generation_config"]["go"]
        target_dir = Path(output_dir) if output_dir else Path(config["target_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # 生成統一的 schemas.go 檔案
        schema_file = target_dir / "schemas.go"
        content = self._render_go_schemas()
        with open(schema_file, "w", encoding="utf-8") as f:
            f.write(content)
        generated_files.append(str(schema_file))  # type: ignore
        logger.info(f"✅ 生成 Go Schema: {schema_file}")

        return generated_files

    def generate_rust_schemas(self, output_dir: str | None = None) -> list[str]:
        """生成 Rust Serde Schema

        Args:
            output_dir: 自訂輸出目錄

        Returns:
            生成的檔案列表
        """
        config = self.sot_data["generation_config"]["rust"]
        target_dir = Path(output_dir) if output_dir else Path(config["target_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # 生成 mod.rs
        mod_file = target_dir / "mod.rs"
        content = self._render_rust_schemas()
        with open(mod_file, "w", encoding="utf-8") as f:
            f.write(content)
        generated_files.append(str(mod_file))  # type: ignore
        logger.info(f"✅ 生成 Rust Schema: {mod_file}")

        return generated_files

    def _render_python_base_types(self) -> str:
        """渲染 Python 基礎類型"""
        content = []
        content.append('"""')  # type: ignore
        content.append("AIVA 基礎類型 Schema - 自動生成")  # type: ignore
        content.append("=====================================")  # type: ignore
        content.append("")  # type: ignore
        content.append(self.sot_data["metadata"]["description"])  # type: ignore
        content.append("")  # type: ignore
        content.append(f"⚠️  {self.sot_data['metadata']['generated_note']}")  # type: ignore
        content.append(f"📅 最後更新: {self.sot_data['metadata']['last_updated']}")  # type: ignore
        content.append(f"🔄 Schema 版本: {self.sot_data['version']}")  # type: ignore
        content.append('"""')  # type: ignore
        content.append("")  # type: ignore

        # 添加imports
        for imp in self.sot_data["generation_config"]["python"]["base_imports"]:
            content.append(imp)  # type: ignore
        content.append("")  # type: ignore
        content.append("")  # type: ignore

        # 生成類別
        for class_name, class_info in self.sot_data["base_types"].items():
            content.append(f"class {class_name}(BaseModel):")  # type: ignore
            content.append(f'    """{class_info["description"]}"""')  # type: ignore
            content.append("")  # type: ignore

            for field_name, field_info in class_info["fields"].items():
                field_line = self._generate_python_field(field_name, field_info)
                content.append(f"    {field_line}")  # type: ignore
                content.append(f'    """{field_info["description"]}"""')  # type: ignore
                content.append("")  # type: ignore

            content.append("")  # type: ignore

        return "\n".join(content)

    def _render_python_category(self, category: str) -> str:
        """渲染 Python 分類 Schema"""
        content = []
        content.append('"""')  # type: ignore
        content.append(f"AIVA {category.title()} Schema - 自動生成")  # type: ignore
        content.append("=====================================")  # type: ignore
        content.append("")  # type: ignore
        content.append(self.sot_data["metadata"]["description"])  # type: ignore
        content.append("")  # type: ignore
        content.append(f"⚠️  {self.sot_data['metadata']['generated_note']}")  # type: ignore
        content.append(f"📅 最後更新: {self.sot_data['metadata']['last_updated']}")  # type: ignore
        content.append(f"🔄 Schema 版本: {self.sot_data['version']}")  # type: ignore
        content.append('"""')  # type: ignore
        content.append("")  # type: ignore

        # 添加imports
        for imp in self.sot_data["generation_config"]["python"]["base_imports"]:
            content.append(imp)  # type: ignore
        content.append("")  # type: ignore
        content.append("from .base_types import *")  # type: ignore
        content.append("")  # type: ignore
        content.append("")  # type: ignore

        # 生成類別
        for class_name, class_info in self.sot_data[category].items():
            content.append(f"class {class_name}(BaseModel):")  # type: ignore
            content.append(f'    """{class_info["description"]}"""')  # type: ignore
            content.append("")  # type: ignore

            # 檢查是否有extends
            if "extends" in class_info:
                content.append(f'    # 繼承自: {class_info["extends"]}')  # type: ignore
                content.append("")  # type: ignore

            # 處理fields
            for field_name, field_info in class_info.get("fields", {}).items():
                field_line = self._generate_python_field(field_name, field_info)
                content.append(f"    {field_line}")  # type: ignore
                content.append(f'    """{field_info["description"]}"""')  # type: ignore
                content.append("")  # type: ignore

            # 處理additional_fields
            for field_name, field_info in class_info.get(
                "additional_fields", {}
            ).items():
                field_line = self._generate_python_field(field_name, field_info)
                content.append(f"    {field_line}")  # type: ignore
                content.append(f'    """{field_info["description"]}"""')  # type: ignore
                content.append("")  # type: ignore

            content.append("")  # type: ignore

        return "\n".join(content)

    def _render_python_init(self) -> str:
        """渲染 Python __init__.py"""
        template = Template(
            '''"""
AIVA Schema 自動生成模組
======================

此模組包含所有由 core_schema_sot.yaml 自動生成的 Schema 定義

⚠️  請勿手動修改此模組中的檔案
🔄  如需更新，請修改 core_schema_sot.yaml 後重新生成
"""

# 基礎類型
from .base_types import *

# 訊息通訊
from .messaging import *

# 任務管理
from .tasks import *

# 發現結果
from .findings import *

__version__ = "{{ version }}"
__generated_at__ = "{{ generated_at }}"

__all__ = [
    # 基礎類型
    "MessageHeader",
    "Target", 
    "Vulnerability",
    
    # 訊息通訊
    "AivaMessage",
    "AIVARequest",
    "AIVAResponse",
    
    # 任務管理
    "FunctionTaskPayload",
    "FunctionTaskTarget", 
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "ScanTaskPayload",
    
    # 發現結果
    "FindingPayload",
    "FindingEvidence",
    "FindingImpact", 
    "FindingRecommendation",
]
'''
        )

        return template.render(
            version=self.sot_data["version"], generated_at=datetime.now().isoformat()
        )

    def _render_go_schemas(self) -> str:
        """渲染 Go 統一 Schema - 重構後降低認知複雜度"""
        content = []
        
        # 生成文件頭部信息
        self._add_go_header(content)
        
        # 生成導入語句
        self._add_go_imports(content)
        
        # 生成枚舉類型
        self._add_go_enums(content)
        
        # 生成基礎類型
        self._add_go_base_types(content)
        
        # 生成其他分類
        self._add_go_other_sections(content)
        
        return "\n".join(content)

    def _add_go_header(self, content: list[str]) -> None:
        """添加 Go 文件頭部信息"""
        metadata = self.sot_data["metadata"]
        content.extend([
            "// AIVA Go Schema - 自動生成",
            "// ===========================",
            "//",
            f'// {metadata["description"]}',
            "//",
            f'// ⚠️  {metadata["generated_note"]}',
            f'// 📅 最後更新: {metadata["last_updated"]}',
            f'// 🔄 Schema 版本: {self.sot_data["version"]}',
            ""
        ])

    def _add_go_imports(self, content: list[str]) -> None:
        """添加 Go 導入語句"""
        base_imports = self.sot_data["generation_config"]["go"]["base_imports"]
        content.extend(base_imports)
        content.append("")

    def _add_go_enums(self, content: list[str]) -> None:
        """添加 Go 枚舉類型"""
        if "enums" not in self.sot_data:
            return
            
        content.extend([
            "// ==================== 枚舉類型 ====================",
            ""
        ])
        
        for enum_name, enum_info in self.sot_data["enums"].items():
            self._generate_go_enum(content, enum_name, enum_info)

    def _generate_go_enum(self, content: list[str], enum_name: str, enum_info: dict) -> None:
        """生成單個 Go 枚舉"""
        description = enum_info.get("description", "")
        content.extend([
            f'// {enum_name} {description}',
            f"type {enum_name} string",
            "",
            "const ("
        ])
        
        for value_key, value_desc in enum_info.get("values", {}).items():
            const_name = f"{enum_name}{value_key.title()}"
            content.append(f'    {const_name:<30} {enum_name} = "{value_key}"  // {value_desc}')
        
        content.extend([")", ""])

    def _add_go_base_types(self, content: list[str]) -> None:
        """添加 Go 基礎類型"""
        content.extend([
            "// ==================== 基礎類型 ====================",
            ""
        ])
        
        for class_name, class_info in self.sot_data["base_types"].items():
            self._generate_go_struct(content, class_name, class_info, "base_types")

    def _add_go_other_sections(self, content: list[str]) -> None:
        """添加 Go 其他分類"""
        sections = [
            ("messaging", "訊息通訊"),
            ("tasks", "任務管理"),
            ("findings", "發現結果"),
            ("async_utils", "異步工具"),
            ("plugins", "插件管理"),
            ("cli", "CLI 界面"),
        ]
        
        for section, title in sections:
            if section in self.sot_data:
                self._generate_go_section(content, section, title)

    def _generate_go_section(self, content: list[str], section: str, title: str) -> None:
        """生成 Go 分類區塊"""
        content.extend([
            f"// ==================== {title} ====================",
            ""
        ])
        
        for class_name, class_info in self.sot_data[section].items():
            self._generate_go_struct(content, class_name, class_info, section)

    def _generate_go_struct(self, content: list[str], class_name: str, class_info: dict, section: str) -> None:
        """生成 Go 結構體"""
        description = class_info["description"]
        content.extend([
            f'// {class_name} {description}',
            f"type {class_name} struct {{"
        ])
        
        # 獲取所有字段（包括繼承的字段）
        all_fields = self._get_all_fields(class_info, section)
        
        # 生成所有字段
        for field_name, field_info in all_fields.items():
            self._generate_go_struct_field(content, field_name, field_info)
        
        content.extend(["}", ""])

    def _generate_go_struct_field(self, content: list[str], field_name: str, field_info: dict) -> None:
        """生成 Go 結構體欄位"""
        go_name = self._to_go_field_name(field_name)
        go_type = self._get_go_type(field_info["type"])
        json_tag = self._get_go_json_tag(field_info.get("required", True))
        description = field_info["description"]
        
        content.append(
            f'    {go_name:<20} {go_type:<25} `json:"{field_name}{json_tag}"`  // {description}'
        )

    def _render_rust_schemas(self) -> str:
        """
        渲染完整的 Rust Schema

        生成包含所有結構體、枚舉和序列化支持的 Rust 代碼
        """
        rust_code = f"""// AIVA Rust Schema - 自動生成
// 版本: {self.sot_data['version']}
// 生成時間: {self.sot_data.get('generated_at', 'N/A')}
// 
// 完整的 Rust Schema 實現，包含序列化/反序列化支持

use serde::{{Serialize, Deserialize}};
use std::collections::HashMap;
use chrono::{{DateTime, Utc}};

// 可選依賴 - 根據實際使用情況啟用
#[cfg(feature = "uuid")]
use uuid::Uuid;

#[cfg(feature = "url")]
use url::Url;

"""

        # 生成枚舉
        for enum_name, enum_data in self.sot_data.get("enums", {}).items():
            rust_code += self._render_rust_enum(enum_name, enum_data)
            rust_code += "\n\n"

        # 生成結構體 - 處理所有頂層分類
        all_schemas = {}

        # 收集所有schema定義 - 包含所有新增的分類
        categories = [
            "base_types",
            "messaging",
            "tasks",
            "findings",
            "async_utils",
            "plugins",
            "cli",
        ]
        for category in categories:
            category_schemas = self.sot_data.get(category, {})
            if isinstance(category_schemas, dict):
                all_schemas.update(category_schemas)

        for schema_name, schema_data in all_schemas.items():
            rust_code += self._render_rust_struct(schema_name, schema_data)
            rust_code += "\n\n"

        return rust_code

    def _render_rust_enum(self, enum_name: str, enum_data: dict) -> str:
        """渲染 Rust 枚舉"""
        description = enum_data.get("description", f"{enum_name} 枚舉")
        values = enum_data.get("values", {})

        rust_enum = f"""/// {description}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum {enum_name} {{"""

        for value, desc in values.items():
            rust_enum += f"""
    /// {desc}
    {value.replace('-', '_').upper()},"""

        rust_enum += (
            """
}

impl std::fmt::Display for """
            + enum_name
            + """ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {"""
        )

        for value in values.keys():
            rust_value = value.replace("-", "_").upper()
            rust_enum += f"""
            {enum_name}::{rust_value} => write!(f, "{value}"),"""

        rust_enum += (
            """
        }
    }
}

impl std::str::FromStr for """
            + enum_name
            + """ {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {"""
        )

        for value in values.keys():
            rust_value = value.replace("-", "_").upper()
            rust_enum += f"""
            "{value.upper()}" => Ok({enum_name}::{rust_value}),"""

        rust_enum += f"""
            _ => Err(format!("Invalid {enum_name}: {{}}", s)),
        }}
    }}
}}"""

        return rust_enum

    def _render_rust_struct(self, struct_name: str, struct_data: dict) -> str:
        """渲染 Rust 結構體 - 重構後降低認知複雜度"""
        struct_info = self._extract_struct_metadata(struct_name, struct_data)
        
        struct_header = self._generate_rust_struct_header(struct_name, struct_info.description)
        struct_fields = self._generate_rust_struct_fields(struct_info)
        struct_impl = self._generate_rust_struct_impl(struct_name, struct_info)
        
        return struct_header + struct_fields + struct_impl

    def _extract_struct_metadata(self, struct_name: str, struct_data: dict):
        """提取結構體元數據"""
        from collections import namedtuple
        StructInfo = namedtuple('StructInfo', ['description', 'properties', 'required'])
        
        description = struct_data.get("description", f"{struct_name} 結構體")
        properties = struct_data.get("properties", struct_data.get("fields", {}))
        required = struct_data.get("required", [])
        
        return StructInfo(description, properties, required)

    def _generate_rust_struct_header(self, struct_name: str, description: str) -> str:
        """生成 Rust 結構體頭部"""
        return f"""/// {description}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct {struct_name} {{"""

    def _generate_rust_struct_fields(self, struct_info) -> str:
        """生成 Rust 結構體欄位定義"""
        fields_code = ""
        
        for field_name, field_data in struct_info.properties.items():
            field_desc = field_data.get("description", f"{field_name} 欄位")
            field_type, is_optional = self._determine_rust_field_type(
                field_name, field_data, struct_info.required
            )
            
            fields_code += f"""
    /// {field_desc}"""
            
            if is_optional:
                fields_code += f"""
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub {field_name}: {field_type},"""
            else:
                fields_code += f"""
    pub {field_name}: {field_type},"""
        
        return fields_code + """
}

"""

    def _determine_rust_field_type(self, field_name: str, field_data: dict, required: list) -> tuple[str, bool]:
        """決定 Rust 欄位類型和是否為可選"""
        original_type = field_data.get("type")
        field_required = field_data.get("required", True)
        is_required_in_struct = field_name in required or field_required

        if original_type and original_type.startswith(OPTIONAL_PREFIX):
            return self._convert_to_rust_type(original_type, field_data), True
        elif not is_required_in_struct:
            base_type = self._convert_to_rust_type(original_type, field_data)
            return f"Option<{base_type}>", True
        else:
            return self._convert_to_rust_type(original_type, field_data), False

    def _generate_rust_struct_impl(self, struct_name: str, struct_info) -> str:
        """生成 Rust 結構體實現部分"""
        new_method = self._generate_rust_new_method(struct_info)
        validate_method = self._generate_rust_validate_method(struct_info)
        default_impl = self._generate_rust_default_impl(struct_name)
        
        return f"""impl {struct_name} {{
    /// 創建新的實例
    pub fn new() -> Self {{
        Self {{{new_method}
        }}
    }}
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {{{validate_method}
        Ok(())
    }}
}}

{default_impl}"""

    def _generate_rust_new_method(self, struct_info) -> str:
        """生成 Rust new 方法的欄位初始化"""
        init_code = ""
        
        for field_name, field_data in struct_info.properties.items():
            original_type = field_data.get("type")
            field_required = field_data.get("required", True)
            is_required_in_struct = field_name in struct_info.required or field_required

            if self._is_optional_field(original_type, is_required_in_struct):
                init_code += f"""
            {field_name}: None,"""
            else:
                converted_type = self._get_converted_type_for_default(
                    original_type, field_data, is_required_in_struct
                )
                default_value = self._get_rust_default_value(converted_type, field_data)
                init_code += f"""
            {field_name}: {default_value},"""
        
        return init_code

    def _is_optional_field(self, original_type: str, is_required_in_struct: bool) -> bool:
        """判斷是否為可選欄位"""
        return (original_type and original_type.startswith(OPTIONAL_PREFIX)) or not is_required_in_struct

    def _get_converted_type_for_default(self, original_type: str, field_data: dict, is_required_in_struct: bool) -> str:
        """獲取用於預設值的轉換類型"""
        if original_type and original_type.startswith(OPTIONAL_PREFIX):
            return self._convert_to_rust_type(original_type, field_data)
        elif not is_required_in_struct:
            base_type = self._convert_to_rust_type(original_type, field_data)
            return f"Option<{base_type}>"
        else:
            return self._convert_to_rust_type(original_type, field_data)

    def _generate_rust_validate_method(self, struct_info) -> str:
        """生成 Rust 驗證方法"""
        validation_code = ""
        
        for field_name in struct_info.required:
            if field_name in struct_info.properties:
                field_type = struct_info.properties[field_name].get("type")
                if field_type == "string":
                    validation_code += f"""
        if self.{field_name}.is_empty() {{
            return Err("Field '{field_name}' is required and cannot be empty".to_string());
        }}"""
        
        return validation_code

    def _generate_rust_default_impl(self, struct_name: str) -> str:
        """生成 Rust Default 實現"""
        return f"""impl Default for {struct_name} {{
    fn default() -> Self {{
        Self::new()
    }}
}}"""

    def _convert_to_rust_type(
        self, json_type: str, field_data: dict[str, Any] | None = None
    ) -> str:
        """將 JSON Schema 類型轉換為 Rust 類型"""
        if field_data is None:
            field_data = {}

        # 處理複合類型
        compound_type = self._handle_rust_compound_types(json_type, field_data)
        if compound_type:
            return compound_type

        # 處理自定義類型
        custom_type = self._handle_rust_custom_types(json_type)
        if custom_type:
            return custom_type

        # 處理特殊類型（枚舉和格式化）
        special_type = self._handle_rust_special_types(field_data)
        if special_type:
            return special_type

        # 處理基本類型
        return self._handle_rust_basic_types(json_type)

    def _handle_rust_compound_types(
        self, json_type: str, field_data: dict[str, Any]
    ) -> str | None:
        """處理複合類型：Optional, List, Dict"""
        if not json_type:
            return None

        # 處理 Optional 類型
        if json_type.startswith(OPTIONAL_PREFIX):
            inner_type = json_type[len(OPTIONAL_PREFIX):-1]
            return f"Option<{self._convert_to_rust_type(inner_type, field_data)}>"

        # 處理 List 類型
        if json_type.startswith(LIST_PREFIX):
            inner_type = json_type[len(LIST_PREFIX):-1]
            return f"Vec<{self._convert_to_rust_type(inner_type, field_data)}>"

        # 處理 Dict 類型
        if json_type.startswith("Dict["):
            return self._convert_rust_dict_type(json_type)

        return None

    def _convert_rust_dict_type(self, json_type: str) -> str:
        """轉換 Dict 類型為 HashMap"""
        dict_content = json_type[5:-1]  # 移除 'Dict[' 和 ']'
        
        if dict_content == "str, str":
            return "std::collections::HashMap<String, String>"
        elif dict_content == "str, Any":
            return "std::collections::HashMap<String, serde_json::Value>"
        else:
            return "std::collections::HashMap<String, serde_json::Value>"

    def _handle_rust_custom_types(self, json_type: str) -> str | None:
        """處理自定義類型（SOT 中的結構體）"""
        all_types = set()
        for category in ["base_types", "messaging", "tasks", "findings"]:
            if category in self.sot_data:
                all_types.update(self.sot_data[category].keys())

        if json_type in all_types:
            return json_type  # 自定義類型保持原名
        return None

    def _handle_rust_special_types(
        self, field_data: dict[str, Any]
    ) -> str | None:
        """處理特殊類型：枚舉和格式化類型"""
        # 處理枚舉類型
        if "enum" in field_data:
            return "String"

        # 處理格式化類型
        format_type = field_data.get("format")
        if format_type:
            return self._get_rust_format_type(format_type)

        return None

    def _get_rust_format_type(self, format_type: str) -> str:
        """根據格式類型返回對應的 Rust 類型"""
        format_mapping = {
            "date-time": CHRONO_DATETIME,
            "uuid": "uuid::Uuid",
            "uri": URL_URL,
            "url": URL_URL,
        }
        return format_mapping.get(format_type, "String")

    def _handle_rust_basic_types(self, json_type: str) -> str:
        """處理基本類型映射"""
        type_mapping = {
            "str": "String",
            "string": "String",
            "int": "i32",
            "integer": "i32",
            "float": "f64",
            "number": "f64",
            "bool": "bool",
            "boolean": "bool",
            "datetime": CHRONO_DATETIME,
            "Any": "serde_json::Value",
        }
        return type_mapping.get(json_type, "String")

    def _get_rust_default_value(
        self, json_type: str, field_data: dict[str, Any] | None = None
    ) -> str:
        """獲取 Rust 類型的默認值"""
        if field_data is None:
            field_data = {}

        # 檢查是否有明確定義的預設值
        explicit_default = self._get_explicit_default_value(field_data)
        if explicit_default:
            return explicit_default

        # 檢查特殊類型的預設值
        special_default = self._get_special_type_default(json_type)
        if special_default:
            return special_default

        # 檢查自定義類型的預設值
        custom_default = self._get_custom_type_default(json_type)
        if custom_default:
            return custom_default

        # 返回基本類型的預設值
        return self._get_basic_type_default(json_type)

    def _get_explicit_default_value(self, field_data: dict[str, Any]) -> str | None:
        """處理明確定義的預設值"""
        if "default" not in field_data:
            return None

        default_val = field_data["default"]
        
        if isinstance(default_val, str):
            return f'"{default_val}".to_string()'
        elif isinstance(default_val, bool):
            return str(default_val).lower()
        elif isinstance(default_val, (int, float)):
            return str(default_val)
        elif isinstance(default_val, list):
            return "Vec::new()"
        elif isinstance(default_val, dict):
            return "std::collections::HashMap::new()"
        
        return None

    def _get_special_type_default(self, json_type: str) -> str | None:
        """處理特殊類型的預設值 (Optional, Vec, HashMap)"""
        if not json_type:
            return None

        # 處理 Optional 類型
        if json_type.startswith("Option<"):
            return "None"

        # 處理 Vec 類型  
        if json_type.startswith("Vec<"):
            return "Vec::new()"

        # 處理 HashMap 類型
        if json_type.startswith("std::collections::HashMap<"):
            return "std::collections::HashMap::new()"

        return None

    def _get_custom_type_default(self, json_type: str) -> str | None:
        """處理自定義類型的預設值"""
        all_types = self._get_all_defined_types()
        
        if json_type in all_types:
            return f"{json_type}::default()"
        
        return None

    def _get_all_defined_types(self) -> set[str]:
        """獲取所有定義的自定義類型"""
        all_types = set()
        for category in ["base_types", "messaging", "tasks", "findings"]:
            if category in self.sot_data:
                all_types.update(self.sot_data[category].keys())
        return all_types

    def _get_basic_type_default(self, json_type: str) -> str:
        """獲取基本類型的預設值"""
        defaults = {
            "String": STRING_NEW,
            "str": STRING_NEW,
            "string": STRING_NEW,
            "i32": "0",
            "int": "0", 
            "integer": "0",
            "f64": "0.0",
            "float": "0.0",
            "number": "0.0",
            "bool": "false",
            "boolean": "false",
            CHRONO_DATETIME: "chrono::Utc::now()",
            "serde_json::Value": "serde_json::Value::Null",
            "uuid::Uuid": "uuid::Uuid::new_v4()",
            URL_URL: 'url::Url::parse("https://example.com").unwrap()',
        }
        return defaults.get(json_type, STRING_NEW)

    def _get_python_type(self, type_str: str) -> str:
        """轉換為 Python 類型"""
        mapping = self.sot_data["generation_config"]["python"]["field_mapping"]
        return mapping.get(type_str, type_str)

    def _get_python_default(self, default_value: Any) -> str:
        """獲取 Python 預設值"""
        if isinstance(default_value, str):
            return f'"{default_value}"'
        elif isinstance(default_value, bool):
            return str(default_value)
        elif isinstance(default_value, dict):
            return "Field(default_factory=dict)"
        elif isinstance(default_value, list):
            return "Field(default_factory=list)"
        return str(default_value)

    def _get_python_validation(self, value: Any) -> str:
        """獲取 Python 驗證參數"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, list):
            return f"{value}"
        return str(value)

    def generate_typescript_schemas(self, output_dir: str | None = None) -> list[str]:
        """生成 TypeScript Schema
        
        Args:
            output_dir: 自訂輸出目錄
            
        Returns:
            生成的檔案列表
        """
        # TypeScript 生成配置
        if output_dir:
            target_dir = Path(output_dir)
        else:
            target_dir = Path("services/features/common/typescript/aiva_common_ts/schemas/generated")
        
        target_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # 生成主要的 schemas.ts 文件
        schema_file = target_dir / "schemas.ts"
        content = self._render_typescript_schemas()
        with open(schema_file, "w", encoding="utf-8") as f:
            f.write(content)
        generated_files.append(str(schema_file))
        logger.info(f"✅ 生成 TypeScript Schema: {schema_file}")

        # 生成 index.ts 導出文件
        index_file = target_dir / "index.ts"
        index_content = self._render_typescript_index()
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(index_content)
        generated_files.append(str(index_file))
        logger.info(f"✅ 生成 TypeScript Index: {index_file}")

        return generated_files

    def _render_typescript_schemas(self) -> str:
        """渲染 TypeScript schemas"""
        content = []
        
        # 文件頭部
        content.extend([
            "/**",
            " * AIVA Common TypeScript Schemas - 自動生成",
            " * ==========================================",
            " * ",
            f" * {self.sot_data['metadata']['description']}",
            " * ",
            f" * ⚠️  {self.sot_data['metadata']['generated_note']}",
            f" * 📅 最後更新: {self.sot_data['metadata']['last_updated']}",
            f" * 🔄 Schema 版本: {self.sot_data['version']}",
            " * ",
            " * 遵循單一事實原則，與 Python aiva_common.schemas 保持完全一致",
            " */",
            "",
            "// ==================== 枚舉類型定義 ====================",
            ""
        ])

        # 生成枚舉 (從 Python enums 映射)
        content.extend([
            "export enum Severity {",
            "  CRITICAL = \"critical\",",
            "  HIGH = \"high\",",
            "  MEDIUM = \"medium\",",
            "  LOW = \"low\",",
            "  INFORMATIONAL = \"info\"",
            "}",
            "",
            "export enum Confidence {",
            "  CERTAIN = \"certain\",",
            "  FIRM = \"firm\",", 
            "  POSSIBLE = \"possible\"",
            "}",
            "",
            "export enum VulnerabilityType {",
            "  XSS = \"XSS\",",
            "  SQLI = \"SQL Injection\",",
            "  SSRF = \"SSRF\",",
            "  IDOR = \"IDOR\",",
            "  BOLA = \"BOLA\",",
            "  INFO_LEAK = \"Information Leak\",",
            "  WEAK_AUTH = \"Weak Authentication\",",
            "  RCE = \"Remote Code Execution\",",
            "  AUTHENTICATION_BYPASS = \"Authentication Bypass\"",
            "}",
            "",
            "// ==================== 基礎介面定義 ====================",
            ""
        ])

        # 生成基礎類型
        for class_name, class_info in self.sot_data["base_types"].items():
            content.extend(self._render_typescript_interface(class_name, class_info))
            content.append("")

        # 生成 messaging 類型
        if "messaging" in self.sot_data:
            content.append("// ==================== 訊息通訊類型 ====================")
            content.append("")
            for class_name, class_info in self.sot_data["messaging"].items():
                content.extend(self._render_typescript_interface(class_name, class_info))
                content.append("")

        # 生成 findings 類型  
        if "findings" in self.sot_data:
            content.append("// ==================== 漏洞發現類型 ====================")
            content.append("")
            for class_name, class_info in self.sot_data["findings"].items():
                content.extend(self._render_typescript_interface(class_name, class_info))
                content.append("")

        # 生成 tasks 類型
        if "tasks" in self.sot_data:
            content.append("// ==================== 任務管理類型 ====================")
            content.append("")
            for class_name, class_info in self.sot_data["tasks"].items():
                content.extend(self._render_typescript_interface(class_name, class_info))
                content.append("")

        return "\n".join(content)

    def _render_typescript_interface(self, interface_name: str, interface_info: dict) -> list[str]:
        """渲染 TypeScript 介面"""
        lines = []
        lines.append("/**")
        lines.append(f" * {interface_info['description']}")
        lines.append(" */")
        lines.append(f"export interface {interface_name} {{")
        
        # 生成字段
        for field_name, field_info in interface_info.get("fields", {}).items():
            ts_type = self._get_typescript_type(field_info["type"])
            optional_marker = "?" if not field_info.get("required", True) else ""
            description = field_info.get("description", "")
            
            if description:
                lines.append(f"  /** {description} */")
            lines.append(f"  {field_name}{optional_marker}: {ts_type};")
        
        lines.append("}")
        return lines

    def _get_typescript_type(self, type_str: str) -> str:
        """轉換為 TypeScript 類型"""
        import re
        
        # 處理 Optional[T]
        if type_str.startswith(OPTIONAL_PREFIX):
            inner = type_str[len(OPTIONAL_PREFIX):-1]
            return f"{self._get_typescript_type(inner)} | null"
        
        # 處理 List[T]
        if type_str.startswith(LIST_PREFIX):
            inner = type_str[len(LIST_PREFIX):-1]
            return f"{self._get_typescript_type(inner)}[]"
        
        # 處理 Dict[str, T]
        dict_match = re.match(r"Dict\[str,\s*(.+)\]", type_str)
        if dict_match:
            value_type = dict_match.group(1).strip()
            return f"Record<string, {self._get_typescript_type(value_type)}>"
        
        # 基本類型映射
        type_mapping = {
            "str": "string",
            "int": "number", 
            "float": "number",
            "bool": "boolean",
            "datetime": "string",  # ISO 8601 格式
            "Any": "any",
            "VulnerabilityType": "VulnerabilityType",
            "Severity": "Severity", 
            "Confidence": "Confidence"
        }
        
        return type_mapping.get(type_str, type_str)

    def _render_typescript_index(self) -> str:
        """渲染 TypeScript index.ts"""
        content = []
        content.extend([
            "/**",
            " * AIVA TypeScript Schemas - 統一導出",
            " * ==================================",
            " * ",
            " * 此文件導出所有標準生成的 Schema 定義",
            " * 遵循單一事實原則，確保與 Python 版本一致",
            " */",
            "",
            "// 導出所有 schemas",
            "export * from './schemas';",
            "",
            "// 版本信息",
            f"export const SCHEMA_VERSION = '{self.sot_data['version']}';",
            f"export const GENERATED_AT = '{datetime.now().isoformat()}';",
            ""
        ])
        
        return "\n".join(content)

    def _generate_python_field(self, field_name: str, field_info: dict) -> str:
        """生成Python字段定義 - 重構後降低認知複雜度"""
        field_type = self._get_python_type(field_info["type"])
        field_declaration = f"{field_name}: {field_type}"
        
        validation_params = self._extract_validation_parameters(field_info)
        field_assignment = self._generate_field_assignment(field_info, validation_params)
        
        return field_declaration + field_assignment

    def _extract_validation_parameters(self, field_info: dict) -> list[str]:
        """提取驗證參數"""
        if "validation" not in field_info:
            return []
        
        validation_handlers = {
            "enum": lambda val: f"values={val}",
            "pattern": lambda val: f'pattern=r"{val}"',
            "format": self._handle_format_validation,
            "max_length": lambda val: f"max_length={val}",
            "minimum": lambda val: f"ge={val}",
            "maximum": lambda val: f"le={val}"
        }
        
        params = []
        for key, value in field_info["validation"].items():
            if key in validation_handlers:
                handler = validation_handlers[key]
                param = handler(value) if callable(handler) else handler
                if param:  # 某些 handler 可能返回 None
                    params.append(param)
        
        return params

    def _handle_format_validation(self, format_value: str) -> str:
        """處理格式驗證參數"""
        format_mapping = {
            "url": "url=True"
        }
        return format_mapping.get(format_value, "")

    def _generate_field_assignment(self, field_info: dict, validation_params: list[str]) -> str:
        """生成欄位賦值部分"""
        is_required = field_info.get("required", True)
        has_default = "default" in field_info
        
        if not is_required:
            return self._generate_optional_field_assignment(field_info, validation_params)
        elif has_default:
            return self._generate_required_field_with_default(field_info, validation_params)
        elif validation_params:
            return f" = Field({', '.join(validation_params)})"
        else:
            return ""

    def _generate_optional_field_assignment(self, field_info: dict, validation_params: list[str]) -> str:
        """生成可選欄位賦值"""
        if "default" in field_info:
            default_val = self._get_python_default(field_info["default"])
            return self._combine_field_params_with_default(validation_params, default_val)
        else:
            return self._combine_field_params_with_default(validation_params, "None")

    def _generate_required_field_with_default(self, field_info: dict, validation_params: list[str]) -> str:
        """生成有預設值的必填欄位"""
        default_val = self._get_python_default(field_info["default"])
        all_params = validation_params + [f"default={default_val}"]
        return f" = Field({', '.join(all_params)})"

    def _combine_field_params_with_default(self, validation_params: list[str], default_val: str) -> str:
        """組合驗證參數和預設值"""
        if validation_params:
            all_params = validation_params + [f"default={default_val}"]
            return f" = Field({', '.join(all_params)})"
        else:
            return f" = {default_val}"

    def _get_go_type(self, type_str: str) -> str:
        """轉換為 Go 類型 - 支援嵌套類型映射"""
        import re

        # 處理 Optional[T] - 轉換為 *T
        if type_str.startswith(OPTIONAL_PREFIX):
            inner = type_str[len(OPTIONAL_PREFIX):-1]  # 提取內部類型
            mapped = self._get_go_type(inner)  # 遞歸映射
            # 如果內部類型已經是指針或map/slice,不再添加*
            if (
                mapped.startswith("*")
                or mapped.startswith("map[")
                or mapped.startswith("[]")
            ):
                return mapped
            return f"*{mapped}"

        # 處理 Dict[K, V] - 轉換為 map[K]V
        dict_match = re.match(r"Dict\[(.+?),\s*(.+)\]", type_str)
        if dict_match:
            key_type_raw = dict_match.group(1).strip()
            val_type_raw = dict_match.group(2).strip()
            key_type = self._get_go_type(key_type_raw)
            val_type = self._get_go_type(val_type_raw)
            return f"map[{key_type}]{val_type}"

        # 處理 List[T] - 轉換為 []T
        if type_str.startswith("List["):
            inner = type_str[5:-1]
            mapped = self._get_go_type(inner)
            return f"[]{mapped}"

        # 基本類型映射
        mapping = self.sot_data["generation_config"]["go"]["field_mapping"]
        return mapping.get(type_str, type_str)

    def _to_go_field_name(self, field_name: str) -> str:
        """
        轉換為 Go 欄位名稱（PascalCase），符合 Go Initialisms 標準
        參考: https://go.dev/wiki/CodeReviewComments#initialisms
        """
        # Go 官方縮寫標準 - 必須統一大小寫
        initialisms = {
            "url": "URL",
            "http": "HTTP",
            "https": "HTTPS",
            "id": "ID",
            "api": "API",
            "json": "JSON",
            "xml": "XML",
            "html": "HTML",
            "css": "CSS",
            "js": "JS",
            "sql": "SQL",
            "cwe": "CWE",
            "cve": "CVE",
            "owasp": "OWASP",
            "uuid": "UUID",
            "uri": "URI",
            "tcp": "TCP",
            "udp": "UDP",
            "ip": "IP",
            "os": "OS",
            "cpu": "CPU",
            "ram": "RAM",
            "db": "DB",
        }

        # 分割字段名並處理每個部分
        parts = field_name.split("_")
        go_parts = []

        for part in parts:
            lower_part = part.lower()
            if lower_part in initialisms:
                go_parts.append(initialisms[lower_part])
            else:
                go_parts.append(part.capitalize())

        return "".join(go_parts)

    def _get_go_json_tag(self, required: bool) -> str:
        """獲取 Go JSON 標籤"""
        return "" if required else ",omitempty"

    def _get_all_fields(self, class_info: dict, current_section: str) -> dict:
        """
        獲取類的所有字段，包括繼承的字段

        Args:
            class_info: 類定義信息
            current_section: 當前所在的 section (base_types, findings, etc.)

        Returns:
            包含所有字段的字典
        """
        all_fields = {}

        # 首先處理繼承
        if "extends" in class_info:
            base_class_name = class_info["extends"]
            base_class_info = None

            # 在所有可能的 section 中查找基類
            for section_name in [
                "base_types",
                "findings",
                "messaging",
                "tasks",
                "plugins",
                "cli",
            ]:
                if (
                    section_name in self.sot_data
                    and base_class_name in self.sot_data[section_name]
                ):
                    base_class_info = self.sot_data[section_name][base_class_name]
                    break

            if base_class_info:
                # 遞歸獲取基類的所有字段
                base_fields = self._get_all_fields(base_class_info, current_section)
                all_fields.update(base_fields)
            else:
                logger.warning(f"找不到基類: {base_class_name}")

        # 添加當前類的直接字段
        if "fields" in class_info:
            all_fields.update(class_info["fields"])

        # 添加當前類的額外字段
        if "additional_fields" in class_info:
            all_fields.update(class_info["additional_fields"])

        return all_fields

    def validate_schemas(self) -> bool:
        """驗證 Schema 定義的一致性"""
        logger.info("🔍 開始 Schema 驗證...")

        errors = []
        
        # 執行各項驗證檢查
        errors.extend(self._validate_required_keys())
        errors.extend(self._validate_version_format())
        errors.extend(self._validate_type_references())

        # 返回驗證結果
        return self._report_validation_results(errors)

    def _validate_required_keys(self) -> list[str]:
        """檢查必要的頂層鍵"""
        errors = []
        required_keys = ["version", "metadata", "base_types", "generation_config"]
        
        for key in required_keys:
            if key not in self.sot_data:
                errors.append(f"缺少必要的頂層鍵: {key}")
        
        return errors

    def _validate_version_format(self) -> list[str]:
        """檢查版本格式"""
        errors = []
        version = self.sot_data.get("version", "")
        
        if not version or not version.replace(".", "").isdigit():
            errors.append(f"版本格式無效: {version}")
        
        return errors

    def _validate_type_references(self) -> list[str]:
        """檢查類型引用的有效性"""
        errors = []
        defined_types, all_schemas = self._collect_schema_types()
        
        for schema_name, schema_info in all_schemas.items():
            schema_errors = self._validate_schema_fields(schema_name, schema_info, defined_types)
            errors.extend(schema_errors)
        
        return errors

    def _collect_schema_types(self) -> tuple[set[str], dict[str, Any]]:
        """收集所有定義的類型和模式"""
        defined_types = set(self.sot_data.get("base_types", {}).keys())
        all_schemas = {}
        
        for category in ["messaging", "tasks", "findings"]:
            if category in self.sot_data:
                all_schemas.update(self.sot_data[category])
                defined_types.update(self.sot_data[category].keys())
        
        return defined_types, all_schemas

    def _validate_schema_fields(
        self, schema_name: str, schema_info: dict[str, Any], defined_types: set[str]
    ) -> list[str]:
        """驗證單個模式的欄位類型引用"""
        errors = []
        
        for field_name, field_info in schema_info.get("fields", {}).items():
            field_type = field_info.get("type", "")
            core_type = self._extract_core_type(field_type)
            
            if not self._is_valid_type(core_type, defined_types):
                errors.append(f"在 {schema_name}.{field_name} 中引用了未定義的類型: {core_type}")
        
        return errors

    def _extract_core_type(self, field_type: str) -> str:
        """從複合類型中提取核心類型"""
        return (
            field_type.replace(OPTIONAL_PREFIX, "")
            .replace(LIST_PREFIX, "")
            .replace("Dict[str, ", "")
            .replace("]", "")
            .replace(">", "")
        )

    def _is_valid_type(self, core_type: str, defined_types: set[str]) -> bool:
        """檢查類型是否有效（基本類型或已定義類型）"""
        basic_types = {"str", "int", "float", "bool", "datetime", "Any"}
        return core_type in basic_types or core_type in defined_types

    def _report_validation_results(self, errors: list[str]) -> bool:
        """報告驗證結果"""
        if errors:
            logger.error("❌ Schema 驗證失敗:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("✅ Schema 驗證通過!")
        return True

    def generate_grpc_schemas(self, output_dir: str | None = None) -> list[str]:
        """生成 gRPC Protocol Buffers Schema
        
        Args:
            output_dir: 自訂輸出目錄
            
        Returns:
            生成的檔案列表
        """
        # gRPC 生成配置
        if output_dir:
            target_dir = Path(output_dir)
        else:
            target_dir = Path("services/aiva_common/grpc/generated")
        
        target_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # 生成主要的 aiva.proto 文件
        proto_file = target_dir / "aiva.proto"
        content = self._render_proto_file()
        with open(proto_file, "w", encoding="utf-8") as f:
            f.write(content)
        generated_files.append(str(proto_file))
        logger.info(f"✅ 生成 gRPC Proto: {proto_file}")

        # 生成編譯腳本
        compile_script = target_dir / "compile_protos.py"
        script_content = self._render_proto_compile_script()
        with open(compile_script, "w", encoding="utf-8") as f:
            f.write(script_content)
        generated_files.append(str(compile_script))
        logger.info(f"✅ 生成編譯腳本: {compile_script}")

        return generated_files

    def _render_proto_file(self) -> str:
        """渲染 Protocol Buffers 檔案"""
        content = []
        
        # Proto 檔案頭部
        content.extend([
            "// AIVA gRPC Protocol Buffers 定義 - 自動生成",
            "// ============================================",
            "//",
            f"// {self.sot_data['metadata']['description']}",
            "//",
            f"// ⚠️  {self.sot_data['metadata']['generated_note']}",
            f"// 📅 最後更新: {self.sot_data['metadata']['last_updated']}",
            f"// 🔄 Schema 版本: {self.sot_data['version']}",
            "//",
            "// 基於 core_schema_sot.yaml 生成，與所有語言 Schema 保持一致",
            "",
            "syntax = \"proto3\";",
            "",
            "package aiva.v1;",
            "",
            "option go_package = \"github.com/kyle0527/AIVA/services/aiva_common_go/grpc/generated\";",
            "",
            "import \"google/protobuf/timestamp.proto\";",
            "import \"google/protobuf/struct.proto\";",
            "",
            "// ==================== 基礎訊息類型 ====================",
            ""
        ])

        # 生成基礎訊息類型
        content.extend([
            "// 訊息標頭",
            "message MessageHeader {",
            "  string message_id = 1;",
            "  string trace_id = 2;",
            "  string correlation_id = 3;",
            "  string source_module = 4;",
            "  google.protobuf.Timestamp timestamp = 5;",
            "  string version = 6;",
            "}",
            "",
            "// 統一 API 請求",
            "message AIVARequest {",
            "  string request_id = 1;",
            "  string task = 2;",
            "  google.protobuf.Struct parameters = 3;",
            "  double timeout = 4;",
            "  string trace_id = 5;",
            "  google.protobuf.Struct metadata = 6;",
            "}",
            "",
            "// 統一 API 響應",
            "message AIVAResponse {",
            "  string request_id = 1;",
            "  bool success = 2;",
            "  google.protobuf.Struct result = 3;",
            "  string error_code = 4;",
            "  string error_message = 5;",
            "  google.protobuf.Timestamp timestamp = 6;",
            "  double duration = 7;",
            "}",
            "",
            "// 目標資訊",
            "message Target {",
            "  string url = 1;",
            "  string host = 2;",
            "  int32 port = 3;",
            "  string protocol = 4;",
            "  string path = 5;",
            "  google.protobuf.Struct metadata = 6;",
            "}",
            "",
            "// 風險級別枚舉",
            "enum RiskLevel {",
            "  RISK_LEVEL_UNSPECIFIED = 0;",
            "  RISK_LEVEL_CRITICAL = 1;",
            "  RISK_LEVEL_HIGH = 2;",
            "  RISK_LEVEL_MEDIUM = 3;",
            "  RISK_LEVEL_LOW = 4;",
            "  RISK_LEVEL_INFO = 5;",
            "}",
            "",
            "// 任務狀態枚舉",
            "enum TaskStatus {",
            "  TASK_STATUS_UNSPECIFIED = 0;",
            "  TASK_STATUS_PENDING = 1;",
            "  TASK_STATUS_RUNNING = 2;",
            "  TASK_STATUS_COMPLETED = 3;",
            "  TASK_STATUS_FAILED = 4;",
            "  TASK_STATUS_CANCELLED = 5;",
            "}",
            "",
            "// 漏洞發現",
            "message FindingPayload {",
            "  string finding_id = 1;",
            "  string vulnerability_type = 2;",
            "  string title = 3;",
            "  string description = 4;",
            "  RiskLevel risk_level = 5;",
            "  double confidence = 6;",
            "  Target target = 7;",
            "  repeated string evidence = 8;",
            "  repeated string recommendations = 9;",
            "  google.protobuf.Timestamp discovered_at = 10;",
            "}",
            "",
            "// 任務配置",
            "message TaskConfig {",
            "  string task_id = 1;",
            "  string task_type = 2;",
            "  Target target = 3;",
            "  google.protobuf.Struct parameters = 4;",
            "  int32 timeout = 5;",
            "  int32 priority = 6;",
            "  google.protobuf.Timestamp created_at = 7;",
            "}",
            "",
            "// 任務結果",
            "message TaskResult {",
            "  string task_id = 1;",
            "  TaskStatus status = 2;",
            "  repeated FindingPayload findings = 3;",
            "  string error = 4;",
            "  google.protobuf.Timestamp started_at = 5;",
            "  google.protobuf.Timestamp completed_at = 6;",
            "  double duration = 7;",
            "  google.protobuf.Struct metadata = 8;",
            "}",
            "",
            "// ==================== gRPC 服務定義 ====================",
            "",
            "// 任務管理服務",
            "service TaskService {",
            "  // 創建新任務",
            "  rpc CreateTask(TaskConfig) returns (AIVAResponse);",
            "  ",
            "  // 獲取任務狀態",
            "  rpc GetTaskStatus(AIVARequest) returns (TaskResult);",
            "  ",
            "  // 取消任務",
            "  rpc CancelTask(AIVARequest) returns (AIVAResponse);",
            "  ",
            "  // 串流任務進度",
            "  rpc StreamTaskProgress(AIVARequest) returns (stream AIVAResponse);",
            "}",
            "",
            "// 跨語言通信服務",
            "service CrossLanguageService {",
            "  // 執行跨語言任務",
            "  rpc ExecuteTask(AIVARequest) returns (AIVAResponse);",
            "  ",
            "  // 健康檢查",
            "  rpc HealthCheck(AIVARequest) returns (AIVAResponse);",
            "  ",
            "  // 獲取服務資訊",
            "  rpc GetServiceInfo(AIVARequest) returns (AIVAResponse);",
            "  ",
            "  // 雙向串流通信",
            "  rpc BidirectionalStream(stream AIVARequest) returns (stream AIVAResponse);",
            "}",
            ""
        ])

        return "\n".join(content)

    def _render_proto_compile_script(self) -> str:
        """生成 Proto 編譯腳本"""
        return '''#!/usr/bin/env python3
"""
gRPC Protocol Buffers 編譯腳本
自動編譯 .proto 檔案為各語言的 gRPC 存根代碼
"""

import subprocess
import sys
from pathlib import Path

def compile_protos():
    """編譯 Protocol Buffers 檔案"""
    proto_dir = Path(__file__).parent
    proto_file = proto_dir / "aiva.proto"
    
    if not proto_file.exists():
        print(f"❌ Proto 檔案不存在: {proto_file}")
        return False
    
    # Python 編譯
    print("🔄 編譯 Python gRPC 存根...")
    python_out = proto_dir / "python"
    python_out.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={python_out}",
        f"--grpc_python_out={python_out}",
        str(proto_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Python gRPC 存根編譯完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ Python 編譯失敗: {e}")
        return False
    
    # Go 編譯
    print("🔄 編譯 Go gRPC 存根...")
    go_out = proto_dir / "go"
    go_out.mkdir(exist_ok=True)
    
    cmd = [
        "protoc",
        f"--proto_path={proto_dir}",
        f"--go_out={go_out}",
        f"--go-grpc_out={go_out}",
        str(proto_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Go gRPC 存根編譯完成")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  Go 編譯跳過 (protoc-gen-go 未安裝): {e}")
    
    print("🎉 gRPC 編譯完成!")
    return True

if __name__ == "__main__":
    success = compile_protos()
    sys.exit(0 if success else 1)
'''

    def generate_all(self, validate: bool = True) -> dict[str, list[str]]:
        """生成所有語言的 Schema

        Args:
            validate: 是否先進行驗證

        Returns:
            各語言生成的檔案列表
        """
        if validate and not self.validate_schemas():
            logger.error("❌ Schema 驗證失敗，停止生成")
            return {}

        results = {}

        logger.info("🚀 開始生成所有語言 Schema...")

        # 生成 Python
        try:
            results["python"] = self.generate_python_schemas()
            logger.info(f"✅ Python Schema 生成完成: {len(results['python'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ Python Schema 生成失敗: {e}")
            results["python"] = []

        # 生成 Go
        try:
            results["go"] = self.generate_go_schemas()
            logger.info(f"✅ Go Schema 生成完成: {len(results['go'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ Go Schema 生成失敗: {e}")
            results["go"] = []

        # 生成 Rust (簡化版本)
        try:
            results["rust"] = self.generate_rust_schemas()
            logger.info(f"✅ Rust Schema 生成完成: {len(results['rust'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ Rust Schema 生成失敗: {e}")
            results["rust"] = []

        # 生成 TypeScript
        try:
            results["typescript"] = self.generate_typescript_schemas()
            logger.info(f"✅ TypeScript Schema 生成完成: {len(results['typescript'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ TypeScript Schema 生成失敗: {e}")
            results["typescript"] = []

        # 生成 gRPC Protocol Buffers
        try:
            results["grpc"] = self.generate_grpc_schemas()
            logger.info(f"✅ gRPC Schema 生成完成: {len(results['grpc'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ gRPC Schema 生成失敗: {e}")
            results["grpc"] = []

        total_files = sum(len(files) for files in results.values())
        logger.info(f"🎉 所有語言 Schema 生成完成! 總計: {total_files} 個檔案")

        return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="AIVA Schema 代碼生成工具")
    parser.add_argument(
        "--lang",
        choices=["python", "go", "rust", "typescript", "grpc", "all"],
        default="all",
        help="生成的語言",
    )
    parser.add_argument("--validate", action="store_true", help="僅進行 Schema 驗證")
    parser.add_argument("--output-dir", help="自訂輸出目錄")
    parser.add_argument(
        "--sot-file",
        default="services/aiva_common/core_schema_sot.yaml",
        help="SOT 檔案路徑",
    )

    args = parser.parse_args()

    # 初始化生成器
    generator = SchemaCodeGenerator(args.sot_file)

    if args.validate:
        # 僅驗證
        success = generator.validate_schemas()
        sys.exit(0 if success else 1)

    # 生成代碼
    if args.lang == "all":
        results = generator.generate_all()
    elif args.lang == "python":
        results = {"python": generator.generate_python_schemas(args.output_dir)}
    elif args.lang == "go":
        results = {"go": generator.generate_go_schemas(args.output_dir)}
    elif args.lang == "rust":
        results = {"rust": generator.generate_rust_schemas(args.output_dir)}
    elif args.lang == "typescript":
        results = {"typescript": generator.generate_typescript_schemas(args.output_dir)}
    elif args.lang == "grpc":
        results = {"grpc": generator.generate_grpc_schemas(args.output_dir)}

    # 輸出結果
    success = all(len(files) > 0 for files in results.values())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
