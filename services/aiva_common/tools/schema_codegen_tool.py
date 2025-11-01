#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Legacy import bridge - moved to plugins/aiva_converters/core/schema_codegen_tool.py""""""

"""

Schema Code Generator - Legacy Import BridgeAIVA Schema Code Generation Tool - Legacy Bridge

try:

    from plugins.aiva_converters.core.schema_codegen_tool import *==========================================================================================

except ImportError:

    from schema_codegen_tool_backup import *



if __name__ == "__main__":⚠️  DEPRECATION NOTICE ⚠️⚠️  DEPRECATION NOTICE ⚠️

    print("Redirected to plugins/aiva_converters/core/schema_codegen_tool.py")
此檔案已移至 AIVA Converters Plugin。此檔案已移至 AIVA Converters Plugin。

請使用新的插件位置：plugins/aiva_converters/core/schema_codegen_tool.py

Original Location: services/aiva_common/tools/schema_codegen_tool.py

New Location: plugins/aiva_converters/core/schema_codegen_tool.pyThis file provides backward compatibility for existing imports.

The actual implementation has been moved to the AIVA Converters Plugin.

This file provides backward compatibility for existing imports.

"""Original Location: services/aiva_common/tools/schema_codegen_tool.py  

New Location: plugins/aiva_converters/core/schema_codegen_tool.py

import sys

import warningsLegacy Usage (still works):

import importlib.util    python services/aiva_common/tools/schema_codegen_tool.py --generate-all

from pathlib import Path    

New Usage (recommended):

# Issue deprecation warning    python plugins/aiva_converters/core/schema_codegen_tool.py --generate-all

warnings.warn("""

    "Importing from 'services.aiva_common.tools.schema_codegen_tool' is deprecated. "

    "Please use 'plugins.aiva_converters.core.schema_codegen_tool' instead.",import argparse

    DeprecationWarning,import logging

    stacklevel=2

)# 設定日誌 - 支援 Unicode

import sys

# Try to import from the new plugin locationfrom datetime import datetime

try:from pathlib import Path

    plugin_path = Path(__file__).parent.parent.parent.parent / "plugins"from typing import Any

    sys.path.insert(0, str(plugin_path))

    import yaml

    from aiva_converters.core.schema_codegen_tool import SchemaCodeGeneratorfrom jinja2 import Environment, FileSystemLoader, Template

    from aiva_converters.core.schema_codegen_tool import *

    # sys.stdout.reconfigure(encoding='utf-8')  # 僅在支持的 Python 版本中可用

    print("✅ Successfully loaded SchemaCodeGenerator from AIVA Converters Plugin")# sys.stderr.reconfigure(encoding='utf-8')  # 僅在支持的 Python 版本中可用

    

except ImportError as e:logging.basicConfig(

    print(f"❌ Plugin import failed: {e}")    level=logging.INFO,

    print("🔄 Trying backup implementation...")    format="%(asctime)s - %(levelname)s - %(message)s",

        handlers=[

    # Load backup if plugin fails        logging.StreamHandler(),

    backup_file = Path(__file__).parent / "schema_codegen_tool_backup.py"        logging.FileHandler("schema_codegen.log", encoding="utf-8"),

    if backup_file.exists():    ],

        try:)

            spec = importlib.util.spec_from_file_location("schema_codegen_backup", backup_file)logger = logging.getLogger(__name__)

            backup_module = importlib.util.module_from_spec(spec)

            spec.loader.exec_module(backup_module)

            SchemaCodeGenerator = backup_module.SchemaCodeGeneratorclass SchemaCodeGenerator:

            print("⚠️  Using backup implementation")    """Schema 代碼生成器 - 支援多語言自動生成"""

        except Exception as backup_error:

            print(f"❌ Backup import also failed: {backup_error}")    def __init__(self, sot_file: str = "services/aiva_common/core_schema_sot.yaml"):

            raise ImportError(        """初始化代碼生成器

                "Cannot import SchemaCodeGenerator from plugin or backup. "

                "Please ensure the AIVA Converters Plugin is properly installed."        Args:

            ) from e            sot_file: Schema SOT YAML 檔案路徑

    else:        """

        raise ImportError(        self.sot_file = Path(sot_file)

            "No backup implementation available. "        self.sot_data: dict[str, Any] = {}

            "Please ensure the AIVA Converters Plugin is properly installed."        self.jinja_env = Environment(

        ) from e            loader=FileSystemLoader(Path(__file__).parent / "templates"),

            trim_blocks=True,

if __name__ == "__main__":            lstrip_blocks=True,

    print("🔄 Schema Code Generator - Legacy Bridge")        )

    print("📂 Original location: services/aiva_common/tools/schema_codegen_tool.py")

    print("📦 New location: plugins/aiva_converters/core/schema_codegen_tool.py")        # 載入 SOT 資料

    print("")        self._load_sot_data()

    print("To use the new plugin directly:")

    print("python plugins/aiva_converters/core/schema_codegen_tool.py [args]")    def _load_sot_data(self) -> None:

            """載入 Schema SOT 資料"""

    # Try to run the main function if available        try:

    try:            with open(self.sot_file, encoding="utf-8") as f:

        if hasattr(sys.modules.get(__name__), 'main'):                self.sot_data = yaml.safe_load(f)

            main()            logger.info(f"✅ 成功載入 SOT 檔案: {self.sot_file}")

        else:        except FileNotFoundError:

            print("No main function available in current module")            logger.error(f"❌ SOT 檔案不存在: {self.sot_file}")

    except Exception as e:            sys.exit(1)

        print(f"Error running main: {e}")        except yaml.YAMLError as e:
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
        """渲染 Go 統一 Schema"""
        content = []
        content.append("// AIVA Go Schema - 自動生成")  # type: ignore
        content.append("// ===========================")  # type: ignore
        content.append("//")  # type: ignore
        content.append(f'// {self.sot_data["metadata"]["description"]}')  # type: ignore
        content.append("//")  # type: ignore
        content.append(f'// ⚠️  {self.sot_data["metadata"]["generated_note"]}')  # type: ignore
        content.append(f'// 📅 最後更新: {self.sot_data["metadata"]["last_updated"]}')  # type: ignore
        content.append(f'// 🔄 Schema 版本: {self.sot_data["version"]}')  # type: ignore
        content.append("")  # type: ignore

        # 添加imports
        for imp in self.sot_data["generation_config"]["go"]["base_imports"]:
            content.append(imp)  # type: ignore
        content.append("")  # type: ignore

        # 枚舉類型
        if "enums" in self.sot_data:
            content.append("// ==================== 枚舉類型 ====================")  # type: ignore
            content.append("")  # type: ignore

            for enum_name, enum_info in self.sot_data["enums"].items():
                content.append(f'// {enum_name} {enum_info.get("description", "")}')  # type: ignore
                content.append(f"type {enum_name} string")  # type: ignore
                content.append("")  # type: ignore
                content.append("const (")  # type: ignore

                for value_key, value_desc in enum_info.get("values", {}).items():
                    const_name = f"{enum_name}{value_key.title()}"
                    content.append(f'    {const_name:<30} {enum_name} = "{value_key}"  // {value_desc}')  # type: ignore

                content.append(")")  # type: ignore
                content.append("")  # type: ignore

        # 基礎類型
        content.append("// ==================== 基礎類型 ====================")  # type: ignore
        content.append("")  # type: ignore

        for class_name, class_info in self.sot_data["base_types"].items():
            content.append(f'// {class_name} {class_info["description"]}')  # type: ignore
            content.append(f"type {class_name} struct {{")  # type: ignore

            for field_name, field_info in class_info["fields"].items():
                go_name = self._to_go_field_name(field_name)
                go_type = self._get_go_type(field_info["type"])
                json_tag = self._get_go_json_tag(field_info.get("required", True))
                content.append(f'    {go_name:<20} {go_type:<25} `json:"{field_name}{json_tag}"`  // {field_info["description"]}')  # type: ignore

            content.append("}")  # type: ignore
            content.append("")  # type: ignore

        # 其他類別 - 包含所有新增的 Schema 分類
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
                content.append(f"// ==================== {title} ====================")  # type: ignore
                content.append("")  # type: ignore

                for class_name, class_info in self.sot_data[section].items():
                    content.append(f'// {class_name} {class_info["description"]}')  # type: ignore
                    content.append(f"type {class_name} struct {{")  # type: ignore

                    # 獲取所有字段（包括繼承的字段）
                    all_fields = self._get_all_fields(class_info, section)

                    # 生成所有字段
                    for field_name, field_info in all_fields.items():
                        go_name = self._to_go_field_name(field_name)
                        go_type = self._get_go_type(field_info["type"])
                        json_tag = self._get_go_json_tag(
                            field_info.get("required", True)
                        )
                        content.append(f'    {go_name:<20} {go_type:<25} `json:"{field_name}{json_tag}"`  // {field_info["description"]}')  # type: ignore

                    content.append("}")  # type: ignore
                    content.append("")  # type: ignore

        return "\n".join(content)

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
        """渲染 Rust 結構體"""
        description = struct_data.get("description", f"{struct_name} 結構體")
        # 支持兩種字段定義格式：properties (標準) 和 fields (AIVA特有)
        properties = struct_data.get("properties", struct_data.get("fields", {}))
        required = struct_data.get("required", [])

        rust_struct = f"""/// {description}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct {struct_name} {{"""

        for field_name, field_data in properties.items():
            field_desc = field_data.get("description", f"{field_name} 欄位")
            original_type = field_data.get("type")
            field_required = field_data.get(
                "required", True
            )  # AIVA schema 中 required 可能在 field 級別
            is_required_in_struct = field_name in required or field_required

            # 如果類型已經是 Optional，不需要再包裝
            if original_type and original_type.startswith("Optional["):
                field_type = self._convert_to_rust_type(original_type, field_data)
                is_optional = True
            elif not is_required_in_struct:
                # 非必填欄位，包裝為 Option
                base_type = self._convert_to_rust_type(original_type, field_data)
                field_type = f"Option<{base_type}>"
                is_optional = True
            else:
                # 必填欄位
                field_type = self._convert_to_rust_type(original_type, field_data)
                is_optional = False

            rust_struct += f"""
    /// {field_desc}"""

            if is_optional:
                rust_struct += f"""
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub {field_name}: {field_type},"""
            else:
                rust_struct += f"""
    pub {field_name}: {field_type},"""

        rust_struct += (
            """
}

impl """
            + struct_name
            + """ {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {"""
        )

        for field_name, field_data in properties.items():
            original_type = field_data.get("type")
            field_required = field_data.get("required", True)
            is_required_in_struct = field_name in required or field_required

            if original_type and original_type.startswith("Optional["):
                rust_struct += f"""
            {field_name}: None,"""
            elif not is_required_in_struct:
                rust_struct += f"""
            {field_name}: None,"""
            else:
                # 需要重新計算 field_type 以獲得正確的預設值
                if original_type and original_type.startswith("Optional["):
                    converted_type = self._convert_to_rust_type(
                        original_type, field_data
                    )
                elif not is_required_in_struct:
                    base_type = self._convert_to_rust_type(original_type, field_data)
                    converted_type = f"Option<{base_type}>"
                else:
                    converted_type = self._convert_to_rust_type(
                        original_type, field_data
                    )

                default_value = self._get_rust_default_value(converted_type, field_data)
                rust_struct += f"""
            {field_name}: {default_value},"""

        rust_struct += """
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {"""

        # 添加必填欄位驗證
        for field_name in required:
            if field_name in properties:
                field_type = properties[field_name].get("type")
                if field_type == "string":
                    rust_struct += f"""
        if self.{field_name}.is_empty() {{
            return Err("Field '{field_name}' is required and cannot be empty".to_string());
        }}"""

        rust_struct += (
            """
        Ok(())
    }
}

impl Default for """
            + struct_name
            + """ {
    fn default() -> Self {
        Self::new()
    }
}"""
        )

        return rust_struct

    def _convert_to_rust_type(
        self, json_type: str, field_data: dict[str, Any] | None = None
    ) -> str:
        """將 JSON Schema 類型轉換為 Rust 類型"""
        if field_data is None:
            field_data = {}

        # 處理 Optional 類型
        if json_type and json_type.startswith("Optional["):
            inner_type = json_type[9:-1]  # 移除 'Optional[' 和 ']'
            return f"Option<{self._convert_to_rust_type(inner_type, field_data)}>"

        # 處理 List 類型
        if json_type and json_type.startswith("List["):
            inner_type = json_type[5:-1]  # 移除 'List[' 和 ']'
            return f"Vec<{self._convert_to_rust_type(inner_type, field_data)}>"

        # 處理 Dict 類型
        if json_type and json_type.startswith("Dict["):
            # Dict[str, str] -> HashMap<String, String>
            # Dict[str, Any] -> HashMap<String, serde_json::Value>
            dict_content = json_type[5:-1]  # 移除 'Dict[' 和 ']'
            if dict_content == "str, str":
                return "std::collections::HashMap<String, String>"
            elif dict_content == "str, Any":
                return "std::collections::HashMap<String, serde_json::Value>"
            else:
                return "std::collections::HashMap<String, serde_json::Value>"

        # 基本類型映射
        type_mapping = {
            "str": "String",
            "string": "String",
            "int": "i32",
            "integer": "i32",
            "float": "f64",
            "number": "f64",
            "bool": "bool",
            "boolean": "bool",
            "datetime": "chrono::DateTime<chrono::Utc>",
            "Any": "serde_json::Value",
        }

        # 檢查是否為自定義類型（存在於 SOT 中的結構體）
        all_types = set()
        for category in ["base_types", "messaging", "tasks", "findings"]:
            if category in self.sot_data:
                all_types.update(self.sot_data[category].keys())

        if json_type in all_types:
            return json_type  # 自定義類型保持原名

        # 處理枚舉類型
        if "enum" in field_data:
            return "String"  # Serde 會處理枚舉驗證

        # 處理格式化類型
        format_type = field_data.get("format")
        if format_type == "date-time":
            return "chrono::DateTime<chrono::Utc>"
        elif format_type == "uuid":
            return "uuid::Uuid"
        elif format_type == "uri" or format_type == "url":
            return "url::Url"

        return type_mapping.get(json_type, "String")

    def _get_rust_default_value(
        self, json_type: str, field_data: dict[str, Any] | None = None
    ) -> str:
        """獲取 Rust 類型的默認值"""
        if field_data is None:
            field_data = {}

        # 檢查是否有預設值定義
        if "default" in field_data:
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

        # 處理 Optional 類型 (已經由上層處理為 Option<T>)
        if json_type and json_type.startswith("Option<"):
            return "None"

        # 處理 Vec 類型
        if json_type and json_type.startswith("Vec<"):
            return "Vec::new()"

        # 處理 HashMap 類型
        if json_type and json_type.startswith("std::collections::HashMap<"):
            return "std::collections::HashMap::new()"

        # 基本類型預設值
        defaults = {
            "String": "String::new()",
            "str": "String::new()",
            "string": "String::new()",
            "i32": "0",
            "int": "0",
            "integer": "0",
            "f64": "0.0",
            "float": "0.0",
            "number": "0.0",
            "bool": "false",
            "boolean": "false",
            "chrono::DateTime<chrono::Utc>": "chrono::Utc::now()",
            "serde_json::Value": "serde_json::Value::Null",
            "uuid::Uuid": "uuid::Uuid::new_v4()",
            "url::Url": 'url::Url::parse("https://example.com").unwrap()',
        }

        # 檢查是否為自定義類型
        all_types = set()
        for category in ["base_types", "messaging", "tasks", "findings"]:
            if category in self.sot_data:
                all_types.update(self.sot_data[category].keys())

        if json_type in all_types:
            return f"{json_type}::default()"

        return defaults.get(json_type, "String::new()")

    def _get_python_type(self, type_str: str) -> str:
        """轉換為 Python 類型"""
        mapping = self.sot_data["generation_config"]["python"]["field_mapping"]
        return mapping.get(type_str, type_str)

    def _get_python_default(self, default_value: Any, type_str: str) -> str:
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

    def _get_python_validation(self, key: str, value: Any) -> str:
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
        lines.append(f"/**")
        lines.append(f" * {interface_info['description']}")
        lines.append(f" */")
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
        if type_str.startswith("Optional["):
            inner = type_str[9:-1]
            return f"{self._get_typescript_type(inner)} | null"
        
        # 處理 List[T]
        if type_str.startswith("List["):
            inner = type_str[5:-1]
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
        """生成Python字段定義"""
        field_type = self._get_python_type(field_info["type"])
        parts = [f"{field_name}: {field_type}"]

        # 處理預設值和Field參數
        field_params = []

        # 添加驗證參數
        if "validation" in field_info:
            for key, value in field_info["validation"].items():
                if key == "enum":
                    field_params.append(f"values={value}")  # type: ignore
                elif key == "pattern":
                    field_params.append(f'pattern=r"{value}"')  # type: ignore
                elif key == "format":
                    # Pydantic v2 format handling
                    if value == "url":
                        field_params.append("url=True")  # type: ignore
                elif key == "max_length":
                    field_params.append(f"max_length={value}")  # type: ignore
                elif key == "minimum":
                    field_params.append(f"ge={value}")  # type: ignore
                elif key == "maximum":
                    field_params.append(f"le={value}")  # type: ignore

        # 處理預設值
        if not field_info.get("required", True):
            if "default" in field_info:
                default_val = self._get_python_default(
                    field_info["default"], field_info["type"]
                )
                if field_params:
                    field_params.append(f"default={default_val}")  # type: ignore
                    parts.append(f" = Field({', '.join(field_params)})")  # type: ignore
                else:
                    parts.append(f" = {default_val}")  # type: ignore
            else:
                if field_params:
                    field_params.append("default=None")  # type: ignore
                    parts.append(f" = Field({', '.join(field_params)})")  # type: ignore
                else:
                    parts.append(" = None")  # type: ignore
        elif "default" in field_info:
            default_val = self._get_python_default(
                field_info["default"], field_info["type"]
            )
            field_params.append(f"default={default_val}")  # type: ignore
            parts.append(f" = Field({', '.join(field_params)})")  # type: ignore
        elif field_params:
            parts.append(f" = Field({', '.join(field_params)})")  # type: ignore

        return "".join(parts)

    def _get_go_type(self, type_str: str) -> str:
        """轉換為 Go 類型 - 支援嵌套類型映射"""
        import re

        # 處理 Optional[T] - 轉換為 *T
        if type_str.startswith("Optional["):
            inner = type_str[9:-1]  # 提取內部類型
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

        # 檢查必要的頂層鍵
        required_keys = ["version", "metadata", "base_types", "generation_config"]
        for key in required_keys:
            if key not in self.sot_data:
                errors.append(f"缺少必要的頂層鍵: {key}")  # type: ignore

        # 檢查版本格式
        version = self.sot_data.get("version", "")
        if not version or not version.replace(".", "").isdigit():
            errors.append(f"版本格式無效: {version}")  # type: ignore

        # 檢查類型引用
        defined_types = set(self.sot_data.get("base_types", {}).keys())
        all_schemas = {}

        for category in ["messaging", "tasks", "findings"]:
            if category in self.sot_data:
                all_schemas.update(self.sot_data[category])
                defined_types.update(self.sot_data[category].keys())

        # 檢查類型引用的有效性
        for schema_name, schema_info in all_schemas.items():
            for field_name, field_info in schema_info.get("fields", {}).items():
                field_type = field_info.get("type", "")
                # 移除泛型包裝檢查核心類型
                core_type = (
                    field_type.replace("Optional[", "")
                    .replace("List[", "")
                    .replace("Dict[str, ", "")
                    .replace("]", "")
                    .replace(">", "")
                )
                if core_type in ["str", "int", "float", "bool", "datetime", "Any"]:
                    continue
                if core_type not in defined_types:
                    errors.append(f"在 {schema_name}.{field_name} 中引用了未定義的類型: {core_type}")  # type: ignore

        if errors:
            logger.error("❌ Schema 驗證失敗:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("✅ Schema 驗證通過!")
        return True

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

        total_files = sum(len(files) for files in results.values())
        logger.info(f"🎉 所有語言 Schema 生成完成! 總計: {total_files} 個檔案")

        return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="AIVA Schema 代碼生成工具")
    parser.add_argument(
        "--lang",
        choices=["python", "go", "rust", "typescript", "all"],
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

    # 輸出結果
    success = all(len(files) > 0 for files in results.values())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
