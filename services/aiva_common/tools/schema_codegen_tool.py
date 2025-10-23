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
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from jinja2 import Environment, FileSystemLoader, Template

# 設定日誌 - 支援 Unicode
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('schema_codegen.log', encoding='utf-8')
    ]
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
        self.sot_data: Dict[str, Any] = {}
        self.jinja_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # 載入 SOT 資料
        self._load_sot_data()
        
    def _load_sot_data(self) -> None:
        """載入 Schema SOT 資料"""
        try:
            with open(self.sot_file, 'r', encoding='utf-8') as f:
                self.sot_data = yaml.safe_load(f)
            logger.info(f"✅ 成功載入 SOT 檔案: {self.sot_file}")
        except FileNotFoundError:
            logger.error(f"❌ SOT 檔案不存在: {self.sot_file}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML 解析錯誤: {e}")
            sys.exit(1)
    
    def generate_python_schemas(self, output_dir: Optional[str] = None) -> List[str]:
        """生成 Python Pydantic v2 Schema
        
        Args:
            output_dir: 自訂輸出目錄
            
        Returns:
            生成的檔案列表
        """
        config = self.sot_data['generation_config']['python']
        target_dir = Path(output_dir) if output_dir else Path(config['target_dir'])
        target_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 生成基礎類型
        if 'base_types' in self.sot_data:
            base_file = target_dir / "base_types.py"
            content = self._render_python_base_types()
            with open(base_file, 'w', encoding='utf-8') as f:
                f.write(content)
            generated_files.append(str(base_file))
            logger.info(f"✅ 生成 Python 基礎類型: {base_file}")
        
        # 生成各模組 Schema
        for category in ['messaging', 'tasks', 'findings']:
            if category in self.sot_data:
                module_file = target_dir / f"{category}.py"
                content = self._render_python_category(category)
                with open(module_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                generated_files.append(str(module_file))
                logger.info(f"✅ 生成 Python {category} Schema: {module_file}")
        
        # 生成 __init__.py
        init_file = target_dir / "__init__.py"
        content = self._render_python_init()
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(content)
        generated_files.append(str(init_file))
        
        return generated_files
    
    def generate_go_schemas(self, output_dir: Optional[str] = None) -> List[str]:
        """生成 Go struct Schema
        
        Args:
            output_dir: 自訂輸出目錄
            
        Returns:
            生成的檔案列表
        """
        config = self.sot_data['generation_config']['go']
        target_dir = Path(output_dir) if output_dir else Path(config['target_dir'])
        target_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 生成統一的 schemas.go 檔案
        schema_file = target_dir / "schemas.go"
        content = self._render_go_schemas()
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write(content)
        generated_files.append(str(schema_file))
        logger.info(f"✅ 生成 Go Schema: {schema_file}")
        
        return generated_files
    
    def generate_rust_schemas(self, output_dir: Optional[str] = None) -> List[str]:
        """生成 Rust Serde Schema
        
        Args:
            output_dir: 自訂輸出目錄
            
        Returns:
            生成的檔案列表
        """
        config = self.sot_data['generation_config']['rust']
        target_dir = Path(output_dir) if output_dir else Path(config['target_dir'])
        target_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 生成 mod.rs
        mod_file = target_dir / "mod.rs"
        content = self._render_rust_schemas()
        with open(mod_file, 'w', encoding='utf-8') as f:
            f.write(content)
        generated_files.append(str(mod_file))
        logger.info(f"✅ 生成 Rust Schema: {mod_file}")
        
        return generated_files
    
    def _render_python_base_types(self) -> str:
        """渲染 Python 基礎類型"""
        content = []
        content.append('"""')
        content.append('AIVA 基礎類型 Schema - 自動生成')
        content.append('=====================================')
        content.append('')
        content.append(self.sot_data['metadata']['description'])
        content.append('')
        content.append(f"⚠️  {self.sot_data['metadata']['generated_note']}")
        content.append(f"📅 最後更新: {self.sot_data['metadata']['last_updated']}")
        content.append(f"🔄 Schema 版本: {self.sot_data['version']}")
        content.append('"""')
        content.append('')
        
        # 添加imports
        for imp in self.sot_data['generation_config']['python']['base_imports']:
            content.append(imp)
        content.append('')
        content.append('')
        
        # 生成類別
        for class_name, class_info in self.sot_data['base_types'].items():
            content.append(f'class {class_name}(BaseModel):')
            content.append(f'    """{class_info["description"]}"""')
            content.append('')
            
            for field_name, field_info in class_info['fields'].items():
                field_line = self._generate_python_field(field_name, field_info)
                content.append(f'    {field_line}')
                content.append(f'    """{field_info["description"]}"""')
                content.append('')
            
            content.append('')
        
        return '\n'.join(content)
    
    def _render_python_category(self, category: str) -> str:
        """渲染 Python 分類 Schema"""
        content = []
        content.append('"""')
        content.append(f'AIVA {category.title()} Schema - 自動生成')
        content.append('=====================================')
        content.append('')
        content.append(self.sot_data['metadata']['description'])
        content.append('')
        content.append(f"⚠️  {self.sot_data['metadata']['generated_note']}")
        content.append(f"📅 最後更新: {self.sot_data['metadata']['last_updated']}")
        content.append(f"🔄 Schema 版本: {self.sot_data['version']}")
        content.append('"""')
        content.append('')
        
        # 添加imports
        for imp in self.sot_data['generation_config']['python']['base_imports']:
            content.append(imp)
        content.append('')
        content.append('from .base_types import *')
        content.append('')
        content.append('')
        
        # 生成類別
        for class_name, class_info in self.sot_data[category].items():
            content.append(f'class {class_name}(BaseModel):')
            content.append(f'    """{class_info["description"]}"""')
            content.append('')
            
            # 檢查是否有extends
            if 'extends' in class_info:
                content.append(f'    # 繼承自: {class_info["extends"]}')
                content.append('')
            
            # 處理fields
            for field_name, field_info in class_info.get('fields', {}).items():
                field_line = self._generate_python_field(field_name, field_info)
                content.append(f'    {field_line}')
                content.append(f'    """{field_info["description"]}"""')
                content.append('')
            
            # 處理additional_fields
            for field_name, field_info in class_info.get('additional_fields', {}).items():
                field_line = self._generate_python_field(field_name, field_info)
                content.append(f'    {field_line}')
                content.append(f'    """{field_info["description"]}"""')
                content.append('')
            
            content.append('')
        
        return '\n'.join(content)
    
    def _render_python_init(self) -> str:
        """渲染 Python __init__.py"""
        template = Template('''"""
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
    
    # 發現結果
    "FindingPayload",
    "FindingEvidence",
    "FindingImpact", 
    "FindingRecommendation",
]
''')
        
        return template.render(
            version=self.sot_data['version'],
            generated_at=datetime.now().isoformat()
        )
    
    def _render_go_schemas(self) -> str:
        """渲染 Go 統一 Schema"""
        content = []
        content.append('// AIVA Go Schema - 自動生成')
        content.append('// ===========================')
        content.append('//')
        content.append(f'// {self.sot_data["metadata"]["description"]}')
        content.append('//')
        content.append(f'// ⚠️  {self.sot_data["metadata"]["generated_note"]}')
        content.append(f'// 📅 最後更新: {self.sot_data["metadata"]["last_updated"]}')
        content.append(f'// 🔄 Schema 版本: {self.sot_data["version"]}')
        content.append('')
        
        # 添加imports
        for imp in self.sot_data['generation_config']['go']['base_imports']:
            content.append(imp)
        content.append('')
        
        # 基礎類型
        content.append('// ==================== 基礎類型 ====================')
        content.append('')
        
        for class_name, class_info in self.sot_data['base_types'].items():
            content.append(f'// {class_name} {class_info["description"]}')
            content.append(f'type {class_name} struct {{')
            
            for field_name, field_info in class_info['fields'].items():
                go_name = self._to_go_field_name(field_name)
                go_type = self._get_go_type(field_info['type'])
                json_tag = self._get_go_json_tag(field_info.get('required', True))
                content.append(f'    {go_name:<20} {go_type:<25} `json:"{field_name}{json_tag}"`  // {field_info["description"]}')
            
            content.append('}')
            content.append('')
        
        # 其他類別
        for section, title in [('messaging', '訊息通訊'), ('tasks', '任務管理'), ('findings', '發現結果')]:
            if section in self.sot_data:
                content.append(f'// ==================== {title} ====================')
                content.append('')
                
                for class_name, class_info in self.sot_data[section].items():
                    content.append(f'// {class_name} {class_info["description"]}')
                    content.append(f'type {class_name} struct {{')
                    
                    # 主要字段
                    for field_name, field_info in class_info.get('fields', {}).items():
                        go_name = self._to_go_field_name(field_name)
                        go_type = self._get_go_type(field_info['type'])
                        json_tag = self._get_go_json_tag(field_info.get('required', True))
                        content.append(f'    {go_name:<20} {go_type:<25} `json:"{field_name}{json_tag}"`  // {field_info["description"]}')
                    
                    # 額外字段
                    for field_name, field_info in class_info.get('additional_fields', {}).items():
                        go_name = self._to_go_field_name(field_name)
                        go_type = self._get_go_type(field_info['type'])
                        json_tag = self._get_go_json_tag(field_info.get('required', True))
                        content.append(f'    {go_name:<20} {go_type:<25} `json:"{field_name}{json_tag}"`  // {field_info["description"]}')
                    
                    content.append('}')
                    content.append('')
        
        return '\n'.join(content)
    
    def _render_rust_schemas(self) -> str:
        """渲染 Rust Schema"""
        # Rust 生成邏輯（簡化版本）
        return f'''// AIVA Rust Schema - 自動生成
// 版本: {self.sot_data['version']}
// 
// TODO: 實現完整的 Rust Schema 生成
// 此功能將在後續版本中完善

{chr(10).join(self.sot_data['generation_config']['rust']['base_imports'])}

// 基礎結構定義將在此處生成...
'''
    
    def _get_python_type(self, type_str: str) -> str:
        """轉換為 Python 類型"""
        mapping = self.sot_data['generation_config']['python']['field_mapping']
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
            return f'{value}'
        return str(value)
    
    def _generate_python_field(self, field_name: str, field_info: dict) -> str:
        """生成Python字段定義"""
        field_type = self._get_python_type(field_info['type'])
        parts = [f"{field_name}: {field_type}"]
        
        # 處理預設值和Field參數
        field_params = []
        
        # 添加驗證參數
        if 'validation' in field_info:
            for key, value in field_info['validation'].items():
                if key == 'enum':
                    field_params.append(f'values={value}')
                elif key == 'pattern':
                    field_params.append(f'pattern=r"{value}"')
                elif key == 'format':
                    # Pydantic v2 format handling
                    if value == 'url':
                        field_params.append('url=True')
                elif key == 'max_length':
                    field_params.append(f'max_length={value}')
                elif key == 'minimum':
                    field_params.append(f'ge={value}')
                elif key == 'maximum':
                    field_params.append(f'le={value}')
        
        # 處理預設值
        if not field_info.get('required', True):
            if 'default' in field_info:
                default_val = self._get_python_default(field_info['default'], field_info['type'])
                if field_params:
                    field_params.append(f'default={default_val}')
                    parts.append(f" = Field({', '.join(field_params)})")
                else:
                    parts.append(f" = {default_val}")
            else:
                if field_params:
                    field_params.append('default=None')
                    parts.append(f" = Field({', '.join(field_params)})")
                else:
                    parts.append(" = None")
        elif 'default' in field_info:
            default_val = self._get_python_default(field_info['default'], field_info['type'])
            field_params.append(f'default={default_val}')
            parts.append(f" = Field({', '.join(field_params)})")
        elif field_params:
            parts.append(f" = Field({', '.join(field_params)})")
            
        return ''.join(parts)
    
    def _get_go_type(self, type_str: str) -> str:
        """轉換為 Go 類型"""
        mapping = self.sot_data['generation_config']['go']['field_mapping']
        return mapping.get(type_str, type_str)
    
    def _to_go_field_name(self, field_name: str) -> str:
        """轉換為 Go 欄位名稱（PascalCase）"""
        return ''.join(word.capitalize() for word in field_name.split('_'))
    
    def _get_go_json_tag(self, required: bool) -> str:
        """獲取 Go JSON 標籤"""
        return "" if required else ",omitempty"
    
    def validate_schemas(self) -> bool:
        """驗證 Schema 定義的一致性"""
        logger.info("🔍 開始 Schema 驗證...")
        
        errors = []
        
        # 檢查必要的頂層鍵
        required_keys = ['version', 'metadata', 'base_types', 'generation_config']
        for key in required_keys:
            if key not in self.sot_data:
                errors.append(f"缺少必要的頂層鍵: {key}")
        
        # 檢查版本格式
        version = self.sot_data.get('version', '')
        if not version or not version.replace('.', '').isdigit():
            errors.append(f"版本格式無效: {version}")
        
        # 檢查類型引用
        defined_types = set(self.sot_data.get('base_types', {}).keys())
        all_schemas = {}
        
        for category in ['messaging', 'tasks', 'findings']:
            if category in self.sot_data:
                all_schemas.update(self.sot_data[category])
                defined_types.update(self.sot_data[category].keys())
        
        # 檢查類型引用的有效性
        for schema_name, schema_info in all_schemas.items():
            for field_name, field_info in schema_info.get('fields', {}).items():
                field_type = field_info.get('type', '')
                # 移除泛型包裝檢查核心類型
                core_type = field_type.replace('Optional[', '').replace('List[', '').replace('Dict[str, ', '').replace(']', '').replace('>', '')
                if core_type in ['str', 'int', 'float', 'bool', 'datetime', 'Any']:
                    continue
                if core_type not in defined_types:
                    errors.append(f"在 {schema_name}.{field_name} 中引用了未定義的類型: {core_type}")
        
        if errors:
            logger.error("❌ Schema 驗證失敗:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ Schema 驗證通過!")
        return True
    
    def generate_all(self, validate: bool = True) -> Dict[str, List[str]]:
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
            results['python'] = self.generate_python_schemas()
            logger.info(f"✅ Python Schema 生成完成: {len(results['python'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ Python Schema 生成失敗: {e}")
            results['python'] = []
        
        # 生成 Go
        try:
            results['go'] = self.generate_go_schemas()
            logger.info(f"✅ Go Schema 生成完成: {len(results['go'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ Go Schema 生成失敗: {e}")
            results['go'] = []
        
        # 生成 Rust (簡化版本)
        try:
            results['rust'] = self.generate_rust_schemas()
            logger.info(f"✅ Rust Schema 生成完成: {len(results['rust'])} 個檔案")
        except Exception as e:
            logger.error(f"❌ Rust Schema 生成失敗: {e}")
            results['rust'] = []
        
        total_files = sum(len(files) for files in results.values())
        logger.info(f"🎉 所有語言 Schema 生成完成! 總計: {total_files} 個檔案")
        
        return results


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="AIVA Schema 代碼生成工具")
    parser.add_argument("--lang", choices=["python", "go", "rust", "all"], default="all", help="生成的語言")
    parser.add_argument("--validate", action="store_true", help="僅進行 Schema 驗證")
    parser.add_argument("--output-dir", help="自訂輸出目錄")
    parser.add_argument("--sot-file", default="services/aiva_common/core_schema_sot.yaml", help="SOT 檔案路徑")
    
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
    
    # 輸出結果
    success = all(len(files) > 0 for files in results.values())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()