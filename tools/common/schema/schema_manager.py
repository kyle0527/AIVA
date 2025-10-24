#!/usr/bin/env python3
"""
AIVA Schema 管理助手工具
功能：新增、修改、驗證、生成多語言 Schema 定義
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import importlib.util

class SchemaManager:
    """Schema 管理核心類別"""
    
    def __init__(self, aiva_root: Path):
        self.aiva_root = aiva_root
        self.schemas_dir = aiva_root / "services" / "aiva_common" / "schemas"
        self.enums_dir = aiva_root / "services" / "aiva_common" / "enums"
        self.output_dir = aiva_root / "schemas"
        
        # 確保目錄存在
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.enums_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_new_schema(self, name: str, category: str, fields: Dict[str, str], 
                         description: Optional[str] = None) -> bool:
        """創建新的 Schema 定義"""
        try:
            # 生成 Schema 類別程式碼
            schema_code = self._generate_schema_template(name, fields, description)
            
            # 目標檔案
            target_file = self.schemas_dir / f"{category}.py"
            
            if target_file.exists():
                # 插入到現有檔案
                self._insert_into_existing_file(target_file, schema_code, name)
            else:
                # 創建新檔案
                self._create_new_schema_file(target_file, schema_code, name, category)
            
            # 更新 __init__.py
            self._update_schema_init(category, name)
            
            print(f"✅ 成功創建 Schema: {name} 在 {category}.py")
            return True
            
        except Exception as e:
            print(f"❌ 創建 Schema 失敗: {e}")
            return False
    
    def create_new_enum(self, name: str, category: str, values: Dict[str, str],
                       description: Optional[str] = None) -> bool:
        """創建新的 Enum 定義"""
        try:
            # 生成 Enum 類別程式碼
            enum_code = self._generate_enum_template(name, values, description)
            
            # 目標檔案
            target_file = self.enums_dir / f"{category}.py"
            
            if target_file.exists():
                # 插入到現有檔案
                self._insert_into_existing_file(target_file, enum_code, name)
            else:
                # 創建新檔案
                self._create_new_enum_file(target_file, enum_code, name, category)
            
            # 更新 __init__.py
            self._update_enum_init(category, name)
            
            print(f"✅ 成功創建 Enum: {name} 在 {category}.py")
            return True
            
        except Exception as e:
            print(f"❌ 創建 Enum 失敗: {e}")
            return False
    
    def validate_all_schemas(self) -> bool:
        """驗證所有 Schema 定義"""
        print("🔍 驗證 Schema 定義...")
        
        try:
            # 檢查 Python 語法
            self._validate_python_syntax()
            
            # 檢查匯入是否正常
            self._validate_imports()
            
            # 檢查 __all__ 列表
            self._validate_exports()
            
            print("✅ 所有 Schema 定義驗證通過")
            return True
            
        except Exception as e:
            print(f"❌ Schema 驗證失敗: {e}")
            return False
    
    def generate_multilang_schemas(self, languages: Optional[List[str]] = None) -> bool:
        """生成多語言 Schema 檔案"""
        print("🔧 生成多語言 Schema 檔案...")
        
        try:
            script_path = self.aiva_root / "tools" / "generate-official-contracts.ps1"
            
            if not script_path.exists():
                print(f"❌ 生成腳本不存在: {script_path}")
                return False
            
            # 根據指定語言決定參數
            if languages:
                args = []
                if "json" in languages:
                    args.append("-GenerateJsonSchema")
                if "typescript" in languages:
                    args.append("-GenerateTypeScript")
                if "go" in languages:
                    args.append("-GenerateGo")
                if "rust" in languages:
                    args.append("-GenerateRust")
            else:
                args = ["-GenerateAll"]
            
            # 執行生成腳本
            cmd = ["pwsh", "-File", str(script_path)] + args
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.aiva_root)
            
            if result.returncode == 0:
                print("✅ 多語言 Schema 生成完成")
                self._show_generated_files()
                return True
            else:
                print(f"❌ 生成失敗:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 生成過程出錯: {e}")
            return False
    
    def list_schemas(self) -> Dict[str, List[str]]:
        """列出所有 Schema 定義"""
        schemas = {}
        
        for schema_file in self.schemas_dir.glob("*.py"):
            if schema_file.name.startswith("__"):
                continue
                
            category = schema_file.stem
            class_names = self._extract_class_names(schema_file)
            schemas[category] = class_names
        
        return schemas
    
    def list_enums(self) -> Dict[str, List[str]]:
        """列出所有 Enum 定義"""
        enums = {}
        
        for enum_file in self.enums_dir.glob("*.py"):
            if enum_file.name.startswith("__"):
                continue
                
            category = enum_file.stem
            class_names = self._extract_class_names(enum_file)
            enums[category] = class_names
        
        return enums
    
    def analyze_schema_usage(self, schema_name: str) -> Dict[str, List[str]]:
        """分析 Schema 在專案中的使用情況"""
        usage = {"files": [], "imports": []}
        
        # 搜尋整個專案
        for py_file in self.aiva_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding="utf-8")
                if schema_name in content:
                    usage["files"].append(str(py_file.relative_to(self.aiva_root)))
                    
                    # 檢查是否為匯入語句
                    for line in content.splitlines():
                        if "import" in line and schema_name in line:
                            usage["imports"].append(line.strip())
                            
            except Exception:
                continue
        
        return usage
    
    def _generate_schema_template(self, name: str, fields: Dict[str, str], 
                                description: Optional[str]) -> str:
        """生成 Schema 類別範本"""
        imports = set(["BaseModel", "Field"])
        
        # 分析欄位型別需要的匯入
        for field_type in fields.values():
            if "Optional" in field_type:
                imports.add("Optional")
            if "List" in field_type:
                imports.add("List")
            if "Dict" in field_type:
                imports.add("Dict") 
            if "datetime" in field_type:
                imports.add("datetime")
        
        # 生成匯入語句
        import_lines = []
        if any(imp in ["Optional", "List", "Dict"] for imp in imports):
            typing_imports = [imp for imp in imports if imp in ["Optional", "List", "Dict"]]
            import_lines.append(f"from typing import {', '.join(typing_imports)}")
        
        if "BaseModel" in imports or "Field" in imports:
            pydantic_imports = [imp for imp in imports if imp in ["BaseModel", "Field"]]
            import_lines.append(f"from pydantic import {', '.join(pydantic_imports)}")
        
        if "datetime" in imports:
            import_lines.append("from datetime import datetime")
        
        # 生成欄位定義
        field_lines = []
        for field_name, field_type in fields.items():
            if "Optional" in field_type:
                field_lines.append(f'    {field_name}: {field_type} = Field(None, description="{field_name} 描述")')
            else:
                field_lines.append(f'    {field_name}: {field_type} = Field(..., description="{field_name} 描述")')
        
        # 組合完整類別
        desc = description or f"{name} 資料模型"
        template = f'''

class {name}(BaseModel):
    """{desc}"""
    
{chr(10).join(field_lines)}

    class Config:
        json_schema_extra = {{
            "example": {{
                # TODO: 添加範例數據
            }}
        }}
'''
        
        return template
    
    def _generate_enum_template(self, name: str, values: Dict[str, str],
                              description: Optional[str]) -> str:
        """生成 Enum 類別範本"""
        desc = description or f"{name} 枚舉"
        
        # 生成枚舉值
        value_lines = []
        for key, value in values.items():
            value_lines.append(f'    {key.upper()} = "{value}"')
        
        template = f'''

class {name}(str, Enum):
    """{desc}"""
    
{chr(10).join(value_lines)}
'''
        
        return template
    
    def _insert_into_existing_file(self, file_path: Path, code: str, class_name: str):
        """插入程式碼到現有檔案"""
        content = file_path.read_text(encoding="utf-8")
        
        # 在檔案末尾添加
        new_content = content + code
        
        file_path.write_text(new_content, encoding="utf-8")
    
    def _create_new_schema_file(self, file_path: Path, schema_code: str, 
                               class_name: str, category: str):
        """創建新的 Schema 檔案"""
        header = f'''"""
{category.title()} Schema 定義
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ..enums import *
'''
        
        content = header + schema_code
        file_path.write_text(content, encoding="utf-8")
    
    def _create_new_enum_file(self, file_path: Path, enum_code: str,
                             class_name: str, category: str):
        """創建新的 Enum 檔案"""
        header = f'''"""
{category.title()} Enum 定義
"""

from __future__ import annotations
from enum import Enum
'''
        
        content = header + enum_code
        file_path.write_text(content, encoding="utf-8")
    
    def _update_schema_init(self, category: str, class_name: str):
        """更新 schemas/__init__.py"""
        init_file = self.schemas_dir / "__init__.py"
        if not init_file.exists():
            return
        
        content = init_file.read_text(encoding="utf-8")
        
        # 檢查是否已經匯入
        if f"from .{category} import" in content:
            # 更新現有匯入
            import_pattern = f"from .{category} import ("
            if import_pattern in content:
                # 在現有匯入中添加
                content = content.replace(
                    f"from .{category} import (",
                    f"from .{category} import (\n    {class_name},"
                )
        else:
            # 添加新匯入
            content += f"\nfrom .{category} import {class_name}\n"
        
        init_file.write_text(content, encoding="utf-8")
    
    def _update_enum_init(self, category: str, class_name: str):
        """更新 enums/__init__.py"""
        init_file = self.enums_dir / "__init__.py"
        if not init_file.exists():
            return
        
        content = init_file.read_text(encoding="utf-8")
        
        # 類似 schema init 的更新邏輯
        if f"from .{category} import" in content:
            import_pattern = f"from .{category} import ("
            if import_pattern in content:
                content = content.replace(
                    f"from .{category} import (",
                    f"from .{category} import (\n    {class_name},"
                )
        else:
            content += f"\nfrom .{category} import {class_name}\n"
        
        init_file.write_text(content, encoding="utf-8")
    
    def _validate_python_syntax(self):
        """驗證 Python 語法"""
        for py_file in self.schemas_dir.rglob("*.py"):
            try:
                compile(py_file.read_text(encoding="utf-8"), str(py_file), 'exec')
            except SyntaxError as e:
                raise Exception(f"語法錯誤在 {py_file}: {e}")
        
        for py_file in self.enums_dir.rglob("*.py"):
            try:
                compile(py_file.read_text(encoding="utf-8"), str(py_file), 'exec')
            except SyntaxError as e:
                raise Exception(f"語法錯誤在 {py_file}: {e}")
    
    def _validate_imports(self):
        """驗證匯入是否正常"""
        try:
            # 添加 services 目錄到 Python 路徑
            import sys
            services_path = str(self.aiva_root / "services")
            if services_path not in sys.path:
                sys.path.insert(0, services_path)
            
            # 測試匯入
            import aiva_common
            from aiva_common import schemas, enums
            
        except Exception as e:
            raise Exception(f"匯入驗證失敗: {e}")
    
    def _validate_exports(self):
        """驗證 __all__ 列表"""
        for init_file in [self.schemas_dir / "__init__.py", self.enums_dir / "__init__.py"]:
            if init_file.exists():
                content = init_file.read_text(encoding="utf-8")
                if "__all__" not in content:
                    print(f"⚠️  {init_file.relative_to(self.aiva_root)} 缺少 __all__ 列表")
    
    def _show_generated_files(self):
        """顯示生成的檔案資訊"""
        print("\n📊 生成檔案統計:")
        for file in self.output_dir.glob("*"):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"   {file.name}: {size_kb:.1f} KB")
    
    def _extract_class_names(self, file_path: Path) -> List[str]:
        """從檔案中提取類別名稱"""
        try:
            content = file_path.read_text(encoding="utf-8")
            import re
            
            # 匹配 class 定義
            class_pattern = r'^class\s+(\w+)'
            matches = re.findall(class_pattern, content, re.MULTILINE)
            return matches
            
        except Exception:
            return []


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AIVA Schema 管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 創建新 Schema
  python schema_manager.py create-schema --name UserProfile --category base \
    --fields '{"user_id": "str", "name": "str", "email": "Optional[str]"}'
  
  # 創建新 Enum  
  python schema_manager.py create-enum --name UserRole --category common \
    --values '{"admin": "administrator", "user": "regular_user"}'
  
  # 驗證所有定義
  python schema_manager.py validate
  
  # 生成多語言檔案
  python schema_manager.py generate --languages json typescript
  
  # 列出所有定義
  python schema_manager.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest="action", help="可用操作")
    
    # 創建 Schema
    create_schema_parser = subparsers.add_parser("create-schema", help="創建新 Schema")
    create_schema_parser.add_argument("--name", required=True, help="Schema 名稱")
    create_schema_parser.add_argument("--category", required=True, help="Schema 分類")
    create_schema_parser.add_argument("--fields", required=True, help="欄位定義 (JSON 格式)")
    create_schema_parser.add_argument("--description", help="Schema 描述")
    
    # 創建 Enum
    create_enum_parser = subparsers.add_parser("create-enum", help="創建新 Enum")
    create_enum_parser.add_argument("--name", required=True, help="Enum 名稱")
    create_enum_parser.add_argument("--category", required=True, help="Enum 分類")
    create_enum_parser.add_argument("--values", required=True, help="枚舉值 (JSON 格式)")
    create_enum_parser.add_argument("--description", help="Enum 描述")
    
    # 驗證
    subparsers.add_parser("validate", help="驗證所有 Schema 定義")
    
    # 生成
    generate_parser = subparsers.add_parser("generate", help="生成多語言 Schema")
    generate_parser.add_argument("--languages", nargs="*", 
                                choices=["json", "typescript", "go", "rust"],
                                help="指定要生成的語言")
    
    # 列出定義
    subparsers.add_parser("list", help="列出所有 Schema 和 Enum")
    
    # 使用情況分析
    usage_parser = subparsers.add_parser("usage", help="分析 Schema 使用情況")
    usage_parser.add_argument("--name", required=True, help="Schema 名稱")
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    # 初始化管理器
    aiva_root = Path(__file__).parent.parent.parent.parent
    manager = SchemaManager(aiva_root)
    
    # 執行對應操作
    success = True
    
    if args.action == "create-schema":
        try:
            fields = json.loads(args.fields)
            success = manager.create_new_schema(
                args.name, args.category, fields, args.description
            )
        except json.JSONDecodeError:
            print("❌ 欄位定義必須是有效的 JSON 格式")
            success = False
    
    elif args.action == "create-enum":
        try:
            values = json.loads(args.values)
            success = manager.create_new_enum(
                args.name, args.category, values, args.description
            )
        except json.JSONDecodeError:
            print("❌ 枚舉值必須是有效的 JSON 格式")
            success = False
    
    elif args.action == "validate":
        success = manager.validate_all_schemas()
    
    elif args.action == "generate":
        success = manager.generate_multilang_schemas(args.languages)
    
    elif args.action == "list":
        schemas = manager.list_schemas()
        enums = manager.list_enums()
        
        print("📋 Schema 定義:")
        for category, classes in schemas.items():
            print(f"  {category}: {', '.join(classes)}")
        
        print("\n📋 Enum 定義:")
        for category, classes in enums.items():
            print(f"  {category}: {', '.join(classes)}")
    
    elif args.action == "usage":
        usage = manager.analyze_schema_usage(args.name)
        print(f"📊 {args.name} 使用情況:")
        print(f"  檔案: {len(usage['files'])} 個")
        for file in usage['files']:
            print(f"    - {file}")
        if usage['imports']:
            print(f"  匯入語句:")
            for imp in usage['imports']:
                print(f"    - {imp}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()