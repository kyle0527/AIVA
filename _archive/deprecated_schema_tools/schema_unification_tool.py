#!/usr/bin/env python3
"""
AIVA Schema 統一整合工具
實現單一事實原則 (Single Source of Truth)

此工具將手動維護的 Schema 作為權威來源，
並自動更新 YAML 配置和生成代碼以保持一致性。
"""

import json
import yaml
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Type, get_type_hints
from pydantic import BaseModel
import sys
import importlib

class SchemaUnificationTool:
    """Schema 統一整合工具"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.schemas_dir = self.project_root / "services" / "aiva_common" / "schemas"
        self.yaml_config_path = self.project_root / "services" / "aiva_common" / "core_schema_sot.yaml"
        
        # 確保 Python 路徑正確
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
    
    def analyze_manual_schemas(self) -> Dict[str, Dict[str, Any]]:
        """分析手動維護的 Schema 定義"""
        print("🔍 分析手動維護的 Schema...")
        
        try:
            # 導入手動 Schema 模組
            base_module = importlib.import_module("services.aiva_common.schemas.base")
            findings_module = importlib.import_module("services.aiva_common.schemas.findings")
            enums_module = importlib.import_module("services.aiva_common.enums")
            
            manual_schemas = {}
            
            # 分析 base.py 中的 Schema
            for name in dir(base_module):
                obj = getattr(base_module, name)
                if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj != BaseModel:
                    schema_info = self.extract_schema_info(obj, "base")
                    manual_schemas[name] = schema_info
                    print(f"   📋 發現 Schema: {name}")
            
            # 分析 findings.py 中的 Schema  
            for name in dir(findings_module):
                obj = getattr(findings_module, name)
                if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj != BaseModel:
                    schema_info = self.extract_schema_info(obj, "findings")
                    manual_schemas[name] = schema_info
                    print(f"   📋 發現 Schema: {name}")
            
            print(f"✅ 共發現 {len(manual_schemas)} 個手動 Schema")
            return manual_schemas
            
        except Exception as e:
            print(f"❌ 手動 Schema 分析失敗: {e}")
            return {}
    
    def extract_schema_info(self, schema_class: Type[BaseModel], module: str) -> Dict[str, Any]:
        """提取 Schema 的詳細資訊"""
        
        try:
            # 獲取 Pydantic 模型的 JSON Schema
            json_schema = schema_class.model_json_schema()
            
            # 獲取欄位註釋和類型提示
            type_hints = get_type_hints(schema_class)
            
            fields = {}
            properties = json_schema.get("properties", {})
            required_fields = json_schema.get("required", [])
            
            for field_name, field_info in properties.items():
                field_type = type_hints.get(field_name, "Any")
                field_config = {
                    "type": self.normalize_type(field_type),
                    "required": field_name in required_fields,
                    "description": field_info.get("description", ""),
                    "default": field_info.get("default"),
                    "validation": {}
                }
                
                # 提取驗證規則
                if "pattern" in field_info:
                    field_config["validation"]["pattern"] = field_info["pattern"]
                if "maxLength" in field_info:
                    field_config["validation"]["max_length"] = field_info["maxLength"]
                if "enum" in field_info:
                    field_config["validation"]["enum"] = field_info["enum"]
                
                fields[field_name] = field_config
            
            return {
                "module": module,
                "description": json_schema.get("description", schema_class.__doc__ or ""),
                "fields": fields,
                "json_schema": json_schema
            }
            
        except Exception as e:
            print(f"⚠️ Schema {schema_class.__name__} 資訊提取失敗: {e}")
            return {
                "module": module,
                "description": schema_class.__doc__ or "",
                "fields": {},
                "error": str(e)
            }
    
    def normalize_type(self, type_hint) -> str:
        """標準化類型表示"""
        
        type_str = str(type_hint)
        
        # 處理常見類型
        type_mapping = {
            "<class 'str'>": "str",
            "<class 'int'>": "int", 
            "<class 'float'>": "float",
            "<class 'bool'>": "bool",
            "<class 'datetime.datetime'>": "datetime",
            "typing.Union[str, NoneType]": "Optional[str]",
            "str | None": "Optional[str]",
            "typing.Optional[str]": "Optional[str]"
        }
        
        # 處理枚舉類型
        if "ModuleName" in type_str:
            return "ModuleName"
        
        # 查找映射
        for pattern, normalized in type_mapping.items():
            if pattern in type_str:
                return normalized
        
        # 處理複雜類型
        if "Dict" in type_str or "dict" in type_str:
            return "Dict[str, Any]"
        if "List" in type_str or "list" in type_str:
            return "List[str]"
        
        return "Any"
    
    def load_current_yaml_config(self) -> Dict[str, Any]:
        """載入當前的 YAML 配置"""
        print("📄 載入當前 YAML 配置...")
        
        try:
            with open(self.yaml_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ YAML 配置載入成功")
            return config
        except Exception as e:
            print(f"❌ YAML 配置載入失敗: {e}")
            return {}
    
    def compare_schemas(self, manual_schemas: Dict, yaml_config: Dict) -> Dict[str, Any]:
        """比較手動 Schema 和 YAML 配置的差異"""
        print("🔍 比較 Schema 差異...")
        
        differences = {
            "missing_in_yaml": [],
            "field_differences": {},
            "type_mismatches": {},
            "validation_differences": {}
        }
        
        yaml_base_types = yaml_config.get("base_types", {})
        
        for schema_name, manual_info in manual_schemas.items():
            if schema_name not in yaml_base_types:
                differences["missing_in_yaml"].append(schema_name)
                continue
            
            yaml_schema = yaml_base_types[schema_name]
            yaml_fields = yaml_schema.get("fields", {})
            manual_fields = manual_info["fields"]
            
            # 比較欄位
            schema_field_diffs = {}
            for field_name, manual_field in manual_fields.items():
                if field_name not in yaml_fields:
                    schema_field_diffs[field_name] = {"status": "missing_in_yaml"}
                    continue
                
                yaml_field = yaml_fields[field_name]
                field_diff = {}
                
                # 比較類型
                if manual_field["type"] != yaml_field.get("type"):
                    field_diff["type"] = {
                        "manual": manual_field["type"],
                        "yaml": yaml_field.get("type")
                    }
                
                # 比較必填性
                if manual_field["required"] != yaml_field.get("required", True):
                    field_diff["required"] = {
                        "manual": manual_field["required"],
                        "yaml": yaml_field.get("required", True)
                    }
                
                # 比較驗證規則
                manual_validation = manual_field.get("validation", {})
                yaml_validation = yaml_field.get("validation", {})
                
                if manual_validation != yaml_validation:
                    field_diff["validation"] = {
                        "manual": manual_validation,
                        "yaml": yaml_validation
                    }
                
                if field_diff:
                    schema_field_diffs[field_name] = field_diff
            
            if schema_field_diffs:
                differences["field_differences"][schema_name] = schema_field_diffs
        
        return differences
    
    def generate_updated_yaml_config(self, manual_schemas: Dict, current_config: Dict) -> Dict[str, Any]:
        """基於手動 Schema 生成更新的 YAML 配置"""
        print("🔄 生成更新的 YAML 配置...")
        
        # 複製當前配置結構
        updated_config = current_config.copy()
        
        # 更新 metadata
        updated_config["metadata"] = {
            "description": "AIVA跨語言Schema統一定義 - 以手動維護版本為準",
            "generated_note": "此配置已同步手動維護的Schema定義，確保單一事實原則",
            "last_updated": datetime.now().isoformat(),
            "sync_source": "手動維護Schema (base.py, findings.py)"
        }
        
        # 更新 base_types
        if "base_types" not in updated_config:
            updated_config["base_types"] = {}
        
        for schema_name, schema_info in manual_schemas.items():
            # 跳過 Task 類別，因為它太複雜且具有前向引用
            if schema_name == "Task":
                continue
                
            updated_config["base_types"][schema_name] = {
                "description": schema_info["description"],
                "fields": self.convert_fields_to_yaml_format(schema_info["fields"])
            }
        
        return updated_config
    
    def convert_fields_to_yaml_format(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """將欄位資訊轉換為 YAML 格式"""
        
        yaml_fields = {}
        
        for field_name, field_info in fields.items():
            yaml_field = {
                "type": field_info["type"],
                "required": field_info["required"],
                "description": field_info["description"]
            }
            
            # 添加預設值
            if field_info.get("default") is not None:
                if field_info["default"] == "datetime.now(UTC)":
                    yaml_field["default"] = "auto_generated"
                else:
                    yaml_field["default"] = field_info["default"]
            
            # 添加驗證規則（如果存在且不為空）
            validation = field_info.get("validation", {})
            if validation:
                yaml_field["validation"] = validation
            
            yaml_fields[field_name] = yaml_field
        
        return yaml_fields
    
    def save_updated_yaml_config(self, updated_config: Dict[str, Any]) -> bool:
        """保存更新的 YAML 配置"""
        print("💾 保存更新的 YAML 配置...")
        
        try:
            # 創建備份
            backup_path = self.yaml_config_path.with_suffix('.yaml.backup')
            if self.yaml_config_path.exists():
                import shutil
                shutil.copy2(self.yaml_config_path, backup_path)
                print(f"📦 已創建備份: {backup_path}")
            
            # 保存更新的配置
            with open(self.yaml_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(updated_config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False, indent=2)
            
            print("✅ YAML 配置更新成功")
            return True
            
        except Exception as e:
            print(f"❌ YAML 配置保存失敗: {e}")
            return False
    
    def run_unification(self) -> bool:
        """執行 Schema 統一整合"""
        print("🚀 開始 AIVA Schema 統一整合")
        print("=" * 60)
        
        # 1. 分析手動 Schema
        manual_schemas = self.analyze_manual_schemas()
        if not manual_schemas:
            print("❌ 無法分析手動 Schema，終止整合")
            return False
        
        # 2. 載入當前 YAML 配置
        current_config = self.load_current_yaml_config()
        
        # 3. 比較差異
        differences = self.compare_schemas(manual_schemas, current_config)
        
        # 4. 顯示差異報告
        self.print_differences_report(differences)
        
        # 5. 生成更新的 YAML 配置
        updated_config = self.generate_updated_yaml_config(manual_schemas, current_config)
        
        # 6. 保存更新的配置
        success = self.save_updated_yaml_config(updated_config)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 Schema 統一整合完成！")
            print("📋 下一步：執行 Schema 重新生成")
            print("   python tools/common/generate_official_schemas.py")
            print("=" * 60)
            return True
        else:
            return False
    
    def print_differences_report(self, differences: Dict[str, Any]):
        """打印差異報告"""
        print("\n📊 Schema 差異分析報告")
        print("-" * 40)
        
        if differences["missing_in_yaml"]:
            print(f"🔴 YAML 中缺失的 Schema ({len(differences['missing_in_yaml'])} 個):")
            for schema in differences["missing_in_yaml"]:
                print(f"   - {schema}")
        
        if differences["field_differences"]:
            print(f"\n🟡 欄位差異 ({len(differences['field_differences'])} 個 Schema):")
            for schema, field_diffs in differences["field_differences"].items():
                print(f"   📋 {schema}:")
                for field, diff in field_diffs.items():
                    print(f"      - {field}: {diff}")
        
        if not differences["missing_in_yaml"] and not differences["field_differences"]:
            print("✅ 未發現顯著差異")


def main():
    """主程式入口"""
    print("🏗️ AIVA Schema 統一整合工具")
    print("實現單一事實原則 (Single Source of Truth)")
    print()
    
    tool = SchemaUnificationTool()
    success = tool.run_unification()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()