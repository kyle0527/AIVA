#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TypeScript Interface Generator
從 JSON Schema 生成 TypeScript 介面定義
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class TypeScriptGenerator:
    """TypeScript 介面生成器"""
    
    def __init__(self):
        self.type_mapping = {
            "string": "string",
            "number": "number", 
            "integer": "number",
            "boolean": "boolean",
            "array": "[]",
            "object": "{}",
            "null": "null"
        }
    
    def json_type_to_ts(self, json_type: Any) -> str:
        """將 JSON Schema 類型轉換為 TypeScript 類型"""
        if isinstance(json_type, list):
            # Union types
            return " | ".join(self.json_type_to_ts(t) for t in json_type)
        elif isinstance(json_type, str):
            return self.type_mapping.get(json_type, json_type)
        else:
            return "any"
    
    def generate_interface(self, name: str, schema: Dict[str, Any]) -> str:
        """生成單個介面定義"""
        lines = [f"export interface {name} {{"]
        
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        for prop_name, prop_schema in properties.items():
            prop_type = self.get_property_type(prop_schema)
            optional = "" if prop_name in required else "?"
            
            # 處理描述
            description = prop_schema.get("description", "")
            if description:
                lines.append(f"  /** {description} */")
            
            lines.append(f"  {prop_name}{optional}: {prop_type};")
        
        lines.append("}")
        return "\n".join(lines)
    
    def get_property_type(self, prop_schema: Dict[str, Any]) -> str:
        """獲取屬性的 TypeScript 類型"""
        # 處理枚舉
        if "enum" in prop_schema:
            enum_values = prop_schema["enum"]
            return " | ".join(f'"{value}"' for value in enum_values)
        
        # 處理 anyOf/oneOf
        if "anyOf" in prop_schema:
            types = []
            for variant in prop_schema["anyOf"]:
                types.append(self.get_property_type(variant))
            return " | ".join(types)
        
        # 處理數組
        if prop_schema.get("type") == "array":
            items = prop_schema.get("items", {})
            item_type = self.get_property_type(items)
            return f"{item_type}[]"
        
        # 處理對象引用
        if "$ref" in prop_schema:
            ref = prop_schema["$ref"]
            if ref.startswith("#/$defs/"):
                return ref.replace("#/$defs/", "")
            return "any"
        
        # 基本類型
        json_type = prop_schema.get("type", "any")
        if isinstance(json_type, list):
            return " | ".join(self.type_mapping.get(t, t) for t in json_type)
        
        return self.type_mapping.get(json_type, "any")
    
    def generate_from_json_schema(self, json_schema_path: Path, output_path: Path):
        """從 JSON Schema 檔案生成 TypeScript 定義"""
        print(f"Generating TypeScript interfaces from {json_schema_path}")
        
        with open(json_schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # 生成標頭
        lines = [
            "// AUTO-GENERATED from JSON Schema by AIVA Official Tools",
            f"// Generated at: {datetime.now().isoformat()}",
            "// Do not edit manually - changes will be overwritten",
            "",
        ]
        
        # 生成介面
        defs = schema_data.get("$defs", {})
        for name, definition in sorted(defs.items()):
            if definition.get("type") == "object" or "properties" in definition:
                interface_code = self.generate_interface(name, definition)
                lines.append(interface_code)
                lines.append("")
        
        # 寫入檔案
        output_path.write_text("\n".join(lines), encoding='utf-8')
        
        file_size = output_path.stat().st_size
        print(f"TypeScript interfaces generated: {output_path}")
        print(f"File size: {file_size // 1024:.1f} KB")
        print(f"Interface count: {len([d for d in defs.values() if d.get('type') == 'object' or 'properties' in d])}")


def main():
    """主函數"""
    project_root = Path.cwd()
    json_schema_path = project_root / "schemas" / "aiva_schemas.json"
    output_path = project_root / "schemas" / "aiva_schemas.d.ts"
    
    if not json_schema_path.exists():
        print(f"JSON Schema file not found: {json_schema_path}")
        print("Please run generate_official_schemas.py first")
        return
    
    generator = TypeScriptGenerator()
    generator.generate_from_json_schema(json_schema_path, output_path)


if __name__ == "__main__":
    main()