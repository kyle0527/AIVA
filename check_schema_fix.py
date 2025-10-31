#!/usr/bin/env python3
"""
簡單的 Schema 合規性檢查
檢查 TypeScript 模組是否使用標準生成的 schemas
"""

import os
from pathlib import Path

def check_typescript_compliance():
    """檢查 TypeScript 模組合規性"""
    ts_module_path = Path("services/features/common/typescript/aiva_common_ts")
    
    print("🔍 檢查 TypeScript Schema 合規性...")
    
    # 1. 檢查是否存在標準生成的 schemas
    generated_schema_path = ts_module_path / "schemas" / "generated" / "schemas.ts"
    if generated_schema_path.exists():
        print("✅ 標準生成的 schemas.ts 存在")
    else:
        print("❌ 標準生成的 schemas.ts 不存在")
        return False
    
    # 2. 檢查是否移除了自定義 schemas
    custom_schema_path = ts_module_path / "schemas.ts"
    if not custom_schema_path.exists():
        print("✅ 自定義 schemas.ts 已移除")
    else:
        print("❌ 自定義 schemas.ts 仍然存在")
        return False
    
    # 3. 檢查 index.ts 是否導入標準 schemas
    index_path = ts_module_path / "index.ts"
    if index_path.exists():
        content = index_path.read_text(encoding='utf-8')
        if "from './schemas/generated'" in content:
            print("✅ index.ts 正確導入標準生成的 schemas")
        else:
            print("❌ index.ts 未導入標準生成的 schemas")
            return False
    
    # 4. 檢查生成的 schema 內容
    if generated_schema_path.exists():
        content = generated_schema_path.read_text(encoding='utf-8')
        if "自動生成" in content and "單一事實原則" in content:
            print("✅ 生成的 schema 包含正確的標頭")
        else:
            print("❌ 生成的 schema 標頭不正確")
            return False
    
    print("🎉 TypeScript Schema 合規性檢查通過！")
    return True

if __name__ == "__main__":
    success = check_typescript_compliance()
    exit(0 if success else 1)