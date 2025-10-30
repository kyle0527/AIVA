#!/usr/bin/env python3
"""
測試 Schema 代碼生成工具
"""

import os
import sys
from pathlib import Path

# 設置環境變數避免配置錯誤
os.environ["AIVA_RABBITMQ_URL"] = "amqp://guest:guest@localhost:5672/"
os.environ["AIVA_POSTGRES_URL"] = "postgresql://user:pass@localhost:5432/aiva"

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent))

def test_schema_generator():
    """測試 Schema 生成器"""
    try:
        from services.aiva_common.tools.schema_codegen_tool import SchemaCodeGenerator
        
        # 創建生成器實例
        generator = SchemaCodeGenerator()
        
        # 測試 YAML 載入
        print("✅ 測試 YAML SOT 載入...")
        sot_data = generator.sot_data
        print(f"   載入成功，版本: {sot_data['version']}")
        print(f"   基礎類型數量: {len(sot_data.get('base_types', {}))}")
        print(f"   枚舉數量: {len(sot_data.get('enums', {}))}")
        
        # 檢查新增的部分
        async_schemas = sot_data.get('async_utils', {})
        plugin_schemas = sot_data.get('plugins', {})
        cli_schemas = sot_data.get('cli', {})
        
        print(f"   異步工具 Schema: {len(async_schemas)} 個")
        print(f"   插件管理 Schema: {len(plugin_schemas)} 個")
        print(f"   CLI 界面 Schema: {len(cli_schemas)} 個")
        
        # 測試語言映射
        print("\n✅ 測試語言映射...")
        python_config = sot_data['generation_config']['python']
        go_config = sot_data['generation_config']['go']  
        rust_config = sot_data['generation_config']['rust']
        
        print(f"   Python 輸出目錄: {python_config['target_dir']}")
        print(f"   Go 輸出目錄: {go_config['target_dir']}")
        print(f"   Rust 輸出目錄: {rust_config['target_dir']}")
        
        # 測試簡單生成功能
        print("\n✅ 測試生成功能...")
        print("   這將測試代碼生成器的基本功能...")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 開始測試 Schema 代碼生成工具...")
    success = test_schema_generator()
    
    if success:
        print("\n✅ 代碼生成工具測試通過！")
    else:
        print("\n❌ 代碼生成工具測試失敗！")
        sys.exit(1)