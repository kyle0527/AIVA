#!/usr/bin/env python3
"""
AIVA 能力註冊中心 - 簡化測試版本
測試基本功能是否正常運行
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 加入 AIVA 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """測試基本導入"""
    print("🔍 測試基本導入...")
    
    try:
        from aiva_common.enums import ProgrammingLanguage, Severity, Confidence
        print("✅ aiva_common.enums 導入成功")
        
        from aiva_common.utils.logging import get_logger
        print("✅ aiva_common.utils.logging 導入成功")
        
        from aiva_common.utils.ids import new_id
        print("✅ aiva_common.utils.ids 導入成功")
        
        return True
    except Exception as e:
        print(f"❌ 導入失敗: {str(e)}")
        return False

def test_models():
    """測試資料模型"""
    print("\n🔧 測試資料模型...")
    
    try:
        from services.integration.capability.models import (
            CapabilityRecord, 
            CapabilityType, 
            CapabilityStatus,
            InputParameter,
            OutputParameter,
            create_capability_id
        )
        from aiva_common.enums import ProgrammingLanguage
        
        # 創建測試能力
        test_capability = CapabilityRecord(
            id=create_capability_id("test", "example", "function"),
            name="測試能力",
            description="這是一個測試能力",
            module="test_module",
            language=ProgrammingLanguage.PYTHON,
            entrypoint="test.module:main",
            capability_type=CapabilityType.UTILITY,
            inputs=[
                InputParameter(
                    name="input_param",
                    type="str", 
                    required=True,
                    description="測試參數"
                )
            ],
            outputs=[
                OutputParameter(
                    name="output_result",
                    type="str",
                    description="測試結果"
                )
            ],
            tags=["test", "example"]
        )
        
        print("✅ 能力模型創建成功")
        print(f"   ID: {test_capability.id}")
        print(f"   名稱: {test_capability.name}")
        print(f"   語言: {test_capability.language.value}")
        print(f"   類型: {test_capability.capability_type.value}")
        
        return test_capability
        
    except Exception as e:
        print(f"❌ 模型測試失敗: {str(e)}")
        return None

def test_config():
    """測試配置系統"""
    print("\n⚙️ 測試配置系統...")
    
    try:
        from services.integration.capability.config import (
            CapabilityRegistryConfig,
            load_config,
            validate_config
        )
        
        # 創建預設配置
        config = CapabilityRegistryConfig()
        print("✅ 預設配置創建成功")
        print(f"   系統名稱: {config.name}")
        print(f"   版本: {config.version}")
        print(f"   環境: {config.environment}")
        print(f"   API 端口: {config.api.port}")
        
        # 驗證配置
        errors = validate_config(config)
        if errors:
            print(f"⚠️ 配置驗證警告: {len(errors)} 個問題")
            for error in errors[:3]:  # 只顯示前3個
                print(f"   • {error}")
        else:
            print("✅ 配置驗證通過")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置測試失敗: {str(e)}")
        return None

def test_basic_functionality():
    """測試基本功能"""
    print("\n🧪 測試基本功能...")
    
    try:
        # 測試 ID 生成
        from aiva_common.utils.ids import new_id
        test_id = new_id("test")
        print(f"✅ ID 生成成功: {test_id}")
        
        # 測試日誌
        from aiva_common.utils.logging import get_logger
        logger = get_logger("test")
        logger.info("測試日誌訊息")
        print("✅ 日誌系統正常")
        
        # 測試枚舉
        from aiva_common.enums import ProgrammingLanguage
        supported_langs = [lang.value for lang in ProgrammingLanguage]
        print(f"✅ 支援的語言: {', '.join(supported_langs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能測試失敗: {str(e)}")
        return False

def show_system_info():
    """顯示系統資訊"""
    print("\n📊 系統資訊摘要")
    print("=" * 50)
    
    print(f"🕒 測試時間: {datetime.now().isoformat()}")
    print(f"🐍 Python 版本: {sys.version.split()[0]}")
    print(f"📁 工作目錄: {Path.cwd()}")
    
    # 檢查重要目錄
    aiva_common_path = Path("services/aiva_common")
    if aiva_common_path.exists():
        print(f"✅ aiva_common 路徑存在: {aiva_common_path}")
    else:
        print(f"❌ aiva_common 路徑不存在: {aiva_common_path}")
    
    capability_path = Path("services/integration/capability")
    if capability_path.exists():
        print(f"✅ 能力註冊中心路徑存在: {capability_path}")
        files = list(capability_path.glob("*.py"))
        print(f"   包含 {len(files)} 個 Python 檔案")
    else:
        print(f"❌ 能力註冊中心路徑不存在: {capability_path}")

def main():
    """主測試函數"""
    print("🚀 AIVA 能力註冊中心 - 基本功能測試")
    print("=" * 60)
    
    # 顯示系統資訊
    show_system_info()
    
    # 運行測試
    test_results = {
        "imports": test_imports(),
        "models": test_models() is not None,
        "config": test_config() is not None,
        "basic_functionality": test_basic_functionality()
    }
    
    # 總結結果
    print(f"\n🎯 測試結果總結")
    print("=" * 30)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 總體結果: {passed}/{total} 測試通過")
    
    if passed == total:
        print("🎉 所有基本功能測試通過！系統準備就緒。")
        
        print(f"\n📚 下一步建議:")
        print("1. 運行完整的示例腳本測試更多功能")
        print("2. 啟動 API 服務器進行完整測試")
        print("3. 執行能力發現和註冊測試")
        
    else:
        print("⚠️ 部分測試失敗，需要進一步調試。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)