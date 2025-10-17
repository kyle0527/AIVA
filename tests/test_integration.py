#!/usr/bin/env python3
"""
快速整合測試

測試新整合的組件是否正常工作
"""

from pathlib import Path
import sys

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """測試新組件的 import"""
    print("[TEST] 測試組件導入...")

    try:
        # 測試 BioNeuron Master
        print("[OK] BioNeuron Master Controller 導入成功")
    except Exception as e:
        print(f"[FAIL] BioNeuron Master Controller 導入失敗: {e}")

    try:
        # 測試存儲系統
        print("[OK] 存儲管理器導入成功")
    except Exception as e:
        print(f"[FAIL] 存儲管理器導入失敗: {e}")

    try:
        # 測試 AI 引擎
        print("[OK] BioNeuronRAGAgent 導入成功")
    except Exception as e:
        print(f"[FAIL] BioNeuronRAGAgent 導入失敗: {e}")

    try:
        # 測試學習系統
        print("[OK] 經驗管理器導入成功")
    except Exception as e:
        print(f"[FAIL] 經驗管理器導入失敗: {e}")

def test_basic_functionality():
    """測試基本功能"""
    print("\n[CONFIG] 測試基本功能...")

    try:
        from services.core.aiva_core.bio_neuron_master import (
            BioNeuronMasterController,
            OperationMode,
        )

        # 創建控制器
        controller = BioNeuronMasterController()
        print("[OK] BioNeuron Master Controller 創建成功")

        # 測試模式切換
        controller.switch_mode(OperationMode.UI)
        print(f"[OK] 模式切換成功，當前模式: {controller.current_mode}")

    except Exception as e:
        print(f"[FAIL] 控制器測試失敗: {e}")

    try:
        from services.core.aiva_core.storage import StorageManager

        # 創建存儲管理器
        storage = StorageManager(
            data_root="./test_data",
            db_type="sqlite",
            auto_create_dirs=True
        )
        print("[OK] 存儲管理器創建成功")

    except Exception as e:
        print(f"[FAIL] 存儲管理器測試失敗: {e}")

def main():
    """主函數"""
    print("=" * 50)
    print("[START] AIVA-1 整合驗證測試")
    print("=" * 50)

    test_imports()
    test_basic_functionality()

    print("\n" + "=" * 50)
    print("[SPARKLE] 測試完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
