"""
測試 AIVA 模組導入 - 用於驗證環境設定
"""
from __future__ import annotations

import sys

print("=" * 60)
print("AIVA 環境診斷測試")
print("=" * 60)

print(f"\n當前工作目錄: {sys.path[0]}")
print(f"\nPython 路徑:")
for i, path in enumerate(sys.path[:5], 1):
    print(f"  {i}. {path}")

print("\n" + "=" * 60)
print("測試模組導入...")
print("=" * 60)

try:
    from services.aiva_common.schemas import (
        FindingPayload,
        MessageHeader,
        AivaMessage,
    )
    print("✅ services.aiva_common.schemas - 導入成功")
except ImportError as e:
    print(f"❌ services.aiva_common.schemas - 導入失敗: {e}")
    sys.exit(1)

try:
    from services.aiva_common.enums import (
        ModuleName,
        Topic,
        Severity,
        Confidence,
    )
    print("✅ services.aiva_common.enums - 導入成功")
except ImportError as e:
    print(f"❌ services.aiva_common.enums - 導入失敗: {e}")
    sys.exit(1)

try:
    from services.aiva_common.utils import get_logger
    print("✅ services.aiva_common.utils - 導入成功")
except ImportError as e:
    print(f"❌ services.aiva_common.utils - 導入失敗: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✨ 所有模組導入測試通過！")
print("=" * 60)

# 測試實例化
print("\n測試 Pydantic 模型實例化...")
try:
    header = MessageHeader(
        message_id="test-001",
        trace_id="trace-001",
        source_module=ModuleName.CORE,
    )
    print(f"✅ MessageHeader 實例化成功: {header.message_id}")
    
    message = AivaMessage(
        header=header,
        topic=Topic.TASK_SCAN_START,
        payload={"test": "data"},
    )
    print(f"✅ AivaMessage 實例化成功: {message.topic}")
    
    print("\n🎉 所有測試完成！環境設定正確！")
except Exception as e:
    print(f"❌ 模型實例化失敗: {e}")
    sys.exit(1)
