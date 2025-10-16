"""測試所有 schemas 模組導入"""

import sys
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 測試導入
try:
    from aiva_common.schemas import MessageHeader, FindingPayload
    print("✅ Base schemas working")
    
    from aiva_common.schemas.enhanced import EnhancedFindingPayload
    print("✅ Enhanced schemas working")
    
    from aiva_common.schemas.system import SessionState, TaskQueue
    print("✅ System schemas working")
    
    from aiva_common.schemas.references import CVEReference, CWEReference
    print("✅ References schemas working")
    
    print("\n🎉 所有新模組導入成功!")
    
except Exception as e:
    print(f"❌ 導入失敗: {e}")
    import traceback
    traceback.print_exc()
