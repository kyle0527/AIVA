"""
測試 Dialog Assistant 對話攻擊能力
依照 AIVA_CORE_使用者手冊.md 的測試流程
"""
import sys
import asyncio
sys.path.insert(0, 'C:/D/fold7/AIVA-git')

from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant

async def test_dialog_attack():
    """測試對話攻擊功能"""
    print("=" * 60)
    print("📋 AIVA Dialog Assistant 攻擊測試")
    print("=" * 60)
    print()
    
    # 初始化助理
    print("1️⃣ 初始化 Dialog Assistant...")
    assistant = AIVADialogAssistant()
    print("✅ 初始化成功")
    print()
    
    # 測試對話攻擊
    target = "http://localhost:3000"
    user_input = f"幫我掃描 {target}"
    
    print(f"2️⃣ 發送攻擊請求: '{user_input}'")
    print(f"   目標: {target}")
    print()
    
    result = await assistant.process_user_input(user_input)
    
    print("3️⃣ 執行結果:")
    print(f"   意圖類型: {result.get('intent')}")
    print(f"   可執行: {result.get('executable')}")
    print()
    
    print("4️⃣ 詳細訊息:")
    print("-" * 60)
    print(result.get('message'))
    print("-" * 60)
    print()
    
    # 總結
    if result.get('executable'):
        print("✅ 測試成功：系統可以執行對話式攻擊")
    else:
        print("⚠️ 測試失敗：系統無法執行攻擊")
    
    print()
    print("=" * 60)
    return result

if __name__ == "__main__":
    asyncio.run(test_dialog_attack())
