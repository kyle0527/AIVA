import sys
import asyncio

# 添加專案路徑
sys.path.insert(0, r"c:\D\fold7\AIVA-git")

from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant

async def test():
    assistant = AIVADialogAssistant()
    
    test_inputs = [
        "掃描 http://localhost:3000",
        " 掃描 http://localhost:3000 ",  # 帶空格
        "掃描http://localhost:3000",  # 無空格
    ]
    
    for input_text in test_inputs:
        print(f"\n測試輸入: '{input_text}'")
        print(f"長度: {len(input_text)}, repr: {repr(input_text)}")
        
        response = await assistant.process_user_input(input_text)
        print(f"回應: {response.get('message', '')[:100]}")

if __name__ == "__main__":
    asyncio.run(test())
