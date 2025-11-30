"""直接讓 AI 對靶場發起攻擊"""
import asyncio
import sys
import os

# 設定 UTF-8 編碼
os.environ['PYTHONIOENCODING'] = 'utf-8'
# sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
# Windows PowerShell已使用UTF-8，無需重新配置
sys.path.append('.')

async def test_ai_attack():
    """讓 AI 接收指令並攻擊 WebGoat"""
    from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant
    
    print('=' * 80)
    print('AI 實戰測試：對 WebGoat 發起攻擊')
    print('=' * 80)
    
    # 初始化 AI 助理
    assistant = AIVADialogAssistant()
    
    # AI 接收自然語言指令
    user_command = "幫我跑 http://localhost:8080/WebGoat 的掃描"
    
    print(f'\n用戶指令: {user_command}')
    print('\nAI 開始處理...\n')
    
    try:
        # AI 處理指令並執行攻擊
        response = await assistant.process_user_input(user_command)
        
        print('AI 響應:')
        print(f'  意圖: {response["intent"]}')
        print(f'  可執行: {response["executable"]}')
        print(f'\nAI 回覆:\n{response["message"]}\n')
        
        # 顯示執行結果
        if response.get('data') and response['data'].get('result'):
            result = response['data']['result']
            print('攻擊執行完成!')
            print(f'  成功: {result["success"]}')
            print(f'  執行時間: {result["execution_time"]:.2f}s')
            if result.get('result'):
                print(f'  結果: {str(result["result"])[:500]}')
            if result.get('error'):
                print(f'  錯誤: {result["error"]}')
        
    except Exception as e:
        print(f'\n錯誤: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # 清理連接
        # 清理連接（如果有的話）
        if hasattr(assistant, '_db_conn') and assistant._db_conn:  # type: ignore[attr-defined]
            if not assistant._db_conn.is_closed():  # type: ignore[attr-defined]
                await assistant._db_conn.close()  # type: ignore[attr-defined]
    
    print('\n' + '=' * 80)
    print('測試完成')
    print('=' * 80)

if __name__ == '__main__':
    asyncio.run(test_ai_attack())
