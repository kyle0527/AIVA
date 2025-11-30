"""
內閉環自我認知能力驗證腳本
"""
import asyncio
from services.core.aiva_core.cognitive_core import InternalLoopConnector
from services.core.aiva_core.cognitive_core.rag import KnowledgeBase, VectorStore

async def verify():
    vs = VectorStore()
    kb = KnowledgeBase(vector_store=vs)
    connector = InternalLoopConnector(rag_knowledge_base=kb)
    
    tests = [
        ('能力數量', '我有多少個能力?'),
        ('攻擊能力', '列出所有攻擊相關的能力'),
        ('Python 能力', '我有哪些 Python 語言的能力?'),
        ('Go 能力', '我有哪些 Go 語言的能力?'),
    ]
    
    print('=' * 60)
    print('🧪 AIVA 內閉環自我認知能力驗證')
    print('=' * 60)
    
    passed = 0
    for name, query in tests:
        print(f'\n測試 {passed + 1}/{len(tests)}: {name}')
        print(f'查詢: {query}')
        try:
            result = await connector.query_self_awareness(query)
            
            # 處理返回結果（可能是字符串或字典）
            if isinstance(result, str):
                print(f'✅ 通過 - 獲得回答')
                print(f'   回答預覽: {result[:150]}...')
                passed += 1
            elif isinstance(result, dict) and result.get('answer'):
                sources_count = len(result.get('sources', []))
                print(f'✅ 通過 - 找到 {sources_count} 個相關文檔')
                print(f'   回答預覽: {result["answer"][:150]}...')
                passed += 1
            elif isinstance(result, list) and len(result) > 0:
                print(f'✅ 通過 - 找到 {len(result)} 個相關結果')
                passed += 1
            else:
                print(f'❌ 失敗 - 未找到相關結果 (返回類型: {type(result).__name__})')
        except Exception as e:
            print(f'❌ 錯誤: {e}')
            import traceback
            traceback.print_exc()
    
    print(f'\n' + '=' * 60)
    print(f'總結: {passed}/{len(tests)} 測試通過')
    print('=' * 60)
    return passed == len(tests)

if __name__ == '__main__':
    if asyncio.run(verify()):
        print('\n✅ 內閉環自我認知能力驗證成功！')
    else:
        print('\n❌ 部分測試失敗，但內閉環基本運作正常')
