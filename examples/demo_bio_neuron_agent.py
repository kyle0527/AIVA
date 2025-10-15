"""
示範如何使用 BioNeuronRAGAgent 操作整個 AIVA 程式
"""

from services.core.aiva_core.ai_engine import BioNeuronRAGAgent


def main() -> None:
    """主函式,展示 RAG Agent 的使用方法."""
    print("="*70)
    print("   BioNeuronRAGAgent 使用示範")
    print("="*70)

    # 初始化代理 (會自動索引整個程式碼庫)
    agent = BioNeuronRAGAgent(codebase_path="c:/D/E/AIVA/AIVA-main")

    # 範例 1: 讀取程式碼
    print("\n\n【範例 1】讀取程式碼檔案")
    print("-" * 70)
    result = agent.invoke(
        query="讀取掃描器的主要入口檔案",
        path="services/scan/aiva_scan/worker.py",
    )
    print(f"\n執行狀態: {result['status']}")
    print(f"使用工具: {result['tool_used']}")
    print(f"信心度: {result['confidence']:.2%}")
    if result['tool_result']['status'] == 'success':
        lines = result['tool_result']['lines']
        print(f"檔案行數: {lines}")

    # 範例 2: 分析程式碼
    print("\n\n【範例 2】分析程式碼結構")
    print("-" * 70)
    result = agent.invoke(
        query="分析核心模組的應用程式結構",
        path="services/core/aiva_core/app.py",
    )
    print(f"\n執行狀態: {result['status']}")
    print(f"使用工具: {result['tool_used']}")
    if result['tool_result']['status'] == 'success':
        print(f"總行數: {result['tool_result']['total_lines']}")
        print(f"函式數: {result['tool_result']['functions']}")
        print(f"類別數: {result['tool_result']['classes']}")

    # 範例 3: 觸發掃描
    print("\n\n【範例 3】觸發漏洞掃描")
    print("-" * 70)
    result = agent.invoke(
        query="對目標網站執行完整的安全掃描",
        target_url="https://example.com",
        scan_type="full",
    )
    print(f"\n執行狀態: {result['status']}")
    print(f"使用工具: {result['tool_used']}")
    if result['tool_result']['status'] == 'success':
        print(f"掃描任務 ID: {result['tool_result']['task_id']}")
        print(f"目標: {result['tool_result']['target']}")

    # 範例 4: 執行命令
    print("\n\n【範例 4】執行系統命令")
    print("-" * 70)
    result = agent.invoke(
        query="檢查 Python 版本",
        command="python --version",
    )
    print(f"\n執行狀態: {result['status']}")
    if result['status'] == 'success':
        print(f"使用工具: {result['tool_used']}")
        if result['tool_result']['status'] == 'success':
            print(f"命令輸出: {result['tool_result']['stdout'].strip()}")
    else:
        print(f"訊息: {result.get('message', '未知錯誤')}")
        print(f"信心度: {result.get('confidence', 0):.2%}")

    # 範例 5: 寫入程式碼 (謹慎使用!)
    print("\n\n【範例 5】寫入測試檔案")
    print("-" * 70)
    test_code = '''"""測試檔案,由 BioNeuronRAGAgent 自動生成."""

def hello_world() -> None:
    """測試函式."""
    print("Hello from BioNeuronRAGAgent!")

if __name__ == "__main__":
    hello_world()
'''
    result = agent.invoke(
        query="建立一個簡單的測試 Python 檔案",
        path="test_agent_generated.py",
        content=test_code,
    )
    print(f"\n執行狀態: {result['status']}")
    if result['status'] == 'success':
        print(f"使用工具: {result['tool_used']}")
        if result['tool_result']['status'] == 'success':
            print(f"寫入位元組數: {result['tool_result']['bytes_written']}")
    else:
        print(f"訊息: {result.get('message', '未知錯誤')}")

    # 顯示知識庫統計
    print("\n\n【知識庫統計】")
    print("-" * 70)
    stats = agent.get_knowledge_stats()
    print(f"程式碼片段總數: {stats['total_chunks']}")
    print(f"關鍵字總數: {stats['total_keywords']}")

    # 顯示執行歷史
    print("\n\n【執行歷史】")
    print("-" * 70)
    history = agent.get_history()
    for i, record in enumerate(history, 1):
        print(f"{i}. {record['query'][:50]}... -> {record['tool_used']}")

    print("\n" + "="*70)
    print("   示範完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
