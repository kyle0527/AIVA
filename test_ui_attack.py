"""測試 UI 攻擊功能的獨立腳本"""
import asyncio
import json
from datetime import datetime

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    from services.core.aiva_core.ui_panel.dashboard import Dashboard
    
    print("="*70)
    print("AIVA v3.0 - UI 攻擊功能完整測試")
    print("="*70)
    
    # 初始化 Dashboard
    print("\n[1/5] 初始化 Dashboard (UI 模式)...")
    dashboard = Dashboard(mode='ui')
    print("✅ Dashboard 初始化成功")
    
    # 測試目標
    targets = [
        "http://localhost:3000",  # juice-shop-live
        "http://localhost:3001",  # ecstatic_ritchie
        "http://localhost:3003",  # vigilant_shockle
    ]
    
    print(f"\n[2/5] 檢測到 {len(targets)} 個可用靶場")
    for i, target in enumerate(targets, 1):
        print(f"  {i}. {target}")
    
    # 選擇第一個目標進行測試
    test_target = targets[0]
    print(f"\n[3/5] 選擇測試目標: {test_target}")
    
    # 執行攻擊（使用 AI 模式，會觸發 AttackExecutor fallback）
    print(f"\n[4/5] 啟動 AI 自主攻擊...")
    print("  - 模式: AI + AttackExecutor fallback")
    print("  - 掃描類型: full")
    print("  - 安全級別: testing")
    
    task = dashboard.create_scan_task(test_target, 'full', use_ai=True)
    
    print(f"\n[5/5] 攻擊執行結果:")
    print("-"*70)
    
    # 格式化輸出（處理 datetime 對象）
    task_json = json.dumps(task, indent=2, ensure_ascii=False, default=json_serial)
    print(task_json)
    
    print("\n" + "="*70)
    print("測試摘要:")
    print("="*70)
    print(f"任務 ID: {task.get('task_id')}")
    print(f"目標: {task.get('target')}")
    print(f"狀態: {task.get('status')}")
    print(f"創建者: {task.get('created_by')}")
    
    if 'metrics' in task:
        metrics = task['metrics']
        print(f"\n執行指標:")
        print(f"  - 總步驟數: {metrics.get('total_steps', 0)}")
        print(f"  - 成功步驟: {metrics.get('successful_steps', 0)}")
        print(f"  - 失敗步驟: {metrics.get('failed_steps', 0)}")
        print(f"  - 執行時間: {metrics.get('total_execution_time', 0):.4f}秒")
    
    if 'execution_result' in task:
        exec_result = task['execution_result']
        if 'recommendations' in exec_result:
            print(f"\n建議:")
            for rec in exec_result['recommendations']:
                print(f"  • {rec}")
    
    print("\n✅ 測試完成")
    print("="*70)

if __name__ == "__main__":
    main()
