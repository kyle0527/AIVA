"""統計從 840 flows 提取的 CLI 指令能力"""
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

# 載入數據
flows_file = PROJECT_ROOT / "services/integration/data/internal_exploration/latest_classification.json"
mappings_file = PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_mappings.json"

with open(flows_file, 'r', encoding='utf-8') as f:
    flows_data = json.load(f)

with open(mappings_file, 'r', encoding='utf-8') as f:
    script_mappings = json.load(f)

# 統計信息
total_flows = len(flows_data['flows'])
total_unique_scripts = len(set(script for flow in flows_data['flows'] for script in flow['path']))
mapped_scripts = len(script_mappings)

print("=" * 80)
print("🎯 從 840 Flows 提取的 CLI 指令能力統計")
print("=" * 80)
print()
print(f"📊 總體數據:")
print(f"   - 總 Flows 數量: {total_flows}")
print(f"   - 唯一腳本數量: {total_unique_scripts}")
print(f"   - 已映射腳本: {mapped_scripts} ({mapped_scripts/total_unique_scripts*100:.1f}%)")
print(f"   - 未映射腳本: {total_unique_scripts - mapped_scripts}")
print()

# 按功能分類
categories = {
    "🤖 機器學習/訓練": [],
    "🔒 安全測試/攻擊": [],
    "📊 監控/性能": [],
    "🎨 UI/可視化": [],
    "🔧 系統管理": [],
    "🔍 分析/診斷": [],
    "💾 狀態管理": [],
    "🌐 API/網絡": []
}

# 分類映射
for script_name, info in script_mappings.items():
    path = info.get('script_path', '')
    
    if 'learning' in path or 'training' in path or 'ai_model' in path or 'trainer' in script_name:
        categories["🤖 機器學習/訓練"].append(script_name)
    elif 'attack' in path or 'exploit' in path or 'scan' in path or 'initial_surface' in script_name:
        categories["🔒 安全測試/攻擊"].append(script_name)
    elif 'monitoring' in path or 'performance' in path:
        categories["📊 監控/性能"].append(script_name)
    elif 'ui' in path or 'dashboard' in path or 'visualizer' in path:
        categories["🎨 UI/可視化"].append(script_name)
    elif 'state' in path or 'session' in path:
        categories["💾 狀態管理"].append(script_name)
    elif 'analysis' in path or 'analyze' in script_name or 'checker' in script_name:
        categories["🔍 分析/診斷"].append(script_name)
    elif 'api' in path or 'app' == script_name:
        categories["🌐 API/網絡"].append(script_name)
    else:
        categories["🔧 系統管理"].append(script_name)

print("=" * 80)
print("📋 已映射能力分類（17 個可用 CLI 指令）")
print("=" * 80)
print()

for category, scripts in categories.items():
    if scripts:
        print(f"{category} ({len(scripts)} 個)")
        print("-" * 80)
        for i, script in enumerate(scripts, 1):
            info = script_mappings[script]
            mock_status = "🟡 MOCK" if info.get('mock', False) else "🟢 REAL"
            method = info.get('method', info.get('function', '?'))
            file = info.get('script_file', '?')
            
            print(f"  {i}. {script}")
            print(f"     狀態: {mock_status}")
            print(f"     文件: {file}")
            print(f"     調用: {info.get('class', 'N/A')}.{method}()")
            print(f"     用途: ", end="")
            
            # 根據名稱推測功能
            if 'monitoring' in script:
                print("收集系統性能指標和健康狀態")
            elif 'optimized_core' in script:
                print("核心系統協調和優化處理")
            elif 'train_classifier' in script:
                print("訓練分類器模型")
            elif 'model_trainer' in script:
                print("通用機器學習模型訓練")
            elif 'training_orchestrator' in script:
                print("編排和協調分布式訓練任務")
            elif 'initial_surface' in script:
                print("分析目標系統的初始攻擊面")
            elif 'exploit_orchestrator' in script:
                print("編排和執行漏洞利用")
            elif 'scan_module_interface' in script:
                print("掃描模組統一接口")
            elif 'app' in script:
                print("FastAPI 應用主入口")
            elif 'session_state_manager' in script:
                print("會話狀態管理和追蹤")
            elif 'system_connectivity_checker' in script:
                print("檢查系統連接性和網絡狀態")
            elif 'dashboard' in script:
                print("主控制面板渲染")
            elif 'improved_ui' in script:
                print("改進的用戶界面顯示")
            elif 'analyze_dataflow_breakpoints' in script:
                print("分析數據流斷點和異常")
            elif 'analyze_missing_function_connections' in script:
                print("分析缺失的函數連接")
            elif 'scalable_bio_trainer' in script:
                print("可擴展生物啟發訓練器")
            elif 'ai_model_manager' in script:
                print("AI 模型管理和版本控制")
            else:
                print("待定義")
            
            print()

print()
print("=" * 80)
print("🎯 CLI 指令執行方式")
print("=" * 80)
print()
print("1. 執行單個 Flow:")
print("   python aiva_flow_runner.py <flow_id>")
print("   例如: python aiva_flow_runner.py 0")
print()
print("2. 測試 10 個代表性 Flows:")
print("   python aiva_flow_runner.py --test")
print()
print("3. Flow 組合示例:")
print("   Flow 0 (2節點): monitoring → optimized_core")
print("   Flow 1 (3節點): monitoring → optimized_core → train_classifier")
print("   Flow 2 (4節點): monitoring → ... → model_trainer")
print("   Flow 3 (5節點): monitoring → ... → training_orchestrator")
print()

print("=" * 80)
print("📈 下一步擴展計劃")
print("=" * 80)
print()
print(f"✅ 已完成: {mapped_scripts}/{total_unique_scripts} 腳本映射 ({mapped_scripts/total_unique_scripts*100:.1f}%)")
print(f"⏳ 待完成: {total_unique_scripts - mapped_scripts} 腳本")
print()
print("優先擴展順序:")
print("1. 完成 10 個測試 flows 的所有節點 (17 個腳本) ✅")
print("2. 添加高頻使用的節點 (30-50 個腳本)")
print("3. 覆蓋所有 6 個入口點的主要路徑")
print("4. 最終目標: 102 個腳本全部映射")
print()

# 統計未映射的高頻腳本
script_frequency = defaultdict(int)
for flow in flows_data['flows']:
    for script in flow['path']:
        script_frequency[script] += 1

unmapped_scripts = [(s, count) for s, count in script_frequency.items() 
                    if s not in script_mappings]
unmapped_scripts.sort(key=lambda x: x[1], reverse=True)

print("🔥 最高頻的 10 個未映射腳本:")
for i, (script, count) in enumerate(unmapped_scripts[:10], 1):
    print(f"   {i}. {script}: 出現在 {count} 個 flows 中")

print()
print("=" * 80)
