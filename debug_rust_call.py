"""
調試 Rust 掃描器調用
檢查完整流程中 Rust 掃描器實際接收的參數
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 導入 Rust Bridge
from services.scan.engines.rust_engine.python_bridge import rust_info_gatherer

def test_rust_bridge():
    """測試 Rust Bridge 調用"""
    print("\n" + "=" * 80)
    print("Rust Bridge 調用測試")
    print("=" * 80)
    
    # 1. 檢查可用性
    print(f"\n可用: {rust_info_gatherer.is_available()}")
    print(f"類型: {rust_info_gatherer.__class__.__name__}")
    
    if rust_info_gatherer.__class__.__name__ == "MockRustInfoGatherer":
        print("\n警告: 使用的是 Mock 版本!")
        return False
    
    # 2. 測試掃描
    target_url = "http://localhost:3000"
    config = {
        "mode": "fast",
        "timeout": 10
    }
    
    print(f"\n目標: {target_url}")
    print(f"配置: {json.dumps(config, indent=2)}")
    
    print("\n執行掃描...")
    result = rust_info_gatherer.scan_target(target_url, config)
    
    print(f"\n結果:")
    print(f"  成功: {result.get('success', False)}")
    print(f"  錯誤: {result.get('error', '無')}")
    
    if result.get("success"):
        results_data = result.get("results", {})
        print(f"  端點數: {len(results_data.get('targets', []))}")
        
        # 顯示第一個目標的數據
        targets = results_data.get("targets", [])
        if targets:
            first_target = targets[0]
            print(f"\n第一個目標:")
            print(f"    URL: {first_target.get('url')}")
            print(f"    端點: {len(first_target.get('endpoints', []))}")
            print(f"    表單: {first_target.get('forms_count', 0)}")
            print(f"    JS 發現: {first_target.get('js_findings_count', 0)}")
    else:
        print(f"\n原始輸出: {result.get('raw_output', '無')}")
    
    return result.get("success", False)

if __name__ == "__main__":
    success = test_rust_bridge()
    sys.exit(0 if success else 1)
