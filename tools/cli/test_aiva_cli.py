#!/usr/bin/env python3
"""
AIVA CLI 測試腳本 - 示範各種功能
"""

import subprocess
import sys

def run_command(cmd):
    """執行CLI命令並顯示結果"""
    print(f"$ {cmd}")
    print("-" * 60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"錯誤: {result.stderr}")
    except Exception as e:
        print(f"執行失敗: {e}")
    print("=" * 60)
    print()

def main():
    """運行示範"""
    print("🚀 AIVA CLI 功能示範")
    print("=" * 60)
    print()
    
    # 測試命令列表
    test_commands = [
        # 基本統計
        "python aiva_cli_implementation.py stats",
        
        # 簡單路徑
        "python aiva_cli_implementation.py reach ai_model_manager --dry-run",
        
        # 指定起點
        "python aiva_cli_implementation.py reach model_trainer --from=scalable_bio_trainer --dry-run",
        
        # 路徑查詢
        "python aiva_cli_implementation.py paths scalable_bio_trainer model_trainer",
        
        # 選擇特定路徑
        "python aiva_cli_implementation.py reach model_trainer --from=scalable_bio_trainer --path=2 --dry-run",
        
        # 流程列表
        "python aiva_cli_implementation.py flow list --module=cognitive_core",
        
        # 顯示流程詳情
        "python aiva_cli_implementation.py flow show 76",
        
        # 執行流程
        "python aiva_cli_implementation.py flow run 76 --dry-run"
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"測試 {i}: {cmd.split(' ')[-2:] if '--' in cmd else cmd.split(' ')[-1:]}")
        run_command(cmd)

if __name__ == "__main__":
    main()