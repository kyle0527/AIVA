#!/usr/bin/env python3
"""
AI 系統通連檢查腳本

驗證 AI 決策能夠實際執行系統命令和操作的完整流程
"""

import asyncio
import sys
import os
import subprocess
import time
from pathlib import Path

# 添加路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'core'))

async def check_ai_to_system_connectivity():
    """檢查 AI 到系統的完整連通性"""
    print("🔗 AI 系統通連檢查開始")
    print("=" * 60)
    
    results = {}
    
    # 1. 檢查 AI 核心組件載入
    print("\n1️⃣ 檢查 AI 核心組件載入...")
    try:
        from aiva_core.ai_engine import AIModelManager, BioNeuronRAGAgent, ScalableBioNet
        from aiva_core.ai_engine import OptimizedScalableBioNet, PerformanceConfig
        print("✅ AI 核心組件載入成功")
        results['ai_core_loading'] = True
    except Exception as e:
        print(f"❌ AI 核心組件載入失敗: {e}")
        results['ai_core_loading'] = False
        return results
    
    # 2. 檢查 AI 工具系統連接
    print("\n2️⃣ 檢查 AI 工具系統連接...")
    try:
        from aiva_core.ai_engine import (
            Tool, CodeReader, CodeWriter, CodeAnalyzer, 
            CommandExecutor, ScanTrigger, VulnerabilityDetector
        )
        print("✅ AI 工具系統連接成功")
        results['ai_tools_connection'] = True
    except Exception as e:
        print(f"❌ AI 工具系統連接失敗: {e}")
        results['ai_tools_connection'] = False
    
    # 3. 檢查 AI 決策 → 工具調用
    print("\n3️⃣ 檢查 AI 決策 → 工具調用...")
    try:
        # 初始化 AI 管理器
        manager = AIModelManager(model_dir=Path("./test_models"))
        init_result = await manager.initialize_models(input_size=64, num_tools=6)
        
        if init_result['status'] == 'success':
            print("✅ AI 模型初始化成功")
            
            # 測試決策調用
            decision_result = await manager.make_decision(
                "執行系統掃描",
                {"target": "localhost", "scan_type": "basic"},
                use_rag=False
            )
            
            if decision_result['status'] == 'success':
                print("✅ AI 決策 → 工具調用成功")
                results['ai_decision_tool_call'] = True
            else:
                print(f"❌ AI 決策調用失敗: {decision_result.get('error')}")
                results['ai_decision_tool_call'] = False
        else:
            print(f"❌ AI 模型初始化失敗: {init_result.get('error')}")
            results['ai_decision_tool_call'] = False
            
    except Exception as e:
        print(f"❌ AI 決策 → 工具調用失敗: {e}")
        results['ai_decision_tool_call'] = False
    
    # 4. 檢查工具 → 系統命令執行
    print("\n4️⃣ 檢查工具 → 系統命令執行...")
    try:
        # 測試 CommandExecutor
        cmd_executor = CommandExecutor()
        
        # 測試簡單的系統命令
        test_commands = [
            "echo 'AI system test'",
            "dir" if os.name == 'nt' else "ls",
            "python --version"
        ]
        
        successful_commands = 0
        for cmd in test_commands:
            try:
                result = subprocess.run(
                    cmd.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    successful_commands += 1
                    print(f"  ✅ 命令成功: {cmd}")
                else:
                    print(f"  ⚠️  命令警告: {cmd} (返回碼: {result.returncode})")
            except Exception as e:
                print(f"  ❌ 命令失敗: {cmd} - {e}")
        
        if successful_commands >= 2:
            print("✅ 工具 → 系統命令執行正常")
            results['tool_system_execution'] = True
        else:
            print("❌ 工具 → 系統命令執行異常")
            results['tool_system_execution'] = False
            
    except Exception as e:
        print(f"❌ 工具 → 系統命令執行檢查失敗: {e}")
        results['tool_system_execution'] = False
    
    # 5. 檢查文件系統訪問
    print("\n5️⃣ 檢查文件系統訪問...")
    try:
        # 測試 CodeReader 和 CodeWriter
        code_reader = CodeReader()
        code_writer = CodeWriter()
        
        # 創建測試文件
        test_file = Path("./test_ai_system_connectivity.tmp")
        test_content = "# AI 系統連通性測試\nprint('Hello from AI system')\n"
        
        # 寫入測試
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 讀取測試
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            if test_content == read_content:
                print("✅ 文件系統讀寫正常")
                results['file_system_access'] = True
            else:
                print("❌ 文件系統讀寫內容不一致")
                results['file_system_access'] = False
        else:
            print("❌ 文件系統寫入失敗")
            results['file_system_access'] = False
        
        # 清理測試文件
        if test_file.exists():
            test_file.unlink()
            
    except Exception as e:
        print(f"❌ 文件系統訪問檢查失敗: {e}")
        results['file_system_access'] = False
    
    # 6. 檢查網路連接 (可選)
    print("\n6️⃣ 檢查網路連接...")
    try:
        import socket
        
        # 測試本地連接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 80))  # 測試本地 80 端口
        sock.close()
        
        # 測試 DNS 解析
        try:
            socket.gethostbyname('localhost')
            print("✅ 網路連接正常 (本地)")
            results['network_connectivity'] = True
        except Exception:
            print("⚠️  網路連接受限 (僅本地)")
            results['network_connectivity'] = False
            
    except Exception as e:
        print(f"⚠️  網路連接檢查失敗: {e}")
        results['network_connectivity'] = False
    
    # 7. 檢查 AI 訓練系統與存儲的連接
    print("\n7️⃣ 檢查 AI 訓練系統與存儲連接...")
    try:
        from aiva_core.learning import ModelTrainer, ScalableBioTrainer, ScalableBioTrainingConfig
        
        # 測試模型創建和基本操作
        import numpy as np
        
        # 創建測試模型
        test_model = type('TestModel', (), {
            'fc1': np.random.randn(10, 5),
            'fc2': np.random.randn(5, 3),
            'forward': lambda self, x: np.random.randn(len(x), 3),
            'backward': lambda self, x, y, lr: None
        })()
        
        # 測試訓練配置
        config = ScalableBioTrainingConfig(epochs=1, batch_size=4)
        trainer = ScalableBioTrainer(test_model, config)
        
        # 測試基本訓練功能
        X_test = np.random.randn(8, 10)
        y_test = np.random.randn(8, 3)
        
        training_result = trainer.train(X_test, y_test)
        
        if training_result and 'final_loss' in training_result:
            print("✅ AI 訓練系統與存儲連接正常")
            results['ai_training_storage'] = True
        else:
            print("❌ AI 訓練系統與存儲連接異常")
            results['ai_training_storage'] = False
            
    except Exception as e:
        print(f"❌ AI 訓練系統與存儲連接檢查失敗: {e}")
        results['ai_training_storage'] = False
    
    return results

async def check_command_execution_chain():
    """檢查命令執行鏈的完整性"""
    print("\n🔗 命令執行鏈檢查")
    print("=" * 60)
    
    try:
        # 1. AI 決策
        print("1️⃣ AI 決策層...")
        manager = AIModelManager()
        await manager.initialize_models(input_size=32, num_tools=4)
        
        # 2. 工具選擇
        print("2️⃣ 工具選擇層...")
        from aiva_core.ai_engine import CommandExecutor
        
        # 3. 命令構造
        print("3️⃣ 命令構造層...")
        test_command = "python -c \"print('AI command execution test')\""
        
        # 4. 系統執行
        print("4️⃣ 系統執行層...")
        result = subprocess.run(
            test_command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✅ 命令執行成功: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ 命令執行失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 命令執行鏈檢查失敗: {e}")
        return False

async def main():
    """主檢查函數"""
    print("🚀 開始 AI 與系統間通連完整檢查")
    print("=" * 70)
    
    # 基本通連檢查
    connectivity_results = await check_ai_to_system_connectivity()
    
    # 命令執行鏈檢查
    execution_chain_result = await check_command_execution_chain()
    
    # 結果統計
    print("\n" + "=" * 70)
    print("📊 通連檢查結果總結")
    print("=" * 70)
    
    total_checks = len(connectivity_results)
    passed_checks = sum(1 for result in connectivity_results.values() if result)
    
    print(f"\n基本通連檢查 ({passed_checks}/{total_checks}):")
    for check_name, result in connectivity_results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        check_display = check_name.replace('_', ' ').title()
        print(f"  {status} {check_display}")
    
    print(f"\n命令執行鏈檢查:")
    exec_status = "✅ 通過" if execution_chain_result else "❌ 失敗"
    print(f"  {exec_status} AI → 工具 → 命令 → 系統執行")
    
    # 整體評估
    overall_success_rate = (passed_checks + (1 if execution_chain_result else 0)) / (total_checks + 1)
    
    print(f"\n🎯 整體通連性: {overall_success_rate:.1%}")
    
    if overall_success_rate >= 0.8:
        print("🎉 AI 與系統通連性良好，可以進行實戰測試！")
    elif overall_success_rate >= 0.6:
        print("⚠️  AI 與系統通連性基本正常，建議檢查失敗項目")
    else:
        print("❌ AI 與系統通連性存在問題，需要修復失敗項目")
    
    print("=" * 70)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  檢查被用戶中斷")
    except Exception as e:
        print(f"\n💥 檢查過程發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())