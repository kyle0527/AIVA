#!/usr/bin/env python3
"""
AIVA 統一啟動腳本
用途: 在項目根目錄提供統一的 AI 持續學習啟動入口
維持五大模組架構的組織方式
"""

import sys
import asyncio
from pathlib import Path

# 添加服務路徑 - 確保可以找到 services 模組
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "services"))

def show_module_info():
    """顯示 AIVA 五大模組架構資訊"""
    print("🏗️  AIVA 五大模組架構")
    print("=" * 60)
    print("1. 🧩 aiva_common - 通用基礎模組")
    print("   └── 共享資料結構、枚舉、工具函數")
    print()
    print("2. 🧠 core - 核心業務模組")
    print("   ├── AI 引擎 (BioNeuron, 抗幻覺)")
    print("   ├── 決策代理 (風險評估, 經驗驅動)")
    print("   └── 任務協調與狀態管理")
    print()
    print("3. 🔍 scan - 掃描發現模組")
    print("   ├── 靶場環境檢測")
    print("   ├── 漏洞掃描引擎")
    print("   └── 資產發現與指紋識別")
    print()
    print("4. 🔗 integration - 整合服務模組")
    print("   ├── AI 持續學習觸發器")
    print("   ├── 操作記錄與監控")
    print("   └── API 閘道與報告系統")
    print()
    print("5. 🎯 features - 功能檢測模組")
    print("   ├── 漏洞檢測功能 (XSS, SQLi, IDOR)")
    print("   ├── 認證繞過功能 (JWT, OAuth)")
    print("   └── 智能檢測管理器")
    print()
    print("💡 API 接點: api/ 目錄提供 FastAPI 後端服務")
    print()

async def start_ai_continuous_learning():
    """啟動 AI 持續學習"""
    try:
        print("🚀 啟動 AIVA AI 持續學習...")
        print("📍 觸發器位置: services/integration/aiva_integration/")
        print()
        
        # 檢查檔案是否存在
        trigger_file = Path(__file__).parent.parent.parent / "services" / "integration" / "aiva_integration" / "trigger_ai_continuous_learning.py"
        
        if not trigger_file.exists():
            print("❌ 找不到觸發器檔案")
            print(f"   預期位置: {trigger_file}")
            return
            
        # 導入 Integration 模組中的觸發器
        try:
            from services.integration.aiva_integration.trigger_ai_continuous_learning import ManualTrainService, main
            await main()
        except ImportError as e:
            print(f"❌ 模組導入失敗: {e}")
            print("💡 嘗試使用替代方式啟動...")
            
            # 替代方式: 直接執行檔案
            import subprocess
            import sys
            print("📋 正在執行觸發器...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(trigger_file),
                cwd=str(trigger_file.parent.parent.parent.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                print(stdout.decode())
            if stderr:
                print(f"⚠️  錯誤輸出: {stderr.decode()}")
        
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")

def show_available_tools():
    """顯示可用工具"""
    print("🛠️  可用工具腳本")
    print("=" * 60)
    
    tools = [
        {
            "name": "AI 持續學習觸發器",
            "path": "services/integration/aiva_integration/trigger_ai_continuous_learning.py",
            "module": "Integration",
            "description": "手動觸發 AI 持續攻擊學習"
        },
        {
            "name": "整合式 AI 訓練器",
            "path": "services/integration/aiva_integration/integrated_ai_trainer.py",
            "module": "Integration",
            "description": "統一的 AI 模型訓練系統"
        },
        {
            "name": "抗幻覺驗證模組",
            "path": "services/core/aiva_core/ai_engine/anti_hallucination_module.py", 
            "module": "Core",
            "description": "防止 AI 生成不合理步驟"
        },
        {
            "name": "BioNeuron 核心引擎",
            "path": "services/core/aiva_core/ai_engine/bio_neuron_core.py",
            "module": "Core",
            "description": "生物神經元啟發的 AI 引擎"
        },
        {
            "name": "靶場環境檢測器",
            "path": "services/scan/aiva_scan/target_environment_detector.py",
            "module": "Scan", 
            "description": "自動檢測靶場狀態"
        },
        {
            "name": "漏洞掃描器",
            "path": "services/scan/aiva_scan/vulnerability_scanner.py",
            "module": "Scan",
            "description": "統一漏洞掃描引擎"
        },
        {
            "name": "AI 操作記錄器",
            "path": "services/integration/aiva_integration/ai_operation_recorder.py",
            "module": "Integration",
            "description": "結構化記錄 AI 操作"
        },
        {
            "name": "決策代理增強模組",
            "path": "services/core/aiva_core/decision/enhanced_decision_agent.py",
            "module": "Core",
            "description": "智能化決策系統"
        },
        {
            "name": "智能檢測管理器",
            "path": "services/features/smart_detection_manager.py",
            "module": "Features",
            "description": "統一功能檢測管理"
        },
        {
            "name": "高價值指南管理器",
            "path": "services/features/high_value_manager.py",
            "module": "Features",
            "description": "高價值漏洞引導系統"
        }
    ]
    
    for i, tool in enumerate(tools, 1):
        # 檢查檔案是否存在
        tool_path = Path(__file__).parent.parent.parent / tool['path']
        status = "✅" if tool_path.exists() else "❌"
        
        print(f"{i}. 📋 {tool['name']} {status}")
        print(f"   🏠 模組: {tool['module']}")
        print(f"   📁 路徑: {tool['path']}")
        print(f"   📝 說明: {tool['description']}")
        print()

def start_api_service():
    """啟動 API 服務"""
    try:
        import subprocess
        import sys
        
        print("🌐 啟動 AIVA API 服務...")
        print("📍 API 服務位置: api/")
        print("📋 正在檢查 API 主檔案...")
        
        api_main = Path(__file__).parent.parent.parent / "api" / "main.py"
        api_start = Path(__file__).parent.parent.parent / "api" / "start_api.py"
        
        if api_start.exists():
            print("✅ 使用 start_api.py 啟動服務")
            print("🔗 API 服務將在背景執行...")
            subprocess.Popen([sys.executable, str(api_start)])
        elif api_main.exists():
            print("✅ 使用 main.py 啟動服務")
            print("🔗 API 服務將在背景執行...")
            subprocess.Popen([sys.executable, str(api_main)])
        else:
            print("❌ 找不到 API 主檔案")
            print("💡 請確認 api/main.py 或 api/start_api.py 存在")
            
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")

def show_system_status():
    """顯示系統狀態"""
    print("📊 AIVA 系統狀態")
    print("=" * 60)
    
    # 檢查主要模組目錄
    project_root = Path(__file__).parent.parent.parent
    modules = [
        ("aiva_common", "services/aiva_common"),
        ("core", "services/core"),
        ("scan", "services/scan"),
        ("integration", "services/integration"),
        ("features", "services/features"),
        ("API", "api")
    ]
    
    for module_name, module_path in modules:
        full_path = project_root / module_path
        status = "✅ 存在" if full_path.exists() else "❌ 缺失"
        print(f"📁 {module_name:12} - {status}")
    
    print()
    print("🐍 Python 環境:")
    print(f"   版本: {sys.version.split()[0]}")
    print(f"   路徑: {sys.executable}")
    print()

def main():
    """主函數"""
    print("🎮 AIVA 統一啟動介面")
    print("=" * 60)
    
    # 顯示系統狀態
    show_system_status()
    
    while True:
        print("\n請選擇操作:")
        print("1. 🚀 啟動 AI 持續學習")
        print("2. 🌐 啟動 API 服務")
        print("3. 🏗️  查看模組架構")
        print("4. 🛠️  查看可用工具")
        print("5. � 重新檢查系統狀態")
        print("6. �🚪 退出")
        
        try:
            choice = input("\n請輸入選項 (1-6): ").strip()
            
            if choice == "1":
                print("\n" + "="*60)
                asyncio.run(start_ai_continuous_learning())
                
            elif choice == "2":
                print("\n" + "="*60)
                start_api_service()
                
            elif choice == "3":
                print("\n" + "="*60)
                show_module_info()
                
            elif choice == "4":
                print("\n" + "="*60)
                show_available_tools()
                
            elif choice == "5":
                print("\n" + "="*60)
                show_system_status()
                
            elif choice == "6":
                print("\n👋 再見！")
                break
                
            else:
                print("❌ 無效選項，請輸入 1-6")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已中斷")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()