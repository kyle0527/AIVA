#!/usr/bin/env python3
"""
AIVA 統一啟動腳本
用途: 在項目根目錄提供統一的 AI 持續學習啟動入口
維持五大模組架構的組織方式
"""

import sys
import asyncio
from pathlib import Path

# 添加服務路徑
sys.path.append(str(Path(__file__).parent))

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
    print("5. ⚙️  function - 功能檢測模組")
    print("   └── 功能測試與檢測執行")
    print()

async def start_ai_continuous_learning():
    """啟動 AI 持續學習"""
    try:
        # 導入 Integration 模組中的觸發器
        from services.integration.aiva_integration.trigger_ai_continuous_learning import ManualTrainService, main
        
        print("🚀 啟動 AIVA AI 持續學習...")
        print("📍 觸發器位置: services/integration/aiva_integration/")
        print()
        
        await main()
        
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        print("💡 請確認 services/integration/aiva_integration/ 目錄存在")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")

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
            "name": "抗幻覺驗證模組",
            "path": "services/core/aiva_core/ai_engine/anti_hallucination_module.py", 
            "module": "Core",
            "description": "防止 AI 生成不合理步驟"
        },
        {
            "name": "靶場環境檢測器",
            "path": "services/scan/aiva_scan/target_environment_detector.py",
            "module": "Scan", 
            "description": "自動檢測靶場狀態"
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
        }
    ]
    
    for i, tool in enumerate(tools, 1):
        print(f"{i}. 📋 {tool['name']}")
        print(f"   🏠 模組: {tool['module']}")
        print(f"   📁 路徑: {tool['path']}")
        print(f"   📝 說明: {tool['description']}")
        print()

def main():
    """主函數"""
    print("🎮 AIVA 統一啟動介面")
    print("=" * 60)
    
    while True:
        print("\n請選擇操作:")
        print("1. 🚀 啟動 AI 持續學習")
        print("2. 🏗️  查看模組架構")
        print("3. 🛠️  查看可用工具")
        print("4. 🚪 退出")
        
        try:
            choice = input("\n請輸入選項 (1-4): ").strip()
            
            if choice == "1":
                print("\n" + "="*60)
                asyncio.run(start_ai_continuous_learning())
                
            elif choice == "2":
                print("\n" + "="*60)
                show_module_info()
                
            elif choice == "3":
                print("\n" + "="*60)
                show_available_tools()
                
            elif choice == "4":
                print("\n👋 再見！")
                break
                
            else:
                print("❌ 無效選項，請輸入 1-4")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已中斷")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()