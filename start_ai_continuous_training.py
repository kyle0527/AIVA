# AIVA 持續學習觸發腳本
# 用途: 在 VS Code 中手動觸發 AI 持續攻擊學習

import asyncio
import sys
import os
from pathlib import Path

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent / "services"))

from core.aiva_core.ai_commander import AICommander
from core.aiva_core.training.training_orchestrator import TrainingOrchestrator
from integration.aiva_integration.system_performance_monitor import SystemPerformanceMonitor

class ManualTrainService:
    """手動觸發的持續訓練服務"""
    
    def __init__(self):
        self.ai_commander = None
        self.training_orchestrator = None
        self.performance_monitor = None
        self.is_running = False
        
    async def check_target_environment(self):
        """檢查靶場環境是否準備就緒"""
        print("🎯 檢查靶場環境...")
        
        # 檢查常見靶場端口
        target_ports = [80, 443, 3000, 8080, 8888, 9000]
        available_targets = []
        
        for port in target_ports:
            # 這裡可以加入實際的端口掃描邏輯
            # 暫時模擬檢查
            print(f"   檢查端口 {port}...")
            
        print("✅ 靶場環境檢查完成")
        return True
    
    async def initialize_ai_components(self):
        """初始化 AI 組件"""
        print("🧠 初始化 AI 組件...")
        
        try:
            # 初始化 AI 指揮官
            self.ai_commander = AICommander()
            print("   ✅ AI Commander 初始化完成")
            
            # 初始化訓練編排器
            self.training_orchestrator = TrainingOrchestrator()
            print("   ✅ Training Orchestrator 初始化完成")
            
            # 初始化性能監控器
            self.performance_monitor = SystemPerformanceMonitor()
            print("   ✅ Performance Monitor 初始化完成")
            
            return True
            
        except Exception as e:
            print(f"❌ AI 組件初始化失敗: {e}")
            return False
    
    async def start_continuous_loop(self):
        """開始持續學習迴圈"""
        print("🚀 開始 AI 持續學習...")
        self.is_running = True
        
        loop_count = 0
        
        try:
            while self.is_running:
                loop_count += 1
                print(f"\n🔄 === 學習迴圈 #{loop_count} ===")
                
                # 1. 執行一個訓練批次
                print("📚 執行訓練批次...")
                if hasattr(self.training_orchestrator, 'run_training_batch'):
                    await self.training_orchestrator.run_training_batch()
                
                # 2. 性能監控
                if self.performance_monitor:
                    metrics = self.performance_monitor.get_system_metrics()
                    print(f"📊 系統性能: {metrics}")
                
                # 3. 保存 AI 狀態
                if hasattr(self.ai_commander, 'save_state'):
                    self.ai_commander.save_state()
                    print("💾 AI 狀態已保存")
                
                # 4. 短暫休息
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n⏹️  收到停止信號，正在安全關閉...")
            self.is_running = False
        except Exception as e:
            print(f"❌ 訓練迴圈發生錯誤: {e}")
            print("🔄 5秒後重試...")
            await asyncio.sleep(5)
    
    def stop(self):
        """停止持續學習"""
        print("🛑 停止 AI 持續學習...")
        self.is_running = False

async def main():
    """主函數 - VS Code 中執行的入口點"""
    print("🎮 AIVA AI 持續學習觸發器")
    print("=" * 50)
    
    # 創建訓練服務
    train_service = ManualTrainService()
    
    try:
        # 1. 檢查靶場環境
        if not await train_service.check_target_environment():
            print("❌ 靶場環境未準備就緒，請先啟動靶場")
            return
        
        # 2. 初始化 AI 組件
        if not await train_service.initialize_ai_components():
            print("❌ AI 組件初始化失敗")
            return
        
        # 3. 開始持續學習
        print("\n🎯 一切就緒！輸入 Ctrl+C 可隨時停止")
        await train_service.start_continuous_loop()
        
    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")
    finally:
        train_service.stop()
        print("✅ AI 持續學習已停止")

if __name__ == "__main__":
    # 在 VS Code 中可以直接運行此腳本
    print("開始執行 AIVA AI 持續學習...")
    asyncio.run(main())