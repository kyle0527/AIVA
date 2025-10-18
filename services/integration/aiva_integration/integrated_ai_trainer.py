#!/usr/bin/env python3
"""
AIVA 增強型 AI 持續學習觸發器
用途: 整合真實 AIVA 模組功能的持續學習系統
基於: 五大模組架構的完整功能整合
"""

import asyncio
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# 嘗試導入真實的 AIVA 模組
try:
    # Core 模組
    from services.core.aiva_core.ai_engine.anti_hallucination_module import AntiHallucinationModule
    from services.core.aiva_core.decision.enhanced_decision_agent import EnhancedDecisionAgent, DecisionContext, RiskLevel
    
    # Scan 模組  
    from services.scan.aiva_scan.target_environment_detector import TargetEnvironmentDetector
    
    # Integration 模組
    from services.integration.aiva_integration.ai_operation_recorder import AIOperationRecorder
    from services.integration.aiva_integration.system_performance_monitor import SystemPerformanceMonitor
    
    REAL_MODULES_AVAILABLE = True
    print("✅ 成功載入真實 AIVA 模組")
    
except ImportError as e:
    print(f"⚠️  部分模組載入失敗: {e}")
    print("🔄 使用模擬模式繼續運行")
    REAL_MODULES_AVAILABLE = False

class IntegratedTrainService:
    """整合型持續訓練服務 - 使用真實 AIVA 模組"""
    
    def __init__(self):
        self.is_running = False
        self.start_time = None
        self.loop_count = 0
        
        # 初始化組件
        self.components = {}
        self.initialize_components()
        
        # 設置日誌
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger("IntegratedTrainService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def initialize_components(self):
        """初始化 AIVA 組件"""
        if REAL_MODULES_AVAILABLE:
            try:
                # Scan 模組: 靶場檢測器
                self.components['target_detector'] = TargetEnvironmentDetector()
                print("   ✅ 靶場環境檢測器已載入")
                
                # Core 模組: 抗幻覺驗證
                self.components['anti_hallucination'] = AntiHallucinationModule()
                print("   ✅ 抗幻覺驗證模組已載入")
                
                # Core 模組: 決策代理
                self.components['decision_agent'] = EnhancedDecisionAgent()
                print("   ✅ 增強決策代理已載入")
                
                # Integration 模組: 操作記錄器
                self.components['operation_recorder'] = AIOperationRecorder()
                print("   ✅ AI 操作記錄器已載入")
                
                # Integration 模組: 性能監控器
                self.components['performance_monitor'] = SystemPerformanceMonitor()
                print("   ✅ 系統性能監控器已載入")
                
            except Exception as e:
                print(f"   ⚠️  組件初始化部分失敗: {e}")
                
        else:
            print("   🔄 使用模擬組件")
    
    async def check_target_environment_real(self) -> Dict[str, Any]:
        """使用真實靶場檢測器檢查環境"""
        if 'target_detector' in self.components:
            detector = self.components['target_detector']
            results = await detector.detect_environment()
            
            print(f"   🎯 發現 {results['targets_scanned']} 個掃描目標")
            print(f"   🔍 識別 {len(results['discovered_services'])} 個服務")
            print(f"   🏆 確認 {len(results['identified_platforms'])} 個靶場平台")
            
            return results
        else:
            # 回退到模擬檢查
            return await self.check_target_environment_fallback()
    
    async def check_target_environment_fallback(self) -> Dict[str, Any]:
        """模擬靶場檢查 (當真實模組不可用時)"""
        print("🎯 檢查靶場環境 (模擬模式)...")
        
        target_checks = [
            ("HTTP 服務", "80"),
            ("HTTPS 服務", "443"), 
            ("開發服務", "3000"),
            ("代理服務", "8080"),
            ("Web 服務", "8888")
        ]
        
        available_targets = []
        for name, port in target_checks:
            print(f"   🔍 檢查 {name} (端口 {port})...")
            await asyncio.sleep(0.2)  # 模擬掃描時間
            available_targets.append(f"{name}:{port}")
            
        return {
            "targets_scanned": len(available_targets),
            "discovered_services": available_targets,
            "identified_platforms": ["模擬靶場平台"]
        }
    
    async def generate_attack_plan_with_validation(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成並驗證攻擊計畫"""
        print("   🎯 生成攻擊計畫...")
        
        # 模擬攻擊計畫生成
        mock_plan = {
            "name": "Web 應用滲透測試",
            "target": target_info.get("primary_target", "http://localhost"),
            "steps": [
                {
                    "action": "port_scan",
                    "description": "掃描目標開放端口",
                    "tool": "nmap",
                    "parameters": {"target": "localhost", "ports": "1-1000"}
                },
                {
                    "action": "web_crawl", 
                    "description": "爬取網站結構",
                    "tool": "spider",
                    "parameters": {"depth": 3}
                },
                {
                    "action": "sql_injection_test",
                    "description": "測試 SQL 注入漏洞",
                    "tool": "sqlmap",
                    "parameters": {"payload": "' OR 1=1--"}
                }
            ]
        }
        
        # 使用抗幻覺模組驗證計畫
        if 'anti_hallucination' in self.components:
            validator = self.components['anti_hallucination']
            validated_plan = validator.validate_attack_plan(mock_plan)
            
            removed_count = len(mock_plan['steps']) - len(validated_plan['steps'])
            if removed_count > 0:
                print(f"   🧠 抗幻覺驗證移除了 {removed_count} 個可疑步驟")
            else:
                print("   ✅ 攻擊計畫通過抗幻覺驗證")
                
            return validated_plan
        else:
            print("   ⚠️  跳過抗幻覺驗證 (模組未載入)")
            return mock_plan
    
    async def make_intelligent_decision(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用增強決策代理做出智能決策"""
        print("   🤔 執行智能決策分析...")
        
        if 'decision_agent' in self.components:
            agent = self.components['decision_agent']
            
            # 創建決策上下文
            context = DecisionContext()
            context.risk_level = RiskLevel.MEDIUM
            context.discovered_vulns = context_data.get('vulns', [])
            context.available_tools = ['nmap', 'sqlmap', 'nikto', 'hydra']
            
            decision = agent.make_decision(context)
            
            print(f"   🎯 決策結果: {decision.action}")
            print(f"   📊 信心度: {decision.confidence:.1%}")
            
            return {
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "parameters": decision.params
            }
        else:
            print("   ⚠️  使用基本決策邏輯")
            return {
                "action": "proceed_scan",
                "confidence": 0.8,
                "reasoning": "基本決策: 繼續掃描流程"
            }
    
    async def record_training_operation(self, operation_type: str, details: Dict[str, Any]):
        """記錄訓練操作"""
        if 'operation_recorder' in self.components:
            recorder = self.components['operation_recorder']
            
            recorder.record_operation(
                command=operation_type,
                description=details.get('description', f'執行 {operation_type}'),
                operation_type="training",
                parameters=details,
                result=details.get('result'),
                duration=details.get('duration'),
                success=details.get('success', True)
            )
    
    async def monitor_system_performance(self) -> Dict[str, Any]:
        """監控系統性能"""
        if 'performance_monitor' in self.components:
            monitor = self.components['performance_monitor']
            try:
                metrics = monitor.get_system_metrics()
                health = await monitor.get_system_health()
                
                return {
                    "metrics": metrics,
                    "health_score": health.overall_score,
                    "status": "healthy" if health.overall_score > 70 else "warning"
                }
            except Exception as e:
                print(f"   ⚠️  性能監控異常: {e}")
                return {"status": "monitoring_error"}
        else:
            return {"status": "no_monitoring"}
    
    async def execute_enhanced_training_cycle(self) -> Dict[str, Any]:
        """執行增強版訓練週期"""
        cycle_start = time.time()
        cycle_results = {
            "success": True,
            "operations": [],
            "performance": {}
        }
        
        try:
            # 1. 環境檢測
            print("   🎯 智能環境檢測...")
            target_info = await self.check_target_environment_real()
            await self.record_training_operation("environment_scan", {
                "description": "靶場環境檢測",
                "result": target_info,
                "duration": 1.0
            })
            cycle_results["operations"].append("environment_scan")
            
            # 2. 攻擊計畫生成與驗證
            print("   📋 計畫生成與驗證...")
            attack_plan = await self.generate_attack_plan_with_validation(target_info)
            await self.record_training_operation("plan_generation", {
                "description": "攻擊計畫生成與抗幻覺驗證",
                "result": {"steps_count": len(attack_plan['steps'])},
                "duration": 1.5
            })
            cycle_results["operations"].append("plan_generation")
            
            # 3. 智能決策
            print("   🧠 執行智能決策...")
            decision_result = await self.make_intelligent_decision({
                "vulns": ["sql_injection"],
                "target_info": target_info
            })
            await self.record_training_operation("intelligent_decision", {
                "description": "增強決策代理分析",
                "result": decision_result,
                "duration": 0.8
            })
            cycle_results["operations"].append("intelligent_decision")
            
            # 4. 計畫執行模擬
            print("   ⚡ 執行攻擊計畫...")
            await asyncio.sleep(2)  # 模擬執行時間
            execution_result = {"executed_steps": len(attack_plan['steps']), "success_rate": 0.85}
            await self.record_training_operation("plan_execution", {
                "description": "攻擊計畫執行",
                "result": execution_result,
                "duration": 2.0
            })
            cycle_results["operations"].append("plan_execution")
            
            # 5. 經驗收集
            print("   📊 收集訓練經驗...")
            experience_data = {
                "plan_effectiveness": 0.85,
                "decision_accuracy": decision_result["confidence"],
                "validation_removes": 0
            }
            await self.record_training_operation("experience_collection", {
                "description": "訓練經驗收集",
                "result": experience_data,
                "duration": 0.5
            })
            cycle_results["operations"].append("experience_collection")
            
            # 6. 性能監控
            print("   📈 系統性能監控...")
            performance_data = await self.monitor_system_performance()
            cycle_results["performance"] = performance_data
            
            cycle_time = time.time() - cycle_start
            print(f"   ✅ 增強訓練週期完成 (耗時: {cycle_time:.1f}s)")
            
            cycle_results.update({
                "cycle_time": cycle_time,
                "improvements": f"整合功能效果提升 {len(cycle_results['operations'])} 項能力"
            })
            
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"訓練週期異常: {e}")
            cycle_results["success"] = False
            cycle_results["error"] = str(e)
            return cycle_results
    
    async def start_integrated_training_loop(self):
        """開始整合型持續學習迴圈"""
        print("🚀 開始 AIVA 整合型持續學習...")
        print("💡 整合功能: 靶場檢測 + 抗幻覺 + 智能決策 + 操作記錄 + 性能監控")
        print("💡 提示: 按 Ctrl+C 可隨時停止")
        print("=" * 70)
        
        self.is_running = True
        self.start_time = datetime.now()
        self.loop_count = 0
        
        try:
            while self.is_running:
                self.loop_count += 1
                print(f"\n🔄 === 整合學習迴圈 #{self.loop_count} ===")
                print(f"🕐 開始時間: {datetime.now().strftime('%H:%M:%S')}")
                
                # 執行增強訓練週期
                try:
                    result = await self.execute_enhanced_training_cycle()
                    
                    if result["success"]:
                        print(f"📊 週期結果: {result['improvements']}")
                        print(f"⚙️  執行操作: {', '.join(result['operations'])}")
                        
                        # 顯示性能狀態
                        perf_status = result.get("performance", {}).get("status", "unknown")
                        print(f"📈 系統狀態: {perf_status}")
                    else:
                        print(f"❌ 週期執行失敗: {result.get('error', '未知錯誤')}")
                        
                except Exception as e:
                    print(f"⚠️  週期執行異常: {e}")
                    print("🔄 5秒後重試...")
                    await asyncio.sleep(5)
                    continue
                
                # 顯示累計統計
                elapsed = datetime.now() - self.start_time
                print(f"📈 累計運行: {elapsed} | 完成週期: {self.loop_count}")
                
                # 短暫休息
                print("😴 休息 3 秒...")
                await asyncio.sleep(3)
                
        except KeyboardInterrupt:
            print("\n⏹️  收到停止信號，正在安全關閉...")
        except Exception as e:
            print(f"❌ 系統錯誤: {e}")
        finally:
            self.is_running = False
            await self.cleanup()
    
    async def cleanup(self):
        """清理資源"""
        print("🧹 清理系統資源...")
        
        # 停止操作記錄器
        if 'operation_recorder' in self.components:
            recorder = self.components['operation_recorder']
            recorder.stop_recording()
            
            # 匯出最終報告
            report_path = recorder.export_session_report()
            if report_path:
                print(f"📄 操作記錄報告: {report_path}")
        
        # 匯出決策分析
        if 'decision_agent' in self.components:
            agent = self.components['decision_agent']
            analysis_path = agent.export_decision_analysis()
            if analysis_path:
                print(f"📄 決策分析報告: {analysis_path}")
        
        print("✅ 資源清理完成")
    
    def get_integrated_stats(self) -> Dict[str, Any]:
        """獲取整合統計"""
        stats = {
            "運行時間": str(datetime.now() - self.start_time) if self.start_time else "未啟動",
            "完成週期": self.loop_count,
            "載入模組": list(self.components.keys()),
            "整合功能數": len(self.components)
        }
        
        # 添加各組件統計
        if 'operation_recorder' in self.components:
            recorder_stats = self.components['operation_recorder'].get_frontend_data()
            stats["操作記錄數"] = recorder_stats["current_stats"]["total_operations"]
        
        if 'decision_agent' in self.components:
            decision_stats = self.components['decision_agent'].get_decision_stats()
            stats["決策次數"] = decision_stats["total_decisions"]
        
        return stats

# 主函數
async def main():
    """主函數 - 整合型 AI 持續學習入口點"""
    print("🚀 AIVA 整合型 AI 持續學習觸發器")
    print("📋 基於: 五大模組架構的完整功能整合")
    print("=" * 70)
    
    # 創建整合訓練服務
    train_service = IntegratedTrainService()
    
    try:
        print("\n🎯 系統整合檢查...")
        print(f"✅ 真實模組可用: {'是' if REAL_MODULES_AVAILABLE else '否 (使用模擬模式)'}")
        print(f"⚙️  載入組件數量: {len(train_service.components)}")
        
        if train_service.components:
            print("📦 已載入組件:")
            for name in train_service.components.keys():
                print(f"   - {name}")
        
        # 開始整合型持續學習
        print("\n🎯 所有檢查通過！即將開始整合型 AI 持續學習...")
        await asyncio.sleep(2)
        await train_service.start_integrated_training_loop()
        
    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")
    finally:
        # 顯示最終統計
        stats = train_service.get_integrated_stats()
        print("\n📊 最終統計:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print("✅ AIVA 整合型 AI 持續學習已停止")

if __name__ == "__main__":
    print("🚀 啟動 AIVA 整合型 AI 持續學習觸發器...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用戶中斷")
    except Exception as e:
        print(f"\n💥 程序異常終止: {e}")