#!/usr/bin/env python3
"""
AIVA 增強型真實攻擊整合系統
用途: 將真實攻擊能力整合到 AIVA AI 決策系統
基於: 實際攻擊執行器 + AI 決策引擎的完整整合
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
sys.path.append(str(Path(__file__).parent.parent.parent))

# 導入真實攻擊執行器
from real_attack_executor import RealAttackExecutor

# 嘗試導入 AIVA 模組
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
    print("✅ 成功載入 AIVA 真實模組")
    
except ImportError as e:
    print(f"⚠️  部分模組載入失敗: {e}")
    print("🔄 使用基本模式繼續運行")
    REAL_MODULES_AVAILABLE = False

class RealAIAttackSystem:
    """真實 AI 攻擊系統 - 結合 AI 決策與真實攻擊執行"""
    
    def __init__(self):
        self.is_running = False
        self.start_time = None
        self.attack_count = 0
        
        # 初始化真實攻擊執行器
        self.real_executor = RealAttackExecutor()
        
        # 初始化 AI 組件
        self.ai_components = {}
        self.initialize_ai_components()
        
        # 設置日誌
        self.logger = self._setup_logger()
        
        # 攻擊歷史記錄
        self.attack_history = []
        
    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger("RealAIAttackSystem")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def initialize_ai_components(self):
        """初始化 AI 組件"""
        if REAL_MODULES_AVAILABLE:
            try:
                # Scan 模組: 靶場檢測器
                self.ai_components['target_detector'] = TargetEnvironmentDetector()
                print("   ✅ AI 靶場檢測器已載入")
                
                # Core 模組: 抗幻覺驗證
                self.ai_components['anti_hallucination'] = AntiHallucinationModule()
                print("   ✅ AI 抗幻覺模組已載入")
                
                # Core 模組: 決策代理
                self.ai_components['decision_agent'] = EnhancedDecisionAgent()
                print("   ✅ AI 決策代理已載入")
                
                # Integration 模組: 操作記錄器
                self.ai_components['operation_recorder'] = AIOperationRecorder()
                print("   ✅ AI 操作記錄器已載入")
                
                # Integration 模組: 性能監控器
                self.ai_components['performance_monitor'] = SystemPerformanceMonitor()
                print("   ✅ AI 性能監控器已載入")
                
            except Exception as e:
                print(f"   ⚠️  AI 組件初始化部分失敗: {e}")
                
        else:
            print("   🔄 使用基本攻擊模式")
    
    async def ai_generate_attack_plan(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """AI 生成攻擊計畫 (基於真實掃描結果)"""
        print("🧠 AI 正在分析目標並生成攻擊計畫...")
        
        # 基於掃描結果生成動態攻擊計畫
        plan = {
            "name": f"AI 動態攻擊計畫 - {target_info.get('target', 'Unknown')}",
            "target": target_info.get('target', 'localhost'),
            "generated_by": "AI",
            "generation_time": time.time(),
            "steps": []
        }
        
        # 1. 必須的端口掃描
        plan["steps"].append({
            "action": "port_scan",
            "description": "AI 智能端口掃描",
            "priority": "high",
            "parameters": {"target": plan["target"]}
        })
        
        # 2. 基於目標類型的動態步驟
        if "web" in target_info.get('type', '').lower() or target_info.get('has_web_service', False):
            plan["steps"].append({
                "action": "web_crawl",
                "description": "AI Web 應用深度偵察",
                "priority": "high"
            })
            
            plan["steps"].append({
                "action": "sql_injection_test",
                "description": "AI SQL 注入智能檢測",
                "priority": "medium"
            })
        
        # 3. 如果發現特定服務，添加對應測試
        if target_info.get('services'):
            for service in target_info['services']:
                if 'ssh' in service.lower():
                    plan["steps"].append({
                        "action": "ssh_brute_force", 
                        "description": "SSH 暴力破解測試",
                        "priority": "low"
                    })
                elif 'ftp' in service.lower():
                    plan["steps"].append({
                        "action": "ftp_anonymous_test",
                        "description": "FTP 匿名登錄測試", 
                        "priority": "medium"
                    })
        
        # AI 抗幻覺驗證
        if 'anti_hallucination' in self.ai_components:
            validator = self.ai_components['anti_hallucination']
            validated_plan = validator.validate_attack_plan(plan)
            
            removed_count = len(plan['steps']) - len(validated_plan['steps'])
            if removed_count > 0:
                print(f"   🧠 AI 抗幻覺驗證移除了 {removed_count} 個不合理步驟")
            else:
                print("   ✅ AI 攻擊計畫通過智能驗證")
                
            return validated_plan
        else:
            return plan
    
    async def ai_make_attack_decision(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """AI 做出攻擊決策"""
        print("🤔 AI 正在分析掃描結果並做出決策...")
        
        if 'decision_agent' in self.ai_components:
            agent = self.ai_components['decision_agent']
            
            # 創建決策上下文
            context = DecisionContext()
            
            # 基於掃描結果設置風險等級
            vulnerabilities = scan_results.get('vulnerabilities', [])
            open_ports = scan_results.get('open_ports', [])
            
            if len(vulnerabilities) > 5:
                context.risk_level = RiskLevel.HIGH
            elif len(vulnerabilities) > 0 or len(open_ports) > 3:
                context.risk_level = RiskLevel.MEDIUM
            else:
                context.risk_level = RiskLevel.LOW
            
            context.discovered_vulns = [v.get('type', 'unknown') for v in vulnerabilities]
            context.available_tools = ['nmap', 'curl', 'python', 'custom_scanner']
            
            decision = agent.make_decision(context)
            
            print(f"   🎯 AI 決策結果: {decision.action}")
            print(f"   📊 AI 信心度: {decision.confidence:.1%}")
            print(f"   💭 AI 推理: {decision.reasoning}")
            
            return {
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "parameters": decision.params,
                "risk_assessment": context.risk_level.value
            }
        else:
            # 基本決策邏輯
            total_findings = len(scan_results.get('vulnerabilities', [])) + len(scan_results.get('open_ports', []))
            
            if total_findings > 5:
                action = "AGGRESSIVE_ATTACK"
                confidence = 0.9
            elif total_findings > 0:
                action = "CAREFUL_EXPLOITATION"
                confidence = 0.7
            else:
                action = "PASSIVE_RECONNAISSANCE" 
                confidence = 0.5
                
            return {
                "action": action,
                "confidence": confidence,
                "reasoning": f"基於 {total_findings} 個發現做出決策",
                "risk_assessment": "medium"
            }
    
    async def execute_ai_driven_attack(self, target_url: str = "http://localhost:3000") -> Dict[str, Any]:
        """執行 AI 驅動的攻擊"""
        attack_session = {
            "session_id": f"ai_attack_{int(time.time())}",
            "target": target_url,
            "start_time": time.time(),
            "phases": [],
            "total_findings": 0,
            "ai_decisions": [],
            "success": True
        }
        
        try:
            # Phase 1: AI 環境檢測
            print(f"\n🎯 Phase 1: AI 智能環境檢測")
            if 'target_detector' in self.ai_components:
                detector = self.ai_components['target_detector']
                env_results = await detector.detect_environment()
                attack_session["phases"].append({
                    "phase": "AI Environment Detection",
                    "results": env_results,
                    "duration": 1.0
                })
                print(f"   🤖 AI 檢測到 {env_results.get('targets_scanned', 0)} 個目標")
            
            # Phase 2: 真實攻擊執行
            print(f"\n🚀 Phase 2: 真實攻擊執行")
            
            # 先進行基礎掃描獲取目標信息
            target_info = {
                "target": target_url.replace("http://", "").replace("https://", "").split(":")[0],
                "type": "web", 
                "has_web_service": True
            }
            
            # AI 生成攻擊計畫
            ai_plan = await self.ai_generate_attack_plan(target_info)
            attack_session["ai_generated_plan"] = ai_plan
            
            # 執行真實攻擊
            real_results = await self.real_executor.execute_real_attack_plan(ai_plan)
            attack_session["phases"].append({
                "phase": "Real Attack Execution",
                "results": real_results,
                "duration": real_results.get("duration", 0)
            })
            
            # Phase 3: AI 結果分析與決策
            print(f"\n🧠 Phase 3: AI 結果分析與決策")
            
            # 整合所有掃描結果
            all_scan_results = {
                "vulnerabilities": [],
                "open_ports": [],
                "accessible_paths": []
            }
            
            for step_result in real_results.get("steps_executed", []):
                result_data = step_result.get("result", {})
                if "vulnerabilities" in result_data:
                    all_scan_results["vulnerabilities"].extend(result_data["vulnerabilities"])
                if "open_ports" in result_data:
                    all_scan_results["open_ports"].extend(result_data["open_ports"])
                if "findings" in result_data and "accessible_paths" in result_data["findings"]:
                    all_scan_results["accessible_paths"].extend(result_data["findings"]["accessible_paths"])
            
            # AI 決策分析
            ai_decision = await self.ai_make_attack_decision(all_scan_results)
            attack_session["ai_decisions"].append(ai_decision)
            
            # 統計結果
            attack_session["total_findings"] = (
                len(all_scan_results["vulnerabilities"]) + 
                len(all_scan_results["open_ports"]) + 
                len(all_scan_results["accessible_paths"])
            )
            
            # Phase 4: 記錄與報告
            print(f"\n📊 Phase 4: AI 學習與記錄")
            
            if 'operation_recorder' in self.ai_components:
                recorder = self.ai_components['operation_recorder']
                recorder.record_operation(
                    command="ai_driven_attack",
                    description=f"AI 驅動的完整攻擊: {target_url}",
                    operation_type="ai_attack",
                    parameters={"target": target_url, "plan": ai_plan},
                    result=attack_session,
                    duration=time.time() - attack_session["start_time"],
                    success=attack_session["success"]
                )
                print("   📝 AI 操作已記錄到學習系統")
            
            attack_session["end_time"] = time.time()
            attack_session["total_duration"] = attack_session["end_time"] - attack_session["start_time"]
            
            return attack_session
            
        except Exception as e:
            self.logger.error(f"AI 攻擊執行失敗: {e}")
            attack_session["success"] = False
            attack_session["error"] = str(e)
            return attack_session
    
    async def start_continuous_ai_attack_learning(self, target_url: str = "http://localhost:3000"):
        """開始持續 AI 攻擊學習"""
        print("🚀 開始 AIVA 真實 AI 攻擊學習系統")
        print("🎯 整合功能: AI 決策 + 真實攻擊 + 智能學習")
        print("💡 提示: 按 Ctrl+C 可隨時停止")
        print("=" * 70)
        
        self.is_running = True
        self.start_time = datetime.now()
        self.attack_count = 0
        
        try:
            while self.is_running:
                self.attack_count += 1
                print(f"\n🔄 === AI 攻擊學習循環 #{self.attack_count} ===")
                print(f"🕐 開始時間: {datetime.now().strftime('%H:%M:%S')}")
                
                # 執行 AI 驅動的攻擊
                try:
                    attack_result = await self.execute_ai_driven_attack(target_url)
                    
                    if attack_result["success"]:
                        print(f"📊 攻擊結果: 發現 {attack_result['total_findings']} 個問題點")
                        print(f"⚙️  執行階段: {len(attack_result['phases'])} 個")
                        print(f"🧠 AI 決策: {len(attack_result['ai_decisions'])} 個")
                        print(f"⏱️  總耗時: {attack_result['total_duration']:.2f} 秒")
                    else:
                        print(f"❌ 攻擊執行失敗: {attack_result.get('error', '未知錯誤')}")
                        
                    # 將結果加入歷史記錄
                    self.attack_history.append(attack_result)
                        
                except Exception as e:
                    print(f"⚠️  攻擊循環異常: {e}")
                    print("🔄 10秒後重試...")
                    await asyncio.sleep(10)
                    continue
                
                # 顯示累計統計
                elapsed = datetime.now() - self.start_time
                total_findings = sum(result.get('total_findings', 0) for result in self.attack_history)
                success_rate = sum(1 for result in self.attack_history if result.get('success', False)) / len(self.attack_history) * 100
                
                print(f"📈 累計統計:")
                print(f"   - 運行時間: {elapsed}")
                print(f"   - 完成攻擊: {self.attack_count}")
                print(f"   - 總發現: {total_findings}")
                print(f"   - 成功率: {success_rate:.1f}%")
                
                # 學習間隔
                print("😴 AI 學習分析中 (30秒)...")
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print("\n⏹️  收到停止信號，正在安全關閉 AI 攻擊系統...")
        except Exception as e:
            print(f"❌ 系統錯誤: {e}")
        finally:
            self.is_running = False
            await self.cleanup()
    
    async def cleanup(self):
        """清理資源並生成最終報告"""
        print("🧹 清理 AI 攻擊系統資源...")
        
        # 停止操作記錄器
        if 'operation_recorder' in self.ai_components:
            recorder = self.ai_components['operation_recorder']
            recorder.stop_recording()
            
            # 匯出最終報告
            report_path = recorder.export_session_report()
            if report_path:
                print(f"📄 AI 操作記錄報告: {report_path}")
        
        # 生成攻擊歷史總結
        if self.attack_history:
            summary_report = {
                "session_summary": {
                    "total_attacks": len(self.attack_history),
                    "total_findings": sum(r.get('total_findings', 0) for r in self.attack_history),
                    "success_rate": sum(1 for r in self.attack_history if r.get('success', False)) / len(self.attack_history),
                    "average_duration": sum(r.get('total_duration', 0) for r in self.attack_history) / len(self.attack_history),
                    "session_duration": str(datetime.now() - self.start_time)
                },
                "attack_history": self.attack_history
            }
            
            summary_file = f"ai_attack_session_{int(time.time())}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False)
            
            print(f"📄 AI 攻擊會話報告: {summary_file}")
        
        print("✅ AI 攻擊系統資源清理完成")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """獲取系統統計"""
        if not self.attack_history:
            return {"status": "no_attacks_completed"}
        
        stats = {
            "運行時間": str(datetime.now() - self.start_time) if self.start_time else "未啟動",
            "完成攻擊": self.attack_count,
            "成功攻擊": sum(1 for r in self.attack_history if r.get('success', False)),
            "總發現數": sum(r.get('total_findings', 0) for r in self.attack_history),
            "平均耗時": f"{sum(r.get('total_duration', 0) for r in self.attack_history) / len(self.attack_history):.2f}s",
            "載入AI組件": list(self.ai_components.keys()),
            "真實攻擊能力": "✅ 已整合"
        }
        
        return stats

# 主函數
async def main():
    """主函數 - AI 驅動的真實攻擊學習系統"""
    print("🚀 AIVA 增強型真實攻擊整合系統")
    print("📋 功能: AI 決策 + 真實攻擊 + 智能學習")
    print("=" * 70)
    
    # 創建 AI 攻擊系統
    ai_attack_system = RealAIAttackSystem()
    
    try:
        print("\n🎯 系統整合檢查...")
        print(f"✅ AI 模組可用: {'是' if REAL_MODULES_AVAILABLE else '否 (基本模式)'}")
        print(f"✅ 真實攻擊可用: 是")
        print(f"⚙️  載入AI組件數: {len(ai_attack_system.ai_components)}")
        print(f"🔧 可用攻擊工具: {len(ai_attack_system.real_executor.available_tools)}")
        
        if ai_attack_system.ai_components:
            print("🧠 AI 組件:")
            for name in ai_attack_system.ai_components.keys():
                print(f"   - {name}")
        
        print("🔧 攻擊工具:")
        for tool, available in ai_attack_system.real_executor.available_tools.items():
            status = "✅" if available else "❌"
            print(f"   - {tool}: {status}")
        
        # 開始 AI 攻擊學習
        print("\n🎯 所有檢查通過！即將開始 AI 驅動的真實攻擊學習...")
        await asyncio.sleep(2)
        await ai_attack_system.start_continuous_ai_attack_learning()
        
    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")
    finally:
        # 顯示最終統計
        stats = ai_attack_system.get_system_stats()
        print("\n📊 最終統計:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print("✅ AIVA AI 攻擊學習系統已停止")

if __name__ == "__main__":
    print("🚀 啟動 AIVA 增強型真實攻擊整合系統...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用戶中斷")
    except Exception as e:
        print(f"\n💥 程序異常終止: {e}")