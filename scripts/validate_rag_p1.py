#!/usr/bin/env python3
"""
RAG P1 靶場驗證 - 使用 AIVA 系統本身的攻擊能力

按照系統設計：
1. 使用 AttackCoordinator 協調攻擊
2. 透過 RAG 決策引擎自動選擇攻擊流程
3. FlowExecutorAdapter 執行真實攻擊
4. 記錄所有執行結果和能力表現
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from services.core.aiva_core.service_backbone.attack_coordinator import AttackCoordinator
from services.core.aiva_core.service_backbone.protocol_models import (
    AttackRequest, AttackMode, TargetType
)


class RAG_P1_Validator:
    """使用真實系統能力進行 P1 驗證"""
    
    def __init__(self):
        self.coordinator = AttackCoordinator()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "testgrounds": {},
            "summary": {
                "total_attacks": 0,
                "successful_attacks": 0,
                "failed_attacks": 0,
                "capabilities_tested": set()
            }
        }
        
    def log(self, message: str):
        """輸出日誌"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    async def attack_target(
        self,
        target_name: str,
        target_url: str,
        capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        對單一靶場發起攻擊
        
        Args:
            target_name: 靶場名稱
            target_url: 靶場 URL
            capabilities: 要測試的能力列表 (如 ["xss", "sqli", "ssrf"])
        """
        self.log(f"🎯 開始攻擊: {target_name}")
        self.log(f"   目標: {target_url}")
        self.log(f"   能力: {', '.join(capabilities)}")
        
        testground_results = {
            "target_url": target_url,
            "capabilities": {},
            "total_attempts": 0,
            "successful_attempts": 0
        }
        
        for capability in capabilities:
            self.log(f"\n   📍 測試能力: {capability.upper()}")
            
            # 構建攻擊請求 - 讓 RAG 系統自動決策
            attack_request = AttackRequest(
                target=target_url,
                target_type=TargetType.WEB_APPLICATION,
                attack_mode=AttackMode.INTELLIGENT,  # 使用智能模式讓 RAG 決策
                capabilities=[capability],
                max_depth=3,
                timeout=120,
                user_id="rag_p1_validation",
                session_id=f"p1_val_{target_name}_{capability}"
            )
            
            try:
                # 使用 AttackCoordinator 執行真實攻擊
                result = await self.coordinator.execute_attack(attack_request)
                
                testground_results["total_attempts"] += 1
                self.results["summary"]["total_attacks"] += 1
                self.results["summary"]["capabilities_tested"].add(capability)
                
                # 分析結果
                success = self._analyze_result(result)
                
                if success:
                    testground_results["successful_attempts"] += 1
                    self.results["summary"]["successful_attacks"] += 1
                    self.log(f"      ✅ 成功 - 發現漏洞或完成攻擊")
                else:
                    self.results["summary"]["failed_attacks"] += 1
                    self.log(f"      ⚠️ 未成功 - 無漏洞或執行失敗")
                
                # 記錄詳細結果
                testground_results["capabilities"][capability] = {
                    "success": success,
                    "flows_executed": result.get("flows_executed", 0),
                    "vulnerabilities_found": result.get("vulnerabilities", []),
                    "execution_time": result.get("execution_time", 0),
                    "error": result.get("error")
                }
                
            except Exception as e:
                self.log(f"      ❌ 執行錯誤: {str(e)}")
                testground_results["capabilities"][capability] = {
                    "success": False,
                    "error": str(e)
                }
                self.results["summary"]["failed_attacks"] += 1
        
        self.results["testgrounds"][target_name] = testground_results
        return testground_results
    
    def _analyze_result(self, result: Dict[str, Any]) -> bool:
        """分析攻擊結果是否成功"""
        # 成功條件：
        # 1. 發現了漏洞
        # 2. 或者執行了流程且沒有錯誤
        if result.get("vulnerabilities"):
            return True
        if result.get("flows_executed", 0) > 0 and not result.get("error"):
            return True
        return False
    
    async def run_validation(self):
        """執行完整 P1 驗證"""
        self.log("=" * 70)
        self.log("🚀 RAG P1 靶場驗證 - 使用 AIVA 真實攻擊能力")
        self.log("=" * 70)
        
        # 檢查 RAG 系統狀態
        if not self.coordinator.rag_available:
            self.log("❌ RAG 系統未就緒")
            return
        
        self.log(f"✅ RAG 系統已就緒")
        self.log(f"   已載入 {len(self.coordinator.cli_engine.all_flows)} 個攻擊流程")
        self.log("")
        
        # 定義靶場配置
        testgrounds = [
            {
                "name": "WebGoat",
                "url": "http://localhost:8080/WebGoat",
                "capabilities": ["xss", "sqli", "ssrf"]
            },
            {
                "name": "JuiceShop_3000",
                "url": "http://localhost:3000",
                "capabilities": ["xss", "sqli", "idor"]
            },
            {
                "name": "JuiceShop_3001",
                "url": "http://localhost:3001",
                "capabilities": ["xss", "sqli"]
            },
            {
                "name": "JuiceShop_3003",
                "url": "http://localhost:3003",
                "capabilities": ["xss", "sqli"]
            }
        ]
        
        # 對每個靶場執行攻擊
        for i, tg in enumerate(testgrounds, 1):
            self.log(f"\n{'=' * 70}")
            self.log(f"靶場 {i}/{len(testgrounds)}: {tg['name']}")
            self.log("=" * 70)
            
            await self.attack_target(
                target_name=tg["name"],
                target_url=tg["url"],
                capabilities=tg["capabilities"]
            )
            
            # 短暫延遲避免過載
            await asyncio.sleep(2)
        
        # 輸出總結
        self._print_summary()
        
        # 保存結果
        self._save_results()
    
    def _print_summary(self):
        """輸出驗證總結"""
        summary = self.results["summary"]
        
        self.log("\n" + "=" * 70)
        self.log("📊 RAG P1 驗證總結")
        self.log("=" * 70)
        
        self.log(f"\n攻擊統計:")
        self.log(f"  • 總攻擊次數: {summary['total_attacks']}")
        self.log(f"  • 成功次數: {summary['successful_attacks']}")
        self.log(f"  • 失敗次數: {summary['failed_attacks']}")
        
        if summary['total_attacks'] > 0:
            success_rate = (summary['successful_attacks'] / summary['total_attacks']) * 100
            self.log(f"  • 成功率: {success_rate:.1f}%")
        
        self.log(f"\n測試能力: {', '.join(sorted(summary['capabilities_tested']))}")
        
        # P1 成功標準檢查
        self.log("\n" + "=" * 70)
        self.log("P1 成功標準檢查:")
        self.log("=" * 70)
        
        capable_count = len([
            cap for cap, results in self._group_by_capability().items()
            if results['success_count'] > 0
        ])
        
        check1 = capable_count >= 3
        check2 = (summary['failed_attacks'] / max(summary['total_attacks'], 1)) < 0.1
        
        self.log(f"  {'✅' if check1 else '❌'} 標準1: 至少 3 種能力成功 (實際: {capable_count})")
        self.log(f"  {'✅' if check2 else '❌'} 標準2: 錯誤率 < 10% (實際: {(summary['failed_attacks']/max(summary['total_attacks'],1))*100:.1f}%)")
        
        if check1 and check2:
            self.log("\n🎉 P1 驗證通過！")
        else:
            self.log("\n⚠️ P1 驗證未完全通過，需要改進")
    
    def _group_by_capability(self) -> Dict[str, Dict[str, int]]:
        """按能力分組統計"""
        capability_stats = {}
        
        for tg_name, tg_results in self.results["testgrounds"].items():
            for cap, cap_result in tg_results["capabilities"].items():
                if cap not in capability_stats:
                    capability_stats[cap] = {"success_count": 0, "total_count": 0}
                
                capability_stats[cap]["total_count"] += 1
                if cap_result.get("success"):
                    capability_stats[cap]["success_count"] += 1
        
        return capability_stats
    
    def _save_results(self):
        """保存驗證結果"""
        # 轉換 set 為 list 以便 JSON 序列化
        self.results["summary"]["capabilities_tested"] = sorted(
            list(self.results["summary"]["capabilities_tested"])
        )
        
        # 保存到 reports 目錄
        report_dir = Path("reports/rag_validation")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"rag_p1_real_attack_{timestamp}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n💾 結果已保存: {report_file}")


async def main():
    """主函數"""
    validator = RAG_P1_Validator()
    await validator.run_validation()


if __name__ == "__main__":
    asyncio.run(main())
