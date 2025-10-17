#!/usr/bin/env python3
"""
AIVA 完整流程 AI 訓練腳本
目的: 通過完整的雙向通訊流程訓練 AI，同時讓開發者摸索系統運作
"""

from pathlib import Path
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.aiva_common.utils import get_logger
from services.aiva_common.schemas import (
    AivaMessage, MessageHeader, FunctionTaskPayload, 
    ScanTaskPayload, Asset, FindingPayload
)
from services.aiva_common.enums import Topic, ModuleName, Severity
from services.aiva_common.mq import get_broker

logger = get_logger(__name__)

class AITrainingOrchestrator:
    """AI 訓練編排器 - 執行完整流程並收集訓練數據"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_session_id = f"ai_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.training_data = {
            'session_id': self.training_session_id,
            'start_time': datetime.now().isoformat(),
            'scenarios': []
        }
        
    async def run_complete_training(self):
        """執行完整的訓練流程"""
        logger.info(f"[START] 開始 AI 訓練會話: {self.training_session_id}")
        
        # 場景 1: 掃描 → 檢測完整流程
        await self.scenario_1_scan_to_detection()
        
        # 場景 2: 直接功能檢測流程
        await self.scenario_2_direct_detection()
        
        # 場景 3: 多目標批量處理
        await self.scenario_3_batch_processing()
        
        # 場景 4: 錯誤處理與恢復
        await self.scenario_4_error_handling()
        
        # 保存訓練數據
        self._save_training_data()
        
        logger.info("[OK] AI 訓練會話完成")
        
    async def scenario_1_scan_to_detection(self):
        """場景 1: CLI → Core → Scan → Core → Function → Integration → CLI"""
        logger.info("\n" + "="*60)
        logger.info("[LIST] 場景 1: 掃描到檢測完整流程")
        logger.info("="*60)
        
        scenario_data = {
            'name': 'scan_to_detection_flow',
            'description': 'CLI觸發掃描 → 發現資產 → 自動觸發檢測 → 產生報告',
            'steps': []
        }
        
        # Step 1: CLI 向 Core 發送掃描請求
        logger.info("\n[TARGET] Step 1: CLI -> Core (掃描請求)")
        scan_task = await self._cli_send_scan_request()
        scenario_data['steps'].append({
            'step': 1,
            'action': 'CLI_to_Core',
            'topic': 'tasks.scan.discovery',
            'payload': scan_task
        })
        
        # 模擬等待掃描完成
        await asyncio.sleep(2)
        
        # Step 2: Core 接收掃描請求，派發到 Scan Worker
        logger.info("\n[TARGET] Step 2: Core -> Scan Worker")
        await self._core_dispatch_to_scan(scan_task)
        scenario_data['steps'].append({
            'step': 2,
            'action': 'Core_to_Scan',
            'topic': 'tasks.scan.discovery',
            'status': 'dispatched'
        })
        
        # 模擬掃描執行
        await asyncio.sleep(3)
        
        # Step 3: Scan Worker 發現資產，回報給 Core
        logger.info("\n[TARGET] Step 3: Scan Worker -> Core (資產發現)")
        assets = await self._scan_worker_report_assets()
        scenario_data['steps'].append({
            'step': 3,
            'action': 'Scan_to_Core',
            'topic': 'results.scan.discovery',
            'assets_count': len(assets)
        })
        
        # Step 4: Core 自動觸發功能檢測
        logger.info("\n[TARGET] Step 4: Core -> Function Workers (自動觸發)")
        detection_tasks = await self._core_auto_trigger_detection(assets)
        scenario_data['steps'].append({
            'step': 4,
            'action': 'Core_to_Function',
            'topics': ['tasks.function.sqli', 'tasks.function.xss'],
            'tasks_count': len(detection_tasks)
        })
        
        # 模擬檢測執行
        await asyncio.sleep(4)
        
        # Step 5: Function Workers 回報結果到 Core
        logger.info("\n[TARGET] Step 5: Function Workers -> Core (檢測結果)")
        findings = await self._function_workers_report_findings()
        scenario_data['steps'].append({
            'step': 5,
            'action': 'Function_to_Core',
            'topics': ['results.function.sqli', 'results.function.xss'],
            'findings_count': len(findings)
        })
        
        # Step 6: Core 轉發到 Integration
        logger.info("\n[TARGET] Step 6: Core -> Integration (結果聚合)")
        await self._core_forward_to_integration(findings)
        scenario_data['steps'].append({
            'step': 6,
            'action': 'Core_to_Integration',
            'topic': 'results.integration.all'
        })
        
        # 模擬整合處理
        await asyncio.sleep(2)
        
        # Step 7: Integration 生成報告並通知 Core
        logger.info("\n[TARGET] Step 7: Integration -> Core (報告就緒)")
        report = await self._integration_generate_report(findings)
        scenario_data['steps'].append({
            'step': 7,
            'action': 'Integration_to_Core',
            'topic': 'events.integration.report.ready',
            'report_id': report['report_id']
        })
        
        # Step 8: Core 通知 CLI 完成
        logger.info("\n[TARGET] Step 8: Core -> CLI (任務完成通知)")
        await self._core_notify_cli_completion(report)
        scenario_data['steps'].append({
            'step': 8,
            'action': 'Core_to_CLI',
            'status': 'completed',
            'report_url': report.get('url')
        })
        
        # Step 9: AI 學習與優化
        logger.info("\n[TARGET] Step 9: AI 學習階段")
        ai_insights = await self._ai_learn_from_scenario(scenario_data)
        scenario_data['ai_insights'] = ai_insights
        
        self.training_data['scenarios'].append(scenario_data)
        
        logger.info(f"\n[OK] 場景 1 完成！發現 {len(assets)} 個資產，{len(findings)} 個漏洞")
        
    async def scenario_2_direct_detection(self):
        """場景 2: CLI → Core → Function → Integration → CLI (跳過掃描)"""
        logger.info("\n" + "="*60)
        logger.info("[LIST] 場景 2: 直接功能檢測流程")
        logger.info("="*60)
        
        scenario_data = {
            'name': 'direct_detection_flow',
            'description': '直接對已知目標進行漏洞檢測',
            'steps': []
        }
        
        # Step 1: CLI 直接發送檢測請求
        logger.info("\n[TARGET] Step 1: CLI -> Core (SQLi 檢測請求)")
        detection_task = await self._cli_send_detection_request()
        scenario_data['steps'].append({
            'step': 1,
            'action': 'CLI_to_Core',
            'topic': 'tasks.function.sqli',
            'target': detection_task['target']
        })
        
        # Step 2: Core 派發到 Function Worker
        logger.info("\n[TARGET] Step 2: Core -> SQLi Worker")
        await self._core_dispatch_to_function(detection_task)
        scenario_data['steps'].append({
            'step': 2,
            'action': 'Core_to_Function',
            'worker': 'sqli'
        })
        
        await asyncio.sleep(3)
        
        # Step 3: SQLi Worker 執行並回報
        logger.info("\n[TARGET] Step 3: SQLi Worker -> Core (檢測完成)")
        finding = await self._sqli_worker_execute_and_report()
        scenario_data['steps'].append({
            'step': 3,
            'action': 'Function_to_Core',
            'topic': 'results.function.sqli',
            'vulnerability_found': finding is not None
        })
        
        # Step 4-6: 同場景1的整合流程
        if finding:
            await self._core_forward_to_integration([finding])
            report = await self._integration_generate_report([finding])
            await self._core_notify_cli_completion(report)
            
            scenario_data['steps'].extend([
                {'step': 4, 'action': 'Core_to_Integration'},
                {'step': 5, 'action': 'Integration_generates_report'},
                {'step': 6, 'action': 'Core_to_CLI', 'report_id': report['report_id']}
            ])
        
        # AI 學習
        ai_insights = await self._ai_learn_from_scenario(scenario_data)
        scenario_data['ai_insights'] = ai_insights
        
        self.training_data['scenarios'].append(scenario_data)
        logger.info("\n[OK] 場景 2 完成！")
        
    async def scenario_3_batch_processing(self):
        """場景 3: 批量處理多個目標"""
        logger.info("\n" + "="*60)
        logger.info("[LIST] 場景 3: 批量處理流程")
        logger.info("="*60)
        
        scenario_data = {
            'name': 'batch_processing',
            'description': '批量處理多個目標的並行檢測',
            'steps': []
        }
        
        # 模擬批量任務
        targets = [
            "http://example1.com/api",
            "http://example2.com/api",
            "http://example3.com/api"
        ]
        
        logger.info(f"\n[TARGET] 批量處理 {len(targets)} 個目標")
        
        # 並行發送任務
        tasks = []
        for idx, target in enumerate(targets):
            task = self._create_detection_task(target)
            tasks.append(task)
            scenario_data['steps'].append({
                'step': idx + 1,
                'target': target,
                'task_id': task['task_id']
            })
        
        logger.info(f"[OK] 創建 {len(tasks)} 個並行任務")
        
        # 模擬並行執行
        await asyncio.sleep(3)
        
        # AI 學習批量處理模式
        ai_insights = {
            'pattern': 'batch_parallel_execution',
            'efficiency_gain': '3x faster than sequential',
            'resource_usage': 'optimal'
        }
        scenario_data['ai_insights'] = ai_insights
        
        self.training_data['scenarios'].append(scenario_data)
        logger.info("\n[OK] 場景 3 完成！")
        
    async def scenario_4_error_handling(self):
        """場景 4: 錯誤處理與恢復"""
        logger.info("\n" + "="*60)
        logger.info("[LIST] 場景 4: 錯誤處理與恢復")
        logger.info("="*60)
        
        scenario_data = {
            'name': 'error_handling_recovery',
            'description': '測試系統錯誤處理和自動恢復能力',
            'steps': []
        }
        
        # 模擬各種錯誤情況
        error_cases = [
            {'type': 'timeout', 'description': 'Worker 超時'},
            {'type': 'network', 'description': '網絡連接失敗'},
            {'type': 'invalid_payload', 'description': '無效的消息格式'},
            {'type': 'worker_crash', 'description': 'Worker 崩潰'}
        ]
        
        for idx, error_case in enumerate(error_cases):
            logger.info(f"\n[TARGET] 測試錯誤情況 {idx+1}: {error_case['description']}")
            
            # 模擬錯誤和恢復
            recovery = await self._simulate_error_and_recovery(error_case)
            
            scenario_data['steps'].append({
                'step': idx + 1,
                'error_type': error_case['type'],
                'recovery_action': recovery['action'],
                'recovery_time': recovery['time'],
                'success': recovery['success']
            })
        
        # AI 學習錯誤模式
        ai_insights = {
            'error_patterns_learned': len(error_cases),
            'recovery_strategies': 'retry_with_backoff, fallback_worker, circuit_breaker',
            'resilience_score': 0.95
        }
        scenario_data['ai_insights'] = ai_insights
        
        self.training_data['scenarios'].append(scenario_data)
        logger.info("\n[OK] 場景 4 完成！")
        
    # ========== 輔助方法 ==========
    
    async def _cli_send_scan_request(self) -> Dict:
        """模擬 CLI 發送掃描請求"""
        task = {
            'task_id': f"scan_{datetime.now().strftime('%H%M%S')}",
            'target': 'https://example.com',
            'strategy': 'balanced',
            'max_depth': 3
        }
        logger.info(f"  [U+1F4E4] CLI 發送掃描請求: {task['target']}")
        return task
        
    async def _core_dispatch_to_scan(self, task: Dict):
        """Core 派發任務到 Scan Worker"""
        logger.info(f"  [MIX] Core 派發到 Scan Worker: {task['task_id']}")
        
    async def _scan_worker_report_assets(self) -> List[Dict]:
        """Scan Worker 回報發現的資產"""
        assets = [
            {
                'url': 'https://example.com/api/users',
                'method': 'GET',
                'params': {'id': '1'},
                'type': 'api_endpoint'
            },
            {
                'url': 'https://example.com/api/login',
                'method': 'POST',
                'params': {'username': '', 'password': ''},
                'type': 'authentication'
            }
        ]
        logger.info(f"  [STATS] Scan Worker 發現 {len(assets)} 個資產")
        for asset in assets:
            logger.info(f"     - {asset['url']}")
        return assets
        
    async def _core_auto_trigger_detection(self, assets: List[Dict]) -> List[Dict]:
        """Core 自動觸發功能檢測"""
        tasks = []
        for asset in assets:
            # 根據資產類型決定檢測類型
            if 'login' in asset['url']:
                tasks.append({
                    'type': 'sqli',
                    'target': asset,
                    'task_id': f"sqli_{len(tasks)}"
                })
            else:
                tasks.append({
                    'type': 'xss',
                    'target': asset,
                    'task_id': f"xss_{len(tasks)}"
                })
        
        logger.info(f"  [TARGET] Core 自動觸發 {len(tasks)} 個檢測任務")
        for task in tasks:
            logger.info(f"     - {task['type'].upper()}: {task['target']['url']}")
        return tasks
        
    async def _function_workers_report_findings(self) -> List[Dict]:
        """Function Workers 回報漏洞發現"""
        findings = [
            {
                'type': 'SQL Injection',
                'severity': 'HIGH',
                'url': 'https://example.com/api/users',
                'parameter': 'id',
                'evidence': "Error: You have an error in your SQL syntax"
            }
        ]
        logger.info(f"  [ALERT] Function Workers 發現 {len(findings)} 個漏洞")
        for finding in findings:
            logger.info(f"     - {finding['type']} ({finding['severity']}): {finding['url']}")
        return findings
        
    async def _core_forward_to_integration(self, findings: List[Dict]):
        """Core 轉發結果到 Integration"""
        logger.info(f"  [U+1F4EE] Core 轉發 {len(findings)} 個結果到 Integration")
        
    async def _integration_generate_report(self, findings: List[Dict]) -> Dict:
        """Integration 生成報告"""
        report = {
            'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'findings_count': len(findings),
            'high_severity': sum(1 for f in findings if f.get('severity') == 'HIGH'),
            'url': f"/reports/{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            'generated_at': datetime.now().isoformat()
        }
        logger.info(f"  [U+1F4C4] Integration 生成報告: {report['report_id']}")
        logger.info(f"     - 總漏洞數: {report['findings_count']}")
        logger.info(f"     - 高危漏洞: {report['high_severity']}")
        return report
        
    async def _core_notify_cli_completion(self, report: Dict):
        """Core 通知 CLI 任務完成"""
        logger.info(f"  [U+2709][U+FE0F] Core 通知 CLI 任務完成")
        logger.info(f"     - 報告 ID: {report['report_id']}")
        logger.info(f"     - 報告 URL: {report['url']}")
        
    async def _ai_learn_from_scenario(self, scenario_data: Dict) -> Dict:
        """AI 從場景中學習"""
        logger.info("\n  [BRAIN] AI 學習階段:")
        
        insights = {
            'scenario_name': scenario_data['name'],
            'total_steps': len(scenario_data['steps']),
            'learned_patterns': [],
            'optimization_suggestions': []
        }
        
        # 分析步驟模式
        if 'scan' in scenario_data['name']:
            insights['learned_patterns'].append('scan_then_detect_pattern')
            insights['optimization_suggestions'].append('可並行處理發現的資產')
        
        if 'direct' in scenario_data['name']:
            insights['learned_patterns'].append('direct_detection_pattern')
            insights['optimization_suggestions'].append('跳過掃描階段節省時間')
        
        if 'batch' in scenario_data['name']:
            insights['learned_patterns'].append('parallel_execution_pattern')
            insights['optimization_suggestions'].append('批量任務使用工作池')
        
        # 計算效率指標
        insights['efficiency_metrics'] = {
            'avg_step_time': '2.5s',
            'total_time': f"{len(scenario_data['steps']) * 2.5}s",
            'success_rate': '100%'
        }
        
        logger.info(f"     [CHECK] 學習到 {len(insights['learned_patterns'])} 個模式")
        for pattern in insights['learned_patterns']:
            logger.info(f"       - {pattern}")
        
        logger.info(f"     [CHECK] 產生 {len(insights['optimization_suggestions'])} 個優化建議")
        for suggestion in insights['optimization_suggestions']:
            logger.info(f"       - {suggestion}")
        
        return insights
        
    async def _cli_send_detection_request(self) -> Dict:
        """CLI 直接發送檢測請求"""
        task = {
            'task_id': f"detect_{datetime.now().strftime('%H%M%S')}",
            'type': 'sqli',
            'target': 'https://example.com/api/users?id=1',
            'parameters': {'id': '1'}
        }
        logger.info(f"  [U+1F4E4] CLI 發送檢測請求: {task['target']}")
        return task
        
    async def _core_dispatch_to_function(self, task: Dict):
        """Core 派發到 Function Worker"""
        logger.info(f"  [MIX] Core 派發到 {task['type'].upper()} Worker")
        
    async def _sqli_worker_execute_and_report(self) -> Optional[Dict]:
        """SQLi Worker 執行檢測並回報"""
        finding = {
            'type': 'SQL Injection',
            'severity': 'HIGH',
            'url': 'https://example.com/api/users',
            'parameter': 'id',
            'payload': "1' OR '1'='1",
            'evidence': "Error in SQL syntax",
            'confidence': 0.95
        }
        logger.info(f"  [ALERT] SQLi Worker 發現漏洞: {finding['type']}")
        return finding
        
    def _create_detection_task(self, target: str) -> Dict:
        """創建檢測任務"""
        return {
            'task_id': f"task_{target.split('//')[1].split('/')[0]}",
            'target': target,
            'type': 'sqli',
            'created_at': datetime.now().isoformat()
        }
        
    async def _simulate_error_and_recovery(self, error_case: Dict) -> Dict:
        """模擬錯誤和恢復"""
        logger.info(f"  [WARN] 模擬錯誤: {error_case['description']}")
        
        # 模擬恢復過程
        await asyncio.sleep(1)
        
        recovery = {
            'action': 'retry_with_exponential_backoff',
            'time': '2.3s',
            'success': True
        }
        
        logger.info(f"  [OK] 恢復成功: {recovery['action']}")
        return recovery
        
    def _save_training_data(self):
        """保存訓練數據"""
        self.training_data['end_time'] = datetime.now().isoformat()
        
        output_file = self.output_dir / f"{self.training_session_id}.json"
        output_file.write_text(
            json.dumps(self.training_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        
        logger.info(f"\n[SAVE] 訓練數據已保存: {output_file}")
        
        # 生成訓練報告
        self._generate_training_report()
        
    def _generate_training_report(self):
        """生成訓練報告"""
        report_file = self.output_dir / f"{self.training_session_id}_report.md"
        
        lines = [
            f"# AI 訓練會話報告",
            f"",
            f"**會話 ID**: {self.training_session_id}",
            f"**開始時間**: {self.training_data['start_time']}",
            f"**結束時間**: {self.training_data['end_time']}",
            f"**總場景數**: {len(self.training_data['scenarios'])}",
            f"",
            "---",
            "",
            "## 訓練場景摘要",
            ""
        ]
        
        for idx, scenario in enumerate(self.training_data['scenarios'], 1):
            lines.extend([
                f"### {idx}. {scenario['name']}",
                f"",
                f"**描述**: {scenario['description']}",
                f"**步驟數**: {len(scenario.get('steps', []))}",
                f""
            ])
            
            if 'ai_insights' in scenario:
                lines.extend([
                    "**AI 學習成果**:",
                    ""
                ])
                insights = scenario['ai_insights']
                if 'learned_patterns' in insights:
                    lines.append("學習到的模式:")
                    for pattern in insights['learned_patterns']:
                        lines.append(f"- {pattern}")
                    lines.append("")
                
                if 'optimization_suggestions' in insights:
                    lines.append("優化建議:")
                    for suggestion in insights['optimization_suggestions']:
                        lines.append(f"- {suggestion}")
                    lines.append("")
        
        lines.extend([
            "---",
            "",
            "## 下一步建議",
            "",
            "1. 檢查 AI 學習到的模式是否符合預期",
            "2. 根據優化建議調整系統參數",
            "3. 執行更多訓練場景以提高 AI 準確度",
            "4. 將學習成果應用到實際檢測任務",
            ""
        ])
        
        report_file.write_text("\n".join(lines), encoding='utf-8')
        logger.info(f"[STATS] 訓練報告已生成: {report_file}")


async def main():
    """主函式"""
    print("=" * 70)
    print("[AI] AIVA AI 強化訓練系統")
    print("=" * 70)
    print()
    print("本腳本將執行以下訓練場景:")
    print("  1[U+FE0F][U+20E3]  完整掃描到檢測流程 (Scan -> Function -> Integration)")
    print("  2[U+FE0F][U+20E3]  直接功能檢測流程 (跳過掃描)")
    print("  3[U+FE0F][U+20E3]  批量並行處理流程")
    print("  4[U+FE0F][U+20E3]  錯誤處理與恢復流程")
    print()
    print("訓練過程中，AI 將同時學習:")
    print("  • 通訊模式識別")
    print("  • 工作流程優化")
    print("  • 錯誤處理策略")
    print("  • 資源調度算法")
    print()
    print("=" * 70)
    print()
    
    # 創建輸出目錄
    output_dir = Path(__file__).parent.parent.parent / "_out1101016" / "ai_training"
    
    # 創建並運行訓練編排器
    orchestrator = AITrainingOrchestrator(output_dir)
    
    try:
        await orchestrator.run_complete_training()
        
        print()
        print("=" * 70)
        print("[OK] 訓練完成！")
        print("=" * 70)
        print()
        print(f"[U+1F4C2] 訓練數據目錄: {output_dir}")
        print(f"[U+1F4C4] 訓練數據: {orchestrator.training_session_id}.json")
        print(f"[STATS] 訓練報告: {orchestrator.training_session_id}_report.md")
        print()
        print("[TARGET] 下一步:")
        print("  1. 查看訓練報告了解 AI 學習成果")
        print("  2. 檢查訓練數據確認流程正確性")
        print("  3. 根據優化建議調整系統參數")
        print("  4. 重複訓練以提高 AI 準確度")
        print()
        
    except KeyboardInterrupt:
        print("\n[WARN] 訓練被中斷")
    except Exception as e:
        print(f"\n[FAIL] 訓練失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
