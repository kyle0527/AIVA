# AIVA 核心模組與掃描模組整合CLI指令系統
"""
AIVA 五模組協同CLI指令系統

本程式展示AIVA核心架構中的指令流程：
1. 核心模組(Core) -> 下令給掃描模組(Scan)
2. 掃描模組(Scan) -> 調用功能模組(Features)  
3. 掃描結果 -> 傳送至整合模組(Integration)
4. 整合模組(Integration) -> 統一輸出和反饋

運作流程：
Core AI Commander -> Scan Engine -> Features Detection -> Integration Service -> Core Analysis
"""

import asyncio
import argparse
import json
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

# 模擬AIVA模組導入
try:
    from services.core.aiva_core.ai_commander import AICommander, AITaskType, AIComponent
    from services.scan.unified_scan_engine import UnifiedScanEngine
    from services.integration.models import AIOperationRecord
except ImportError:
    print("⚠️  實際模組未完全載入，使用模擬模式")

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class CommandType(str, Enum):
    """CLI指令類型"""
    QUICK_SCAN = "quick-scan"           # 快速掃描
    DEEP_SCAN = "deep-scan"             # 深度掃描
    TARGET_DISCOVERY = "discovery"      # 目標發現
    VULNERABILITY_ASSESSMENT = "vuln"   # 漏洞評估
    INTELLIGENCE_GATHERING = "intel"    # 情報收集
    COMPREHENSIVE_AUDIT = "audit"       # 綜合審計

class ModuleName(str, Enum):
    """模組名稱"""
    CORE = "core"
    SCAN = "scan"
    FEATURES = "features"
    INTEGRATION = "integration"
    COMMON = "common"

@dataclass
class CoreCommand:
    """核心模組指令"""
    command_id: str
    command_type: CommandType
    target: str
    parameters: Dict[str, Any]
    timestamp: datetime
    requested_by: str = "CLI"

@dataclass
class ScanTask:
    """掃描任務"""
    task_id: str
    command_id: str
    scan_type: str
    target: str
    strategy: str
    modules_required: List[str]
    timeout: int = 300

@dataclass
class ExecutionResult:
    """執行結果"""
    task_id: str
    module: str
    status: str
    findings: List[Dict[str, Any]]
    execution_time: float
    error_message: Optional[str] = None

class MockAICommander:
    """模擬AI指揮官"""
    
    def __init__(self):
        self.active_commands = {}
        logger.info("🧠 Mock AI Commander initialized")
    
    async def execute_command(self, command: CoreCommand) -> Dict[str, Any]:
        """執行核心指令"""
        logger.info(f"🎯 [CORE] Executing command: {command.command_type.value}")
        logger.info(f"   Target: {command.target}")
        logger.info(f"   Parameters: {json.dumps(command.parameters, indent=2)}")
        
        # 分析指令並生成掃描任務
        scan_tasks = await self._analyze_and_generate_tasks(command)
        
        # 下令給掃描模組
        scan_results = []
        for task in scan_tasks:
            result = await self._delegate_to_scan_module(task)
            scan_results.append(result)
        
        # 整合結果
        integrated_result = await self._integrate_results(command, scan_results)
        
        return integrated_result
    
    async def _analyze_and_generate_tasks(self, command: CoreCommand) -> List[ScanTask]:
        """分析指令並生成掃描任務"""
        logger.info("🔍 [CORE] Analyzing command and generating scan tasks...")
        
        tasks = []
        task_id_base = f"task_{command.command_id}_{int(time.time())}"
        
        if command.command_type == CommandType.QUICK_SCAN:
            tasks.append(ScanTask(
                task_id=f"{task_id_base}_quick",
                command_id=command.command_id,
                scan_type="quick_vulnerability_scan",
                target=command.target,
                strategy="FAST",
                modules_required=["vulnerability_scanner", "port_scanner"],
                timeout=120
            ))
        
        elif command.command_type == CommandType.DEEP_SCAN:
            tasks.extend([
                ScanTask(
                    task_id=f"{task_id_base}_discovery",
                    command_id=command.command_id,
                    scan_type="comprehensive_discovery",
                    target=command.target,
                    strategy="COMPREHENSIVE",
                    modules_required=["network_scanner", "service_detector", "fingerprint_manager"],
                    timeout=300
                ),
                ScanTask(
                    task_id=f"{task_id_base}_vuln",
                    command_id=command.command_id,
                    scan_type="deep_vulnerability_scan",
                    target=command.target,
                    strategy="COMPREHENSIVE",
                    modules_required=["vulnerability_scanner", "auth_manager", "payload_generator"],
                    timeout=600
                )
            ])
        
        elif command.command_type == CommandType.INTELLIGENCE_GATHERING:
            tasks.append(ScanTask(
                task_id=f"{task_id_base}_intel",
                command_id=command.command_id,
                scan_type="intelligence_collection",
                target=command.target,
                strategy="STEALTH",
                modules_required=["info_gatherer_rust", "osint_collector", "metadata_analyzer"],
                timeout=240
            ))
        
        logger.info(f"   Generated {len(tasks)} scan tasks")
        for task in tasks:
            logger.info(f"   - {task.task_id}: {task.scan_type} ({task.strategy})")
        
        return tasks
    
    async def _delegate_to_scan_module(self, task: ScanTask) -> ExecutionResult:
        """委派任務給掃描模組"""
        logger.info(f"📤 [CORE->SCAN] Delegating task: {task.task_id}")
        
        # 模擬調用掃描引擎
        mock_scan_engine = MockUnifiedScanEngine()
        result = await mock_scan_engine.execute_scan_task(task)
        
        return result
    
    async def _integrate_results(self, command: CoreCommand, scan_results: List[ExecutionResult]) -> Dict[str, Any]:
        """整合掃描結果"""
        logger.info("🔗 [CORE] Integrating scan results...")
        
        # 統計結果
        total_findings = sum(len(result.findings) for result in scan_results)
        successful_scans = sum(1 for result in scan_results if result.status == "completed")
        total_execution_time = sum(result.execution_time for result in scan_results)
        
        # 分類發現
        findings_by_severity = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
        
        for result in scan_results:
            for finding in result.findings:
                severity = finding.get("severity", "info").lower()
                if severity in findings_by_severity:
                    findings_by_severity[severity].append(finding)
        
        integrated_result = {
            "command_id": command.command_id,
            "command_type": command.command_type.value,
            "target": command.target,
            "execution_summary": {
                "total_tasks": len(scan_results),
                "successful_tasks": successful_scans,
                "total_findings": total_findings,
                "total_execution_time": round(total_execution_time, 2),
                "findings_by_severity": {k: len(v) for k, v in findings_by_severity.items()}
            },
            "detailed_findings": findings_by_severity,
            "scan_results": [asdict(result) for result in scan_results],
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if successful_scans == len(scan_results) else "partial_success"
        }
        
        # 傳送至整合模組
        await self._send_to_integration_module(integrated_result)
        
        logger.info(f"✅ [CORE] Integration completed: {total_findings} findings found")
        
        return integrated_result
    
    async def _send_to_integration_module(self, result: Dict[str, Any]):
        """傳送結果至整合模組"""
        logger.info("📨 [CORE->INTEGRATION] Sending results to integration module")
        
        mock_integration = MockIntegrationService()
        await mock_integration.process_scan_results(result)

class MockUnifiedScanEngine:
    """模擬統一掃描引擎"""
    
    async def execute_scan_task(self, task: ScanTask) -> ExecutionResult:
        """執行掃描任務"""
        logger.info(f"🔍 [SCAN] Executing scan task: {task.scan_type}")
        logger.info(f"   Target: {task.target}")
        logger.info(f"   Strategy: {task.strategy}")
        logger.info(f"   Required modules: {', '.join(task.modules_required)}")
        
        start_time = time.time()
        
        # 模擬調用功能模組
        findings = []
        for module in task.modules_required:
            module_findings = await self._call_feature_module(module, task)
            findings.extend(module_findings)
        
        execution_time = time.time() - start_time
        
        result = ExecutionResult(
            task_id=task.task_id,
            module="scan",
            status="completed",
            findings=findings,
            execution_time=execution_time
        )
        
        logger.info(f"✅ [SCAN] Task completed: {len(findings)} findings in {execution_time:.2f}s")
        
        return result
    
    async def _call_feature_module(self, module_name: str, task: ScanTask) -> List[Dict[str, Any]]:
        """調用功能模組"""
        logger.info(f"🎯 [SCAN->FEATURES] Calling feature module: {module_name}")
        
        # 模擬不同模組的發現
        mock_findings = {
            "vulnerability_scanner": [
                {"type": "sql_injection", "severity": "high", "location": "/api/users", "confidence": 0.9},
                {"type": "xss", "severity": "medium", "location": "/search", "confidence": 0.8}
            ],
            "port_scanner": [
                {"type": "open_port", "severity": "info", "port": 80, "service": "http"},
                {"type": "open_port", "severity": "info", "port": 443, "service": "https"},
                {"type": "open_port", "severity": "low", "port": 22, "service": "ssh"}
            ],
            "network_scanner": [
                {"type": "subdomain", "severity": "info", "value": "api.example.com"},
                {"type": "subdomain", "severity": "info", "value": "admin.example.com"}
            ],
            "service_detector": [
                {"type": "service_version", "severity": "medium", "service": "apache", "version": "2.4.29"}
            ],
            "fingerprint_manager": [
                {"type": "technology", "severity": "info", "tech": "react", "version": "17.0.2"}
            ],
            "info_gatherer_rust": [
                {"type": "sensitive_info", "severity": "high", "content": "api_key_found", "location": "/js/config.js"}
            ],
            "auth_manager": [
                {"type": "auth_bypass", "severity": "critical", "method": "jwt_none_alg", "location": "/api/auth"}
            ]
        }
        
        # 模擬處理時間
        await asyncio.sleep(0.5)
        
        findings = mock_findings.get(module_name, [])
        logger.info(f"   📋 [FEATURES] {module_name} found {len(findings)} items")
        
        return findings

class MockIntegrationService:
    """模擬整合服務"""
    
    async def process_scan_results(self, results: Dict[str, Any]):
        """處理掃描結果"""
        logger.info("🔗 [INTEGRATION] Processing scan results...")
        
        # 模擬處理過程
        await asyncio.sleep(0.3)
        
        # 生成報告
        report = await self._generate_report(results)
        
        # 儲存至資料庫
        await self._save_to_database(results)
        
        # 觸發後續動作
        await self._trigger_follow_up_actions(results)
        
        logger.info("✅ [INTEGRATION] Results processed successfully")
    
    async def _generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成報告"""
        logger.info("📊 [INTEGRATION] Generating comprehensive report...")
        
        report = {
            "report_id": f"report_{results['command_id']}",
            "target": results["target"],
            "scan_date": datetime.now().isoformat(),
            "executive_summary": {
                "total_findings": results["execution_summary"]["total_findings"],
                "critical_issues": results["execution_summary"]["findings_by_severity"]["critical"],
                "high_issues": results["execution_summary"]["findings_by_severity"]["high"],
                "risk_score": self._calculate_risk_score(results)
            },
            "detailed_findings": results["detailed_findings"],
            "recommendations": self._generate_recommendations(results)
        }
        
        # 儲存報告
        report_path = Path(f"reports/scan_report_{results['command_id']}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   📄 Report saved: {report_path}")
        
        return report
    
    def _calculate_risk_score(self, results: Dict[str, Any]) -> float:
        """計算風險分數"""
        severity_weights = {"critical": 10, "high": 7, "medium": 4, "low": 2, "info": 1}
        
        total_score = 0
        for severity, count in results["execution_summary"]["findings_by_severity"].items():
            total_score += count * severity_weights.get(severity, 0)
        
        # 正規化到0-100分
        max_possible = 100
        risk_score = min((total_score / max_possible) * 100, 100)
        
        return round(risk_score, 2)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成建議"""
        recommendations = []
        
        findings_count = results["execution_summary"]["findings_by_severity"]
        
        if findings_count["critical"] > 0:
            recommendations.append("🚨 立即修復所有Critical等級漏洞，這些漏洞可能導致系統完全妥協")
        
        if findings_count["high"] > 0:
            recommendations.append("⚠️ 優先處理High等級漏洞，建議在48小時內完成修復")
        
        if findings_count["medium"] > 5:
            recommendations.append("📋 中等風險漏洞數量較多，建議制定系統性修復計畫")
        
        if findings_count["info"] > 10:
            recommendations.append("ℹ️ 資訊類發現較多，建議review系統配置和安全基線")
        
        return recommendations
    
    async def _save_to_database(self, results: Dict[str, Any]):
        """儲存至資料庫"""
        logger.info("💾 [INTEGRATION] Saving to database...")
        # 模擬資料庫操作
        await asyncio.sleep(0.2)
    
    async def _trigger_follow_up_actions(self, results: Dict[str, Any]):
        """觸發後續動作"""
        logger.info("🔄 [INTEGRATION] Triggering follow-up actions...")
        
        critical_count = results["execution_summary"]["findings_by_severity"]["critical"]
        if critical_count > 0:
            logger.info(f"   🚨 Alert: {critical_count} critical vulnerabilities found - notifying security team")
        
        # 模擬通知
        await asyncio.sleep(0.1)

class CoreScanCLI:
    """核心掃描CLI"""
    
    def __init__(self):
        self.ai_commander = MockAICommander()
        logger.info("🚀 AIVA Core-Scan Integration CLI initialized")
    
    async def execute_command(self, command_type: CommandType, target: str, **kwargs) -> Dict[str, Any]:
        """執行CLI指令"""
        command_id = f"cmd_{int(time.time())}"
        
        command = CoreCommand(
            command_id=command_id,
            command_type=command_type,
            target=target,
            parameters=kwargs,
            timestamp=datetime.now()
        )
        
        logger.info("=" * 80)
        logger.info(f"🎯 Starting AIVA command execution: {command_type.value}")
        logger.info(f"   Command ID: {command_id}")
        logger.info(f"   Target: {target}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            result = await self.ai_commander.execute_command(command)
            execution_time = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info("✅ COMMAND EXECUTION COMPLETED")
            logger.info(f"   Total execution time: {execution_time:.2f} seconds")
            logger.info(f"   Total findings: {result['execution_summary']['total_findings']}")
            logger.info(f"   Status: {result['status']}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("=" * 80)
            logger.error("❌ COMMAND EXECUTION FAILED")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   Execution time: {execution_time:.2f} seconds")
            logger.error("=" * 80)
            raise

def create_cli_parser():
    """創建CLI參數解析器"""
    parser = argparse.ArgumentParser(
        description="AIVA 核心模組與掃描模組整合CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python core_scan_integration_cli.py quick-scan https://example.com
  python core_scan_integration_cli.py deep-scan https://target.com --timeout 600
  python core_scan_integration_cli.py intel https://victim.com --stealth
  python core_scan_integration_cli.py audit https://app.com --comprehensive
        """
    )
    
    parser.add_argument(
        "command",
        choices=[cmd.value for cmd in CommandType],
        help="掃描指令類型"
    )
    
    parser.add_argument(
        "target",
        help="目標URL或IP地址"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="掃描超時時間(秒) [預設: 300]"
    )
    
    parser.add_argument(
        "--output",
        default="console",
        choices=["console", "json", "report"],
        help="輸出格式 [預設: console]"
    )
    
    parser.add_argument(
        "--stealth",
        action="store_true",
        help="啟用隱匿模式"
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="啟用綜合模式"
    )
    
    parser.add_argument(
        "--modules",
        nargs="+",
        help="指定特定模組"
    )
    
    return parser

async def main():
    """主函數"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # 創建CLI實例
    cli = CoreScanCLI()
    
    # 準備參數
    command_type = CommandType(args.command)
    kwargs = {
        "timeout": args.timeout,
        "output_format": args.output,
        "stealth_mode": args.stealth,
        "comprehensive_mode": args.comprehensive,
    }
    
    if args.modules:
        kwargs["specific_modules"] = args.modules
    
    try:
        # 執行指令
        result = await cli.execute_command(command_type, args.target, **kwargs)
        
        # 輸出結果
        if args.output == "json":
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.output == "report":
            print("\n📊 SCAN REPORT SUMMARY")
            print("=" * 50)
            print(f"Target: {result['target']}")
            print(f"Command: {result['command_type']}")
            print(f"Status: {result['status']}")
            print(f"Total Findings: {result['execution_summary']['total_findings']}")
            print(f"Execution Time: {result['execution_summary']['total_execution_time']}s")
            print("\n🔍 Findings by Severity:")
            for severity, count in result['execution_summary']['findings_by_severity'].items():
                if count > 0:
                    print(f"  {severity.upper()}: {count}")
        else:
            # Console 輝出已在執行過程中顯示
            print(f"\n✅ Command completed successfully. Results saved to reports/")
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Command interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Command failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    
    print("""
🎯 AIVA 核心模組與掃描模組整合CLI
═══════════════════════════════════════════════════════════════
展示五模組協同工作流程：
Core -> Scan -> Features -> Integration -> Analysis
═══════════════════════════════════════════════════════════════
    """)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)