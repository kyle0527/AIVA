#!/usr/bin/env python3
"""
AIVA 系統 SOP 合規性檢查工具

遵循單一真實來源原則和分層責任架構進行全面檢查
"""

import asyncio
import sys
import os
import json
import importlib.util
from pathlib import Path
from typing import Any, List
import logging

# 設置專案根目錄
project_root = Path(__file__).parent.parent.parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 嘗試導入核心模組
try:
    from services.core.aiva_core.ai_engine import (
        AIModelManager, Tool, CodeReader, CodeWriter,
        CodeAnalyzer, CommandExecutor, ScanTrigger, VulnerabilityDetector
    )
    from services.core.aiva_core.cognitive_core.neural.real_neural_core import (
        RealDecisionEngine, RealAICore
    )
    from services.core.aiva_core.learning import ModelTrainer, ScalableBioTrainer
    IMPORT_SUCCESS = True
    logger.info("✅ 核心 Python 模組導入成功 (5M Decision Engine)")
except ImportError as e:
    logger.error(f"❌ 核心 Python 模組導入失敗: {e}")
    IMPORT_SUCCESS = False

class AIVASOPComplianceChecker:
    """AIVA SOP 合規性檢查器"""
    
    def __init__(self, aiva_root: Path):
        self.aiva_root = aiva_root
        self.schemas_dir = aiva_root / "services" / "aiva_common" / "schemas"
        self.enums_dir = aiva_root / "services" / "aiva_common" / "enums"
        self.ai_core_dir = aiva_root / "services" / "core" / "aiva_core"
        
        self.check_results = {}
    
    async def run_comprehensive_check(self):
        """執行全面的 SOP 合規性檢查"""
        print("🔍 AIVA 系統 SOP 合規性檢查")
        print("=" * 70)
        print("檢查範圍: Schema 定義、AI 核心、系統整合、命令執行")
        print("=" * 70)
        
        # 1. Schema 定義體系檢查
        await self.check_schema_definitions()
        
        # 2. AI 核心模組檢查
        await self.check_ai_core_modules()
        
        # 3. 系統工具連接檢查
        await self.check_system_tools_connectivity()
        
        # 4. 命令執行鏈檢查
        await self.check_command_execution_chain()
        
        # 生成報告
        await self.generate_final_report()
    
    async def check_schema_definitions(self):
        """檢查 Schema 定義體系"""
        print("\n📋 1. Schema 定義體系檢查")
        print("-" * 50)
        
        schema_checks = {}
        
        # 檢查權威定義來源
        print("1.1 檢查權威定義來源...")
        try:
            core_schema_files = [
                "base.py", "messaging.py", "tasks.py", "findings.py",
                "ai.py", "api_testing.py", "assets.py", "risk.py", "telemetry.py"
            ]
            
            missing_files = []
            for schema_file in core_schema_files:
                file_path = self.schemas_dir / schema_file
                if not file_path.exists():
                    missing_files.append(schema_file)
            
            if not missing_files:
                print("✅ 權威 Schema 文件完整")
                schema_checks['authority_source'] = True
            else:
                print(f"❌ 缺少 Schema 文件: {missing_files}")
                schema_checks['authority_source'] = False
                
        except Exception as e:
            print(f"❌ 權威定義檢查失敗: {e}")
            schema_checks['authority_source'] = False
        
        # 檢查 Enum 定義
        print("1.2 檢查 Enum 定義...")
        try:
            core_enum_files = [
                "common.py", "modules.py", "security.py", "assets.py"
            ]
            
            enum_missing = []
            for enum_file in core_enum_files:
                file_path = self.enums_dir / enum_file
                if not file_path.exists():
                    enum_missing.append(enum_file)
            
            if not enum_missing:
                print("✅ Enum 定義文件完整")
                schema_checks['enum_definitions'] = True
            else:
                print(f"❌ 缺少 Enum 文件: {enum_missing}")
                schema_checks['enum_definitions'] = False
                
        except Exception as e:
            print(f"❌ Enum 定義檢查失敗: {e}")
            schema_checks['enum_definitions'] = False
        
        # 檢查導入導出完整性
        print("1.3 檢查導入導出完整性...")
        try:
            spec = importlib.util.spec_from_file_location(
                "aiva_common",
                self.aiva_root / "services" / "aiva_common" / "__init__.py"
            )
            if spec and spec.loader:
                aiva_common = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(aiva_common)
                print("✅ aiva_common 模組導入成功")
                schema_checks['import_export'] = True
            else:
                print("❌ aiva_common 模組規範載入失敗")
                schema_checks['import_export'] = False
                
        except Exception as e:
            print(f"❌ 導入導出檢查失敗: {e}")
            schema_checks['import_export'] = False
        
        self.check_results['schema_definitions'] = schema_checks
    
    async def check_ai_core_modules(self):
        """檢查 AI 核心模組整合"""
        print("\n🧠 2. AI 核心模組檢查")
        print("-" * 50)
        
        ai_checks = {}
        
        # 檢查 AI 引擎核心
        print("2.1 檢查 AI 引擎核心...")
        try:
            from services.core.aiva_core.cognitive_core.neural.real_neural_core import (
                RealDecisionEngine, RealAICore
            )
            from services.core.aiva_core.ai_engine import (
                AIModelManager, OptimizedScalableBioNet
            )
            print("✅ AI 引擎核心模組載入成功 (5M Decision Engine)")
            ai_checks['ai_engine_core'] = True
        except Exception as e:
            print(f"❌ AI 引擎核心載入失敗: {e}")
            ai_checks['ai_engine_core'] = False
        
        # 檢查工具系統
        print("2.2 檢查工具系統...")
        try:
            from services.core.aiva_core.ai_engine import (
                Tool, CodeReader, CodeWriter, CodeAnalyzer,
                CommandExecutor, ScanTrigger, VulnerabilityDetector
            )
            print("✅ 工具系統載入成功")
            ai_checks['tool_system'] = True
        except Exception as e:
            print(f"❌ 工具系統載入失敗: {e}")
            ai_checks['tool_system'] = False
        
        # 檢查學習系統
        print("2.3 檢查學習系統...")
        try:
            from services.core.aiva_core.learning import (
                ModelTrainer, ScalableBioTrainer, ScalableBioTrainingConfig
            )
            print("✅ 學習系統載入成功")
            ai_checks['learning_system'] = True
        except Exception as e:
            print(f"❌ 學習系統載入失敗: {e}")
            ai_checks['learning_system'] = False
        
        self.check_results['ai_core_modules'] = ai_checks
    
    async def check_system_tools_connectivity(self):
        """檢查系統工具連接性"""
        print("\n🔧 3. 系統工具連接檢查")
        print("-" * 50)
        
        tools_checks = {}
        
        # 檢查文件操作工具
        print("3.1 檢查文件操作工具...")
        try:
            from services.core.aiva_core.ai_engine import CodeReader, CodeWriter
            reader = CodeReader(codebase_path=".")
            writer = CodeWriter(codebase_path=".")
            print("✅ 文件操作工具可用")
            tools_checks['file_operations'] = True
        except Exception as e:
            print(f"❌ 文件操作工具檢查失敗: {e}")
            tools_checks['file_operations'] = False
        
        # 檢查命令執行工具
        print("3.2 檢查命令執行工具...")
        try:
            from services.core.aiva_core.ai_engine import CommandExecutor
            executor = CommandExecutor(codebase_path=".")
            print("✅ 命令執行工具可用")
            tools_checks['command_execution'] = True
        except Exception as e:
            print(f"❌ 命令執行工具檢查失敗: {e}")
            tools_checks['command_execution'] = False
        
        # 檢查掃描觸發工具
        print("3.3 檢查掃描觸發工具...")
        try:
            from services.core.aiva_core.ai_engine import ScanTrigger
            scanner = ScanTrigger()
            print("✅ 掃描觸發工具可用")
            tools_checks['scan_trigger'] = True
        except Exception as e:
            print(f"❌ 掃描觸發工具檢查失敗: {e}")
            tools_checks['scan_trigger'] = False
        
        self.check_results['system_tools'] = tools_checks
    
    async def check_command_execution_chain(self):
        """檢查命令執行鏈"""
        print("\n⚙️  4. 命令執行鏈檢查")
        print("-" * 50)
        
        exec_checks = {}
        
        # 檢查命令構建
        print("4.1 檢查命令構建...")
        try:
            import subprocess
            check_cmd = "python --version"
            result = subprocess.run(
                check_cmd.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ 命令構建成功: {result.stdout.strip()}")
                exec_checks['command_building'] = True
            else:
                print("❌ 命令構建失敗")
                exec_checks['command_building'] = False
        except Exception as e:
            print(f"❌ 命令構建檢查失敗: {e}")
            exec_checks['command_building'] = False
        
        # 檢查系統執行
        print("4.2 檢查系統執行...")
        try:
            import subprocess
            check_cmd = "echo SOP compliance check" if os.name != 'nt' else "echo SOP compliance check"
            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✅ 系統執行正常")
                exec_checks['system_execution'] = True
            else:
                print("❌ 系統執行失敗")
                exec_checks['system_execution'] = False
        except Exception as e:
            print(f"❌ 系統執行檢查失敗: {e}")
            exec_checks['system_execution'] = False
        
        self.check_results['command_execution'] = exec_checks
    
    async def generate_final_report(self):
        """生成最終報告"""
        print("\n" + "=" * 70)
        print("📊 SOP 合規性檢查結果總結")
        print("=" * 70)
        
        total_checks = 0
        passed_checks = 0
        
        for category, checks in self.check_results.items():
            category_total = len(checks)
            category_passed = sum(1 for v in checks.values() if v)
            total_checks += category_total
            passed_checks += category_passed
            
            print(f"\n{category.replace('_', ' ').title()} ({category_passed}/{category_total}):")
            for check_name, result in checks.items():
                status = "✅ 通過" if result else "❌ 失敗"
                check_display = check_name.replace('_', ' ').title()
                print(f"  {status} {check_display}")
        
        compliance_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        print(f"\n🎯 整體合規率: {compliance_rate:.1f}% ({passed_checks}/{total_checks})")
        
        if compliance_rate >= 90:
            print("🎉 系統高度符合 SOP 標準！")
        elif compliance_rate >= 70:
            print("⚠️  系統基本符合 SOP 標準，建議改善失敗項目")
        else:
            print("❌ 系統需要加強 SOP 合規性")
        
        print("=" * 70)
        
        # 保存報告
        report_file = f"sop_compliance_report_{Path(__file__).stem}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.check_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 詳細報告已保存: {report_file}")


async def main():
    """主函數"""
    if not IMPORT_SUCCESS:
        print("❌ 無法導入核心模組，檢查中止")
        return 1
    
    checker = AIVASOPComplianceChecker(project_root)
    await checker.run_comprehensive_check()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n⏹️  檢查被用戶中斷")
    except Exception as e:
        print(f"\n💥 檢查過程發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
