#!/usr/bin/env python3
"""
AIVA 系統通連及定義檢查 (按照 SCHEMA_MANAGEMENT_SOP.md 標準) - v1.1 (Import 修復)

遵循單一真實來源原則和分層責任架構進行全面檢查
"""

import asyncio
import sys
import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple
import aiofiles
import logging
import traceback

# ================== Import 修復 Start ==================
# 計算專案根目錄 (AIVA-main) 的絕對路徑
# __file__ 是目前腳本的路徑 (e.g., /path/to/AIVA-main/aiva_system_connectivity_sop_check.py)
# .parent 會得到 /path/to/AIVA-main
project_root = Path(__file__).parent.resolve()

# 將專案根目錄添加到 sys.path 的最前面，優先於其他路徑
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 驗證是否添加成功 (可選)
# print(f"[*] Project root added to sys.path: {project_root}")
# print(f"[*] Current sys.path: {sys.path}")
# ================== Import 修復 End ====================

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 現在可以嘗試導入 services 模組了
try:
    from services.core.aiva_core.ai_engine import (
        BioNeuronRAGAgent, AIModelManager, Tool, CodeReader, CodeWriter,
        CodeAnalyzer, CommandExecutor, ScanTrigger, VulnerabilityDetector
    )
    from services.core.aiva_core.learning import (
        ModelTrainer, ScalableBioTrainer,
        ScalableBioTrainingConfig
    )
    from services.aiva_common.ai import AIVAExperienceManager as ExperienceManager
    from services.core.aiva_core.ai_engine import PerformanceConfig, MemoryManager, ComponentPool
    # 嘗試導入 aiva_common (如果前面 sys.path 設置正確，這裡應該能成功)
    import services.aiva_common
    IMPORT_SUCCESS = True
    logger.info("✅ 核心 Python 模組導入成功")
except ImportError as e:
    logger.error(f"❌ 核心 Python 模組導入失敗: {e}")
    logger.error("   請確認您的 Python 環境以及 AIVA-main 目錄結構是否正確。")
    logger.error(f"   目前的 sys.path: {sys.path}")
    IMPORT_SUCCESS = False
except Exception as e:
    logger.error(f"❌ 導入過程中發生非預期的錯誤: {e}")
    logger.error(traceback.format_exc())
    IMPORT_SUCCESS = False

class AIVASystemConnectivityChecker:
    """AIVA 系統連通性檢查器 (遵循 SOP 標準)"""
    
    def __init__(self, aiva_root: Path):
        self.aiva_root = aiva_root
        self.schemas_dir = aiva_root / "services" / "aiva_common" / "schemas"
        self.enums_dir = aiva_root / "services" / "aiva_common" / "enums"
        self.generated_schemas_dir = aiva_root / "schemas"
        self.ai_core_dir = aiva_root / "services" / "core" / "aiva_core"
        
        self.check_results = {}
    
    async def run_comprehensive_check(self):
        """執行全面的系統連通性和定義檢查"""
        print("🔍 AIVA 系統通連及定義檢查 (SOP 標準)")
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
        
        # 5. 多語言轉換檢查
        await self.check_multilang_generation()
        
        # 生成報告
        await self.generate_final_report()
    
    async def check_schema_definitions(self):
        """檢查 Schema 定義體系 (按照 SOP 第 2 章)"""
        print("\n📋 1. Schema 定義體系檢查")
        print("-" * 50)
        
        schema_checks = {}
        
        # 1.1 檢查權威定義來源
        print("1.1 檢查權威定義來源...")
        try:
            # 檢查核心 Schema 文件
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
        
        # 1.2 檢查 Enum 定義
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
        
        # 1.3 檢查導入導出完整性
        print("1.3 檢查導入導出完整性...")
        try:
            # 動態導入 aiva_common 檢查
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
        
        # 2.1 檢查 AI 引擎核心
        print("2.1 檢查 AI 引擎核心...")
        try:
            from services.core.aiva_core.ai_engine import (
                BioNeuronRAGAgent, ScalableBioNet, 
                AIModelManager, OptimizedScalableBioNet
            )
            print("✅ AI 引擎核心模組載入成功")
            ai_checks['ai_engine_core'] = True
        except Exception as e:
            print(f"❌ AI 引擎核心載入失敗: {e}")
            ai_checks['ai_engine_core'] = False
        
        # 2.2 檢查統一訓練系統
        print("2.2 檢查統一訓練系統...")
        try:
            from services.core.aiva_core.learning import (
                ModelTrainer, ScalableBioTrainer, 
                ScalableBioTrainingConfig
            )
            from services.aiva_common.ai import AIVAExperienceManager as ExperienceManager
            print("✅ 統一訓練系統載入成功")
            ai_checks['training_system'] = True
        except Exception as e:
            print(f"❌ 統一訓練系統載入失敗: {e}")
            ai_checks['training_system'] = False
        
        # 2.3 檢查性能優化組件
        print("2.3 檢查性能優化組件...")
        try:
            from services.core.aiva_core.ai_engine import (
                PerformanceConfig, MemoryManager, ComponentPool
            )
            print("✅ 性能優化組件載入成功")
            ai_checks['performance_optimization'] = True
        except Exception as e:
            print(f"❌ 性能優化組件載入失敗: {e}")
            ai_checks['performance_optimization'] = False
        
        # 2.4 測試 AI 模型基本功能
        print("2.4 測試 AI 模型基本功能...")
        try:
            # 初始化 AI 管理器
            manager = None
            manager_imported = False
            try:
                from services.core.aiva_core.ai_engine import AIModelManager
                manager = AIModelManager(model_dir=Path("./test_models"))
                manager_imported = True
            except Exception as e:
                print(f"  ⚠️  AIModelManager 導入問題: {e}")
            
            if manager_imported and manager is not None:
                # 測試模型初始化
                init_result = await manager.initialize_models(input_size=32, num_tools=4)
                if init_result.get('status') == 'success':
                    print("✅ AI 模型基本功能正常")
                    ai_checks['ai_basic_function'] = True
                else:
                    print(f"❌ AI 模型初始化失敗: {init_result.get('error')}")
                    ai_checks['ai_basic_function'] = False
            else:
                print("❌ 無法測試 AI 基本功能")
                ai_checks['ai_basic_function'] = False
                
        except Exception as e:
            print(f"❌ AI 基本功能測試失敗: {e}")
            ai_checks['ai_basic_function'] = False
        
        self.check_results['ai_core_modules'] = ai_checks
    
    async def check_system_tools_connectivity(self):
        """檢查系統工具連接性"""
        print("\n🔧 3. 系統工具連接檢查")
        print("-" * 50)
        
        tools_checks = {}
        
        # 3.1 檢查工具類別導入
        print("3.1 檢查工具類別導入...")
        try:
            from services.core.aiva_core.ai_engine import (
                Tool, CodeReader, CodeWriter, CodeAnalyzer,
                CommandExecutor, ScanTrigger, VulnerabilityDetector
            )
            print("✅ 工具類別導入成功")
            tools_checks['tools_import'] = True
        except Exception as e:
            print(f"❌ 工具類別導入失敗: {e}")
            tools_checks['tools_import'] = False
            return
        
        # 3.2 檢查工具實例化 (帶預設參數)
        print("3.2 檢查工具實例化...")
        try:
            # 使用當前目錄作為預設路徑
            current_dir = str(Path.cwd())
            
            # 測試各個工具的實例化
            tools_to_test = [
                ("CodeReader", lambda: CodeReader(current_dir)),
                ("CodeWriter", lambda: CodeWriter(current_dir)),
                ("CodeAnalyzer", lambda: CodeAnalyzer(current_dir)),
                ("CommandExecutor", lambda: CommandExecutor(current_dir)),
                ("ScanTrigger", lambda: ScanTrigger()),
                ("VulnerabilityDetector", lambda: VulnerabilityDetector())
            ]
            
            successful_tools = []
            failed_tools = []
            
            for tool_name, tool_factory in tools_to_test:
                try:
                    tool_instance = tool_factory()
                    successful_tools.append(tool_name)
                    print(f"  ✅ {tool_name} 實例化成功")
                except Exception as e:
                    failed_tools.append((tool_name, str(e)))
                    print(f"  ❌ {tool_name} 實例化失敗: {e}")
            
            if len(successful_tools) >= 4:  # 大部分工具成功
                print("✅ 工具實例化基本正常")
                tools_checks['tools_instantiation'] = True
            else:
                print(f"❌ 工具實例化問題較多: {failed_tools}")
                tools_checks['tools_instantiation'] = False
                
        except Exception as e:
            print(f"❌ 工具實例化檢查失敗: {e}")
            tools_checks['tools_instantiation'] = False
        
        # 3.3 檢查文件系統訪問
        print("3.3 檢查文件系統訪問...")
        try:
            # 創建測試文件
            test_file = Path("./test_system_connectivity.tmp")
            test_content = "# AIVA 系統連通性測試\\nprint('System connectivity test')"
            
            # 寫入測試
            test_file.write_text(test_content, encoding='utf-8')
            
            # 讀取測試
            read_content = test_file.read_text(encoding='utf-8')
            
            if test_content == read_content:
                print("✅ 文件系統訪問正常")
                tools_checks['file_system_access'] = True
            else:
                print("❌ 文件系統讀寫內容不一致")
                tools_checks['file_system_access'] = False
            
            # 清理
            if test_file.exists():
                test_file.unlink()
                
        except Exception as e:
            print(f"❌ 文件系統訪問檢查失敗: {e}")
            tools_checks['file_system_access'] = False
        
        self.check_results['system_tools'] = tools_checks
    
    async def check_command_execution_chain(self):
        """檢查命令執行鏈"""
        print("\n⚡ 4. 命令執行鏈檢查")
        print("-" * 50)
        
        exec_checks = {}
        
        # 4.1 檢查基本命令執行
        print("4.1 檢查基本命令執行...")
        try:
            import subprocess
            
            # 測試命令列表
            if os.name == 'nt':  # Windows
                test_commands = [
                    ("echo", ["cmd", "/c", "echo", "AIVA system test"], "系統回音測試"),
                    ("python_version", ["python", "--version"], "Python 版本檢查"),
                    ("dir_list", ["cmd", "/c", "dir"], "目錄列表"),
                ]
            else:  # Unix/Linux
                test_commands = [
                    ("echo", ["echo", "AIVA system test"], "系統回音測試"),
                    ("python_version", ["python", "--version"], "Python 版本檢查"),
                    ("dir_list", ["ls"], "目錄列表"),
                ]
            
            successful_commands = 0
            for cmd_name, cmd_args, description in test_commands:
                try:
                    result = subprocess.run(
                        cmd_args,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        print(f"  ✅ {description} 成功")
                        successful_commands += 1
                    else:
                        print(f"  ⚠️  {description} 警告 (返回碼: {result.returncode})")
                        
                except Exception as e:
                    print(f"  ❌ {description} 失敗: {e}")
            
            if successful_commands >= 2:
                print("✅ 基本命令執行正常")
                exec_checks['basic_command_execution'] = True
            else:
                print("❌ 基本命令執行異常")
                exec_checks['basic_command_execution'] = False
                
        except Exception as e:
            print(f"❌ 基本命令執行檢查失敗: {e}")
            exec_checks['basic_command_execution'] = False
        
        # 4.2 檢查 AI → 系統 決策執行鏈
        print("4.2 檢查 AI → 系統 決策執行鏈...")
        try:
            # 簡化的決策執行測試
            decision_made = False
            command_executed = False
            
            # 1. 模擬 AI 決策
            try:
                from services.core.aiva_core.ai_engine import AIModelManager
                manager = AIModelManager()
                await manager.initialize_models(input_size=16, num_tools=3)
                
                # 執行決策 (不依賴 RAG)
                decision_result = await manager.make_decision(
                    "執行系統檢查",
                    {"type": "connectivity_test"},
                    use_rag=False
                )
                
                if decision_result.get('status') == 'success':
                    decision_made = True
                    print("  ✅ AI 決策層正常")
                else:
                    print(f"  ❌ AI 決策失敗: {decision_result.get('error')}")
                    
            except Exception as e:
                print(f"  ❌ AI 決策層失敗: {e}")
            
            # 2. 系統命令執行
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "-c", "print('AI-triggered system execution')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    command_executed = True
                    print("  ✅ 系統執行層正常")
                else:
                    print("  ❌ 系統執行層失敗")
                    
            except Exception as e:
                print(f"  ❌ 系統執行層失敗: {e}")
            
            # 3. 整體評估
            if decision_made and command_executed:
                print("✅ AI → 系統 決策執行鏈正常")
                exec_checks['ai_system_chain'] = True
            else:
                print("❌ AI → 系統 決策執行鏈存在問題")
                exec_checks['ai_system_chain'] = False
                
        except Exception as e:
            print(f"❌ 決策執行鏈檢查失敗: {e}")
            exec_checks['ai_system_chain'] = False
        
        self.check_results['command_execution'] = exec_checks
    
    async def check_multilang_generation(self):
        """檢查多語言轉換功能 (按照 SOP 第 5 章)"""
        print("\n🌐 5. 多語言轉換檢查")
        print("-" * 50)
        
        multilang_checks = {}
        
        # 5.1 檢查生成工具可用性
        print("5.1 檢查生成工具可用性...")
        try:
            import subprocess
            
            # 檢查 PowerShell
            ps_result = subprocess.run(
                ["pwsh", "-Command", "Write-Host 'PowerShell available'"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if ps_result.returncode == 0:
                print("  ✅ PowerShell 可用")
                multilang_checks['powershell_available'] = True
            else:
                print("  ❌ PowerShell 不可用")
                multilang_checks['powershell_available'] = False
            
            # 檢查生成腳本
            generate_script = self.aiva_root / "tools" / "generate-official-contracts.ps1"
            if generate_script.exists():
                print("  ✅ 官方生成腳本存在")
                multilang_checks['generate_script_exists'] = True
            else:
                print("  ❌ 官方生成腳本不存在")
                multilang_checks['generate_script_exists'] = False
                
        except Exception as e:
            print(f"❌ 生成工具檢查失敗: {e}")
            multilang_checks['powershell_available'] = False
            multilang_checks['generate_script_exists'] = False
        
        # 5.2 檢查已生成的多語言文件
        print("5.2 檢查已生成的多語言文件...")
        try:
            expected_files = [
                "aiva_schemas.json",
                "aiva_schemas.d.ts", 
                "enums.ts",
                "aiva_schemas.go",
                "aiva_schemas.rs"
            ]
            
            existing_files = []
            missing_files = []
            
            for file_name in expected_files:
                file_path = self.generated_schemas_dir / file_name
                if file_path.exists():
                    existing_files.append(file_name)
                else:
                    missing_files.append(file_name)
            
            print(f"  ✅ 已存在文件: {existing_files}")
            if missing_files:
                print(f"  ⚠️  缺少文件: {missing_files}")
            
            if len(existing_files) >= 3:  # 至少有3個語言文件
                print("✅ 多語言文件基本齊全")
                multilang_checks['multilang_files'] = True
            else:
                print("❌ 多語言文件不足")
                multilang_checks['multilang_files'] = False
                
        except Exception as e:
            print(f"❌ 多語言文件檢查失敗: {e}")
            multilang_checks['multilang_files'] = False
        
        self.check_results['multilang_generation'] = multilang_checks
    
    async def generate_final_report(self):
        """生成最終檢查報告"""
        print("\n" + "=" * 70)
        print("📊 AIVA 系統通連及定義檢查報告")
        print("=" * 70)
        
        # 統計各模組檢查結果
        module_stats = {}
        total_checks = 0
        passed_checks = 0
        
        for module_name, checks in self.check_results.items():
            module_passed = sum(1 for result in checks.values() if result)
            module_total = len(checks)
            module_stats[module_name] = (module_passed, module_total)
            
            total_checks += module_total
            passed_checks += module_passed
        
        # 顯示詳細結果
        print(f"\\n📋 詳細檢查結果:")
        for module_name, (passed, total) in module_stats.items():
            percentage = (passed / total * 100) if total > 0 else 0
            module_display = module_name.replace('_', ' ').title()
            print(f"  {module_display}: {passed}/{total} ({percentage:.1f}%)")
            
            # 顯示失敗項目
            failed_items = [
                check_name for check_name, result 
                in self.check_results[module_name].items() 
                if not result
            ]
            
            if failed_items:
                print(f"    ❌ 失敗項目: {', '.join(failed_items)}")
        
        # 整體評估
        overall_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        print(f"\\n🎯 整體系統通連性: {passed_checks}/{total_checks} ({overall_percentage:.1f}%)")
        
        # 給出建議
        if overall_percentage >= 85:
            print("🎉 系統通連性優秀！可以進行實戰靶場測試")
            recommendation = "READY_FOR_PRODUCTION"
        elif overall_percentage >= 70:
            print("✅ 系統通連性良好，建議修復少數問題後進行測試")
            recommendation = "READY_WITH_MINOR_FIXES"
        elif overall_percentage >= 50:
            print("⚠️  系統通連性基本可用，需要修復關鍵問題")
            recommendation = "NEEDS_MAJOR_FIXES"
        else:
            print("❌ 系統通連性存在嚴重問題，需要全面檢修")
            recommendation = "NEEDS_COMPLETE_OVERHAUL"
        
        # 保存檢查報告 (使用異步文件操作)
        report_data = {
            "timestamp": "2025-01-18",
            "overall_percentage": overall_percentage,
            "recommendation": recommendation,
            "module_stats": module_stats,
            "detailed_results": self.check_results
        }
        
        report_file = self.aiva_root / "SYSTEM_CONNECTIVITY_REPORT.json"
        async with aiofiles.open(report_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report_data, indent=2, ensure_ascii=False))
        
        print(f"\\n📄 詳細報告已保存: {report_file}")
        print("=" * 70)

async def main():
    """主函數"""
    aiva_root = Path(__file__).parent
    checker = AIVASystemConnectivityChecker(aiva_root)
    await checker.run_comprehensive_check()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n⏹️  檢查被用戶中斷")
    except Exception as e:
        print(f"\\n💥 檢查過程發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
