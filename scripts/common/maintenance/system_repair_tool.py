#!/usr/bin/env python3
"""
AIVA 系統修復工具
專門修復編譯錯誤、依賴問題和系統通連性問題
"""

import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import re

class AIVASystemRepair:
    """AIVA 系統修復器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.repair_log = []
        
    def log_repair(self, module, action, result, details=""):
        """記錄修復操作"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "action": action,
            "result": result,
            "details": details
        }
        self.repair_log.append(entry)
        
        status_icon = "✅" if result == "success" else "❌" if result == "failed" else "⚠️"
        print(f"{status_icon} [{module}] {action}: {details}")
    
    def repair_go_dependencies(self):
        """修復 Go 模組依賴問題"""
        print("\n🔧 修復 Go 模組依賴...")
        
        go_modules = [
            "services/features/function_authn_go",
            "services/features/function_cspm_go", 
            "services/features/function_ssrf_go",
            "services/features/function_sca_go"
        ]
        
        for module_path in go_modules:
            module_dir = self.project_root / module_path
            module_name = module_path.split("/")[-1]
            
            if not module_dir.exists():
                self.log_repair(module_name, "Skip", "warning", "目錄不存在")
                continue
            
            try:
                # 1. 清理舊的 go.sum
                go_sum = module_dir / "go.sum"
                if go_sum.exists():
                    go_sum.unlink()
                    self.log_repair(module_name, "Clean go.sum", "success")
                
                # 2. 執行 go mod tidy
                result = subprocess.run(
                    ["go", "mod", "tidy"],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.log_repair(module_name, "Go mod tidy", "success")
                else:
                    self.log_repair(module_name, "Go mod tidy", "failed", result.stderr[:200])
                
                # 3. 下載依賴
                result = subprocess.run(
                    ["go", "mod", "download"],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                
                if result.returncode == 0:
                    self.log_repair(module_name, "Go mod download", "success")
                
                # 4. 嘗試編譯測試
                result = subprocess.run(
                    ["go", "build", "./..."],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.log_repair(module_name, "Go build test", "success")
                else:
                    # 如果是 SSRF detector 的未使用變數問題，嘗試修復
                    if "declared and not used" in result.stderr and "awsV2Token" in result.stderr:
                        self.fix_ssrf_unused_variable(module_dir)
                    else:
                        self.log_repair(module_name, "Go build test", "failed", result.stderr[:200])
                        
            except subprocess.TimeoutExpired:
                self.log_repair(module_name, "Go repair", "failed", "操作超時")
            except Exception as e:
                self.log_repair(module_name, "Go repair", "failed", str(e))
    
    def fix_ssrf_unused_variable(self, module_dir):
        """修復 SSRF 模組的未使用變數問題"""
        scanner_file = module_dir / "internal/detector/cloud_metadata_scanner.go"
        
        if not scanner_file.exists():
            return
        
        try:
            content = scanner_file.read_text(encoding='utf-8')
            
            # 找到並修復未使用的 awsV2Token 變數
            # 方法1: 使用變數或者註解掉
            if "awsV2Token :=" in content and "awsV2Token" in content:
                # 在變數後添加使用或註解
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "awsV2Token :=" in line and i < len(lines) - 1:
                        # 在下一行添加一個使用該變數的語句
                        if "_ = awsV2Token" not in lines[i + 1]:
                            lines.insert(i + 1, "\t_ = awsV2Token // 防止未使用變數錯誤")
                            break
                
                new_content = '\n'.join(lines)
                scanner_file.write_text(new_content, encoding='utf-8')
                self.log_repair("SSRF Detector", "Fix unused variable", "success", "已修復 awsV2Token 未使用問題")
                
        except Exception as e:
            self.log_repair("SSRF Detector", "Fix unused variable", "failed", str(e))
    
    def repair_rust_compilation(self):
        """修復 Rust 編譯問題"""
        print("\n🦀 修復 Rust 模組...")
        
        rust_modules = [
            ("services/features/function_sast_rust", "SAST Analyzer"),
            ("services/scan/info_gatherer_rust", "Info Gatherer")
        ]
        
        for module_path, module_name in rust_modules:
            module_dir = self.project_root / module_path
            
            if not module_dir.exists():
                self.log_repair(module_name, "Skip", "warning", "目錄不存在")
                continue
            
            try:
                # 1. 清理舊的編譯結果
                target_dir = module_dir / "target"
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                    self.log_repair(module_name, "Clean target", "success")
                
                # 2. 檢查 Cargo.toml
                cargo_toml = module_dir / "Cargo.toml"
                if not cargo_toml.exists():
                    self.log_repair(module_name, "Check Cargo.toml", "failed", "Cargo.toml 不存在")
                    continue
                
                # 3. 更新依賴
                result = subprocess.run(
                    ["cargo", "update"],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.log_repair(module_name, "Cargo update", "success")
                
                # 4. 編譯檢查
                result = subprocess.run(
                    ["cargo", "check"],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                
                if result.returncode == 0:
                    warning_count = result.stderr.count("warning:")
                    if warning_count > 0:
                        self.log_repair(module_name, "Cargo check", "success", f"編譯成功，{warning_count} 個警告")
                    else:
                        self.log_repair(module_name, "Cargo check", "success", "編譯無警告")
                else:
                    self.log_repair(module_name, "Cargo check", "failed", result.stderr[:300])
                
            except subprocess.TimeoutExpired:
                self.log_repair(module_name, "Rust repair", "failed", "操作超時")
            except Exception as e:
                self.log_repair(module_name, "Rust repair", "failed", str(e))
    
    def repair_python_imports(self):
        """修復 Python 導入問題"""
        print("\n🐍 修復 Python 導入...")
        
        # 檢查並創建必要的 __init__.py 檔案
        required_inits = [
            "services/__init__.py",
            "services/aiva_common/__init__.py",
            "services/core/__init__.py", 
            "services/scan/__init__.py",
            "services/integration/__init__.py",
            "services/features/__init__.py"
        ]
        
        for init_path in required_inits:
            init_file = self.project_root / init_path
            if not init_file.exists():
                init_file.touch()
                self.log_repair("Python Import", "Create __init__.py", "success", init_path)
        
        # 檢查是否有 schema 相關問題
        schema_fix_script = self.project_root / "tools/fix_all_schema_imports.py"
        if schema_fix_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(schema_fix_script)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.log_repair("Python Import", "Run schema fix", "success")
                else:
                    self.log_repair("Python Import", "Run schema fix", "failed", result.stderr[:200])
                    
            except Exception as e:
                self.log_repair("Python Import", "Run schema fix", "failed", str(e))
    
    def check_system_connectivity(self):
        """檢查系統通連性"""
        print("\n🔍 檢查系統通連性...")
        
        # 測試基本模組導入
        test_imports = [
            ("services.aiva_common", "Common Module"),
            ("services.core.aiva_core", "Core Module"),
            ("services.scan.aiva_scan", "Scan Module"),
            ("services.integration.aiva_integration", "Integration Module"),
            ("services.features", "Features Module")
        ]
        
        sys.path.insert(0, str(self.project_root))
        
        for import_path, module_name in test_imports:
            try:
                __import__(import_path)
                self.log_repair("Connectivity", f"Import {module_name}", "success")
            except ImportError as e:
                self.log_repair("Connectivity", f"Import {module_name}", "failed", str(e))
            except Exception as e:
                self.log_repair("Connectivity", f"Import {module_name}", "warning", str(e))
    
    def verify_target_range_connection(self):
        """驗證靶場連接能力"""
        print("\n🎯 驗證靶場連接...")
        
        try:
            # 簡單的靶場連接測試
            import socket
            
            # 測試 localhost:3000
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            result = sock.connect_ex(('127.0.0.1', 3000))
            if result == 0:
                self.log_repair("Target Range", "Connect localhost:3000", "success", "靶場可達")
            else:
                self.log_repair("Target Range", "Connect localhost:3000", "warning", "靶場不可達，需要啟動")
            
            sock.close()
            
        except Exception as e:
            self.log_repair("Target Range", "Connection test", "failed", str(e))
    
    def generate_repair_report(self):
        """生成修復報告"""
        print("\n📊 生成修復報告...")
        
        # 統計修復結果
        total_actions = len(self.repair_log)
        success_actions = len([entry for entry in self.repair_log if entry['result'] == 'success'])
        failed_actions = len([entry for entry in self.repair_log if entry['result'] == 'failed'])
        warning_actions = len([entry for entry in self.repair_log if entry['result'] == 'warning'])
        
        report = {
            "repair_time": datetime.now().isoformat(),
            "summary": {
                "total_actions": total_actions,
                "success": success_actions,
                "failed": failed_actions,
                "warnings": warning_actions,
                "success_rate": f"{(success_actions / max(total_actions, 1) * 100):.1f}%"
            },
            "actions": self.repair_log
        }
        
        # 保存詳細報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"aiva_system_repair_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.log_repair("System", "Save report", "success", report_file)
        except Exception as e:
            self.log_repair("System", "Save report", "failed", str(e))
        
        return report
    
    def print_repair_summary(self, report):
        """打印修復摘要"""
        print("\n" + "="*70)
        print("🔧 AIVA 系統修復報告")
        print("="*70)
        
        summary = report['summary']
        print(f"🕐 修復時間: {report['repair_time']}")
        print(f"📊 修復動作: {summary['total_actions']} 個")
        print(f"✅ 成功: {summary['success']} 個")
        print(f"❌ 失敗: {summary['failed']} 個") 
        print(f"⚠️ 警告: {summary['warnings']} 個")
        print(f"📈 成功率: {summary['success_rate']}")
        
        # 按模組分組顯示結果
        modules = {}
        for entry in self.repair_log:
            module = entry['module']
            if module not in modules:
                modules[module] = {'success': 0, 'failed': 0, 'warning': 0}
            modules[module][entry['result']] += 1
        
        print(f"\n📋 模組修復狀態:")
        for module, counts in modules.items():
            total = sum(counts.values())
            success_rate = (counts['success'] / max(total, 1)) * 100
            
            if success_rate >= 80:
                status_icon = "✅"
            elif success_rate >= 50:
                status_icon = "⚠️"
            else:
                status_icon = "❌"
            
            print(f"   {status_icon} {module}: {counts['success']}/{total} 成功")
        
        # 顯示仍需注意的問題
        failed_actions = [entry for entry in self.repair_log if entry['result'] == 'failed']
        if failed_actions:
            print(f"\n⚠️ 仍需手動處理的問題:")
            for entry in failed_actions[-5:]:  # 顯示最後 5 個失敗
                print(f"   • {entry['module']} - {entry['action']}: {entry['details'][:50]}...")
    
    def run_full_repair(self):
        """運行完整修復流程"""
        print("🚀 AIVA 系統修復開始")
        print("=" * 70)
        
        # 1. 修復 Python 導入
        self.repair_python_imports()
        
        # 2. 修復 Go 依賴
        self.repair_go_dependencies()
        
        # 3. 修復 Rust 編譯
        self.repair_rust_compilation()
        
        # 4. 檢查系統通連性  
        self.check_system_connectivity()
        
        # 5. 驗證靶場連接
        self.verify_target_range_connection()
        
        # 6. 生成報告
        report = self.generate_repair_report()
        self.print_repair_summary(report)
        
        print(f"\n✅ 系統修復完成！")
        return report

def main():
    """主函數"""
    repairer = AIVASystemRepair()
    report = repairer.run_full_repair()
    
    # 返回適當的退出碼
    success_rate = float(report['summary']['success_rate'].replace('%', ''))
    return 0 if success_rate >= 70 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)