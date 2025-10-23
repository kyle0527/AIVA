#!/usr/bin/env python3
"""
AIVA 補包快速驗證工具
====================

此工具用於快速驗證補包的完整性和系統準備狀態

使用方式:
    python aiva_package_validator.py
    python aiva_package_validator.py --detailed
    python aiva_package_validator.py --export-report
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class AIVAPackageValidator:
    """AIVA補包驗證器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.validation_results = {}
        
    def validate_schema_system(self) -> dict:
        """驗證Schema自動化系統"""
        results = {
            'status': 'unknown',
            'details': {},
            'errors': []
        }
        
        try:
            # 檢查核心SOT檔案
            sot_file = self.project_root / "services/aiva_common/core_schema_sot.yaml"
            results['details']['sot_file'] = sot_file.exists()
            
            # 檢查工具檔案
            tools = [
                "services/aiva_common/tools/schema_codegen_tool.py",
                "services/aiva_common/tools/schema_validator.py", 
                "services/aiva_common/tools/module_connectivity_tester.py"
            ]
            
            tool_status = {}
            for tool in tools:
                tool_path = self.project_root / tool
                tool_status[Path(tool).name] = tool_path.exists()
            
            results['details']['tools'] = tool_status
            
            # 檢查生成的Schema
            generated_py = self.project_root / "services/aiva_common/schemas/generated"
            generated_go = self.project_root / "services/features/common/go/aiva_common_go/schemas/generated"
            
            py_schemas = list(generated_py.glob("*.py")) if generated_py.exists() else []
            go_schemas = list(generated_go.glob("*.go")) if generated_go.exists() else []
            
            results['details']['generated_schemas'] = {
                'python_count': len(py_schemas),
                'go_count': len(go_schemas),
                'python_files': [f.name for f in py_schemas],
                'go_files': [f.name for f in go_schemas]
            }
            
            # 判定整體狀態
            if (results['details']['sot_file'] and 
                all(tool_status.values()) and 
                len(py_schemas) >= 4 and len(go_schemas) >= 1):
                results['status'] = 'healthy'
            else:
                results['status'] = 'incomplete'
                
        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(str(e))
            
        return results
    
    def validate_module_structure(self) -> dict:
        """驗證五大模組結構"""
        results = {
            'status': 'unknown',
            'modules': {},
            'summary': {}
        }
        
        modules = {
            'AI核心引擎': 'services/core/aiva_core/ai_engine',
            '攻擊執行引擎': 'services/core/aiva_core/attack',
            '掃描引擎': 'services/scan',
            '整合服務': 'services/integration', 
            '功能檢測': 'services/features'
        }
        
        total_modules = len(modules)
        healthy_modules = 0
        
        for name, path in modules.items():
            module_path = self.project_root / path
            
            if module_path.exists():
                py_files = len(list(module_path.rglob("*.py")))
                go_files = len(list(module_path.rglob("*.go")))
                rs_files = len(list(module_path.rglob("*.rs")))
                
                results['modules'][name] = {
                    'exists': True,
                    'python_files': py_files,
                    'go_files': go_files,
                    'rust_files': rs_files,
                    'healthy': py_files > 0
                }
                
                if py_files > 0:
                    healthy_modules += 1
            else:
                results['modules'][name] = {
                    'exists': False,
                    'healthy': False
                }
        
        results['summary'] = {
            'total': total_modules,
            'healthy': healthy_modules,
            'health_rate': (healthy_modules / total_modules) * 100
        }
        
        results['status'] = 'healthy' if healthy_modules == total_modules else 'incomplete'
        
        return results
    
    def validate_phase_i_readiness(self) -> dict:
        """驗證Phase I準備狀態"""
        results = {
            'status': 'unknown',
            'readiness_factors': {},
            'blockers': []
        }
        
        # 檢查關鍵依賴
        factors = {
            'schema_system': self.validate_schema_system()['status'] == 'healthy',
            'module_structure': self.validate_module_structure()['status'] == 'healthy',
            'documentation': (self.project_root / "AIVA_PHASE_0_COMPLETE_PHASE_I_ROADMAP.md").exists(),
            'tools_available': all([
                (self.project_root / "services/aiva_common/tools/schema_codegen_tool.py").exists(),
                (self.project_root / "services/aiva_common/tools/schema_validator.py").exists()
            ])
        }
        
        results['readiness_factors'] = factors
        
        # 識別阻塞因素
        for factor, status in factors.items():
            if not status:
                results['blockers'].append(factor)
        
        # 判定整體準備狀態
        ready_count = sum(factors.values())
        total_count = len(factors)
        
        if ready_count == total_count:
            results['status'] = 'ready'
        elif ready_count >= total_count * 0.8:
            results['status'] = 'mostly_ready'
        else:
            results['status'] = 'not_ready'
            
        return results
    
    def run_connectivity_test(self) -> dict:
        """執行快速通連性測試"""
        results = {
            'status': 'unknown',
            'test_results': {},
            'errors': []
        }
        
        try:
            # 嘗試導入核心Schema
            sys.path.insert(0, str(self.project_root))
            
            # 測試基礎Schema導入
            try:
                from services.aiva_common.schemas.generated.base_types import MessageHeader
                results['test_results']['base_schema_import'] = True
            except ImportError as e:
                results['test_results']['base_schema_import'] = False
                results['errors'].append(f"基礎Schema導入失敗: {e}")
            
            # 測試消息Schema導入  
            try:
                from services.aiva_common.schemas.generated.messaging import AivaMessage
                results['test_results']['messaging_schema_import'] = True
            except ImportError as e:
                results['test_results']['messaging_schema_import'] = False
                results['errors'].append(f"消息Schema導入失敗: {e}")
            
            # 判定狀態
            passed_tests = sum(results['test_results'].values())
            total_tests = len(results['test_results'])
            
            if passed_tests == total_tests:
                results['status'] = 'passed'
            elif passed_tests > 0:
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
                
        except Exception as e:
            results['status'] = 'error'  
            results['errors'].append(f"測試執行錯誤: {e}")
            
        return results
    
    def generate_validation_report(self) -> dict:
        """生成完整驗證報告"""
        report = {
            'validation_time': datetime.now().isoformat(),
            'package_version': 'v2.5.1',
            'overall_status': 'unknown',
            'components': {}
        }
        
        # 執行所有驗證
        print("🔍 驗證Schema自動化系統...")
        report['components']['schema_system'] = self.validate_schema_system()
        
        print("🏗️ 驗證模組結構...")
        report['components']['module_structure'] = self.validate_module_structure()
        
        print("🚀 檢查Phase I準備狀態...")
        report['components']['phase_i_readiness'] = self.validate_phase_i_readiness()
        
        print("📡 執行通連性測試...")
        report['components']['connectivity_test'] = self.run_connectivity_test()
        
        # 計算整體狀態
        component_scores = {
            'schema_system': 1 if report['components']['schema_system']['status'] == 'healthy' else 0,
            'module_structure': 1 if report['components']['module_structure']['status'] == 'healthy' else 0,
            'phase_i_readiness': 1 if report['components']['phase_i_readiness']['status'] in ['ready', 'mostly_ready'] else 0,
            'connectivity_test': 1 if report['components']['connectivity_test']['status'] == 'passed' else 0
        }
        
        total_score = sum(component_scores.values())
        max_score = len(component_scores)
        
        if total_score == max_score:
            report['overall_status'] = 'excellent'
        elif total_score >= max_score * 0.8:
            report['overall_status'] = 'good'
        elif total_score >= max_score * 0.6:
            report['overall_status'] = 'acceptable'
        else:
            report['overall_status'] = 'needs_improvement'
            
        report['score'] = f"{total_score}/{max_score}"
        
        return report
    
    def print_summary_report(self, report: dict):
        """印出摘要報告"""
        print("\n" + "="*60)
        print("📋 AIVA補包驗證報告摘要")
        print("="*60)
        
        status_icons = {
            'excellent': '🟢 優秀',
            'good': '🟡 良好', 
            'acceptable': '🟠 可接受',
            'needs_improvement': '🔴 需改善'
        }
        
        print(f"⏰ 驗證時間: {report['validation_time']}")
        print(f"📦 補包版本: {report['package_version']}")
        print(f"🎯 整體狀態: {status_icons.get(report['overall_status'], report['overall_status'])}")
        print(f"📊 評分: {report['score']}")
        
        print("\n📋 組件狀態:")
        
        for component, details in report['components'].items():
            status = details['status']
            if status in ['healthy', 'ready', 'passed', 'excellent']:
                icon = "✅"
            elif status in ['mostly_ready', 'partial', 'good']:
                icon = "⚠️"
            else:
                icon = "❌"
                
            component_names = {
                'schema_system': 'Schema自動化系統',
                'module_structure': '五大模組結構',
                'phase_i_readiness': 'Phase I準備狀態',
                'connectivity_test': '通連性測試'
            }
            
            name = component_names.get(component, component)
            print(f"  {icon} {name}: {status}")
        
        # 顯示關鍵統計
        if 'module_structure' in report['components']:
            module_summary = report['components']['module_structure']['summary']
            print(f"\n📊 模組統計: {module_summary['healthy']}/{module_summary['total']} 模組健康")
        
        if 'schema_system' in report['components']:
            schema_details = report['components']['schema_system']['details']
            if 'generated_schemas' in schema_details:
                py_count = schema_details['generated_schemas']['python_count']
                go_count = schema_details['generated_schemas']['go_count']
                print(f"📋 Schema統計: {py_count} Python + {go_count} Go 檔案")
        
        # 顯示準備狀態
        if report['overall_status'] == 'excellent':
            print(f"\n🎉 補包狀態完美！可立即開始Phase I開發")
        elif report['overall_status'] == 'good':
            print(f"\n✅ 補包狀態良好，建議解決小問題後開始Phase I")
        else:
            print(f"\n⚠️ 補包需要改善，請檢查問題項目")


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA補包驗證工具")
    parser.add_argument("--detailed", action="store_true", help="顯示詳細報告")
    parser.add_argument("--export-report", action="store_true", help="匯出JSON報告")
    
    args = parser.parse_args()
    
    validator = AIVAPackageValidator()
    report = validator.generate_validation_report()
    
    # 顯示摘要
    validator.print_summary_report(report)
    
    # 顯示詳細資訊
    if args.detailed:
        print(f"\n📄 詳細報告:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    
    # 匯出報告
    if args.export_report:
        report_file = f"aiva_package_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📄 詳細報告已匯出: {report_file}")
    
    # 返回狀態碼
    success = report['overall_status'] in ['excellent', 'good']
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())