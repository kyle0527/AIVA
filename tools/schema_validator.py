#!/usr/bin/env python3
"""
AIVA Schema 驗證和管理工具
版本: 2.0
建立日期: 2025-10-18
用途: 驗證 aiva_common 中的所有 Schema 和 Enum 定義
"""

import sys
import os
import json
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback

# 確保 services 目錄在 Python 路徑中
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

@dataclass
class ValidationResult:
    """驗證結果資料類別"""
    category: str
    item: str
    status: str  # "✅", "⚠️", "❌"
    passed: bool
    details: str = ""
    error_message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SchemaValidator:
    """Schema 驗證器主類別"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.stats = {
            'total_enums': 0,
            'valid_enums': 0,
            'total_schemas': 0,
            'valid_schemas': 0,
            'total_utils': 0,
            'valid_utils': 0
        }
    
    def log(self, message: str, level: str = "INFO"):
        """記錄訊息"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def add_result(self, category: str, item: str, passed: bool, 
                   details: str = "", error_message: str = ""):
        """新增驗證結果"""
        status = "✅" if passed else "❌"
        result = ValidationResult(
            category=category,
            item=item,
            status=status,
            passed=passed,
            details=details,
            error_message=error_message
        )
        self.results.append(result)
        
        if not passed and error_message:
            self.log(f"❌ {category} - {item}: {error_message}", "ERROR")
        elif passed:
            self.log(f"✅ {category} - {item}: {details}", "INFO")
    
    def validate_enums(self) -> bool:
        """驗證所有 Enum 定義"""
        self.log("🔍 開始驗證 Enums...", "INFO")
        
        try:
            from aiva_common import enums
            enum_modules = []
            
            # 獲取所有 enum 模組
            enums_path = Path("services/aiva_common/enums")
            if not enums_path.exists():
                self.add_result("Enums", "目錄結構", False, 
                              error_message="enums 目錄不存在")
                return False
            
            # 遍歷 enum 文件
            for py_file in enums_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                module_name = py_file.stem
                try:
                    module = importlib.import_module(f"aiva_common.enums.{module_name}")
                    enum_modules.append((module_name, module))
                    self.stats['total_enums'] += 1
                    
                    # 檢查模組中的 Enum 類別
                    enum_classes = []
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and hasattr(obj, '__members__'):
                            enum_classes.append(name)
                    
                    if enum_classes:
                        self.add_result("Enums", f"{module_name} 模組", True, 
                                      f"包含 {len(enum_classes)} 個 enum 類別")
                        self.stats['valid_enums'] += 1
                    else:
                        self.add_result("Enums", f"{module_name} 模組", False, 
                                      error_message="未找到 enum 類別")
                        
                except Exception as e:
                    self.add_result("Enums", f"{module_name} 模組", False, 
                                  error_message=str(e))
                    
            return self.stats['valid_enums'] > 0
            
        except ImportError as e:
            self.add_result("Enums", "模組匯入", False, 
                          error_message=f"無法匯入 aiva_common.enums: {e}")
            return False
    
    def validate_schemas(self) -> bool:
        """驗證所有 Schema 定義"""
        self.log("🔍 開始驗證 Schemas...", "INFO")
        
        try:
            from aiva_common import schemas
            schema_modules = []
            
            # 獲取所有 schema 模組
            schemas_path = Path("services/aiva_common/schemas")
            if not schemas_path.exists():
                self.add_result("Schemas", "目錄結構", False, 
                              error_message="schemas 目錄不存在")
                return False
            
            # 遍歷 schema 文件
            for py_file in schemas_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                module_name = py_file.stem
                try:
                    module = importlib.import_module(f"aiva_common.schemas.{module_name}")
                    schema_modules.append((module_name, module))
                    self.stats['total_schemas'] += 1
                    
                    # 檢查模組中的 Pydantic 模型
                    schema_classes = []
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj):
                            # 檢查是否為 Pydantic 模型
                            if hasattr(obj, '__pydantic_core_schema__') or hasattr(obj, '__fields__'):
                                schema_classes.append(name)
                    
                    if schema_classes:
                        self.add_result("Schemas", f"{module_name} 模組", True, 
                                      f"包含 {len(schema_classes)} 個 schema 類別")
                        self.stats['valid_schemas'] += 1
                    else:
                        self.add_result("Schemas", f"{module_name} 模組", False, 
                                      error_message="未找到 schema 類別")
                        
                except Exception as e:
                    self.add_result("Schemas", f"{module_name} 模組", False, 
                                  error_message=str(e))
                    
            return self.stats['valid_schemas'] > 0
            
        except ImportError as e:
            self.add_result("Schemas", "模組匯入", False, 
                          error_message=f"無法匯入 aiva_common.schemas: {e}")
            return False
    
    def validate_utils(self) -> bool:
        """驗證工具模組"""
        self.log("🔍 開始驗證 Utils...", "INFO")
        
        try:
            from aiva_common import utils
            
            # 獲取所有 utils 模組
            utils_path = Path("services/aiva_common/utils")
            if not utils_path.exists():
                self.add_result("Utils", "目錄結構", False, 
                              error_message="utils 目錄不存在")
                return False
            
            # 遍歷 utils 文件
            for py_file in utils_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                module_name = py_file.stem
                try:
                    module = importlib.import_module(f"aiva_common.utils.{module_name}")
                    self.stats['total_utils'] += 1
                    
                    # 檢查模組中的函數和類別
                    functions = []
                    classes = []
                    for name, obj in inspect.getmembers(module):
                        if not name.startswith("_"):
                            if inspect.isfunction(obj):
                                functions.append(name)
                            elif inspect.isclass(obj):
                                classes.append(name)
                    
                    if functions or classes:
                        details = f"函數: {len(functions)}, 類別: {len(classes)}"
                        self.add_result("Utils", f"{module_name} 模組", True, details)
                        self.stats['valid_utils'] += 1
                    else:
                        self.add_result("Utils", f"{module_name} 模組", False, 
                                      error_message="未找到公開的函數或類別")
                        
                except Exception as e:
                    self.add_result("Utils", f"{module_name} 模組", False, 
                                  error_message=str(e))
                    
            return self.stats['valid_utils'] > 0
            
        except ImportError as e:
            self.add_result("Utils", "模組匯入", False, 
                          error_message=f"無法匯入 aiva_common.utils: {e}")
            return False
    
    def validate_main_module(self) -> bool:
        """驗證主模組"""
        self.log("🔍 開始驗證主模組...", "INFO")
        
        try:
            import aiva_common
            
            # 檢查版本
            version = getattr(aiva_common, '__version__', 'unknown')
            self.add_result("主模組", "版本資訊", True, f"版本: {version}")
            
            # 檢查主要匯出
            expected_modules = ['enums', 'schemas', 'utils']
            for module_name in expected_modules:
                has_module = hasattr(aiva_common, module_name)
                self.add_result("主模組", f"{module_name} 模組", has_module, 
                              "可用" if has_module else "")
            
            return True
            
        except ImportError as e:
            self.add_result("主模組", "匯入測試", False, 
                          error_message=f"無法匯入 aiva_common: {e}")
            return False
    
    def run_validation(self) -> Dict[str, Any]:
        """執行完整驗證"""
        self.log("🚀 開始 AIVA Schema 驗證...", "INFO")
        start_time = datetime.now()
        
        # 執行各項驗證
        main_ok = self.validate_main_module()
        enums_ok = self.validate_enums()
        schemas_ok = self.validate_schemas()
        utils_ok = self.validate_utils()
        
        # 計算統計
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.passed])
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 生成報告
        report = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'overall_status': 'PASS' if success_rate >= 80 else 'FAIL',
            'success_rate': round(success_rate, 1),
            'statistics': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': total_checks - passed_checks,
                'enums': {
                    'total': self.stats['total_enums'],
                    'valid': self.stats['valid_enums']
                },
                'schemas': {
                    'total': self.stats['total_schemas'],
                    'valid': self.stats['valid_schemas']
                },
                'utils': {
                    'total': self.stats['total_utils'],
                    'valid': self.stats['valid_utils']
                }
            },
            'results': [
                {
                    'category': r.category,
                    'item': r.item,
                    'status': r.status,
                    'passed': r.passed,
                    'details': r.details,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """列印驗證摘要"""
        print("\n" + "="*60)
        print("🔍 AIVA Schema 驗證報告")
        print("="*60)
        
        status_color = "🟢" if report['overall_status'] == 'PASS' else "🔴"
        print(f"{status_color} 整體狀態: {report['overall_status']}")
        print(f"📊 成功率: {report['success_rate']}%")
        print(f"⏱️ 執行時間: {report['duration_seconds']:.1f} 秒")
        
        stats = report['statistics']
        print(f"\n📋 詳細統計:")
        print(f"   總檢查項目: {stats['total_checks']}")
        print(f"   通過項目: {stats['passed_checks']}")
        print(f"   失敗項目: {stats['failed_checks']}")
        
        print(f"\n📦 模組統計:")
        print(f"   Enums: {stats['enums']['valid']}/{stats['enums']['total']} 有效")
        print(f"   Schemas: {stats['schemas']['valid']}/{stats['schemas']['total']} 有效")
        print(f"   Utils: {stats['utils']['valid']}/{stats['utils']['total']} 有效")
        
        # 顯示失敗項目
        failed_results = [r for r in report['results'] if not r['passed']]
        if failed_results:
            print(f"\n❌ 失敗項目:")
            for result in failed_results:
                print(f"   {result['category']} - {result['item']}: {result['error_message']}")
        else:
            print(f"\n🎉 所有檢查都通過了！")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Schema 驗證工具")
    parser.add_argument('-v', '--verbose', action='store_true', help="詳細輸出")
    parser.add_argument('-o', '--output', help="輸出 JSON 報告到指定文件")
    parser.add_argument('--stats-only', action='store_true', help="只顯示統計資訊")
    
    args = parser.parse_args()
    
    # 檢查是否在正確的目錄
    if not Path("services/aiva_common").exists():
        print("❌ 錯誤: 找不到 services/aiva_common 目錄")
        print("請確認您在 AIVA 專案根目錄中執行此腳本")
        sys.exit(1)
    
    # 執行驗證
    validator = SchemaValidator(verbose=args.verbose)
    report = validator.run_validation()
    
    # 顯示結果
    if not args.stats_only:
        validator.print_summary(report)
    else:
        stats = report['statistics']
        print(f"Enums: {stats['enums']['valid']}/{stats['enums']['total']}")
        print(f"Schemas: {stats['schemas']['valid']}/{stats['schemas']['total']}")
        print(f"Utils: {stats['utils']['valid']}/{stats['utils']['total']}")
        print(f"Success Rate: {report['success_rate']}%")
    
    # 儲存報告
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n💾 報告已儲存至: {args.output}")
        except Exception as e:
            print(f"\n❌ 無法儲存報告: {e}")
    
    # 設定退出碼
    sys.exit(0 if report['overall_status'] == 'PASS' else 1)

if __name__ == "__main__":
    main()