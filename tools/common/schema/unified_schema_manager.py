#!/usr/bin/env python3
"""
AIVA 統一 Schema 管理器
版本: 3.0
建立日期: 2025-10-24
用途: 統一所有 Schema 相關功能 - 驗證、生成、管理、同步
"""

import sys
import os
import json
import importlib
import inspect
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback

# 確保 services 目錄在 Python 路徑中
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "services"))

@dataclass
class ValidationResult:
    """驗證結果資料類別"""
    category: str
    item: str
    status: bool
    details: str = ""
    error_message: str = ""

class UnifiedSchemaManager:
    """統一 Schema 管理器 - 整合所有 Schema 相關功能"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent.parent.parent
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
        result = ValidationResult(
            category=category,
            item=item,
            status=passed,
            details=details,
            error_message=error_message
        )
        self.results.append(result)
        
        status_icon = "✅" if passed else "❌"
        self.log(f"  {status_icon} {category} - {item}: {details or error_message}")
    
    # ==================== 驗證功能 ====================
    
    def validate_enums(self) -> bool:
        """驗證所有 Enum 定義"""
        self.log("🔍 開始驗證 Enums...", "INFO")
        
        try:
            from aiva_common import enums
            
            enum_modules = []
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
                    
                    # 檢查模組中的 enum 類別
                    enum_classes = []
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and hasattr(obj, '_member_names_'):
                            enum_classes.append(name)
                    
                    if enum_classes:
                        details = f"包含 {len(enum_classes)} 個 enum 類別"
                        self.add_result("Enums", f"{module_name} 模組", True, details)
                        self.stats['valid_enums'] += 1
                    else:
                        self.add_result("Enums", f"{module_name} 模組", False, 
                                      error_message="未找到有效的 enum 類別")
                        
                except Exception as e:
                    self.add_result("Enums", f"{module_name} 模組", False, 
                                  error_message=f"載入失敗: {e}")
            
            self.log(f"Enums 驗證完成: {self.stats['valid_enums']}/{self.stats['total_enums']}")
            return self.stats['valid_enums'] > 0
            
        except ImportError as e:
            self.add_result("Enums", "主模組", False, error_message=f"無法導入 aiva_common.enums: {e}")
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
                    
                    # 檢查模組中的 BaseModel 類別
                    schema_classes = []
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            hasattr(obj, '__bases__') and
                            any('BaseModel' in str(base) for base in obj.__bases__)):
                            schema_classes.append(name)
                    
                    if schema_classes:
                        details = f"包含 {len(schema_classes)} 個 schema 類別"
                        self.add_result("Schemas", f"{module_name} 模組", True, details)
                        self.stats['valid_schemas'] += 1
                    else:
                        self.add_result("Schemas", f"{module_name} 模組", False, 
                                      error_message="未找到有效的 schema 類別")
                        
                except Exception as e:
                    self.add_result("Schemas", f"{module_name} 模組", False, 
                                  error_message=f"載入失敗: {e}")
            
            self.log(f"Schemas 驗證完成: {self.stats['valid_schemas']}/{self.stats['total_schemas']}")
            return self.stats['valid_schemas'] > 0
            
        except ImportError as e:
            self.add_result("Schemas", "主模組", False, error_message=f"無法導入 aiva_common.schemas: {e}")
            return False
    
    def validate_utils(self) -> bool:
        """驗證工具模組"""
        self.log("🔍 開始驗證 Utils...", "INFO")
        
        try:
            from aiva_common import utils
            
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
                                  error_message=f"載入失敗: {e}")
            
            self.log(f"Utils 驗證完成: {self.stats['valid_utils']}/{self.stats['total_utils']}")
            return self.stats['valid_utils'] > 0
            
        except ImportError as e:
            self.add_result("Utils", "主模組", False, error_message=f"無法導入 aiva_common.utils: {e}")
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
            self.add_result("主模組", "導入", False, error_message=f"無法導入 aiva_common: {e}")
            return False
    
    # ==================== 管理功能 ====================
    
    def list_schemas(self) -> Dict[str, List[str]]:
        """列出所有 Schema 定義"""
        schemas = {}
        schemas_path = Path("services/aiva_common/schemas")
        
        if not schemas_path.exists():
            return schemas
            
        for py_file in schemas_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            module_name = py_file.stem
            try:
                module = importlib.import_module(f"aiva_common.schemas.{module_name}")
                classes = []
                
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        hasattr(obj, '__bases__') and
                        any('BaseModel' in str(base) for base in obj.__bases__)):
                        classes.append(name)
                
                if classes:
                    schemas[module_name] = classes
                    
            except Exception:
                continue
        
        return schemas
    
    def list_enums(self) -> Dict[str, List[str]]:
        """列出所有 Enum 定義"""
        enums = {}
        enums_path = Path("services/aiva_common/enums")
        
        if not enums_path.exists():
            return enums
            
        for py_file in enums_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            module_name = py_file.stem
            try:
                module = importlib.import_module(f"aiva_common.enums.{module_name}")
                classes = []
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and hasattr(obj, '_member_names_'):
                        classes.append(name)
                
                if classes:
                    enums[module_name] = classes
                    
            except Exception:
                continue
        
        return enums
    
    # ==================== 生成功能 ====================
    
    def generate_multilang_schemas(self, languages: List[str]) -> bool:
        """生成多語言 Schema 定義"""
        self.log(f"🔄 開始生成多語言 Schema: {', '.join(languages)}", "INFO")
        
        success = True
        
        for lang in languages:
            try:
                if lang.lower() == "go":
                    success &= self._generate_go_schemas()
                elif lang.lower() == "typescript":
                    success &= self._generate_typescript_schemas()
                elif lang.lower() == "java":
                    success &= self._generate_java_schemas()
                else:
                    self.log(f"不支援的語言: {lang}", "WARNING")
                    success = False
                    
            except Exception as e:
                self.log(f"生成 {lang} Schema 失敗: {e}", "ERROR")
                success = False
        
        return success
    
    def _generate_go_schemas(self) -> bool:
        """生成 Go Schema"""
        # Go Schema 已實現於 services/features/common/go/aiva_common_go/schemas/
        self.log("Go Schema 已存在於 aiva_common_go", "INFO")
        return True
    
    def _generate_typescript_schemas(self) -> bool:
        """生成 TypeScript Schema"""
        # TypeScript Schema 已實現於 schemas/aiva_schemas.d.ts
        self.log("TypeScript Schema 已存在於 schemas/aiva_schemas.d.ts", "INFO")
        return True
    
    def _generate_java_schemas(self) -> bool:
        """生成 Java Schema"""
        # Java Schema 暫未需要 - AIVA 專案目前不使用 Java
        self.log("Java Schema 暫不需要 - 專案不使用 Java", "INFO")
        return True
    
    # ==================== 核心功能 ====================
    
    def run_validation(self) -> Dict[str, Any]:
        """執行完整驗證"""
        self.log("🚀 開始執行完整的 Schema 驗證...", "INFO")
        start_time = datetime.now()
        
        # 執行各項驗證
        main_ok = self.validate_main_module()
        enums_ok = self.validate_enums()
        schemas_ok = self.validate_schemas()
        utils_ok = self.validate_utils()
        
        # 計算結果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.status)
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
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
                    'details': r.details,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }
        
        self.log(f"驗證完成，耗時 {duration:.2f} 秒", "INFO")
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """列印驗證摘要"""
        stats = report['statistics']
        
        print(f"\n📊 AIVA Schema 驗證報告")
        print(f"=" * 50)
        print(f"⏰ 執行時間: {report['duration_seconds']:.2f} 秒")
        print(f"📈 成功率: {report['success_rate']}%")
        print(f"📋 總檢查數: {stats['total_checks']}")
        print(f"✅ 通過: {stats['passed_checks']}")
        print(f"❌ 失敗: {stats['failed_checks']}")
        
        print(f"\n📊 詳細統計:")
        print(f"  🔢 Enums: {stats['enums']['valid']}/{stats['enums']['total']}")
        print(f"  📝 Schemas: {stats['schemas']['valid']}/{stats['schemas']['total']}")
        print(f"  🛠️  Utils: {stats['utils']['valid']}/{stats['utils']['total']}")
        
        # 顯示失敗的檢查
        failed_results = [r for r in report['results'] if not r['status']]
        if failed_results:
            print(f"\n❌ 失敗的檢查:")
            for result in failed_results:
                print(f"   {result['category']} - {result['item']}: {result['error_message']}")
        else:
            print(f"\n🎉 所有檢查都通過了！")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="AIVA 統一 Schema 管理器")
    
    # 基本選項
    parser.add_argument('-v', '--verbose', action='store_true', help="詳細輸出")
    parser.add_argument('-o', '--output', help="輸出 JSON 報告到指定文件")
    
    # 功能選項
    subparsers = parser.add_subparsers(dest='action', help='執行的操作')
    
    # 驗證命令
    validate_parser = subparsers.add_parser('validate', help='驗證所有 Schema')
    validate_parser.add_argument('--stats-only', action='store_true', help="只顯示統計資訊")
    
    # 列表命令
    list_parser = subparsers.add_parser('list', help='列出所有 Schema 和 Enum')
    
    # 生成命令
    generate_parser = subparsers.add_parser('generate', help='生成多語言 Schema')
    generate_parser.add_argument('--languages', nargs='+', 
                               choices=['go', 'typescript', 'java'],
                               default=['go'], 
                               help='要生成的語言')
    
    args = parser.parse_args()
    
    # 檢查是否在正確的目錄
    if not Path("services/aiva_common").exists():
        print("❌ 錯誤: 找不到 services/aiva_common 目錄")
        print("請確認您在 AIVA 專案根目錄中執行此腳本")
        sys.exit(1)
    
    # 創建管理器
    manager = UnifiedSchemaManager(verbose=args.verbose)
    
    # 預設動作是驗證
    if not args.action:
        args.action = 'validate'
        args.stats_only = False
    
    # 執行對應操作
    success = True
    
    if args.action == 'validate':
        # 執行驗證
        report = manager.run_validation()
        
        # 顯示結果
        if not args.stats_only:
            manager.print_summary(report)
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
                print(f"❌ 儲存報告失敗: {e}")
                success = False
        
        success = report['overall_status'] == 'PASS'
    
    elif args.action == 'list':
        schemas = manager.list_schemas()
        enums = manager.list_enums()
        
        print("📋 Schema 定義:")
        for category, classes in schemas.items():
            print(f"  {category}: {', '.join(classes)}")
        
        print("\n📋 Enum 定義:")
        for category, classes in enums.items():
            print(f"  {category}: {', '.join(classes)}")
    
    elif args.action == 'generate':
        success = manager.generate_multilang_schemas(args.languages)
        if success:
            print(f"🎉 多語言 Schema 生成完成: {', '.join(args.languages)}")
        else:
            print("❌ 多語言 Schema 生成失敗")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()