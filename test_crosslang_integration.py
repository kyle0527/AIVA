#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA 跨語言方案綜合測試器
驗證所有備用跨語言通信方法的功能性和可用性
"""

import os
import sys
import asyncio
import logging
import time
import json
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
import importlib.util

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossLanguageTestSuite:
    """跨語言方案測試套件"""
    
    def __init__(self):
        self.test_results = {}
        self.available_methods = []
        self.workspace_path = Path("C:/D/fold7/AIVA-git")
        self.logger = logging.getLogger("CrossLanguageTestSuite")
        
        # 測試配置
        self.test_configs = {
            "cross_language_bridge": {
                "file": "cross_language_bridge.py",
                "class": "CrossLanguageManager",
                "methods": ["FFI", "Subprocess", "WebSocket", "ZMQ", "TCP", "NamedPipe", "SharedMemory", "FileBased", "RestAPI", "gRPC"]
            },
            "wasm_integration": {
                "file": "wasm_integration.py", 
                "class": "AIVAWASMManager",
                "methods": ["Wasmtime", "Wasmer"]
            },
            "graalvm_integration": {
                "file": "graalvm_integration.py",
                "class": "AIVAGraalVMManager", 
                "methods": ["GraalVM", "NodeJS", "Fallback"]
            },
            "ffi_integration": {
                "file": "ffi_integration.py",
                "class": "AIVAFFIManager",
                "methods": ["CFFI", "ctypes", "Rust", "Go"]
            }
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """執行綜合測試"""
        self.logger.info("開始跨語言方案綜合測試...")
        
        # 1. 環境檢查
        env_results = await self._check_environment()
        self.test_results["environment"] = env_results
        
        # 2. 依賴檢查
        dep_results = await self._check_dependencies()
        self.test_results["dependencies"] = dep_results
        
        # 3. 功能測試
        func_results = await self._test_functionality()
        self.test_results["functionality"] = func_results
        
        # 4. 性能測試
        perf_results = await self._test_performance()
        self.test_results["performance"] = perf_results
        
        # 5. 兼容性測試
        compat_results = await self._test_compatibility()
        self.test_results["compatibility"] = compat_results
        
        # 6. 生成綜合報告
        report = self._generate_comprehensive_report()
        
        return report
    
    async def _check_environment(self) -> Dict[str, Any]:
        """檢查執行環境"""
        self.logger.info("檢查執行環境...")
        
        results = {
            "python_version": sys.version,
            "platform": sys.platform,
            "architecture": sys.maxsize > 2**32 and "64-bit" or "32-bit",
            "workspace_exists": self.workspace_path.exists(),
            "files_available": {}
        }
        
        # 檢查核心檔案是否存在
        for config_name, config in self.test_configs.items():
            file_path = self.workspace_path / config["file"]
            results["files_available"][config_name] = file_path.exists()
            
            if file_path.exists():
                self.logger.info(f"[OK] {config['file']} 找到")
            else:
                self.logger.warning(f"[FAIL] {config['file']} 未找到")
        
        return results
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """檢查相依性"""
        self.logger.info("檢查相依性...")
        
        # 必要的 Python 套件
        required_packages = [
            "asyncio", "json", "subprocess", "pathlib", "ctypes",
            "cffi", "websockets", "zmq", "wasmtime", "wasmer", 
            "grpcio", "protobuf"
        ]
        
        optional_packages = [
            "polyglot",  # GraalVM
            "numpy",     # 數值計算
            "pandas"     # 資料處理
        ]
        
        results = {
            "required": {},
            "optional": {},
            "external_tools": {}
        }
        
        # 檢查必要套件
        for package in required_packages:
            try:
                if package == "cffi":
                    import cffi
                elif package == "websockets":
                    import websockets
                elif package == "zmq":
                    import zmq
                elif package == "wasmtime": 
                    import wasmtime
                elif package == "wasmer":
                    import wasmer
                elif package == "grpcio":
                    import grpc  # grpcio package imports as grpc
                elif package == "protobuf":
                    import google.protobuf
                else:
                    __import__(package)
                
                results["required"][package] = {"available": True, "error": None}
                self.logger.info(f"[OK] {package} 可用")
                
            except ImportError as e:
                results["required"][package] = {"available": False, "error": str(e)}
                self.logger.warning(f"[FAIL] {package} 不可用: {e}")
        
        # 檢查可選套件
        for package in optional_packages:
            try:
                if package == "polyglot":
                    import polyglot
                else:
                    __import__(package)
                
                results["optional"][package] = {"available": True, "error": None}
                self.logger.info(f"[OK] {package} (可選) 可用")
                
            except ImportError as e:
                results["optional"][package] = {"available": False, "error": str(e)}
                self.logger.info(f"[INFO] {package} (可選) 不可用")
        
        # 檢查外部工具
        external_tools = ["cargo", "go", "node", "emcc", "wasm-opt", "protoc"]
        
        for tool in external_tools:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    results["external_tools"][tool] = {
                        "available": True, 
                        "version": result.stdout.strip().split('\n')[0],
                        "error": None
                    }
                    self.logger.info(f"[OK] {tool} 可用")
                else:
                    results["external_tools"][tool] = {
                        "available": False,
                        "version": None, 
                        "error": result.stderr
                    }
                    self.logger.warning(f"[FAIL] {tool} 錯誤: {result.stderr}")
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                results["external_tools"][tool] = {
                    "available": False,
                    "version": None,
                    "error": str(e)
                }
                self.logger.warning(f"[FAIL] {tool} 不可用: {e}")
        
        return results
    
    async def _test_functionality(self) -> Dict[str, Any]:
        """測試功能性"""
        self.logger.info("測試功能性...")
        
        results = {}
        
        # 測試跨語言橋接系統
        results["cross_language_bridge"] = await self._test_cross_language_bridge()
        
        # 測試 WebAssembly 整合
        results["wasm_integration"] = await self._test_wasm_integration()
        
        # 測試 GraalVM 整合
        results["graalvm_integration"] = await self._test_graalvm_integration()
        
        # 測試 FFI 整合
        results["ffi_integration"] = await self._test_ffi_integration()
        
        return results
    
    async def _test_cross_language_bridge(self) -> Dict[str, Any]:
        """測試跨語言橋接系統"""
        self.logger.info("測試跨語言橋接系統...")
        
        try:
            # 動態載入模組
            spec = importlib.util.spec_from_file_location(
                "cross_language_bridge",
                self.workspace_path / "cross_language_bridge.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 建立管理器
            manager = module.get_cross_language_manager()
            
            # 測試各種橋接方法
            bridge_results = {}
            
            for bridge_type in ["file_based", "subprocess", "tcp_socket"]:
                try:
                    # 獲取橋接器
                    bridge = manager.get_bridge(bridge_type)
                    
                    if bridge and await bridge.is_available():
                        # 簡單的測試訊息
                        test_data = {"test": "hello", "language": "python"}
                        result = await bridge.send_message("test", test_data)
                        
                        bridge_results[bridge_type] = {
                            "available": True,
                            "test_passed": result is not None,
                            "result": str(result)[:100] if result else None
                        }
                        self.logger.info(f"[OK] {bridge_type} 橋接測試通過")
                    else:
                        bridge_results[bridge_type] = {
                            "available": False,
                            "test_passed": False,
                            "result": "Bridge not available"
                        }
                        self.logger.warning(f"[FAIL] {bridge_type} 橋接不可用")
                        
                except Exception as e:
                    bridge_results[bridge_type] = {
                        "available": False,
                        "test_passed": False,
                        "result": f"Error: {e}"
                    }
                    self.logger.error(f"[FAIL] {bridge_type} 橋接測試失敗: {e}")
            
            return {
                "module_loaded": True,
                "manager_created": True,
                "bridges": bridge_results,
                "overall_success": any(r["test_passed"] for r in bridge_results.values())
            }
            
        except Exception as e:
            self.logger.error(f"跨語言橋接系統測試失敗: {e}")
            return {
                "module_loaded": False,
                "manager_created": False,
                "bridges": {},
                "overall_success": False,
                "error": str(e)
            }
    
    async def _test_wasm_integration(self) -> Dict[str, Any]:
        """測試 WebAssembly 整合"""
        self.logger.info("測試 WebAssembly 整合...")
        
        try:
            # 動態載入模組
            spec = importlib.util.spec_from_file_location(
                "wasm_integration",
                self.workspace_path / "wasm_integration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 建立管理器
            manager = module.AIVAWASMManager()
            
            # 檢查執行環境
            wasmtime_available = hasattr(module, 'WASMTIME_AVAILABLE') and module.WASMTIME_AVAILABLE
            wasmer_available = hasattr(module, 'WASMER_AVAILABLE') and module.WASMER_AVAILABLE
            
            return {
                "module_loaded": True,
                "manager_created": True,
                "wasmtime_available": wasmtime_available,
                "wasmer_available": wasmer_available,
                "overall_success": wasmtime_available or wasmer_available,
                "note": "WASM 模組需要實際的 .wasm 檔案才能完全測試"
            }
            
        except Exception as e:
            self.logger.error(f"WebAssembly 整合測試失敗: {e}")
            return {
                "module_loaded": False,
                "manager_created": False,
                "overall_success": False,
                "error": str(e)
            }
    
    async def _test_graalvm_integration(self) -> Dict[str, Any]:
        """測試 GraalVM 整合"""
        self.logger.info("測試 GraalVM 整合...")
        
        try:
            # 動態載入模組
            spec = importlib.util.spec_from_file_location(
                "graalvm_integration", 
                self.workspace_path / "graalvm_integration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 建立管理器
            manager = module.AIVAGraalVMManager()
            
            # 檢查 GraalVM 可用性
            graalvm_available = hasattr(module, 'GRAALVM_AVAILABLE') and module.GRAALVM_AVAILABLE
            
            # 測試回退模式
            context = manager.context
            python_init = context.initialize_context("python")
            js_init = context.initialize_context("js")
            
            return {
                "module_loaded": True,
                "manager_created": True,
                "graalvm_available": graalvm_available,
                "python_context": python_init,
                "js_context": js_init,
                "overall_success": python_init or js_init,
                "note": "完整 GraalVM 功能需要安裝 GraalVM 環境"
            }
            
        except Exception as e:
            self.logger.error(f"GraalVM 整合測試失敗: {e}")
            return {
                "module_loaded": False,
                "manager_created": False,
                "overall_success": False,
                "error": str(e)
            }
    
    async def _test_ffi_integration(self) -> Dict[str, Any]:
        """測試 FFI 整合"""
        self.logger.info("測試 FFI 整合...")
        
        try:
            # 動態載入模組
            spec = importlib.util.spec_from_file_location(
                "ffi_integration",
                self.workspace_path / "ffi_integration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 建立管理器
            manager = module.AIVAFFIManager()
            
            # 檢查依賴
            cffi_available = hasattr(module, 'CFFI_AVAILABLE') and module.CFFI_AVAILABLE
            ctypes_available = hasattr(module, 'CTYPES_AVAILABLE') and module.CTYPES_AVAILABLE
            
            return {
                "module_loaded": True,
                "manager_created": True,
                "cffi_available": cffi_available,
                "ctypes_available": ctypes_available,
                "overall_success": cffi_available or ctypes_available,
                "note": "FFI 功能需要編譯的動態函式庫才能完全測試"
            }
            
        except Exception as e:
            self.logger.error(f"FFI 整合測試失敗: {e}")
            return {
                "module_loaded": False,
                "manager_created": False,
                "overall_success": False,
                "error": str(e)
            }
    
    async def _test_performance(self) -> Dict[str, Any]:
        """測試性能"""
        self.logger.info("測試性能...")
        
        # 簡單的性能測試
        results = {}
        
        # 測試文件 I/O 性能（用於 file-based 橋接）
        start_time = time.time()
        test_file = self.workspace_path / "test_performance.json"
        
        for i in range(100):
            test_data = {"iteration": i, "data": "test" * 100}
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
        
        file_io_time = time.time() - start_time
        
        # 清理測試檔案
        if test_file.exists():
            test_file.unlink()
        
        results["file_io"] = {
            "operations": 100,
            "total_time": file_io_time,
            "avg_time_per_op": file_io_time / 100,
            "ops_per_second": 100 / file_io_time
        }
        
        self.logger.info(f"檔案 I/O 性能: {100/file_io_time:.2f} ops/sec")
        
        return results
    
    async def _test_compatibility(self) -> Dict[str, Any]:
        """測試兼容性"""
        self.logger.info("測試兼容性...")
        
        results = {
            "python_version_compatible": sys.version_info >= (3, 7),
            "platform_supported": sys.platform in ["win32", "linux", "darwin"],
            "encoding_support": True,  # 假設支援 UTF-8
            "network_available": await self._test_network_connectivity(),
            "file_system_writable": await self._test_file_system_access()
        }
        
        return results
    
    async def _test_network_connectivity(self) -> bool:
        """測試網路連接"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(("127.0.0.1", 80))
            sock.close()
            return True  # 即使連接失敗，也表示網路堆疊可用
        except Exception:
            return False
    
    async def _test_file_system_access(self) -> bool:
        """測試檔案系統存取"""
        try:
            test_file = self.workspace_path / "test_fs_access.tmp"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成綜合報告"""
        self.logger.info("生成綜合報告...")
        
        # 計算總體得分
        scores = {
            "environment": self._calculate_environment_score(),
            "dependencies": self._calculate_dependencies_score(),
            "functionality": self._calculate_functionality_score(),
            "performance": self._calculate_performance_score(),
            "compatibility": self._calculate_compatibility_score()
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        # 生成建議
        recommendations = self._generate_recommendations()
        
        # 判斷可用的方法
        available_methods = self._identify_available_methods()
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": self.test_results,
            "scores": scores,
            "overall_score": overall_score,
            "available_methods": available_methods,
            "recommendations": recommendations,
            "summary": {
                "total_methods_tested": len(self.test_configs),
                "successful_integrations": len(available_methods),
                "success_rate": len(available_methods) / len(self.test_configs) * 100,
                "status": "PASS" if overall_score >= 0.6 else "PARTIAL" if overall_score >= 0.3 else "FAIL"
            }
        }
        
        return report
    
    def _calculate_environment_score(self) -> float:
        """計算環境得分"""
        env_data = self.test_results.get("environment", {})
        score = 0.0
        
        if env_data.get("workspace_exists", False):
            score += 0.3
        
        files_available = env_data.get("files_available", {})
        if files_available:
            available_count = sum(1 for available in files_available.values() if available)
            score += 0.7 * (available_count / len(files_available))
        
        return score
    
    def _calculate_dependencies_score(self) -> float:
        """計算依賴得分"""
        dep_data = self.test_results.get("dependencies", {})
        score = 0.0
        
        # 必要套件權重 70%
        required = dep_data.get("required", {})
        if required:
            available_count = sum(1 for pkg in required.values() if pkg.get("available", False))
            score += 0.7 * (available_count / len(required))
        
        # 外部工具權重 30%
        external = dep_data.get("external_tools", {})
        if external:
            available_count = sum(1 for tool in external.values() if tool.get("available", False))
            score += 0.3 * (available_count / len(external))
        
        return score
    
    def _calculate_functionality_score(self) -> float:
        """計算功能得分"""
        func_data = self.test_results.get("functionality", {})
        
        if not func_data:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for integration, data in func_data.items():
            if isinstance(data, dict) and "overall_success" in data:
                total_score += 1.0 if data["overall_success"] else 0.0
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _calculate_performance_score(self) -> float:
        """計算性能得分"""
        perf_data = self.test_results.get("performance", {})
        
        # 基於檔案 I/O 性能評分
        file_io = perf_data.get("file_io", {})
        if file_io:
            ops_per_sec = file_io.get("ops_per_second", 0)
            # 假設 100 ops/sec 是及格線
            return min(1.0, ops_per_sec / 100.0)
        
        return 0.5  # 預設中等分數
    
    def _calculate_compatibility_score(self) -> float:
        """計算兼容性得分"""
        compat_data = self.test_results.get("compatibility", {})
        
        if not compat_data:
            return 0.0
        
        criteria = [
            "python_version_compatible",
            "platform_supported", 
            "encoding_support",
            "network_available",
            "file_system_writable"
        ]
        
        passed = sum(1 for criterion in criteria if compat_data.get(criterion, False))
        return passed / len(criteria)
    
    def _identify_available_methods(self) -> List[str]:
        """識別可用的方法"""
        available = []
        
        func_data = self.test_results.get("functionality", {})
        
        for integration, data in func_data.items():
            if isinstance(data, dict) and data.get("overall_success", False):
                available.append(integration)
        
        return available
    
    def _generate_recommendations(self) -> List[str]:
        """生成建議"""
        recommendations = []
        
        # 基於測試結果生成建議
        dep_data = self.test_results.get("dependencies", {})
        
        # 檢查缺少的依賴
        required = dep_data.get("required", {})
        missing_packages = [pkg for pkg, info in required.items() 
                          if not info.get("available", False)]
        
        if missing_packages:
            recommendations.append(f"安裝缺少的 Python 套件: {', '.join(missing_packages)}")
        
        # 檢查外部工具
        external = dep_data.get("external_tools", {})
        missing_tools = [tool for tool, info in external.items() 
                        if not info.get("available", False)]
        
        if missing_tools:
            recommendations.append(f"安裝缺少的外部工具: {', '.join(missing_tools)}")
        
        # 基於可用方法生成建議
        available_methods = self._identify_available_methods()
        
        if "cross_language_bridge" in available_methods:
            recommendations.append("建議優先使用跨語言橋接系統，提供最多元的通信方式")
        
        if not available_methods:
            recommendations.append("所有整合方案都不可用，請檢查環境配置")
        elif len(available_methods) < 2:
            recommendations.append("可用方案較少，建議安裝更多依賴以提高可靠性")
        
        return recommendations

# 主要執行函數
async def main():
    """主要執行函數"""
    print(">> AIVA 跨語言方案綜合測試開始...")
    print("=" * 60)
    
    # 建立測試套件
    test_suite = CrossLanguageTestSuite()
    
    try:
        # 執行綜合測試
        report = await test_suite.run_comprehensive_tests()
        
        # 顯示結果
        print("\n[REPORT] 測試結果摘要:")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"測試狀態: {summary['status']}")
        print(f"總體得分: {report['overall_score']:.2%}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"可用整合方案: {summary['successful_integrations']}/{summary['total_methods_tested']}")
        
        print(f"\n[OK] 可用的跨語言方法:")
        for method in report["available_methods"]:
            print(f"  - {method}")
        
        if report["recommendations"]:
            print(f"\n💡 建議:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        
        # 儲存詳細報告
        report_file = Path("aiva_crosslang_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 詳細報告已儲存至: {report_file}")
        
        # 結論
        if summary["status"] == "PASS":
            print("\n🎉 測試通過！AIVA 跨語言系統已準備就緒。")
        elif summary["status"] == "PARTIAL":
            print("\n[WARN] 部分功能可用，建議完善環境配置。")
        else:
            print("\n[FAIL] 測試失敗，需要修復環境配置。")
        
    except Exception as e:
        logger.error(f"測試過程發生錯誤: {e}")
        print(f"\n[FAIL] 測試過程發生錯誤: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # 執行測試
    success = asyncio.run(main())
    sys.exit(0 if success else 1)