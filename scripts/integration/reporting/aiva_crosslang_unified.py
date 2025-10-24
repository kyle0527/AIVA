#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA 跨語言整合統一接口
將所有跨語言方案整合到 AIVA 主系統中，提供統一的調用接口
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass
import importlib.util

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrossLanguageTask:
    """跨語言任務定義"""
    task_id: str
    target_language: str
    function_name: str
    parameters: Dict[str, Any]
    requirements: Optional[Dict[str, Any]] = None
    timeout: int = 30
    retry_count: int = 3
    priority: str = "normal"  # low, normal, high, critical

@dataclass
class CrossLanguageResult:
    """跨語言執行結果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0
    method_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AIVACrossLanguageUnified:
    """AIVA 跨語言統一接口"""
    
    def __init__(self, workspace_path: str = "C:/D/fold7/AIVA-git"):
        self.workspace_path = Path(workspace_path)
        self.integrations = {}
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.task_history = []
        self.logger = logging.getLogger("AIVACrossLanguageUnified")
        
        # 統計資訊
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "method_usage": {},
            "language_usage": {},
            "avg_execution_time": 0
        }
    
    async def initialize(self) -> bool:
        """初始化所有跨語言整合"""
        self.logger.info("初始化 AIVA 跨語言統一接口...")
        
        success_count = 0
        total_count = 0
        
        # 1. 載入跨語言橋接系統
        if await self._load_cross_language_bridge():
            success_count += 1
            self.logger.info("✅ 跨語言橋接系統載入成功")
        else:
            self.logger.warning("❌ 跨語言橋接系統載入失敗")
        total_count += 1
        
        # 2. 載入 WebAssembly 整合
        if await self._load_wasm_integration():
            success_count += 1
            self.logger.info("✅ WebAssembly 整合載入成功")
        else:
            self.logger.warning("❌ WebAssembly 整合載入失敗")
        total_count += 1
        
        # 3. 載入 GraalVM 整合
        if await self._load_graalvm_integration():
            success_count += 1
            self.logger.info("✅ GraalVM 整合載入成功")
        else:
            self.logger.warning("❌ GraalVM 整合載入失敗")
        total_count += 1
        
        # 4. 載入 FFI 整合
        if await self._load_ffi_integration():
            success_count += 1
            self.logger.info("✅ FFI 整合載入成功")
        else:
            self.logger.warning("❌ FFI 整合載入失敗")
        total_count += 1
        
        # 5. 載入智能選擇器
        if await self._load_smart_selector():
            success_count += 1
            self.logger.info("✅ 智能選擇器載入成功")
        else:
            self.logger.warning("❌ 智能選擇器載入失敗")
        total_count += 1
        
        success_rate = success_count / total_count
        self.logger.info(f"初始化完成，成功率: {success_rate:.1%} ({success_count}/{total_count})")
        
        # 啟動任務處理器
        if success_count > 0:
            asyncio.create_task(self._task_processor())
            self.logger.info("任務處理器已啟動")
        
        return success_count > 0
    
    async def _load_cross_language_bridge(self) -> bool:
        """載入跨語言橋接系統"""
        try:
            spec = importlib.util.spec_from_file_location(
                "cross_language_bridge",
                self.workspace_path / "cross_language_bridge.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            manager = module.get_cross_language_manager()
            self.integrations["bridge"] = {
                "module": module,
                "manager": manager,
                "type": "bridge"
            }
            return True
        except Exception as e:
            self.logger.error(f"載入跨語言橋接系統失敗: {e}")
            return False
    
    async def _load_wasm_integration(self) -> bool:
        """載入 WebAssembly 整合"""
        try:
            spec = importlib.util.spec_from_file_location(
                "wasm_integration",
                self.workspace_path / "wasm_integration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            manager = module.AIVAWASMManager()
            self.integrations["wasm"] = {
                "module": module,
                "manager": manager,
                "type": "wasm"
            }
            return True
        except Exception as e:
            self.logger.error(f"載入 WebAssembly 整合失敗: {e}")
            return False
    
    async def _load_graalvm_integration(self) -> bool:
        """載入 GraalVM 整合"""
        try:
            spec = importlib.util.spec_from_file_location(
                "graalvm_integration",
                self.workspace_path / "graalvm_integration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            manager = module.AIVAGraalVMManager()
            await manager.initialize_all_languages()
            
            self.integrations["graalvm"] = {
                "module": module,
                "manager": manager,
                "type": "polyglot"
            }
            return True
        except Exception as e:
            self.logger.error(f"載入 GraalVM 整合失敗: {e}")
            return False
    
    async def _load_ffi_integration(self) -> bool:
        """載入 FFI 整合"""
        try:
            spec = importlib.util.spec_from_file_location(
                "ffi_integration",
                self.workspace_path / "ffi_integration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            manager = module.AIVAFFIManager()
            self.integrations["ffi"] = {
                "module": module,
                "manager": manager,
                "type": "ffi"
            }
            return True
        except Exception as e:
            self.logger.error(f"載入 FFI 整合失敗: {e}")
            return False
    
    async def _load_smart_selector(self) -> bool:
        """載入智能選擇器"""
        try:
            spec = importlib.util.spec_from_file_location(
                "smart_communication_selector",
                self.workspace_path / "smart_communication_selector.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            selector = module.AIVASmartCommunicationManager()
            self.integrations["selector"] = {
                "module": module,
                "manager": selector,
                "type": "selector"
            }
            return True
        except Exception as e:
            self.logger.error(f"載入智能選擇器失敗: {e}")
            return False
    
    async def execute_task(self, task: CrossLanguageTask) -> CrossLanguageResult:
        """執行跨語言任務"""
        start_time = time.time()
        self.stats["total_tasks"] += 1
        
        try:
            # 根據語言和需求選擇最佳方法
            method_info = await self._select_execution_method(task)
            
            if not method_info:
                raise RuntimeError("沒有可用的執行方法")
            
            # 執行任務
            result = await self._execute_with_method(task, method_info)
            
            # 記錄成功
            execution_time = time.time() - start_time
            self.stats["successful_tasks"] += 1
            self._update_stats(task, method_info["name"], execution_time)
            
            return CrossLanguageResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                method_used=method_info["name"],
                metadata=method_info.get("metadata")
            )
            
        except Exception as e:
            # 記錄失敗
            execution_time = time.time() - start_time
            self.stats["failed_tasks"] += 1
            
            self.logger.error(f"任務 {task.task_id} 執行失敗: {e}")
            
            return CrossLanguageResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _select_execution_method(self, task: CrossLanguageTask) -> Optional[Dict[str, Any]]:
        """選擇執行方法"""
        language = task.target_language
        function_name = task.function_name
        
        # 檢查智能選擇器是否可用
        if "selector" in self.integrations:
            try:
                # 建立需求規格
                from smart_communication_selector import CommunicationRequirement, PerformanceLevel, SecurityLevel, ReliabilityLevel
                
                # 根據任務參數建立需求
                performance = PerformanceLevel.HIGH if task.priority == "critical" else PerformanceLevel.MEDIUM
                security = SecurityLevel.HIGH if task.requirements and task.requirements.get("secure", False) else SecurityLevel.STANDARD
                reliability = ReliabilityLevel.HIGH
                
                requirement = CommunicationRequirement(
                    performance=performance,
                    security=security,
                    reliability=reliability,
                    languages=["python", language] if language != "python" else ["python"]
                )
                
                # 獲取最佳配置
                selector = self.integrations["selector"]["manager"]
                config = await selector.get_optimal_communication_setup(requirement)
                
                primary_method = config["primary"]["name"]
                
                # 根據選擇的方法返回對應的執行資訊
                return self._get_method_execution_info(primary_method, task)
                
            except Exception as e:
                self.logger.warning(f"智能選擇失敗，使用預設方法: {e}")
        
        # 回退到基於語言的預設選擇
        return self._get_default_method_for_language(language, task)
    
    def _get_method_execution_info(self, method_name: str, task: CrossLanguageTask) -> Optional[Dict[str, Any]]:
        """獲取方法執行資訊"""
        language = task.target_language
        
        # FFI 方法
        if "ffi" in method_name.lower():
            if language in ["rust", "go", "c", "cpp"] and "ffi" in self.integrations:
                return {
                    "name": f"FFI_{language}",
                    "integration": "ffi",
                    "executor": self.integrations["ffi"]["manager"],
                    "metadata": {"method": method_name}
                }
        
        # GraalVM 方法
        elif "graalvm" in method_name.lower() or language in ["javascript", "java", "ruby"]:
            if "graalvm" in self.integrations:
                return {
                    "name": f"GraalVM_{language}",
                    "integration": "graalvm",
                    "executor": self.integrations["graalvm"]["manager"],
                    "metadata": {"method": method_name}
                }
        
        # WebAssembly 方法
        elif "wasm" in method_name.lower():
            if "wasm" in self.integrations:
                return {
                    "name": f"WASM_{language}",
                    "integration": "wasm",
                    "executor": self.integrations["wasm"]["manager"],
                    "metadata": {"method": method_name}
                }
        
        # 橋接方法
        else:
            if "bridge" in self.integrations:
                return {
                    "name": f"Bridge_{method_name}",
                    "integration": "bridge",
                    "executor": self.integrations["bridge"]["manager"],
                    "metadata": {"method": method_name}
                }
        
        return None
    
    def _get_default_method_for_language(self, language: str, task: CrossLanguageTask) -> Optional[Dict[str, Any]]:
        """獲取語言的預設方法"""
        # Rust -> FFI (如果可用) 或 橋接
        if language == "rust":
            if "ffi" in self.integrations:
                return {
                    "name": "FFI_rust",
                    "integration": "ffi",
                    "executor": self.integrations["ffi"]["manager"],
                    "metadata": {"default": True}
                }
        
        # JavaScript -> GraalVM (如果可用) 或 橋接
        elif language == "javascript":
            if "graalvm" in self.integrations:
                return {
                    "name": "GraalVM_javascript",
                    "integration": "graalvm",
                    "executor": self.integrations["graalvm"]["manager"],
                    "metadata": {"default": True}
                }
        
        # Go -> FFI (如果可用) 或 橋接
        elif language == "go":
            if "ffi" in self.integrations:
                return {
                    "name": "FFI_go",
                    "integration": "ffi",
                    "executor": self.integrations["ffi"]["manager"],
                    "metadata": {"default": True}
                }
        
        # 回退到橋接方法
        if "bridge" in self.integrations:
            return {
                "name": "Bridge_default",
                "integration": "bridge",
                "executor": self.integrations["bridge"]["manager"],
                "metadata": {"default": True, "fallback": True}
            }
        
        return None
    
    async def _execute_with_method(self, task: CrossLanguageTask, method_info: Dict[str, Any]) -> Any:
        """使用指定方法執行任務"""
        integration_type = method_info["integration"]
        executor = method_info["executor"]
        
        if integration_type == "ffi":
            return await self._execute_with_ffi(task, executor)
        elif integration_type == "graalvm":
            return await self._execute_with_graalvm(task, executor)
        elif integration_type == "wasm":
            return await self._execute_with_wasm(task, executor)
        elif integration_type == "bridge":
            return await self._execute_with_bridge(task, executor)
        else:
            raise ValueError(f"未知的整合類型: {integration_type}")
    
    async def _execute_with_ffi(self, task: CrossLanguageTask, executor) -> Any:
        """使用 FFI 執行任務"""
        # 簡化的 FFI 調用邏輯
        library_name = f"aiva_{task.target_language}_ffi"
        function_name = f"{task.function_name}_ffi"
        
        try:
            # 檢查函式庫是否載入
            if library_name not in executor.libraries:
                # 嘗試建構並載入函式庫
                self.logger.info(f"嘗試建構 {library_name} 函式庫...")
                # 這裡需要實際的建構邏輯，目前返回模擬結果
                return {"status": "simulated", "message": f"FFI call to {function_name} simulated"}
            
            # 調用函數
            result = executor.call_function(library_name, function_name, **task.parameters)
            return result
            
        except Exception as e:
            raise RuntimeError(f"FFI 執行失敗: {e}")
    
    async def _execute_with_graalvm(self, task: CrossLanguageTask, executor) -> Any:
        """使用 GraalVM 執行任務"""
        try:
            # 建構程式碼
            if task.target_language == "javascript":
                code = self._build_javascript_code(task)
            elif task.target_language == "python":
                code = self._build_python_code(task)
            else:
                raise ValueError(f"GraalVM 不支援語言: {task.target_language}")
            
            # 執行程式碼
            result = executor.context.execute_code(task.target_language, code, task.parameters)
            return result
            
        except Exception as e:
            raise RuntimeError(f"GraalVM 執行失敗: {e}")
    
    async def _execute_with_wasm(self, task: CrossLanguageTask, executor) -> Any:
        """使用 WebAssembly 執行任務"""
        try:
            # 檢查模組是否載入
            module_name = f"{task.target_language}_{task.function_name}"
            
            if module_name not in executor.modules:
                # 嘗試載入模組
                self.logger.info(f"嘗試載入 WASM 模組: {module_name}")
                # 這裡需要實際的載入邏輯，目前返回模擬結果
                return {"status": "simulated", "message": f"WASM call to {task.function_name} simulated"}
            
            # 調用函數
            result = await executor.call_module_function(module_name, task.function_name, **task.parameters)
            return result
            
        except Exception as e:
            raise RuntimeError(f"WASM 執行失敗: {e}")
    
    async def _execute_with_bridge(self, task: CrossLanguageTask, executor) -> Any:
        """使用橋接方法執行任務"""
        try:
            # 選擇可用的橋接器
            bridge_type = "file_based"  # 預設使用檔案系統橋接
            bridge = executor.get_bridge(bridge_type)
            
            if not bridge or not await bridge.is_available():
                # 嘗試其他橋接器
                for alternative in ["tcp_socket", "subprocess"]:
                    bridge = executor.get_bridge(alternative)
                    if bridge and await bridge.is_available():
                        bridge_type = alternative
                        break
                else:
                    raise RuntimeError("沒有可用的橋接器")
            
            # 建構訊息
            message = {
                "function": task.function_name,
                "parameters": task.parameters,
                "language": task.target_language
            }
            
            # 發送訊息
            result = await bridge.send_message(task.task_id, message)
            return result
            
        except Exception as e:
            raise RuntimeError(f"橋接執行失敗: {e}")
    
    def _build_javascript_code(self, task: CrossLanguageTask) -> str:
        """建構 JavaScript 程式碼"""
        function_name = task.function_name
        params = json.dumps(task.parameters)
        
        # 基本的 JavaScript 程式碼模板
        code = f"""
        function {function_name}(params) {{
            // 這裡應該是實際的函數邏輯
            console.log('Executing {function_name} with params:', params);
            return {{ status: 'success', function: '{function_name}', params: params }};
        }}
        
        const params = {params};
        const result = {function_name}(params);
        result;
        """
        
        return code
    
    def _build_python_code(self, task: CrossLanguageTask) -> str:
        """建構 Python 程式碼"""
        function_name = task.function_name
        params = task.parameters
        
        # 基本的 Python 程式碼模板
        code = f"""
def {function_name}(params):
    # 這裡應該是實際的函數邏輯
    print(f'Executing {function_name} with params: {{params}}')
    return {{'status': 'success', 'function': '{function_name}', 'params': params}}

params = {params}
result = {function_name}(params)
"""
        
        return code
    
    def _update_stats(self, task: CrossLanguageTask, method_name: str, execution_time: float):
        """更新統計資訊"""
        # 更新方法使用統計
        if method_name not in self.stats["method_usage"]:
            self.stats["method_usage"][method_name] = 0
        self.stats["method_usage"][method_name] += 1
        
        # 更新語言使用統計
        language = task.target_language
        if language not in self.stats["language_usage"]:
            self.stats["language_usage"][language] = 0
        self.stats["language_usage"][language] += 1
        
        # 更新平均執行時間
        total_tasks = self.stats["successful_tasks"]
        if total_tasks > 1:
            self.stats["avg_execution_time"] = (
                (self.stats["avg_execution_time"] * (total_tasks - 1) + execution_time) / total_tasks
            )
        else:
            self.stats["avg_execution_time"] = execution_time
    
    async def _task_processor(self):
        """任務處理器"""
        self.logger.info("任務處理器開始運行...")
        
        while True:
            try:
                # 從佇列獲取任務
                task = await self.task_queue.get()
                
                # 執行任務
                result = await self.execute_task(task)
                
                # 儲存結果
                self.task_history.append(result)
                
                # 標記任務完成
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"任務處理器錯誤: {e}")
                await asyncio.sleep(1)
    
    async def submit_task(self, task: CrossLanguageTask) -> str:
        """提交任務到佇列"""
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        self.logger.info(f"任務 {task.task_id} 已提交")
        return task.task_id
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計資訊"""
        return self.stats.copy()
    
    def get_task_result(self, task_id: str) -> Optional[CrossLanguageResult]:
        """獲取任務結果"""
        for result in self.task_history:
            if result.task_id == task_id:
                return result
        return None
    
    def list_available_integrations(self) -> List[str]:
        """列出可用的整合"""
        return list(self.integrations.keys())
    
    def get_integration_info(self, integration_name: str) -> Optional[Dict[str, Any]]:
        """獲取整合資訊"""
        if integration_name in self.integrations:
            integration = self.integrations[integration_name]
            return {
                "name": integration_name,
                "type": integration["type"],
                "module": integration["module"].__name__,
                "manager": type(integration["manager"]).__name__
            }
        return None

# 使用範例和測試
async def demo_unified_interface():
    """示範統一接口功能"""
    print("🚀 AIVA 跨語言統一接口示範")
    print("=" * 50)
    
    # 建立統一接口
    unified = AIVACrossLanguageUnified()
    
    # 初始化
    success = await unified.initialize()
    if not success:
        print("❌ 初始化失敗")
        return
    
    print("✅ 初始化成功")
    
    # 顯示可用整合
    integrations = unified.list_available_integrations()
    print(f"\n📋 可用整合: {', '.join(integrations)}")
    
    # 建立測試任務
    tasks = [
        CrossLanguageTask(
            task_id="task_001",
            target_language="rust",
            function_name="scan_vulnerabilities",
            parameters={"code": "fn main() { println!(\"Hello\"); }", "language": "rust"},
            priority="high"
        ),
        CrossLanguageTask(
            task_id="task_002",
            target_language="javascript",
            function_name="analyze_data",
            parameters={"data": [1, 2, 3, 4, 5]},
            priority="normal"
        ),
        CrossLanguageTask(
            task_id="task_003",
            target_language="go",
            function_name="gather_info",
            parameters={"target": "localhost"},
            priority="low"
        )
    ]
    
    # 執行任務
    print("\n🔄 執行跨語言任務...")
    for task in tasks:
        result = await unified.execute_task(task)
        
        status = "✅" if result.success else "❌"
        print(f"{status} 任務 {task.task_id} ({task.target_language}): {result.method_used or 'N/A'} - {result.execution_time:.3f}s")
        
        if result.error:
            print(f"   錯誤: {result.error}")
        elif result.result:
            result_str = str(result.result)[:100]
            print(f"   結果: {result_str}{'...' if len(str(result.result)) > 100 else ''}")
    
    # 顯示統計資訊
    stats = unified.get_stats()
    print(f"\n📊 執行統計:")
    print(f"  總任務: {stats['total_tasks']}")
    print(f"  成功: {stats['successful_tasks']}")
    print(f"  失敗: {stats['failed_tasks']}")
    print(f"  平均執行時間: {stats['avg_execution_time']:.3f}s")
    
    if stats['method_usage']:
        print(f"  方法使用: {stats['method_usage']}")
    
    if stats['language_usage']:
        print(f"  語言使用: {stats['language_usage']}")

if __name__ == "__main__":
    # 執行示範
    asyncio.run(demo_unified_interface())