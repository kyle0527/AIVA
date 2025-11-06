"""Multi-Language AI Coordinator
多語言 AI 協調器 - V2 gRPC 統一架構

負責協調 Python/Rust/Go/TypeScript 等多語言 AI 模組
使用統一的 gRPC 框架進行跨語言通訊，替代舊版 HTTP/subprocess 方式
"""

import asyncio
import time
from typing import Any

from services.aiva_common.cross_language.core import CrossLanguageService, CrossLanguageConfig
from services.aiva_common.enums import ProgrammingLanguage

from .utils.logging_formatter import get_aiva_logger, log_cross_language_call

logger = get_aiva_logger("multilang_coordinator")


class MultiLanguageAICoordinator:
    """多語言 AI 協調器 - V2 gRPC 統一版本"""

    def __init__(self, config: CrossLanguageConfig | None = None):
        self.available_ai_modules: dict[ProgrammingLanguage, bool] = {
            ProgrammingLanguage.PYTHON: True,  # 主要 AI 引擎
            ProgrammingLanguage.RUST: False,  # Rust AI 模組（gRPC 服務）
            ProgrammingLanguage.GO: False,  # Go AI 模組（gRPC 服務）
            ProgrammingLanguage.TYPESCRIPT: False,  # TypeScript AI 模組（gRPC 服務）
        }
        self.module_status: dict[str, Any] = {}
        
        # V2 統一 gRPC 服務
        self.cross_lang_service = CrossLanguageService(config)
        
        # gRPC 服務端點配置
        self.service_endpoints = {
            ProgrammingLanguage.RUST: "localhost:50052",
            ProgrammingLanguage.GO: "localhost:50053", 
            ProgrammingLanguage.TYPESCRIPT: "localhost:50054"
        }

        # 非同步初始化（需要在異步上下文中調用）
        self._initialized = False

    async def initialize(self):
        """異步初始化 gRPC 連接和模組檢查"""
        if self._initialized:
            return
            
        logger.info("正在初始化多語言 AI 協調器...")
        
        # 檢查各語言 gRPC 服務的可用性
        await self._check_rust_service()
        await self._check_go_service() 
        await self._check_typescript_service()
        
        self._initialized = True
        logger.info("✅ 多語言 AI 協調器初始化完成")

    def check_module_availability(self, language: ProgrammingLanguage) -> bool:
        """檢查特定語言的 AI 模組是否可用"""
        return self.available_ai_modules.get(language, False)

    async def execute_task(
        self, task: str, language: ProgrammingLanguage | None = None, **kwargs
    ) -> dict[str, Any]:
        """執行 AI 任務

        Args:
            task: 任務類型
            language: 指定使用的語言（None 則自動選擇）
            **kwargs: 任務參數

        Returns:
            任務執行結果
        """
        if language is None:
            # 自動選擇可用的語言模組
            language = self._select_best_language(task)

        logger.info(f"執行 AI 任務: {task}, 使用語言: {language}")

        # 根據語言調用對應的模組
        try:
            if language == ProgrammingLanguage.RUST:
                return await self.call_rust_ai(task, **kwargs)
            elif language == ProgrammingLanguage.GO:
                return await self.call_go_ai(task, **kwargs)
            elif language == ProgrammingLanguage.TYPESCRIPT:
                return await self.call_typescript_ai(task, **kwargs)
            elif language == ProgrammingLanguage.PYTHON:
                # Python 本地處理
                return await self._execute_python_task(task, **kwargs)
            else:
                return {
                    "success": False,
                    "error": f"不支持的語言: {language}",
                    "language": language,
                }
        except Exception as e:
            logger.error(f"任務執行異常: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "language": language,
            }

    async def _execute_python_task(self, task: str, **kwargs) -> dict[str, Any]:
        """執行 Python AI 任務"""
        logger.info(f"執行 Python AI 任務: {task}")

        # 這裡可以調用本地的 Python AI 功能
        # 例如 BioNeuronRAGAgent 或其他 AI 組件

        # 模擬處理
        import asyncio

        await asyncio.sleep(0.1)  # 模擬處理時間

        return {
            "success": True,
            "task": task,
            "language": "python",
            "result": f"Python AI 任務 '{task}' 執行完成",
            "details": kwargs,
            "processed_by": "MultiLanguageAICoordinator",
        }

    def _select_best_language(self, task: str) -> ProgrammingLanguage:
        """根據任務選擇最佳語言"""
        # 優先使用 Python（主要 AI 引擎）
        if self.available_ai_modules[ProgrammingLanguage.PYTHON]:
            return ProgrammingLanguage.PYTHON

        # 性能密集型任務優先使用 Rust
        performance_intensive = ["vulnerability_scan", "fuzzing", "exploit"]
        if any(keyword in task.lower() for keyword in performance_intensive):
            if self.available_ai_modules[ProgrammingLanguage.RUST]:
                return ProgrammingLanguage.RUST

        # 併發任務優先使用 Go
        concurrent_tasks = ["parallel", "distributed", "concurrent"]
        if any(keyword in task.lower() for keyword in concurrent_tasks):
            if self.available_ai_modules[ProgrammingLanguage.GO]:
                return ProgrammingLanguage.GO

        # 默認使用 Python
        return ProgrammingLanguage.PYTHON

    def get_status(self) -> dict[str, Any]:
        """獲取協調器狀態"""
        return {
            "available_modules": {
                lang.value: available
                for lang, available in self.available_ai_modules.items()
            },
            "module_status": self.module_status,
        }

    def enable_module(self, language: ProgrammingLanguage) -> bool:
        """啟用特定語言模組"""
        try:
            self.available_ai_modules[language] = True
            logger.info(f"已啟用 {language} AI 模組")
            return True
        except Exception as e:
            logger.error(f"啟用 {language} 模組失敗: {e}")
            return False

    def disable_module(self, language: ProgrammingLanguage) -> bool:
        """禁用特定語言模組"""
        try:
            self.available_ai_modules[language] = False
            logger.info(f"已禁用 {language} AI 模組")
            return True
        except Exception as e:
            logger.error(f"禁用 {language} 模組失敗: {e}")
            return False

    async def _check_rust_service(self) -> None:
        """檢查 Rust gRPC 服務可用性"""
        try:
            logger.info("正在檢查 Rust gRPC 服務...")
            endpoint = self.service_endpoints[ProgrammingLanguage.RUST]
            
            # 使用 V2 gRPC 健康檢查
            is_available = await self.cross_lang_service.health_check(endpoint)
            
            if is_available:
                self.available_ai_modules[ProgrammingLanguage.RUST] = True
                logger.info("✅ Rust gRPC 服務已就緒")
                self.module_status[ProgrammingLanguage.RUST] = {
                    "status": "ready",
                    "endpoint": endpoint,
                    "protocol": "gRPC",
                    "checked_at": time.time(),
                }
            else:
                logger.info("📝 Rust gRPC 服務未運行，保持禁用狀態")
                self.module_status[ProgrammingLanguage.RUST] = {
                    "status": "unavailable", 
                    "endpoint": endpoint,
                    "protocol": "gRPC"
                }

        except Exception as e:
            logger.error(f"Rust 服務檢查異常: {e}")
            self.available_ai_modules[ProgrammingLanguage.RUST] = False

    async def _check_go_service(self) -> None:
        """檢查 Go gRPC 服務可用性"""
        try:
            logger.info("正在檢查 Go gRPC 服務...")
            endpoint = self.service_endpoints[ProgrammingLanguage.GO]
            
            # 使用 V2 gRPC 健康檢查
            is_available = await self.cross_lang_service.health_check(endpoint)
            
            if is_available:
                self.available_ai_modules[ProgrammingLanguage.GO] = True
                logger.info("✅ Go gRPC 服務已就緒")
                self.module_status[ProgrammingLanguage.GO] = {
                    "status": "ready",
                    "endpoint": endpoint,
                    "protocol": "gRPC",
                    "checked_at": time.time(),
                }
            else:
                logger.info("📝 Go gRPC 服務未運行，保持禁用狀態")
                self.module_status[ProgrammingLanguage.GO] = {
                    "status": "unavailable",
                    "endpoint": endpoint,
                    "protocol": "gRPC"
                }

        except Exception as e:
            logger.error(f"Go 服務檢查異常: {e}")
            self.available_ai_modules[ProgrammingLanguage.GO] = False

    async def _check_typescript_service(self) -> None:
        """檢查 TypeScript gRPC 服務可用性"""
        try:
            logger.info("正在檢查 TypeScript gRPC 服務...")
            endpoint = self.service_endpoints[ProgrammingLanguage.TYPESCRIPT]
            
            # 使用 V2 gRPC 健康檢查
            is_available = await self.cross_lang_service.health_check(endpoint)
            
            if is_available:
                self.available_ai_modules[ProgrammingLanguage.TYPESCRIPT] = True
                logger.info("✅ TypeScript gRPC 服務已就緒")
                self.module_status[ProgrammingLanguage.TYPESCRIPT] = {
                    "status": "ready",
                    "endpoint": endpoint,
                    "protocol": "gRPC",
                    "checked_at": time.time(),
                }
            else:
                logger.info("📝 TypeScript gRPC 服務未運行，保持禁用狀態")
                self.module_status[ProgrammingLanguage.TYPESCRIPT] = {
                    "status": "unavailable",
                    "endpoint": endpoint,
                    "protocol": "gRPC"
                }

        except Exception as e:
            logger.error(f"TypeScript 服務檢查異常: {e}")
            self.available_ai_modules[ProgrammingLanguage.TYPESCRIPT] = False

    async def call_rust_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 Rust AI 模組 - V2 gRPC 版本"""
        if not self.available_ai_modules[ProgrammingLanguage.RUST]:
            return {"success": False, "error": "Rust AI 模組未啟用"}

        start_time = time.time()
        try:
            # 確保已初始化
            if not self._initialized:
                await self.initialize()
            
            # 根據任務類型選擇適當的 gRPC 服務
            endpoint = self.service_endpoints[ProgrammingLanguage.RUST]
            
            # 構建 gRPC 請求
            if "reasoning" in task.lower():
                # 使用 AI 推理服務
                from services.aiva_common.protocols.aiva_services_pb2 import ReasoningRequest
                from services.aiva_common.protocols.aiva_services_pb2_grpc import AIServiceStub
                
                request = ReasoningRequest(
                    query=kwargs.get("query", task),
                    session_id=kwargs.get("session_id", "default"),
                    context_items=kwargs.get("context", []),
                    constraints=kwargs.get("constraints", {})
                )
                
                response = await self.cross_lang_service.call_service(
                    AIServiceStub, "ExecuteReasoning", request, endpoint
                )
                
                result = {
                    "response": response.response,
                    "confidence": response.confidence,
                    "reasoning_steps": list(response.reasoning_steps)
                }
            else:
                # 使用安全掃描服務（默認）
                from services.aiva_common.protocols.aiva_services_pb2 import ScanRequest, ScanConfig
                from services.aiva_common.protocols.aiva_services_pb2_grpc import SecurityScannerStub
                
                config = ScanConfig(
                    max_depth=kwargs.get("max_depth", 10),
                    timeout_seconds=kwargs.get("timeout", 30),
                    aggressive_mode=kwargs.get("aggressive", False)
                )
                
                request = ScanRequest(
                    scan_id=kwargs.get("scan_id", f"rust_{int(time.time())}"),
                    target=kwargs.get("target", ""),
                    scan_type=task,
                    config=config
                )
                
                response = await self.cross_lang_service.call_service(
                    SecurityScannerStub, "StartScan", request, endpoint
                )
                
                result = {
                    "scan_id": response.scan_id,
                    "status": response.status,
                    "findings_count": len(response.findings),
                    "metrics": response.metrics
                }

            log_cross_language_call(
                logger,
                "python",
                "rust",
                task,
                kwargs,
                result,
                None,
                time.time() - start_time,
            )
            
            return {
                "success": True,
                "language": "rust",
                "task": task,
                "result": result,
                "protocol": "gRPC"
            }

        except Exception as e:
            error_msg = f"gRPC 調用異常: {e}"
            log_cross_language_call(
                logger,
                "python",
                "rust", 
                task,
                kwargs,
                None,
                error_msg,
                time.time() - start_time,
            )
            logger.error(f"調用 Rust gRPC 服務異常: {e}")
            return {"success": False, "error": error_msg}

    async def call_go_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 Go AI 模組 - V2 gRPC 版本"""
        if not self.available_ai_modules[ProgrammingLanguage.GO]:
            return {"success": False, "error": "Go AI 模組未啟用"}

        start_time = time.time()
        try:
            # 確保已初始化
            if not self._initialized:
                await self.initialize()
            
            # 獲取 Go gRPC 服務端點
            endpoint = self.service_endpoints[ProgrammingLanguage.GO]
            
            # 根據任務類型選擇服務
            if "data_analysis" in task.lower() or "analyze" in task.lower():
                # 使用數據分析服務
                from services.aiva_common.protocols.aiva_services_pb2 import DataAnalysisRequest
                from services.aiva_common.protocols.aiva_services_pb2_grpc import DataAnalyzerStub
                
                request = DataAnalysisRequest(
                    analysis_id=kwargs.get("analysis_id", f"go_{int(time.time())}"),
                    data_source=kwargs.get("data_source", ""),
                    analysis_type=task,
                    parameters=kwargs.get("parameters", {})
                )
                
                response = await self.cross_lang_service.call_service(
                    DataAnalyzerStub, "AnalyzeData", request, endpoint
                )
                
                result = {
                    "analysis_id": response.analysis_id,
                    "status": response.status,
                    "insights_count": len(response.insights),
                    "summary": response.summary
                }
                
            elif "code" in task.lower():
                # 使用代碼生成服務
                from services.aiva_common.protocols.aiva_services_pb2 import CodeGenerationRequest
                from services.aiva_common.protocols.aiva_services_pb2_grpc import CodeGeneratorStub
                
                request = CodeGenerationRequest(
                    generation_id=kwargs.get("generation_id", f"go_{int(time.time())}"),
                    template_type=kwargs.get("template_type", "standard"),
                    target_language=kwargs.get("target_language", "go"),
                    parameters=kwargs.get("parameters", {}),
                    specification=kwargs.get("specification", task)
                )
                
                response = await self.cross_lang_service.call_service(
                    CodeGeneratorStub, "GenerateCode", request, endpoint
                )
                
                result = {
                    "generation_id": response.generation_id,
                    "status": response.status,
                    "files_count": len(response.files),
                    "warnings": list(response.warnings)
                }
                
            else:
                # 默認使用 AI 服務
                from services.aiva_common.protocols.aiva_services_pb2 import ReasoningRequest
                from services.aiva_common.protocols.aiva_services_pb2_grpc import AIServiceStub
                
                request = ReasoningRequest(
                    query=kwargs.get("query", task),
                    session_id=kwargs.get("session_id", "go_session"),
                    context_items=kwargs.get("context", [])
                )
                
                response = await self.cross_lang_service.call_service(
                    AIServiceStub, "ExecuteReasoning", request, endpoint
                )
                
                result = {
                    "response": response.response,
                    "confidence": response.confidence,
                    "reasoning_steps": list(response.reasoning_steps)
                }

            log_cross_language_call(
                logger,
                "python",
                "go", 
                task,
                kwargs,
                result,
                None,
                time.time() - start_time,
            )
            
            return {
                "success": True,
                "language": "go",
                "task": task,
                "result": result,
                "protocol": "gRPC"
            }

        except Exception as e:
            error_msg = f"gRPC 調用異常: {e}"
            log_cross_language_call(
                logger,
                "python",
                "go",
                task,
                kwargs,
                None,
                error_msg,
                time.time() - start_time,
            )
            logger.error(f"調用 Go gRPC 服務異常: {e}")
            return {"success": False, "error": error_msg}

    async def call_typescript_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 TypeScript AI 模組 - V2 gRPC 版本"""
        if not self.available_ai_modules[ProgrammingLanguage.TYPESCRIPT]:
            return {"success": False, "error": "TypeScript AI 模組未啟用"}

        start_time = time.time()
        try:
            # 確保已初始化
            if not self._initialized:
                await self.initialize()
            
            # 獲取 TypeScript gRPC 服務端點
            endpoint = self.service_endpoints[ProgrammingLanguage.TYPESCRIPT]
            
            # 根據任務類型選擇服務（TypeScript 主要用於 Web 相關任務）
            if "web" in task.lower() or "http" in task.lower():
                # 使用 Web 服務
                from services.aiva_common.protocols.aiva_services_pb2 import ScanRequest, ScanConfig
                from services.aiva_common.protocols.aiva_services_pb2_grpc import WebServiceStub
                
                config = ScanConfig(
                    max_depth=kwargs.get("max_depth", 5),
                    timeout_seconds=kwargs.get("timeout", 30),
                    aggressive_mode=kwargs.get("aggressive", False)
                )
                
                request = ScanRequest(
                    scan_id=kwargs.get("scan_id", f"ts_{int(time.time())}"),
                    target=kwargs.get("target", kwargs.get("url", "")),
                    scan_type=task,
                    config=config
                )
                
                # 使用流式掃描（適用於 Web 掃描）
                async for web_result in self.cross_lang_service.call_service(
                    WebServiceStub, "ScanWebsite", request, endpoint
                ):
                    # 處理流式結果（這裡簡化為取第一個結果）
                    result = {
                        "scan_id": web_result.scan_id,
                        "request_info": {
                            "method": web_result.request.method,
                            "url": web_result.request.url
                        },
                        "response_info": {
                            "status_code": web_result.response.status_code,
                            "response_time": web_result.response.response_time_ms
                        },
                        "findings_count": len(web_result.findings)
                    }
                    break  # 取第一個結果作為示例
                    
            else:
                # 默認使用 AI 服務進行命令分析
                from services.aiva_common.protocols.aiva_services_pb2 import CommandAnalysisRequest
                from services.aiva_common.protocols.aiva_services_pb2_grpc import AIServiceStub
                
                request = CommandAnalysisRequest(
                    command=task,
                    session_id=kwargs.get("session_id", "ts_session"),
                    context=kwargs.get("context", {}),
                    user_id=kwargs.get("user_id", "default")
                )
                
                response = await self.cross_lang_service.call_service(
                    AIServiceStub, "AnalyzeCommand", request, endpoint
                )
                
                result = {
                    "intent": response.intent,
                    "recommended_service": response.recommended_service,
                    "parameters": dict(response.parameters),
                    "confidence": response.confidence,
                    "suggestions": list(response.suggestions)
                }

            log_cross_language_call(
                logger,
                "python",
                "typescript",
                task,
                kwargs,
                result,
                None,
                time.time() - start_time,
            )
            
            return {
                "success": True,
                "language": "typescript",
                "task": task,
                "result": result,
                "protocol": "gRPC"
            }

        except Exception as e:
            error_msg = f"gRPC 調用異常: {e}"
            log_cross_language_call(
                logger,
                "python",
                "typescript",
                task,
                kwargs,
                None,
                error_msg,
                time.time() - start_time,
            )
            logger.error(f"調用 TypeScript gRPC 服務異常: {e}")
            return {"success": False, "error": error_msg}
