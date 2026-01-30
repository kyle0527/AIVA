"""Multi-Language AI Coordinator
多語言 AI 協調器 - V2 gRPC 統一架構

負責協調 Python/Rust/Go/TypeScript 等多語言 AI 模組
使用統一的 gRPC 框架進行跨語言通訊，替代舊版 HTTP/subprocess 方式

整合功能：
- Rust 模組：高效能掃描和計算
- Go 模組：並發處理和微服務
- TypeScript 模組：動態分析和前端整合
- Python 核心：AI 決策和策略規劃
"""

import asyncio
import time
from typing import Any, Optional

from aiva_common.cross_language.core import CrossLanguageService, CrossLanguageConfig
from aiva_common.cross_language.adapters import (
    RustAdapter, RustConfig, create_rust_adapter,
    GoAdapter, GoConfig, create_go_adapter
)
from aiva_common.enums.modules import ProgrammingLanguage
from aiva_common.utils import get_logger

# 使用标准日志工具
from aiva_common.utils.logging import get_logger

logger = get_logger("multilang_coordinator")


def log_cross_language_call(
    logger_instance,
    src: str,
    dst: str,
    task: str,
    kwargs: dict,
    result: Any,
    error: Optional[Exception | str],
    duration: float
) -> None:
    """记录跨语言调用日志
    
    Args:
        logger_instance: 日志实例
        src: 源语言/模块
        dst: 目标语言/模块
        task: 任务描述
        kwargs: 调用参数
        result: 调用结果
        error: 错误信息（异常对象或字符串）
        duration: 执行时长（秒）
    """
    status = "ERROR" if error else "SUCCESS"
    logger_instance.info(
        f"Cross-language call {src}->{dst}: task={task}, "
        f"status={status}, duration={duration:.3f}s"
    )
    if error:
        logger_instance.error(f"Error details: {error}")


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
        
        # 語言適配器實例
        self.rust_adapter: RustAdapter | None = None
        self.go_adapter: GoAdapter | None = None
        self.typescript_adapter: Any | None = None  # TypeScript adapter placeholder
        
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
        
        # 創建 Rust 適配器
        try:
            self.rust_adapter = await create_rust_adapter(RustConfig())
            if await self.rust_adapter.initialize():
                self.available_ai_modules[ProgrammingLanguage.RUST] = True
                logger.info("✅ Rust 模組初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ Rust 模組初始化失敗: {e}")
        
        # 創建 Go 適配器
        try:
            self.go_adapter = await create_go_adapter(GoConfig())
            if await self.go_adapter.initialize():
                self.available_ai_modules[ProgrammingLanguage.GO] = True
                logger.info("✅ Go 模組初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ Go 模組初始化失敗: {e}")
        
        # 檢查 TypeScript 服務
        await self._check_typescript_service()
        
        self._initialized = True
        
        # 輸出初始化狀態
        available_langs = [lang.value for lang, available in self.available_ai_modules.items() if available]
        logger.info(f"✅ 多語言 AI 協調器初始化完成 - 可用語言: {', '.join(available_langs)}")

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
        """執行 Python AI 任務
        
        調用真實的 Python AI 組件執行任務。
        
        Raises:
            RuntimeError: 任務執行失敗時拋出
        """
        logger.info(f"執行 Python AI 任務: {task}")

        # 使用 5M Decision Engine（特化決策引擎）
        try:
            from ..cognitive_core.neural.real_neural_core import RealDecisionEngine
            
            # 獲取或創建 decision engine 實例
            if not hasattr(self, '_python_agent') or self._python_agent is None:
                self._python_agent = RealDecisionEngine(use_5m_model=True)
            
            # 執行任務（使用 generate_decision 方法）
            # task_description: 任務描述，context: 上下文資訊
            context_str = str(kwargs) if kwargs else ""
            result = self._python_agent.generate_decision(
                task_description=task,
                context=context_str
            )
            
            return {
                "success": True,
                "task": task,
                "language": "python",
                "result": result,
                "details": kwargs,
                "processed_by": "5M_DecisionEngine",
            }
        except ImportError as e:
            raise RuntimeError(
                f"Python AI 組件不可用: {e}。"
                "請確認 cognitive_core.neural.real_neural_core 模組已正確安裝。"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Python AI 任務執行失敗: {e}") from e

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
        """調用 Rust AI 模組 - 使用 RustAdapter
        
        Raises:
            RuntimeError: Rust 模組未啟用或調用失敗時拋出
        """
        if not self.available_ai_modules[ProgrammingLanguage.RUST]:
            raise RuntimeError(
                "Rust AI 模組未啟用。請確認 Rust 服務已啟動並可連接。"
            )

        if not self.rust_adapter:
            raise RuntimeError(
                "Rust 適配器未初始化。請先調用 initialize() 方法。"
            )

        start_time = time.time()
        try:
            # 確保已初始化
            if not self._initialized:
                await self.initialize()
            
            # 通過適配器執行任務
            if not hasattr(self.rust_adapter, 'execute_rust_function'):
                raise RuntimeError(
                    "Rust 適配器缺少 execute_rust_function 方法。"
                    "請確認 RustAdapter 實現完整。"
                )
            
            result = await self.rust_adapter.execute_rust_function(task, **kwargs)

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
                "protocol": "adapter"
            }

        except Exception as e:
            error_msg = f"Rust適配器調用失敗: {e}"
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
            raise RuntimeError(error_msg) from e

    async def call_go_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 Go AI 模組 - V2 gRPC 版本"""
        if not self.available_ai_modules[ProgrammingLanguage.GO]:
            raise RuntimeError(
                "Go AI 模組未啟用。"
                "請在配置中啟用 Go 語言支持並確保相關依賴已安裝。"
            )

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
                from aiva_common.protocols.aiva_services_pb2 import DataAnalysisRequest  # type: ignore[attr-defined]
                from aiva_common.protocols.aiva_services_pb2_grpc import DataAnalyzerStub  # type: ignore[attr-defined]
                
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
                    "analysis_id": response.analysis_id,  # type: ignore[attr-defined]
                    "status": response.status,  # type: ignore[attr-defined]
                    "insights_count": len(response.insights),  # type: ignore[attr-defined]
                    "summary": response.summary  # type: ignore[attr-defined]
                }
                
            elif "code" in task.lower():
                # 使用代碼生成服務
                from aiva_common.protocols.aiva_services_pb2 import CodeGenerationRequest  # type: ignore[attr-defined]
                from aiva_common.protocols.aiva_services_pb2_grpc import CodeGeneratorStub  # type: ignore[attr-defined]
                
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
                    "generation_id": response.generation_id,  # type: ignore[attr-defined]
                    "status": response.status,  # type: ignore[attr-defined]
                    "files_count": len(response.files),  # type: ignore[attr-defined]
                    "warnings": list(response.warnings)  # type: ignore[attr-defined]
                }
                
            else:
                # 默認 AI 推理
                from aiva_common.protocols.aiva_services_pb2 import ReasoningRequest  # type: ignore[attr-defined]
                from aiva_common.protocols.aiva_services_pb2_grpc import AIServiceStub  # type: ignore[attr-defined]
                
                request = ReasoningRequest(
                    query=kwargs.get("query", task),
                    session_id=kwargs.get("session_id", "default"),
                    context_items=kwargs.get("context", [])
                )
                
                response = await self.cross_lang_service.call_service(
                    AIServiceStub, "ExecuteReasoning", request, endpoint
                )
                
                result = {
                    "response": response.response,  # type: ignore[attr-defined]
                    "confidence": response.confidence,  # type: ignore[attr-defined]
                    "reasoning_steps": list(response.reasoning_steps)  # type: ignore[attr-defined]
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
            raise RuntimeError(
                f"Go gRPC 服務調用失敗: {e}。"
                "請確認 Go gRPC 服務已啟動並正確配置端點。"
            ) from e

    async def call_typescript_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 TypeScript AI 模組 - V2 gRPC 版本"""
        if not self.available_ai_modules[ProgrammingLanguage.TYPESCRIPT]:
            raise RuntimeError(
                "TypeScript AI 模組未啟用。"
                "請在配置中啟用 TypeScript 語言支持並確保相關依賴已安裝。"
            )

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
                from aiva_common.protocols.aiva_services_pb2 import ScanRequest, ScanConfig  # type: ignore[attr-defined]
                from aiva_common.protocols.aiva_services_pb2_grpc import WebServiceStub  # type: ignore[attr-defined]
                
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
                async for web_result in self.cross_lang_service.call_service(  # type: ignore[misc]
                    WebServiceStub, "ScanWebsite", request, endpoint
                ):
                    # 處理流式結果（這裡簡化為取第一個結果）
                    result = {
                        "scan_id": web_result.scan_id,  # type: ignore[attr-defined]
                        "request_info": {
                            "method": web_result.request.method,  # type: ignore[attr-defined]
                            "url": web_result.request.url  # type: ignore[attr-defined]
                        },
                        "response_info": {
                            "status_code": web_result.response.status_code,  # type: ignore[attr-defined]
                            "response_time": web_result.response.response_time_ms  # type: ignore[attr-defined]
                        },
                        "findings_count": len(web_result.findings)  # type: ignore[attr-defined]
                    }
                    break  # 取第一個結果作為示例
                    
            else:
                # 默認使用 AI 服務進行命令分析
                from aiva_common.protocols.aiva_services_pb2 import CommandAnalysisRequest  # type: ignore[attr-defined]
                from aiva_common.protocols.aiva_services_pb2_grpc import AIServiceStub  # type: ignore[attr-defined]
                
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
                    "intent": response.intent,  # type: ignore[attr-defined]
                    "recommended_service": response.recommended_service,  # type: ignore[attr-defined]
                    "parameters": dict(response.parameters),  # type: ignore[attr-defined]
                    "confidence": response.confidence,  # type: ignore[attr-defined]
                    "suggestions": list(response.suggestions)  # type: ignore[attr-defined]
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
            raise RuntimeError(
                f"TypeScript gRPC 服務調用失敗: {e}。"
                "請確認 TypeScript gRPC 服務已啟動並正確配置端點。"
            ) from e

