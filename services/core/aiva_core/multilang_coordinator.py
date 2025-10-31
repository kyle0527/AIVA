"""Multi-Language AI Coordinator
多語言 AI 協調器

負責協調 Python/Rust/Go/TypeScript 等多語言 AI 模組
"""

import time
from typing import Any

from services.aiva_common.enums import ProgrammingLanguage

from .utils.logging_formatter import get_aiva_logger, log_cross_language_call

logger = get_aiva_logger("multilang_coordinator")


class MultiLanguageAICoordinator:
    """多語言 AI 協調器"""

    def __init__(self):
        self.available_ai_modules: dict[ProgrammingLanguage, bool] = {
            ProgrammingLanguage.PYTHON: True,  # 主要 AI 引擎
            ProgrammingLanguage.RUST: False,  # Rust AI 模組（需啟動）
            ProgrammingLanguage.GO: False,  # Go AI 模組（需啟動）
            ProgrammingLanguage.TYPESCRIPT: False,  # TypeScript AI 模組（需啟動）
        }
        self.module_status: dict[str, Any] = {}

        # 初始化 Rust 和 Go 子模組
        self._initialize_rust_module()
        self._initialize_go_module()
        self._initialize_typescript_module()

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

    def _initialize_rust_module(self) -> None:
        """初始化 Rust AI 模組"""
        try:
            # TODO: 實際實現 Rust AI 模組的初始化
            # 這裡可以調用 Rust 二進制文件或通過 FFI
            logger.info("正在初始化 Rust AI 模組...")

            # 模擬檢查 Rust 模組是否可用
            import subprocess

            # 檢查是否有 Rust 執行檔案
            rust_module_path = (
                "services/features/rust_ai_module/target/release/ai_processor"
            )
            try:
                result = subprocess.run(
                    [rust_module_path, "--version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    self.available_ai_modules[ProgrammingLanguage.RUST] = True
                    logger.info("✅ Rust AI 模組已就緒")
                    self.module_status[ProgrammingLanguage.RUST] = {
                        "status": "ready",
                        "version": result.stdout.decode().strip(),
                        "initialized_at": logger.info.__name__,
                    }
                else:
                    logger.warning("⚠️ Rust AI 模組初始化失敗")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.info("📝 Rust AI 模組尚未構建，保持禁用狀態")

        except Exception as e:
            logger.error(f"Rust 模組初始化異常: {e}")

    def _initialize_go_module(self) -> None:
        """初始化 Go AI 模組"""
        try:
            logger.info("正在初始化 Go AI 模組...")

            # 檢查 Go AI 服務是否運行
            import requests

            go_service_url = "http://localhost:8081/health"  # Go AI 服務的健康檢查端點

            try:
                response = requests.get(go_service_url, timeout=2)
                if response.status_code == 200:
                    self.available_ai_modules[ProgrammingLanguage.GO] = True
                    logger.info("✅ Go AI 模組已就緒")
                    self.module_status[ProgrammingLanguage.GO] = {
                        "status": "ready",
                        "service_url": go_service_url,
                        "response_time": response.elapsed.total_seconds(),
                    }
                else:
                    logger.warning("⚠️ Go AI 服務響應異常")
            except requests.RequestException:
                logger.info("📝 Go AI 服務未運行，保持禁用狀態")

        except Exception as e:
            logger.error(f"Go 模組初始化異常: {e}")

    def _initialize_typescript_module(self) -> None:
        """初始化 TypeScript AI 模組"""
        try:
            logger.info("正在初始化 TypeScript AI 模組...")

            # 檢查 Node.js AI 服務
            import requests

            ts_service_url = (
                "http://localhost:3001/api/health"  # TypeScript AI 服務端點
            )

            try:
                response = requests.get(ts_service_url, timeout=2)
                if response.status_code == 200:
                    self.available_ai_modules[ProgrammingLanguage.TYPESCRIPT] = True
                    logger.info("✅ TypeScript AI 模組已就緒")
                    self.module_status[ProgrammingLanguage.TYPESCRIPT] = {
                        "status": "ready",
                        "service_url": ts_service_url,
                        "response_time": response.elapsed.total_seconds(),
                    }
                else:
                    logger.warning("⚠️ TypeScript AI 服務響應異常")
            except requests.RequestException:
                logger.info("📝 TypeScript AI 服務未運行，保持禁用狀態")

        except Exception as e:
            logger.error(f"TypeScript 模組初始化異常: {e}")

    async def call_rust_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 Rust AI 模組"""
        if not self.available_ai_modules[ProgrammingLanguage.RUST]:
            return {"success": False, "error": "Rust AI 模組未啟用"}

        start_time = time.time()
        try:
            import json
            import subprocess

            # 構建調用參數
            input_data = {
                "task": task,
                "parameters": kwargs,
                "timestamp": str(logger.info.__name__),
            }

            # 調用 Rust 執行檔
            rust_module_path = (
                "services/features/rust_ai_module/target/release/ai_processor"
            )
            process = subprocess.run(
                [rust_module_path, "process"],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if process.returncode == 0:
                result = json.loads(process.stdout)
                log_cross_language_call(
                    logger,
                    "python",
                    "rust",
                    task,
                    kwargs,
                    result,
                    None,
                    time.time() - start_time if "start_time" in locals() else None,
                )
                return {
                    "success": True,
                    "language": "rust",
                    "task": task,
                    "result": result,
                }
            else:
                error_msg = process.stderr or "未知錯誤"
                log_cross_language_call(
                    logger,
                    "python",
                    "rust",
                    task,
                    kwargs,
                    None,
                    error_msg,
                    time.time() - start_time if "start_time" in locals() else None,
                )
                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"調用 Rust AI 模組異常: {e}")
            return {"success": False, "error": str(e)}

    async def call_go_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 Go AI 模組"""
        if not self.available_ai_modules[ProgrammingLanguage.GO]:
            return {"success": False, "error": "Go AI 模組未啟用"}

        start_time = time.time()
        try:

            import requests

            # 構建請求數據
            request_data = {
                "task": task,
                "parameters": kwargs,
                "timestamp": str(logger.info.__name__),
            }

            # 發送 HTTP 請求到 Go 服務
            go_service_url = "http://localhost:8081/api/ai/process"

            response = requests.post(go_service_url, json=request_data, timeout=30)

            if response.status_code == 200:
                result = response.json()
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
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
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
                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"調用 Go AI 模組異常: {e}")
            return {"success": False, "error": str(e)}

    async def call_typescript_ai(self, task: str, **kwargs) -> dict[str, Any]:
        """調用 TypeScript AI 模組"""
        if not self.available_ai_modules[ProgrammingLanguage.TYPESCRIPT]:
            return {"success": False, "error": "TypeScript AI 模組未啟用"}

        start_time = time.time()
        try:

            import requests

            # 構建請求數據
            request_data = {
                "task": task,
                "parameters": kwargs,
                "timestamp": str(logger.info.__name__),
            }

            # 發送請求到 TypeScript/Node.js 服務
            ts_service_url = "http://localhost:3001/api/ai/process"

            response = requests.post(ts_service_url, json=request_data, timeout=30)

            if response.status_code == 200:
                result = response.json()
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
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
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
                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"調用 TypeScript AI 模組異常: {e}")
            return {"success": False, "error": str(e)}
