"""
AIVA Common AI Cross Language Bridge - 跨語言橋接器

此文件提供符合 aiva_common 規範的跨語言橋接器實現，
支援 Python/Go/Rust 子進程通信和結果同步機制。

設計特點:
- 實現 ICrossLanguageBridge 介面
- 支援多語言子進程執行 (Python, Go, Rust, Node.js)
- 異步結果同步機制
- 與現有 aiva_common 工具整合
- 使用現有 Schema 和枚舉定義

架構位置:
- 屬於 Common 層的共享組件
- 支援五大模組架構的跨語言需求
- 與 schema_codegen_tool 和多語言 Schema 整合
"""



import asyncio
import json
import logging
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from ..enums import ProgrammingLanguage
from ..schemas.languages import (
    CrossLanguageAnalysis,
    LanguageInteroperability,
    MultiLanguageCodebase,
)
from .interfaces import ICrossLanguageBridge

logger = logging.getLogger(__name__)


class BridgeConfig:
    """橋接器配置類"""
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        max_concurrent_processes: int = 5,
        enable_process_pooling: bool = True,
        temp_dir: Optional[str] = None,
        environment_isolation: bool = True,
        result_compression: bool = False,
        debug_mode: bool = False
    ):
        self.default_timeout = default_timeout
        self.max_concurrent_processes = max_concurrent_processes
        self.enable_process_pooling = enable_process_pooling
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.environment_isolation = environment_isolation
        self.result_compression = result_compression
        self.debug_mode = debug_mode
        
        # 支援的語言和預設執行器路徑
        self.language_executors = {
            ProgrammingLanguage.PYTHON: "python",
            ProgrammingLanguage.GO: "go",  
            ProgrammingLanguage.RUST: "cargo",
            ProgrammingLanguage.JAVASCRIPT: "node",
            ProgrammingLanguage.TYPESCRIPT: "ts-node",
        }
        
        # 語言特定的參數模板
        self.language_args_templates = {
            ProgrammingLanguage.PYTHON: ["-c"],
            ProgrammingLanguage.GO: ["run"],
            ProgrammingLanguage.RUST: ["run", "--"],
            ProgrammingLanguage.JAVASCRIPT: [],
            ProgrammingLanguage.TYPESCRIPT: [],
        }


class ProcessPool:
    """進程池管理器"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.active_processes: Dict[str, subprocess.Popen[bytes]] = {}
        self.process_semaphore = asyncio.Semaphore(max_size)
        self._lock = asyncio.Lock()
    
    async def acquire_slot(self) -> str:
        """獲取進程槽位"""
        await self.process_semaphore.acquire()
        slot_id = f"slot_{uuid4().hex[:8]}"
        return slot_id
    
    def release_slot(self, slot_id: str) -> None:
        """釋放進程槽位"""
        if slot_id in self.active_processes:
            try:
                process = self.active_processes[slot_id]
                if process.poll() is None:  # 進程仍在運行
                    process.terminate()
                del self.active_processes[slot_id]
            except Exception as e:
                logger.warning(f"Error releasing process slot {slot_id}: {e}")
        
        self.process_semaphore.release()
    
    async def cleanup(self) -> None:
        """清理所有活躍進程"""
        async with self._lock:
            for slot_id, process in self.active_processes.items():
                try:
                    if process.poll() is None:
                        process.terminate()
                        await asyncio.sleep(0.1)  # 給予進程終止時間
                        if process.poll() is None:
                            process.kill()
                except Exception as e:
                    logger.warning(f"Error cleaning up process {slot_id}: {e}")
            
            self.active_processes.clear()


class AIVACrossLanguageBridge(ICrossLanguageBridge):
    """AIVA 跨語言橋接器實現
    
    此類提供符合 aiva_common 規範的跨語言通信功能，
    支援多語言子進程執行和結果同步。
    """

    def __init__(
        self,
        config: Optional[BridgeConfig] = None
    ):
        """初始化跨語言橋接器
        
        Args:
            config: 橋接器配置
        """
        self.config = config or BridgeConfig()
        self.process_pool = ProcessPool(self.config.max_concurrent_processes)
        
        # 語言支援檢查
        self._supported_languages = self._detect_supported_languages()
        
        # 結果快取和同步狀態
        self._result_cache: Dict[str, Dict[str, Any]] = {}
        self._sync_operations: Dict[str, asyncio.Task[Any]] = {}
        
        logger.info(
            f"AIVACrossLanguageBridge initialized. "
            f"Supported languages: {[lang.value for lang in self._supported_languages]}"
        )

    async def execute_subprocess(
        self,
        language: str,
        executable_path: str,
        args: List[str],
        timeout: float = 30.0,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """執行子進程
        
        Args:
            language: 程式語言 (python/go/rust/node)
            executable_path: 可執行文件路徑
            args: 參數列表
            timeout: 超時秒數
            env: 環境變數
            
        Returns:
            執行結果 (包含 stdout, stderr, exit_code)
        """
        # 轉換語言字符串為枚舉
        try:
            prog_lang = ProgrammingLanguage(language.lower())
        except ValueError:
            return self._create_error_result(
                f"Unsupported language: {language}",
                "UNSUPPORTED_LANGUAGE"
            )
        
        if prog_lang not in self._supported_languages:
            return self._create_error_result(
                f"Language {language} is not available on this system",
                "LANGUAGE_NOT_AVAILABLE"
            )
        
        execution_id = f"exec_{uuid4().hex[:12]}"
        start_time = datetime.now(UTC)
        
        logger.info(
            f"Starting subprocess execution {execution_id}: "
            f"{language} {executable_path} {args}"
        )
        
        # 獲取進程槽位
        slot_id = await self.process_pool.acquire_slot()
        
        try:
            # 構建完整的執行命令
            command = self._build_command(prog_lang, executable_path, args)
            
            # 準備環境變數
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            if self.config.environment_isolation:
                # 隔離環境變數，移除可能的干擾
                isolation_vars = ["PYTHONPATH", "GOPATH", "CARGO_HOME"]
                for var in isolation_vars:
                    process_env.pop(var, None)  # type: ignore
            
            # 執行子進程
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                cwd=Path(executable_path).parent if os.path.isfile(executable_path) else None
            )
            
            # 記錄活躍進程
            self.process_pool.active_processes[slot_id] = process  # type: ignore[assignment]
            
            # 等待執行完成
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                exit_code = process.returncode
                
            except asyncio.TimeoutError:
                logger.warning(f"Process {execution_id} timed out after {timeout}s")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                
                return self._create_timeout_result(execution_id, timeout)
            
            # 處理結果
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            
            result = {
                "execution_id": execution_id,
                "language": language,
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "execution_time": execution_time,
                "command": ' '.join(command),
                "timestamp": start_time.isoformat(),
            }
            
            # 嘗試解析 JSON 輸出
            if result["success"] and str(result["stdout"]).strip() if result["stdout"] else "":
                try:
                    result["parsed_output"] = json.loads(str(result["stdout"]) if result["stdout"] is not None else "{}")
                except json.JSONDecodeError:
                    result["parsed_output"] = None
            
            # 快取結果
            self._result_cache[execution_id] = result
            
            logger.info(
                f"Subprocess execution {execution_id} completed: "
                f"exit_code={exit_code}, time={execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Subprocess execution {execution_id} failed: {e}", exc_info=True)
            return self._create_error_result(str(e), "EXECUTION_ERROR", execution_id)
            
        finally:
            # 釋放進程槽位
            self.process_pool.release_slot(slot_id)

    async def sync_results(
        self,
        source_language: str,
        target_language: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """同步跨語言結果
        
        Args:
            source_language: 來源語言
            target_language: 目標語言 
            data: 要同步的數據
            
        Returns:
            同步後的結果
        """
        sync_id = f"sync_{uuid4().hex[:8]}"
        
        logger.info(
            f"Starting cross-language sync {sync_id}: "
            f"{source_language} -> {target_language}"
        )
        
        try:
            # 驗證語言支援
            source_lang = ProgrammingLanguage(source_language.lower())
            target_lang = ProgrammingLanguage(target_language.lower())
            
            if source_lang not in self._supported_languages:
                return self._create_error_result(
                    f"Source language {source_language} not supported",
                    "UNSUPPORTED_SOURCE_LANGUAGE"
                )
            
            if target_lang not in self._supported_languages:
                return self._create_error_result(
                    f"Target language {target_language} not supported", 
                    "UNSUPPORTED_TARGET_LANGUAGE"
                )
            
            # 執行語言特定的數據轉換
            converted_data = await self._convert_data_format(
                data, source_lang, target_lang
            )
            
            # 驗證轉換結果
            validation_result = await self._validate_converted_data(
                converted_data, target_lang
            )
            
            if not validation_result["valid"]:
                return self._create_error_result(
                    f"Data conversion validation failed: {validation_result['errors']}",
                    "CONVERSION_VALIDATION_FAILED"
                )
            
            # 創建同步結果
            sync_result = {
                "sync_id": sync_id,
                "source_language": source_language,
                "target_language": target_language,
                "original_data": data,
                "converted_data": converted_data,
                "conversion_metadata": {
                    "conversion_time": datetime.now(UTC).isoformat(),
                    "data_size_before": len(str(data)),  # type: ignore
                    "data_size_after": len(str(converted_data)),  # type: ignore
                    "conversion_method": f"{source_language}_to_{target_language}",
                },
                "validation_result": validation_result,
                "success": True,
            }
            
            logger.info(f"Cross-language sync {sync_id} completed successfully")
            return sync_result
            
        except Exception as e:
            logger.error(f"Cross-language sync {sync_id} failed: {e}", exc_info=True)
            return self._create_error_result(str(e), "SYNC_ERROR", sync_id)

    def get_supported_languages(self) -> List[str]:
        """獲取支援的語言列表
        
        Returns:
            支援的程式語言列表
        """
        return [lang.value for lang in self._supported_languages]  # type: ignore

    async def get_language_interoperability_analysis(
        self,
        source_language: str,
        target_language: str
    ) -> LanguageInteroperability:
        """獲取語言互操作性分析
        
        Args:
            source_language: 來源語言
            target_language: 目標語言
            
        Returns:
            語言互操作性分析結果
        """
        try:
            source_lang = ProgrammingLanguage(source_language.lower())
            target_lang = ProgrammingLanguage(target_language.lower())
            
            # 分析互操作性
            interop_method = self._determine_interop_method(source_lang, target_lang)
            security_considerations = self._analyze_security_considerations(
                source_lang, target_lang, interop_method
            )
            performance_impact = self._assess_performance_impact(
                source_lang, target_lang, interop_method
            )
            compatibility_issues = self._identify_compatibility_issues(
                source_lang, target_lang
            )
            recommendations = self._generate_interop_recommendations(
                source_lang, target_lang, interop_method
            )
            
            return LanguageInteroperability(
                source_language=source_lang,
                target_language=target_lang,
                interop_method=interop_method,
                security_considerations=security_considerations,
                performance_impact=performance_impact,
                compatibility_issues=compatibility_issues,
                recommendations=recommendations,
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze language interoperability: {e}")
            raise

    async def create_cross_language_analysis(
        self,
        project_name: str,
        codebase: MultiLanguageCodebase
    ) -> CrossLanguageAnalysis:
        """創建跨語言分析
        
        Args:
            project_name: 專案名稱
            codebase: 多語言程式碼庫
            
        Returns:
            跨語言分析結果
        """
        analysis_id = f"analysis_{uuid4().hex[:12]}"
        
        try:
            # 分析跨語言問題
            cross_language_issues = []
            integration_points = []
            security_boundaries = []
            data_flow_risks = []
            
            languages = list(codebase.languages.keys())
            
            # 分析每對語言的互操作性
            for i, lang1 in enumerate(languages):
                for lang2 in languages[i+1:]:
                    interop = await self.get_language_interoperability_analysis(
                        lang1.value, lang2.value
                    )
                    
                    integration_points.append(  # type: ignore
                        f"{lang1.value} <-> {lang2.value} via {interop.interop_method}"
                    )
                    
                    if interop.security_considerations:
                        security_boundaries.extend(interop.security_considerations)  # type: ignore
                    
                    if interop.compatibility_issues:
                        cross_language_issues.extend(interop.compatibility_issues)  # type: ignore
            
            # 分析數據流風險
            if len(languages) > 1:  # type: ignore
                data_flow_risks = [
                    "跨語言數據序列化/反序列化風險",
                    "類型安全性在語言邊界的丟失",
                    "錯誤處理機制不一致",
                    "記憶體管理差異導致的安全風險",
                ]
            
            # 生成建議
            recommendations = self._generate_cross_language_recommendations(
                languages, codebase
            )
            
            # 計算風險評分
            risk_score = self._calculate_cross_language_risk_score(
                len(cross_language_issues),  # type: ignore
                len(security_boundaries),  # type: ignore
                len(data_flow_risks),  # type: ignore
                len(languages)  # type: ignore
            )
            
            return CrossLanguageAnalysis(
                analysis_id=analysis_id,
                project_name=project_name,
                languages_analyzed=languages,
                cross_language_issues=cross_language_issues,  # type: ignore
                integration_points=integration_points,  # type: ignore
                security_boundaries=security_boundaries,  # type: ignore
                data_flow_risks=data_flow_risks,
                recommendations=recommendations,
                risk_score=risk_score,
            )
            
        except Exception as e:
            logger.error(f"Failed to create cross-language analysis: {e}")
            raise

    def _detect_supported_languages(self) -> List[ProgrammingLanguage]:
        """檢測系統支援的語言"""
        supported = []
        
        for lang, executor in self.config.language_executors.items():
            try:
                # 檢查執行器是否可用
                result = subprocess.run(
                    [executor, "--version"],
                    capture_output=True,
                    timeout=5.0,
                    text=True
                )
                if result.returncode == 0:
                    supported.append(lang)  # type: ignore
                    if self.config.debug_mode:
                        logger.debug(f"Detected {lang.value}: {result.stdout.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                if self.config.debug_mode:
                    logger.debug(f"Language {lang.value} not available")
        
        return supported  # type: ignore

    def _build_command(
        self,
        language: ProgrammingLanguage,
        executable_path: str,
        args: List[str]
    ) -> List[str]:
        """構建執行命令"""
        executor = self.config.language_executors[language]
        base_args = self.config.language_args_templates.get(language, [])
        
        if language == ProgrammingLanguage.PYTHON:
            if os.path.isfile(executable_path):
                return [executor, executable_path] + args
            else:
                # 直接執行 Python 代碼
                return [executor] + base_args + [executable_path] + args
        
        elif language == ProgrammingLanguage.GO:
            return [executor] + base_args + [executable_path] + args
        
        elif language == ProgrammingLanguage.RUST:
            # Cargo 需要在專案目錄中執行
            return [executor] + base_args + args
        
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            return [executor, executable_path] + args
        
        else:
            return [executor, executable_path] + args

    async def _convert_data_format(
        self,
        data: Dict[str, Any],
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """轉換數據格式"""
        # 這裡實現語言特定的數據轉換邏輯
        # 目前返回基本的 JSON 序列化兼容格式
        
        converted = {}
        
        for key, value in data.items():
            # 處理不同語言的命名慣例
            converted_key = self._convert_naming_convention(key, source_lang, target_lang)
            
            # 處理不同語言的數據類型
            converted_value = self._convert_data_type(value, source_lang, target_lang)
            
            converted[converted_key] = converted_value
        
        return converted  # type: ignore

    def _convert_naming_convention(
        self,
        name: str,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage
    ) -> str:
        """轉換命名慣例"""
        # Python/Go: snake_case
        # JavaScript/TypeScript: camelCase
        # Rust: snake_case
        
        if source_lang in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.GO, ProgrammingLanguage.RUST]:
            if target_lang in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                # snake_case -> camelCase
                components = name.split('_')
                return components[0] + ''.join(word.capitalize() for word in components[1:])
        
        elif source_lang in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            if target_lang in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.GO, ProgrammingLanguage.RUST]:
                # camelCase -> snake_case
                import re
                return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name

    def _convert_data_type(
        self,
        value: Any,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage
    ) -> Any:
        """轉換數據類型"""
        # 處理基本類型轉換
        if isinstance(value, dict):
            # 明確類型轉換，確保字典鍵為字符串類型
            converted_dict: Dict[str, Any] = {}
            # 使用類型斷言來明確指定字典的類型
            dict_value = cast(Dict[Any, Any], value)
            for k, v in dict_value.items():
                if isinstance(k, str):
                    converted_key = self._convert_naming_convention(k, source_lang, target_lang)
                    converted_dict[converted_key] = self._convert_data_type(v, source_lang, target_lang)
                else:
                    # 非字符串鍵直接轉換為字符串
                    key_str = str(k) if k is not None else ""
                    converted_key = self._convert_naming_convention(key_str, source_lang, target_lang)
                    converted_dict[converted_key] = self._convert_data_type(v, source_lang, target_lang)
            return converted_dict
        elif isinstance(value, list):
            return [self._convert_data_type(item, source_lang, target_lang) for item in value]  # type: ignore
        else:
            return value

    async def _validate_converted_data(
        self,
        data: Dict[str, Any],
        target_lang: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """驗證轉換後的數據"""
        errors = []
        warnings = []
        
        try:
            # 檢查 JSON 序列化兼容性
            json.dumps(data)
        except (TypeError, ValueError) as e:
            errors.append(f"JSON serialization failed: {e}")  # type: ignore
        
        # 語言特定驗證
        if target_lang == ProgrammingLanguage.GO:
            # Go 的特定驗證
            for key in data.keys():
                if not key[0].isupper() and not key.startswith('_'):
                    warnings.append(f"Go field '{key}' should start with uppercase for export")  # type: ignore
        
        return {
            "valid": len(errors) == 0,  # type: ignore
            "errors": errors,
            "warnings": warnings,
        }

    def _determine_interop_method(
        self,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage
    ) -> str:
        """確定互操作方法"""
        interop_matrix = {
            (ProgrammingLanguage.PYTHON, ProgrammingLanguage.GO): "subprocess_json",
            (ProgrammingLanguage.GO, ProgrammingLanguage.PYTHON): "subprocess_json", 
            (ProgrammingLanguage.PYTHON, ProgrammingLanguage.RUST): "subprocess_json",
            (ProgrammingLanguage.RUST, ProgrammingLanguage.PYTHON): "subprocess_json",
            (ProgrammingLanguage.GO, ProgrammingLanguage.RUST): "subprocess_json",
            (ProgrammingLanguage.RUST, ProgrammingLanguage.GO): "subprocess_json",
            (ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.PYTHON): "subprocess_json",
            (ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT): "subprocess_json",
        }
        
        return interop_matrix.get((source_lang, target_lang), "subprocess_generic")

    def _analyze_security_considerations(
        self,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage,
        interop_method: str
    ) -> List[str]:
        """分析安全考量"""
        considerations = []
        
        if interop_method == "subprocess_json":
            considerations.extend([  # type: ignore
                "JSON 反序列化攻擊風險",
                "命令注入風險 (如果處理用戶輸入)",
                "資源耗盡攻擊 (大量子進程)",
            ])
        
        # 語言特定的安全考量
        if source_lang == ProgrammingLanguage.PYTHON:
            considerations.append("Python pickle 反序列化風險 (如果使用)")  # type: ignore
        
        if target_lang == ProgrammingLanguage.GO:
            considerations.append("Go race condition 風險 (並發存取)")  # type: ignore
        
        return considerations  # type: ignore

    def _assess_performance_impact(
        self,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage,
        interop_method: str
    ) -> str:
        """評估性能影響"""
        if interop_method == "subprocess_json":
            return "中等 - 子進程創建和 JSON 序列化開銷"
        else:
            return "低 - 基本的進程間通信開銷"

    def _identify_compatibility_issues(
        self,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage
    ) -> List[str]:
        """識別兼容性問題"""
        issues = []
        
        # 數值精度問題
        if source_lang == ProgrammingLanguage.JAVASCRIPT:
            issues.append("JavaScript 數值精度限制 (IEEE 754)")  # type: ignore
        
        # 字符編碼問題
        issues.append("字符編碼不一致風險 (UTF-8 vs 其他編碼)")  # type: ignore
        
        # 日期時間格式
        issues.append("日期時間格式差異")  # type: ignore
        
        return issues  # type: ignore

    def _generate_interop_recommendations(
        self,
        source_lang: ProgrammingLanguage,
        target_lang: ProgrammingLanguage,
        interop_method: str
    ) -> List[str]:
        """生成互操作建議"""
        recommendations = [
            "使用結構化數據格式 (JSON/MessagePack)",
            "實施輸入驗證和清理",
            "建立錯誤處理機制",
            "考慮使用 Protocol Buffers 或 Apache Avro",
        ]
        
        if interop_method == "subprocess_json":
            recommendations.extend([  # type: ignore
                "限制子進程數量和執行時間",
                "使用進程池提高效率",
                "實施結果快取機制",
            ])
        
        return recommendations  # type: ignore

    def _generate_cross_language_recommendations(
        self,
        languages: List[ProgrammingLanguage],
        codebase: MultiLanguageCodebase
    ) -> List[str]:
        """生成跨語言建議"""
        recommendations = []
        
        if len(languages) > 2:  # type: ignore
            recommendations.append("考慮建立統一的 API 層")  # type: ignore
            recommendations.append("使用 gRPC 或 GraphQL 進行服務間通信")  # type: ignore
        
        if ProgrammingLanguage.JAVASCRIPT in languages:
            recommendations.append("注意 JavaScript 的類型強制轉換問題")  # type: ignore
        
        if codebase.total_lines > 10000:
            recommendations.append("實施自動化的跨語言測試")  # type: ignore
            recommendations.append("建立持續整合流程")  # type: ignore
        
        recommendations.extend([  # type: ignore
            "建立統一的錯誤處理策略",
            "實施跨語言的日誌記錄標準",
            "考慮使用容器化部署",
        ])
        
        return recommendations  # type: ignore

    def _calculate_cross_language_risk_score(
        self,
        issues_count: int,
        security_boundaries_count: int,
        data_flow_risks_count: int,
        languages_count: int
    ) -> float:
        """計算跨語言風險評分"""
        base_score = 2.0  # 基礎風險
        
        # 問題數量影響
        issues_impact = min(issues_count * 0.5, 3.0)
        
        # 安全邊界影響  
        security_impact = min(security_boundaries_count * 0.3, 2.0)
        
        # 數據流風險影響
        data_flow_impact = min(data_flow_risks_count * 0.4, 2.0)
        
        # 語言複雜度影響
        complexity_impact = min((languages_count - 1) * 0.5, 1.5)
        
        total_score = base_score + issues_impact + security_impact + data_flow_impact + complexity_impact
        
        return min(total_score, 10.0)

    def _create_error_result(
        self,
        error_message: str,
        error_code: str,
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """創建錯誤結果"""
        return {
            "execution_id": execution_id or f"error_{uuid4().hex[:8]}",
            "success": False,
            "error_code": error_code,
            "error_message": error_message,
            "stdout": "",
            "stderr": error_message,
            "exit_code": -1,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _create_timeout_result(
        self,
        execution_id: str,
        timeout: float
    ) -> Dict[str, Any]:
        """創建超時結果"""
        return {
            "execution_id": execution_id,
            "success": False,
            "error_code": "TIMEOUT",
            "error_message": f"Process execution timeout after {timeout}s",
            "stdout": "",
            "stderr": f"Process terminated due to timeout ({timeout}s)",
            "exit_code": -1,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def cleanup(self) -> None:
        """清理資源"""
        try:
            # 清理進程池
            await self.process_pool.cleanup()
            
            # 清理快取
            self._result_cache.clear()
            
            # 取消同步操作
            for task in self._sync_operations.values():
                if not task.done():
                    task.cancel()
            self._sync_operations.clear()
            
            logger.info("AIVACrossLanguageBridge cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """析構函數"""
        # 確保資源清理
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
        except Exception:
            pass


# ============================================================================
# Factory Function (工廠函數)
# ============================================================================


def create_cross_language_bridge(
    config: Optional[BridgeConfig] = None,
    **kwargs  # type: ignore
) -> AIVACrossLanguageBridge:
    """創建跨語言橋接器實例
    
    Args:
        config: 橋接器配置
        **kwargs: 其他參數  # type: ignore
        
    Returns:
        跨語言橋接器實例
    """
    return AIVACrossLanguageBridge(config=config)