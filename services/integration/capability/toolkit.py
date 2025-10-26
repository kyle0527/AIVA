"""
AIVA 能力註冊系統工具集
充分利用 aiva_common 中的現有工具和插件功能

功能包括:
- 自動化模式生成和驗證
- 跨語言程式碼產生
- 連接性測試和診斷
- 統一的日誌和追蹤
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id
from aiva_common.enums import ProgrammingLanguage, Severity, Confidence

from .models import CapabilityRecord, CapabilityEvidence, CapabilityStatus

# 設定結構化日誌
logger = get_logger(__name__)


class CapabilityToolkit:
    """
    能力管理工具集
    整合 aiva_common 中的各種工具和插件
    """
    
    def __init__(self):
        self.aiva_common_path = Path("services/aiva_common")
        self.tools_path = self.aiva_common_path / "tools"
        
    async def validate_capability_schema(
        self, 
        capability: CapabilityRecord
    ) -> Tuple[bool, List[str]]:
        """
        使用 aiva_common 的模式驗證工具驗證能力定義
        
        Returns:
            (是否有效, 錯誤列表)
        """
        try:
            # 使用 aiva_common 的 schema_validator.py
            validator_path = self.tools_path / "schema_validator.py"
            
            if not validator_path.exists():
                logger.warning("schema_validator.py 不存在，跳過驗證")
                return True, []
            
            # 將能力轉換為 JSON 用於驗證
            capability_json = capability.model_dump_json()
            
            # 執行驗證器
            result = await self._run_python_tool(
                str(validator_path),
                ["--input", capability_json, "--schema", "CapabilityRecord"]
            )
            
            if result["success"]:
                logger.info(
                    "能力模式驗證通過",
                    capability_id=capability.id
                )
                return True, []
            else:
                errors = result.get("errors", [result.get("error", "未知錯誤")])
                logger.warning(
                    "能力模式驗證失敗",
                    capability_id=capability.id,
                    errors=errors
                )
                return False, errors
                
        except Exception as e:
            logger.error(
                "模式驗證過程中出現異常",
                capability_id=capability.id,
                error=str(e),
                exc_info=True
            )
            return False, [f"驗證異常: {str(e)}"]
    
    async def generate_cross_language_bindings(
        self, 
        capability: CapabilityRecord
    ) -> Dict[str, str]:
        """
        使用 aiva_common 的程式碼產生工具產生跨語言綁定
        
        Returns:
            語言 -> 產生的程式碼 的對映
        """
        bindings = {}
        
        try:
            # 使用 aiva_common 的 schema_codegen_tool.py
            codegen_path = self.tools_path / "schema_codegen_tool.py"
            
            if not codegen_path.exists():
                logger.warning("schema_codegen_tool.py 不存在，跳過程式碼產生")
                return bindings
            
            # 為不同語言產生綁定
            target_languages = [
                ProgrammingLanguage.PYTHON,
                ProgrammingLanguage.GO,
                ProgrammingLanguage.RUST,
                ProgrammingLanguage.TYPESCRIPT
            ]
            
            capability_json = capability.model_dump_json()
            
            for lang in target_languages:
                try:
                    result = await self._run_python_tool(
                        str(codegen_path),
                        [
                            "--input", capability_json,
                            "--target", lang.value,
                            "--template", "capability_binding"
                        ]
                    )
                    
                    if result["success"]:
                        bindings[lang.value] = result.get("generated_code", "")
                        logger.info(
                            f"成功產生 {lang.value} 綁定",
                            capability_id=capability.id
                        )
                    else:
                        logger.warning(
                            f"產生 {lang.value} 綁定失敗",
                            capability_id=capability.id,
                            error=result.get("error")
                        )
                        
                except Exception as e:
                    logger.error(
                        f"產生 {lang.value} 綁定時發生異常",
                        capability_id=capability.id,
                        error=str(e)
                    )
            
        except Exception as e:
            logger.error(
                "跨語言程式碼產生過程中出現異常",
                capability_id=capability.id,
                error=str(e),
                exc_info=True
            )
        
        return bindings
    
    async def test_capability_connectivity(
        self, 
        capability: CapabilityRecord
    ) -> CapabilityEvidence:
        """
        使用 aiva_common 的連接性測試工具測試能力
        
        Returns:
            測試證據
        """
        trace_id = new_id("trace")
        start_time = datetime.utcnow()
        
        try:
            # 使用 aiva_common 的 module_connectivity_tester.py
            tester_path = self.tools_path / "module_connectivity_tester.py"
            
            if not tester_path.exists():
                logger.warning("module_connectivity_tester.py 不存在，使用基本測試")
                return await self._basic_connectivity_test(capability, trace_id)
            
            # 根據能力類型選擇測試方法
            test_config = {
                "capability_id": capability.id,
                "entrypoint": capability.entrypoint,
                "language": capability.language.value,
                "timeout": capability.timeout_seconds,
                "trace_id": trace_id
            }
            
            result = await self._run_python_tool(
                str(tester_path),
                ["--config", json.dumps(test_config)]
            )
            
            end_time = datetime.utcnow()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # 創建證據記錄
            evidence = CapabilityEvidence(
                capability_id=capability.id,
                timestamp=start_time,
                probe_type="connectivity",
                success=result.get("success", False),
                latency_ms=latency_ms,
                trace_id=trace_id,
                sample_input=test_config,
                sample_output=result.get("output"),
                error_message=result.get("error") if not result.get("success") else None,
                metadata={
                    "test_method": "module_connectivity_tester",
                    "tool_version": result.get("tool_version", "unknown")
                }
            )
            
            logger.info(
                "連接性測試完成",
                capability_id=capability.id,
                success=evidence.success,
                latency_ms=evidence.latency_ms,
                trace_id=trace_id
            )
            
            return evidence
            
        except Exception as e:
            end_time = datetime.utcnow()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.error(
                "連接性測試失敗",
                capability_id=capability.id,
                error=str(e),
                trace_id=trace_id,
                exc_info=True
            )
            
            return CapabilityEvidence(
                capability_id=capability.id,
                timestamp=start_time,
                probe_type="connectivity",
                success=False,
                latency_ms=latency_ms,
                trace_id=trace_id,
                error_message=str(e),
                metadata={"test_method": "exception_fallback"}
            )
    
    async def _basic_connectivity_test(
        self, 
        capability: CapabilityRecord, 
        trace_id: str
    ) -> CapabilityEvidence:
        """基本連接性測試（當專用工具不可用時）"""
        
        start_time = datetime.utcnow()
        
        try:
            if capability.language == ProgrammingLanguage.PYTHON:
                # Python 模組測試
                success = await self._test_python_module(capability.entrypoint)
            elif capability.language == ProgrammingLanguage.GO:
                # Go 服務測試
                success = await self._test_go_service(capability.entrypoint)
            elif capability.language == ProgrammingLanguage.RUST:
                # Rust 程式測試
                success = await self._test_rust_program(capability.entrypoint)
            else:
                # 其他語言暫時標記為未知
                success = False
            
            end_time = datetime.utcnow()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return CapabilityEvidence(
                capability_id=capability.id,
                timestamp=start_time,
                probe_type="basic_connectivity",
                success=success,
                latency_ms=latency_ms,
                trace_id=trace_id,
                metadata={"test_method": "basic_test"}
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return CapabilityEvidence(
                capability_id=capability.id,
                timestamp=start_time,
                probe_type="basic_connectivity",
                success=False,
                latency_ms=latency_ms,
                trace_id=trace_id,
                error_message=str(e),
                metadata={"test_method": "basic_test_failed"}
            )
    
    async def _test_python_module(self, entrypoint: str) -> bool:
        """測試 Python 模組是否可導入"""
        try:
            module_path, function_name = entrypoint.rsplit(':', 1) if ':' in entrypoint else (entrypoint, None)
            
            # 嘗試導入模組
            result = await self._run_python_command([
                "-c", 
                f"import {module_path}; print('Import successful')"
            ])
            
            return result["success"]
            
        except Exception as e:
            logger.debug(f"Python 模組測試失敗: {str(e)}")
            return False
    
    async def _test_go_service(self, entrypoint: str) -> bool:
        """測試 Go 服務是否可達"""
        try:
            if entrypoint.startswith('http://') or entrypoint.startswith('https://'):
                # HTTP 服務測試
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(entrypoint, timeout=10) as response:
                        return response.status < 500
            else:
                # 其他類型的 Go 服務
                logger.debug(f"暫不支援的 Go 服務類型: {entrypoint}")
                return False
                
        except Exception as e:
            logger.debug(f"Go 服務測試失敗: {str(e)}")
            return False
    
    async def _test_rust_program(self, entrypoint: str) -> bool:
        """測試 Rust 程式是否存在"""
        try:
            rust_path = Path(entrypoint)
            
            if rust_path.exists() and rust_path.is_file():
                # 檢查是否為可執行檔案
                return rust_path.stat().st_mode & 0o111 != 0
            else:
                # 嘗試執行（可能在 PATH 中）
                result = await self._run_command([entrypoint, "--version"])
                return result["success"]
                
        except Exception as e:
            logger.debug(f"Rust 程式測試失敗: {str(e)}")
            return False
    
    async def _run_python_tool(
        self, 
        tool_path: str, 
        args: List[str]
    ) -> Dict[str, Any]:
        """執行 Python 工具"""
        try:
            cmd = [sys.executable, tool_path] + args
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            result = {
                "success": process.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode
            }
            
            # 如果輸出是 JSON，嘗試解析
            if result["success"] and result["stdout"]:
                try:
                    output_data = json.loads(result["stdout"])
                    result.update(output_data)
                except json.JSONDecodeError:
                    result["output"] = result["stdout"]
            
            if not result["success"]:
                result["error"] = result["stderr"] or f"Tool failed with code {process.returncode}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_python_command(self, args: List[str]) -> Dict[str, Any]:
        """執行 Python 命令"""
        return await self._run_command([sys.executable] + args)
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """執行通用命令"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_capability_documentation(
        self, 
        capability: CapabilityRecord
    ) -> str:
        """
        產生能力的標準化文件
        
        Returns:
            Markdown 格式的文件
        """
        
        doc_template = f"""# {capability.name}

## 基本資訊

- **ID**: `{capability.id}`
- **版本**: `{capability.version}`
- **模組**: `{capability.module}`
- **語言**: `{capability.language.value}`
- **類型**: `{capability.capability_type.value}`
- **狀態**: `{capability.status.value}`

## 描述

{capability.description}

## 入口點

```
{capability.entrypoint}
```

## 輸入參數

"""
        
        if capability.inputs:
            doc_template += "| 參數名 | 類型 | 必需 | 描述 | 默認值 |\n"
            doc_template += "|--------|------|------|------|--------|\n"
            
            for param in capability.inputs:
                default_val = param.default if param.default is not None else "N/A"
                required = "是" if param.required else "否"
                doc_template += f"| {param.name} | {param.type} | {required} | {param.description} | {default_val} |\n"
        else:
            doc_template += "無輸入參數。\n"
        
        doc_template += "\n## 輸出參數\n\n"
        
        if capability.outputs:
            doc_template += "| 輸出名 | 類型 | 描述 | 示例值 |\n"
            doc_template += "|--------|------|------|--------|\n"
            
            for output in capability.outputs:
                sample = output.sample_value if output.sample_value is not None else "N/A"
                doc_template += f"| {output.name} | {output.type} | {output.description} | {sample} |\n"
        else:
            doc_template += "無輸出參數。\n"
        
        # 依賴關係
        if capability.dependencies:
            doc_template += "\n## 依賴關係\n\n"
            for dep in capability.dependencies:
                doc_template += f"- `{dep}`\n"
        
        # 前置條件
        if capability.prerequisites:
            doc_template += "\n## 前置條件\n\n"
            for prereq in capability.prerequisites:
                doc_template += f"- {prereq}\n"
        
        # 標籤
        if capability.tags:
            doc_template += "\n## 標籤\n\n"
            tag_list = ", ".join([f"`{tag}`" for tag in capability.tags])
            doc_template += f"{tag_list}\n"
        
        # 配置資訊
        doc_template += f"""
## 執行配置

- **超時時間**: {capability.timeout_seconds} 秒
- **重試次數**: {capability.retry_count} 次
- **優先級**: {capability.priority}/100

## 時間戳

- **創建時間**: {capability.created_at.isoformat()}
- **更新時間**: {capability.updated_at.isoformat()}
"""
        
        if capability.last_probe:
            doc_template += f"- **最後探測**: {capability.last_probe.isoformat()}\n"
        
        if capability.last_success:
            doc_template += f"- **最後成功**: {capability.last_success.isoformat()}\n"
        
        return doc_template
    
    async def export_capabilities_summary(
        self, 
        capabilities: List[CapabilityRecord]
    ) -> Dict[str, Any]:
        """
        匯出能力摘要統計
        
        Returns:
            統計資訊字典
        """
        
        summary = {
            "total_count": len(capabilities),
            "by_language": {},
            "by_type": {},
            "by_status": {},
            "by_module": {},
            "health_overview": {
                "healthy": 0,
                "issues": 0,
                "unknown": 0
            },
            "recent_updates": [],
            "top_dependencies": {},
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # 統計各種分類
        for cap in capabilities:
            # 語言統計
            lang = cap.language.value
            summary["by_language"][lang] = summary["by_language"].get(lang, 0) + 1
            
            # 類型統計
            cap_type = cap.capability_type.value
            summary["by_type"][cap_type] = summary["by_type"].get(cap_type, 0) + 1
            
            # 狀態統計
            status = cap.status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            # 模組統計
            module = cap.module
            summary["by_module"][module] = summary["by_module"].get(module, 0) + 1
            
            # 健康狀態概覽
            if status == "healthy":
                summary["health_overview"]["healthy"] += 1
            elif status in ["degraded", "failed"]:
                summary["health_overview"]["issues"] += 1
            else:
                summary["health_overview"]["unknown"] += 1
            
            # 依賴統計
            for dep in cap.dependencies:
                summary["top_dependencies"][dep] = summary["top_dependencies"].get(dep, 0) + 1
        
        # 最近更新的能力（按更新時間排序，取前10個）
        sorted_caps = sorted(capabilities, key=lambda x: x.updated_at, reverse=True)
        summary["recent_updates"] = [
            {
                "id": cap.id,
                "name": cap.name,
                "updated_at": cap.updated_at.isoformat(),
                "status": cap.status.value
            }
            for cap in sorted_caps[:10]
        ]
        
        # 排序依賴統計（按使用次數降序）
        summary["top_dependencies"] = dict(
            sorted(summary["top_dependencies"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return summary


# 全局工具集實例
toolkit = CapabilityToolkit()