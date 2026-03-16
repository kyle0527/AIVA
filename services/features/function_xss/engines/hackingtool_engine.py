"""
HackingTool XSS 跨語言檢測引擎
支援 Go, Ruby, Python, Rust 等多語言 XSS 檢測工具
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, Union
import re

from ..hackingtool_config import (
    HackingToolXSSConfig, 
    XSSToolConfig, 
    get_xss_tools_config,
    XSS_SCAN_MODES
)


@dataclass
class XSSDetectionResult:
    """XSS 檢測結果資料結構"""
    tool_name: str
    language: str
    target_url: str
    vulnerability_found: bool
    confidence: float
    payloads: List[str]
    execution_time: float
    raw_output: str
    error_message: Optional[str] = None
    severity: Optional[str] = None
    context: Optional[str] = None
    method: Optional[str] = None
    parameter: Optional[str] = None


@dataclass
class LanguageEnvironment:
    """語言環境檢測結果"""
    language: str
    available: bool
    version: Optional[str] = None
    binary_path: Optional[str] = None
    error_message: Optional[str] = None


class DetectionEngineProtocol(Protocol):
    """檢測引擎協議介面"""
    async def detect(self, target_url: str, **kwargs) -> List[XSSDetectionResult]:
        """執行 XSS 檢測"""
        ...


class CrossLanguageXSSEngine:
    """跨語言 XSS 檢測引擎"""
    
    def __init__(self, config: Optional[HackingToolXSSConfig] = None):
        self.config = config or get_xss_tools_config()
        self.logger = logging.getLogger(__name__)
        self.language_environments: Dict[str, LanguageEnvironment] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="aiva_xss_"))
        
    async def initialize(self) -> bool:
        """初始化檢測引擎,檢測語言環境並驗證工具可用性
        
        ✅ 修復: 添加工具可用性檢查,防止偽陰性
        
        Raises:
            RuntimeError: 當必要工具未安裝或不可用時
        """
        try:
            # 檢測語言環境
            await self._detect_language_environments()
            
            # ✅ 新增: 驗證至少有一個工具可用
            available_tools = await self._validate_tool_availability()
            
            if not available_tools:
                raise RuntimeError(
                    "❌ XSS 檢測引擎初始化失敗: 沒有可用的工具\n"
                    "\n"
                    "已檢查的工具:\n"
                    "  - Dalfox (Go):   需要安裝 'go install github.com/hahwul/dalfox/v2@latest'\n"
                    "  - XSpear (Ruby): 需要安裝 'gem install XSpear'\n"
                    "  - xsser (Python): 需要安裝 'pip install xsser'\n"
                    "\n"
                    f"當前語言環境:\n"
                    f"  - Go:     {'✅' if self.language_environments.get('go', LanguageEnvironment('go', False)).available else '❌'}\n"
                    f"  - Ruby:   {'✅' if self.language_environments.get('ruby', LanguageEnvironment('ruby', False)).available else '❌'}\n"
                    f"  - Python: {'✅' if self.language_environments.get('python', LanguageEnvironment('python', False)).available else '❌'}\n"
                    f"  - Rust:   {'✅' if self.language_environments.get('rust', LanguageEnvironment('rust', False)).available else '❌'}\n"
                    "\n"
                    "請至少安裝一個 XSS 檢測工具"
                )
            
            self.logger.info(
                f"✅ XSS Detection Engine 初始化成功 "
                f"({len(available_tools)} 個工具可用: {', '.join(available_tools)})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"❌ XSS 引擎初始化失敗: {e}")
            raise  # ✅ 向上傳播異常,不返回 False
    
    async def _validate_tool_availability(self) -> List[str]:
        """驗證工具可用性——降低複雜度
        
        Returns:
            可用工具名稱列表
        """
        available_tools = []
        
        # 拆分為單獨的檢查方法
        dalfox_available = await self._check_dalfox_availability()
        if dalfox_available:
            available_tools.append("Dalfox")
            
        xspear_available = await self._check_xspear_availability()
        if xspear_available:
            available_tools.append("XSpear")
            
        xsser_available = await self._check_xsser_availability()
        if xsser_available:
            available_tools.append("xsser")
        
        return available_tools
    
    async def _check_dalfox_availability(self) -> bool:
        """檢查 Dalfox 工具可用性"""
        go_env = self.language_environments.get("go", LanguageEnvironment("go", False))
        if not go_env.available or not shutil.which("dalfox"):
            return False
            
        try:
            result = await self._run_command(["dalfox", "version"], timeout_seconds=5)
            if result.returncode == 0:
                self.logger.debug("✅ Dalfox 可用")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️  Dalfox 檢查失敗: {e}")
        return False
    
    async def _check_xspear_availability(self) -> bool:
        """檢查 XSpear 工具可用性"""
        ruby_env = self.language_environments.get("ruby", LanguageEnvironment("ruby", False))
        if not ruby_env.available or not shutil.which("XSpear"):
            return False
            
        try:
            result = await self._run_command(["XSpear", "--version"], timeout_seconds=5)
            if result.returncode == 0:
                self.logger.debug("✅ XSpear 可用")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️  XSpear 檢查失敗: {e}")
        return False
    
    async def _check_xsser_availability(self) -> bool:
        """檢查 xsser 工具可用性"""
        python_env = self.language_environments.get("python", LanguageEnvironment("python", False))
        if not python_env.available:
            return False
            
        try:
            result = await self._run_command(["python3", "-c", "import xsser"], timeout_seconds=5)
            if result.returncode == 0:
                self.logger.debug("✅ xsser 可用")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️  xsser 檢查失敗: {e}")
        return False
        return available_tools
    
    async def _detect_language_environments(self) -> None:
        """檢測各語言環境可用性"""
        language_checkers = {
            "go": self._check_go_environment,
            "ruby": self._check_ruby_environment,
            "python": self._check_python_environment,
            "rust": self._check_rust_environment
        }
        
        for language, checker in language_checkers.items():
            try:
                env = await checker()
                self.language_environments[language] = env
                self.logger.info(f"{language.upper()} environment: {'Available' if env.available else 'Not available'}")
            except Exception as e:
                self.logger.error(f"Error checking {language} environment: {e}")
                self.language_environments[language] = LanguageEnvironment(
                    language=language,
                    available=False,
                    error_message=str(e)
                )
    
    async def _check_go_environment(self) -> LanguageEnvironment:
        """檢測 Go 語言環境"""
        try:
            # 使用內建超時機制而非timeout參數
            async with asyncio.timeout(10):
                result = await self._run_command(["go", "version"])
                if result.returncode == 0:
                    version = result.stdout.strip()
                    go_path = shutil.which("go")
                    return LanguageEnvironment(
                        language="go",
                        available=True,
                        version=version,
                        binary_path=go_path
                    )
        except Exception as e:
            return LanguageEnvironment(
                language="go",
                available=False,
                error_message=f"Go not available: {e}"
            )
        
        return LanguageEnvironment(language="go", available=False)
    
    async def _check_ruby_environment(self) -> LanguageEnvironment:
        """檢測 Ruby 語言環境"""
        try:
            async with asyncio.timeout(10):
                result = await self._run_command(["ruby", "--version"])
                if result.returncode == 0:
                    version = result.stdout.strip()
                    ruby_path = shutil.which("ruby")
                    return LanguageEnvironment(
                        language="ruby",
                        available=True,
                        version=version,
                        binary_path=ruby_path
                    )
        except Exception as e:
            return LanguageEnvironment(
                language="ruby",
                available=False,
                error_message=f"Ruby not available: {e}"
            )
        
        return LanguageEnvironment(language="ruby", available=False)
    
    async def _check_python_environment(self) -> LanguageEnvironment:
        """檢測 Python 語言環境"""
        try:
            result = await self._run_command(["python3", "--version"], timeout_seconds=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                python_path = shutil.which("python3")
                return LanguageEnvironment(
                    language="python",
                    available=True,
                    version=version,
                    binary_path=python_path
                )
        except Exception as e:
            return LanguageEnvironment(
                language="python",
                available=False,
                error_message=f"Python not available: {e}"
            )
        
        return LanguageEnvironment(language="python", available=False)
    
    async def _check_rust_environment(self) -> LanguageEnvironment:
        """檢測 Rust 語言環境"""
        try:
            result = await self._run_command(["cargo", "--version"], timeout_seconds=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                cargo_path = shutil.which("cargo")
                return LanguageEnvironment(
                    language="rust",
                    available=True,
                    version=version,
                    binary_path=cargo_path
                )
        except Exception as e:
            return LanguageEnvironment(
                language="rust",
                available=False,
                error_message=f"Rust not available: {e}"
            )
        
        return LanguageEnvironment(language="rust", available=False)
    
    async def detect(
        self,
        target_url: str,
        mode: str = "comprehensive",
        max_concurrent: int = 3,
        tools_filter: Optional[List[str]] = None
    ) -> List[XSSDetectionResult]:
        """執行 XSS 檢測"""
        if not target_url:
            raise ValueError("Target URL is required")
        
        # 獲取可用工具
        available_tools = self._get_available_execution_plans(target_url, mode, tools_filter)
        if not available_tools:
            self.logger.warning("No tools available for execution")
            return []
        
        # 並行執行檢測
        return await self._execute_parallel_detection(available_tools, target_url, max_concurrent)
    
    def _get_available_execution_plans(
        self, 
        target_url: str, 
        mode: str, 
        tools_filter: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """獲取可用的執行計劃"""
        execution_plan = self.config.get_execution_plan(target_url, mode)
        
        # 過濾工具
        if tools_filter:
            execution_plan = [
                plan for plan in execution_plan
                if plan["tool_name"].lower() in [t.lower() for t in tools_filter]
            ]
        
        # 過濾可用語言的工具
        available_tools = []
        for plan in execution_plan:
            tool_config = self.config.get_tool_config(plan["tool_name"].lower())
            if tool_config and self._is_language_available(tool_config.language):
                available_tools.append(plan)
            else:
                tool_lang = tool_config.language if tool_config else 'unknown'
                self.logger.warning(f"Skipping {plan['tool_name']} - {tool_lang} environment not available")
        
        return available_tools
    
    async def _execute_parallel_detection(
        self, 
        available_tools: List[Dict[str, Any]], 
        target_url: str, 
        max_concurrent: int
    ) -> List[XSSDetectionResult]:
        """並行執行檢測任務"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        for plan in available_tools:
            default_timeout = plan.get("timeout", 300)
            task = self._execute_tool_detection(plan, target_url, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 處理結果
        detection_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Detection task failed: {result}")
            elif result:
                detection_results.append(result)
        
        return detection_results
    
    async def _execute_tool_detection(
        self,
        execution_plan: Dict[str, Any],
        target_url: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[XSSDetectionResult]:
        """執行單一工具檢測"""
        async with semaphore:
            tool_name = execution_plan["tool_name"]
            start_time = time.time()
            
            try:
                self.logger.info(f"Starting {tool_name} detection for {target_url}")
                
                tool_config = self.config.get_tool_config(tool_name.lower())
                if not tool_config:
                    raise ValueError(f"Tool config not found: {tool_name}")
                
                # 使用timeout情境管理器並傳遞參數
                timeout = execution_plan.get("timeout", 300)
                async with asyncio.timeout(timeout):
                    if tool_config.language == "go":
                        result = await self._execute_go_tool(tool_config, target_url, timeout)
                    elif tool_config.language == "ruby":
                        result = await self._execute_ruby_tool(tool_config, target_url, timeout)
                    elif tool_config.language == "python":
                        result = await self._execute_python_tool(tool_config, target_url, timeout)
                    elif tool_config.language == "rust":
                        result = await self._execute_rust_tool(tool_config, target_url, timeout)
                    else:
                        raise ValueError(f"Unsupported language: {tool_config.language}")
                
                execution_time = time.time() - start_time
                
                # 解析結果
                parsed_result = self._parse_tool_output(
                    tool_config, 
                    target_url, 
                    result.stdout, 
                    result.stderr,
                    execution_time
                )
                
                self.logger.info(f"{tool_name} completed in {execution_time:.2f}s")
                return parsed_result
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                error_msg = f"{tool_name} 執行超時"
                self.logger.error(error_msg)
                # ✅ 修復: 拋出異常耏非回傳空結果
                raise RuntimeError(
                    f"XSS 檢測工具執行超時: {tool_name}\n"
                    f"目標: {target_url}\n"
                    f"超時時間: {timeout}s\n"
                    f"建議: 增加 timeout 設置或檢查目標可訪問性"
                ) from None
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"{tool_name} 執行失敗: {e}")
                # ✅ 修復: 拋出異常而非回傳空結果 (移除偽陰性)
                raise RuntimeError(
                    f"XSS 檢測工具執行失敗: {tool_name}\n"
                    f"目標: {target_url}\n"
                    f"錯誤: {str(e)}\n"
                    f"請檢查:\n"
                    f"  1. 工具是否正確安裝\n"
                    f"  2. 目標 URL 是否可訪問\n"
                    f"  3. 網絡連接是否正常"
                ) from e
    
    async def _execute_go_tool(
        self, 
        tool_config: XSSToolConfig, 
        target_url: str,
        timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """執行 Go 語言工具"""
        if tool_config.name == "Dalfox":
            output_file = self.temp_dir / f"dalfox_output_{int(time.time())}.json"
            command = [
                "dalfox", "url", target_url,
                "--format", "json",
                "--output", str(output_file),
                "--silence"
            ]
            
            result = await self._run_command(command, timeout_seconds=timeout)
            
            # 讀取輸出檔案內容
            if output_file.exists():
                try:
                    json_output = output_file.read_text(encoding='utf-8')
                    result.stdout = json_output
                except Exception as e:
                    self.logger.warning(f"Failed to read Dalfox output file: {e}")
                finally:
                    output_file.unlink()  # 清理暫存檔案
            
            return result
        else:
            raise ValueError(f"Unsupported Go tool: {tool_config.name}")
    
    async def _execute_ruby_tool(
        self, 
        tool_config: XSSToolConfig, 
        target_url: str,
        timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """執行 Ruby 語言工具"""
        if tool_config.name == "XSpear":
            output_file = self.temp_dir / f"xspear_output_{int(time.time())}.json"
            command = [
                "XSpear", "-u", target_url,
                "--json-report", str(output_file)
            ]
            
            result = await self._run_command(command, timeout_seconds=timeout)
            
            # 讀取輸出檔案內容
            if output_file.exists():
                try:
                    json_output = output_file.read_text(encoding='utf-8')
                    result.stdout = json_output
                except Exception as e:
                    self.logger.warning(f"Failed to read XSpear output file: {e}")
                finally:
                    output_file.unlink()  # 清理暫存檔案
            
            return result
        else:
            raise ValueError(f"Unsupported Ruby tool: {tool_config.name}")
    
    async def _execute_python_tool(
        self, 
        tool_config: XSSToolConfig, 
        target_url: str,
        timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """執行 Python 語言工具"""
        # 根據工具名稱構建命令
        if "payload" in tool_config.name.lower():
            # XSSPayloadGenerator 類型工具
            command = ["python3", "-c", f"""
import requests
import json
target = '{target_url}'
payloads = ['<script>alert(1)</script>', '\\"\\'>alert(1)', '<img src=x onerror=alert(1)>']
results = []
for payload in payloads:
    try:
        response = requests.get(target, params={{'q': payload}}, timeout=10)
        if payload in response.text:
            results.append({{'payload': payload, 'vulnerable': True}})
        else:
            results.append({{'payload': payload, 'vulnerable': False}})
    except Exception:
        results.append({{'payload': payload, 'vulnerable': False, 'error': 'Request failed'}})
print(json.dumps(results))
"""]
        else:
            # 其他 Python 工具
            run_pattern = tool_config.run_pattern
            output_file = self.temp_dir / f"{tool_config.name.lower()}_output_{int(time.time())}.json"
            
            command_str = run_pattern.format(
                target=target_url,
                output_file=str(output_file)
            )
            command = ["bash", "-c", command_str]
        
        return await self._run_command(command, timeout_seconds=timeout)
    
    async def _execute_rust_tool(
        self, 
        tool_config: XSSToolConfig, 
        target_url: str,
        timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """執行 Rust 語言工具"""
        if tool_config.name == "RVuln":
            command = [
                "./target/release/RVuln",
                "-u", target_url,
                "--output", "json"
            ]
            return await self._run_command(command, timeout_seconds=timeout)
        else:
            raise ValueError(f"Unsupported Rust tool: {tool_config.name}")
    
    def _parse_tool_output(
        self,
        tool_config: XSSToolConfig,
        target_url: str,
        stdout: str,
        stderr: str,
        execution_time: float
    ) -> XSSDetectionResult:
        """解析工具輸出結果"""
        # 嘗試 JSON 解析
        json_result = self._parse_json_output(stdout)
        if json_result:
            return self._create_result_from_json(
                tool_config, target_url, json_result, execution_time, stderr
            )
        
        # 正則表達式解析
        regex_result = self._parse_regex_output(tool_config, stdout)
        return self._create_result_from_regex(
            tool_config, target_url, regex_result, execution_time, stdout, stderr
        )
    
    def _parse_json_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """解析 JSON 格式輸出"""
        try:
            if stdout.strip().startswith('{') or stdout.strip().startswith('['):
                return json.loads(stdout)
        except json.JSONDecodeError:
            pass
        return None
    
    def _create_result_from_json(
        self,
        tool_config: XSSToolConfig,
        target_url: str,
        json_data: Dict[str, Any],
        execution_time: float,
        stderr: str
    ) -> XSSDetectionResult:
        """從 JSON 數據創建結果"""
        if isinstance(json_data, dict):
            vulnerability_found = json_data.get('vulnerable', json_data.get('vulnerability', False))
            confidence = json_data.get('confidence', 0.5)
            payloads = json_data.get('payloads', [json_data.get('payload')] if json_data.get('payload') else [])
        elif isinstance(json_data, list):
            vulnerability_found = any(item.get('vulnerable', False) for item in json_data)
            payloads = [item.get('payload', '') for item in json_data if item.get('payload')]
            confidence = 0.7 if vulnerability_found else 0.0
        else:
            vulnerability_found = False
            confidence = 0.0
            payloads = []
        
        return XSSDetectionResult(
            tool_name=tool_config.name,
            language=tool_config.language,
            target_url=target_url,
            vulnerability_found=vulnerability_found,
            confidence=confidence,
            payloads=payloads,
            execution_time=execution_time,
            raw_output=str(json_data)[:1000],
            error_message=stderr if stderr else None
        )
    
    def _parse_regex_output(self, tool_config: XSSToolConfig, stdout: str) -> Dict[str, Any]:
        """使用正則表達式解析輸出"""
        result = {
            'vulnerability_found': False,
            'confidence': 0.0,
            'payloads': [],
            'severity': None
        }
        
        for pattern in tool_config.result_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                self._process_regex_matches(pattern, matches, result)
        
        return result
    
    def _process_regex_matches(self, pattern: str, matches: List[str], result: Dict[str, Any]) -> None:
        """處理正則表達式匹配結果"""
        pattern_lower = pattern.lower()
        
        if "vulnerability" in pattern_lower or "vulnerable" in pattern_lower:
            result['vulnerability_found'] = True
            result['confidence'] = 0.8
        elif "payload" in pattern_lower:
            if isinstance(matches[0], tuple):
                result['payloads'].extend([match[0] for match in matches])
            else:
                result['payloads'].extend(matches)
        elif "severity" in pattern_lower:
            result['severity'] = matches[0] if matches else None
        elif "confidence" in pattern_lower:
            try:
                result['confidence'] = float(matches[0]) / 100.0 if matches else 0.0
            except (ValueError, IndexError):
                result['confidence'] = 0.5
    
    def _create_result_from_regex(
        self,
        tool_config: XSSToolConfig,
        target_url: str,
        regex_result: Dict[str, Any],
        execution_time: float,
        stdout: str,
        stderr: str
    ) -> XSSDetectionResult:
        """從正則表達式結果創建結果"""
        return XSSDetectionResult(
            tool_name=tool_config.name,
            language=tool_config.language,
            target_url=target_url,
            vulnerability_found=regex_result['vulnerability_found'],
            confidence=regex_result['confidence'],
            payloads=regex_result['payloads'],
            execution_time=execution_time,
            raw_output=stdout[:1000],
            error_message=stderr if stderr else None,
            severity=regex_result['severity']
        )
    
    async def _run_command(
        self, 
        command: List[str], 
        cwd: Optional[str] = None,
        timeout_seconds: int = 300
    ) -> subprocess.CompletedProcess:
        """執行系統命令"""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout_seconds
            )
            
            return subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore')
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise asyncio.TimeoutError(f"Command timed out: {' '.join(command)}")
    
    def _is_language_available(self, language: str) -> bool:
        """檢查語言環境是否可用"""
        env = self.language_environments.get(language.lower())
        return env is not None and env.available
    
    def get_available_tools(self) -> List[str]:
        """獲取可用工具列表"""
        available_tools = []
        for tool_name, tool_config in self.config.tools.items():
            if self._is_language_available(tool_config.language):
                available_tools.append(tool_name)
        return available_tools
    
    def get_language_status(self) -> Dict[str, LanguageEnvironment]:
        """獲取語言環境狀態"""
        return self.language_environments.copy()
    
    def cleanup(self) -> None:
        """清理暫存資源"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """析構函數，確保資源清理"""
        self.cleanup()


# 創建全域引擎實例
_xss_engine_instance: Optional[CrossLanguageXSSEngine] = None


async def get_xss_engine() -> CrossLanguageXSSEngine:
    """獲取 XSS 檢測引擎實例"""
    global _xss_engine_instance
    if _xss_engine_instance is None:
        _xss_engine_instance = CrossLanguageXSSEngine()
        await _xss_engine_instance.initialize()
    return _xss_engine_instance


# 快捷方法
async def detect_xss(
    target_url: str,
    mode: str = "fast",
    tools: Optional[List[str]] = None
) -> List[XSSDetectionResult]:
    """快速 XSS 檢測方法"""
    engine = await get_xss_engine()
    return await engine.detect(target_url, mode=mode, tools_filter=tools)