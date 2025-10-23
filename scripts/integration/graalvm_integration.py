#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA GraalVM 多語言整合器
支援 Python、JavaScript、Java、Ruby 等語言間的無縫互操作
"""

import os
import json
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path

# GraalVM Python 支援
try:
    import polyglot
    GRAALVM_AVAILABLE = True
except ImportError:
    GRAALVM_AVAILABLE = False
    print("Warning: GraalVM polyglot not available. Please install GraalVM with Python support.")

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraalVMContext:
    """GraalVM 多語言執行環境"""
    
    def __init__(self, allowed_languages: Optional[List[str]] = None):
        self.allowed_languages = allowed_languages or ["python", "js", "java", "ruby", "R"]
        self.contexts = {}
        self.shared_data = {}
        self.logger = logging.getLogger("GraalVMContext")
        
        if not GRAALVM_AVAILABLE:
            self.logger.warning("GraalVM not available, using fallback mode")
    
    def initialize_context(self, language: str) -> bool:
        """初始化語言環境"""
        try:
            if not GRAALVM_AVAILABLE:
                return self._initialize_fallback_context(language)
            
            if language not in self.allowed_languages:
                raise ValueError(f"Language {language} not allowed")
            
            # 建立 GraalVM 環境
            context = polyglot.Context.create(language)
            self.contexts[language] = context
            
            # 設定共享資料
            context.get_bindings(language).put_member("shared", self.shared_data)
            
            self.logger.info(f"GraalVM context initialized for {language}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {language} context: {e}")
            return False
    
    def _initialize_fallback_context(self, language: str) -> bool:
        """回退模式初始化"""
        if language == "python":
            # Python 原生支援
            self.contexts[language] = {"type": "native_python"}
            return True
        elif language == "js":
            # 檢查 Node.js
            try:
                subprocess.run(["node", "--version"], check=True, capture_output=True)
                self.contexts[language] = {"type": "nodejs"}
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        else:
            return False
    
    def execute_code(self, language: str, code: str, 
                    context_vars: Optional[Dict[str, Any]] = None) -> Any:
        """執行多語言程式碼"""
        try:
            if language not in self.contexts:
                if not self.initialize_context(language):
                    raise RuntimeError(f"Cannot initialize {language} context")
            
            if GRAALVM_AVAILABLE:
                return self._execute_graalvm(language, code, context_vars)
            else:
                return self._execute_fallback(language, code, context_vars)
                
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            raise
    
    def _execute_graalvm(self, language: str, code: str, 
                        context_vars: Optional[Dict[str, Any]] = None) -> Any:
        """使用 GraalVM 執行程式碼"""
        context = self.contexts[language]
        
        # 設定變數
        if context_vars:
            for key, value in context_vars.items():
                context.get_bindings(language).put_member(key, value)
        
        # 執行程式碼
        return context.eval(language, code)
    
    def _execute_fallback(self, language: str, code: str, 
                         context_vars: Optional[Dict[str, Any]] = None) -> Any:
        """回退模式執行程式碼"""
        if language == "python":
            # 建立執行環境
            exec_globals = dict(self.shared_data)
            if context_vars:
                exec_globals.update(context_vars)
            
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            
            # 回傳 return 值或最後的變數
            if 'result' in exec_locals:
                return exec_locals['result']
            elif exec_locals:
                return exec_locals
            else:
                return None
        
        elif language == "js":
            # 使用 Node.js 執行
            return self._execute_nodejs(code, context_vars)
        
        else:
            raise NotImplementedError(f"Fallback execution not implemented for {language}")
    
    def _execute_nodejs(self, code: str, context_vars: Optional[Dict[str, Any]] = None) -> Any:
        """使用 Node.js 執行 JavaScript"""
        try:
            # 建立臨時檔案
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                # 寫入變數設定
                if context_vars:
                    for key, value in context_vars.items():
                        f.write(f"const {key} = {json.dumps(value)};\n")
                
                # 寫入程式碼
                f.write(code)
                f.write("\n")
                
                # 如果程式碼沒有 console.log，加入回傳邏輯
                if "console.log" not in code and "return" not in code:
                    f.write("console.log(JSON.stringify(result || {}));\n")
                
                temp_file = f.name
            
            # 執行 Node.js
            result = subprocess.run(
                ["node", temp_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            # 清理臨時檔案
            os.unlink(temp_file)
            
            # 解析結果
            output = result.stdout.strip()
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return output
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Node.js execution failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"JavaScript execution error: {e}")
    
    def call_function(self, language: str, function_name: str, *args) -> Any:
        """調用跨語言函數"""
        if language not in self.contexts:
            raise ValueError(f"Context for {language} not initialized")
        
        if GRAALVM_AVAILABLE:
            context = self.contexts[language]
            bindings = context.get_bindings(language)
            function = bindings.get_member(function_name)
            
            if function and function.can_execute():
                return function.execute(*args)
            else:
                raise ValueError(f"Function {function_name} not found or not executable")
        else:
            # 回退模式調用
            return self._call_function_fallback(language, function_name, *args)
    
    def _call_function_fallback(self, language: str, function_name: str, *args) -> Any:
        """回退模式函數調用"""
        if language == "python":
            # 從 shared_data 或全域空間尋找函數
            if function_name in self.shared_data:
                func = self.shared_data[function_name]
                if callable(func):
                    return func(*args)
            raise ValueError(f"Function {function_name} not found")
        else:
            raise NotImplementedError(f"Function call not implemented for {language}")
    
    def share_object(self, name: str, obj: Any):
        """在語言間共享物件"""
        self.shared_data[name] = obj
        
        # 更新所有已初始化的環境
        for language, context in self.contexts.items():
            try:
                if GRAALVM_AVAILABLE and hasattr(context, 'get_bindings'):
                    context.get_bindings(language).put_member(name, obj)
            except Exception as e:
                self.logger.warning(f"Failed to share object {name} with {language}: {e}")

class AIVASecurityScanner:
    """AIVA 多語言安全掃描器"""
    
    def __init__(self, graalvm_context: GraalVMContext):
        self.context = graalvm_context
        self.logger = logging.getLogger("AIVASecurityScanner")
        
        # 初始化掃描規則
        self.vulnerability_patterns = {
            "python": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"__import__\s*\(",
            ],
            "javascript": [
                r"eval\s*\(",
                r"Function\s*\(",
                r"document\.write\s*\(",
                r"innerHTML\s*=",
                r"setTimeout\s*\(",
            ],
            "java": [
                r"Runtime\.getRuntime\(\)\.exec",
                r"ProcessBuilder",
                r"Class\.forName",
                r"Method\.invoke",
            ]
        }
    
    async def scan_code_multilang(self, codes: Dict[str, str]) -> Dict[str, Any]:
        """多語言程式碼安全掃描"""
        results = {}
        
        for language, code in codes.items():
            try:
                # 使用對應語言的掃描邏輯
                if language == "python":
                    result = await self._scan_python(code)
                elif language == "javascript":
                    result = await self._scan_javascript(code)
                elif language == "java":
                    result = await self._scan_java(code)
                else:
                    result = await self._scan_generic(language, code)
                
                results[language] = result
                
            except Exception as e:
                self.logger.error(f"Scan failed for {language}: {e}")
                results[language] = {
                    "vulnerabilities": [],
                    "error": str(e)
                }
        
        return results
    
    async def _scan_python(self, code: str) -> Dict[str, Any]:
        """Python 程式碼掃描"""
        # 可以整合 bandit 或其他 Python 安全工具
        python_scanner_code = """
import re
import ast

def scan_python_code(code):
    vulnerabilities = []
    
    # 靜態模式匹配
    patterns = [
        (r'eval\\s*\\(', 'Code Injection - eval()'),
        (r'exec\\s*\\(', 'Code Injection - exec()'),
        (r'os\\.system\\s*\\(', 'Command Injection - os.system()'),
        (r'subprocess\\.call\\s*\\(', 'Command Injection - subprocess.call()'),
        (r'__import__\\s*\\(', 'Dynamic Import - __import__()'),
        (r'pickle\\.loads\\s*\\(', 'Deserialization - pickle.loads()'),
    ]
    
    for pattern, description in patterns:
        matches = re.findall(pattern, code)
        if matches:
            vulnerabilities.append({
                'type': 'pattern_match',
                'description': description,
                'pattern': pattern,
                'matches': len(matches)
            })
    
    # AST 分析
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        vulnerabilities.append({
                            'type': 'ast_analysis',
                            'description': f'Dangerous function call: {node.func.id}',
                            'line': node.lineno
                        })
    except SyntaxError:
        vulnerabilities.append({
            'type': 'syntax_error',
            'description': 'Code contains syntax errors'
        })
    
    return {
        'vulnerabilities': vulnerabilities,
        'scan_time': time.time(),
        'language': 'python'
    }

import time
result = scan_python_code(code_to_scan)
"""
        
        try:
            # 在 GraalVM 中執行掃描
            self.context.share_object("code_to_scan", code)
            result = self.context.execute_code("python", python_scanner_code)
            return result if result else {"vulnerabilities": [], "error": "No result"}
            
        except Exception as e:
            self.logger.error(f"Python scan error: {e}")
            return {"vulnerabilities": [], "error": str(e)}
    
    async def _scan_javascript(self, code: str) -> Dict[str, Any]:
        """JavaScript 程式碼掃描"""
        js_scanner_code = """
const code = code_to_scan;
const vulnerabilities = [];

// 危險模式檢查
const patterns = [
    [/eval\\s*\\(/g, 'Code Injection - eval()'],
    [/Function\\s*\\(/g, 'Code Injection - Function()'],
    [/document\\.write\\s*\\(/g, 'XSS - document.write()'],
    [/innerHTML\\s*=/g, 'XSS - innerHTML'],
    [/setTimeout\\s*\\(/g, 'Code Injection - setTimeout()'],
    [/setInterval\\s*\\(/g, 'Code Injection - setInterval()'],
];

patterns.forEach(([pattern, description]) => {
    const matches = code.match(pattern);
    if (matches) {
        vulnerabilities.push({
            type: 'pattern_match',
            description: description,
            matches: matches.length
        });
    }
});

// 檢查危險的 DOM 操作
if (code.includes('document.cookie')) {
    vulnerabilities.push({
        type: 'cookie_access',
        description: 'Direct cookie access detected'
    });
}

const result = {
    vulnerabilities: vulnerabilities,
    scan_time: Date.now(),
    language: 'javascript'
};

result;
"""
        
        try:
            self.context.share_object("code_to_scan", code)
            result = self.context.execute_code("js", js_scanner_code)
            return result if result else {"vulnerabilities": [], "error": "No result"}
            
        except Exception as e:
            self.logger.error(f"JavaScript scan error: {e}")
            return {"vulnerabilities": [], "error": str(e)}
    
    async def _scan_java(self, code: str) -> Dict[str, Any]:
        """Java 程式碼掃描 (簡化版)"""
        import re
        
        vulnerabilities = []
        
        # 模式匹配
        patterns = [
            (r'Runtime\.getRuntime\(\)\.exec', 'Command Injection - Runtime.exec()'),
            (r'ProcessBuilder', 'Command Injection - ProcessBuilder'),
            (r'Class\.forName', 'Reflection - Class.forName()'),
            (r'Method\.invoke', 'Reflection - Method.invoke()'),
            (r'PreparedStatement.*\+', 'SQL Injection - String concatenation'),
        ]
        
        for pattern, description in patterns:
            matches = re.findall(pattern, code)
            if matches:
                vulnerabilities.append({
                    'type': 'pattern_match',
                    'description': description,
                    'matches': len(matches)
                })
        
        return {
            'vulnerabilities': vulnerabilities,
            'scan_time': time.time(),
            'language': 'java'
        }
    
    async def _scan_generic(self, language: str, code: str) -> Dict[str, Any]:
        """通用程式碼掃描"""
        import re
        
        vulnerabilities = []
        
        # 通用危險模式
        patterns = [
            (r'system\s*\(', 'Potential system call'),
            (r'exec\s*\(', 'Potential code execution'),
            (r'eval\s*\(', 'Potential code evaluation'),
            (r'shell_exec\s*\(', 'Shell execution'),
        ]
        
        for pattern, description in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                vulnerabilities.append({
                    'type': 'pattern_match',
                    'description': description,
                    'matches': len(matches)
                })
        
        return {
            'vulnerabilities': vulnerabilities,
            'scan_time': time.time(),
            'language': language
        }

class AIVAGraalVMManager:
    """AIVA GraalVM 整合管理器"""
    
    def __init__(self):
        self.context = GraalVMContext()
        self.scanner = AIVASecurityScanner(self.context)
        self.logger = logging.getLogger("AIVAGraalVMManager")
    
    async def initialize_all_languages(self) -> Dict[str, bool]:
        """初始化所有支援的語言"""
        results = {}
        
        for language in self.context.allowed_languages:
            success = self.context.initialize_context(language)
            results[language] = success
            
            if success:
                self.logger.info(f"✅ {language} context initialized")
            else:
                self.logger.warning(f"❌ {language} context initialization failed")
        
        return results
    
    async def run_multilang_security_scan(self, project_path: str) -> Dict[str, Any]:
        """執行多語言專案安全掃描"""
        project_path = Path(project_path)
        if not project_path.exists():
            raise FileNotFoundError(f"Project path not found: {project_path}")
        
        # 尋找不同語言的程式碼檔案
        code_files = {
            "python": list(project_path.rglob("*.py")),
            "javascript": list(project_path.rglob("*.js")) + list(project_path.rglob("*.ts")),
            "java": list(project_path.rglob("*.java")),
            "go": list(project_path.rglob("*.go")),
            "rust": list(project_path.rglob("*.rs")),
        }
        
        all_results = {}
        
        for language, files in code_files.items():
            if not files:
                continue
            
            language_results = []
            
            for file_path in files[:10]:  # 限制每種語言最多掃描 10 個檔案
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # 執行掃描
                    if language in ["python", "javascript", "java"]:
                        scan_result = await self.scanner.scan_code_multilang({language: code})
                        result = scan_result.get(language, {})
                    else:
                        result = await self.scanner._scan_generic(language, code)
                    
                    result['file'] = str(file_path)
                    language_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to scan {file_path}: {e}")
                    language_results.append({
                        'file': str(file_path),
                        'vulnerabilities': [],
                        'error': str(e)
                    })
            
            if language_results:
                all_results[language] = language_results
        
        return all_results
    
    async def execute_cross_language_workflow(self, workflow_config: Dict[str, Any]) -> Any:
        """執行跨語言工作流程"""
        """
        workflow_config 範例:
        {
            "steps": [
                {
                    "language": "python",
                    "code": "result = [1, 2, 3, 4, 5]",
                    "output_var": "numbers"
                },
                {
                    "language": "javascript", 
                    "code": "const sum = numbers.reduce((a, b) => a + b, 0); result = sum;",
                    "input_vars": ["numbers"],
                    "output_var": "sum"
                },
                {
                    "language": "python",
                    "code": "final_result = f'Total sum: {sum}'",
                    "input_vars": ["sum"]
                }
            ]
        }
        """
        
        shared_vars = {}
        results = []
        
        for step in workflow_config.get("steps", []):
            language = step.get("language")
            code = step.get("code")
            input_vars = step.get("input_vars", [])
            output_var = step.get("output_var")
            
            # 準備輸入變數
            context_vars = {}
            for var_name in input_vars:
                if var_name in shared_vars:
                    context_vars[var_name] = shared_vars[var_name]
            
            # 執行程式碼
            try:
                result = self.context.execute_code(language, code, context_vars)
                
                # 儲存輸出變數
                if output_var and result is not None:
                    shared_vars[output_var] = result
                
                step_result = {
                    "language": language,
                    "success": True,
                    "result": result,
                    "output_var": output_var
                }
                
                results.append(step_result)
                self.logger.info(f"Step completed: {language} -> {output_var}")
                
            except Exception as e:
                self.logger.error(f"Step failed: {language} - {e}")
                step_result = {
                    "language": language,
                    "success": False,
                    "error": str(e)
                }
                results.append(step_result)
                break  # 工作流程中斷
        
        return {
            "steps": results,
            "shared_vars": shared_vars,
            "success": all(step.get("success", False) for step in results)
        }

# 使用範例
async def demo_graalvm_integration():
    """示範 GraalVM 多語言整合"""
    manager = AIVAGraalVMManager()
    
    try:
        # 初始化語言環境
        print("初始化多語言環境...")
        init_results = await manager.initialize_all_languages()
        
        for lang, success in init_results.items():
            status = "✅" if success else "❌"
            print(f"{status} {lang}")
        
        # 執行跨語言工作流程
        print("\n執行跨語言工作流程...")
        workflow = {
            "steps": [
                {
                    "language": "python",
                    "code": "result = list(range(1, 11))",
                    "output_var": "numbers"
                },
                {
                    "language": "javascript",
                    "code": "const doubled = numbers.map(x => x * 2); result = doubled;",
                    "input_vars": ["numbers"],
                    "output_var": "doubled_numbers"
                },
                {
                    "language": "python",
                    "code": "result = sum(doubled_numbers)",
                    "input_vars": ["doubled_numbers"],
                    "output_var": "final_sum"
                }
            ]
        }
        
        workflow_result = await manager.execute_cross_language_workflow(workflow)
        print(f"工作流程結果: {workflow_result}")
        
        # 執行安全掃描
        print("\n執行多語言安全掃描...")
        scan_results = await manager.run_multilang_security_scan("C:/D/fold7/AIVA-git")
        
        for language, results in scan_results.items():
            print(f"\n{language.upper()} 掃描結果:")
            for result in results[:3]:  # 只顯示前3個結果
                vuln_count = len(result.get('vulnerabilities', []))
                file_name = Path(result.get('file', 'unknown')).name
                print(f"  {file_name}: {vuln_count} 個潛在問題")
        
    except Exception as e:
        print(f"示範過程發生錯誤: {e}")

if __name__ == "__main__":
    # 檢查 GraalVM 環境
    if GRAALVM_AVAILABLE:
        print("GraalVM 多語言環境可用")
    else:
        print("GraalVM 不可用，使用回退模式")
        print("要獲得完整功能，請安裝 GraalVM 和 Python 支援")
    
    # 執行示範
    import time
    asyncio.run(demo_graalvm_integration())