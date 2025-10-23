#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA WebAssembly 整合器
支援將 Rust/C++ 模組編譯為 WASM 並在 Python 中執行
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import subprocess
import tempfile
import shutil

# WebAssembly 執行環境
try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False
    print("Warning: wasmtime not available. Install with: pip install wasmtime-py")

try:
    import wasmer
    WASMER_AVAILABLE = True
except ImportError:
    WASMER_AVAILABLE = False
    print("Warning: wasmer not available. Install with: pip install wasmer wasmer-compiler-cranelift")

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WASMModule:
    """WebAssembly 模組封裝"""
    
    def __init__(self, name: str, wasm_path: str, runtime: str = "wasmtime"):
        self.name = name
        self.wasm_path = Path(wasm_path)
        self.runtime = runtime
        self.module = None
        self.instance = None
        self.store = None
        self.memory = None
        self.logger = logging.getLogger(f"WASM.{name}")
        
        if not self.wasm_path.exists():
            raise FileNotFoundError(f"WASM file not found: {wasm_path}")
    
    async def load(self) -> bool:
        """載入 WASM 模組"""
        try:
            if self.runtime == "wasmtime" and WASMTIME_AVAILABLE:
                return await self._load_wasmtime()
            elif self.runtime == "wasmer" and WASMER_AVAILABLE:
                return await self._load_wasmer()
            else:
                raise RuntimeError(f"Runtime {self.runtime} not available")
        except Exception as e:
            self.logger.error(f"Failed to load WASM module: {e}")
            return False
    
    async def _load_wasmtime(self) -> bool:
        """使用 Wasmtime 載入模組"""
        engine = wasmtime.Engine()
        self.store = wasmtime.Store(engine)
        
        with open(self.wasm_path, 'rb') as f:
            wasm_bytes = f.read()
        
        self.module = wasmtime.Module(engine, wasm_bytes)
        self.instance = wasmtime.Instance(self.store, self.module, [])
        
        # 獲取記憶體
        memory_export = self.instance.exports(self.store).get("memory")
        if memory_export:
            self.memory = memory_export
        
        self.logger.info(f"Wasmtime module loaded: {self.name}")
        return True
    
    async def _load_wasmer(self) -> bool:
        """使用 Wasmer 載入模組"""
        store = wasmer.Store()
        
        with open(self.wasm_path, 'rb') as f:
            wasm_bytes = f.read()
        
        self.module = wasmer.Module(store, wasm_bytes)
        
        # 建立匯入物件 (如果需要)
        import_object = wasmer.ImportObject()
        
        self.instance = wasmer.Instance(self.module, import_object)
        self.store = store
        
        self.logger.info(f"Wasmer module loaded: {self.name}")
        return True
    
    def call_function(self, func_name: str, *args) -> Any:
        """調用 WASM 函數"""
        try:
            if self.runtime == "wasmtime":
                func = self.instance.exports(self.store)[func_name]
                return func(self.store, *args)
            elif self.runtime == "wasmer":
                func = self.instance.exports.__getattribute__(func_name)
                return func(*args)
        except Exception as e:
            self.logger.error(f"Function call failed: {e}")
            raise
    
    def get_exported_functions(self) -> List[str]:
        """獲取匯出函數列表"""
        try:
            if self.runtime == "wasmtime":
                exports = self.instance.exports(self.store)
                return [name for name, export in exports.items() 
                       if isinstance(export, wasmtime.Func)]
            elif self.runtime == "wasmer":
                return list(self.instance.exports.__dict__.keys())
        except Exception as e:
            self.logger.error(f"Failed to get exports: {e}")
            return []
    
    def read_memory(self, offset: int, length: int) -> bytes:
        """讀取 WASM 記憶體"""
        if not self.memory:
            raise RuntimeError("Memory not available")
        
        if self.runtime == "wasmtime":
            data = self.memory.data(self.store)
            return data[offset:offset + length]
        elif self.runtime == "wasmer":
            view = self.memory.uint8_view()
            return bytes(view[offset:offset + length])
    
    def write_memory(self, offset: int, data: bytes):
        """寫入 WASM 記憶體"""
        if not self.memory:
            raise RuntimeError("Memory not available")
        
        if self.runtime == "wasmtime":
            memory_data = self.memory.data(self.store)
            memory_data[offset:offset + len(data)] = data
        elif self.runtime == "wasmer":
            view = self.memory.uint8_view()
            for i, byte in enumerate(data):
                view[offset + i] = byte

class WASMCompiler:
    """WebAssembly 編譯器"""
    
    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("WASMCompiler")
    
    async def compile_rust_to_wasm(self, source_dir: str, target_name: str, 
                                 features: Optional[List[str]] = None) -> Optional[str]:
        """編譯 Rust 專案為 WASM"""
        try:
            source_path = Path(source_dir)
            if not source_path.exists():
                raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
            # 檢查是否有 Cargo.toml
            cargo_toml = source_path / "Cargo.toml"
            if not cargo_toml.exists():
                raise FileNotFoundError("Cargo.toml not found in source directory")
            
            # 建構編譯命令
            cmd = ["cargo", "build", "--target", "wasm32-unknown-unknown", "--release"]
            
            if features:
                cmd.extend(["--features", ",".join(features)])
            
            # 執行編譯
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=source_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Rust compilation failed: {stderr.decode()}")
                return None
            
            # 尋找生成的 WASM 檔案
            wasm_file = source_path / "target" / "wasm32-unknown-unknown" / "release" / f"{target_name}.wasm"
            
            if not wasm_file.exists():
                self.logger.error(f"WASM file not found: {wasm_file}")
                return None
            
            # 複製到工作目錄
            output_path = self.work_dir / f"{target_name}.wasm"
            shutil.copy2(wasm_file, output_path)
            
            self.logger.info(f"Rust WASM compiled: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Rust compilation error: {e}")
            return None
    
    async def compile_c_to_wasm(self, source_files: List[str], target_name: str,
                              include_dirs: Optional[List[str]] = None,
                              libraries: Optional[List[str]] = None) -> Optional[str]:
        """編譯 C/C++ 為 WASM (需要 Emscripten)"""
        try:
            # 建構 emcc 命令
            cmd = ["emcc"]
            cmd.extend(source_files)
            cmd.extend(["-o", f"{target_name}.wasm"])
            cmd.extend(["-O3", "-s", "WASM=1", "-s", "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']"])
            
            if include_dirs:
                for inc_dir in include_dirs:
                    cmd.extend(["-I", inc_dir])
            
            if libraries:
                for lib in libraries:
                    cmd.extend(["-l", lib])
            
            # 執行編譯
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"C/C++ compilation failed: {stderr.decode()}")
                return None
            
            output_path = self.work_dir / f"{target_name}.wasm"
            if not output_path.exists():
                self.logger.error(f"WASM file not found: {output_path}")
                return None
            
            self.logger.info(f"C/C++ WASM compiled: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"C/C++ compilation error: {e}")
            return None
    
    def optimize_wasm(self, wasm_path: str) -> bool:
        """優化 WASM 檔案 (需要 wasm-opt)"""
        try:
            cmd = ["wasm-opt", "-O3", wasm_path, "-o", wasm_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"WASM optimized: {wasm_path}")
                return True
            else:
                self.logger.error(f"WASM optimization failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            self.logger.warning("wasm-opt not found, skipping optimization")
            return False
        except Exception as e:
            self.logger.error(f"WASM optimization error: {e}")
            return False

class WASMSecurityScanner:
    """WebAssembly 安全掃描器"""
    
    def __init__(self, wasm_module: WASMModule):
        self.wasm_module = wasm_module
        self.logger = logging.getLogger("WASMSecurityScanner")
    
    async def scan_code(self, code: str, language: str = "rust") -> Dict[str, Any]:
        """使用 WASM 模組掃描程式碼"""
        try:
            # 將程式碼寫入 WASM 記憶體
            code_bytes = code.encode('utf-8')
            
            # 分配記憶體 (假設有 malloc 函數)
            try:
                malloc_func = self.wasm_module.call_function("malloc", len(code_bytes))
                code_ptr = malloc_func
            except:
                # 如果沒有 malloc，使用固定位置
                code_ptr = 0
            
            # 寫入程式碼
            self.wasm_module.write_memory(code_ptr, code_bytes)
            
            # 調用掃描函數
            scan_result = self.wasm_module.call_function("scan_vulnerabilities", 
                                                       code_ptr, len(code_bytes))
            
            # 讀取結果
            if scan_result > 0:
                result_data = self.wasm_module.read_memory(scan_result, 1024)
                result_str = result_data.decode('utf-8').rstrip('\x00')
                return json.loads(result_str)
            else:
                return {"vulnerabilities": [], "error": "Scan failed"}
                
        except Exception as e:
            self.logger.error(f"WASM scan error: {e}")
            return {"vulnerabilities": [], "error": str(e)}
    
    async def batch_scan(self, files: List[str]) -> List[Dict[str, Any]]:
        """批次掃描多個檔案"""
        results = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # 根據檔案副檔名判斷語言
                suffix = Path(file_path).suffix.lower()
                language_map = {
                    '.rs': 'rust',
                    '.go': 'go',
                    '.py': 'python',
                    '.js': 'javascript',
                    '.ts': 'typescript',
                    '.c': 'c',
                    '.cpp': 'cpp',
                    '.java': 'java'
                }
                language = language_map.get(suffix, 'unknown')
                
                scan_result = await self.scan_code(code, language)
                scan_result['file'] = file_path
                scan_result['language'] = language
                
                results.append(scan_result)
                
            except Exception as e:
                self.logger.error(f"Failed to scan {file_path}: {e}")
                results.append({
                    'file': file_path,
                    'vulnerabilities': [],
                    'error': str(e)
                })
        
        return results

class AIVAWASMManager:
    """AIVA WebAssembly 管理器"""
    
    def __init__(self, modules_dir: str = "wasm_modules"):
        self.modules_dir = Path(modules_dir)
        self.modules_dir.mkdir(exist_ok=True)
        self.modules: Dict[str, WASMModule] = {}
        self.compiler = WASMCompiler()
        self.logger = logging.getLogger("AIVAWASMManager")
    
    async def load_module(self, name: str, wasm_path: str, runtime: str = "wasmtime") -> bool:
        """載入 WASM 模組"""
        try:
            module = WASMModule(name, wasm_path, runtime)
            success = await module.load()
            
            if success:
                self.modules[name] = module
                self.logger.info(f"Module loaded: {name}")
                return True
            else:
                self.logger.error(f"Failed to load module: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Load module error: {e}")
            return False
    
    async def compile_and_load_rust_sast(self) -> bool:
        """編譯並載入 Rust SAST 模組"""
        source_dir = "C:/D/fold7/AIVA-git/services/features/function_sast_rust"
        
        # 編譯為 WASM
        wasm_path = await self.compiler.compile_rust_to_wasm(
            source_dir=source_dir,
            target_name="aiva_sast",
            features=["wasm"]
        )
        
        if not wasm_path:
            return False
        
        # 載入模組
        return await self.load_module("rust_sast", wasm_path)
    
    async def compile_and_load_rust_info_gatherer(self) -> bool:
        """編譯並載入 Rust Info Gatherer 模組"""
        source_dir = "C:/D/fold7/AIVA-git/services/scan/info_gatherer_rust"
        
        # 編譯為 WASM
        wasm_path = await self.compiler.compile_rust_to_wasm(
            source_dir=source_dir,
            target_name="aiva_info_gatherer",
            features=["wasm"]
        )
        
        if not wasm_path:
            return False
        
        # 載入模組
        return await self.load_module("rust_info_gatherer", wasm_path)
    
    def get_module(self, name: str) -> Optional[WASMModule]:
        """獲取 WASM 模組"""
        return self.modules.get(name)
    
    def list_modules(self) -> List[str]:
        """列出所有載入的模組"""
        return list(self.modules.keys())
    
    async def call_module_function(self, module_name: str, function_name: str, 
                                 *args) -> Any:
        """調用模組函數"""
        module = self.modules.get(module_name)
        if not module:
            raise ValueError(f"Module not found: {module_name}")
        
        return module.call_function(function_name, *args)
    
    def create_security_scanner(self, module_name: str) -> Optional[WASMSecurityScanner]:
        """建立安全掃描器"""
        module = self.modules.get(module_name)
        if not module:
            return None
        
        return WASMSecurityScanner(module)

# 使用範例
async def demo_wasm_integration():
    """示範 WebAssembly 整合"""
    manager = AIVAWASMManager()
    
    try:
        # 編譯並載入 Rust SAST 模組
        print("編譯 Rust SAST 模組...")
        if await manager.compile_and_load_rust_sast():
            print("✅ Rust SAST 模組載入成功")
            
            # 建立安全掃描器
            scanner = manager.create_security_scanner("rust_sast")
            if scanner:
                # 測試掃描
                test_code = """
                fn main() {
                    let user_input = std::env::args().nth(1).unwrap();
                    println!("User input: {}", user_input);
                }
                """
                
                result = await scanner.scan_code(test_code, "rust")
                print(f"掃描結果: {result}")
        else:
            print("❌ Rust SAST 模組載入失敗")
        
        # 列出所有模組
        modules = manager.list_modules()
        print(f"已載入的模組: {modules}")
        
    except Exception as e:
        print(f"示範過程發生錯誤: {e}")

if __name__ == "__main__":
    # 檢查依賴
    if not WASMTIME_AVAILABLE and not WASMER_AVAILABLE:
        print("請安裝 WebAssembly 執行環境:")
        print("pip install wasmtime-py")
        print("或")
        print("pip install wasmer wasmer-compiler-cranelift")
    else:
        print("WebAssembly 執行環境可用")
        # 執行示範
        asyncio.run(demo_wasm_integration())