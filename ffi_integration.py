#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA FFI (Foreign Function Interface) 整合器
支援 Python 與 C/C++、Rust、Go 等語言的直接函數調用
"""

import os
import json
import logging
import asyncio
import tempfile
import subprocess
import ctypes
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import platform

# FFI 函式庫支援
try:
    import cffi
    CFFI_AVAILABLE = True
except ImportError:
    CFFI_AVAILABLE = False
    print("Warning: cffi not available. Install with: pip install cffi")

try:
    from ctypes import cdll, CDLL, c_char_p, c_int, c_double, c_void_p, Structure, pointer
    CTYPES_AVAILABLE = True
except ImportError:
    CTYPES_AVAILABLE = False

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FFILibrary:
    """FFI 函式庫封裝"""
    
    def __init__(self, name: str, library_path: str, use_cffi: bool = True):
        self.name = name
        self.library_path = Path(library_path)
        self.use_cffi = use_cffi and CFFI_AVAILABLE
        self.library = None
        self.ffi = None
        self.functions = {}
        self.logger = logging.getLogger(f"FFI.{name}")
        
        if not self.library_path.exists():
            raise FileNotFoundError(f"Library not found: {library_path}")
    
    def load_library(self) -> bool:
        """載入動態函式庫"""
        try:
            if self.use_cffi:
                return self._load_with_cffi()
            else:
                return self._load_with_ctypes()
        except Exception as e:
            self.logger.error(f"Failed to load library: {e}")
            return False
    
    def _load_with_cffi(self) -> bool:
        """使用 CFFI 載入函式庫"""
        self.ffi = cffi.FFI()
        
        # 載入函式庫
        self.library = self.ffi.dlopen(str(self.library_path))
        
        self.logger.info(f"CFFI library loaded: {self.name}")
        return True
    
    def _load_with_ctypes(self) -> bool:
        """使用 ctypes 載入函式庫"""
        if platform.system() == "Windows":
            self.library = ctypes.WinDLL(str(self.library_path))
        else:
            self.library = ctypes.CDLL(str(self.library_path))
        
        self.logger.info(f"ctypes library loaded: {self.name}")
        return True
    
    def define_function(self, func_name: str, signature: str, 
                       return_type: str = "int") -> bool:
        """定義函數簽名"""
        try:
            if self.use_cffi:
                return self._define_function_cffi(func_name, signature, return_type)
            else:
                return self._define_function_ctypes(func_name, signature, return_type)
        except Exception as e:
            self.logger.error(f"Failed to define function {func_name}: {e}")
            return False
    
    def _define_function_cffi(self, func_name: str, signature: str, 
                             return_type: str) -> bool:
        """使用 CFFI 定義函數"""
        # 建構完整的函數聲明
        full_signature = f"{return_type} {func_name}({signature});"
        
        # 定義函數
        self.ffi.cdef(full_signature)
        
        # 獲取函數引用
        self.functions[func_name] = getattr(self.library, func_name)
        
        self.logger.info(f"CFFI function defined: {func_name}")
        return True
    
    def _define_function_ctypes(self, func_name: str, signature: str, 
                              return_type: str) -> bool:
        """使用 ctypes 定義函數"""
        # 獲取函數
        func = getattr(self.library, func_name)
        
        # 設定回傳類型
        type_map = {
            "int": c_int,
            "double": c_double,
            "char*": c_char_p,
            "void*": c_void_p,
            "void": None
        }
        
        func.restype = type_map.get(return_type, c_int)
        
        # TODO: 解析參數類型並設定 argtypes
        # 這裡需要更複雜的簽名解析邏輯
        
        self.functions[func_name] = func
        
        self.logger.info(f"ctypes function defined: {func_name}")
        return True
    
    def call_function(self, func_name: str, *args) -> Any:
        """調用函數"""
        if func_name not in self.functions:
            raise ValueError(f"Function {func_name} not defined")
        
        try:
            func = self.functions[func_name]
            return func(*args)
        except Exception as e:
            self.logger.error(f"Function call failed: {e}")
            raise
    
    def get_function_list(self) -> List[str]:
        """獲取已定義的函數列表"""
        return list(self.functions.keys())

class RustFFIBuilder:
    """Rust FFI 函式庫建構器"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.logger = logging.getLogger("RustFFIBuilder")
    
    async def build_ffi_library(self, target_name: str = None) -> Optional[str]:
        """建構 Rust FFI 函式庫"""
        try:
            if not (self.project_path / "Cargo.toml").exists():
                raise FileNotFoundError("Cargo.toml not found")
            
            # 修改 Cargo.toml 以支援 FFI
            await self._prepare_cargo_toml(target_name)
            
            # 建構為動態函式庫
            cmd = ["cargo", "build", "--release"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Rust build failed: {stderr.decode()}")
                return None
            
            # 尋找生成的動態函式庫
            lib_path = await self._find_generated_library(target_name)
            
            if lib_path:
                self.logger.info(f"Rust FFI library built: {lib_path}")
                return str(lib_path)
            else:
                self.logger.error("Generated library not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Rust FFI build error: {e}")
            return None
    
    async def _prepare_cargo_toml(self, target_name: str = None):
        """準備 Cargo.toml 以支援 FFI"""
        cargo_toml_path = self.project_path / "Cargo.toml"
        
        # 讀取現有的 Cargo.toml
        with open(cargo_toml_path, 'r') as f:
            content = f.read()
        
        # 檢查是否已經有 [lib] 部分
        if "[lib]" not in content:
            # 新增 FFI 配置
            ffi_config = """
[lib]
name = "{}"
crate-type = ["cdylib"]
""".format(target_name or "aiva_ffi")
            
            content += ffi_config
            
            # 寫入修改後的 Cargo.toml
            with open(cargo_toml_path, 'w') as f:
                f.write(content)
            
            self.logger.info("Cargo.toml updated for FFI support")
    
    async def _find_generated_library(self, target_name: str = None) -> Optional[Path]:
        """尋找生成的動態函式庫"""
        target_dir = self.project_path / "target" / "release"
        
        if not target_dir.exists():
            return None
        
        # 根據作業系統尋找對應的檔案
        system = platform.system()
        lib_name = target_name or "aiva_ffi"
        
        if system == "Windows":
            lib_file = target_dir / f"{lib_name}.dll"
        elif system == "Darwin":
            lib_file = target_dir / f"lib{lib_name}.dylib"
        else:
            lib_file = target_dir / f"lib{lib_name}.so"
        
        return lib_file if lib_file.exists() else None
    
    def generate_rust_ffi_code(self, functions: List[Dict[str, Any]]) -> str:
        """生成 Rust FFI 程式碼"""
        """
        functions 範例:
        [
            {
                "name": "scan_vulnerabilities",
                "params": [("code", "String"), ("language", "String")],
                "return_type": "String"
            }
        ]
        """
        
        code = """//! AIVA Rust FFI Library
//! Auto-generated FFI bindings

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

"""
        
        for func in functions:
            func_name = func["name"]
            params = func.get("params", [])
            return_type = func.get("return_type", "String")
            
            # 生成 FFI 函數
            code += f"""
#[no_mangle]
pub extern "C" fn {func_name}_ffi("""
            
            # 參數
            ffi_params = []
            for param_name, param_type in params:
                if param_type == "String":
                    ffi_params.append(f"{param_name}: *const c_char")
                elif param_type == "i32":
                    ffi_params.append(f"{param_name}: i32")
                elif param_type == "f64":
                    ffi_params.append(f"{param_name}: f64")
            
            code += ", ".join(ffi_params)
            
            # 回傳類型
            if return_type == "String":
                code += ") -> *mut c_char {\n"
            elif return_type == "i32":
                code += ") -> i32 {\n"
            else:
                code += ") -> *mut c_char {\n"
            
            # 函數主體
            code += f"    // Convert C strings to Rust strings\n"
            for param_name, param_type in params:
                if param_type == "String":
                    code += f"""    let {param_name}_str = unsafe {{
        CStr::from_ptr({param_name}).to_string_lossy().into_owned()
    }};
"""
            
            # 調用實際的 Rust 函數
            param_names = [name + ("_str" if ptype == "String" else "") 
                          for name, ptype in params]
            
            code += f"""    
    // Call the actual Rust function
    let result = {func_name}({", ".join(param_names)});
    
"""
            
            # 處理回傳值
            if return_type == "String":
                code += """    // Convert result to C string
    let c_string = CString::new(result).unwrap();
    c_string.into_raw()
"""
            else:
                code += "    result\n"
            
            code += "}\n\n"
        
        # 新增清理函數
        code += """
#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    unsafe {
        if s.is_null() { return }
        CString::from_raw(s)
    };
}
"""
        
        return code

class GoFFIBuilder:
    """Go FFI 函式庫建構器"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.logger = logging.getLogger("GoFFIBuilder")
    
    async def build_ffi_library(self, package_name: str = "main") -> Optional[str]:
        """建構 Go FFI 函式庫"""
        try:
            # Go 建構為共享函式庫
            cmd = [
                "go", "build", 
                "-buildmode=c-shared",
                "-o", "aiva_go_ffi.so" if platform.system() != "Windows" else "aiva_go_ffi.dll"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Go build failed: {stderr.decode()}")
                return None
            
            # 尋找生成的函式庫
            lib_file = "aiva_go_ffi.dll" if platform.system() == "Windows" else "aiva_go_ffi.so"
            lib_path = self.project_path / lib_file
            
            if lib_path.exists():
                self.logger.info(f"Go FFI library built: {lib_path}")
                return str(lib_path)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Go FFI build error: {e}")
            return None
    
    def generate_go_ffi_code(self, functions: List[Dict[str, Any]]) -> str:
        """生成 Go FFI 程式碼"""
        code = """// AIVA Go FFI Library
// Auto-generated FFI bindings

package main

import "C"
import (
    "encoding/json"
    "unsafe"
)

"""
        
        for func in functions:
            func_name = func["name"]
            params = func.get("params", [])
            return_type = func.get("return_type", "string")
            
            # 生成 export 函數
            code += f"""
//export {func_name}_ffi
func {func_name}_ffi("""
            
            # 參數
            ffi_params = []
            for param_name, param_type in params:
                if param_type == "string":
                    ffi_params.append(f"{param_name} *C.char")
                elif param_type == "int":
                    ffi_params.append(f"{param_name} C.int")
                elif param_type == "float64":
                    ffi_params.append(f"{param_name} C.double")
            
            code += ", ".join(ffi_params)
            
            if return_type == "string":
                code += ") *C.char {\n"
            elif return_type == "int":
                code += ") C.int {\n"
            else:
                code += ") *C.char {\n"
            
            # 轉換參數
            for param_name, param_type in params:
                if param_type == "string":
                    code += f"    {param_name}Go := C.GoString({param_name})\n"
                elif param_type == "int":
                    code += f"    {param_name}Go := int({param_name})\n"
                elif param_type == "float64":
                    code += f"    {param_name}Go := float64({param_name})\n"
            
            # 調用實際函數
            param_names = [name + "Go" for name, _ in params]
            code += f"\n    result := {func_name}({', '.join(param_names)})\n"
            
            # 回傳結果
            if return_type == "string":
                code += """    
    cResult := C.CString(result)
    return cResult
"""
            else:
                code += "    return C.int(result)\n"
            
            code += "}\n"
        
        code += "\nfunc main() {}\n"
        
        return code

class AIVAFFIManager:
    """AIVA FFI 整合管理器"""
    
    def __init__(self):
        self.libraries: Dict[str, FFILibrary] = {}
        self.rust_builder = None
        self.go_builder = None
        self.logger = logging.getLogger("AIVAFFIManager")
    
    async def build_and_load_rust_library(self, project_path: str, 
                                        functions: List[Dict[str, Any]],
                                        target_name: str = "aiva_rust_ffi") -> bool:
        """建構並載入 Rust FFI 函式庫"""
        try:
            self.rust_builder = RustFFIBuilder(project_path)
            
            # 生成 FFI 程式碼
            ffi_code = self.rust_builder.generate_rust_ffi_code(functions)
            
            # 寫入 lib.rs
            lib_rs_path = Path(project_path) / "src" / "lib.rs"
            with open(lib_rs_path, 'w') as f:
                f.write(ffi_code)
            
            self.logger.info("Rust FFI code generated")
            
            # 建構函式庫
            lib_path = await self.rust_builder.build_ffi_library(target_name)
            
            if not lib_path:
                return False
            
            # 載入函式庫
            library = FFILibrary(target_name, lib_path)
            if not library.load_library():
                return False
            
            # 定義函數
            for func in functions:
                func_name = func["name"] + "_ffi"
                # 簡化的簽名處理
                signature = "const char*, const char*"  # 根據實際需求調整
                return_type = "char*"
                
                library.define_function(func_name, signature, return_type)
            
            self.libraries[target_name] = library
            self.logger.info(f"Rust FFI library loaded: {target_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rust FFI setup failed: {e}")
            return False
    
    async def build_and_load_go_library(self, project_path: str,
                                      functions: List[Dict[str, Any]],
                                      target_name: str = "aiva_go_ffi") -> bool:
        """建構並載入 Go FFI 函式庫"""
        try:
            self.go_builder = GoFFIBuilder(project_path)
            
            # 生成 FFI 程式碼
            ffi_code = self.go_builder.generate_go_ffi_code(functions)
            
            # 寫入 main.go
            main_go_path = Path(project_path) / "main.go"
            with open(main_go_path, 'w') as f:
                f.write(ffi_code)
            
            self.logger.info("Go FFI code generated")
            
            # 建構函式庫
            lib_path = await self.go_builder.build_ffi_library()
            
            if not lib_path:
                return False
            
            # 載入函式庫
            library = FFILibrary(target_name, lib_path)
            if not library.load_library():
                return False
            
            # 定義函數
            for func in functions:
                func_name = func["name"] + "_ffi"
                signature = "const char*, const char*"  # 根據實際需求調整
                return_type = "char*"
                
                library.define_function(func_name, signature, return_type)
            
            self.libraries[target_name] = library
            self.logger.info(f"Go FFI library loaded: {target_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Go FFI setup failed: {e}")
            return False
    
    def call_function(self, library_name: str, function_name: str, *args) -> Any:
        """調用 FFI 函數"""
        if library_name not in self.libraries:
            raise ValueError(f"Library {library_name} not loaded")
        
        library = self.libraries[library_name]
        return library.call_function(function_name, *args)
    
    def list_libraries(self) -> List[str]:
        """列出已載入的函式庫"""
        return list(self.libraries.keys())
    
    def get_library_functions(self, library_name: str) -> List[str]:
        """獲取函式庫的函數列表"""
        if library_name not in self.libraries:
            return []
        
        return self.libraries[library_name].get_function_list()

# 使用範例
async def demo_ffi_integration():
    """示範 FFI 整合"""
    manager = AIVAFFIManager()
    
    try:
        # 定義要匯出的函數
        rust_functions = [
            {
                "name": "scan_vulnerabilities",
                "params": [("code", "String"), ("language", "String")],
                "return_type": "String"
            },
            {
                "name": "analyze_performance",
                "params": [("code", "String")],
                "return_type": "String"
            }
        ]
        
        go_functions = [
            {
                "name": "gather_system_info",
                "params": [("target", "string")],
                "return_type": "string"
            },
            {
                "name": "scan_network",
                "params": [("network", "string"), ("timeout", "int")],
                "return_type": "string"
            }
        ]
        
        # 建構並載入 Rust FFI 函式庫
        print("建構 Rust FFI 函式庫...")
        rust_success = await manager.build_and_load_rust_library(
            "C:/D/fold7/AIVA-git/services/features/function_sast_rust",
            rust_functions
        )
        
        if rust_success:
            print("✅ Rust FFI 函式庫載入成功")
            
            # 測試調用
            try:
                result = manager.call_function(
                    "aiva_rust_ffi", 
                    "scan_vulnerabilities_ffi",
                    b"fn main() { println!(\"Hello\"); }",
                    b"rust"
                )
                print(f"Rust 函數調用結果: {result}")
            except Exception as e:
                print(f"Rust 函數調用失敗: {e}")
        else:
            print("❌ Rust FFI 函式庫載入失敗")
        
        # 建構並載入 Go FFI 函式庫
        print("\n建構 Go FFI 函式庫...")
        go_success = await manager.build_and_load_go_library(
            "C:/D/fold7/AIVA-git/services/scan/info_gatherer_go",
            go_functions
        )
        
        if go_success:
            print("✅ Go FFI 函式庫載入成功")
            
            # 測試調用
            try:
                result = manager.call_function(
                    "aiva_go_ffi",
                    "gather_system_info_ffi",
                    b"localhost"
                )
                print(f"Go 函數調用結果: {result}")
            except Exception as e:
                print(f"Go 函數調用失敗: {e}")
        else:
            print("❌ Go FFI 函式庫載入失敗")
        
        # 列出所有載入的函式庫
        libraries = manager.list_libraries()
        print(f"\n已載入的 FFI 函式庫: {libraries}")
        
        for lib_name in libraries:
            functions = manager.get_library_functions(lib_name)
            print(f"{lib_name} 函數: {functions}")
        
    except Exception as e:
        print(f"FFI 示範過程發生錯誤: {e}")

if __name__ == "__main__":
    # 檢查 FFI 支援
    print("FFI 支援檢查:")
    print(f"ctypes: {'✅' if CTYPES_AVAILABLE else '❌'}")
    print(f"cffi: {'✅' if CFFI_AVAILABLE else '❌'}")
    
    if not CFFI_AVAILABLE:
        print("建議安裝 cffi: pip install cffi")
    
    # 執行示範
    asyncio.run(demo_ffi_integration())