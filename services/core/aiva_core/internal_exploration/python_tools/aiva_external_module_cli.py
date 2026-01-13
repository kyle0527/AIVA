#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA Multi-Language CLI Executor
多語言統一 CLI 執行器

支援 4 種語言的能力調用：
- Python: 動態導入與執行
- Rust: cargo run 調用
- Go: go run 調用
- TypeScript: npx ts-node 調用

使用方式:
    # 列出所有可用能力
    python multilang_cli_executor.py --list
    
    # 列出特定語言的能力
    python multilang_cli_executor.py --list --lang python
    python multilang_cli_executor.py --list --lang rust
    
    # 執行 Python 能力
    python multilang_cli_executor.py --lang python --flow 1
    
    # 執行 Rust 能力（帶參數）
    python multilang_cli_executor.py --lang rust --func analyze_cookies --cookies-json '[]' --url 'https://example.com'
    
    # 執行 Go 能力
    python multilang_cli_executor.py --lang go --func DialBroker --broker-url 'amqp://localhost'
    
    # 執行 TypeScript 能力
    python multilang_cli_executor.py --lang typescript --func analyzeClientSideAuthBypass --target 'https://example.com'
    
    # Dry run 模式（僅顯示命令，不執行）
    python multilang_cli_executor.py --lang rust --func analyze_cookies --dry-run
"""

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib

# ==========================================
# 設定與常數
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
INTEGRATION_DATA = PROJECT_ROOT / "services" / "integration" / "data" / "internal_exploration"

# 各語言分析結果位置
ANALYSIS_PATHS = {
    "python": PROJECT_ROOT / "features_classification" / "classification_data.json",
    "rust": INTEGRATION_DATA / "analysis_results" / "rust" / "analysis_results.json",
    "go": PROJECT_ROOT / "services" / "features" / "function_authn_go" / "analysis_output" / "analysis_results.json",
    "typescript": PROJECT_ROOT / "services" / "scan" / "typescript_engine" / "analysis_output" / "analysis_results.json",
}

# Rust 專案路徑
RUST_PROJECT_PATH = PROJECT_ROOT / "services" / "features" / "function_crypto" / "rust_core"


class MultiLangExecutor:
    """多語言統一執行器"""
    
    def __init__(self):
        self.capabilities = {
            "python": [],
            "rust": [],
            "go": [],
            "typescript": []
        }
        self.flows_info = {
            "python": {"flows": 0, "functions": 0},
            "rust": {"flows": 0, "functions": 0},
            "go": {"flows": 0, "functions": 0},
            "typescript": {"flows": 0, "functions": 0}
        }
        self._load_all_capabilities()
    
    def _load_all_capabilities(self):
        """載入所有語言的能力定義"""
        for lang, path in ANALYSIS_PATHS.items():
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 記錄 flows 和 functions 數量
                    flows_count = len(data.get("flows", []))
                    
                    if lang == "python":
                        self._load_python_capabilities(data)
                        functions_count = len(self.capabilities["python"])
                    else:
                        self._load_compiled_lang_capabilities(lang, data)
                        functions_count = len(data.get("functions", []))
                    
                    self.flows_info[lang] = {
                        "flows": flows_count,
                        "functions": functions_count
                    }
                    
                    print(f"[OK] 載入 {lang.upper()}: {functions_count} 個函數, {flows_count} 個數據流")
                except Exception as e:
                    print(f"[WARN] 載入 {lang} 能力失敗: {e}")
            else:
                print(f"[WARN] {lang} 分析結果不存在: {path}")
    
    def _load_python_capabilities(self, data: Dict):
        """載入 Python 能力（從 flows）"""
        if "flows" in data:
            for flow in data["flows"]:
                # 從 path 提取模組（如果 path 是 list）
                path_list = flow.get("full_path", [])
                if isinstance(path_list, list) and path_list:
                    # 提取模組名稱（通常在 path 中）
                    module_name = flow.get("primary_module", "unknown")
                else:
                    module_name = "unknown"
                
                self.capabilities["python"].append({
                    "id": flow.get("id"),
                    "name": flow.get("start", "unknown"),
                    "description": f"{flow.get('start')} → {flow.get('end')}",
                    "path": path_list,
                    "module": module_name,
                    "primary_component_type": flow.get("primary_component_type", ""),
                    "parameters": []  # Python flows 不直接包含參數定義
                })
    
    def _load_compiled_lang_capabilities(self, lang: str, data: Dict):
        """載入編譯型語言能力（Rust/Go/TypeScript）"""
        functions = data.get("functions", [])
        flows = data.get("flows", [])
        
        # 從 functions 提取能力
        for func in functions:
            capability = {
                "name": func.get("function_name"),
                "module": func.get("module"),
                "description": func.get("description", ""),
                "category": func.get("category", "other"),
                "source_file": func.get("source_file", ""),
                "inputs": func.get("inputs", []),
                "parameters": self._extract_lang_params(lang, func)
            }
            self.capabilities[lang].append(capability)
    
    def _extract_python_params(self, flow: Dict) -> List[Dict]:
        """提取 Python 能力的參數（從 flow 定義推斷）"""
        # 基於 CLI command 提取參數
        cli = flow.get("cli_command", "")
        params = []
        
        # 簡單解析（實際應該更複雜）
        if "--" in cli:
            parts = cli.split("--")
            for part in parts[1:]:
                param_name = part.split()[0] if part else ""
                if param_name:
                    params.append({
                        "name": param_name,
                        "required": False,
                        "type": "string"
                    })
        
        return params
    
    def _extract_lang_params(self, lang: str, func: Dict) -> List[Dict]:
        """提取編譯型語言的參數"""
        inputs = func.get("inputs", [])
        params = []
        
        for inp in inputs:
            params.append({
                "name": inp,
                "required": True,
                "type": "string"  # 簡化處理
            })
        
        return params
    
    def list_capabilities(self, lang: Optional[str] = None, category: Optional[str] = None):
        """列出可用能力"""
        languages = [lang] if lang else ["python", "rust", "go", "typescript"]
        
        for language in languages:
            caps = self.capabilities.get(language, [])
            
            if category:
                caps = [c for c in caps if c.get("category") == category]
            
            if not caps:
                continue
            
            print(f"\n{'='*60}")
            print(f"  {language.upper()} 能力 ({len(caps)} 個)")
            print(f"{'='*60}\n")
            
            for idx, cap in enumerate(caps, 1):
                print(f"{idx}. {cap.get('name')}")
                print(f"   模組: {cap.get('module', 'N/A')}")
                print(f"   分類: {cap.get('category', 'N/A')}")
                
                if cap.get('description'):
                    print(f"   說明: {cap['description']}")
                
                if cap.get('parameters'):
                    params = cap['parameters']
                    param_str = ", ".join([p['name'] for p in params])
                    print(f"   參數: {param_str}")
                
                print()
    
    def execute_python(self, flow_id: Optional[int] = None, func_name: Optional[str] = None, 
                      dry_run: bool = False, **kwargs):
        """執行 Python 能力"""
        if flow_id:
            caps = [c for c in self.capabilities["python"] if c.get("id") == flow_id]
        elif func_name:
            caps = [c for c in self.capabilities["python"] if c.get("name") == func_name]
        else:
            print("[錯誤] 請指定 --flow 或 --func")
            return
        
        if not caps:
            print(f"[錯誤] 找不到指定的 Python 能力")
            return
        
        cap = caps[0]
        cli_command = cap.get("cli_command", "")
        
        # 替換參數
        for key, value in kwargs.items():
            cli_command = cli_command.replace(f"--{key} <value>", f"--{key} {value}")
        
        print(f"\n[執行] Python 能力: {cap['name']}")
        print(f"[指令] {cli_command}\n")
        
        if dry_run:
            print("[Dry Run] 不實際執行")
            return
        
        # 實際執行
        try:
            result = subprocess.run(cli_command, shell=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            print(result.stdout)
            if result.stderr:
                print(f"[警告] 錯誤: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"[錯誤] 執行失敗: {e}")
            return False
    
    def execute_rust(self, func_name: str, dry_run: bool = False, **kwargs):
        """執行 Rust 能力"""
        caps = [c for c in self.capabilities["rust"] if c.get("name") == func_name]
        
        if not caps:
            print(f"[錯誤] 找不到 Rust 函數: {func_name}")
            return
        
        cap = caps[0]
        
        # 根據函數構建 cargo 命令
        if func_name in ["scan_javascript", "analyze_tls", "analyze_cookies", "analyze_headers"]:
            # 這些是 main.rs 中的子命令
            cmd_map = {
                "scan_javascript": "scan-js",
                "analyze_tls": "analyze-tls",
                "analyze_cookies": "analyze-cookies",
                "analyze_headers": "analyze-headers"
            }
            
            subcommand = cmd_map.get(func_name, "")
            cmd = f"cargo run --manifest-path {RUST_PROJECT_PATH}/Cargo.toml -- {subcommand}"
            
            # 添加參數
            for param in cap.get('parameters', []):
                param_name = param['name'].replace('_', '-')
                value = kwargs.get(param['name'], kwargs.get(param_name, ''))
                if value:
                    cmd += f" --{param_name} \"{value}\""
        else:
            # analyzer 的輔助函數
            source_file = cap.get('source_file', '')
            cmd = f"cargo run --manifest-path {RUST_PROJECT_PATH}/Cargo.toml -- --file {source_file} --func {func_name}"
        
        print(f"\n[執行] Rust 能力: {func_name}")
        print(f"[指令] {cmd}\n")
        
        if dry_run:
            print("[Dry Run] 不實際執行")
            return
        
        # 實際執行
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                # Rust 的 warning 也會在 stderr，但不是錯誤
                if "error" in result.stderr.lower():
                    print(f"[錯誤] {result.stderr}")
                else:
                    print(f"[警告] {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"[錯誤] 執行失敗: {e}")
            return False
    
    def execute_go(self, func_name: str, dry_run: bool = False, **kwargs):
        """執行 Go 能力"""
        caps = [c for c in self.capabilities["go"] if c.get("name") == func_name]
        
        if not caps:
            print(f"[錯誤] 找不到 Go 函數: {func_name}")
            return
        
        cap = caps[0]
        source_file = cap.get('source_file', '')
        
        # 構建 go run 命令
        cmd = f"go run {source_file}"
        
        # 添加參數（Go 通常用 flags）
        for param in cap.get('parameters', []):
            param_name = param['name']
            value = kwargs.get(param_name, '')
            if value:
                cmd += f" -{param_name}=\"{value}\""
        
        print(f"\n[執行] Go 能力: {func_name}")
        print(f"[指令] {cmd}\n")
        
        if dry_run:
            print("[Dry Run] 不實際執行")
            return
        
        # 實際執行
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"[警告] {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"[錯誤] 執行失敗: {e}")
            return False
    
    def execute_typescript(self, func_name: str, dry_run: bool = False, **kwargs):
        """執行 TypeScript 能力"""
        caps = [c for c in self.capabilities["typescript"] if c.get("name") == func_name]
        
        if not caps:
            print(f"[錯誤] 找不到 TypeScript 函數: {func_name}")
            return
        
        cap = caps[0]
        source_file = cap.get('source_file', '')
        
        # 構建 npx ts-node 命令
        ts_engine_path = PROJECT_ROOT / "services" / "scan" / "typescript_engine"
        cmd = f"npx ts-node {source_file} --func {func_name}"
        
        # 添加參數
        for param in cap.get('parameters', []):
            param_name = param['name']
            value = kwargs.get(param_name, '')
            if value:
                cmd += f" --{param_name}=\"{value}\""
        
        print(f"\n[執行] TypeScript 能力: {func_name}")
        print(f"[指令] {cmd}\n")
        
        if dry_run:
            print("[Dry Run] 不實際執行")
            return
        
        # 實際執行
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ts_engine_path))
            print(result.stdout)
            if result.stderr:
                print(f"[警告] {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"[錯誤] 執行失敗: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="AIVA Multi-Language CLI Executor - 多語言統一CLI執行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 列出所有能力
  python multilang_cli_executor.py --list
  
  # 列出 Rust 能力
  python multilang_cli_executor.py --list --lang rust
  
  # 執行 Rust Cookie 分析（dry run）
  python multilang_cli_executor.py --lang rust --func analyze_cookies \\
    --cookies-json '["sessionid=abc"]' --url 'https://example.com' --dry-run
  
  # 實際執行 Rust Cookie 分析
  python multilang_cli_executor.py --lang rust --func analyze_cookies \\
    --cookies-json '["sessionid=abc"]' --url 'https://example.com'
  
  # 執行 Python 流程
  python multilang_cli_executor.py --lang python --flow 1
  
  # 執行 Go 函數
  python multilang_cli_executor.py --lang go --func DialBroker --broker-url 'amqp://localhost'
        """
    )
    
    # 主要操作
    parser.add_argument("--list", action="store_true", help="列出所有可用能力")
    parser.add_argument("--lang", choices=["python", "rust", "go", "typescript"], help="指定語言")
    parser.add_argument("--category", help="篩選特定分類 (analysis, reconnaissance, exploitation, etc.)")
    
    # 執行選項
    parser.add_argument("--flow", type=int, help="Python flow ID")
    parser.add_argument("--func", help="函數名稱")
    parser.add_argument("--dry-run", action="store_true", help="Dry run 模式（僅顯示命令）")
    
    # 動態參數（用於不同能力）
    parser.add_argument("--target", help="目標 URL")
    parser.add_argument("--cookies-json", help="Cookies JSON 字串")
    parser.add_argument("--headers-json", help="Headers JSON 字串")
    parser.add_argument("--url", help="URL")
    parser.add_argument("--content", help="內容（用於 JS 分析）")
    parser.add_argument("--port", type=int, default=443, help="端口（用於 TLS 分析）")
    parser.add_argument("--broker-url", help="Message broker URL")
    
    args, unknown = parser.parse_known_args()
    
    # 解析未知參數（支援動態參數）
    kwargs = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:].replace("-", "_")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                kwargs[key] = unknown[i + 1]
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    
    # 合併已知參數到 kwargs
    for key, value in vars(args).items():
        if value is not None and key not in ["list", "lang", "category", "flow", "func", "dry_run"]:
            kwargs[key] = value
    
    executor = MultiLangExecutor()
    
    if args.list:
        executor.list_capabilities(lang=args.lang, category=args.category)
        return
    
    if not args.lang:
        print("[錯誤] 請指定語言 --lang [python|rust|go|typescript]")
        parser.print_help()
        return
    
    # 執行能力
    if args.lang == "python":
        executor.execute_python(flow_id=args.flow, func_name=args.func, dry_run=args.dry_run, **kwargs)
    elif args.lang == "rust":
        if not args.func:
            print("[錯誤] Rust 需要指定函數名稱 --func")
            return
        executor.execute_rust(func_name=args.func, dry_run=args.dry_run, **kwargs)
    elif args.lang == "go":
        if not args.func:
            print("[錯誤] Go 需要指定函數名稱 --func")
            return
        executor.execute_go(func_name=args.func, dry_run=args.dry_run, **kwargs)
    elif args.lang == "typescript":
        if not args.func:
            print("[錯誤] TypeScript 需要指定函數名稱 --func")
            return
        executor.execute_typescript(func_name=args.func, dry_run=args.dry_run, **kwargs)


if __name__ == "__main__":
    main()
