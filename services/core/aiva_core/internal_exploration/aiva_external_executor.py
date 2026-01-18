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
from collections import defaultdict
from datetime import datetime
import importlib

# ==========================================
# 設定與常數
# ==========================================

# PROJECT_ROOT: internal_exploration → aiva_core → core → services → AIVA-git
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# 各語言分析結果位置 - 統一從 classification_data.json 讀取
CLASSIFICATION_DATA = PROJECT_ROOT / "features_classification" / "classification_data.json"

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
        """載入所有語言的能力定義（從統一的 classification_data.json）"""
        if not CLASSIFICATION_DATA.exists():
            print(f"[ERROR] 找不到 classification_data.json: {CLASSIFICATION_DATA}")
            print(f"[提示] 請先執行: python aiva_external_classifier.py")
            return
        
        try:
            with open(CLASSIFICATION_DATA, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 從 metadata 獲取統計
            metadata = data.get("metadata", {})
            stats = metadata.get("statistics", {})
            by_language = stats.get("by_language", {})
            
            # 從 flows 中按語言分組
            all_flows = data.get("flows", [])
            flows_by_lang = defaultdict(list)
            
            for flow in all_flows:
                lang = flow.get("language", "Python").lower()
                flows_by_lang[lang].append(flow)
            
            # 載入各語言的能力
            for lang in ["python", "rust", "go", "typescript"]:
                flows = flows_by_lang.get(lang, [])
                if flows:
                    self.capabilities[lang] = flows
                    self.flows_info[lang] = {
                        "flows": len(flows),
                        "functions": len(flows)  # 每個 flow 代表一個功能
                    }
                    print(f"[OK] 載入 {lang.upper()}: {len(flows)} 個流程")
            
            print(f"\n[OK] 成功載入統一分類數據")
            print(f"     總流程: {metadata.get('total_flows', 0)}")
            print(f"     總模組: {metadata.get('total_modules', 0)}")
            print(f"     語言: {', '.join(metadata.get('languages', []))}")
            
        except Exception as e:
            print(f"[ERROR] 載入分類數據失敗: {e}")
            import traceback
            traceback.print_exc()
    
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
                caps = [c for c in caps if c.get("module_type") == category]
            
            if not caps:
                continue
            
            print(f"\n{'='*60}")
            print(f"  {language.upper()} 能力 ({len(caps)} 個)")
            print(f"{'='*60}\n")
            
            # 按模組分組顯示
            by_module = defaultdict(list)
            for cap in caps:
                by_module[cap.get('module', 'unknown')].append(cap)
            
            for module_name, module_caps in sorted(by_module.items()):
                module_type = module_caps[0].get('module_type', 'unknown')
                module_category = module_caps[0].get('module_category', 'unknown')
                module_desc = module_caps[0].get('module_description', module_name)
                use_case = module_caps[0].get('use_case', '')
                
                print(f"[{module_name}]")
                print(f"  類型: {module_type} / {module_category}")
                print(f"  說明: {module_desc}")
                if use_case:
                    print(f"  用途: {use_case}")
                print(f"  流程數: {len(module_caps)}\n")
                
                # 顯示前 5 個流程示例
                for idx, cap in enumerate(module_caps[:5], 1):
                    flow_id = cap.get('id', '?')
                    start = cap.get('start', 'unknown')
                    end = cap.get('end', 'unknown')
                    length = cap.get('length', 0)
                    
                    # 清理函數名（移除類別前綴）
                    if '.' in start:
                        start_clean = start.split('.')[-1]
                    else:
                        start_clean = start
                    
                    if '.' in end:
                        end_clean = end.split('.')[-1]
                    else:
                        end_clean = end
                    
                    print(f"  {flow_id:3d}. {start_clean} → {end_clean} (長度: {length})")
                
                if len(module_caps) > 5:
                    print(f"  ... 還有 {len(module_caps) - 5} 個流程")
                
                print()
    
    def execute_python(self, flow_id: Optional[int] = None, func_name: Optional[str] = None, 
                      dry_run: bool = False, **kwargs):
        """執行 Python 能力"""
        if flow_id:
            caps = [c for c in self.capabilities["python"] if c.get("id") == flow_id]
        elif func_name:
            caps = [c for c in self.capabilities["python"] if c.get("start") == func_name or c.get("end") == func_name]
        else:
            print("[錯誤] 請指定 --flow 或 --func")
            return
        
        if not caps:
            print(f"[錯誤] 找不到指定的 Python 能力 (flow_id={flow_id})")
            return
        
        cap = caps[0]
        
        # 顯示流程信息
        print(f"\n{'='*60}")
        print(f"  執行 Python 流程 #{cap['id']}")
        print(f"{'='*60}\n")
        
        print(f"[模組] {cap['module']}")
        print(f"[類型] {cap['module_type']} / {cap['module_category']}")
        print(f"[描述] {cap.get('module_description', cap['module'])}")
        print(f"[用途] {cap.get('use_case', '通用安全檢測')}")
        print(f"\n[流程] {cap['start']} → {cap['end']}")
        print(f"[路徑] {' → '.join(cap['path'])}")
        print(f"[長度] {cap['length']} 步驟")
        
        # 構建執行命令（需要從實際模組導入並執行）
        module_name = cap['module']
        start_func = cap['start']
        target = kwargs.get('target', 'http://localhost:3000')
        
        print(f"\n[目標] {target}")
        
        if dry_run:
            print(f"\n{'='*60}")
            print("  Dry Run 模式 - 不實際執行")
            print(f"{'='*60}\n")
            print(f"[說明] 這個流程將會：")
            print(f"  1. 導入模組: services/features/{module_name}")
            print(f"  2. 執行入口: {start_func}")
            print(f"  3. 測試目標: {target}")
            print(f"\n[建議] 何時使用此流程：")
            use_case = cap.get('use_case', '')
            if use_case:
                print(f"  {use_case}")
            return
        
        # 實際執行需要動態導入模組
        print(f"\n{'='*60}")
        print("  開始執行")
        print(f"{'='*60}\n")
        print("[提示] 完整的執行邏輯待實現（需要動態導入和實例化）")
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
    
    def generate_reference_docs(self, output_format: str = "md", output_dir: Optional[str] = None):
        """生成外部模組 CLI 指令參考文件
        
        支援兩種格式:
        - Markdown (.md): 適合人類閱讀
        - JSON (.json): 適合 AI 檢索
        
        Args:
            output_format: 輸出格式 'md' 或 'json'
            output_dir: 輸出目錄，預設為 features_classification
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "features_classification"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_format == "md":
            self._generate_markdown_reference(output_dir)
        elif output_format == "json":
            self._generate_json_database(output_dir)
    
    def _generate_markdown_reference(self, output_dir: Path):
        """生成 Markdown 參考手冊"""
        filename = output_dir / "EXTERNAL_CLI_COMMANDS_REFERENCE.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 標題和元數據
                f.write("# AIVA 外部模組 CLI 指令參考手冊\n\n")
                f.write(f"> 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"> 資料來源: classification_data.json\n")
                
                # 統計總覽
                total_flows = sum(len(caps) for caps in self.capabilities.values())
                f.write(f"> 總流程數: {total_flows}\n\n")
                
                f.write("## 模組總覽\n\n")
                f.write("| 語言 | 流程數 | 模組數 |\n")
                f.write("|------|--------|--------|\n")
                
                for lang, caps in self.capabilities.items():
                    if caps:
                        modules = set(c.get('module', '') for c in caps)
                        f.write(f"| {lang.upper()} | {len(caps)} | {len(modules)} |\n")
                
                f.write("\n---\n\n")
                
                # 按語言分組的流程列表
                for lang in ["python", "rust", "go", "typescript"]:
                    caps = self.capabilities.get(lang, [])
                    if not caps:
                        continue
                    
                    f.write(f"## {lang.upper()} 流程\n\n")
                    
                    # 按模組分組
                    by_module = defaultdict(list)
                    for cap in caps:
                        by_module[cap.get('module', 'unknown')].append(cap)
                    
                    for module_name in sorted(by_module.keys()):
                        module_caps = by_module[module_name]
                        module_type = module_caps[0].get('module_type', '')
                        module_desc = module_caps[0].get('module_description', '')
                        use_case = module_caps[0].get('use_case', '')
                        
                        f.write(f"### {module_name}\n\n")
                        f.write(f"**類型**: {module_type}\n\n")
                        f.write(f"**描述**: {module_desc}\n\n")
                        if use_case:
                            f.write(f"**用途**: {use_case}\n\n")
                        
                        # 流程表格
                        f.write("| Flow ID | 流程路徑 | 長度 | CLI 指令 |\n")
                        f.write("|---------|----------|------|----------|\n")
                        
                        for cap in module_caps[:10]:  # 只顯示前10個
                            flow_id = cap.get('id', '?')
                            path = ' → '.join([p.split('.')[-1] for p in cap.get('path', [])])
                            length = cap.get('length', 0)
                            
                            if lang == "python":
                                cli = f"`python aiva_external_executor.py --lang python --flow {flow_id} --target <URL>`"
                            else:
                                func_name = cap.get('start', '')
                                cli = f"`python aiva_external_executor.py --lang {lang} --func {func_name}`"
                            
                            f.write(f"| {flow_id} | {path} | {length} | {cli} |\n")
                        
                        if len(module_caps) > 10:
                            f.write(f"\n*... 還有 {len(module_caps) - 10} 個流程*\n")
                        
                        f.write("\n")
                    
                    f.write("---\n\n")
                
                # 使用指南
                f.write("## 使用指南\n\n")
                f.write("### 基本用法\n\n")
                f.write("```bash\n")
                f.write("# 列出所有可用能力\n")
                f.write("python aiva_external_executor.py --list\n\n")
                f.write("# 列出特定語言\n")
                f.write("python aiva_external_executor.py --list --lang python\n\n")
                f.write("# 執行 Python 流程（dry-run）\n")
                f.write("python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run\n\n")
                f.write("# 實際執行\n")
                f.write("python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000\n")
                f.write("```\n\n")
                
                f.write("### 按攻擊類型分類\n\n")
                attack_types = {}
                for lang, caps in self.capabilities.items():
                    for cap in caps:
                        attack_type = cap.get('module_type', 'unknown')
                        if attack_type not in attack_types:
                            attack_types[attack_type] = []
                        attack_types[attack_type].append(cap)
                
                for attack_type, caps in sorted(attack_types.items()):
                    modules = set(c.get('module', '') for c in caps)
                    f.write(f"- **{attack_type}**: {len(caps)} 個流程，涵蓋 {', '.join(sorted(modules))}\n")
            
            print(f"[成功] Markdown 參考文件已生成: {filename}")
            
        except IOError as e:
            print(f"[錯誤] 無法寫入文件: {e}")
    
    def _generate_json_database(self, output_dir: Path):
        """生成 JSON 資料庫"""
        filename = output_dir / "external_cli_commands_db.json"
        
        db_records = []
        
        for lang, caps in self.capabilities.items():
            for cap in caps:
                flow_id = cap.get('id')
                module_name = cap.get('module', 'unknown')
                start = cap.get('start', '')
                end = cap.get('end', '')
                
                if lang == "python":
                    cli_command = f"python aiva_external_executor.py --lang python --flow {flow_id} --target <URL>"
                else:
                    cli_command = f"python aiva_external_executor.py --lang {lang} --func {start}"
                
                record = {
                    "flow_id": flow_id,
                    "language": lang,
                    "module": module_name,
                    "module_type": cap.get('module_type', ''),
                    "module_category": cap.get('module_category', ''),
                    "description": cap.get('module_description', ''),
                    "use_case": cap.get('use_case', ''),
                    "flow_path": cap.get('path', []),
                    "length": cap.get('length', 0),
                    "cli_command": cli_command,
                    "start_function": start,
                    "end_function": end
                }
                db_records.append(record)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(db_records, f, indent=2, ensure_ascii=False)
            print(f"[成功] JSON 資料庫已生成: {filename}")
        except IOError as e:
            print(f"[錯誤] 無法寫入文件: {e}")


class InteractiveMenu:
    """
    互動式選單介面（外部模組版本）
    
    提供三層聚合顯示:
    1. 第一層: 按模組分類
    2. 第二層: 按「起點->終點」聚合能力（相同能力的不同實現）
    3. 第三層: 顯示具體中間路徑變體
    """
    
    def __init__(self, executor: MultiLangExecutor):
        self.executor = executor
        self.capabilities = executor.capabilities
        self._build_indexes()
    
    def _build_indexes(self):
        """建立雙層索引: [Language][Module][(Start, End)] -> [Flows]"""
        from collections import defaultdict
        self.lang_module_groups = {}
        
        for lang, caps in self.capabilities.items():
            if not caps:
                continue
            
            module_groups = defaultdict(lambda: defaultdict(list))
            for flow in caps:
                module = flow.get('module', 'unknown')
                start = flow.get('start', '?')
                end = flow.get('end', '?')
                module_groups[module][(start, end)].append(flow)
            
            self.lang_module_groups[lang] = module_groups
    
    def clear(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run(self):
        """主選單循環"""
        while True:
            self.clear()
            total = sum(len(caps) for caps in self.capabilities.values())
            
            print("=" * 70)
            print(f" AIVA 外部模組 - 能力控制台 (Flows: {total})")
            print("=" * 70)
            
            # 按語言分類顯示
            langs = [l for l, c in self.capabilities.items() if c]
            
            for i, lang in enumerate(langs):
                caps = self.capabilities[lang]
                modules = self.lang_module_groups.get(lang, {})
                module_count = len(modules)
                
                print(f" [{i+1}] {lang.upper():<15} | {len(caps)} 個流程 ({module_count} 個模組)")
            
            print(" [0] 離開")
            choice = input("\n> 請選擇語言: ").strip()
            
            if choice == "0":
                break
            if choice.isdigit() and 0 < int(choice) <= len(langs):
                self._show_module_menu(langs[int(choice)-1])
    
    def _show_module_menu(self, language: str):
        """顯示模組列表"""
        module_groups = self.lang_module_groups.get(language, {})
        sorted_modules = sorted(module_groups.keys())
        
        while True:
            self.clear()
            print(f"\n=== {language.upper()} 模組列表 ===")
            print(f"{'No.':<4} | {'模組名稱':<30} | {'能力數'} | {'總流程'}")
            print("-" * 80)
            
            for i, module in enumerate(sorted_modules):
                caps = module_groups[module]
                unique_caps = len(caps)
                total_flows = sum(len(v) for v in caps.values())
                
                # 獲取模組描述
                first_flow = next(iter(caps.values()))[0]
                desc = first_flow.get('module_description', module)
                
                print(f" [{i+1:<2}] | {module:<30} | {unique_caps:<7} | {total_flows}")
                print(f"       {desc}")
            
            print("-" * 80)
            print(" [b] 返回  [輸入數字] 選擇模組")
            
            choice = input("\n> ").strip()
            if choice == 'b':
                break
            
            if choice.isdigit() and 0 < int(choice) <= len(sorted_modules):
                module_name = sorted_modules[int(choice)-1]
                self._show_capability_menu(language, module_name, module_groups[module_name])
    
    def _show_capability_menu(self, language: str, module_name: str, caps: dict):
        """顯示能力的聚合列表 (起點->終點)"""
        sorted_caps = sorted(caps.keys())
        
        while True:
            self.clear()
            # 獲取模組資訊
            first_flow = next(iter(caps.values()))[0]
            module_desc = first_flow.get('module_description', module_name)
            module_type = first_flow.get('module_type', '')
            
            print(f"\n=== {module_desc} ({module_name}) ===")
            print(f"類型: {module_type} | 語言: {language.upper()}")
            print()
            print(f"{'No.':<4} | {'起點 (Start)':<30} -> {'終點 (End)':<30} | {'變體'}")
            print("-" * 100)
            
            for i, (start, end) in enumerate(sorted_caps):
                count = len(caps[(start, end)])
                # 縮短顯示名稱
                start_short = start.split('.')[-1] if '.' in start else start
                end_short = end.split('.')[-1] if '.' in end else end
                
                print(f" [{i+1:<2}] | {start_short:<30} -> {end_short:<30} | {count}")
            
            print("-" * 100)
            print(" [b] 返回  [輸入數字] 選擇能力")
            
            choice = input("\n> ").strip()
            if choice == 'b':
                break
            
            if choice.isdigit() and 0 < int(choice) <= len(sorted_caps):
                key = sorted_caps[int(choice)-1]
                self._show_variant_menu(language, key, caps[key])
    
    def _show_variant_menu(self, language: str, cap_key: tuple, flows: list):
        """顯示具體路徑變體"""
        while True:
            self.clear()
            start_short = cap_key[0].split('.')[-1] if '.' in cap_key[0] else cap_key[0]
            end_short = cap_key[1].split('.')[-1] if '.' in cap_key[1] else cap_key[1]
            
            print(f"\n=== 執行路徑: {start_short} -> {end_short} ({language.upper()}) ===")
            
            # 顯示用途說明
            if flows:
                use_case = flows[0].get('use_case', '')
                if use_case:
                    print(f"用途: {use_case}")
            
            print("-" * 100)
            
            # 按長度排序
            flows.sort(key=lambda x: x['length'])
            
            for i, flow in enumerate(flows):
                flow_id = flow.get('id', '?')
                length = flow.get('length', 0)
                
                # 顯示中間路徑
                path = flow.get('path', [])
                if len(path) > 2:
                    mid = [p.split('.')[-1] for p in path[1:-1]]
                    mid_str = " -> ".join(mid)
                    print(f" [{i+1}] Flow {flow_id} (長度:{length}): ... -> {mid_str} -> ...")
                else:
                    print(f" [{i+1}] Flow {flow_id} (長度:{length}): 直連")
            
            print("-" * 100)
            print(" [b] 返回  [輸入數字] 執行  [v 數字] 預覽 (dry-run)")
            
            cmd = input("\n> ").strip()
            if cmd == 'b':
                break
            
            is_view = False
            idx = -1
            
            if cmd.startswith('v '):
                is_view = True
                try:
                    idx = int(cmd.split()[1]) - 1
                except (ValueError, IndexError):
                    pass
            elif cmd.isdigit():
                idx = int(cmd) - 1
            
            if 0 <= idx < len(flows):
                target = flows[idx]
                flow_id = target.get('id')
                
                print("\n" + "=" * 50)
                if language == "python":
                    self.executor.execute_python(flow_id=flow_id, dry_run=is_view)
                elif language == "rust":
                    func_name = target.get('start', '')
                    self.executor.execute_rust(func_name=func_name, dry_run=is_view)
                elif language == "go":
                    func_name = target.get('start', '')
                    self.executor.execute_go(func_name=func_name, dry_run=is_view)
                elif language == "typescript":
                    func_name = target.get('start', '')
                    self.executor.execute_typescript(func_name=func_name, dry_run=is_view)
                
                input("\n[按 Enter 繼續...]")


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
    parser.add_argument("--menu", action="store_true", help="啟動互動式選單（按模組和能力瀏覽）")
    parser.add_argument("--generate-doc", choices=["md", "json"], help="生成參考文件 (md=Markdown, json=JSON資料庫)")
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
    
    # 如果沒有任何參數，預設啟動互動選單
    if not any([args.flow, args.func, args.generate_doc, args.list, args.menu, args.lang]):
        InteractiveMenu(executor).run()
        return
    
    # 啟動互動選單
    if args.menu:
        InteractiveMenu(executor).run()
        return
    
    # 生成文件
    if args.generate_doc:
        executor.generate_reference_docs(output_format=args.generate_doc)
        return
    
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
