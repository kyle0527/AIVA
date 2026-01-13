#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA External Module Classifier v1.0
====================================
Classifier for external function modules (features/scan)

Target modules:
- services/features/function_*
- services/scan/*

Features:
- Classify by function modules (function_sqli, function_xss, etc.)
- Multi-language detection (Python/Go/Rust/TypeScript)
- Entry point identification
- CLI command generation
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

SECTION_SEPARATOR = "---\n\n"


class ExternalModuleClassifier:
    """External module classifier for features/scan modules
    
    [核心差異] 從數據中動態提取模組信息，而非預先定義
    
    與內部分類器的差異：
    - 內部（aiva_flow_classifier.py）：AI Core 5大模組架構
      cognitive_core, internal_exploration, task_planning, 
      core_capabilities, service_backbone
    
    - 外部（本檔案）：從數據路徑中動態提取
      功能模組: services/features/.../function_*
      掃描引擎: services/scan/*
    
    分類維度：
    1. 模組歸屬（從路徑提取：function_sqli, scan_engine, etc.）
    2. 程式語言（Python/Go/Rust/TypeScript）
    3. 入口點識別（main, handler, executor, etc.）
    4. 攻擊類型（從路徑推斷：SQL注入、XSS、SSRF等）
    """
    
    # [不再使用] 改為從數據動態提取
    # 攻擊類型推斷規則（從模組名稱推斷）
    ATTACK_TYPE_INFERENCE = {
        "sqli": {"type": "injection", "category": "database_security", "name": "SQL 注入檢測"},
        "xss": {"type": "injection", "category": "web_security", "name": "XSS 漏洞檢測"},
        "ssrf": {"type": "ssrf", "category": "network_security", "name": "SSRF 漏洞檢測"},
        "idor": {"type": "access_control", "category": "authorization", "name": "IDOR 漏洞檢測"},
        "info_leak": {"type": "information_disclosure", "category": "data_protection", "name": "信息洩露檢測"},
        "bizlogic": {"type": "business_logic", "category": "logic_flaw", "name": "業務邏輯漏洞"},
        "crypto": {"type": "cryptographic", "category": "encryption", "name": "加密相關漏洞"},
        "authn": {"type": "authentication", "category": "identity", "name": "身份驗證"},
        "authz": {"type": "authorization", "category": "access_control", "name": "權限控制"}
    }
    
    # 掃描引擎類型推斷
    SCAN_TYPE_INFERENCE = {
        "scan_engine": {"type": "engine", "category": "scanning", "name": "掃描引擎"},
        "scan_orchestrator": {"type": "orchestrator", "category": "coordination", "name": "掃描編排器"
        },
        "typescript_engine": {
            "name": "TypeScript 掃描引擎",
            "component_type": "engine",
            "category": "language_specific"
        }
    }
    
    # 多語言檢測
    LANGUAGE_EXTENSIONS = {
        ".py": "Python",
        ".go": "Go",
        ".rs": "Rust",
        ".ts": "TypeScript",
        ".js": "JavaScript"
    }
    
    # 入口點關鍵字
    ENTRY_POINT_KEYWORDS = [
        "main", "handler", "executor", "scanner", "detector",
        "checker", "validator", "analyzer", "engine"
    ]
    
    def __init__(self, input_dir: str, output_dir: str, verbose: bool = False):
        """初始化分類器
        
        Args:
            input_dir: 包含 analysis_results.json 的目錄
            output_dir: 輸出目錄
            verbose: 是否顯示詳細訊息
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.flows_data = None
        self.classified_flows = []
        self.module_stats = defaultdict(int)
        self.language_stats = defaultdict(int)
        
        # 初始化模組字典（將從數據動態生成）
        self.FUNCTION_MODULES: Dict[str, Dict[str, str]] = {}
        self.SCAN_MODULES: Dict[str, Dict[str, str]] = {}
        
    def load_flow_data(self):
        """載入 AST 分析結果
        
        支援兩種格式：
        - 'flows': 已轉換的流程對象列表（來自 classifier）
        - 'flow_chains': 原始流程鏈列表（來自 analyzer）
        """
        input_file = self.input_dir / "analysis_results.json"
        
        if not input_file.exists():
            raise FileNotFoundError(f"找不到分析結果: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            self.flows_data = json.load(f)
        
        # 檢查是否有 flow_chains（analyzer 直接輸出）
        flow_chains = self.flows_data.get('flow_chains', [])
        flows = self.flows_data.get('flows', [])
        
        # 如果有 flow_chains 但沒有 flows，需要轉換
        if flow_chains and not flows:
            print(f"   檢測到 {len(flow_chains)} 個 flow_chains，轉換為 flows 格式...")
            flows = self._convert_flow_chains_to_flows(flow_chains)
            self.flows_data['flows'] = flows
        
        # 從數據中動態提取模組信息
        self._extract_modules_from_data(flows)
        
        print(f"✅ 載入 {len(flows)} 個數據流")
        print(f"✅ 提取到 {len(self.FUNCTION_MODULES)} 個功能模組, {len(self.SCAN_MODULES)} 個掃描模組")
    
    def _convert_flow_chains_to_flows(self, flow_chains: List[List[str]]) -> List[Dict[str, Any]]:
        """將 flow_chains 轉換為 flows 格式
        
        flow_chains: [['func1', 'func2'], ['func3', 'func4'], ...]
        flows: [{id: 1, path: ['func1', 'func2'], file_path: '...', ...}, ...]
        """
        flows = []
        
        # 安全檢查 flows_data
        if not self.flows_data:
            return flows
            
        function_details = self.flows_data.get('function_details', {})
        function_map = function_details.get('function_map', {})
        
        for idx, chain in enumerate(flow_chains, 1):
            if not chain:
                continue
            
            # 從第一個函數獲取文件路徑
            first_func = chain[0]
            func_info = function_map.get(first_func, {})
            file_path = func_info.get('file_path', '')
            
            flow = {
                'id': idx,
                'path': chain,  # 函數名稱列表
                'file_path': file_path,
                'length': len(chain),
                'func_names': chain
            }
            
            flows.append(flow)
        
        return flows
    
    def _extract_modules_from_data(self, flows: List[Dict[str, Any]]):
        """從數據中動態提取模組信息
        
        根據文件路徑識別模組名稱並生成模組字典
        例如: services/features/.../function_sqli/ → function_sqli
              services/scan/typescript_engine/ → typescript_engine
        """
        import re
        
        # 收集所有唯一的模組名稱
        module_paths = set()
        
        for flow in flows:
            file_path = flow.get('file_path', '')
            
            # 提取 function_* 模組
            function_match = re.search(r'function_(\w+)', file_path)
            if function_match:
                module_name = f"function_{function_match.group(1)}"
                module_paths.add(('function', module_name))
            
            # 提取 scan/* 模組
            scan_match = re.search(r'/scan/(\w+)', file_path)
            if scan_match:
                module_name = scan_match.group(1)
                module_paths.add(('scan', module_name))
        
        # 生成 FUNCTION_MODULES
        for module_type, module_name in module_paths:
            if module_type == 'function':
                # 從模組名稱推斷攻擊類型
                attack_key = module_name.replace('function_', '')
                attack_info = self.ATTACK_TYPE_INFERENCE.get(attack_key, {
                    "type": "unknown",
                    "category": "security",
                    "name": f"{module_name} 檢測"
                })
                
                self.FUNCTION_MODULES[module_name] = {
                    "name": attack_info.get("name", f"{module_name} 檢測"),
                    "attack_type": attack_info.get("type", "unknown"),
                    "category": attack_info.get("category", "security")
                }
            
            elif module_type == 'scan':
                # 掃描模組信息
                scan_info = self.SCAN_TYPE_INFERENCE.get(module_name, {
                    "name": f"{module_name} 掃描引擎",
                    "component_type": "engine",
                    "category": "scanning"
                })
                
                self.SCAN_MODULES[module_name] = {
                    "name": scan_info.get("name", f"{module_name} 掃描引擎"),
                    "component_type": scan_info.get("component_type", "engine"),
                    "category": scan_info.get("category", "scanning")
                }
    
    def classify_flows(self):
        """分類所有數據流"""
        # 安全檢查 flows_data
        if not self.flows_data:
            logger.warning("[警告] flows_data 為 None，無法分類")
            return
            
        flows = self.flows_data.get('flows', [])
        
        # 新增：分類統計
        category_stats = defaultdict(int)  # features vs scan
        attack_type_stats = defaultdict(int)  # injection, ssrf, access_control等
        
        for flow in flows:
            classified = self._classify_single_flow(flow)
            self.classified_flows.append(classified)
            
            # 統計
            module = classified.get('function_module', 'unknown')
            language = classified.get('language', 'unknown')
            category = classified.get('module_category', 'unknown')
            attack_type = classified.get('attack_type', 'unknown')
            
            self.module_stats[module] += 1
            self.language_stats[language] += 1
            category_stats[category] += 1
            if attack_type != 'unknown':
                attack_type_stats[attack_type] += 1
        
        # 保存額外統計
        self.category_stats = category_stats
        self.attack_type_stats = attack_type_stats
        
        print(f"[OK] 分類完成: {len(self.classified_flows)} 個流程")
        print(f"   - 功能模組: {len(self.module_stats)} 個")
        print(f"   - 語言類型: {len(self.language_stats)} 個")
        print(f"   - 模組類別: Features({category_stats['features']}) / Scan({category_stats['scan']})")
        if attack_type_stats:
            print(f"   - 攻擊類型: {len(attack_type_stats)} 種")
    
    def _classify_single_flow(self, flow: Dict) -> Dict:
        """分類單個數據流
        
        與內部分類的差異：
        - 內部：根據模組路徑分類（cognitive_core, task_planning等）
        - 外部：根據功能目錄分類（function_sqli, function_xss等）
        
        Args:
            flow: 原始 flow 數據
        
        Returns:
            添加分類標籤的 flow
        """
        # 複製原始數據
        classified = flow.copy()
        
        # 1. 檢測功能模組（features/scan）
        function_module, module_category = self._detect_function_module(flow)
        classified['function_module'] = function_module
        classified['module_category'] = module_category  # 新增：功能/掃描分類
        
        # 取得模組描述
        if function_module in self.FUNCTION_MODULES:
            module_info = self.FUNCTION_MODULES[function_module]
            classified['function_module_desc'] = module_info['name']
            classified['attack_type'] = module_info['attack_type']
            classified['security_category'] = module_info['category']
        elif function_module in self.SCAN_MODULES:
            module_info = self.SCAN_MODULES[function_module]
            classified['function_module_desc'] = module_info['name']
            classified['component_type'] = module_info['component_type']
        else:
            classified['function_module_desc'] = "未知模組"
        
        # 2. 檢測語言
        language = self._detect_language(flow)
        classified['language'] = language
        
        # 3. 檢測入口點（與內部不同：專注於攻擊模組的入口）
        entry_points = self._detect_entry_points(flow)
        classified['entry_points'] = entry_points
        classified['has_entry_point'] = len(entry_points) > 0
        
        # 4. 生成執行指令 (如果是 Python)
        if language == "Python" and entry_points:
            classified['cli_command'] = self._generate_cli_command(
                flow, function_module, entry_points[0]
            )
        
        return classified
    
    def _detect_function_module(self, flow: Dict) -> Tuple[str, str]:
        """檢測功能模組和類別
        
        與內部檢測的差異：
        - 內部：從模組路徑檢測（cognitive_core/neural_network）
        - 外部：從功能目錄檢測（features/function_sqli 或 scan/typescript_engine）
        
        從文件路徑中識別功能模組名稱
        例如: services/features/function_sqli/scanner.py
              → ('function_sqli', 'features')
              services/scan/typescript_engine/src/index.ts
              → ('typescript_engine', 'scan')
        
        Returns:
            (模組名稱, 模組類別) 
            模組類別: 'features' 或 'scan'
        """
        path = flow.get('path', [])
        file_path = flow.get('file_path', '')
        
        # 優先檢測功能模組 (features/function_*)
        for module_name in self.FUNCTION_MODULES.keys():
            if module_name in file_path:
                return (module_name, "features")
        
        # 檢測掃描模組 (services/scan/*)
        for module_name in self.SCAN_MODULES.keys():
            if module_name in file_path:
                return (module_name, "scan")
        
        # 從 path 檢測
        for node in path:
            for module_name in list(self.FUNCTION_MODULES.keys()) + list(self.SCAN_MODULES.keys()):
                if module_name in node:
                    # 根據模組名稱判斷類別
                    if module_name.startswith('function_'):
                        return (module_name, "features")
                    else:
                        return (module_name, "scan")
        
        return ("unknown", "unknown")
    
    def _detect_language(self, flow: Dict) -> str:
        """檢測程式語言"""
        file_path = flow.get('file_path', '')
        
        for ext, lang in self.LANGUAGE_EXTENSIONS.items():
            if file_path.endswith(ext):
                return lang
        
        return "unknown"
    
    def _detect_entry_points(self, flow: Dict) -> List[str]:
        """檢測入口點
        
        Returns:
            入口點函數名稱列表
        """
        entry_points = []
        path = flow.get('path', [])
        
        for node in path:
            node_lower = node.lower()
            for keyword in self.ENTRY_POINT_KEYWORDS:
                if keyword in node_lower:
                    entry_points.append(node)
                    break
        
        return entry_points
    
    def _generate_cli_command(
        self,
        flow: Dict,
        module: str,
        entry_point: str
    ) -> str:
        """生成 CLI 執行指令
        
        Args:
            flow: flow 數據
            module: 功能模組名稱
            entry_point: 入口點函數名
        
        Returns:
            CLI 指令字符串
        """
        file_path = flow.get('file_path', '')
        
        # 基本格式: python <file_path>
        cmd = f"python {file_path}"
        
        # 如果有特定入口點，添加參數
        if entry_point and entry_point != "main":
            cmd += f" --function {entry_point}"
        
        # 添加目標參數占位符
        cmd += " --target <TARGET>"
        
        return cmd
    
    def generate_reports(self):
        """生成分類報告"""
        # 1. JSON 數據
        self._save_classification_json()
        
        # 2. Markdown 摘要
        self._save_classification_summary()
        
        # 3. 詳細報告
        self._save_complete_details()
    
    def _save_classification_json(self):
        """保存分類 JSON"""
        output_file = self.output_dir / "classification_data.json"
        
        data = {
            "metadata": {
                "type": "external_modules",
                "total_flows": len(self.classified_flows),
                "total_modules": len(self.module_stats),
                "total_languages": len(self.language_stats),
                "generated_at": datetime.now().isoformat()
            },
            "modules": dict(self.module_stats),
            "languages": dict(self.language_stats),
            "flows": self.classified_flows
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存分類數據: {output_file}")
    
    def _save_classification_summary(self):
        """保存分類摘要（外部模組版本）
        
        與內部摘要的差異：
        - 內部：5大模組 + AI組件類型統計
        - 外部：功能模組 + 掃描模組 + 攻擊類型 + 多語言統計
        """
        output_file = self.output_dir / "classification_summary.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 標題和基本信息
            f.write("# 外部模組分類摘要\n\n")
            f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**分析範圍**: 功能模組 (features) + 掃描引擎 (scan)\n\n")
            f.write("---\n\n")
            
            # 總體統計
            f.write("## 📊 總體統計\n\n")
            f.write(f"- **總流程數**: {len(self.classified_flows)}\n")
            f.write(f"- **功能模組數**: {len(self.module_stats)}\n")
            f.write(f"- **程式語言數**: {len(self.language_stats)}\n")
            
            # 模組類別統計
            if hasattr(self, 'category_stats'):
                features_count = self.category_stats.get('features', 0)
                scan_count = self.category_stats.get('scan', 0)
                f.write(f"- **功能模組流程**: {features_count}\n")
                f.write(f"- **掃描引擎流程**: {scan_count}\n")
            
            f.write("\n---\n\n")
            
            # 功能模組分佈（按攻擊類型分組）
            f.write("## 🎯 功能模組分佈\n\n")
            f.write("### 攻擊檢測模組\n\n")
            f.write("| 模組 | 說明 | 攻擊類型 | 流程數 | 語言 |\n")
            f.write("|------|------|----------|--------|------|\n")
            
            for module, count in sorted(self.module_stats.items(), key=lambda x: x[1], reverse=True):
                if module in self.FUNCTION_MODULES:
                    info = self.FUNCTION_MODULES[module]
                    # 統計該模組使用的語言
                    module_flows = [f for f in self.classified_flows if f.get('function_module') == module]
                    langs = set(f.get('language', 'unknown') for f in module_flows)
                    langs_str = ', '.join(sorted(langs))
                    
                    f.write(f"| {module} | {info['name']} | {info['attack_type']} | {count} | {langs_str} |\n")
            
            f.write("\n### 掃描引擎模組\n\n")
            f.write("| 模組 | 說明 | 組件類型 | 流程數 | 語言 |\n")
            f.write("|------|------|----------|--------|------|\n")
            
            for module, count in sorted(self.module_stats.items(), key=lambda x: x[1], reverse=True):
                if module in self.SCAN_MODULES:
                    info = self.SCAN_MODULES[module]
                    module_flows = [f for f in self.classified_flows if f.get('function_module') == module]
                    langs = set(f.get('language', 'unknown') for f in module_flows)
                    langs_str = ', '.join(sorted(langs))
                    
                    f.write(f"| {module} | {info['name']} | {info['component_type']} | {count} | {langs_str} |\n")
            
            f.write("\n---\n\n")
            
            # 攻擊類型統計
            if hasattr(self, 'attack_type_stats') and self.attack_type_stats:
                f.write("## ⚔️ 攻擊類型分佈\n\n")
                f.write("| 攻擊類型 | 流程數 | 占比 |\n")
                f.write("|----------|--------|------|\n")
                
                total = sum(self.attack_type_stats.values())
                for attack_type, count in sorted(self.attack_type_stats.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total * 100) if total > 0 else 0
                    f.write(f"| {attack_type} | {count} | {percentage:.1f}% |\n")
                
                f.write("\n---\n\n")
            
            # 語言分佈（詳細統計）
            f.write("## 💻 程式語言分佈\n\n")
            f.write("| 語言 | 流程數 | 占比 | 關聯模組 |\n")
            f.write("|------|--------|------|----------|\n")
            
            for lang, count in sorted(self.language_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.classified_flows) * 100) if self.classified_flows else 0
                
                # 統計該語言關聯的模組
                lang_flows = [f for f in self.classified_flows if f.get('language') == lang]
                modules = set(f.get('function_module', 'unknown') for f in lang_flows)
                modules_str = ', '.join(sorted(modules))
                
                f.write(f"| {lang} | {count} | {percentage:.1f}% | {modules_str} |\n")
            
            f.write("\n---\n\n")
            
            # 多語言模組統計
            f.write("## 🌐 多語言支援統計\n\n")
            
            # 找出支援多語言的模組
            module_languages = defaultdict(set)
            for flow in self.classified_flows:
                module = flow.get('function_module', 'unknown')
                lang = flow.get('language', 'unknown')
                module_languages[module].add(lang)
            
            multilang_modules = {m: langs for m, langs in module_languages.items() if len(langs) > 1}
            
            if multilang_modules:
                f.write(f"共有 {len(multilang_modules)} 個模組支援多種程式語言：\n\n")
                f.write("| 模組 | 支援語言 | 語言數 |\n")
                f.write("|------|----------|--------|\n")
                
                for module, langs in sorted(multilang_modules.items(), key=lambda x: len(x[1]), reverse=True):
                    langs_str = ', '.join(sorted(langs))
                    f.write(f"| {module} | {langs_str} | {len(langs)} |\n")
            else:
                f.write("目前所有模組均為單一語言實現。\n")
        
        print(f"[OK] 保存摘要報告: {output_file}")
    
    def _save_complete_details(self):
        """保存完整細節（外部模組版本）
        
        與內部詳細報告的差異：
        - 內部：按5大模組分組，顯示AI組件類型
        - 外部：按功能/掃描分組，顯示攻擊類型和多語言實現
        """
        output_file = self.output_dir / "complete_flow_details.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 標題
            f.write("# 外部模組完整流程細節\n\n")
            f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**總流程數**: {len(self.classified_flows)}\n\n")
            f.write("---\n\n")
            
            # 按模組分組
            flows_by_module = defaultdict(list)
            for flow in self.classified_flows:
                module = flow.get('function_module', 'unknown')
                flows_by_module[module].append(flow)
            
            # 功能模組
            f.write("# 功能模組 (Features)\n\n")
            
            for module in sorted(flows_by_module.keys()):
                if module not in self.FUNCTION_MODULES:
                    continue
                    
                flows = flows_by_module[module]
                info = self.FUNCTION_MODULES[module]
                
                f.write(f"## {module}\n\n")
                f.write(f"**說明**: {info['name']}\n")
                f.write(f"**攻擊類型**: {info['attack_type']}\n")
                f.write(f"**安全類別**: {info['category']}\n")
                f.write(f"**流程數量**: {len(flows)}\n\n")
                
                # 按語言分組顯示
                flows_by_lang = defaultdict(list)
                for flow in flows:
                    lang = flow.get('language', 'unknown')
                    flows_by_lang[lang].append(flow)
                
                for lang in sorted(flows_by_lang.keys()):
                    lang_flows = flows_by_lang[lang]
                    f.write(f"### {lang} 實現 ({len(lang_flows)} 個流程)\n\n")
                    
                    for flow in lang_flows:
                        self._write_flow_detail(f, flow)
                
                f.write("\n---\n\n")
            
            # 掃描模組
            f.write("# 掃描引擎 (Scan)\n\n")
            
            for module in sorted(flows_by_module.keys()):
                if module not in self.SCAN_MODULES:
                    continue
                    
                flows = flows_by_module[module]
                info = self.SCAN_MODULES[module]
                
                f.write(f"## {module}\n\n")
                f.write(f"**說明**: {info['name']}\n")
                f.write(f"**組件類型**: {info['component_type']}\n")
                f.write(f"**流程數量**: {len(flows)}\n\n")
                
                for flow in flows:
                    self._write_flow_detail(f, flow)
                
                f.write("\n---\n\n")
            
            # 未分類模組
            unknown_flows = flows_by_module.get('unknown', [])
            if unknown_flows:
                f.write("# 未分類模組\n\n")
                f.write(f"**流程數量**: {len(unknown_flows)}\n\n")
                
                for flow in unknown_flows:
                    self._write_flow_detail(f, flow)
                
                f.write("\n---\n\n")
        
        print(f"[OK] 保存完整細節: {output_file}")
    
    def _write_flow_detail(self, f, flow: Dict):
        """寫入單個流程的詳細信息"""
        f.write(f"#### Flow {flow['id']}\n\n")
        f.write(f"- **語言**: {flow.get('language', 'unknown')}\n")
        f.write(f"- **文件**: `{flow.get('file_path', 'unknown')}`\n")
        f.write(f"- **路徑長度**: {flow.get('length', 0)} 步\n")
        
        # 攻擊類型（如果有）
        if flow.get('attack_type'):
            f.write(f"- **攻擊類型**: {flow['attack_type']}\n")
        
        # 安全類別（如果有）
        if flow.get('security_category'):
            f.write(f"- **安全類別**: {flow['security_category']}\n")
        
        # 入口點
        if flow.get('has_entry_point'):
            entry_points = flow.get('entry_points', [])
            f.write(f"- **入口點**: {', '.join(entry_points)}\n")
        
        # CLI 指令
        if flow.get('cli_command'):
            f.write(f"- **CLI 指令**: `{flow['cli_command']}`\n")
        
        # 執行路徑
        path = flow.get('path', [])
        if path:
            f.write(f"- **執行路徑**: `{' → '.join(path)}`\n")
        
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="AIVA 外部模組分類器 - 分析 features/scan 模組",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="包含 analysis_results.json 的目錄"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="輸出目錄"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="顯示詳細訊息"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 AIVA 外部模組分類器")
    print("=" * 60)
    
    classifier = ExternalModuleClassifier(
        input_dir=args.input,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    try:
        classifier.load_flow_data()
        classifier.classify_flows()
        classifier.generate_reports()
        
        print("\n✨ 分類完成!")
        
    except Exception as e:
        print(f"\n❌ 分類失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    # 設定 Windows Console 編碼
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    sys.exit(main())
