#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA External Module Batch Classifier
====================================
批次掃描所有外部模組並生成整合報告

功能：
- 自動掃描所有 function_* 和 scan_* 模組
- 整合所有模組的分析結果
- 生成統一的綜合報告
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import argparse

# 重用 v2 的模組信息推斷邏輯
ATTACK_TYPE_PATTERNS = {
    "sqli": {"type": "injection", "category": "database_security", "name": "SQL 注入檢測"},
    "xss": {"type": "injection", "category": "web_security", "name": "XSS 漏洞檢測"},
    "ssrf": {"type": "ssrf", "category": "network_security", "name": "SSRF 漏洞檢測"},
    "idor": {"type": "access_control", "category": "authorization", "name": "IDOR 漏洞檢測"},
    "info_leak": {"type": "information_disclosure", "category": "data_protection", "name": "信息洩露檢測"},
    "infoleak": {"type": "information_disclosure", "category": "data_protection", "name": "信息洩露檢測"},
    "bizlogic": {"type": "business_logic", "category": "logic_flaw", "name": "業務邏輯漏洞"},
    "crypto": {"type": "cryptographic", "category": "encryption", "name": "加密相關漏洞"},
    "authn": {"type": "authentication", "category": "identity", "name": "身份驗證"},
    "authz": {"type": "authorization", "category": "access_control", "name": "權限控制"},
    "scan": {"type": "scanner", "category": "scanning", "name": "掃描引擎"},
    "typescript": {"type": "language_engine", "category": "analysis", "name": "TypeScript 分析引擎"},
    "rust": {"type": "language_engine", "category": "analysis", "name": "Rust 分析引擎"},
    "go": {"type": "language_engine", "category": "analysis", "name": "Go 分析引擎"}
}

LANGUAGE_EXTENSIONS = {
    ".py": "Python",
    ".rs": "Rust",
    ".go": "Go",
    ".ts": "TypeScript",
    ".js": "JavaScript"
}


class ExternalModuleBatchClassifier:
    """批次外部模組分類器"""
    
    def __init__(self, workspace_root: str, output_dir: str, verbose: bool = False):
        self.workspace_root = Path(workspace_root)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.all_modules = {}  # {module_name: module_data}
        self.all_flows = []
        self.total_stats = defaultdict(int)
        
    def scan_modules(self) -> List[Path]:
        """掃描所有外部模組目錄"""
        module_dirs = []
        
        # 掃描 module_analysis/function_*
        if (self.workspace_root / "module_analysis").exists():
            for path in (self.workspace_root / "module_analysis").iterdir():
                if path.is_dir() and path.name.startswith("function_"):
                    analysis_file = path / "analysis_results.json"
                    if analysis_file.exists():
                        module_dirs.append(path)
        
        # 掃描 services/features/.../function_*
        features_dir = self.workspace_root / "services" / "features"
        if features_dir.exists():
            for analysis_file in features_dir.rglob("analysis_results.json"):
                parent = analysis_file.parent
                if "function_" in str(parent):
                    module_dirs.append(parent)
        
        # 掃描 services/scan/*
        scan_dir = self.workspace_root / "services" / "scan"
        if scan_dir.exists():
            for analysis_file in scan_dir.rglob("analysis_results.json"):
                module_dirs.append(analysis_file.parent)
        
        if self.verbose:
            print(f"[OK] 發現 {len(module_dirs)} 個模組目錄")
        
        return module_dirs
    
    def process_module(self, module_dir: Path) -> Dict[str, Any]:
        """處理單個模組"""
        analysis_file = module_dir / "analysis_results.json"
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 從路徑提取模組名稱
            module_name, module_category = self._extract_module_from_path(str(module_dir))
            
            # 推斷模組信息
            module_info = self._infer_module_info(module_name)
            
            # 處理流程
            flows = data.get('flows', [])
            if not flows and 'flow_chains' in data:
                flows = self._convert_flow_chains(data['flow_chains'], data.get('target', ''))
            
            # 分類流程
            classified_flows = []
            languages = set()
            
            for flow in flows:
                lang = self._detect_language(flow)
                languages.add(lang)
                
                classified_flow = {
                    **flow,
                    'module_name': module_name,
                    'module_category': module_category,
                    'module_info': module_info,
                    'language': lang
                }
                classified_flows.append(classified_flow)
                self.all_flows.append(classified_flow)
            
            module_data = {
                'name': module_name,
                'category': module_category,
                'info': module_info,
                'path': str(module_dir),
                'flow_count': len(classified_flows),
                'languages': sorted(languages),
                'flows': classified_flows
            }
            
            if self.verbose:
                print(f"  [{module_name}] {len(flows)} 個流程, 語言: {', '.join(languages)}")
            
            return module_data
            
        except Exception as e:
            if self.verbose:
                print(f"  [錯誤] {module_dir.name}: {e}")
            return {}
    
    def _extract_module_from_path(self, path_str: str) -> tuple:
        """從路徑提取模組名稱"""
        normalized = path_str.replace('\\', '/').lower()
        
        # function_*
        if match := re.search(r'function_([a-z_0-9]+)', normalized):
            return (f"function_{match.group(1)}", "features")
        
        # scan/*
        if '/scan/' in normalized:
            if 'typescript' in normalized:
                return ('typescript_engine', 'scan')
            elif 'rust' in normalized:
                return ('rust_engine', 'scan')
            else:
                return ('scan_engine', 'scan')
        
        return ("unknown", "unknown")
    
    def _infer_module_info(self, module_name: str) -> Dict[str, str]:
        """推斷模組信息"""
        for pattern, info in ATTACK_TYPE_PATTERNS.items():
            if pattern in module_name.lower():
                return info
        
        return {
            'name': module_name.replace('_', ' ').title(),
            'type': 'unknown',
            'category': 'unknown'
        }
    
    def _convert_flow_chains(self, flow_chains: List[List[str]], target: str) -> List[Dict]:
        """轉換 flow_chains 為 flows"""
        flows = []
        for i, chain in enumerate(flow_chains, 1):
            if chain:
                flows.append({
                    'id': i,
                    'path': chain,
                    'file_path': chain[0],
                    'full_path': chain,
                    'length': len(chain),
                    'start': chain[0],
                    'end': chain[-1],
                    'target': target
                })
        return flows
    
    def _detect_language(self, flow: Dict) -> str:
        """檢測語言"""
        if 'language' in flow:
            return flow['language'].capitalize()
        
        file_path = flow.get('file_path', '') or (flow.get('full_path', [''])[0] if flow.get('full_path') else '')
        
        for ext, lang in LANGUAGE_EXTENSIONS.items():
            if file_path.endswith(ext):
                return lang
        
        return "Unknown"
    
    def generate_integrated_report(self):
        """生成整合報告"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 計算統計
        stats = self._calculate_statistics()
        
        # 生成報告
        self._save_summary_report(stats)
        self._save_detailed_report()
        self._save_json_data(stats)
        
        if self.verbose:
            print(f"\n[OK] 整合報告已生成:")
            print(f"    {self.output_dir / 'integrated_summary.md'}")
            print(f"    {self.output_dir / 'integrated_details.md'}")
            print(f"    {self.output_dir / 'integrated_data.json'}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """計算統計信息"""
        by_module = defaultdict(int)
        by_category = defaultdict(int)
        by_attack_type = defaultdict(int)
        by_language = defaultdict(int)
        
        for flow in self.all_flows:
            by_module[flow['module_name']] += 1
            by_category[flow['module_category']] += 1
            by_attack_type[flow['module_info']['type']] += 1
            by_language[flow['language']] += 1
        
        return {
            'total_modules': len(self.all_modules),
            'total_flows': len(self.all_flows),
            'by_module': dict(by_module),
            'by_category': dict(by_category),
            'by_attack_type': dict(by_attack_type),
            'by_language': dict(by_language)
        }
    
    def _save_summary_report(self, stats: Dict):
        """保存摘要報告"""
        output_file = self.output_dir / "integrated_summary.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 外部模組整合分類報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 總體統計
            f.write("## 總體統計\n\n")
            f.write(f"- **總模組數**: {stats['total_modules']}\n")
            f.write(f"- **總流程數**: {stats['total_flows']}\n")
            f.write(f"- **支援語言**: {', '.join(stats['by_language'].keys())}\n\n")
            f.write("---\n\n")
            
            # 模組列表
            f.write("## 發現的模組\n\n")
            f.write("| 模組名稱 | 類別 | 描述 | 攻擊類型 | 流程數 | 語言 |\n")
            f.write("|---------|------|------|---------|--------|------|\n")
            
            for module_name, module_data in sorted(self.all_modules.items(), 
                                                   key=lambda x: x[1]['flow_count'], reverse=True):
                category = module_data['category']
                info = module_data['info']
                count = module_data['flow_count']
                languages = ', '.join(module_data['languages'])
                
                f.write(f"| {module_name} | {category} | {info['name']} | "
                       f"{info['type']} | {count} | {languages} |\n")
            
            f.write("\n---\n\n")
            
            # 類別分布
            f.write("## 模組類別分布\n\n")
            f.write("| 類別 | 模組數 | 流程數 | 百分比 |\n")
            f.write("|------|--------|--------|--------|\n")
            
            total = sum(stats['by_category'].values())
            for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
                module_count = sum(1 for m in self.all_modules.values() if m['category'] == category)
                percentage = (count / total * 100) if total > 0 else 0
                f.write(f"| {category} | {module_count} | {count} | {percentage:.1f}% |\n")
            
            f.write("\n---\n\n")
            
            # 攻擊類型分布
            f.write("## 攻擊類型分布\n\n")
            f.write("| 攻擊類型 | 流程數 | 百分比 | 相關模組 |\n")
            f.write("|---------|--------|--------|----------|\n")
            
            total = sum(stats['by_attack_type'].values())
            for attack_type, count in sorted(stats['by_attack_type'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                modules = [m for m, d in self.all_modules.items() if d['info']['type'] == attack_type]
                f.write(f"| {attack_type} | {count} | {percentage:.1f}% | {', '.join(modules[:3])} |\n")
            
            f.write("\n---\n\n")
            
            # 語言分布
            f.write("## 語言分布\n\n")
            f.write("| 語言 | 流程數 | 百分比 | 模組數 |\n")
            f.write("|------|--------|--------|--------|\n")
            
            total = sum(stats['by_language'].values())
            for lang, count in sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                module_count = sum(1 for m in self.all_modules.values() if lang in m['languages'])
                f.write(f"| {lang} | {count} | {percentage:.1f}% | {module_count} |\n")
    
    def _save_detailed_report(self):
        """保存詳細報告"""
        output_file = self.output_dir / "integrated_details.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 外部模組詳細流程信息\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 按類別分組
            by_category = defaultdict(list)
            for module_name, module_data in self.all_modules.items():
                by_category[module_data['category']].append((module_name, module_data))
            
            for category in sorted(by_category.keys()):
                f.write(f"## {category.upper()}\n\n")
                
                for module_name, module_data in sorted(by_category[category], key=lambda x: x[1]['flow_count'], reverse=True):
                    f.write(f"### {module_name}\n\n")
                    f.write(f"**描述**: {module_data['info']['name']}\n\n")
                    f.write(f"**攻擊類型**: {module_data['info']['type']}\n\n")
                    f.write(f"**流程數**: {module_data['flow_count']}\n\n")
                    f.write(f"**語言**: {', '.join(module_data['languages'])}\n\n")
                    
                    # 顯示前3個流程
                    flows = module_data['flows'][:3]
                    if flows:
                        f.write("**流程範例**:\n\n")
                        for i, flow in enumerate(flows, 1):
                            path = flow.get('path', [])
                            if isinstance(path, list) and len(path) > 0:
                                # 提取檔案名稱
                                parts = []
                                for p in path[:3]:
                                    if '\\' in p:
                                        parts.append(p.split('\\')[-1])
                                    else:
                                        parts.append(p.split('/')[-1])
                                path_str = ' → '.join(parts)
                                f.write(f"{i}. {path_str}")
                                if len(path) > 3:
                                    f.write(f" ... (+{len(path)-3} 更多)")
                                f.write("\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
    
    def _save_json_data(self, stats: Dict):
        """保存 JSON 數據"""
        output_file = self.output_dir / "integrated_data.json"
        
        # 轉換 set 為 list
        serializable_modules = {}
        for module_name, module_data in self.all_modules.items():
            serializable_modules[module_name] = {
                **module_data,
                'flows': module_data['flows'][:10]  # 只保存前10個流程
            }
        
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_modules': stats['total_modules'],
                'total_flows': stats['total_flows']
            },
            'modules': serializable_modules,
            'statistics': stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def run(self):
        """執行批次分類"""
        print("[執行] 掃描外部模組...")
        module_dirs = self.scan_modules()
        
        if not module_dirs:
            print("[警告] 未發現任何外部模組")
            return
        
        print(f"[執行] 處理 {len(module_dirs)} 個模組...")
        for module_dir in module_dirs:
            module_data = self.process_module(module_dir)
            if module_data:
                self.all_modules[module_data['name']] = module_data
        
        print(f"[執行] 生成整合報告...")
        self.generate_integrated_report()
        
        print(f"\n[OK] 批次分類完成!")
        print(f"    處理模組: {len(self.all_modules)}")
        print(f"    總流程數: {len(self.all_flows)}")
        print(f"    輸出目錄: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='外部模組批次分類器 - 生成整合報告')
    parser.add_argument('-w', '--workspace', default='.',
                       help='工作區根目錄 (預設: 當前目錄)')
    parser.add_argument('-o', '--output', default='external_module_reports',
                       help='輸出目錄 (預設: external_module_reports)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='顯示詳細訊息')
    
    args = parser.parse_args()
    
    try:
        classifier = ExternalModuleBatchClassifier(
            workspace_root=args.workspace,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        classifier.run()
        
    except Exception as e:
        print(f"[錯誤] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
