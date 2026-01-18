#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA External Module Multi-Language Classifier
==============================================
整合所有語言的外部模組分析結果，生成統一的 classification_data.json

功能：
1. 掃描所有外部模組的 analysis_results.json
   - module_analysis/function_xss/analysis_results.json (Python)
   - module_analysis/function_sqli/analysis_results.json (Python)
   - module_analysis/function_crypto/analysis_results.json (Rust)
   - module_analysis/function_authn_go/analysis_output/analysis_results.json (Go)
   - services/scan/typescript_engine/analysis_output/analysis_results.json (TypeScript)

2. 整合所有 flows 到統一格式

3. 生成 features_classification/classification_data.json
   讓 aiva_external_executor.py 可以執行

架構位置：
- internal_exploration/aiva_external_classifier.py  (本文件 - 多語言整合)
- internal_exploration/aiva_external_executor.py    (執行器)
- internal_exploration/python_tools/                (Python AST 分析)
- internal_exploration/rust_tools/                  (Rust AST 分析)
- internal_exploration/go_tools/                    (Go AST 分析)
- internal_exploration/typescript_tools/            (TypeScript AST 分析)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import argparse

# ==========================================
# 模組類型推斷規則
# ==========================================

ATTACK_TYPE_PATTERNS = {
    "sqli": {"type": "injection", "category": "database_security", "name": "SQL 注入檢測"},
    "xss": {"type": "injection", "category": "web_security", "name": "XSS 漏洞檢測"},
    "ssrf": {"type": "ssrf", "category": "network_security", "name": "SSRF 漏洞檢測"},
    "idor": {"type": "access_control", "category": "authorization", "name": "IDOR 漏洞檢測"},
    "infoleak": {"type": "information_disclosure", "category": "data_protection", "name": "信息洩露檢測"},
    "bizlogic": {"type": "business_logic", "category": "logic_flaw", "name": "業務邏輯漏洞"},
    "crypto": {"type": "cryptographic", "category": "encryption", "name": "加密相關漏洞"},
    "authn": {"type": "authentication", "category": "identity", "name": "身份驗證"},
    "authz": {"type": "authorization", "category": "access_control", "name": "權限控制"},
    "typescript": {"type": "language_engine", "category": "analysis", "name": "TypeScript 分析引擎"},
}

LANGUAGE_EXTENSIONS = {
    ".py": "Python",
    ".rs": "Rust",
    ".go": "Go",
    ".ts": "TypeScript",
    ".js": "JavaScript"
}


class MultiLanguageClassifier:
    """多語言外部模組分類器
    
    整合 Python/Rust/Go/TypeScript 的分析結果
    生成統一的 classification_data.json
    """
    
    def __init__(self, workspace_root: str, output_dir: str, verbose: bool = False):
        self.workspace_root = Path(workspace_root)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.all_modules = {}  # {module_name: module_data}
        self.all_flows = []    # 統一的流程列表
        self.flow_id_counter = 1
        
    def scan_all_modules(self) -> List[tuple]:
        """掃描所有外部模組的分析結果
        
        Returns:
            List[tuple]: [(module_path, language), ...]
        """
        modules = []
        
        # 1. Python 模組: module_analysis/function_*
        module_analysis = self.workspace_root / "module_analysis"
        if module_analysis.exists():
            for path in module_analysis.iterdir():
                if path.is_dir() and path.name.startswith("function_"):
                    analysis_file = path / "analysis_results.json"
                    if analysis_file.exists():
                        lang = self._detect_module_language(path)
                        modules.append((path, lang))
                        if self.verbose:
                            print(f"  [發現] {path.name} ({lang})")
        
        # 2. TypeScript 引擎: services/scan/typescript_engine
        ts_engine = self.workspace_root / "services" / "scan" / "typescript_engine" / "analysis_output"
        if (ts_engine / "analysis_results.json").exists():
            modules.append((ts_engine.parent, "TypeScript"))
            if self.verbose:
                print(f"  [發現] typescript_engine (TypeScript)")
        
        return modules
    
    def _detect_module_language(self, module_path: Path) -> str:
        """檢測模組的主要語言"""
        module_name = module_path.name.lower()
        
        # 從模組名稱判斷
        if "go" in module_name or "authn_go" in module_name:
            return "Go"
        
        if "rust" in module_name or "crypto" in module_name:
            return "Rust"
        
        if "typescript" in module_name:
            return "TypeScript"
        
        # 檢查特定的目錄結構
        if (module_path / "rust_core").exists() or (module_path / "Cargo.toml").exists():
            return "Rust"
        
        if (module_path / "go.mod").exists() or any(module_path.glob("*.go")):
            return "Go"
        
        if (module_path / "tsconfig.json").exists() or any(module_path.glob("*.ts")):
            return "TypeScript"
        
        # 預設 Python
        return "Python"
    
    def process_module(self, module_path: Path, language: str) -> Dict[str, Any]:
        """處理單個模組的分析結果"""
        # 找到 analysis_results.json
        analysis_file = module_path / "analysis_results.json"
        if not analysis_file.exists():
            analysis_file = module_path / "analysis_output" / "analysis_results.json"
        
        if not analysis_file.exists():
            if self.verbose:
                print(f"    [跳過] 找不到 analysis_results.json: {module_path.name}")
            return {}
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取模組資訊
            module_name = module_path.name
            module_info = self._infer_module_info(module_name)
            
            # 處理 flows
            flows = data.get('flows', [])
            if not flows and 'flow_chains' in data:
                flows = self._convert_flow_chains(data['flow_chains'])
            
            # 整合到統一格式
            classified_flows = []
            for flow in flows:
                classified_flow = self._normalize_flow(flow, module_name, module_info, language)
                classified_flows.append(classified_flow)
                self.all_flows.append(classified_flow)
            
            module_data = {
                'name': module_name,
                'language': language,
                'info': module_info,
                'flow_count': len(classified_flows),
                'flows': classified_flows
            }
            
            if self.verbose:
                print(f"    [{module_name}] {len(classified_flows)} 個流程 ({language})")
            
            return module_data
            
        except Exception as e:
            if self.verbose:
                print(f"    [錯誤] {module_path.name}: {e}")
            return {}
    
    def _normalize_flow(self, flow: Dict, module_name: str, module_info: Dict, language: str) -> Dict:
        """將不同語言的 flow 標準化為統一格式"""
        flow_id = self.flow_id_counter
        self.flow_id_counter += 1
        
        # 統一的 flow 格式
        normalized = {
            'id': flow_id,
            'module': module_name,
            'module_type': module_info['type'],
            'module_category': module_info['category'],
            'module_description': module_info['name'],
            'language': language,
            'path': flow.get('path', []),
            'full_path': flow.get('full_path', flow.get('path', [])),
            'length': flow.get('length', len(flow.get('path', []))),
            'start': flow.get('start', ''),
            'end': flow.get('end', ''),
        }
        
        # 提取功能描述（從 docstring 或其他元數據）
        if 'description' in flow:
            normalized['description'] = flow['description']
        
        # 保留原始數據（可選）
        if 'file_path' in flow:
            normalized['file_path'] = flow['file_path']
        if 'target' in flow:
            normalized['target'] = flow['target']
        
        # 推斷使用場景
        normalized['use_case'] = self._infer_use_case(module_name, normalized['start'], normalized['end'])
        
        return normalized
    
    def _infer_use_case(self, module_name: str, start: str, end: str) -> str:
        """根據模組、起點和終點函數名稱推斷詳細使用場景
        
        為每個具體的流程路徑生成清晰的用途說明，讓 AI 能夠理解：
        1. 這個流程在做什麼
        2. 什麼時候該使用它
        3. 它的效能特性如何
        """
        # 提取函數名的關鍵詞
        start_lower = start.lower()
        end_lower = end.lower()
        combined = f"{start_lower} {end_lower}"
        
        # === XSS 模組詳細場景 ===
        if module_name == 'function_xss':
            if 'bruteforcer' in start_lower:
                if 'url' in end_lower and 'get' in end_lower:
                    return '[XSS暴力測試] 從目標提取所有可測試 URL，用於批量掃描多個端點'
                elif 'param' in end_lower and 'get' in end_lower:
                    return '[XSS參數提取] 提取 URL 中的所有參數，用於識別潛在注入點'
                elif 'converter' in end_lower or 'convert' in end_lower:
                    return '[XSS編碼轉換] 將 payload 進行各種編碼，用於繞過 WAF 和過濾器'
            elif 'stored' in start_lower:
                return '[XSS存儲型] 檢測存儲型 XSS，payload 會保存到資料庫並在其他頁面觸發'
            elif 'blind' in start_lower:
                return '[XSS盲注] 使用 OAST 技術檢測盲 XSS，適合無法直接觀察輸出的場景'
            elif 'dom' in start_lower:
                return '[XSS DOM型] 檢測 DOM-based XSS，payload 在客戶端 JavaScript 中觸發'
            elif 'reflected' in start_lower:
                return '[XSS反射型] 檢測反射型 XSS，payload 在當前響應中立即觸發'
            elif 'detect_language' in start_lower:
                return '[XSS環境偵測] 檢測目標使用的程式語言和框架，用於選擇最佳 payload'
            elif 'context' in start_lower or 'context' in end_lower:
                return '[XSS上下文分析] 分析 payload 插入位置的 HTML/JS 上下文，選擇合適的注入方式'
            else:
                return '[XSS通用] 跨站腳本漏洞檢測，掃描 Web 應用的輸入/輸出點'
        
        # === SQLi 模組詳細場景 ===
        elif module_name == 'function_sqli':
            if 'sqlmap' in combined:
                return '[SQLi工具整合] 調用 SQLMap 進行深度 SQL 注入測試，支持多種資料庫'
            elif 'custom' in start_lower:
                return '[SQLi自訂掃描] 使用自訂 payload 進行 SQL 注入測試，適合特殊場景'
            elif 'nosql' in combined:
                return '[NoSQL注入] 檢測 MongoDB/Redis 等 NoSQL 資料庫的注入漏洞'
            elif 'blind' in combined:
                return '[SQLi盲注] 使用時間延遲或布爾盲注技術，適合無錯誤訊息的場景'
            elif 'wrapper' in end_lower or 'encode' in end_lower:
                return '[SQLi編碼繞過] 對 payload 進行編碼和混淆，用於繞過 WAF'
            elif 'bounty' in combined:
                return '[SQLi賞金模式] 針對高價值目標的深度測試，使用高級繞過技術'
            elif 'parse' in combined:
                return '[SQLi結果解析] 分析 SQL 注入測試結果，提取資料庫資訊'
            else:
                return '[SQLi通用] SQL 注入漏洞檢測，測試資料庫查詢接口'
        
        # === SSRF 模組詳細場景 ===
        elif module_name == 'function_ssrf':
            if 'param' in combined and 'semantic' in combined:
                return '[SSRF參數語義] 分析參數語義判斷是否可能觸發後端請求（如 url=, callback= 等）'
            elif 'internal' in combined and 'address' in combined:
                return '[SSRF內網探測] 檢測是否可以訪問內部網路（127.0.0.1, 192.168.x.x, 10.x.x.x）'
            elif 'oast' in combined:
                return '[SSRF帶外檢測] 使用 OAST 技術驗證盲 SSRF，通過 DNS/HTTP 回調確認'
            elif 'telemetry' in combined:
                return '[SSRF遙測記錄] 記錄 SSRF 測試的詳細數據，用於分析和報告'
            elif 'resolve' in combined and 'payload' in combined:
                return '[SSRF Payload 生成] 根據目標環境生成最佳 SSRF payload'
            elif 'issue' in combined and 'request' in combined:
                return '[SSRF請求發送] 實際發送 SSRF 測試請求並分析響應'
            else:
                return '[SSRF通用] 服務器端請求偽造檢測，測試後端 URL 請求功能'
        
        # === IDOR 模組詳細場景 ===
        elif module_name == 'function_idor':
            if 'extract' in combined:
                return '[IDOR資源提取] 從 URL 和響應中提取資源 ID（如 /user/123, /order/456）'
            elif 'context' in combined:
                return '[IDOR上下文檢測] 分析資源訪問上下文，判斷權限控制是否正確'
            elif 'worker' in end_lower:
                return '[IDOR異步掃描] 後台 Worker 模式，適合大量資源 ID 的批量測試'
            elif 'telemetry' in combined:
                return '[IDOR遙測記錄] 記錄越權測試的詳細數據和結果'
            elif 'engine' in combined:
                return '[IDOR檢測引擎] 核心越權檢測邏輯，嘗試不同用戶訪問相同資源'
            else:
                return '[IDOR通用] 越權訪問檢測，測試資源 ID 的權限控制'
        
        # === BizLogic 模組詳細場景 ===
        elif module_name == 'function_bizlogic':
            if 'race' in combined:
                return '[業務邏輯-競態] 檢測競態條件漏洞，同時發送多個請求測試鎖機制'
            elif 'price' in combined:
                return '[業務邏輯-價格] 檢測價格操縱漏洞，嘗試修改金額、折扣、數量等'
            elif 'workflow' in combined:
                return '[業務邏輯-流程] 檢測工作流繞過，嘗試跳過必要步驟（如支付驗證）'
            elif 'comprehensive' in combined:
                return '[業務邏輯-全面掃描] 運行所有業務邏輯測試，包括競態、價格、流程'
            else:
                return '[業務邏輯通用] 業務流程漏洞檢測，測試支付、訂單等關鍵業務'
        
        # === Authn 模組詳細場景 ===
        elif module_name == 'function_authn_go':
            if 'broker' in combined:
                return '[認證-消息隊列] 連接 RabbitMQ 接收認證測試任務'
            elif 'analyze' in combined and 'cookie' in combined:
                return '[認證-Cookie分析] 分析 Cookie 的安全性（HttpOnly, Secure, SameSite）'
            elif 'analyze' in combined and 'jwt' in combined:
                return '[認證-JWT分析] 檢測 JWT 的弱點（弱簽名、算法混淆、過期時間）'
            elif 'session' in combined:
                return '[認證-會話管理] 測試會話固定、會話劫持等漏洞'
            else:
                return '[認證通用] 身份驗證漏洞檢測，測試登錄、會話、權限'
        
        # === TypeScript Engine 詳細場景 ===
        elif module_name == 'typescript_engine':
            if 'network' in combined and 'asset' in combined:
                return '[前端-網路資產] 提取前端代碼中的所有網路請求（API、AJAX、Fetch）'
            elif 'api' in combined and 'request' in combined:
                return '[前端-API請求] 分析前端 API 調用模式，識別敏感端點'
            elif 'ajax' in combined:
                return '[前端-AJAX分析] 檢測 XMLHttpRequest 和 jQuery AJAX 請求'
            elif 'pattern' in combined:
                return '[前端-模式分析] 分析請求模式，識別認證機制和參數結構'
            else:
                return '[前端通用] 前端代碼和網路請求分析，檢測客戶端漏洞'
        
        # === 通用場景（fallback）===
        else:
            return f'[{module_name}] 安全檢測功能'
    
    def _convert_flow_chains(self, flow_chains: List[List[str]]) -> List[Dict]:
        """轉換 flow_chains 為標準 flows 格式"""
        flows = []
        for i, chain in enumerate(flow_chains, 1):
            if chain:
                flows.append({
                    'id': i,
                    'path': chain,
                    'full_path': chain,
                    'length': len(chain),
                    'start': chain[0] if chain else '',
                    'end': chain[-1] if chain else ''
                })
        return flows
    
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
    
    def generate_classification_data(self):
        """生成統一的 classification_data.json
        
        格式與內部分類器產生的相同，讓執行器可以使用
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "classification_data.json"
        
        # 統計資訊
        by_language = defaultdict(int)
        by_module = defaultdict(int)
        by_type = defaultdict(int)
        
        for flow in self.all_flows:
            by_language[flow['language']] += 1
            by_module[flow['module']] += 1
            by_type[flow['module_type']] += 1
        
        # 生成分類數據
        classification_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'aiva_external_classifier.py',
                'description': '外部模組多語言整合分類數據',
                'total_flows': len(self.all_flows),
                'total_modules': len(self.all_modules),
                'languages': list(by_language.keys()),
                'statistics': {
                    'by_language': dict(by_language),
                    'by_module': dict(by_module),
                    'by_type': dict(by_type)
                }
            },
            'modules': self.all_modules,
            'flows': self.all_flows
        }
        
        # 寫入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classification_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n[OK] 生成 classification_data.json")
            print(f"    檔案: {output_file}")
            print(f"    總流程: {len(self.all_flows)}")
            print(f"    總模組: {len(self.all_modules)}")
            print(f"    語言: {', '.join(by_language.keys())}")
    
    def generate_summary_report(self):
        """生成摘要報告（可選）"""
        output_file = self.output_dir / "classification_summary.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 外部模組多語言分類報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 總體統計
            f.write("## 總體統計\n\n")
            f.write(f"- **總模組數**: {len(self.all_modules)}\n")
            f.write(f"- **總流程數**: {len(self.all_flows)}\n\n")
            
            # 模組列表
            f.write("## 模組列表\n\n")
            f.write("| 模組名稱 | 語言 | 類型 | 流程數 |\n")
            f.write("|---------|------|------|--------|\n")
            
            for module_name, module_data in sorted(self.all_modules.items(), 
                                                   key=lambda x: x[1]['flow_count'], reverse=True):
                lang = module_data['language']
                info = module_data['info']
                count = module_data['flow_count']
                
                f.write(f"| {module_name} | {lang} | {info['type']} | {count} |\n")
            
            f.write("\n")
        
        if self.verbose:
            print(f"    摘要: {output_file}")
    
    def run(self):
        """執行多語言整合分類"""
        print("[執行] 掃描所有外部模組...")
        modules = self.scan_all_modules()
        
        if not modules:
            print("[警告] 未發現任何外部模組")
            return
        
        print(f"\n[執行] 處理 {len(modules)} 個模組...")
        for module_path, language in modules:
            module_data = self.process_module(module_path, language)
            if module_data:
                self.all_modules[module_data['name']] = module_data
        
        print(f"\n[執行] 生成統一分類數據...")
        self.generate_classification_data()
        self.generate_summary_report()
        
        print(f"\n[完成] 多語言整合分類完成!")
        print(f"    處理模組: {len(self.all_modules)}")
        print(f"    總流程數: {len(self.all_flows)}")
        print(f"    輸出目錄: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='外部模組多語言分類器 - 整合所有語言的分析結果'
    )
    parser.add_argument('-w', '--workspace', 
                       default='.',
                       help='工作區根目錄 (預設: 當前目錄)')
    parser.add_argument('-o', '--output', 
                       default='features_classification',
                       help='輸出目錄 (預設: features_classification)')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='顯示詳細訊息')
    
    args = parser.parse_args()
    
    try:
        classifier = MultiLanguageClassifier(
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
