#!/usr/bin/env python3
"""
AIVA External Module Multi-Language Classifier (v3.3)
======================================================
整合所有語言的外部模組分析結果，生成統一的 classification_data.json

版本歷史:
---------
v3.3 (2026-01-21) - 🔧 統一路徑配置
  - 整合 paths_config.py 作為路徑的單一數據源 (SOT)
  - 外部分類輸出至 external_classification.json
  
v3.2 - 多語言支援完善

功能：
1. 掃描所有外部模組的 analysis_results.json
   - services/features/function_* (Python/Rust)
   - services/scan/*_engine (TypeScript)

2. 整合所有 flows 到統一格式

3. 生成 external_classification.json (SOT)
   讓 aiva_external_executor.py 可以執行

數據存放位置:
-------------
輸出目錄: services/integration/data/internal_exploration/
分類文件: external_classification.json
摘要文件: external_classification_summary.md

架構位置：
- internal_exploration/aiva_external_classifier.py  (本文件 - 多語言整合)
- internal_exploration/aiva_external_executor.py    (執行器)
- internal_exploration/python_tools/                (Python AST 分析)
- internal_exploration/rust_tools/                  (Rust AST 分析)
- internal_exploration/go_tools/                    (Go AST 分析)
- internal_exploration/typescript_tools/            (TypeScript AST 分析)
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

# ==========================================
# 路徑配置導入
# ==========================================

try:
    PATHS_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "integration" / "data" / "internal_exploration"
    
    from services.integration.data.internal_exploration.paths_config import (
        DATA_ROOT,
        PROJECT_ROOT,
        CombinedPaths,
        ExternalPaths,
        ensure_all_dirs,
    )
    USING_NEW_PATHS = True
except ImportError:
    USING_NEW_PATHS = False
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    DATA_ROOT = Path(__file__).resolve().parent / "features_classification"

# ==========================================
# 常量定義
# ==========================================

ANALYSIS_RESULTS_JSON = "analysis_results.json"

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

# ==========================================
# 多語言能力識別原則 (基於原則.md)
# ==========================================
# 5大原則: 邊界原則、序列化原則、拓撲學原則、命名慣例、框架約定

CAPABILITY_PRINCIPLES = {
    "Python": {
        "boundary": "動態導入邊界 - 透過 importlib 動態載入 services.features 下的模組",
        "serialization": "基本型別 - CLI 參數解析為 str, int, bool, List, Dict",
        "topology": "模組入口 - 尋找 is_entry_point=True 的頂層函數",
        "naming": "操作型命名 - 如 run_reflected_test，動詞明確",
        "framework": "features 目錄 - 位於 services/features/ 下的模組",
        
        # 判斷標準
        "include_criteria": [
            "參數無 self 或 self 可透過無參建構子初始化",
            "參數類型有 Type Hint 且為基本型別 (str, int, bool, List, Dict)",
            "函數名非 _ 開頭",
        ],
        "exclude_criteria": [
            "參數包含無法解析的物件 (如 Request 物件, Session 實例)",
            "位於 __init__.py 但無具體邏輯",
            "裝飾器顯示依賴 Web Context (如 @app.route 的 request 參數)",
        ],
    },
    "Rust": {
        "boundary": "CLI 參數邊界 - 透過 clap 接收命令行參數 (--url)",
        "serialization": "字串化輸入 - 所有輸入 (cookies-json, url) 都是可序列化的字串或 JSON",
        "topology": "Manifest Target - Cargo.toml 定義的 binary target",
        "naming": "子命令映射 - 如 scan-js, analyze-cookies，使用祈使句動詞",
        "framework": "rust_core 目錄 - 位於特定子目錄且有 Cargo.toml",
        
        # 判斷標準
        "include_criteria": [
            "函數參數為基本型別 (String, u32, Vec<u8>)",
            "結構體有 #[derive(Deserialize)] (表示可從 JSON 構建)",
        ],
        "exclude_criteria": [
            "參數包含 &mut self 且該 Struct 無法簡單 new()",
            "參數包含 Lifetime 標記 'a 且指向非靜態資源",
        ],
    },
    "Go": {
        "boundary": "Stdin/Struct 邊界 - 接收 JSON 輸入，解析 Request struct",
        "serialization": "JSON Struct - 利用 Go 的 struct tag (json:\"field\") 定義輸入",
        "topology": "Main Package - go run 只能執行 main package 中的入口檔案",
        "naming": "服務命名 - 如 DialBroker，描述具體動作而非狀態",
        "framework": "微服務結構 - 獨立的 Go 模組資料夾",
        
        # 判斷標準
        "include_criteria": [
            "函數接收 Struct (且該 Struct 有 json tag) 或基本型別",
            "函數簽名不包含 interface{} 或 chan",
            "位於 main package 或 public (大寫開頭) 的 Library 函數",
        ],
        "exclude_criteria": [
            "函數接收 context.Context 作為唯一參數",
            "函數接收 *sql.DB, *http.Client (依賴熱狀態)",
            "函數返回 func() (閉包)",
        ],
    },
    "TypeScript": {
        "boundary": "Script 參數邊界 - 透過 process.argv 接收參數",
        "serialization": "CLI Args - 參數如 --target 都是純文字",
        "topology": "執行腳本 - 執行具體的 .ts 檔案，而非被 import 的模組",
        "naming": "功能命名 - 如 analyzeClientSideAuthBypass，描述具體分析行為",
        "framework": "engine 目錄 - 以 _engine 結尾的目錄被識別為掃描引擎",
        
        # 判斷標準
        "include_criteria": [
            "Export 的函數",
            "參數為 JSON 可序列化型別",
            "無 callback 參數",
        ],
        "exclude_criteria": [
            "參數包含 DOM 元素 (HTMLElement) 或 Node Stream",
            "依賴閉包變數",
        ],
    },
}

# 語言執行機制
LANGUAGE_EXECUTION_MECHANISM = {
    "Python": "動態導入 (Dynamic Import) - 利用 importlib 直接載入模組",
    "Go": "結構映射 (Struct Mapping) - AST 讀取 Request struct，參數轉 JSON 灌入",
    "Rust": "Manifest 驅動 - 透過 cargo run 指向特定 target",
    "TypeScript": "直接執行 (Direct Exec) - 利用 ts-node 直接執行腳本",
}


class MultiLanguageClassifier:
    """多語言外部模組分類器
    
    ⚠️ 重要架構說明：
    
    外部分類器需要讀取 4 個語言的分析結果：
    1. Python:     services/features/function_*/analysis_results.json
    2. Rust:       services/features/function_crypto/analysis_results.json  
    3. Go:         services/features/function_authn_go/analysis_results.json
    4. TypeScript: services/scan/typescript_engine/analysis_results.json
    
    然後整合輸出至 external_classification.json (SOT)
    外部執行器只需要讀取這個文件
    """
    
    def __init__(self, workspace_root: str, output_dir: str, verbose: bool = False):
        self.workspace_root = Path(workspace_root)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.all_modules = {}  # {module_name: module_data}
        self.all_flows = []    # 統一的流程列表
        self.flow_id_counter = 1
        # 分語言計數器 (每種語言獨立編號)
        self.lang_counters = {
            'Python': 1,
            'Go': 1,
            'TypeScript': 1,
            'Rust': 1,
            'JavaScript': 1
        }
        
        # 定義 4 個語言的輸入路徑
        self._setup_input_paths()
        
    def _setup_input_paths(self):
        """設置 4 個語言的分析結果輸入路徑"""
        if USING_NEW_PATHS:
            self.analysis_inputs = ExternalPaths.ANALYSIS_INPUTS
        else:
            # 降級方案：手動定義
            features_dir = self.workspace_root / "services" / "features"
            scan_dir = self.workspace_root / "services" / "scan"
            
            self.analysis_inputs = {
                "python": [
                    features_dir / "function_xss" / ANALYSIS_RESULTS_JSON,
                    features_dir / "function_sqli" / ANALYSIS_RESULTS_JSON,
                    features_dir / "function_ssrf" / ANALYSIS_RESULTS_JSON,
                    features_dir / "function_idor" / ANALYSIS_RESULTS_JSON,
                    features_dir / "function_infoleak" / ANALYSIS_RESULTS_JSON,
                    features_dir / "function_bizlogic" / ANALYSIS_RESULTS_JSON,
                ],
                "rust": [
                    features_dir / "function_crypto" / ANALYSIS_RESULTS_JSON,
                ],
                "go": [
                    features_dir / "function_authn_go" / ANALYSIS_RESULTS_JSON,
                ],
                "typescript": [
                    scan_dir / "typescript_engine" / ANALYSIS_RESULTS_JSON,
                ],
            }
        
    def scan_all_modules(self) -> list[tuple]:
        """掃描所有外部模組的分析結果
        
        讀取 4 個語言的分析結果文件
        
        Returns:
            List[tuple]: [(analysis_file_path, language), ...]
        """
        modules = []
        
        if self.verbose:
            print("\n[掃描] 讀取 4 個語言的分析結果:")
        
        # 遍歷每種語言的輸入路徑
        for lang, paths in self.analysis_inputs.items():
            lang_display = lang.capitalize() if lang != "typescript" else "TypeScript"
            
            for analysis_file in paths:
                if analysis_file.exists():
                    # ✅ 修正：直接傳文件路徑而不是 parent 目錄
                    modules.append((analysis_file, lang_display))
                    if self.verbose:
                        print(f"  ✅ [{lang_display}] {analysis_file.stem}")
                else:
                    if self.verbose:
                        print(f"  ⚠️  [{lang_display}] 找不到: {analysis_file.name}")
        
        # ⚠️ 移除額外掃描邏輯（已通過 ANALYSIS_INPUTS 集中管理）
        # 所有分析結果已集中到 integration/data/.../external/ 目錄
        # 不再需要從原始模組目錄掃描
        
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
    
    def process_module(self, analysis_file: Path, language: str) -> dict[str, Any]:
        """處理單個模組的分析結果
        
        Args:
            analysis_file: 分析結果 JSON 文件的完整路徑
            language: 模組語言
        """
        if not analysis_file.exists():
            if self.verbose:
                print(f"    [跳過] 找不到分析文件: {analysis_file.name}")
            return {}
        
        try:
            with open(analysis_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取模組資訊（從文件名推斷）
            module_name = analysis_file.stem  # 例如 function_sqli
            module_info = self._infer_module_info(module_name)
            
            # 處理 flows - 優先從 function_details 提取有調用鏈的入口點
            flows = []
            
            # ✅ 新增：優先處理 function_details（有真實調用鏈的入口點）
            if 'function_details' in data:
                flows = self._extract_entry_points_from_function_details(data['function_details'], language)
                if flows and self.verbose:
                    print(f"    [function_details] 提取 {len(flows)} 個入口點（有調用鏈）")
            
            # 如果沒有 function_details，fallback 到舊邏輯
            if not flows:
                flows = data.get('flows', [])
                
                # 處理舊格式的 graphs（不做任何過濾，全部轉換）
                if not flows and 'graphs' in data:
                    flows = self._convert_graphs_to_flows(data['graphs'])
                    if self.verbose:
                        print(f"    [graphs] 轉換 {len(data.get('graphs', []))} → {len(flows)} flows")
                
                if not flows and 'flow_chains' in data:
                    # 從 flow_chains 和 function_details 構建完整的 flows
                    flows = self._convert_flow_chains(data['flow_chains'], data.get('function_details', []))
            
            # ✅ 新增：處理有結構體定義但無流程的 Go 模組（stdin JSON 模式）
            if not flows and 'struct_definitions' in data and language == 'Go':
                flows = self._convert_struct_to_flows(data['struct_definitions'], data.get('function_details', []))
            
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
                print(f"    [錯誤] {analysis_file.stem}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _normalize_flow(self, flow: dict, module_name: str, module_info: dict, language: str) -> dict:
        """將不同語言的 flow 標準化為統一格式
        
        命名規則（按層級分類）：
        1. 頂層模組分類：features 或 scan
        2. 子模組名稱：xss, sqli, ssrf, idor, bizlogic, web_scanner 等
        3. 流程編號：該子模組內的流程序號
        
        格式：aiva_{top_module}_{sub_module}_{flow_num}
        範例：
        - aiva_features_xss_1
        - aiva_features_sqli_1
        - aiva_scan_python_1
        - aiva_scan_typescript_1
        """
        # 全局唯一 ID
        flow_id = self.flow_id_counter
        self.flow_id_counter += 1
        
        # 判斷頂層模組（features 或 scan）
        if module_name.startswith('function_'):
            top_module = 'features'
            sub_module = module_name.replace('function_', '')  # function_xss -> xss
        elif module_name.endswith('_engine'):
            top_module = 'scan'
            sub_module = module_name.replace('_engine', '')  # python_engine -> python
        else:
            top_module = 'unknown'
            sub_module = module_name
        
        # 為每個子模組維護獨立的計數器
        counter_key = f"{top_module}_{sub_module}"
        if counter_key not in self.lang_counters:
            self.lang_counters[counter_key] = 1
        
        flow_num = self.lang_counters[counter_key]
        self.lang_counters[counter_key] += 1
        
        # 統一命名格式：aiva_{top_module}_{sub_module}_{num}
        unified_name = f"aiva_{top_module}_{sub_module}_{flow_num}"
        
        # 統一的 flow 格式
        normalized = {
            'id': flow_id,
            'name': unified_name,  # 層級命名: aiva_features_xss_1, aiva_scan_python_1
            'top_module': top_module,  # features 或 scan
            'sub_module': sub_module,  # xss, sqli, python, typescript 等
            'module': module_name,  # 原始模組名: function_xss, python_engine
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
        
        # ✅ 保留來源檔案資訊（用於執行器定位類/函數）
        if 'file_path' in flow:
            normalized['file_path'] = flow['file_path']
        if 'file_name' in flow:
            normalized['file_name'] = flow['file_name']
        if 'script_name' in flow:
            normalized['script_name'] = flow['script_name']
        if 'target' in flow:
            normalized['target'] = flow['target']
        
        # ✅ 提取參數資訊（支援所有語言）
        normalized['parameters'] = self._extract_parameters(flow)
        
        # ✅ 判斷是否為可操作的能力（基於原則.md）
        normalized['is_operable'] = self._is_operable_capability(flow, language)
        normalized['operability_reason'] = self._get_operability_reason(flow, language)
        
        # 推斷使用場景
        normalized['use_case'] = self._infer_use_case(module_name, normalized['start'], normalized['end'])
        
        return normalized
    
    def _is_operable_capability(self, flow: dict, language: str) -> bool:
        """判斷能力是否可操作（基於原則.md 的判斷標準）
        
        根據5大原則判斷：
        - 邊界原則：數據來源是否可序列化
        - 序列化原則：參數類型是否為基本型別
        - 拓撲學原則：是否為入口函數
        - 命名慣例：函數名是否明確表達操作
        - 框架約定：是否符合目錄結構規範
        """
        if language not in CAPABILITY_PRINCIPLES:
            return True  # 未知語言預設可操作
        
        # 獲取函數資訊
        start_func = flow.get('start', '')
        params = flow.get('parameters', flow.get('inputs', []))
        
        # Python 排除標準
        if language == 'Python':
            # 檢查是否依賴 Web Context
            if any(p in str(params) for p in ['request', 'Request', 'session', 'Session']):
                return False
            # 檢查是否為私有函數
            func_name = start_func.split('.')[-1] if '.' in start_func else start_func
            if func_name.startswith('_') and not func_name.startswith('__'):
                return False
        
        # Go 排除標準
        elif language == 'Go':
            # 檢查是否僅接收 context
            if params == ['context.Context'] or params == ['ctx']:
                return False
            # 檢查是否依賴熱狀態
            if any(p in str(params) for p in ['*sql.DB', '*http.Client']):
                return False
        
        # Rust 排除標準
        elif language == 'Rust':
            # 檢查是否有 lifetime 標記指向非靜態資源
            if "'a" in str(params) and 'static' not in str(params).lower():
                return False
        
        # TypeScript 排除標準
        elif language == 'TypeScript':
            # 檢查是否依賴 DOM
            if any(p in str(params) for p in ['HTMLElement', 'Document', 'Window']):
                return False
        
        return True
    
    def _get_operability_reason(self, flow: dict, language: str) -> str:
        """獲取可操作性判斷的原因"""
        if language not in CAPABILITY_PRINCIPLES:
            return "未知語言，預設可操作"
        
        start_func = flow.get('start', '')
        params = flow.get('parameters', flow.get('inputs', []))
        
        if language == 'Python':
            if any(p in str(params) for p in ['request', 'Request', 'session', 'Session']):
                return "❌ 依賴 Web Context (request/session 參數)"
            func_name = start_func.split('.')[-1] if '.' in start_func else start_func
            if func_name.startswith('_') and not func_name.startswith('__'):
                return "❌ 私有函數 (以 _ 開頭)"
        
        elif language == 'Go':
            if params == ['context.Context'] or params == ['ctx']:
                return "❌ 僅接收 context.Context 參數"
            if any(p in str(params) for p in ['*sql.DB', '*http.Client']):
                return "❌ 依賴熱狀態 (sql.DB/http.Client)"
        
        elif language == 'Rust':
            if "'a" in str(params) and 'static' not in str(params).lower():
                return "❌ Lifetime 指向非靜態資源"
        
        elif language == 'TypeScript':
            if any(p in str(params) for p in ['HTMLElement', 'Document', 'Window']):
                return "❌ 依賴 DOM 元素"
        
        return "✅ 可操作 - 符合序列化原則"

    def _extract_parameters(self, flow: dict) -> dict[str, Any]:
        """提取流程的參數資訊（支援所有語言）
        
        從原始分析結果中提取參數，格式化為統一的參數描述
        支援 Python, Go, Rust, TypeScript 等所有語言
        
        Returns:
            Dict: {
                'start_function_params': [...],  # 起點函數的參數
                'end_function_params': [...],    # 終點函數的參數
                'all_params': [...],             # 流程中所有參數的合集
                'has_params': bool               # 是否有參數
            }
        """
        params_info = {
            'start_function_params': [],
            'end_function_params': [],
            'all_params': [],
            'has_params': False
        }
        
        # 從 flow 中提取參數（如果有的話）
        if 'start_params' in flow:
            params_info['start_function_params'] = flow['start_params']
            params_info['has_params'] = True
        
        if 'end_params' in flow:
            params_info['end_function_params'] = flow['end_params']
            params_info['has_params'] = True
        
        # 嘗試從其他欄位提取（不同語言的分析器可能放在不同位置）
        if 'parameters' in flow:
            params_info['all_params'] = flow['parameters']
            params_info['has_params'] = True
        elif 'inputs' in flow:
            params_info['all_params'] = flow['inputs']
            params_info['has_params'] = True
        
        return params_info
    
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
        return f'[{module_name}] 安全檢測功能'
    
    def _extract_entry_points_from_function_details(self, function_details, language: str) -> list[dict]:
        """從 function_details 提取有調用鏈的入口點
        
        function_details 結構: 
        - Python: {module_type: {func_name: func_data, ...}, ...}
        - Go/Rust: [{name: ..., inputs: ...}, ...]
        
        Args:
            function_details: 函數詳細資訊（dict 或 list）
            language: 語言類型
        
        Returns:
            有調用鏈的入口點列表
        """
        flows = []
        
        # Go/Rust 使用 list 格式，直接返回空（它們沒有 Python 式的調用鏈分析）
        if isinstance(function_details, list):
            return []
        
        # Python: 從 function_map 提取函數
        function_map = function_details.get('function_map', {})
        
        for func_name, func_data in function_map.items():
            called_functions = func_data.get('called_functions', [])
            
            # 過濾：只保留有調用鏈的函數
            if not called_functions or len(called_functions) == 0:
                continue
            
            # 過濾標準庫調用
            valid_calls = [
                call for call in called_functions
                if not any(call.startswith(x) for x in [
                    'unknown_call', 're.', 'json.', 'os.', 'sys.',
                    'asyncio.', 'aiohttp.', 'httpx.', 'subprocess.',
                    'logger.', 'console.', 'time.', 'urlparse',
                    'len', 'sum', 'int', 'str', 'any', 'bool', 'list', 'dict'
                ])
            ]
            
            # 如果沒有有效調用，跳過
            if not valid_calls:
                continue
            
            # 構建調用鏈
            call_chain = valid_calls[:10]  # 最多保留前 10 個調用
            
            # 提取參數
            parameters = func_data.get('parameters', [])
            is_entry_point = func_data.get('is_entry_point', False)
            is_async = func_data.get('is_async', False)
            
            # ✅ 提取檔案資訊（來自 AST 分析結果）
            file_name = func_data.get('file_name', '')
            file_path = func_data.get('file_path', '')
            script_name = func_data.get('script_name', '')
            
            # 構建 flow
            flow = {
                'id': len(flows) + 1,
                'path': [func_name] + call_chain[:5],  # 路徑包含起點和前5個調用
                'full_path': [func_name] + call_chain,
                'length': len(call_chain) + 1,
                'start': func_name,
                'end': call_chain[-1] if call_chain else func_name,
                'call_chain': call_chain,  # ✅ 關鍵：保留完整調用鏈
                'parameters': parameters,
                'parameter_count': len(parameters),
                'is_entry_point': is_entry_point,
                'is_async': is_async,
                'language': language,
                'source': 'function_details',  # 標記來源
                'file_name': file_name,        # ✅ 新增：來源檔名
                'file_path': file_path,        # ✅ 新增：來源完整路徑
                'script_name': script_name,    # ✅ 新增：腳本模組名（無 .py）
            }
            
            flows.append(flow)
        
        return flows
    
    def _convert_struct_to_flows(self, struct_definitions: list[dict], function_details: list[dict] | None = None) -> list[dict]:
        """將 Go struct 定義轉換為標準 flows 格式（用於 stdin JSON 微服務模式）
        
        對於使用 JSON stdin/stdout 模式的 Go 微服務，struct 欄位定義就是參數定義。
        此方法將這些 struct 定義轉換為統一的 flow 格式，讓 AI 可以理解如何呼叫這些服務。
        
        Args:
            struct_definitions: Go struct 定義列表，每個包含 struct_name, fields, source_file
            function_details: 可選的函數詳細資訊，用於補充額外的上下文
        
        Returns:
            包含路徑、參數和使用說明的 flows 列表
        """
        if function_details is None:
            function_details = []
        flows = []
        
        # 只處理 Request/Config 類型的 struct（這些是輸入參數定義）
        for struct_def in struct_definitions:
            struct_name = struct_def.get('struct_name', '')
            
            # 識別這是否為請求/配置結構（輸入參數）
            if not any(keyword in struct_name.lower() for keyword in ['request', 'config', 'input', 'param']):
                continue  # 跳過 Response/Result 等輸出結構
            
            fields = struct_def.get('fields', [])
            source_file = struct_def.get('source_file', '')
            
            # 從檔案路徑提取掃描器名稱（例如 ssrf-scanner, sca-scanner）
            scanner_name = 'unknown'
            if 'cmd' in source_file:
                parts = source_file.split('\\')
                cmd_idx = parts.index('cmd') if 'cmd' in parts else -1
                if cmd_idx >= 0 and cmd_idx + 1 < len(parts):
                    scanner_name = parts[cmd_idx + 1]
            
            # 將 struct fields 轉換為參數格式
            parameters = []
            for field in fields:
                param = {
                    'name': field.get('json_tag', field.get('name', '')),  # 優先使用 json_tag
                    'type': field.get('type', 'string'),
                    'description': field.get('description', f"{field.get('name', '')} 參數"),
                    'required': field.get('name', '').lower() in ['scanid', 'targets'],  # 假設這些為必填
                    'kind': 'json_field'  # 標記為 JSON 欄位
                }
                parameters.append(param)
            
            # 建立 flow 物件（模擬 Python 函數呼叫的格式）
            flow = {
                'id': len(flows) + 1,
                'path': [f'{scanner_name}_stdin'],  # 路徑表示這是 stdin 模式
                'full_path': [f'{scanner_name}_stdin'],
                'length': 1,
                'start': f'{scanner_name}_stdin',
                'end': f'{scanner_name}_stdout',
                'parameters': parameters,  # ✅ 關鍵：將 struct fields 作為參數
                'cli_usage': f'echo \'{{"targets":["url"],...}}\' | ./{scanner_name}',  # stdin JSON 使用範例
                'method': 'stdin_json',  # 標記呼叫方式
                'struct_name': struct_name,  # 保留原始 struct 名稱
                'source_file': source_file
            }
            
            flows.append(flow)
        
        return flows
    
    def _convert_graphs_to_flows(self, graphs: list[dict]) -> list[dict]:
        """轉換舊格式的 graphs 為標準 flows 格式（全部轉換，不過濾）
        
        Args:
            graphs: 圖結構列表，每個 graph 包含 nodes
        
        Returns:
            包含路徑和參數資訊的 flows 列表
        """
        flows = []
        
        for i, graph in enumerate(graphs, 1):
            graph_name = graph.get('name', f'graph_{i}')
            graph_type = graph.get('type', 'unknown')
            nodes = graph.get('nodes', [])
            file_path = graph.get('file_path', '')
            
            # 提取路徑（從節點中提取調用鏈）
            path = [graph_name]  # 起點是 graph 名稱
            
            # 找出所有 call 節點
            for node in nodes:
                if node.get('kind') == 'call':
                    call_label = node.get('label', '')
                    if call_label and call_label not in path:
                        # 清理 call 標籤（移除 "Call: " 前綴）
                        clean_label = call_label.replace('Call: ', '').strip()
                        path.append(clean_label)
            
            # 如果沒有 call 節點，就只有起點本身
            if len(path) == 1:
                path.append(f'{graph_name}_end')
            
            # 提取參數資訊
            start_node = next((n for n in nodes if n.get('kind') == 'start'), None)
            parameters = []
            
            if start_node and 'metadata' in start_node:
                meta = start_node['metadata']
                # 提取函數參數
                if 'parameters' in meta:
                    parameters = meta['parameters']
                # 提取返回類型
                return_type = meta.get('return_type', '')
                docstring = meta.get('docstring', '')
            
            # 構建 flow
            flow_dict = {
                'id': i,
                'path': path,
                'full_path': path,
                'length': len(path),
                'start': path[0],
                'end': path[-1],
                'graph_type': graph_type,  # 保留原始類型
                'source_file': Path(file_path).name if file_path else '',
                'node_count': len(nodes),  # 記錄節點數量
            }
            
            # 添加參數資訊
            if parameters:
                flow_dict['start_params'] = parameters
                flow_dict['parameters'] = parameters
            
            if return_type:
                flow_dict['return_type'] = return_type
            
            if docstring:
                flow_dict['docstring'] = docstring
            
            flows.append(flow_dict)
        
        return flows
    
    def _convert_flow_chains(self, flow_chains: list[list[str]], function_details: list[dict] | None = None) -> list[dict]:
        """轉換 flow_chains 為標準 flows 格式，並附加參數資訊
        
        Args:
            flow_chains: 函數調用鏈列表
            function_details: 函數詳細資訊（可能是字典結構）
        
        Returns:
            包含路徑和參數資訊的 flows 列表
        """
        # 建立函數名稱到參數的映射
        func_params_map = {}
        if function_details:
            # Python 分析器的格式：{'function_map': {函數名: 資訊}}
            if isinstance(function_details, dict):
                func_map = function_details.get('function_map', function_details)
                for func_name, func_info in func_map.items():
                    if isinstance(func_info, dict):
                        params = func_info.get('parameters', [])
                        if params:
                            func_params_map[func_name] = params
            # Go/Rust 格式：[{name: ..., inputs: ...}]
            elif isinstance(function_details, list):
                for func in function_details:
                    if isinstance(func, dict):
                        func_name = func.get('function_name') or func.get('name', '')
                        params = func.get('inputs', func.get('parameters', []))
                        if func_name and params:
                            func_params_map[func_name] = params
        
        flows = []
        for i, chain in enumerate(flow_chains, 1):
            if chain:
                # 提取起點和終點函數的參數
                start_func = chain[0] if chain else ''
                end_func = chain[-1] if chain else ''
                
                flow_dict = {
                    'id': i,
                    'path': chain,
                    'full_path': chain,
                    'length': len(chain),
                    'start': start_func,
                    'end': end_func
                }
                
                # 添加參數資訊
                if start_func in func_params_map:
                    flow_dict['start_params'] = func_params_map[start_func]
                if end_func in func_params_map:
                    flow_dict['end_params'] = func_params_map[end_func]
                
                flows.append(flow_dict)
        return flows
    
    def _infer_module_info(self, module_name: str) -> dict[str, str]:
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
        
        v3.3 (2026-01-21): 統一路徑配置
        - 輸出至 external_classification.json (SOT)
        - 同時輸出至 output_dir/classification_data.json (向後相容)
        - 新增可操作性統計（基於原則.md判斷標準）
        
        格式與內部分類器產生的相同，讓執行器可以使用
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "classification_data.json"
        
        # 統計資訊
        by_language = defaultdict(int)
        by_module = defaultdict(int)
        by_type = defaultdict(int)
        
        # v3.3: 可操作性統計（基於原則.md判斷標準）
        operable_count = 0
        non_operable_count = 0
        operable_by_language = defaultdict(int)
        non_operable_by_language = defaultdict(int)
        non_operable_reasons = defaultdict(int)
        
        for flow in self.all_flows:
            by_language[flow['language']] += 1
            by_module[flow['module']] += 1
            by_type[flow['module_type']] += 1
            
            # 統計可操作性
            if flow.get('is_operable', True):
                operable_count += 1
                operable_by_language[flow['language']] += 1
            else:
                non_operable_count += 1
                non_operable_by_language[flow['language']] += 1
                reason = flow.get('operability_reason', '未知原因')
                non_operable_reasons[reason] += 1
        
        # 生成分類數據
        classification_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'aiva_external_classifier.py',
                'version': '3.3',
                'description': '外部模組多語言整合分類數據',
                'total_flows': len(self.all_flows),
                'total_modules': len(self.all_modules),
                'languages': list(by_language.keys()),
                'classification_type': 'external',  # 標記為外部分類
                'schema_version': '3.3',
                'ai_compatible': True,
                'statistics': {
                    'by_language': dict(by_language),
                    'by_module': dict(by_module),
                    'by_type': dict(by_type)
                },
                # v3.3: 可操作性統計（基於原則.md 的5大判斷原則）
                'operability': {
                    'total_operable': operable_count,
                    'total_non_operable': non_operable_count,
                    'operable_rate': round(operable_count / len(self.all_flows) * 100, 2) if self.all_flows else 0,
                    'by_language': {
                        'operable': dict(operable_by_language),
                        'non_operable': dict(non_operable_by_language)
                    },
                    'non_operable_reasons': dict(non_operable_reasons),
                    'judgment_principles': [
                        '邊界原則 - 數據來源邊界',
                        '序列化原則 - 參數類型判斷', 
                        '拓撲學原則 - 入口函數識別',
                        '命名慣例 - 函數名稱規範',
                        '框架約定 - 目錄結構規範'
                    ]
                }
            },
            'modules': self.all_modules,
            'flows': self.all_flows
        }
        
        # 寫入到 output_dir (向後相容)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classification_data, f, indent=2, ensure_ascii=False)
        
        # v3.3: 同時輸出到統一路徑 (SOT)
        if USING_NEW_PATHS:
            sot_file = ExternalPaths.CLASSIFICATION_FILE
            sot_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sot_file, 'w', encoding='utf-8') as f:
                json.dump(classification_data, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print("\n[OK] 生成 external_classification.json (SOT)")
                print(f"    檔案: {sot_file}")
        
        if self.verbose:
            print("\n[OK] 生成 classification_data.json")
            print(f"    檔案: {output_file}")
            print(f"    總流程: {len(self.all_flows)}")
            print(f"    可操作: {operable_count} ({round(operable_count / len(self.all_flows) * 100, 1) if self.all_flows else 0}%)")
            print(f"    總模組: {len(self.all_modules)}")
            print(f"    語言: {', '.join(by_language.keys())}")
    
    def generate_summary_report(self):
        """生成摘要報告（可選）
        
        v3.3: 新增可操作性分析區段
        """
        output_file = self.output_dir / "classification_summary.md"
        
        # 計算可操作性統計
        operable_count = sum(1 for f in self.all_flows if f.get('is_operable', True))
        non_operable_count = len(self.all_flows) - operable_count
        
        # 按語言統計可操作性
        operable_by_lang = defaultdict(int)
        non_operable_by_lang = defaultdict(int)
        for flow in self.all_flows:
            if flow.get('is_operable', True):
                operable_by_lang[flow['language']] += 1
            else:
                non_operable_by_lang[flow['language']] += 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 外部模組多語言分類報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 總體統計
            f.write("## 總體統計\n\n")
            f.write(f"- **總模組數**: {len(self.all_modules)}\n")
            f.write(f"- **總流程數**: {len(self.all_flows)}\n\n")
            
            # v3.3: 可操作性統計（基於原則.md）
            f.write("## 可操作性分析\n\n")
            f.write("> 基於原則.md 的5大判斷原則（邊界、序列化、拓撲學、命名慣例、框架約定）\n\n")
            f.write(f"- ✅ **可操作流程**: {operable_count} ({round(operable_count / len(self.all_flows) * 100, 1) if self.all_flows else 0}%)\n")
            f.write(f"- ❌ **不可操作流程**: {non_operable_count} ({round(non_operable_count / len(self.all_flows) * 100, 1) if self.all_flows else 0}%)\n\n")
            
            # 按語言分類
            f.write("### 按語言分類\n\n")
            f.write("| 語言 | 可操作 | 不可操作 | 可操作率 |\n")
            f.write("|------|--------|----------|----------|\n")
            all_languages = set(operable_by_lang.keys()) | set(non_operable_by_lang.keys())
            for lang in sorted(all_languages):
                op = operable_by_lang.get(lang, 0)
                non_op = non_operable_by_lang.get(lang, 0)
                total = op + non_op
                rate = round(op / total * 100, 1) if total > 0 else 0
                f.write(f"| {lang} | {op} | {non_op} | {rate}% |\n")
            f.write("\n")
            
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
        for analysis_file, language in modules:
            module_data = self.process_module(analysis_file, language)
            if module_data:
                self.all_modules[module_data['name']] = module_data
        
        print("\n[執行] 生成統一分類數據...")
        self.generate_classification_data()
        self.generate_summary_report()
        
        print("\n[完成] 多語言整合分類完成!")
        print(f"    處理模組: {len(self.all_modules)}")
        print(f"    總流程數: {len(self.all_flows)}")
        print(f"    輸出目錄: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='外部模組多語言分類器 (v3.3) - 整合所有語言的分析結果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️ 注意: 此分類器用於外部模組 (Features/Scan) 的多語言整合
        內部模組請使用 aiva_internal_classifier.py

數據存放位置:
  services/integration/data/internal_exploration/
  ├── external_classification.json      # 分類數據 (SOT)
  ├── external_classification_summary.md   # 摘要報告
  └── analysis_results/external/        # 詳細分析

示例:
  # 使用預設路徑 (推薦)
  python aiva_external_classifier.py
  
  # 指定工作區和輸出目錄
  python aiva_external_classifier.py -w /path/to/workspace -o ./custom_output -v
        """
    )
    
    # 設定預設路徑
    if USING_NEW_PATHS:
        default_workspace = str(PROJECT_ROOT)
        default_output = str(ExternalPaths.OUTPUT_DIR)
    else:
        default_workspace = '.'
        default_output = 'features_classification'
    
    parser.add_argument('-w', '--workspace', 
                       default=default_workspace,
                       help=f'工作區根目錄 (預設: {default_workspace})')
    parser.add_argument('-o', '--output', 
                       default=default_output,
                       help=f'輸出目錄 (預設: {default_output})')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='顯示詳細訊息')
    
    args = parser.parse_args()
    
    # 確保目錄存在
    if USING_NEW_PATHS:
        ensure_all_dirs()
    
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
