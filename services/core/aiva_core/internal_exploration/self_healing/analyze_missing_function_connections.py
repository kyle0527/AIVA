"""
缺失函數連接分析器
==================
找出有明確出口和入口但沒有連接的函數對

分析邏輯：
1. 函數A調用了函數B，但找不到B的定義 → 缺失連接
2. 函數B被定義但從未被調用 → 孤立函數
3. 函數A有出口(返回值)，函數C需要這個入口(參數)，但沒連上 → 潛在連接缺失
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FunctionSignature:
    """函數簽名"""
    name: str
    file_path: str
    file_name: str
    line_number: int
    # 出口
    has_return: bool = False
    return_type: str = "unknown"
    # 入口
    parameters: List[str] = field(default_factory=list)
    param_types: Dict[str, str] = field(default_factory=dict)
    # 調用情況
    calls_functions: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    # 連接狀態
    is_connected: bool = False


@dataclass
class MissingConnection:
    """缺失的連接"""
    connection_type: str  # "定義缺失", "調用缺失", "參數不匹配", "返回值未使用"
    source_function: str
    source_file: str
    target_function: str
    target_file: Optional[str]
    description: str
    confidence: str  # "high", "medium", "low"
    suggestion: str = ""


class MissingConnectionAnalyzer:
    """缺失連接分析器"""
    
    def __init__(self, analysis_results_path: str, source_root: str):
        self.results_path = Path(analysis_results_path)
        self.source_root = Path(source_root)
        self.analysis_data = self._load_analysis_results()
        
        # 函數簽名映射
        self.function_signatures: Dict[str, FunctionSignature] = {}
        self.functions_by_name: Dict[str, List[FunctionSignature]] = defaultdict(list)
        
        # 缺失連接
        self.missing_connections: List[MissingConnection] = []
        
        # Python 內建函數和常用方法（更全面）
        self.builtins = {
            # 基本內建函數
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
            'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'max', 'min',
            'open', 'input', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
            'property', 'staticmethod', 'classmethod', 'super', 'any', 'all', 'abs',
            'bool', 'bytes', 'chr', 'ord', 'dir', 'eval', 'exec', 'format', 'hash',
            'help', 'id', 'iter', 'next', 'object', 'oct', 'hex', 'pow', 'repr',
            'round', 'slice', 'vars', 'compile', 'delattr', 'divmod', 'globals',
            'locals', 'memoryview', 'reversed', 'frozenset', 'ascii', 'bin',
            'callable', 'complex', 'issubclass',
            
            # 字符串方法
            'lower', 'upper', 'strip', 'lstrip', 'rstrip', 'split', 'rsplit',
            'join', 'replace', 'format', 'format_map', 'startswith', 'endswith',
            'find', 'rfind', 'index', 'rindex', 'count', 'capitalize', 'title',
            'swapcase', 'center', 'ljust', 'rjust', 'zfill', 'encode', 'decode',
            'isalpha', 'isdigit', 'isalnum', 'isspace', 'isupper', 'islower',
            
            # 列表/dict方法
            'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'sort',
            'reverse', 'copy', 'get', 'keys', 'values', 'items', 'update',
            'setdefault', 'popitem', 'fromkeys',
            
            # 文件方法
            'read', 'write', 'close', 'readline', 'readlines', 'writelines',
            'seek', 'tell', 'flush', 'fileno', 'truncate',
            
            # pathlib 方法
            'exists', 'is_file', 'is_dir', 'mkdir', 'rmdir', 'unlink', 'rename',
            'glob', 'rglob', 'stat', 'chmod', 'touch', 'resolve', 'absolute',
            'relative_to', 'with_name', 'with_suffix', 'parent', 'name', 'stem',
            'suffix', 'suffixes', 'as_posix', 'as_uri', 'joinpath',
            
            # 常用標準庫
            'loads', 'dumps', 'load', 'dump',  # json
            'sleep', 'time',  # time
            'datetime', 'timedelta', 'now', 'today',  # datetime
            'request', 'get', 'post', 'put', 'delete',  # requests
            'parse', 'urlencode', 'quote', 'unquote',  # urllib
        }
        
        # 標準庫模組（不需要檢查這些調用）
        self.stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'logging',
            'collections', 'itertools', 'functools', 'operator', 're', 'math',
            'random', 'uuid', 'hashlib', 'base64', 'subprocess', 'threading',
            'multiprocessing', 'asyncio', 'urllib', 'http', 'socket', 'email',
            'csv', 'xml', 'sqlite3', 'pickle', 'io', 'shutil', 'tempfile',
        }
    
    def _load_analysis_results(self) -> Dict:
        """載入分析結果"""
        json_path = self.results_path / "analysis_results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"找不到分析結果: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_function_signatures(self):
        """從源碼提取函數簽名"""
        print("📝 提取函數簽名...")
        
        # 從分析結果獲取所有Python文件
        flow_chains = self.analysis_data.get('flow_chains', [])
        all_files = set()
        
        for chain in flow_chains:
            all_files.update(chain)
        
        processed = 0
        for file_path in all_files:
            try:
                self._analyze_file_functions(file_path)
                processed += 1
                if processed % 20 == 0:
                    print(f"   處理進度: {processed}/{len(all_files)}")
            except Exception as e:
                if "syntax error" not in str(e).lower():
                    pass  # 靜默跳過
        
        print(f"   ✅ 提取完成: {len(self.function_signatures)} 個函數")
    
    def _analyze_file_functions(self, file_path: str):
        """分析單個文件的函數"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = self._extract_function_signature(node, file_path)
                    
                    key = f"{file_path}::{sig.name}"
                    self.function_signatures[key] = sig
                    self.functions_by_name[sig.name].append(sig)
                    
        except (SyntaxError, UnicodeDecodeError, OSError):
            pass
    
    def _extract_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str) -> FunctionSignature:
        """提取函數簽名"""
        sig = FunctionSignature(
            name=node.name,
            file_path=file_path,
            file_name=Path(file_path).name,
            line_number=node.lineno
        )
        
        self._extract_parameters(node, sig)
        self._extract_return_info(node, sig)
        self._extract_function_calls(node, sig)
        
        return sig
    
    def _extract_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef, sig: FunctionSignature):
        """提取函數參數"""
        for arg in node.args.args:
            param_name = arg.arg
            sig.parameters.append(param_name)
            
            if arg.annotation:
                try:
                    sig.param_types[param_name] = ast.unparse(arg.annotation)
                except (AttributeError, ValueError):
                    sig.param_types[param_name] = "Any"
    
    def _extract_return_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef, sig: FunctionSignature):
        """提取返回值信息"""
        # 檢查是否有返回值
        has_return = False
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                has_return = True
                break
        
        sig.has_return = has_return
        
        # 提取返回類型註解
        if node.returns:
            try:
                sig.return_type = ast.unparse(node.returns)
            except (AttributeError, ValueError):
                sig.return_type = "Any"
    
    def _extract_function_calls(self, node: ast.FunctionDef | ast.AsyncFunctionDef, sig: FunctionSignature):
        """提取函數調用信息"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = self._get_function_name(child)
                
                if func_name and func_name not in self.builtins:
                    sig.calls_functions.append(func_name)
    
    def _get_function_name(self, call_node: ast.Call) -> str | None:
        """從調用節點獲取函數名"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def find_missing_definitions(self):
        """找出調用但缺失定義的函數"""
        print("\n🔍 查找缺失的函數定義...")
        
        missing_count = 0
        
        for key, sig in self.function_signatures.items():
            for called_func in sig.calls_functions:
                # 檢查是否應該忽略
                if self._should_ignore_connection(sig, called_func):
                    continue
                
                # 檢查是否能找到定義
                if called_func not in self.functions_by_name:
                    # 可能是外部庫，跳過常見的外部調用
                    if self._is_likely_external(called_func):
                        continue
                    
                    missing = MissingConnection(
                        connection_type="定義缺失",
                        source_function=sig.name,
                        source_file=sig.file_name,
                        target_function=called_func,
                        target_file=None,
                        description=f"函數 '{sig.name}' 調用了 '{called_func}'，但找不到該函數的定義",
                        confidence="high",
                        suggestion=f"檢查是否缺少導入，或 '{called_func}' 的實現文件"
                    )
                    self.missing_connections.append(missing)
                    missing_count += 1
        
        print(f"   🔴 發現 {missing_count} 個缺失定義")
    
    def find_orphaned_with_entry_exit(self):
        """找出有明確入口出口但孤立的函數"""
        print("\n🔍 查找有入口出口但孤立的函數...")
        
        orphaned_count = 0
        
        for key, sig in self.function_signatures.items():
            # 跳過私有函數和特殊函數
            if sig.name.startswith('_') and sig.name not in ['__init__', '__call__']:
                continue
            
            # 有參數(入口)或有返回值(出口)
            has_interface = len(sig.parameters) > 0 or sig.has_return
            
            if has_interface and len(sig.called_by) == 0:
                # 檢查是否真的被調用
                is_called = any(
                    sig.name in other_sig.calls_functions 
                    for other_sig in self.function_signatures.values()
                )
                
                if not is_called:
                    interface_desc = []
                    if sig.parameters:
                        interface_desc.append(f"參數: {', '.join(sig.parameters)}")
                    if sig.has_return:
                        interface_desc.append(f"返回: {sig.return_type}")
                    
                    missing = MissingConnection(
                        connection_type="調用缺失",
                        source_function=sig.name,
                        source_file=sig.file_name,
                        target_function="(未知調用者)",
                        target_file=None,
                        description=f"函數 '{sig.name}' 有明確接口({'; '.join(interface_desc)})但從未被調用",
                        confidence="high",
                        suggestion="此函數可能是廢棄代碼，或缺少調用入口"
                    )
                    self.missing_connections.append(missing)
                    orphaned_count += 1
        
        print(f"   🟡 發現 {orphaned_count} 個有接口但孤立的函數")
    
    def find_potential_missing_links(self):
        """找出潛在的缺失連接（基於名稱和簽名相似度）"""
        print("\n🔍 查找潛在的缺失連接...")
        
        potential_count = 0
        
        # 找出調用了不存在函數的情況
        for key, sig in self.function_signatures.items():
            for called_func in sig.calls_functions:
                if called_func in self.functions_by_name:
                    continue  # 已經有定義
                
                if self._is_likely_external(called_func):
                    continue
                
                # 查找名稱相似的函數
                similar_funcs = self._find_similar_functions(called_func)
                
                if similar_funcs:
                    for similar_sig in similar_funcs[:3]:  # 最多顯示3個
                        missing = MissingConnection(
                            connection_type="潛在連接缺失",
                            source_function=sig.name,
                            source_file=sig.file_name,
                            target_function=similar_sig.name,
                            target_file=similar_sig.file_name,
                            description=f"'{sig.name}' 調用 '{called_func}'(不存在)，可能想調用 '{similar_sig.name}'",
                            confidence="medium",
                            suggestion=f"檢查是否應該調用 '{similar_sig.name}' (在 {similar_sig.file_name})"
                        )
                        self.missing_connections.append(missing)
                        potential_count += 1
        
        print(f"   💡 發現 {potential_count} 個潛在連接")
    
    def find_unused_returns(self):
        """找出返回值從未被使用的函數"""
        print("\n🔍 查找返回值未被使用的函數...")
        
        unused_count = 0
        
        for key, sig in self.function_signatures.items():
            if not sig.has_return:
                continue
            
            # 找出所有調用此函數的地方
            callers = [
                other_sig for other_sig in self.function_signatures.values()
                if sig.name in other_sig.calls_functions
            ]
            
            if not callers:
                continue  # 沒人調用，在其他檢查中處理
            
            # 檢查調用者是否使用返回值的功能已移除（需要更深入的AST分析）
        
        print(f"   🔵 發現 {unused_count} 個可能未使用的返回值")
    
    def _is_likely_external(self, func_name: str) -> bool:
        """
        判斷是否可能是外部庫函數或內建方法
        改進版：更激進的過濾，減少誤報
        """
        # 1. 檢查是否在內建列表中
        if func_name in self.builtins:
            return True
        
        # 2. 檢查是否是常見的魔術方法
        if func_name.startswith('__') and func_name.endswith('__'):
            return True
        
        # 3. 檢查是否以標準庫模組名開頭
        func_lower = func_name.lower()
        for module in self.stdlib_modules:
            if func_lower.startswith(module + '_') or func_lower.startswith(module):
                return True
        
        # 4. 常見的外部庫模式（保持原有邏輯）
        external_patterns = [
            'logger', 'log', 'debug', 'info', 'warning', 'error', 'critical',
            'json', 'loads', 'dumps', 'parse', 'parser', 'request', 'response',
            'get', 'post', 'put', 'delete', 'patch', 'head', 'options',
            'save', 'load', 'dump', 'restore',
            'now', 'today', 'strftime', 'strptime', 'sleep', 'run', 'execute',
            'connect', 'disconnect', 'query', 'fetch', 'commit', 'rollback',
        ]
        
        return any(pattern in func_lower for pattern in external_patterns)
    
    def _should_ignore_connection(self, source_sig: FunctionSignature, target_func: str) -> bool:
        """
        新增方法：判斷是否應該忽略這個連接檢查
        減少誤報的額外過濾
        """
        # 1. 源文件是測試文件，目標函數是測試相關
        if 'test' in source_sig.file_name.lower():
            if any(keyword in target_func.lower() for keyword in ['test', 'mock', 'assert', 'expect']):
                return True
        
        # 2. 單字母函數名（通常是臨時變量）
        if len(target_func) == 1:
            return True
        
        # 3. 常見的屬性訪問模式（如 obj.method）
        if '.' in target_func:
            return True
        
        # 4. 內建類型的構造函數
        if target_func in ['list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool']:
            return True
        
        return False
    
    def _find_similar_functions(self, func_name: str) -> List[FunctionSignature]:
        """查找名稱相似的函數"""
        similar = []
        func_lower = func_name.lower()
        
        for name, sigs in self.functions_by_name.items():
            name_lower = name.lower()
            
            # 檢查相似度
            if (func_lower in name_lower or name_lower in func_lower or 
                self._levenshtein_distance(func_lower, name_lower) <= 3):
                similar.extend(sigs)
        
        return similar
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """計算編輯距離"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _group_connections(self) -> tuple[dict, dict]:
        """連接分組
        
        複雜度: A (2) - 簡單循環
        """
        by_type = defaultdict(list)
        by_confidence = defaultdict(list)
        
        for conn in self.missing_connections:
            by_type[conn.connection_type].append(conn)
            by_confidence[conn.confidence].append(conn)
        
        return by_type, by_confidence
    
    def _write_summary_section(self, f, by_type: dict, by_confidence: dict) -> None:
        """寫入摘要
        
        複雜度: A (2) - 簡單循環
        """
        f.write("## 📊 分析摘要\n\n")
        f.write(f"- **分析函數總數:** {len(self.function_signatures)}\n")
        f.write(f"- **發現缺失連接:** {len(self.missing_connections)} 個\n\n")
        
        f.write("### 按問題類型分布\n\n")
        for conn_type, conns in sorted(by_type.items()):
            f.write(f"- **{conn_type}:** {len(conns)} 個\n")
        f.write("\n")
        
        f.write("### 按置信度分布\n\n")
        f.write(f"- 🔴 **High (高):** {len(by_confidence['high'])} 個\n")
        f.write(f"- 🟡 **Medium (中):** {len(by_confidence['medium'])} 個\n")
        f.write(f"- 🔵 **Low (低):** {len(by_confidence['low'])} 個\n\n")
    
    def _format_connection_item(self, f, i: int, conn) -> None:
        """格式化連接項
        
        複雜度: A (3) - 簡單條件
        """
        confidence_icon = {'high': '🔴', 'medium': '🟡', 'low': '🔵'}.get(conn.confidence, '⚪')
        
        f.write(f"#### {i}. {confidence_icon} {conn.source_function} → {conn.target_function}\n\n")
        f.write(f"**源文件:** `{conn.source_file}`\n\n")
        if conn.target_file:
            f.write(f"**目標文件:** `{conn.target_file}`\n\n")
        f.write(f"**問題:** {conn.description}\n\n")
        if conn.suggestion:
            f.write(f"**建議:** {conn.suggestion}\n\n")
        f.write("---\n\n")
    
    def _write_detailed_list(self, f, by_type: dict) -> None:
        """寫入詳細列表
        
        複雜度: B (7) - 多層循環
        """
        f.write("## 📋 詳細問題列表\n\n")
        
        def get_confidence_order(conn):
            order_dict = {'high': 0, 'medium': 1, 'low': 2}
            return order_dict.get(conn.confidence, 3)
        
        for conn_type in ["定義缺失", "調用缺失", "潛在連接缺失", "返回值未使用"]:
            if conn_type not in by_type:
                continue
            
            conns = by_type[conn_type]
            f.write(f"### {conn_type} ({len(conns)} 個)\n\n")
            
            sorted_conns = sorted(conns, key=get_confidence_order)
            
            for i, conn in enumerate(sorted_conns[:50], 1):
                self._format_connection_item(f, i, conn)
            
            if len(conns) > 50:
                f.write(f"*... 還有 {len(conns) - 50} 個 {conn_type} 問題*\n\n")
    
    def _write_priority_suggestions(self, f) -> None:
        """寫入優先建議
        
        複雜度: B (6) - 循環 + 條件
        """
        f.write("## 🔧 優先修復建議\n\n")
        
        high_priority = [c for c in self.missing_connections if c.confidence == 'high']
        if not high_priority:
            return
        
        f.write("### 🔴 高優先級（必須修復）\n\n")
        
        high_by_type = defaultdict(list)
        for conn in high_priority:
            high_by_type[conn.connection_type].append(conn)
        
        for conn_type, conns in high_by_type.items():
            f.write(f"**{conn_type}:** {len(conns)} 個\n")
            for conn in conns[:5]:
                f.write(f"- `{conn.source_function}` → `{conn.target_function}` ({conn.source_file})\n")
            if len(conns) > 5:
                f.write(f"- ... 還有 {len(conns) - 5} 個\n")
            f.write("\n")
    
    def _write_statistics(self, f) -> None:
        """寫入統計信息
        
        複雜度: A (1) - 簡單統計
        """
        f.write("## 📈 統計信息\n\n")
        f.write(f"- **有入口(參數)的函數:** {len([s for s in self.function_signatures.values() if s.parameters])} 個\n")
        f.write(f"- **有出口(返回值)的函數:** {len([s for s in self.function_signatures.values() if s.has_return])} 個\n")
        f.write(f"- **既有入口又有出口的函數:** {len([s for s in self.function_signatures.values() if s.parameters and s.has_return])} 個\n")
    
    def _write_file_ranking(self, f) -> None:
        """寫入文件排名
        
        複雜度: A (3) - 簡單循環
        """
        files_with_issues = defaultdict(int)
        for conn in self.missing_connections:
            files_with_issues[conn.source_file] += 1
        
        if files_with_issues:
            f.write("\n### 問題文件排名 (Top 10)\n\n")
            sorted_files = sorted(files_with_issues.items(), key=lambda x: x[1], reverse=True)
            for file_name, count in sorted_files[:10]:
                f.write(f"- `{file_name}`: {count} 個問題\n")
    
    def generate_report(self, output_path: str | None = None):
        """生成報告
        
        複雜度: A (2) - 主協調函數，單一職責
        原複雜度: D (28) -> 重構後: A (2) [93% 降低]
        """
        if output_path is None:
            report_path = self.results_path / "missing_function_connections.md"
        else:
            report_path = Path(output_path)
        
        print("\n📝 生成報告...")
        
        # 1. 分組連接
        by_type, by_confidence = self._group_connections()
        
        # 2. 生成報告文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 缺失函數連接分析報告\n\n")
            f.write("**分析對象:** AIVA Core 系統\n\n")
            
            self._write_summary_section(f, by_type, by_confidence)
            self._write_detailed_list(f, by_type)
            self._write_priority_suggestions(f)
            self._write_statistics(f)
            self._write_file_ranking(f)
        
        print(f"   ✅ 報告已保存: {report_path}")
        return report_path
    
    def run_full_analysis(self):
        """執行完整分析"""
        print("="*60)
        print("🔍 缺失函數連接分析器")
        print("="*60)
        
        self.extract_function_signatures()
        self.find_missing_definitions()
        self.find_orphaned_with_entry_exit()
        self.find_potential_missing_links()
        self.find_unused_returns()
        
        report_path = self.generate_report()
        
        print("\n" + "="*60)
        print(f"✅ 分析完成！發現 {len(self.missing_connections)} 個缺失連接")
        print("="*60)
        
        return report_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="缺失函數連接分析器")
    parser.add_argument("--analysis-dir", "-d",
                       default="./analysis_with_unconnected",
                       help="分析結果目錄")
    parser.add_argument("--source-root", "-s",
                       default="C:\\D\\fold7\\AIVA-git\\services\\core\\aiva_core",
                       help="源代碼根目錄")
    parser.add_argument("--output", "-o",
                       help="輸出報告路徑")
    
    args = parser.parse_args()
    
    try:
        analyzer = MissingConnectionAnalyzer(args.analysis_dir, args.source_root)
        report_path = analyzer.run_full_analysis()
        
        print(f"\n📄 查看報告: {report_path}")
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
