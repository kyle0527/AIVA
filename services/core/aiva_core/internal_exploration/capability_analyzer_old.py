"""Capability Analyzer - 基於 py2mermaid 邏輯的數據流追蹤

核心思路 (參考 py2mermaid.py):
1. 用 py2mermaid 為每個函數生成流程圖 (start → 語句 → 調用 → end)
2. 提取圖中的函數調用節點 (ast.Call)
3. 比對不同圖的 start 和 end (函數名稱匹配)
4. 自動拼接成完整的跨函數/跨檔案調用鏈
5. 簡化成可調整參數的指令

範例:
  Manager.detect() 的圖: start → validate_input → call Scanner.scan → end
  Scanner.scan() 的圖: start → http_request → parse_result → end
  
  拼接後: Manager.detect → Scanner.scan → http_request
  簡化指令: detect(target="...", scan_type="...")

遵循規範: aiva_common README 規範, 維持六大模組架構
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sys
import importlib.util

logger = logging.getLogger(__name__)

# 動態導入 py2mermaid
def _import_py2mermaid():
    """動態導入 py2mermaid 工具"""
    py2mermaid_path = Path(__file__).parent.parent.parent.parent.parent / "tools" / "common" / "development" / "py2mermaid.py"
    
    if not py2mermaid_path.exists():
        logger.warning(f"py2mermaid.py not found at {py2mermaid_path}")
        return None
    
    spec = importlib.util.spec_from_file_location("py2mermaid", py2mermaid_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None

py2mermaid = _import_py2mermaid()


# ==================== 數據結構定義 ====================

@dataclass
class FunctionCallNode:
    """函數調用節點 - 從 py2mermaid 圖中提取"""
    function_name: str  # 被調用的函數名稱
    call_signature: str  # 完整調用簽名 (如 Scanner.scan, http_request)
    file_path: str
    line_number: int
    caller_function: str  # 調用者函數名稱
    arguments: List[str] = field(default_factory=list)  # 調用時的參數


@dataclass
class FunctionFlowGraph:
    """函數流程圖 - 由 py2mermaid 生成"""
    function_name: str
    file_path: str
    class_name: str | None
    module_name: str
    
    # py2mermaid 生成的圖
    mermaid_code: str  # 完整的 Mermaid 代碼
    
    # 從圖中提取的調用節點
    call_nodes: List[FunctionCallNode] = field(default_factory=list)
    
    # 函數元資訊
    is_async: bool = False
    is_entry_point: bool = False  # Manager/Coordinator 公開方法
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_signature(self) -> str:
        """獲取函數簽名"""
        if self.class_name:
            return f"{self.class_name}.{self.function_name}"
        return self.function_name


# ==================== 基於 py2mermaid 的數據流追蹤器 ====================

class DataFlowTracer:
    """基於 py2mermaid 的數據流追蹤器
    
    核心流程:
    1. 使用 py2mermaid.Builder 為每個函數生成流程圖
    2. 從流程圖的 AST 中提取函數調用 (ast.Call)
    3. 比對函數名稱,拼接成完整調用鏈
    4. 簡化成可調整參數的指令
    """
    
    def __init__(self, project_root: Path, storage_path: Path | None = None):
        self.project_root = project_root
        self.storage_path = storage_path or (project_root / "data" / "integration" / "data_flows")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 所有函數的流程圖索引: function_signature -> FunctionFlowGraph
        self.function_graphs: Dict[str, List[FunctionFlowGraph]] = {}
        
        # 檢查 py2mermaid 是否可用
        if py2mermaid is None:
            logger.error("py2mermaid module not available!")
        
        logger.info(f"🎨 DataFlowTracer initialized (using py2mermaid logic)")
    
    def analyze_file(self, file_path: Path) -> List[FunctionFlowGraph]:
        """分析檔案中的所有函數 (使用 py2mermaid)"""
        if py2mermaid is None:
            logger.warning("py2mermaid not available, skipping file")
            return []
        
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path.name}: {e}")
            return []
        
        # 提取模組名稱
        try:
            module_name = str(file_path.relative_to(self.project_root)).replace("\\", ".").replace("/", ".").replace(".py", "")
        except Exception:
            module_name = file_path.stem
        
        graphs = []
        
        # 遍歷所有類和函數
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                # 分析類中的方法
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        graph = self._analyze_function(item, file_path, module_name, class_name)
                        if graph:
                            graphs.append(graph)
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 確保不重複處理類方法
                is_class_method = any(
                    node in cls.body 
                    for cls in ast.walk(tree) 
                    if isinstance(cls, ast.ClassDef)
                )
                
                if not is_class_method:
                    graph = self._analyze_function(node, file_path, module_name, None)
                    if graph:
                        graphs.append(graph)
        
        return graphs
    
    def _analyze_function(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, 
                         file_path: Path, module_name: str, class_name: str | None) -> FunctionFlowGraph | None:
        """使用 py2mermaid 分析單個函數"""
        if py2mermaid is None:
            return None
        
        func_name = func_node.name
        
        # 使用 py2mermaid.Builder 生成流程圖
        try:
            builder = py2mermaid.Builder(f"Function: {func_name}", {})
            graph = builder.build_function(func_node)
            mermaid_code = graph.to_mermaid()
        except Exception as e:
            logger.debug(f"Failed to build graph for {func_name}: {e}")
            mermaid_code = ""
        
        # 提取函數參數
        parameters = []
        for arg in func_node.args.args:
            param_info = {
                "name": arg.arg,
                "annotation": ast.unparse(arg.annotation) if arg.annotation else None
            }
            parameters.append(param_info)
        
        # 提取函數調用 (從 AST 直接提取,比解析 Mermaid 更準確)
        call_nodes = self._extract_function_calls(func_node, func_name, str(file_path))
        
        # 判斷是否為入口點
        is_entry = self._is_entry_point(class_name, func_name, func_node)
        
        return FunctionFlowGraph(
            function_name=func_name,
            file_path=str(file_path),
            class_name=class_name,
            module_name=module_name,
            mermaid_code=mermaid_code,
            call_nodes=call_nodes,
            is_async=isinstance(func_node, ast.AsyncFunctionDef),
            is_entry_point=is_entry,
            parameters=parameters
        )
    
    def _extract_function_calls(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, 
                               caller_name: str, file_path: str) -> List[FunctionCallNode]:
        """從函數 AST 中提取所有函數調用"""
        calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                call_info = self._parse_call_node(node, caller_name, file_path)
                if call_info:
                    calls.append(call_info)
        
        return calls
    
    def _parse_call_node(self, call_node: ast.Call, caller_name: str, file_path: str) -> FunctionCallNode | None:
        """解析單個 Call 節點"""
        try:
            # 提取被調用的函數名稱
            if isinstance(call_node.func, ast.Name):
                # 簡單調用: func()
                func_name = call_node.func.id
                call_sig = func_name
            elif isinstance(call_node.func, ast.Attribute):
                # 方法調用: obj.method() 或 Class.method()
                if isinstance(call_node.func.value, ast.Name):
                    obj_name = call_node.func.value.id
                    method_name = call_node.func.attr
                    
                    if obj_name == "self":
                        # self.method() → 只記錄 method
                        func_name = method_name
                        call_sig = method_name
                    else:
                        # obj.method() → 完整記錄
                        func_name = method_name
                        call_sig = f"{obj_name}.{method_name}"
                else:
                    func_name = call_node.func.attr
                    call_sig = func_name
            else:
                return None
            
            # 提取參數
            arguments = []
            for arg in call_node.args:
                try:
                    arg_str = ast.unparse(arg)
                    arguments.append(arg_str)
                except Exception:
                    arguments.append("<arg>")
            
            return FunctionCallNode(
                function_name=func_name,
                call_signature=call_sig,
                file_path=file_path,
                line_number=call_node.lineno,
                caller_function=caller_name,
                arguments=arguments
            )
        except Exception:
            return None
    
    def _is_entry_point(self, class_name: str | None, func_name: str, 
                       func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """判斷是否為入口點"""
        if func_name.startswith('_'):
            return False
        
        if class_name and class_name.endswith("Manager"):
            if isinstance(func_node, ast.AsyncFunctionDef):
                return True
        
        if class_name and "Coordinator" in class_name:
            if func_name.startswith("execute_"):
                return True
        
        if class_name == "UnifiedFunctionCaller" and func_name == "call_function":
            return True
        
        return False
    
    def build_function_index(self) -> None:
        """建立函數索引 (用於拼接)"""
        self.function_graphs.clear()
        
        # 這裡需要先收集所有 FunctionFlowGraph
        # (在 CapabilityAnalyzer 中會先調用 analyze_file 收集所有圖)
        
        logger.info(f"📇 Built function index: {len(self.function_graphs)} unique functions")
    
    def add_graph(self, graph: FunctionFlowGraph) -> None:
        """添加函數圖到索引"""
        signature = graph.get_signature()
        if signature not in self.function_graphs:
            self.function_graphs[signature] = []
        self.function_graphs[signature].append(graph)
    
    def stitch_call_chain(self, entry_signature: str, max_depth: int = 10) -> List[List[FunctionFlowGraph]]:
        """拼接完整調用鏈
        
        這就是 "比對每張圖的開始跟結束,把能組合的組合起來"
        """
        if entry_signature not in self.function_graphs:
            logger.warning(f"Entry point not found: {entry_signature}")
            return []
        
        entry_graphs = self.function_graphs[entry_signature]
        all_chains = []
        
        for entry_graph in entry_graphs:
            chains = self._dfs_stitch(entry_graph, set(), [entry_graph], 0, max_depth)
            all_chains.extend(chains)
        
        logger.info(f"🔗 Stitched {len(all_chains)} complete chains from {entry_signature}")
        return all_chains
    
    def _dfs_stitch(self, current: FunctionFlowGraph, visited: Set[str], 
                   current_chain: List[FunctionFlowGraph], depth: int, max_depth: int) -> List[List[FunctionFlowGraph]]:
        """深度優先搜索拼接調用鏈"""
        # 限制深度避免無限遞歸
        if depth >= max_depth:
            return [current_chain.copy()]
        
        # 如果沒有調用其他函數,視為終點
        if not current.call_nodes:
            return [current_chain.copy()]
        
        all_chains = []
        found_match = False
        
        # 遍歷所有被調用的函數
        for call_node in current.call_nodes:
            call_sig = call_node.call_signature
            
            # 跳過已訪問的函數 (避免循環)
            if call_sig in visited:
                continue
            
            # 在索引中查找匹配的函數圖
            matching_graphs = self._find_matching_graphs(call_sig, current)
            
            if not matching_graphs:
                continue
            
            found_match = True
            
            # 對每個匹配的圖遞歸拼接
            for next_graph in matching_graphs:
                new_visited = visited.copy()
                new_visited.add(call_sig)
                new_chain = current_chain.copy()
                new_chain.append(next_graph)
                
                chains = self._dfs_stitch(next_graph, new_visited, new_chain, depth + 1, max_depth)
                all_chains.extend(chains)
        
        # 如果沒找到匹配,當前鏈也是有效的
        if not found_match:
            all_chains.append(current_chain.copy())
        
        return all_chains
    
    def _find_matching_graphs(self, call_sig: str, caller: FunctionFlowGraph) -> List[FunctionFlowGraph]:
        """查找匹配的函數圖"""
        candidates = []
        
        # 1. 完全匹配
        if call_sig in self.function_graphs:
            candidates.extend(self.function_graphs[call_sig])
        
        # 2. 類內方法匹配 (self.method → ClassName.method)
        if caller.class_name and '.' not in call_sig:
            class_method_sig = f"{caller.class_name}.{call_sig}"
            if class_method_sig in self.function_graphs:
                candidates.extend(self.function_graphs[class_method_sig])
        
        # 3. 如果是 obj.method,嘗試只用 method 匹配
        if '.' in call_sig:
            method_only = call_sig.split('.')[-1]
            # 這裡需要更智能的匹配,暫時保守處理
            pass
        
        return candidates
    
    def save_analysis_results(self, entry_points: List[str]) -> None:
        """保存分析結果"""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_functions": sum(len(graphs) for graphs in self.function_graphs.values()),
            "entry_points": entry_points,
            "call_chains": {}
        }
        
        for entry in entry_points:
            chains = self.stitch_call_chain(entry)
            results["call_chains"][entry] = [
                [
                    {
                        "signature": g.get_signature(),
                        "file": g.file_path,
                        "calls": [c.call_signature for c in g.call_nodes]
                    }
                    for g in chain
                ]
                for chain in chains[:10]  # 只保留前 10 條
            ]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.storage_path / f"call_chains_{timestamp}.json"
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        
        logger.info(f"💾 Analysis results saved to {output_file}")


# ==================== 能力分析器 ====================

class CapabilityAnalyzer:
    """能力分析器 - 使用 py2mermaid 邏輯追蹤數據流"""
    
    def __init__(self):
        self.stats = {
            "total_files": 0,
            "analyzed_files": 0,
            "entry_points_found": 0,
            "call_chains_traced": 0
        }
        logger.info("CapabilityAnalyzer initialized (py2mermaid-based mode)")
    
    async def analyze_capabilities(self, modules_info: dict) -> list[dict[str, Any]]:
        """分析模組能力 (使用 py2mermaid 邏輯)"""
        logger.info(f"🔍 Starting py2mermaid-based analysis for {len(modules_info)} modules...")
        
        if py2mermaid is None:
            logger.error("py2mermaid not available! Cannot proceed.")
            return []
        
        # 1. 初始化追蹤器
        project_root = self._get_project_root()
        tracer = DataFlowTracer(project_root)
        
        # 2. 為每個檔案的每個函數建立流程圖
        for module_name, module_data in modules_info.items():
            logger.info(f"  📁 Module: {module_name}")
            module_path = Path(module_data["path"])
            
            for file_info in module_data["files"]:
                file_path = module_path / file_info["path"]
                
                if file_path.name == "__init__.py" or file_path.suffix != ".py":
                    continue
                
                self.stats["total_files"] += 1
                
                # 使用 py2mermaid 分析檔案
                logger.debug(f"    🔄 Analyzing {file_path.name}")
                graphs = tracer.analyze_file(file_path)
                
                # 添加到索引
                for graph in graphs:
                    tracer.add_graph(graph)
                
                self.stats["analyzed_files"] += 1
        
        # 3. 建立函數索引
        tracer.build_function_index()
        
        # 4. 自動識別入口點
        entry_points = self._identify_entry_points(tracer)
        self.stats["entry_points_found"] = len(entry_points)
        logger.info(f"  🎯 Identified {len(entry_points)} entry points")
        
        # 5. 拼接調用鏈並創建能力記錄
        capabilities = []
        for entry_sig in entry_points:
            chains = tracer.stitch_call_chain(entry_sig)
            
            if chains:
                cap = self._create_capability_from_chains(entry_sig, chains, tracer)
                capabilities.append(cap)
                self.stats["call_chains_traced"] += len(chains)
        
        # 6. 保存結果
        tracer.save_analysis_results(entry_points)
        
        # 7. 統計
        logger.info(f"✅ py2mermaid-based analysis completed:")
        logger.info(f"   - Files analyzed: {self.stats['analyzed_files']}/{self.stats['total_files']}")
        logger.info(f"   - Entry points: {self.stats['entry_points_found']}")
        logger.info(f"   - Capabilities: {len(capabilities)}")
        logger.info(f"   - Call chains: {self.stats['call_chains_traced']}")
        
        return capabilities
    
    def _get_project_root(self) -> Path:
        """獲取專案根目錄"""
        current = Path(__file__).parent
        while current.name != "AIVA-git" and current.parent != current:
            current = current.parent
        return current
    
    def _identify_entry_points(self, tracer: DataFlowTracer) -> List[str]:
        """自動識別入口點"""
        entry_points = []
        
        for signature, graphs in tracer.function_graphs.items():
            for graph in graphs:
                if graph.is_entry_point:
                    entry_points.append(signature)
                    break
        
        return entry_points
    
    def _create_capability_from_chains(self, entry_sig: str, chains: List[List[FunctionFlowGraph]], 
                                      tracer: DataFlowTracer) -> dict[str, Any]:
        """從調用鏈創建能力記錄"""
        entry_graph = chains[0][0]
        
        return {
            "name": entry_graph.function_name,
            "module": entry_graph.module_name,
            "class_name": entry_graph.class_name,
            "description": f"{entry_graph.get_signature()} - Auto-discovered with py2mermaid",
            "parameters": entry_graph.parameters,
            "file_path": entry_graph.file_path,
            "return_type": None,
            "is_async": entry_graph.is_async,
            "decorators": [],
            "language": "python",
            "is_entry_point": True,
            "call_chains": [
                {
                    "chain_id": i,
                    "steps": [
                        {
                            "signature": g.get_signature(),
                            "file": g.file_path,
                            "mermaid_code": g.mermaid_code[:200] + "..." if len(g.mermaid_code) > 200 else g.mermaid_code,
                            "calls": [c.call_signature for c in g.call_nodes]
                        }
                        for g in chain
                    ]
                }
                for i, chain in enumerate(chains[:5], 1)
            ],
            "total_chains": len(chains)
        }
    
    def _save_broken_chains(self, broken: List[Dict], storage_path: Path) -> None:
        """保存斷鏈診斷"""
        output_file = storage_path / "broken_chains.json"
        output_file.write_text(json.dumps(broken, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"  💾 Broken chains saved to {output_file}")
