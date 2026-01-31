"""
aiva_flow_analyzer.py (v3.0 - Refactored)
-----------------------------------------
AIVA 結構化數據流分析器 v3.0

核心邏輯：
1. AST 解析 (基於 py2mermaid)：精準解析控制流與程式結構。
2. 參數提取 (保留 v2 功能)：為 CLI 自動化提供參數元數據。
3. 尾接頭原則 (Stitching)：基於完全名稱匹配，連接 Call (尾) 與 Definition (頭)。
4. 輸出：標準化 JSON，不含 Mermaid 語法。

設計原則：
- No Guessing: 不推測模組來源，僅依賴名稱匹配。
- No Degradation: 完整保留 AST 細節。
- Memory-First: 建立全域註冊表進行記憶體內組裝。
"""

import ast
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ==========================================
# Part 1: 基礎數據結構 (Graph & Node)
# ==========================================

class Node:
    """代表流程圖中的一個節點"""
    def __init__(self, id_: str, label: str, kind: str = "op", metadata: Optional[Dict] = None):
        self.id = self._sanitize_id(id_)
        self.label = label
        self.kind = kind  # start, end, op, cond, call
        self.nexts: List[str] = []  # 存儲下一個節點的 ID
        self.metadata = metadata if metadata is not None else {}

    def _sanitize_id(self, id_str: str) -> str:
        import re
        # 確保 ID 是合法的識別符，避免 JSON 或後續處理出錯
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", id_str)
        if cleaned and (cleaned[0].isdigit() or cleaned[0] in "_-"):
            cleaned = f"n_{cleaned}"
        return cleaned or "node"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "nexts": self.nexts,
            "metadata": self.metadata
        }

class Graph:
    """代表一個函數或模組的完整流程圖"""
    def __init__(self, name: str, file_path: str, graph_type: str = "function"):
        self.name = name  # 用於匹配的名稱 (Head Name)
        self.file_path = file_path
        self.type = graph_type # function, class, module
        self.nodes: List[Node] = []
        self.counter = 0
        
        # 建立固定的開始與結束節點
        self.start = self.add("start", name, {"type": "head", "name": name})
        self.end = self.add("end", "End")
        
        # 為了快速組圖，記錄這張圖對外的所有呼叫
        self.outgoing_calls: List[Node] = [] 

    def add(self, kind: str, label: str, metadata: Optional[Dict] = None) -> Node:
        self.counter += 1
        # 產生全域唯一的節點 ID (包含 graph name 以防衝突)
        node_id = f"{self.name}_{self.counter}"
        node_id = node_id.replace("(", "_").replace(")", "_").replace(".", "_").replace("<", "").replace(">", "")
        
        node = Node(node_id, label, kind, metadata if metadata is not None else {})
        self.nodes.append(node)
        
        if kind == 'call':
            self.outgoing_calls.append(node)
            
        return node

    def link(self, a: Node, b: Node):
        if b.id not in a.nexts:
            a.nexts.append(b.id)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "type": self.type,
            "nodes": [n.to_dict() for n in self.nodes],
            "start_node_id": self.start.id,
            "end_node_id": self.end.id
        }

# ==========================================
# Part 2: 參數提取器 (移植自 v2)
# ==========================================

class ParameterExtractor:
    """專門負責從 AST 提取函數參數與型別資訊"""
    
    def extract_parameters(self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict]:
        params = []
        args = func_node.args
        num_args = len(args.args)
        num_defaults = len(args.defaults)
        
        for i, arg in enumerate(args.args):
            if arg.arg in ['self', 'cls']: continue
                
            default_val = None
            required = True
            
            # 處理預設值對齊
            if num_defaults > 0:
                defaults_start_index = num_args - num_defaults
                if i >= defaults_start_index:
                    default_idx = i - defaults_start_index
                    default_val = self._ast_to_value(args.defaults[default_idx])
                    required = False
            
            params.append({
                "name": arg.arg,
                "type": self._annotation_to_string(arg.annotation),
                "default": default_val,
                "required": required,
                "kind": "positional_or_keyword"
            })
        
        if args.vararg:
            params.append({"name": f"*{args.vararg.arg}", "type": "tuple", "required": False, "kind": "var_positional"})
        if args.kwarg:
            params.append({"name": f"**{args.kwarg.arg}", "type": "dict", "required": False, "kind": "var_keyword"})
            
        return params

    def _ast_to_value(self, node: Optional[ast.AST]) -> Any:
        if node is None: return None
        try:
            if isinstance(node, ast.Constant): return node.value
            return ast.unparse(node)
        except: return "<complex>"

    def _annotation_to_string(self, annotation: Optional[ast.AST]) -> str:
        if annotation is None: return "any"
        try:
            return ast.unparse(annotation)
        except: return "complex"

# ==========================================
# Part 3: 流程構建器 (AST Builder - 核心邏輯)
# ==========================================

class FlowBuilder(ast.NodeVisitor):
    """
    遍歷 AST 並構建 Graph。
    繼承自 py2mermaid 的邏輯，但增加了對 ClassDef 的支援以及 Call 的特殊處理。
    """
    def __init__(self, name: str, file_path: str, graph_type: str = "function"):
        self.graph = Graph(name, file_path, graph_type)
        self.current_node = self.graph.start
        self.param_extractor = ParameterExtractor()

    def build(self, tree: ast.AST) -> Graph:
        # 如果是函數，先提取參數放到 Start Node 的 metadata
        if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = self.param_extractor.extract_parameters(tree)
            docstring = ast.get_docstring(tree) or ""
            self.graph.start.metadata.update({
                "parameters": params,
                "docstring": docstring,
                "return_type": self.param_extractor._annotation_to_string(tree.returns)
            })
            if tree.body:
                self._visit_block(tree.body)
        
        elif isinstance(tree, ast.ClassDef):
            # 類別定義也視為一個 Graph，主要為了讓 Constructor Call (e.g. MessageBroker()) 能連上
            docstring = ast.get_docstring(tree) or ""
            self.graph.start.metadata.update({
                "docstring": docstring,
                "is_class": True
            })
            # 類別的 Body 包含方法定義，這裡我們暫時只掃描賦值等初始化操作
            # 方法定義會被主程序掃描為獨立的 Graph
            if tree.body:
                self._visit_block(tree.body)

        elif isinstance(tree, ast.Module):
            if tree.body:
                self._visit_block(tree.body)
        
        self.graph.link(self.current_node, self.graph.end)
        return self.graph

    def _visit_block(self, stmts: list):
        for stmt in stmts:
            self.visit(stmt)

    # --- 關鍵：處理函數調用 (Tail) ---
    def visit_Call(self, node: ast.Call):
        func_name = self._resolve_name(node.func)
        
        # 建立 Call 節點 (Tail)
        # 注意：我們記錄 target_func 名稱，這將用於後續的「尾接頭」匹配
        call_node = self.graph.add(
            kind="call", 
            label=f"Call: {func_name}",
            metadata={
                "type": "tail",
                "target_func": func_name
            }
        )
        
        self.graph.link(self.current_node, call_node)
        self.current_node = call_node

    def _resolve_name(self, node) -> str:
        """解析函數名稱，支援 a.b.c 格式"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._resolve_name(node.value)}.{node.attr}"
        return "unknown_call"

    # --- 流程控制邏輯 (保留 py2mermaid 的精華) ---
    def visit_If(self, node: ast.If):
        try:
            cond_text = ast.unparse(node.test)
        except:
            cond_text = "condition"
            
        cond_node = self.graph.add("cond", f"if {cond_text}")
        self.graph.link(self.current_node, cond_node)
        
        entry = cond_node
        
        # Then branch
        self.current_node = entry
        self._visit_block(node.body)
        then_end = self.current_node
        
        # Else branch
        else_end = None
        if node.orelse:
            self.current_node = entry
            self._visit_block(node.orelse)
            else_end = self.current_node
            
        # Merge point
        merge_node = self.graph.add("op", "") # 空標籤代表匯合點
        self.graph.link(then_end, merge_node)
        if else_end:
            self.graph.link(else_end, merge_node)
        else:
            self.graph.link(entry, merge_node)
            
        self.current_node = merge_node

    def visit_While(self, node: ast.While):
        try:
            cond_text = ast.unparse(node.test)
        except: cond_text = "loop"
            
        cond_node = self.graph.add("cond", f"while {cond_text}")
        self.graph.link(self.current_node, cond_node)
        
        self.current_node = cond_node
        self._visit_block(node.body)
        self.graph.link(self.current_node, cond_node) # Loop back
        
        exit_node = self.graph.add("op", "")
        self.graph.link(cond_node, exit_node)
        self.current_node = exit_node

    def visit_For(self, node: ast.For):
        try:
            target = ast.unparse(node.target)
            iter_ = ast.unparse(node.iter)
            label = f"for {target} in {iter_}"
        except: label = "for loop"
        
        for_node = self.graph.add("cond", label)
        self.graph.link(self.current_node, for_node)
        
        self.current_node = for_node
        self._visit_block(node.body)
        self.graph.link(self.current_node, for_node) # Loop back
        
        exit_node = self.graph.add("op", "")
        self.graph.link(for_node, exit_node)
        self.current_node = exit_node
        
    def visit_Return(self, node: ast.Return):
        try:
            val = ast.unparse(node.value) if node.value else "None"
        except: val = "..."
        
        return_node = self.graph.add("op", f"return {val}")
        self.graph.link(self.current_node, return_node)
        self.graph.link(return_node, self.graph.end)
        self.current_node = return_node

    # 忽略 FunctionDef 和 ClassDef 的內部遍歷，它們會被主迴圈作為獨立 Graph 處理
    # 除非它們是嵌套定義，但為了簡化，v3 暫不深入嵌套定義的內部細節
    def visit_FunctionDef(self, node): pass
    def visit_AsyncFunctionDef(self, node): pass
    def visit_ClassDef(self, node): pass

# ==========================================
# Part 4: 縫合器 (FlowStitcher - 全域組裝)
# ==========================================

class FlowStitcher:
    """
    全域註冊表與連接器
    
    流程：
    1. register_graph() - 暫存每張 AST 分析結果
    2. stitch() - 執行尾接頭完全匹配，產生單步連接
    3. build_flow_chains() - 從單步連接遞歸構建完整路徑
    4. build_function_details() - 產生 function_map 和 script_functions
    """
    def __init__(self):
        # Registry: Key = Function/Class Name (Head), Value = Graph Object
        self.registry: Dict[str, Graph] = {}
        self.all_graphs: List[Graph] = []
        
        # 連接資料 (由 stitch() 填充)
        self.connections: List[Dict] = []
        
        # 鄰接表：用於快速查找 source → targets (由 stitch() 建立)
        self.adjacency: Dict[str, List[str]] = {}  # graph_name → [connected_graph_names]

    def register_graph(self, graph: Graph):
        """暫存一張 AST 分析結果"""
        self.all_graphs.append(graph)
        # 註冊 Head Name。如果是類別方法，可能需要處理 module.Class.method 的命名
        # 但依據「完全匹配」原則，我們先用最單純的名稱註冊
        self.registry[graph.name] = graph
        
        # 如果需要支援模組限定名 (e.g. utils.helper)，可以在這裡擴充
        # 例如: self.registry[f"{module_name}.{graph.name}"] = graph

    def stitch(self) -> List[Dict]:
        """執行尾接頭：遍歷所有 Call，若 target_func 存在於 registry，則建立連接"""
        self.connections = []
        self.adjacency = {}
        
        for source_graph in self.all_graphs:
            for call_node in source_graph.outgoing_calls:
                target_name = call_node.metadata.get("target_func")
                
                # --- 核心邏輯：完全匹配 (Exact Match) ---
                if target_name in self.registry:
                    target_graph = self.registry[target_name]
                    
                    # 建立邏輯連接 (在 JSON 中表現為一條邊)
                    connection = {
                        "source_graph": source_graph.name,
                        "source_node_id": call_node.id,
                        "target_graph": target_graph.name,
                        "target_node_id": target_graph.start.id,
                        "type": "call"
                    }
                    self.connections.append(connection)
                    
                    # 更新 Call Node 的 Metadata，標記已解決的連接
                    call_node.metadata["connected_to"] = target_graph.name
                    
                    # 建立鄰接表
                    if source_graph.name not in self.adjacency:
                        self.adjacency[source_graph.name] = []
                    if target_graph.name not in self.adjacency[source_graph.name]:
                        self.adjacency[source_graph.name].append(target_graph.name)

        return self.connections

    def build_flow_chains(self, max_depth: int = 10) -> List[List[str]]:
        """
        從單步連接構建完整的數據流路徑
        
        邏輯：
        1. 找到頭節點 (只當 source，不當 target 的 graph)
        2. 從每個頭節點遞歸追蹤所有分支路徑
        3. 回傳所有路徑 (每條路徑是 graph name 的列表)
        """
        if not self.connections:
            return []
        
        # 找出所有 source 和 target
        all_sources = set(self.adjacency.keys())
        all_targets = set()
        for targets in self.adjacency.values():
            all_targets.update(targets)
        
        # 頭節點：只當 source，不當 target (純數據提供者)
        head_nodes = all_sources - all_targets
        
        # 如果沒有純頭節點，選擇出度最高的節點
        if not head_nodes:
            sorted_by_outgoing = sorted(
                self.adjacency.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            head_nodes = {sorted_by_outgoing[0][0]} if sorted_by_outgoing else set()
        
        # 從每個頭節點遞歸構建路徑
        all_chains: List[List[str]] = []
        for head in head_nodes:
            self._build_paths_recursive(head, [head], all_chains, max_depth, set())
        
        # 去重並過濾長度 >= 2 的路徑
        unique_chains = []
        for chain in all_chains:
            if len(chain) >= 2 and chain not in unique_chains:
                unique_chains.append(chain)
        
        return unique_chains

    def _build_paths_recursive(
        self, 
        current: str, 
        current_path: List[str], 
        all_paths: List[List[str]], 
        max_depth: int,
        visited: set
    ):
        """遞歸構建所有分支路徑"""
        if len(current_path) >= max_depth:
            if len(current_path) >= 2:
                all_paths.append(current_path[:])
            return
        
        # 避免循環
        if current in visited:
            if len(current_path) >= 2:
                all_paths.append(current_path[:])
            return
        
        visited_copy = visited | {current}
        
        # 取得下一步可到達的節點
        next_nodes = self.adjacency.get(current, [])
        
        if not next_nodes:
            # 葉節點，記錄完整路徑
            if len(current_path) >= 2:
                all_paths.append(current_path[:])
            return
        
        # 為每個分支遞歸
        for next_node in next_nodes:
            self._build_paths_recursive(
                next_node, 
                current_path + [next_node], 
                all_paths, 
                max_depth,
                visited_copy
            )

    def build_function_details(self) -> Dict:
        """
        產生 function_details 格式 (兼容舊版 classifier)
        
        Returns:
            {
                "function_map": {func_name: {file_path, parameters, ...}},
                "script_functions": {script_name: {functions: {...}, ...}}
            }
        """
        function_map: Dict[str, Dict] = {}
        script_functions: Dict[str, Dict] = {}
        
        for graph in self.all_graphs:
            script_name = Path(graph.file_path).stem
            
            # 初始化 script_functions 結構
            if script_name not in script_functions:
                script_functions[script_name] = {
                    "file_path": graph.file_path,
                    "functions": {},
                    "entry_points": [],
                    "export_functions": []
                }
            
            # 從 Graph 的 start node 提取參數資訊
            params = graph.start.metadata.get("parameters", [])
            docstring = graph.start.metadata.get("docstring", "")
            module_doc = graph.start.metadata.get("module_doc", "")  # ✅ 新增：從 Metadata 獲取 README 資訊
            return_type = graph.start.metadata.get("return_type", "any")
            is_class = graph.start.metadata.get("is_class", False)
            
            # 收集這個 graph 呼叫的函數
            called_functions = [
                call.metadata.get("target_func", "") 
                for call in graph.outgoing_calls
            ]
            
            # 全域 function_map
            function_map[graph.name] = {
                "file_name": Path(graph.file_path).name,
                "file_path": graph.file_path,
                "script_name": script_name,
                "line_number": 0,  # AST 不保留行號，設為 0
                "is_async": False,  # 可從 graph.type 判斷
                "is_entry_point": graph.type in ["function", "class"],
                "is_class": is_class,
                "parameters": params,
                "docstring": docstring,
                "module_doc": module_doc, # ✅ 新增
                "return_type": return_type,
                "called_functions": called_functions
            }
            
            # script_functions 內的函數資訊
            script_functions[script_name]["functions"][graph.name] = {
                "line_number": 0,
                "is_async": False,
                "is_entry_point": graph.type in ["function", "class"],
                "parameters": params,
                "module_doc": module_doc, # ✅ 新增
                "called_functions": called_functions
            }
            
            # 記錄 export_functions
            if graph.type in ["function", "class"]:
                if graph.name not in script_functions[script_name]["export_functions"]:
                    script_functions[script_name]["export_functions"].append(graph.name)
        
        return {
            "function_map": function_map,
            "script_functions": script_functions
        }

# ==========================================
# Part 5: 主程序與相容層
# ==========================================

# ==========================================
# Part 4.5: README 提取器 (ReadmeExtractor)
# ==========================================

class ReadmeExtractor:
    """
    負責提取目錄下的 README.md 內容
    """
    def __init__(self):
        self._cache = {}  # dir_path -> readme_content

    def get_readme_content(self, file_path: str) -> Optional[str]:
        """
        嘗試獲取檔案所在目錄（或上層）的 README 內容
        
        Args:
            file_path: Python 檔案路徑
            
        Returns:
            README 內容摘要 (第一段或描述)
        """
        path = Path(file_path).resolve()
        dir_path = path.parent
        
        if dir_path in self._cache:
            return self._cache[dir_path]
            
        # 嘗試在當前目錄尋找 README.md
        readme_path = dir_path / "README.md"
        content = None
        
        if readme_path.exists():
            try:
                raw_content = readme_path.read_text(encoding='utf-8')
                content = self._extract_summary(raw_content)
            except Exception as e:
                print(f"Error reading README at {readme_path}: {e}")
        else:
            # 如果當前目錄沒有，嘗試往上一層找 (由調用者決定是否需要遞歸，這裡暫時只找當前)
            # 視需求可擴展遞歸邏輯
            pass
            
        self._cache[dir_path] = content
        return content

    def _extract_summary(self, content: str) -> str:
        """提取 README 的摘要 (標題 + 第一段描述)"""
        lines = content.splitlines()
        summary = []
        capture = False
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 標題
            if line.startswith('# '):
                summary.append(line.replace('# ', '').strip())
                capture = True
                continue
                
            # 二級標題，停止抓取
            if line.startswith('## '):
                break
                
            if capture:
                summary.append(line)
        
        return " - ".join(summary[:3])  # 只取前幾行作為簡述

# ==========================================
# Part 5: 主程序與相容層
# ==========================================

def analyze_and_generate(target_dir: str, output_file: Optional[str] = "aiva_flow_analysis_v3.json") -> tuple:
    """
    執行分析並生成結果
    
    流程：
    1. Phase 1: AST 解析 - 一張張分析，暫存到 FlowStitcher
    2. Phase 2: 縫合 - 尾接頭完全匹配
    3. Phase 3: 組圖 - 從連接構建完整路徑
    4. Phase 4: 輸出 - 產生 flow_chains 和 function_details
    
    Returns:
        (output_data, stitcher) - 分析結果字典和 stitcher 物件
    """
    root_path = Path(target_dir)
    stitcher = FlowStitcher()
    readme_extractor = ReadmeExtractor() # 初始化 README 提取器
    files = list(root_path.rglob("*.py"))
    
    print(f"1. Scanning {len(files)} files in {root_path} (Included README extraction)...")
    
    # --- Phase 1: AST 解析 (一張張分析，暫存到 registry) ---
    for file_path in files:
        if "test" in file_path.name or "__init__" in file_path.name: continue
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # 獲取該檔案對應的 README 描述
            readme_desc = readme_extractor.get_readme_content(str(file_path))
            
            # 遍歷頂層節點，為每個 Function 和 Class 建立獨立的 Graph
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    builder = FlowBuilder(node.name, str(file_path), "function")
                    graph = builder.build(node)
                    
                    # 注入 README 資訊到 Metadata
                    if readme_desc:
                        graph.start.metadata["module_doc"] = readme_desc
                        
                    stitcher.register_graph(graph)
                    
                elif isinstance(node, ast.ClassDef):
                    # 1. 為 Class 本身建立 Graph (處理 Constructor)
                    class_builder = FlowBuilder(node.name, str(file_path), "class")
                    class_graph = class_builder.build(node)
                    if readme_desc:
                        class_graph.start.metadata["module_doc"] = readme_desc
                    stitcher.register_graph(class_graph)
                    
                    # 2. 為 Class 的每個方法建立 Graph
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # 方法名稱使用 Class.method 格式，避免與全域函數衝突
                            method_name = f"{node.name}.{item.name}"
                            builder = FlowBuilder(method_name, str(file_path), "method")
                            graph = builder.build(item)
                            if readme_desc:
                                graph.start.metadata["module_doc"] = readme_desc
                            stitcher.register_graph(graph)
                            
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    print(f"2. Registered {len(stitcher.all_graphs)} graphs (暫存完成). Stitching...")
    
    # --- Phase 2: 縫合 (尾接頭完全匹配) ---
    connections = stitcher.stitch()
    print(f"3. Found {len(connections)} valid connections (完全匹配).")
    
    # --- Phase 3: 組圖 (從連接構建完整路徑) ---
    flow_chains = stitcher.build_flow_chains()
    print(f"4. Built {len(flow_chains)} flow chains (組圖完成).")
    
    # --- Phase 4: 產生 function_details ---
    function_details = stitcher.build_function_details()
    print(f"5. Built function_details with {len(function_details['function_map'])} functions.")
    
    # --- Phase 5: 輸出 (兼容格式) ---
    output_data = {
        # 新格式
        "graphs": [g.to_dict() for g in stitcher.all_graphs],
        "connections": connections,
        
        # 舊格式 (兼容 classifier)
        "flow_chains": flow_chains,
        "function_details": function_details,
        
        # 摘要
        "summary": {
            "total_files": len(files),
            "total_graphs": len(stitcher.all_graphs),
            "total_connections": len(connections),
            "total_flow_chains": len(flow_chains)
        }
    }
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Done. Analysis saved to {output_file}")
    
    return output_data, stitcher


# ==========================================
# 相容層：提供與 pipeline 相容的 API
# ==========================================

class AIVAFlowAnalyzer:
    """
    相容層：為 aiva_exploration_pipeline.py 提供相容的 API
    
    內部流程：
    1. analyze_directory() - 執行 AST 分析、縫合、組圖
    2. save_results() - 輸出 flow_chains、function_details、Mermaid 流程圖
    """
    
    def __init__(self, target_dir: str = "."):
        self.target_dir = target_dir
        self.results: Optional[Dict] = None
        self.stitcher: Optional[FlowStitcher] = None
        self._output_dir: Optional[str] = None
    
    def analyze_directory(self, target: str, depth: int = 5, verbose: bool = False) -> Dict:
        """
        分析目標目錄
        
        Args:
            target: 目標路徑
            depth: 遍歷深度（v3 使用 rglob，此參數保留供兼容）
            verbose: 詳細輸出
            
        Returns:
            分析結果字典 (包含 graphs, connections, flow_chains, function_details)
        """
        self.results, self.stitcher = analyze_and_generate(target, output_file=None)
        return self.results if self.results is not None else {}
    
    def save_results(self, output_dir: str) -> None:
        """
        保存分析結果到指定目錄
        
        輸出內容：
        - analysis_results.json (完整 AST 分析結果，供後續工具直接使用)
        
        Args:
            output_dir: 輸出目錄路徑
        """
        if self.results is None:
            raise RuntimeError("請先執行 analyze_directory()")
        
        self._output_dir = output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 結果 (供 classifier 等後續工具直接使用)
        results_file = output_path / "analysis_results.json"
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 分析結果已保存至: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIVA Flow Analyzer v3")
    parser.add_argument("target_dir", help="Directory to analyze")
    parser.add_argument("--output", default="aiva_flow_analysis_v3.json", help="Output JSON file")
    
    args = parser.parse_args()
    analyze_and_generate(args.target_dir, args.output)