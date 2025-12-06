"""
AIVA 流程組圖分析器
==================
基於 py2mermaid 完整代碼的流程圖生成與智能組合工具

核心設計：
1. 產圖階段：移植完整 py2mermaid.py 邏輯，生成單檔案流程圖
2. 組圖階段：分析每個圖的輸入輸出介面，進行頭尾智能匹配
3. 輸出階段：生成組合後的完整數據流程圖
"""

import argparse
import ast
import re
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================================
# py2mermaid 完整代碼 (直接 Copy)
# ============================================================================

class Node:
    def __init__(self, id_: str, label: str, kind: str = "op"):
        self.id = self._sanitize_id(id_)
        self.label = label
        self.kind = kind
        self.nexts: list[Node] = []
        self.edge_info: dict[str, dict[str, str]] = {}  # Store edge labels and styles

    def _sanitize_id(self, id_str: str) -> str:
        import re

        # Mermaid node IDs should be alphanumeric with underscores/hyphens
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", id_str)
        # Ensure it starts with a letter
        if cleaned and (cleaned[0].isdigit() or cleaned[0] in "_-"):
            cleaned = f"n_{cleaned}"
        # Remove consecutive underscores
        cleaned = re.sub(r"_{2,}", "_", cleaned)
        # Ensure not empty
        return cleaned or "node"


class Graph:
    def __init__(self, title: str, direction: str = "TD"):
        self.title = title
        self.direction = self._validate_direction(direction)
        self.nodes: list[Node] = []
        self.counter = 0
        self.start = self.add("start", "開始")
        self.end = self.add("end", "結束")

    def _validate_direction(self, direction: str) -> str:
        # Valid Mermaid flowchart directions
        valid = ["TD", "TB", "BT", "RL", "LR"]
        direction = direction.upper()
        # TD is deprecated, use TB instead
        if direction == "TD":
            direction = "TB"
        return direction if direction in valid else "TB"

    def add(self, id_: str, label: str, kind: str = "op") -> Node:
        node = Node(id_, label, kind)
        self.nodes.append(node)
        return node

    def link(self, from_: Node, to: Node, label: str = "", style: str = ""):
        from_.nexts.append(to)
        if label or style:
            from_.edge_info[to.id] = {"label": label, "style": style}

    def render_mermaid(self) -> str:
        lines = [f"flowchart {self.direction}"]
        
        # Generate node definitions
        self._add_node_definitions(lines)
        
        # Generate connections
        self._add_node_connections(lines)
        
        return "\n".join(lines)
    
    def _add_node_definitions(self, lines: list[str]) -> None:
        """Add node definitions to mermaid lines"""
        for node in self.nodes:
            node_def = self._get_node_definition(node)
            lines.append(f"    {node_def}")
    
    def _get_node_definition(self, node: Node) -> str:
        """Get the mermaid definition for a node"""
        if node.kind == "condition":
            return f"{node.id}{{{node.label}}}"
        elif node.kind in ["start", "end"]:
            return f"{node.id}(({node.label}))"
        else:
            return f"{node.id}[{node.label}]"
    
    def _add_node_connections(self, lines: list[str]) -> None:
        """Add node connections to mermaid lines"""
        for node in self.nodes:
            for next_node in node.nexts:
                connection = self._get_connection_definition(node, next_node)
                lines.append(f"    {connection}")
    
    def _get_connection_definition(self, node: Node, next_node: Node) -> str:
        """Get the connection definition between two nodes"""
        if next_node.id in node.edge_info:
            edge_label = node.edge_info[next_node.id]["label"]
            if edge_label:
                return f"{node.id} -->|{edge_label}| {next_node.id}"
        return f"{node.id} --> {next_node.id}"


class Builder(ast.NodeVisitor):
    def __init__(self, graph: Graph, debug_mode: bool = False):
        self.graph = graph
        self.debug_mode = debug_mode
        self.current = graph.start
        self.break_stack: list[Node] = []
        self.continue_stack: list[Node] = []
        self.return_statements: list[Node] = []
        self.nested_level = 0

    def _debug_print(self, msg: str):
        if self.debug_mode:
            indent = "  " * self.nested_level
            print(f"DEBUG: {indent}{msg}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._debug_print(f"Processing function: {node.name}")
        
        # Create function start node
        func_start = self.graph.add(f"func_start_{node.name}", f"函數: {node.name}")
        self.graph.link(self.current, func_start)
        self.current = func_start
        
        # Process function body
        self.nested_level += 1
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                # Connect to return processing
                return_node = self.graph.add(f"return_{node.lineno}", "返回")
                self.graph.link(self.current, return_node)
                self.return_statements.append(return_node)
                self.current = return_node
                if stmt.value:
                    self.visit(stmt.value)
            else:
                self.visit(stmt)
        self.nested_level -= 1
        
        # Connect any unconnected return statements to end
        for ret_node in self.return_statements:
            self.graph.link(ret_node, self.graph.end)
        
        # If no explicit return, connect to end
        if self.current != self.graph.end and not self.return_statements:
            self.graph.link(self.current, self.graph.end)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._debug_print(f"Processing async function: {node.name}")
        
        # Create async function start node
        func_start = self.graph.add(f"async_func_start_{node.name}", f"異步函數: {node.name}")
        self.graph.link(self.current, func_start)
        self.current = func_start
        
        # Process function body (similar to regular function)
        self.nested_level += 1
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                return_node = self.graph.add(f"return_{node.lineno}", "返回")
                self.graph.link(self.current, return_node)
                self.return_statements.append(return_node)
                self.current = return_node
                if stmt.value:
                    self.visit(stmt.value)
            else:
                self.visit(stmt)
        self.nested_level -= 1
        
        # Connect return statements to end
        for ret_node in self.return_statements:
            self.graph.link(ret_node, self.graph.end)
        
        if self.current != self.graph.end and not self.return_statements:
            self.graph.link(self.current, self.graph.end)

    def visit_If(self, node: ast.If) -> Any:
        self._debug_print("Processing If statement")
        
        # Create condition node
        condition_id = f"if_condition_{node.lineno}"
        condition_text = "條件判斷"
        
        # Try to extract condition text
        try:
            condition_text = ast.unparse(node.test).strip()
        except Exception:
            condition_text = f"條件 (行 {node.lineno})"
            
        condition_node = self.graph.add(condition_id, condition_text, "condition")
        self.graph.link(self.current, condition_node)

        # Create merge node for after if-else
        merge_node = self.graph.add(f"if_merge_{node.lineno}", "合併點")

        # Process True branch
        self.current = condition_node
        if_start = self.graph.add(f"if_true_{node.lineno}", "True 分支")
        self.graph.link(condition_node, if_start, "True")
        self.current = if_start

        self.nested_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.nested_level -= 1
        
        self.graph.link(self.current, merge_node)

        # Process False branch (else/elif)
        if node.orelse:
            else_start = self.graph.add(f"if_false_{node.lineno}", "False 分支")
            self.graph.link(condition_node, else_start, "False")
            self.current = else_start

            self.nested_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.nested_level -= 1
            
            self.graph.link(self.current, merge_node)
        else:
            # No else branch, link condition directly to merge
            self.graph.link(condition_node, merge_node, "False")

        self.current = merge_node

    def visit_For(self, node: ast.For) -> Any:
        self._debug_print("Processing For loop")
        
        # Create loop start node
        loop_start = self.graph.add(f"for_start_{node.lineno}", "For 迴圈開始")
        self.graph.link(self.current, loop_start)

        # Create loop condition node
        loop_condition = self.graph.add(f"for_condition_{node.lineno}", "迴圈條件", "condition")
        self.graph.link(loop_start, loop_condition)

        # Create loop body start
        loop_body = self.graph.add(f"for_body_{node.lineno}", "迴圈主體")
        self.graph.link(loop_condition, loop_body, "True")

        # Create loop end node
        loop_end = self.graph.add(f"for_end_{node.lineno}", "迴圈結束")
        self.graph.link(loop_condition, loop_end, "False")

        # Process loop body
        self.current = loop_body
        self.break_stack.append(loop_end)
        self.continue_stack.append(loop_condition)

        self.nested_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.nested_level -= 1

        self.break_stack.pop()
        self.continue_stack.pop()

        # Link body back to condition
        self.graph.link(self.current, loop_condition)

        # Process else clause if exists
        if node.orelse:
            else_start = self.graph.add(f"for_else_{node.lineno}", "For Else")
            self.graph.link(loop_end, else_start)
            self.current = else_start
            
            self.nested_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.nested_level -= 1
            
            self.graph.link(self.current, loop_end)

        self.current = loop_end

    def visit_While(self, node: ast.While) -> Any:
        self._debug_print("Processing While loop")
        
        # Create while condition node
        while_condition = self.graph.add(f"while_condition_{node.lineno}", "While 條件", "condition")
        self.graph.link(self.current, while_condition)

        # Create while body
        while_body = self.graph.add(f"while_body_{node.lineno}", "While 主體")
        self.graph.link(while_condition, while_body, "True")

        # Create while end
        while_end = self.graph.add(f"while_end_{node.lineno}", "While 結束")
        self.graph.link(while_condition, while_end, "False")

        # Process body
        self.current = while_body
        self.break_stack.append(while_end)
        self.continue_stack.append(while_condition)

        self.nested_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.nested_level -= 1

        self.break_stack.pop()
        self.continue_stack.pop()

        # Link body back to condition
        self.graph.link(self.current, while_condition)

        # Process else clause
        if node.orelse:
            else_start = self.graph.add(f"while_else_{node.lineno}", "While Else")
            self.graph.link(while_end, else_start)
            self.current = else_start
            
            self.nested_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.nested_level -= 1

        self.current = while_end

    def visit_Break(self, node: ast.Break) -> Any:
        if self.break_stack:
            break_node = self.graph.add(f"break_{node.lineno}", "Break")
            self.graph.link(self.current, break_node)
            self.graph.link(break_node, self.break_stack[-1])
            self.current = break_node

    def visit_Continue(self, node: ast.Continue) -> Any:
        if self.continue_stack:
            continue_node = self.graph.add(f"continue_{node.lineno}", "Continue")
            self.graph.link(self.current, continue_node)
            self.graph.link(continue_node, self.continue_stack[-1])
            self.current = continue_node

    def visit_Try(self, node: ast.Try) -> Any:
        self._debug_print("Processing Try block")
        
        # Create try start node and merge node
        try_start = self.graph.add(f"try_start_{node.lineno}", "Try 開始")
        merge_node = self.graph.add(f"try_merge_{node.lineno}", "Try 合併點")
        self.graph.link(self.current, try_start)

        # Process try body
        self._process_try_body(node, try_start, merge_node)
        
        # Process exception handlers
        self._process_exception_handlers(node, try_start, merge_node)

        # Process else and finally clauses
        self._process_try_else_finally(node, try_start, merge_node)
        
        self.current = merge_node
    
    def _process_try_body(self, node: ast.Try, try_start: Node, merge_node: Node) -> None:
        """Process the try body statements"""
        self.current = try_start
        self.nested_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.nested_level -= 1
        self.graph.link(self.current, merge_node)
    
    def _process_exception_handlers(self, node: ast.Try, try_start: Node, merge_node: Node) -> None:
        """Process exception handlers"""
        for i, handler in enumerate(node.handlers):
            handler_name = self._get_exception_handler_name(handler)
            except_start = self.graph.add(f"except_{i}_{node.lineno}", f"Except: {handler_name}")
            self.graph.link(try_start, except_start, "Exception")
            
            self.current = except_start
            self.nested_level += 1
            for stmt in handler.body:
                self.visit(stmt)
            self.nested_level -= 1
            self.graph.link(self.current, merge_node)
    
    def _get_exception_handler_name(self, handler) -> str:
        """Get the name of the exception handler"""
        if not handler.type:
            return "Any"
        try:
            return ast.unparse(handler.type)
        except Exception:
            return getattr(handler.type, 'id', 'Exception')
    
    def _process_try_else_finally(self, node: ast.Try, try_start: Node, merge_node: Node) -> None:
        """Process else and finally clauses"""
        # Process else clause
        if node.orelse:
            self._process_else_clause(node, try_start, merge_node)
        
        # Process finally clause
        if node.finalbody:
            self._process_finally_clause(node, merge_node)
    
    def _process_else_clause(self, node: ast.Try, try_start: Node, merge_node: Node) -> None:
        """Process the else clause of try statement"""
        else_start = self.graph.add(f"try_else_{node.lineno}", "Try Else")
        self.graph.link(try_start, else_start, "No Exception")
        self.current = else_start
        
        self.nested_level += 1
        for stmt in node.orelse:
            self.visit(stmt)
        self.nested_level -= 1
        self.graph.link(self.current, merge_node)
    
    def _process_finally_clause(self, node: ast.Try, merge_node: Node) -> None:
        """Process the finally clause of try statement"""
        finally_start = self.graph.add(f"finally_{node.lineno}", "Finally")
        self.graph.link(merge_node, finally_start)
        self.current = finally_start
        
        self.nested_level += 1
        for stmt in node.finalbody:
            self.visit(stmt)
        self.nested_level -= 1

    def visit_With(self, node: ast.With) -> Any:
        self._debug_print("Processing With statement")
        
        # Create with start node
        with_start = self.graph.add(f"with_start_{node.lineno}", "With 開始")
        self.graph.link(self.current, with_start)
        self.current = with_start

        # Process with body
        self.nested_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.nested_level -= 1

        # Create with end node
        with_end = self.graph.add(f"with_end_{node.lineno}", "With 結束")
        self.graph.link(self.current, with_end)
        self.current = with_end

    def visit_Call(self, node: ast.Call) -> Any:
        self._debug_print("Processing function call")
        
        # Extract function name
        func_name = "函數調用"
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{node.func.attr}"
            else:
                func_name = f"?.{node.func.attr}"

        # Create function call node
        call_node = self.graph.add(f"call_{func_name}_{node.lineno}", f"調用: {func_name}")
        self.graph.link(self.current, call_node)
        self.current = call_node

        # Visit arguments
        for arg in node.args:
            self.visit(arg)

    def visit_Assign(self, node: ast.Assign) -> Any:
        self._debug_print("Processing assignment")
        
        # Create assignment node
        assign_node = self.graph.add(f"assign_{node.lineno}", f"賦值 (行 {node.lineno})")
        self.graph.link(self.current, assign_node)
        self.current = assign_node

        # Visit the value being assigned
        if node.value:
            self.visit(node.value)

    def visit_Expr(self, node: ast.Expr) -> Any:
        # For expression statements, just visit the expression
        self.visit(node.value)

    def generic_visit(self, node: ast.AST) -> Any:
        # For any other node types, continue traversing
        super().generic_visit(node)


def process_python_file(file_path: Path, debug_mode: bool = False) -> Graph:
    """
    處理單一 Python 檔案並生成流程圖
    
    Args:
        file_path: Python 檔案路徑
        debug_mode: 是否啟用除錯模式
        
    Returns:
        Graph: 生成的流程圖物件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"❌ 語法錯誤 in {file_path}: {e}")
        # Return empty graph instead of None
        return Graph(f"empty_{file_path.name}")

    # Create graph with file name as title
    graph = Graph(f"流程圖: {file_path.name}")
    builder = Builder(graph, debug_mode)

    # Process all top-level statements
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            builder.visit(node)

    return graph


# ============================================================================
# 組圖功能：頭尾拼接
# ============================================================================

@dataclass 
class ScriptNode:
    """腳本節點：表示一個Python腳本的輸入輸出"""
    script_path: str
    script_name: str
    
    # 頭：腳本的輸入端（入口函數）
    entry_points: list[str] = field(default_factory=list)
    
    # 尾：腳本的輸出端（被其他腳本調用的函數）
    export_functions: list[str] = field(default_factory=list)
    
    # 依賴：這個腳本import了哪些其他腳本
    import_dependencies: list[str] = field(default_factory=list)
    
    # 調用：這個腳本調用了哪些外部函數
    external_calls: list[tuple[str, str]] = field(default_factory=list)  # (函數名, 模組名)


class DataFlowStitcher:
    """真正的數據流拼接器 - 基於腳本間的頭尾拼接"""
    
    def __init__(self):
        self.script_nodes: dict[str, ScriptNode] = {}
        self.real_connections: list[tuple[str, str]] = []  # (from_script, to_script)
        
    def add_script(self, script_path: str, graph):
        """添加腳本分析結果"""
        script_name = Path(script_path).stem
        
        node = ScriptNode(
            script_path=script_path,
            script_name=script_name
        )
        
        # 分析腳本的頭尾
        self._analyze_script_head_tail(node, graph, script_path)
        
        self.script_nodes[script_path] = node
    
    def _analyze_script_head_tail(self, node: ScriptNode, graph, script_path: str):
        """分析腳本的頭尾特徵"""
        
        # 讀取源代碼分析import和函數定義
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            tree = ast.parse(source_code)
        except Exception:
            return
        
        # 分析import關係
        for node_ast in ast.walk(tree):
            if isinstance(node_ast, ast.Import):
                for alias in node_ast.names:
                    node.import_dependencies.append(alias.name)
            elif isinstance(node_ast, ast.ImportFrom):
                if node_ast.module:
                    node.import_dependencies.append(node_ast.module)
        
        # 分析函數定義和調用
        defined_functions = []
        called_functions = []
        
        for node_ast in ast.walk(tree):
            if isinstance(node_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_functions.append(node_ast.name)
            elif isinstance(node_ast, ast.Call):
                if isinstance(node_ast.func, ast.Name):
                    called_functions.append(node_ast.func.id)
                elif isinstance(node_ast.func, ast.Attribute):
                    if isinstance(node_ast.func.value, ast.Name):
                        # module.function 形式的調用
                        module_name = node_ast.func.value.id
                        func_name = node_ast.func.attr
                        node.external_calls.append((func_name, module_name))
        
        # 確定頭（入口點）：有main或沒有被內部調用的函數
        node.entry_points = [f for f in defined_functions if f not in called_functions or f in ['main', 'run', 'execute']]
        
        # 確定尾（輸出端）：所有定義的函數都可能被外部調用
        node.export_functions = defined_functions[:]
    
    def find_real_connections(self):
        """找到真實的腳本間連接"""
        self.real_connections.clear()
        
        for script_path, script_node in self.script_nodes.items():
            # 檢查這個腳本的每個外部調用
            for func_name, module_name in script_node.external_calls:
                # 找到提供這個函數的腳本
                provider_script = self._find_function_provider(func_name, module_name)
                if provider_script and provider_script != script_path:
                    # 發現真實連接：provider_script(數據源) → script_path(數據消費者)
                    connection = (provider_script, script_path)
                    if connection not in self.real_connections:
                        self.real_connections.append(connection)
        
        if len(self.real_connections) == 0:
            print("⚠️ 未找到任何真實的腳本間連接")
        else:
            print(f"🔗 找到 {len(self.real_connections)} 個真實連接")
    
    def _find_function_provider(self, func_name: str, module_hint: str) -> Optional[str]:
        """找到提供某個函數的腳本"""
        # 首先嘗試精確匹配模組名
        for script_path, script_node in self.script_nodes.items():
            if module_hint in script_node.script_name or script_node.script_name in module_hint:
                if func_name in script_node.export_functions:
                    return script_path
        
        # 模糊匹配：在所有腳本中找這個函數
        for script_path, script_node in self.script_nodes.items():
            if func_name in script_node.export_functions:
                return script_path
        
        return None
    
    def build_data_flow_chains(self, max_depth: int = 5) -> list[list[str]]:
        """構建數據流鏈路：支援分支圖而非強制鏈式"""
        chains = []
        
        # 首先確保找到真實連接
        self.find_real_connections()
        
        if not self.real_connections:
            return chains
        
        # 找到所有頭節點（數據流起點）
        head_scripts = self._find_head_scripts()
        
        print(f"🎯 找到 {len(head_scripts)} 個數據源頭節點腳本")
        
        for head_script in head_scripts:
            # 從每個頭節點開始構建所有可能的分支路徑
            self._build_all_paths_from_head(head_script, [head_script], chains, max_depth)
        
        # 去重和排序 - 保留所有有意義的路徑(包含分支)
        unique_chains = []
        for chain in chains:
            if len(chain) >= 2 and chain not in unique_chains:  
                unique_chains.append(chain)
        
        print(f"📊 生成 {len(unique_chains)} 條獨立數據流路徑(含分支)")
        return unique_chains
    
    def _find_head_scripts(self) -> list[str]:
        """找到頭節點：數據流起點(被最多模組導入的核心服務)"""
        all_sources = {source for source, _ in self.real_connections}
        all_targets = {target for _, target in self.real_connections}
        
        # 統計每個腳本作為數據提供者的次數
        provider_counts = {}
        for source, _ in self.real_connections:
            provider_counts[source] = provider_counts.get(source, 0) + 1
        
        # 頭節點策略：
        # 1. 只提供數據不消費數據的腳本(純數據源)
        pure_providers = all_sources - all_targets
        
        # 2. 如果沒有純提供者，選擇提供次數最多的核心模組
        if not pure_providers:
            # 選擇被導入次數最多的前幾個模組作為頭節點
            sorted_providers = sorted(provider_counts.items(), key=lambda x: x[1], reverse=True)
            head_scripts = [script for script, count in sorted_providers[:5] if count > 1]
        else:
            head_scripts = list(pure_providers)
        
        return head_scripts
    
    def _build_all_paths_from_head(self, current_script: str, current_path: list[str], 
                                 all_paths: list[list[str]], max_depth: int):
        """從頭節點遞歸構建所有分支路徑 - 記錄每個有意義的路徑"""
        if len(current_path) >= max_depth:
            return
        
        # 找到當前腳本連接到的下一個腳本(避免循環)
        next_scripts = [target for source, target in self.real_connections 
                       if source == current_script and target not in current_path]
        
        if not next_scripts:
            # 到達葉節點，記錄完整路徑
            if len(current_path) >= 2:
                all_paths.append(current_path[:])
            return
        
        # 如果有分支，為每個分支生成獨立路徑
        for next_script in next_scripts:
            new_path = current_path + [next_script]
            # 記錄這個階段的路徑(代表一個數據流分支)
            if len(new_path) >= 2:
                all_paths.append(new_path[:])
            
            # 繼續構建更深層的路徑
            self._build_all_paths_from_head(next_script, new_path, all_paths, max_depth)
    def analyze_branch_patterns(self) -> dict[str, Any]:
        """分析數據流的分支模式"""
        if not self.real_connections:
            return {}
        
        # 統計分支模式
        outgoing_counts = {}  # 每個節點的出度(分支數)
        incoming_counts = {}  # 每個節點的入度(匯聚數)
        
        for source, target in self.real_connections:
            outgoing_counts[source] = outgoing_counts.get(source, 0) + 1
            incoming_counts[target] = incoming_counts.get(target, 0) + 1
        
        # 識別關鍵節點類型
        fan_out_nodes = {k: v for k, v in outgoing_counts.items() if v > 3}  # 扇出節點
        fan_in_nodes = {k: v for k, v in incoming_counts.items() if v > 3}   # 扇入節點
        
        return {
            'fan_out_nodes': fan_out_nodes,
            'fan_in_nodes': fan_in_nodes,
            'total_connections': len(self.real_connections),
            'unique_sources': len({s for s, _ in self.real_connections}),
            'unique_targets': len({t for _, t in self.real_connections})
        }

@dataclass
class FlowGraph:
    """單一函數流程圖"""
    function_name: str
    file_path: str
    graph: Graph
    # 關鍵改進：提取實際的函數調用和定義
    defined_functions: list[str] = field(default_factory=list)  # 此圖定義的函數
    called_functions: list[str] = field(default_factory=list)   # 此圖調用的函數
    entry_signatures: list[str] = field(default_factory=list)   # 可當入口的函數簽名
    exit_signatures: list[str] = field(default_factory=list)    # 可當出口的函數簽名


class SmartFlowStitcher:
    """智能流程圖拼接器 - 支援跨圖搜索和動態函數匹配"""
    
    def __init__(self):
        self.graphs: dict[str, FlowGraph] = {}  # file_path -> FlowGraph
        self.function_index: dict[str, list[str]] = defaultdict(list)  # func_name -> [file_paths]
        self.call_index: dict[str, list[str]] = defaultdict(list)     # called_func -> [caller_files]
        
    def add_graph(self, flow_graph: FlowGraph):
        """添加流程圖並建立索引"""
        file_key = flow_graph.file_path
        self.graphs[file_key] = flow_graph
        
        # 建立函數定義索引
        for func_def in flow_graph.defined_functions:
            self.function_index[func_def].append(file_key)
        
        # 建立函數調用索引
        for func_call in flow_graph.called_functions:
            self.call_index[func_call].append(file_key)
    
    def find_stitchable_graphs(self, entry_func: str, max_depth: int = 5) -> list[list[str]]:
        """
        智能搜索可拼接的圖序列
        
        Args:
            entry_func: 入口函數名
            max_depth: 最大搜索深度
            
        Returns:
            可拼接的圖檔案路徑序列列表
        """
        stitched_sequences = []
        
        # 找到定義入口函數的所有圖
        entry_graphs = self.function_index.get(entry_func, [])
        
        for entry_graph in entry_graphs:
            # 從每個入口圖開始深度優先搜索
            self._dfs_search_stitchable(entry_graph, [entry_graph], 
                                      stitched_sequences, 0, max_depth)
        
        return stitched_sequences
    
    def _dfs_search_stitchable(self, current_graph: str, current_path: list[str], 
                              all_sequences: list[list[str]], depth: int, max_depth: int):
        """深度優先搜索可拼接的圖"""
        if depth >= max_depth or len(current_path) > 10:  # 避免無限遞歸
            return
            
        if current_graph not in self.graphs:
            return
            
        current_flow = self.graphs[current_graph]
        
        # 檢查當前圖調用的函數
        found_connections = False
        for called_func in current_flow.called_functions:
            # 找到定義這個函數的圖
            target_graphs = self.function_index.get(called_func, [])
            
            for target_graph in target_graphs:
                if target_graph not in current_path:  # 避免循環
                    new_path = current_path + [target_graph]
                    
                    # 記錄找到的序列
                    if len(new_path) > 1:
                        all_sequences.append(new_path[:])
                        found_connections = True
                    
                    # 繼續搜索
                    self._dfs_search_stitchable(target_graph, new_path, 
                                               all_sequences, depth + 1, max_depth)
        
        # 如果沒有找到連接，也記錄單圖序列
        if not found_connections and len(current_path) == 1:
            all_sequences.append(current_path[:])
    
    def generate_stitched_mermaid(self, graph_sequence: list[str]) -> str:
        """根據圖序列生成拼接後的 Mermaid 圖"""
        if not graph_sequence:
            return "flowchart TD\n    start((開始))\n    end((結束))\n    start --> end"
        
        lines = ["flowchart TD"]
        
        # 為每個圖在序列中創建節點
        for i, graph_file in enumerate(graph_sequence):
            if graph_file not in self.graphs:
                continue
                
            flow_graph = self.graphs[graph_file]
            
            # 創建圖的代表節點
            graph_name = Path(graph_file).stem
            node_id = f"graph_{i}_{graph_name}"
            
            # 顯示圖中定義的主要函數
            main_functions = flow_graph.defined_functions[:3]  # 限制顯示數量
            if main_functions:
                func_list = ", ".join(main_functions)
                lines.append(f"    {node_id}[{graph_name}\\n{func_list}]")
            else:
                lines.append(f"    {node_id}[{graph_name}]")
            
            # 連接到下一個圖
            if i < len(graph_sequence) - 1:
                next_graph_file = graph_sequence[i + 1]
                next_graph_name = Path(next_graph_file).stem
                next_node_id = f"graph_{i+1}_{next_graph_name}"
                
                # 找到連接的函數調用
                connection_func = self._find_connection_function(graph_file, next_graph_file)
                if connection_func:
                    lines.append(f"    {node_id} -->|{connection_func}| {next_node_id}")
                else:
                    lines.append(f"    {node_id} --> {next_node_id}")
        
        return "\n".join(lines)
    
    def _find_connection_function(self, from_graph: str, to_graph: str) -> str:
        """找到兩個圖之間的連接函數"""
        if from_graph not in self.graphs or to_graph not in self.graphs:
            return ""
            
        from_flow = self.graphs[from_graph]
        to_flow = self.graphs[to_graph]
        
        # 找到 from_graph 調用的函數中，在 to_graph 中定義的函數
        for called_func in from_flow.called_functions:
            if called_func in to_flow.defined_functions:
                return called_func
                
        return ""


# ============================================================================
# AIVA 整合功能
# ============================================================================

class AIVAFlowAnalyzer:
    """AIVA 流程分析器主類"""
    
    def __init__(self, target_dir: Optional[str] = None):
        # 使用 services 目錄作為預設掃描根目錄
        default_target = Path(__file__).parent.parent.parent.parent
        self.target_dir = Path(target_dir) if target_dir else default_target
        self.stitcher = DataFlowStitcher()  # 使用新的數據流拼接器
        self.results = {}
        
    def _get_target_directory(self, target: str) -> Path:
        """根據 target 參數決定掃描目錄"""
        if target == "all":
            return self.target_dir
        elif target == "core":
            return self.target_dir / "services" / "core"
        elif target.startswith("/") or target.startswith("C:"):
            # 絕對路徑
            return Path(target)
        else:
            # 相對路徑，在專案中尋找
            possible_paths = [
                self.target_dir / target,
                self.target_dir / "services" / target,
                self.target_dir / "services" / "core" / target,
                self.target_dir / "api" / target,
                self.target_dir / "plugins" / target
            ]
            for path in possible_paths:
                if path.exists():
                    return path
            # 如果都找不到，返回預設路徑
            return self.target_dir / target
        
    def analyze_directory(self, target: str = "core", depth: int = 3, 
                               max_paths: int = 20, verbose: bool = False) -> dict:
        """
        分析指定目錄的 Python 檔案
        
        Args:
            target: 分析目標 (all/core/module_name)
            depth: 追蹤深度
            max_paths: 最大路徑數
            verbose: 詳細輸出
            
        Returns:
            分析結果字典
        """
        if verbose:
            print(f"🔍 開始分析 {target} 目標...")
        
        # 決定掃描範圍
        scan_dir = self._get_target_directory(target)
        if verbose:
            print(f"📂 目標範圍: {scan_dir}")
        
        if not scan_dir.exists():
            print(f"❌ 目標目錄不存在: {scan_dir}")
            return {}
        
        # 1. 產圖階段：掃描並分析所有 Python 檔案
        python_files = list(scan_dir.rglob("*.py"))
        total_files = len(python_files)
        
        if verbose:
            print(f"📁 發現 {total_files} 個 Python 檔案")
        
        processed = 0
        for py_file in python_files:
            try:
                if verbose and processed % 10 == 0:
                    print(f"📊 處理進度: {processed}/{total_files}")
                
                # 使用 py2mermaid 處理檔案
                graph = process_python_file(py_file, debug_mode=verbose)
                if graph:
                    # 將腳本添加到數據流拼接器
                    self.stitcher.add_script(str(py_file), graph)
                    processed += 1
                    
            except Exception as e:
                if verbose:
                    print(f"❌ 處理檔案 {py_file} 時出錯: {e}")
                continue
        
        if verbose:
            print(f"✅ 產圖階段完成，共處理 {processed} 個檔案")
        
        # 2. 組圖階段：修正後的數據流鏈路分析
        if verbose:
            print("🔗 開始修正後的數據流鏈路分析...")
            print("   💡 修正內容: 正確的頭尾判斷 + 分支路徑支援")
        
        # 分析分支模式
        branch_info = self.stitcher.analyze_branch_patterns()
        
        # 構建修正後的數據流鏈路(含分支)  
        flow_chains = self.stitcher.build_data_flow_chains()
        
        combined_results = {}
        if flow_chains:
            if verbose:
                print(f"🎯 找到 {len(flow_chains)} 條數據流路徑(含分支)")
                print(f"📊 分支統計: 扇出節點 {len(branch_info.get('fan_out_nodes', {}))} 個, 扇入節點 {len(branch_info.get('fan_in_nodes', {}))} 個")
            
            # 為每條鏈路創建結果
            for i, chain in enumerate(flow_chains, 1):
                chain_name = f"data_flow_chain_{i}"
                chain_scripts = [Path(script).stem for script in chain]
                
                # 生成簡化的 Mermaid 圖
                mermaid_diagram = self._generate_chain_mermaid(chain)
                
                combined_results[chain_name] = {
                    "sequences": [{
                        "sequence": chain_scripts,
                        "mermaid": mermaid_diagram,
                        "file_paths": chain
                    }],
                    "total_sequences": 1
                }
        else:
            if verbose:
                print("⚠️ 未發現真實的跨腳本數據流鏈路")
        
        if verbose:
            total_chains = len(flow_chains)
            print(f"✅ 真實數據流分析完成，共找到 {total_chains} 條鏈路")
        
        self.results = {
            "target": target,
            "depth": depth,
            "total_files_processed": processed,
            "flow_chains": flow_chains,
            "combined_flows": combined_results,
            "summary": {
                "total_graphs": len(self.stitcher.script_nodes),
                "total_flow_chains": len(flow_chains),
                "real_connections": len(self.stitcher.real_connections),
                "script_nodes": len(self.stitcher.script_nodes)
            }
        }
        
        return self.results
    
    def _generate_chain_mermaid(self, chain: list[str]) -> str:
        """為數據流鏈路生成 Mermaid 圖"""
        mermaid = "graph TD\n"
        
        # 為每個腳本創建節點
        for i, script_path in enumerate(chain):
            script_name = Path(script_path).stem
            mermaid += f"    {script_name}[{script_name}]\n"
        
        # 添加連接
        for i in range(len(chain) - 1):
            from_script = Path(chain[i]).stem
            to_script = Path(chain[i + 1]).stem
            mermaid += f"    {from_script} --> {to_script}\n"
        
        return mermaid
        
        return self.results
    
    def _extract_functions_from_graph(self, graph: Graph) -> tuple[list[str], list[str]]:
        """
        從圖中提取函數定義和調用
        
        Returns:
            (defined_functions, called_functions)
        """
        defined_functions = []
        called_functions = []
        
        for node in graph.nodes:
            # 提取函數定義（通常是函數開始節點）
            if "函數:" in node.label or "異步函數:" in node.label:
                func_name = node.label.split(":", 1)[1].strip()
                defined_functions.append(func_name)
            
            # 提取函數調用
            elif "調用:" in node.label:
                func_name = node.label.replace("調用: ", "").strip()
                # 清理函數名稱
                if func_name and not func_name.startswith(("print", "len", "str", "int", "float")):
                    called_functions.append(func_name)
        
        return defined_functions, called_functions
    
    def _find_meaningful_entry_functions(self) -> list[str]:
        """找出有意義的入口函數（基於腳本節點分析）"""
        meaningful_entries = []
        
        # 從DataFlowStitcher的腳本節點中提取入口函數
        for script_path, script_node in self.stitcher.script_nodes.items():
            # 取每個腳本的入口點
            for entry_point in script_node.entry_points:
                if entry_point and entry_point not in meaningful_entries:
                    meaningful_entries.append(entry_point)
        
        # 如果沒有找到入口點，選擇一些常見的函數名作為起點
        if not meaningful_entries:
            common_entries = ['main', 'run', 'execute', 'start', 'process', 'handle']
            
            # 從所有腳本中查找這些常見入口函數
            for script_path, script_node in self.stitcher.script_nodes.items():
                for func in script_node.export_functions:
                    if any(entry in func.lower() for entry in common_entries):
                        if func not in meaningful_entries:
                            meaningful_entries.append(func)
        
        # 限制返回數量避免過多的序列
        return meaningful_entries[:10]
    
    def save_results(self, output_dir: str = "aiva_flow_analysis") -> None:
        """保存分析結果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 結果
        with open(output_path / "analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存每條數據流鏈路的 Mermaid 圖
        for chain_name, result in self.results.get("combined_flows", {}).items():
            for i, sequence_data in enumerate(result.get("sequences", [])):
                sequence_name = "_".join(sequence_data["sequence"][:3])  # 前3個圖名
                mermaid_file = output_path / f"{chain_name}_{i}_{sequence_name}_flow.md"
                
                with open(mermaid_file, "w", encoding="utf-8") as f:
                    f.write(f"# {chain_name} 數據流鏈路 {i+1}\n\n")
                    f.write("```mermaid\n")
                    f.write(sequence_data["mermaid"])
                    f.write("\n```\n\n")
                    f.write("## 腳本序列\n\n")
                    for j, script_name in enumerate(sequence_data["sequence"], 1):
                        f.write(f"{j}. {script_name}\n")
                    f.write("\n## 檔案路徑\n\n")
                    for path in sequence_data["file_paths"]:
                        f.write(f"- {path}\n")
        
        # 生成摘要報告
        summary_file = output_path / "data_flow_summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("# AIVA 數據流分析摘要\n\n")
            f.write(f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            summary = self.results.get("summary", {})
            f.write("## 統計概況\n\n")
            f.write(f"- 處理文件數：{self.results.get('total_files_processed', 0)}\n")
            f.write(f"- 腳本節點數：{summary.get('script_nodes', 0)}\n") 
            f.write(f"- 真實連接數：{summary.get('real_connections', 0)}\n")
            f.write(f"- 數據流鏈路：{summary.get('total_flow_chains', 0)}\n\n")
            
            # 真實連接列表
            if self.stitcher.real_connections:
                f.write("## 真實腳本連接\n\n")
                for from_script, to_script in self.stitcher.real_connections:
                    from_name = Path(from_script).stem
                    to_name = Path(to_script).stem
                    f.write(f"- `{from_name}` → `{to_name}`\n")
        
        print(f"📁 結果已保存到: {output_path}")


# ============================================================================
# CLI 介面
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AIVA 流程分析器 - 基於 py2mermaid 的數據流追蹤與組圖工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--target", "-t", default="core",
                       help="分析目標：'all'(全部), 'core'(核心), 或指定目錄名稱")
    parser.add_argument("--depth", "-d", type=int, default=3,
                       help="最大追蹤深度 (預設: 3)")
    parser.add_argument("--max-paths", "-mp", type=int, default=10,
                       help="每個入口點的最大路徑數 (預設: 10)")
    parser.add_argument("--output", "-o", default="./aiva_flow_analysis",
                       help="輸出目錄 (預設: ./aiva_flow_analysis)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="顯示詳細處理過程")
    
    args = parser.parse_args()
    
    print("="*50)
    print("AIVA 流程分析器 v1.0")
    print("產圖 + 組圖 一體化工具")
    print("="*50)
    print(f"分析目標: {args.target}")
    print(f"追蹤深度: {args.depth}")
    print(f"最大路徑: {args.max_paths}")
    print("="*50)
    
    def run_analysis():
        try:
            analyzer = AIVAFlowAnalyzer()
            results = analyzer.analyze_directory(
                target=args.target,
                depth=args.depth,
                max_paths=args.max_paths,
                verbose=args.verbose
            )
            
            # 保存結果
            analyzer.save_results(args.output)
            
            # 顯示摘要
            summary = results["summary"]
            print("\n📊 數據流分析完成摘要:")
            print(f"   - 處理檔案數: {results['total_files_processed']}")
            print(f"   - 腳本節點數: {summary['script_nodes']}")
            print(f"   - 真實連接數: {summary['real_connections']}")
            print(f"   - 數據流鏈路: {summary['total_flow_chains']}")
            print(f"   - 鏈路條數: {len(results.get('flow_chains', []))}")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析失敗: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    # 執行分析
    success = run_analysis()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()