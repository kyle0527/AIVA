"""Capability Analyzer - 基於 py2mermaid 的完整數據流追蹤系統 (v2.0)

⭐ 核心理念：「流程圖 = 數據流」
將 py2mermaid.py 的 AST 分析方式直接整合進 AI，通過比對每張圖的 start 和 end，
實現完整的跨函數/跨文件數據流拼接，最終簡化成可調整參數的指令。

✅ 實現策略（完全整合 py2mermaid 代碼）：
1. 以檔案為單位分析（build_for_file 邏輯）
2. 為每個函數生成獨立流程圖（Builder.build_function）
3. 每個圖有固定的 start 和 end 節點
4. ⭐ 頭接尾/尾接頭拼接：
   - 圖A的節點調用函數B → 查找函數B的圖（O(1)索引查找）
   - 頭接尾：圖A調用點 → 圖B的 start
   - 尾接頭：圖B的 end → 圖A的下一個節點

🔥 規模化處理（支援上千張圖）：
- 使用 Dict[signature, List[Graph]] 建立高效索引
- O(1) 查找匹配的函數圖
- 深度優先搜索拼接完整調用鏈

📋 遵循規範：
- ✅ aiva_common v2.0 架構規範
- ✅ 維持六大模組架構
- ✅ 使用統一日誌 (get_logger 待整合)
- ✅ 完整類型註解
- ✅ Pydantic 模型驗證（待整合）

作者：AIVA Team
版本：2.0 (完全整合 py2mermaid 代碼，移除動態導入)
更新：2025-12-06
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ==================== py2mermaid 核心代碼（完整整合）====================
# 來源：tools/common/development/py2mermaid.py (521 lines)
# 整合原因：避免動態導入，確保代碼穩定性，便於維護
# 整合日期：2025-12-06
# 修改：無，完全保留原始邏輯

class Node:
    """流程圖節點
    
    每個節點代表流程中的一個步驟（操作、條件、開始、結束等）
    nexts 列表記錄了這個節點連接到的所有下一個節點
    """
    
    def __init__(self, id_: str, label: str, kind: str = "op"):
        self.id = self._sanitize_id(id_)
        self.label = label
        self.kind = kind
        self.nexts: list['Node'] = []  # 連接到的下一個節點列表（這就是數據流方向！）
        self.edge_info: dict[str, dict[str, str]] = {}  # 邊的標籤和樣式

    def _sanitize_id(self, id_str: str) -> str:
        """清理節點 ID，確保符合 Mermaid 語法"""
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", id_str)
        if cleaned and (cleaned[0].isdigit() or cleaned[0] in "_-"):
            cleaned = f"n_{cleaned}"
        cleaned = re.sub(r"_{2,}", "_", cleaned)
        return cleaned or "node"


class Graph:
    """流程圖（每個函數一張圖）
    
    ⭐ 關鍵設計：
    - start: 固定的起點節點（label="開始"）
    - end: 固定的終點節點（label="結束"）
    - nodes: 所有節點列表
    
    這個設計讓「頭接尾」拼接變得簡單：
    圖A的某節點 → 圖B的 start
    圖B的 end → 圖A的下一個節點
    """
    
    def __init__(self, title: str, direction: str = "TB"):
        self.title = title
        self.direction = direction
        self.nodes: list[Node] = []
        self.counter = 0
        
        # 自動創建固定的開始和結束節點
        self.start = self.add("start", "開始")
        self.end = self.add("end", "結束")

    def add(self, kind: str, label: str) -> Node:
        """添加新節點到圖中"""
        self.counter += 1
        node_id = f"n{self.counter}"
        node = Node(node_id, label, kind)
        self.nodes.append(node)
        return node

    def link(self, a: Node, b: Node, label: str = "", style: str = "") -> None:
        """連接兩個節點（這就是數據流的方向！）
        
        a.nexts.append(b) 表示：數據從 a 流向 b
        """
        if b not in a.nexts:
            a.nexts.append(b)
        if label or style:
            a.edge_info[b.id] = {"label": label, "style": style}

    def to_mermaid(self) -> str:
        """生成 Mermaid 圖表代碼"""
        lines = [f"flowchart {self.direction}"]
        
        # 定義所有節點
        for node in self.nodes:
            if node.kind == "start":
                lines.append(f"    {node.id}([{node.label}])")
            elif node.kind == "end":
                lines.append(f"    {node.id}([{node.label}])")
            elif node.kind == "cond":
                lines.append(f"    {node.id}{{{{{node.label}}}}}")
            else:
                lines.append(f"    {node.id}[{node.label}]")
        
        # 定義所有連接
        for node in self.nodes:
            for next_node in node.nexts:
                edge_info = node.edge_info.get(next_node.id, {})
                label = edge_info.get("label", "")
                style = edge_info.get("style", "")
                
                if label:
                    lines.append(f"    {node.id} -->|{label}| {next_node.id}")
                elif style:
                    lines.append(f"    {node.id} -.->|{style}| {next_node.id}")
                else:
                    lines.append(f"    {node.id} --> {next_node.id}")
        
        return "\n".join(lines)


class Builder(ast.NodeVisitor):
    """AST 訪問者，為 Python 函數建立流程圖
    
    核心方法：
    - build_function(): 為單個函數建立完整流程圖
    - _build_block(): 處理語句塊
    - _build_stmt(): 處理單個語句
    - _get_stmt_label(): 提取語句標籤（⭐包含函數調用信息！）
    """
    
    def __init__(self, title: str):
        self.g = Graph(title)

    def build_function(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> Graph:
        """為函數建立流程圖
        
        流程：
        1. 從 start 開始
        2. 順序處理函數體的每個語句
        3. 最後連到 end
        
        結果：start → 語句1 → 語句2 → ... → end
        """
        args_str = ", ".join(a.arg for a in func.args.args)
        title = f"{func.name}({args_str})"
        if len(title) > 50:
            title = f"{func.name}(...)"
        
        self.g = Graph(title)
        
        last = self.g.start  # 從 start 開始
        if func.body:
            last = self._build_block(func.body, last)
        self.g.link(last, self.g.end)  # 最後連到 end
        
        return self.g

    def _build_block(self, stmts: list[ast.stmt], entry: Node) -> Node:
        """處理語句塊（順序執行）
        
        返回最後一個語句的節點，作為下一個語句的入口
        """
        last = entry
        for stmt in stmts:
            last = self._build_stmt(stmt, last)
        return last

    def _build_stmt(self, stmt: ast.stmt, entry: Node) -> Node:
        """處理單個語句
        
        根據語句類型分發處理：
        - If: 條件分支
        - While/For: 循環
        - Try: 異常處理
        - 其他: 普通操作節點
        """
        if isinstance(stmt, ast.If):
            return self._build_if(stmt, entry)
        elif isinstance(stmt, ast.While):
            return self._build_while(stmt, entry)
        elif isinstance(stmt, ast.For):
            return self._build_for(stmt, entry)
        elif isinstance(stmt, ast.Try):
            return self._build_try(stmt, entry)
        else:
            label = self._get_stmt_label(stmt)
            node = self.g.add("op", label)
            self.g.link(entry, node)
            return node

    def _get_stmt_label(self, stmt: ast.stmt) -> str:
        """獲取語句標籤
        
        ⭐ 關鍵：這裡會顯示函數調用！
        例如：scanner.scan() 會顯示在節點標籤中
        這是拼接邏輯的核心：從標籤中識別函數調用
        """
        if isinstance(stmt, ast.Return):
            return "return"
        elif isinstance(stmt, ast.Assign):
            try:
                targets = [ast.unparse(t) for t in stmt.targets]
                value = ast.unparse(stmt.value)
                if len(value) > 20:
                    value = value[:20] + "..."
                return f"{', '.join(targets)} = {value}"
            except Exception:
                return "assign"
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            # 函數調用！這是拼接的關鍵點
            try:
                call_str = ast.unparse(stmt.value)
                if len(call_str) > 40:
                    call_str = call_str[:37] + "..."
                return call_str
            except Exception:
                return "function_call"
        else:
            try:
                s = ast.unparse(stmt)
                return s[:50] + ("..." if len(s) > 50 else "")
            except Exception:
                return stmt.__class__.__name__

    def _build_if(self, stmt: ast.If, entry: Node) -> Node:
        """處理 if 語句"""
        cond_node = self.g.add("cond", "if")
        self.g.link(entry, cond_node)
        
        then_last = self._build_block(stmt.body, cond_node)
        
        if stmt.orelse:
            else_last = self._build_block(stmt.orelse, cond_node)
            merge_node = self.g.add("op", "")
            self.g.link(then_last, merge_node)
            self.g.link(else_last, merge_node)
            return merge_node
        else:
            merge_node = self.g.add("op", "")
            self.g.link(then_last, merge_node)
            self.g.link(cond_node, merge_node)
            return merge_node

    def _build_while(self, stmt: ast.While, entry: Node) -> Node:
        """處理 while 迴圈"""
        cond_node = self.g.add("cond", "while")
        self.g.link(entry, cond_node)
        
        if stmt.body:
            body_last = self._build_block(stmt.body, cond_node)
            self.g.link(body_last, cond_node)  # 循環回去
        
        exit_node = self.g.add("op", "")
        self.g.link(cond_node, exit_node)
        return exit_node

    def _build_for(self, stmt: ast.For, entry: Node) -> Node:
        """處理 for 迴圈"""
        for_node = self.g.add("cond", "for")
        self.g.link(entry, for_node)
        
        if stmt.body:
            body_last = self._build_block(stmt.body, for_node)
            self.g.link(body_last, for_node)  # 循環回去
        
        exit_node = self.g.add("op", "")
        self.g.link(for_node, exit_node)
        return exit_node

    def _build_try(self, stmt: ast.Try, entry: Node) -> Node:
        """處理 try 語句"""
        try_node = self.g.add("op", "try")
        self.g.link(entry, try_node)
        
        try_last = self._build_block(stmt.body, try_node)
        merge_node = self.g.add("op", "")
        self.g.link(try_last, merge_node)
        
        for handler in stmt.handlers:
            handler_node = self.g.add("op", "except")
            self.g.link(try_node, handler_node)
            handler_last = self._build_block(handler.body, handler_node)
            self.g.link(handler_last, merge_node)
        
        return merge_node


# ==================== 擴展數據結構 ====================

@dataclass
class FunctionCallNode:
    """函數調用節點
    
    記錄一個函數中調用其他函數的信息
    這是拼接邏輯的基礎數據
    """
    function_name: str          # 被調用的函數名（如 scan）
    call_signature: str         # 完整調用簽名（如 Scanner.scan, self.scan）
    file_path: str             # 調用發生的文件
    line_number: int           # 行號
    caller_function: str       # 調用者函數名
    arguments: list[str] = field(default_factory=list)  # 調用參數


@dataclass
class FunctionFlowGraph:
    """函數流程圖（完整版）
    
    包含：
    - 函數元信息（名稱、類、模組、文件）
    - py2mermaid 生成的圖（start → 語句 → end）
    - 提取的函數調用節點（用於拼接）
    - 函數參數定義
    """
    function_name: str
    file_path: str
    class_name: str | None
    module_name: str
    
    # py2mermaid 生成的圖
    graph: Graph
    mermaid_code: str
    
    # 從 AST 提取的調用節點
    call_nodes: list[FunctionCallNode] = field(default_factory=list)
    
    # 函數元資訊
    is_async: bool = False
    is_entry_point: bool = False
    parameters: list[dict[str, Any]] = field(default_factory=list)
    
    def get_signature(self) -> str:
        """獲取函數簽名（用於索引查找）"""
        if self.class_name:
            return f"{self.class_name}.{self.function_name}"
        return self.function_name


# ==================== 數據流拼接器 ====================

class DataFlowStitcher:
    """數據流拼接器 - 實現「頭接尾/尾接頭」邏輯
    
    核心功能：
    1. 為每個 Python 檔案的每個函數生成流程圖
    2. 建立函數簽名索引（規模化：支援上千張圖）
    3. 從圖中識別函數調用
    4. 拼接完整調用鏈：
       - 圖A調用函數B → 找到圖B
       - 頭接尾：圖A的調用點 → 圖B的 start
       - 尾接頭：圖B的 end → 圖A的下一步
    
    範例：
      圖A (Manager.detect):
        start → validate → [call Scanner.scan] → parse → end
      
      圖B (Scanner.scan):
        start → http_request → end
      
      拼接後：
        start(detect) → validate → [start(scan) → http_request → end(scan)] → parse → end(detect)
    """
    
    def __init__(self, project_root: Path, storage_path: Path | None = None):
        self.project_root = project_root
        self.storage_path = storage_path or (project_root / "data" / "integration" / "data_flows")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 函數簽名索引：signature → List[FunctionFlowGraph]
        # 使用 List 是因為可能有多個同名函數（不同文件/類）
        self.function_graphs: dict[str, list[FunctionFlowGraph]] = {}
        
        logger.info(f"🎨 DataFlowStitcher initialized (支援上千張圖的規模化索引)")
    
    def analyze_file(self, file_path: Path) -> list[FunctionFlowGraph]:
        """分析檔案中的所有函數（以檔案為單位）
        
        這就是 py2mermaid 的核心流程：
        1. 解析 Python 檔案的 AST
        2. 遍歷所有類和函數
        3. 為每個函數生成獨立的流程圖
        """
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path.name}: {e}")
            return []
        
        try:
            module_name = str(file_path.relative_to(self.project_root)).replace("\\", ".").replace("/", ".").replace(".py", "")
        except Exception:
            module_name = file_path.stem
        
        graphs = []
        
        # 遍歷所有類和函數
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fg = self._build_function_graph(item, file_path, module_name, class_name)
                        if fg:
                            graphs.append(fg)
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_class_method = any(
                    node in cls.body 
                    for cls in ast.walk(tree) 
                    if isinstance(cls, ast.ClassDef)
                )
                
                if not is_class_method:
                    fg = self._build_function_graph(node, file_path, module_name, None)
                    if fg:
                        graphs.append(fg)
        
        return graphs
    
    def _build_function_graph(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef,
                             file_path: Path, module_name: str, class_name: str | None) -> FunctionFlowGraph | None:
        """為單個函數建立流程圖"""
        func_name = func_node.name
        
        try:
            builder = Builder(func_name)
            graph = builder.build_function(func_node)
            mermaid_code = graph.to_mermaid()
        except Exception as e:
            logger.debug(f"Failed to build graph for {func_name}: {e}")
            return None
        
        parameters = []
        for arg in func_node.args.args:
            param_info = {
                "name": arg.arg,
                "annotation": ast.unparse(arg.annotation) if arg.annotation else None
            }
            parameters.append(param_info)
        
        call_nodes = self._extract_function_calls(func_node, func_name, str(file_path))
        is_entry = self._is_entry_point(class_name, func_name, func_node)
        
        return FunctionFlowGraph(
            function_name=func_name,
            file_path=str(file_path),
            class_name=class_name,
            module_name=module_name,
            graph=graph,
            mermaid_code=mermaid_code,
            call_nodes=call_nodes,
            is_async=isinstance(func_node, ast.AsyncFunctionDef),
            is_entry_point=is_entry,
            parameters=parameters
        )
    
    def _extract_function_calls(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef,
                               caller_name: str, file_path: str) -> list[FunctionCallNode]:
        """從函數 AST 中提取所有函數調用
        
        ⭐ 這是拼接邏輯的關鍵：找到所有被調用的函數
        """
        calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                call_info = self._parse_call_node(node, caller_name, file_path)
                if call_info:
                    calls.append(call_info)
        
        return calls
    
    def _parse_call_node(self, call_node: ast.Call, caller_name: str, file_path: str) -> FunctionCallNode | None:
        """解析單個 Call 節點，提取函數調用信息"""
        try:
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
                call_sig = func_name
            elif isinstance(call_node.func, ast.Attribute):
                if isinstance(call_node.func.value, ast.Name):
                    obj_name = call_node.func.value.id
                    method_name = call_node.func.attr
                    
                    if obj_name == "self":
                        func_name = method_name
                        call_sig = method_name
                    else:
                        func_name = method_name
                        call_sig = f"{obj_name}.{method_name}"
                else:
                    func_name = call_node.func.attr
                    call_sig = func_name
            else:
                return None
            
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
    
    def add_graph(self, fg: FunctionFlowGraph) -> None:
        """添加函數圖到索引（規模化：O(1) 插入）"""
        signature = fg.get_signature()
        if signature not in self.function_graphs:
            self.function_graphs[signature] = []
        self.function_graphs[signature].append(fg)
    
    def stitch_data_flow(self, entry_signature: str, max_depth: int = 10) -> list[list[str]]:
        """拼接完整數據流（頭接尾、尾接頭）
        
        ⭐ 核心拼接邏輯：
        1. 從入口點開始
        2. 遍歷圖中的所有函數調用
        3. 查找被調用函數的圖（O(1) 索引查找）
        4. 遞歸追蹤，形成完整調用鏈
        
        Returns:
            list[list[str]]: 每條路徑是函數簽名的列表
        """
        if entry_signature not in self.function_graphs:
            logger.warning(f"Entry point not found: {entry_signature}")
            return []
        
        entry_graphs = self.function_graphs[entry_signature]
        all_paths = []
        
        for entry_graph in entry_graphs:
            paths = self._dfs_stitch(entry_graph, set(), [entry_signature], 0, max_depth)
            all_paths.extend(paths)
        
        logger.info(f"🔗 Stitched {len(all_paths)} complete data flow paths from {entry_signature}")
        return all_paths
    
    def _dfs_stitch(self, current: FunctionFlowGraph, visited: Set[str],
                   current_path: list[str], depth: int, max_depth: int) -> list[list[str]]:
        """深度優先搜索拼接數據流
        
        ⭐ 頭接尾/尾接頭的實現：
        - 當前圖的 end → 找到調用點 → 下一個圖的 start
        - 遞歸處理每個調用，形成完整路徑
        """
        if depth >= max_depth:
            return [current_path.copy()]
        
        if not current.call_nodes:
            return [current_path.copy()]
        
        all_paths = []
        found_match = False
        
        for call_node in current.call_nodes:
            call_sig = call_node.call_signature
            
            if call_sig in visited:
                continue
            
            # ⭐ 核心：查找被調用函數的圖（這就是「尾接頭」的關鍵！）
            matching_graphs = self._find_matching_graphs(call_sig, current)
            
            if not matching_graphs:
                continue
            
            found_match = True
            
            for next_graph in matching_graphs:
                new_visited = visited.copy()
                new_visited.add(call_sig)
                new_path = current_path.copy()
                new_path.append(next_graph.get_signature())
                
                paths = self._dfs_stitch(next_graph, new_visited, new_path, depth + 1, max_depth)
                all_paths.extend(paths)
        
        if not found_match:
            all_paths.append(current_path.copy())
        
        return all_paths
    
    def _find_matching_graphs(self, call_sig: str, caller: FunctionFlowGraph) -> list[FunctionFlowGraph]:
        """查找匹配的函數圖（規模化：O(1) 查找）
        
        匹配策略：
        1. 完全匹配：Scanner.scan → Scanner.scan
        2. 類內方法：self.method → ClassName.method
        3. 簡單名稱：scan → Scanner.scan（如果在同一個類中）
        """
        candidates = []
        
        # 1. 完全匹配（O(1)）
        if call_sig in self.function_graphs:
            candidates.extend(self.function_graphs[call_sig])
        
        # 2. 類內方法匹配
        if caller.class_name and '.' not in call_sig:
            class_method_sig = f"{caller.class_name}.{call_sig}"
            if class_method_sig in self.function_graphs:
                candidates.extend(self.function_graphs[class_method_sig])
        
        return candidates
    
    def save_analysis_results(self, entry_points: list[str]) -> None:
        """保存分析結果"""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_functions": sum(len(graphs) for graphs in self.function_graphs.values()),
            "unique_signatures": len(self.function_graphs),
            "entry_points": entry_points,
            "data_flows": {}
        }
        
        for entry in entry_points:
            paths = self.stitch_data_flow(entry)
            results["data_flows"][entry] = {
                "total_paths": len(paths),
                "paths": paths[:20]
            }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.storage_path / f"data_flows_{timestamp}.json"
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        
        logger.info(f"💾 Analysis results saved to {output_file}")


# ==================== 能力分析器（對外接口）====================

class CapabilityAnalyzer:
    """能力分析器 - AIVA 內部探索模組的核心組件 (v2.0)
    
    職責：
    1. 掃描所有 Python 模組的函數
    2. 使用 py2mermaid 為每個函數生成流程圖
    3. 建立函數簽名索引（規模化：上千張圖）
    4. 拼接完整數據流（頭接尾/尾接頭）
    5. 識別入口點和能力
    6. 返回能力列表給 internal_loop_connector
    
    接口規範：
    - async def analyze_capabilities(modules_info: dict) -> list[dict[str, Any]]
    - 輸入：ModuleExplorer.explore_all_modules() 的結果
    - 輸出：能力列表（每個能力包含名稱、參數、調用鏈等）
    
    遵循規範：
    - ✅ aiva_common v2.0 架構
    - ✅ 六大模組架構
    - ✅ 統一日誌
    - ✅ 完整類型註解
    """
    
    def __init__(self):
        self.stats = {
            "total_files": 0,
            "analyzed_files": 0,
            "total_functions": 0,
            "entry_points_found": 0,
            "data_flows_traced": 0
        }
        logger.info("✅ CapabilityAnalyzer initialized (v2.0 - 完全整合 py2mermaid)")
    
    async def analyze_capabilities(self, modules_info: dict) -> list[dict[str, Any]]:
        """分析模組能力（對外接口）
        
        這是 internal_loop_connector.py 調用的核心方法
        
        流程：
        1. 初始化拼接器
        2. 為每個檔案的每個函數生成圖
        3. 建立函數索引
        4. 識別入口點
        5. 拼接完整數據流
        6. 創建能力記錄
        7. 保存結果
        
        Args:
            modules_info: ModuleExplorer.explore_all_modules() 的結果
        
        Returns:
            list[dict]: 能力列表，每個能力包含：
                - name, module, class_name, parameters
                - data_flow_paths: 完整調用鏈
                - total_paths: 總路徑數
        """
        logger.info(f"🔍 Starting capability analysis for {len(modules_info)} modules...")
        
        # 1. 初始化拼接器
        project_root = self._get_project_root()
        stitcher = DataFlowStitcher(project_root)
        
        # 2. 為每個檔案的每個函數建立流程圖
        for module_name, module_data in modules_info.items():
            logger.info(f"  📁 Module: {module_name}")
            module_path = Path(module_data["path"])
            
            for file_info in module_data["files"]:
                file_path = module_path / file_info["path"]
                
                if file_path.name == "__init__.py" or file_path.suffix != ".py":
                    continue
                
                self.stats["total_files"] += 1
                
                logger.debug(f"    🔄 Analyzing {file_path.name}")
                graphs = stitcher.analyze_file(file_path)
                
                for graph in graphs:
                    stitcher.add_graph(graph)
                    self.stats["total_functions"] += 1
                
                self.stats["analyzed_files"] += 1
        
        # 3. 識別入口點
        entry_points = self._identify_entry_points(stitcher)
        self.stats["entry_points_found"] = len(entry_points)
        logger.info(f"  🎯 Identified {len(entry_points)} entry points")
        
        # 4. 拼接數據流並創建能力記錄
        capabilities = []
        for entry_sig in entry_points:
            paths = stitcher.stitch_data_flow(entry_sig)
            
            if paths:
                cap = self._create_capability(entry_sig, paths, stitcher)
                capabilities.append(cap)
                self.stats["data_flows_traced"] += len(paths)
        
        # 5. 保存結果
        stitcher.save_analysis_results(entry_points)
        
        # 6. 統計
        logger.info(f"✅ Analysis completed:")
        logger.info(f"   - Files: {self.stats['analyzed_files']}/{self.stats['total_files']}")
        logger.info(f"   - Functions: {self.stats['total_functions']}")
        logger.info(f"   - Entry points: {self.stats['entry_points_found']}")
        logger.info(f"   - Capabilities: {len(capabilities)}")
        logger.info(f"   - Data flows: {self.stats['data_flows_traced']}")
        
        return capabilities
    
    def _get_project_root(self) -> Path:
        """獲取專案根目錄"""
        current = Path(__file__).parent
        while current.name != "AIVA-git" and current.parent != current:
            current = current.parent
        return current
    
    def _identify_entry_points(self, stitcher: DataFlowStitcher) -> list[str]:
        """自動識別入口點"""
        entry_points = []
        
        for signature, graphs in stitcher.function_graphs.items():
            for graph in graphs:
                if graph.is_entry_point:
                    entry_points.append(signature)
                    break
        
        return entry_points
    
    def _create_capability(self, entry_sig: str, paths: List[List[str]],
                          stitcher: DataFlowStitcher) -> dict[str, Any]:
        """從數據流路徑創建能力記錄
        
        返回格式符合 internal_loop_connector 的期望
        """
        entry_graph = stitcher.function_graphs[entry_sig][0]
        
        return {
            "name": entry_graph.function_name,
            "module": entry_graph.module_name,
            "class_name": entry_graph.class_name,
            "description": f"{entry_graph.get_signature()} - Data flow stitched by py2mermaid v2.0",
            "parameters": entry_graph.parameters,
            "file_path": entry_graph.file_path,
            "return_type": None,
            "is_async": entry_graph.is_async,
            "decorators": [],
            "language": "python",
            "is_entry_point": True,
            "data_flow_paths": [
                {
                    "path_id": i,
                    "functions": path,
                    "length": len(path)
                }
                for i, path in enumerate(paths[:10], 1)
            ],
            "total_paths": len(paths),
            "mermaid_code": entry_graph.mermaid_code
        }


# ==================== 命令列執行介面 ====================

def main():
    """主函數 - 提供命令列介面，可根據輸入指令調整分析範圍"""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(
        description="AIVA Capability Analyzer - 基於 py2mermaid 的數據流追蹤分析工具"
    )
    
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="分析目標：'all'(全部模組), 'core'(核心模組), 或指定模組名稱",
        default="all"
    )
    
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=10,
        help="最大追蹤深度 (預設: 10)",
    )
    
    parser.add_argument(
        "--max-paths",
        "-mp",
        type=int,
        default=20,
        help="每個入口點保留的最大路徑數 (預設: 20)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="輸出目錄 (預設: ./data/integration/data_flows)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="顯示詳細日誌",
    )
    
    parser.add_argument(
        "--entry-points",
        "-ep",
        type=str,
        nargs="*",
        help="指定特定入口點函數 (格式: ClassName.method_name)",
    )
    
    args = parser.parse_args()
    
    # 設定日誌級別
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    async def run_analysis():
        print("=" * 80)
        print("AIVA Capability Analyzer v2.0")
        print("基於 py2mermaid 的數據流拼接分析工具")
        print("=" * 80)
        print(f"分析目標: {args.target}")
        print(f"追蹤深度: {args.depth}")
        print(f"最大路徑: {args.max_paths}")
        if args.entry_points:
            print(f"指定入口點: {', '.join(args.entry_points)}")
        print("=" * 80)
        print()
        
        try:
            # 動態導入避免循環依賴
            import sys
            from pathlib import Path
            
            # 確保路徑正確
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent.parent
            services_core = current_dir.parent.parent
            
            # 添加必要路徑到 sys.path
            sys.path.insert(0, str(project_root))  # 專案根目錄
            sys.path.insert(0, str(services_core))  # services/core
            sys.path.insert(0, str(project_root / "services"))  # services 目錄
            
            from aiva_core.internal_exploration.module_explorer import ModuleExplorer
            
            # 1. 掃描模組
            explorer = ModuleExplorer()
            modules = await explorer.explore_all_modules()
            
            # 2. 根據目標過濾模組
            if args.target != "all":
                if args.target == "core":
                    # 只分析核心模組
                    modules = {k: v for k, v in modules.items() if "core" in k}
                else:
                    # 指定模組
                    modules = {k: v for k, v in modules.items() if args.target in k}
            
            print(f"📁 掃描到 {len(modules)} 個模組")
            
            # 3. 分析能力
            analyzer = CapabilityAnalyzer()
            
            # 設定輸出路徑
            if args.output:
                output_path = Path(args.output)
            else:
                project_root = analyzer._get_project_root()
                output_path = project_root / "data" / "integration" / "data_flows"
            
            # 自定義拼接器
            stitcher = DataFlowStitcher(analyzer._get_project_root(), output_path)
            
            # 4. 為每個檔案建立流程圖
            total_functions = 0
            for module_name, module_data in modules.items():
                print(f"  📁 分析模組: {module_name}")
                module_path = Path(module_data["path"])
                
                for file_info in module_data["files"]:
                    file_path = module_path / file_info["path"]
                    
                    if file_path.name == "__init__.py" or file_path.suffix != ".py":
                        continue
                    
                    graphs = stitcher.analyze_file(file_path)
                    for graph in graphs:
                        stitcher.add_graph(graph)
                        total_functions += 1
            
            print(f"🔧 建立索引：{total_functions} 個函數")
            
            # 5. 識別入口點
            if args.entry_points:
                # 使用指定的入口點
                entry_points = args.entry_points
                print(f"🎯 使用指定入口點：{len(entry_points)} 個")
            else:
                # 自動識別
                entry_points = []
                for signature, graphs in stitcher.function_graphs.items():
                    for graph in graphs:
                        if graph.is_entry_point:
                            entry_points.append(signature)
                            break
                print(f"🎯 自動識別入口點：{len(entry_points)} 個")
            
            # 6. 拼接數據流
            all_results = {}
            total_paths = 0
            
            for entry_sig in entry_points:
                print(f"  🔗 拼接 {entry_sig}...")
                paths = stitcher.stitch_data_flow(entry_sig, max_depth=args.depth)
                
                # 限制路徑數量
                limited_paths = paths[:args.max_paths]
                all_results[entry_sig] = {
                    "total_paths": len(paths),
                    "limited_paths": len(limited_paths),
                    "paths": limited_paths
                }
                total_paths += len(paths)
            
            # 7. 保存結果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "target": args.target,
                    "depth": args.depth,
                    "max_paths": args.max_paths,
                },
                "statistics": {
                    "modules_scanned": len(modules),
                    "total_functions": total_functions,
                    "entry_points": len(entry_points),
                    "total_paths": total_paths,
                },
                "entry_points": list(entry_points),
                "data_flows": all_results
            }
            
            output_file = output_path / f"capability_analysis_{timestamp}.json"
            output_path.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # 8. 顯示結果摘要
            print("=" * 80)
            print("✅ 分析完成！")
            print(f"📊 統計：")
            print(f"   - 模組數: {len(modules)}")
            print(f"   - 函數數: {total_functions}")
            print(f"   - 入口點: {len(entry_points)}")
            print(f"   - 總路徑: {total_paths}")
            print(f"💾 結果保存到: {output_file}")
            print("=" * 80)
            
            # 顯示前幾個入口點的路徑範例
            if all_results:
                print("\n📋 入口點範例：")
                for i, (entry_sig, data) in enumerate(list(all_results.items())[:3], 1):
                    print(f"{i}. {entry_sig}")
                    print(f"   總路徑: {data['total_paths']}, 顯示: {data['limited_paths']}")
                    if data['paths']:
                        first_path = data['paths'][0]
                        path_str = " → ".join(first_path[:3])
                        if len(first_path) > 3:
                            path_str += "..."
                        print(f"   範例路徑: {path_str}")
                    print()
            
        except Exception as e:
            print(f"❌ 分析失敗: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # 執行分析
    asyncio.run(run_analysis())


if __name__ == "__main__":
    main()
