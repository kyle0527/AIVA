"""
py2mermaid.py
-------------
Python AST 解析與 Mermaid 流程圖產生
"""

import ast
from pathlib import Path
from typing import Any


class Node:
    def __init__(self, id_: str, label: str, kind: str = "op"):
        self.id = self._sanitize_id(id_)
        self.label = label
        self.kind = kind
        self.nexts: list[Node] = []

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

    def add(self, kind: str, label: str) -> Node:
        self.counter += 1
        node_id = f"n{self.counter}"
        node = Node(node_id, label, kind)
        self.nodes.append(node)
        return node

    def link(self, a: Node, b: Node, label: str = "", style: str = "-->"):
        if b not in a.nexts:
            a.nexts.append(b)

    def to_mermaid(self) -> str:
        lines = [f"flowchart {self.direction}"]

        def sanitize_text(text: str) -> str:
            """Properly escape text for Mermaid syntax"""
            # Basic character escaping for Mermaid
            text = (
                text.replace('"', "&quot;")
                .replace("'", "&#39;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("&", "&amp;")
                .replace("#", "&#35;")
                .replace("|", "&#124;")
                .replace("(", "&#40;")
                .replace(")", "&#41;")
                .replace("[", "&#91;")
                .replace("]", "&#93;")
                .replace("{", "&#123;")
                .replace("}", "&#125;")
            )

            # Handle newlines and tabs
            text = text.replace("\n", "<br/>").replace("\t", "    ")

            # Limit length for readability
            if len(text) > 60:
                text = text[:57] + "..."

            return text

        def fmt(n: Node) -> str:
            text = sanitize_text(n.label)

            if n.kind == "cond":
                # Diamond shape for conditionals
                return f"{n.id}{{{text}}}"
            elif n.kind in ("end", "start"):
                # Stadium/pill shape for start/end
                return f"{n.id}([{text}])"
            elif n.kind == "subgraph":
                # Trapezoid shape for subgraphs
                return f"{n.id}[/{text}/]"
            else:
                # Rectangle shape for operations
                return f"{n.id}[{text}]"

        # Add node definitions
        for n in self.nodes:
            lines.append(f"    {fmt(n)}")

        # Add connections
        for n in self.nodes:
            for i, m in enumerate(n.nexts):
                if n.kind == "cond":
                    if i == 0:
                        lines.append(f"    {n.id} -->|Yes| {m.id}")
                    elif i == 1:
                        lines.append(f"    {n.id} -->|No| {m.id}")
                    else:
                        lines.append(f"    {n.id} --> {m.id}")
                else:
                    lines.append(f"    {n.id} --> {m.id}")

        return "\n".join(lines)


class Builder(ast.NodeVisitor):
    def __init__(self, title: str, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.g = Graph(title, self.config.get("direction", "TD"))

    def build_module(self, node: ast.Module) -> Graph:
        last = self.g.start
        if node.body:
            last = self._build_block(node.body, last)
        self.g.link(last, self.g.end)
        return self.g

    def build_function(self, func: ast.FunctionDef) -> Graph:
        args_str = ", ".join(a.arg for a in func.args.args)
        title = f"{func.name}({args_str})"
        if len(title) > 50:
            title = f"{func.name}(...)"
        self.g = Graph(title, self.config.get("direction", "TD"))
        last = self.g.start
        if func.body:
            last = self._build_block(func.body, last)
        self.g.link(last, self.g.end)
        return self.g

    def _build_block(self, stmts: list[ast.stmt], entry: Node) -> Node:
        last = entry
        for stmt in stmts:
            last = self._build_stmt(stmt, last)
        return last

    def _build_stmt(self, stmt: ast.stmt, entry: Node) -> Node:
        if isinstance(stmt, ast.If):
            return self._build_if(stmt, entry)
        elif isinstance(stmt, ast.While):
            return self._build_while(stmt, entry)
        elif isinstance(stmt, ast.For):
            return self._build_for(stmt, entry)
        elif isinstance(stmt, ast.Try):
            return self._build_try(stmt, entry)
        elif isinstance(stmt, ast.With):
            return self._build_with(stmt, entry)
        else:
            label = self._get_stmt_label(stmt)
            node = self.g.add("op", label)
            self.g.link(entry, node)
            return node

    def _get_stmt_label(self, stmt: ast.stmt) -> str:
        if isinstance(stmt, ast.Return):
            value = getattr(stmt, "value", None)
            if value is not None:
                try:
                    val = ast.unparse(value)
                except Exception:
                    val = "value"
                return f"return {val[:30]}..." if len(val) > 30 else f"return {val}"
            return "return"
        elif isinstance(stmt, ast.Assign):
            try:
                targets = [ast.unparse(t) for t in stmt.targets]
                value = ast.unparse(stmt.value)
            except Exception:
                return "assign"
            if len(value) > 20:
                value = value[:20] + "..."
            return f"{', '.join(targets)} = {value}"
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            return f"def {stmt.name}(...)"
        elif isinstance(stmt, ast.ClassDef):
            return f"class {stmt.name}(...)"
        else:
            try:
                s = ast.unparse(stmt)
                return s[:50] + ("..." if len(s) > 50 else "")
            except Exception:
                return stmt.__class__.__name__

    def _build_if(self, stmt: ast.If, entry: Node) -> Node:
        try:
            cond_text = ast.unparse(stmt.test)
            if len(cond_text) > 30:
                cond_text = cond_text[:27] + "..."
        except Exception:
            cond_text = "condition"

        cond_node = self.g.add("cond", f"if {cond_text}")
        self.g.link(entry, cond_node)

        # Build the 'then' branch
        then_last = self._build_block(stmt.body, cond_node)

        if stmt.orelse:
            # Build the 'else' branch
            else_last = self._build_block(stmt.orelse, cond_node)
            # Create merge point
            merge_node = self.g.add("op", "")
            self.g.link(then_last, merge_node)
            self.g.link(else_last, merge_node)
            return merge_node
        else:
            # For if without else, create a merge node for better flow
            merge_node = self.g.add("op", "")
            self.g.link(then_last, merge_node)
            self.g.link(cond_node, merge_node)  # Direct path for "No" condition
            return merge_node

    def _build_while(self, stmt: ast.While, entry: Node) -> Node:
        try:
            cond_text = ast.unparse(stmt.test)
            if len(cond_text) > 25:
                cond_text = cond_text[:22] + "..."
            cond_text = f"while {cond_text}"
        except Exception:
            cond_text = "while condition"

        cond_node = self.g.add("cond", cond_text)
        self.g.link(entry, cond_node)

        if stmt.body:
            body_last = self._build_block(stmt.body, cond_node)
            self.g.link(body_last, cond_node)  # Loop back

        # Create exit node for when condition is false
        exit_node = self.g.add("op", "")
        self.g.link(cond_node, exit_node)
        return exit_node

    def _build_for(self, stmt: ast.For, entry: Node) -> Node:
        try:
            target = ast.unparse(stmt.target)
            iter_expr = ast.unparse(stmt.iter)
            if len(iter_expr) > 20:
                iter_expr = iter_expr[:17] + "..."
            label = f"for {target} in {iter_expr}"
        except Exception:
            label = "for loop"

        for_node = self.g.add("cond", label)
        self.g.link(entry, for_node)

        if stmt.body:
            body_last = self._build_block(stmt.body, for_node)
            self.g.link(body_last, for_node)  # Loop back

        # Create exit node for when loop is done
        exit_node = self.g.add("op", "")
        self.g.link(for_node, exit_node)
        return exit_node

    def _build_try(self, stmt: ast.Try, entry: Node) -> Node:
        try_node = self.g.add("op", "try")
        self.g.link(entry, try_node)

        try_last = self._build_block(stmt.body, try_node)
        merge_node = self.g.add("op", "")
        self.g.link(try_last, merge_node)

        # Handle exception handlers
        for handler in stmt.handlers:
            exc_type = handler.type
            try:
                exc_name = ast.unparse(exc_type) if exc_type else "Exception"
                if len(exc_name) > 15:
                    exc_name = exc_name[:12] + "..."
            except Exception:
                exc_name = "Exception"

            handler_node = self.g.add("op", f"except {exc_name}")
            self.g.link(try_node, handler_node)
            handler_last = self._build_block(handler.body, handler_node)
            self.g.link(handler_last, merge_node)

        # Handle finally block
        if stmt.finalbody:
            finally_node = self.g.add("op", "finally")
            self.g.link(merge_node, finally_node)
            finally_last = self._build_block(stmt.finalbody, finally_node)
            return finally_last

        return merge_node

    def _build_with(self, stmt: ast.With, entry: Node) -> Node:
        items = []
        for item in stmt.items:
            try:
                ctx_expr = ast.unparse(item.context_expr)
                if len(ctx_expr) > 12:
                    ctx_expr = ctx_expr[:9] + "..."
                items.append(ctx_expr)
            except Exception:
                items.append("context")

        # Limit the number of items shown
        if len(items) > 2:
            items = items[:2] + ["..."]

        label = f"with {', '.join(items)}"
        with_node = self.g.add("op", label)
        self.g.link(entry, with_node)
        return self._build_block(stmt.body, with_node)


def scan_py_files(root: Path, ignore: list[str], max_files: int) -> list[Path]:
    files = []
    for py_file in root.rglob("*.py"):
        if any(ig in str(py_file) for ig in ignore):
            continue
        files.append(py_file)
        if len(files) >= max_files:
            break
    return files


def build_for_file(
    file_path: Path, config: dict[str, Any] | None = None
) -> list[tuple]:
    config = config or {}
    enc = config.get("encoding", "utf-8")
    try:
        source = Path(file_path).read_text(encoding=enc)
    except UnicodeDecodeError:
        try:
            source = Path(file_path).read_text(encoding="cp950")
        except UnicodeDecodeError:
            return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    charts = []
    builder = Builder(f"Module: {Path(file_path).name}", config)
    graph = builder.build_module(tree)
    charts.append((("Module", graph.to_mermaid())))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            builder = Builder(f"Function: {node.name}", config)
            graph = builder.build_function(node)
            charts.append((f"Function: {node.name}", graph.to_mermaid()))
    return charts


def analyze_and_generate(
    root_path: Path,
    output_dir: Path,
    ignore_patterns: list[str] | None = None,
    max_files: int = 50,
) -> None:
    """分析 Python 檔案並生成 Mermaid 流程圖.
    
    Args:
        root_path: 要分析的根目錄
        output_dir: 輸出目錄
        ignore_patterns: 要忽略的路徑模式
        max_files: 最大處理檔案數
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "_out",
            "emoji_backups",
            ".pytest_cache",
            "target",
            "build",
            "dist",
        ]
    
    files = scan_py_files(root_path, ignore_patterns, max_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"找到 {len(files)} 個 Python 檔案")
    print(f"開始生成 Mermaid 流程圖...")
    
    total_charts = 0
    for i, file_path in enumerate(files, 1):
        if i % 10 == 0:
            print(f"進度: {i}/{len(files)}")
        
        charts = build_for_file(file_path)
        if not charts:
            continue
        
        # 為每個函數生成單獨的檔案
        rel_path = file_path.relative_to(root_path)
        file_prefix = str(rel_path).replace("/", "_").replace("\\", "_").replace(".py", "")
        
        for chart_name, mermaid_code in charts:
            safe_name = chart_name.replace(" ", "_").replace(":", "_")
            output_file = output_dir / f"{file_prefix}_{safe_name}.mmd"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
            
            total_charts += 1
    
    print(f"完成！共生成 {total_charts} 個流程圖檔案")
    print(f"輸出目錄: {output_dir}")


def main():
    """主函數 - 提供命令列介面."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Python AST 解析與 Mermaid 流程圖產生工具"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="輸入目錄或檔案路徑 (預設: ./services)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="輸出目錄 (預設: ./docs/diagrams)",
    )
    parser.add_argument(
        "--max-files",
        "-m",
        type=int,
        default=50,
        help="最大處理檔案數 (預設: 50)",
    )
    parser.add_argument(
        "--direction",
        "-d",
        choices=["TB", "BT", "LR", "RL"],
        default="TB",
        help="流程圖方向 (預設: TB = 從上到下)",
    )
    
    args = parser.parse_args()
    
    # 設定預設路徑
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_path = args.input or (project_root / "services")
    output_path = args.output or (project_root / "docs" / "diagrams")
    
    print("=" * 80)
    print("Python 程式碼流程圖生成工具")
    print("=" * 80)
    print(f"輸入路徑: {input_path}")
    print(f"輸出路徑: {output_path}")
    print(f"最大檔案數: {args.max_files}")
    print(f"流程圖方向: {args.direction}")
    print("=" * 80)
    print()
    
    config = {"direction": args.direction}
    
    if input_path.is_file():
        # 處理單一檔案
        charts = build_for_file(input_path, config)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for chart_name, mermaid_code in charts:
            safe_name = chart_name.replace(" ", "_").replace(":", "_")
            output_file = output_path / f"{input_path.stem}_{safe_name}.mmd"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
            
            print(f"生成: {output_file}")
        
        print(f"\n完成！共生成 {len(charts)} 個流程圖")
    else:
        # 處理目錄
        analyze_and_generate(
            root_path=input_path,
            output_dir=output_path,
            max_files=args.max_files,
        )


if __name__ == "__main__":
    main()
