from __future__ import annotations
import os, json, ast, hashlib
from typing import Dict, Any, List
from ..registry import CapabilityRegistry

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def register_builtins(reg: CapabilityRegistry) -> None:
    reg.register("echo", echo, desc="echo text back")
    reg.register("index_repo", index_repo, desc="index files under a root folder")
    reg.register("parse_ast", parse_ast, desc="parse python files and extract simple summaries")
    reg.register("build_graph", build_graph, desc="build simple call graph from AST summaries")
    reg.register("render_report", render_report, desc="render a markdown report from artifacts")

def echo(text: str) -> Dict[str, Any]:
    return {"ok": True, "result": {"echo": text}}

def index_repo(root: str = ".", exts: List[str] | None = None, out_dir: str | None = None) -> Dict[str, Any]:
    exts = exts or [".py"]
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        # skip common vendor dirs
        if any(part in dirpath for part in (".git", ".cache", "node_modules", "__pycache__")):
            continue
        for fn in filenames:
            if any(fn.endswith(e) for e in exts):
                files.append(os.path.join(dirpath, fn))
    artifacts = {}
    os.makedirs(out_dir or "data/artifacts", exist_ok=True)
    idx_path = os.path.join(out_dir or "data/artifacts", f"index_{_sha1(root)}.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"root": root, "count": len(files), "files": files}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "metrics": {"files": len(files)}, "artifacts": {"index": idx_path}}

def parse_ast(index_artifact: str, out_dir: str | None = None) -> Dict[str, Any]:
    with open(index_artifact, "r", encoding="utf-8") as f:
        idx = json.load(f)
    summaries = []
    for path in idx["files"]:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                src = fp.read()
            tree = ast.parse(src, filename=path)
            funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            summaries.append({
                "path": path,
                "functions": len(funcs),
                "calls": len(calls),
            })
        except Exception:
            continue
    os.makedirs(out_dir or "data/artifacts", exist_ok=True)
    ast_path = os.path.join(out_dir or "data/artifacts", f"ast_{_sha1(index_artifact)}.json")
    with open(ast_path, "w", encoding="utf-8") as f:
        json.dump({"files": summaries}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "metrics": {"files": len(summaries)}, "artifacts": {"ast": ast_path}}

def build_graph(ast_artifact: str, out_dir: str | None = None) -> Dict[str, Any]:
    with open(ast_artifact, "r", encoding="utf-8") as f:
        data = json.load(f)
    # very naive "graph": node per file, edge weight = calls count (self loop)
    nodes = [{"id": i, "path": f["path"], "functions": f["functions"], "calls": f["calls"]}
             for i, f in enumerate(data["files"])]
    edges = [{"source": n["id"], "target": n["id"], "weight": n["calls"]} for n in nodes]
    os.makedirs(out_dir or "data/artifacts", exist_ok=True)
    gpath = os.path.join(out_dir or "data/artifacts", f"graph_{_sha1(ast_artifact)}.json")
    with open(gpath, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "artifacts": {"graph": gpath}}

def render_report(index_artifact: str, ast_artifact: str, graph_artifact: str, out_dir: str | None = None) -> Dict[str, Any]:
    with open(index_artifact, "r", encoding="utf-8") as f:
        idx = json.load(f)
    with open(ast_artifact, "r", encoding="utf-8") as f:
        astd = json.load(f)
    with open(graph_artifact, "r", encoding="utf-8") as f:
        g = json.load(f)
    md = []
    md.append(f"# AIVA Scan Report")
    md.append(f"- Indexed files: {idx.get('count')} under {idx.get('root')}")
    md.append(f"- AST summarized files: {len(astd.get('files', []))}")
    md.append("## Top files by calls")
    top = sorted(astd.get("files", []), key=lambda x: x.get("calls", 0), reverse=True)[:10]
    for it in top:
        md.append(f"- `{it['path']}` : functions={it['functions']} calls={it['calls']}")
    os.makedirs(out_dir or "reports", exist_ok=True)
    rp = os.path.join(out_dir or "reports", f"report_{_sha1(index_artifact)}.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return {"ok": True, "artifacts": {"report": rp}}
