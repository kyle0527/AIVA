// rs2mermaid.rs - Rust AST 解析與 Mermaid 流程圖產生工具
// 使用 syn crate 解析 Rust 程式碼

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use syn::{visit::Visit, Block, Expr, ExprBlock, ExprIf, ExprLoop, ExprWhile, ExprForLoop, ExprMatch, Item, Stmt};

/// 流程圖節點
#[derive(Debug, Clone)]
struct Node {
    id: String,
    label: String,
    kind: NodeKind,
    nexts: Vec<String>,
    edge_info: HashMap<String, EdgeStyle>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeKind {
    Op,
    Cond,
    Start,
    End,
    Subgraph,
}

#[derive(Debug, Clone)]
struct EdgeStyle {
    label: String,
    style: String,
}

impl Node {
    fn new(id: String, label: String, kind: NodeKind) -> Self {
        Node {
            id: Self::sanitize_id(&id),
            label,
            kind,
            nexts: Vec::new(),
            edge_info: HashMap::new(),
        }
    }

    fn sanitize_id(id: &str) -> String {
        let cleaned = id
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect::<String>();

        let cleaned = cleaned.trim_matches('_');

        if cleaned.is_empty() {
            return "node".to_string();
        }

        if cleaned.chars().next().unwrap().is_numeric() || cleaned.starts_with('_') || cleaned.starts_with('-') {
            format!("n_{}", cleaned)
        } else {
            cleaned.to_string()
        }
    }
}

/// 流程圖
struct Graph {
    title: String,
    direction: String,
    nodes: Vec<Node>,
    counter: usize,
    start_id: String,
    end_id: String,
}

impl Graph {
    fn new(title: String, direction: String) -> Self {
        let direction = Self::validate_direction(&direction);
        let mut graph = Graph {
            title,
            direction,
            nodes: Vec::new(),
            counter: 0,
            start_id: String::new(),
            end_id: String::new(),
        };

        let start = graph.add(NodeKind::Start, "開始".to_string());
        graph.start_id = start;
        let end = graph.add(NodeKind::End, "結束".to_string());
        graph.end_id = end;

        graph
    }

    fn validate_direction(dir: &str) -> String {
        let dir = dir.to_uppercase();
        let valid = vec!["TB", "TD", "BT", "LR", "RL"];

        if valid.contains(&dir.as_str()) {
            if dir == "TD" {
                return "TB".to_string(); // TD deprecated
            }
            dir
        } else {
            "TB".to_string()
        }
    }

    fn add(&mut self, kind: NodeKind, label: String) -> String {
        self.counter += 1;
        let id = format!("n{}", self.counter);
        let node = Node::new(id.clone(), label, kind);
        self.nodes.push(node);
        id
    }

    fn link(&mut self, from: &str, to: &str, label: Option<&str>, style: Option<&str>) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == from) {
            if !node.nexts.contains(&to.to_string()) {
                node.nexts.push(to.to_string());
            }
            if label.is_some() || style.is_some() {
                node.edge_info.insert(
                    to.to_string(),
                    EdgeStyle {
                        label: label.unwrap_or("").to_string(),
                        style: style.unwrap_or("-->").to_string(),
                    },
                );
            }
        }
    }

    fn to_mermaid(&self) -> String {
        let mut lines = vec![format!("flowchart {}", self.direction)];

        // 節點定義
        for node in &self.nodes {
            lines.push(format!("    {}", self.format_node(node)));
        }

        lines.push(String::new());

        // 連接
        for node in &self.nodes {
            for (i, next_id) in node.nexts.iter().enumerate() {
                let edge = self.format_edge(node, next_id, i);
                lines.push(format!("    {}", edge));
            }
        }

        lines.join("\n")
    }

    fn format_node(&self, node: &Node) -> String {
        let text = Self::sanitize_text(&node.label);

        match node.kind {
            NodeKind::Cond => format!("{}{{\"{}\" }}", node.id, text),
            NodeKind::Start | NodeKind::End => format!("{}([\"{}\" ])", node.id, text),
            NodeKind::Subgraph => format!("{}[/\"{}\"/]", node.id, text),
            NodeKind::Op => format!("{}[\"{}\" ]", node.id, text),
        }
    }

    fn format_edge(&self, from: &Node, to_id: &str, index: usize) -> String {
        if from.kind == NodeKind::Cond {
            if index == 0 {
                return format!("{} -->|Yes| {}", from.id, to_id);
            } else if index == 1 {
                return format!("{} -->|No| {}", from.id, to_id);
            }
        }

        if let Some(edge) = from.edge_info.get(to_id) {
            if !edge.label.is_empty() {
                return format!("{} -->|{}| {}", from.id, edge.label, to_id);
            }
        }

        format!("{} --> {}", from.id, to_id)
    }

    fn sanitize_text(text: &str) -> String {
        let mut result = text
            .replace('&', "&amp;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('#', "&#35;")
            .replace('|', "&#124;")
            .replace('(', "&#40;")
            .replace(')', "&#41;")
            .replace('[', "&#91;")
            .replace(']', "&#93;")
            .replace('{', "&#123;")
            .replace('}', "&#125;")
            .replace('\n', "<br/>")
            .replace('\t', "    ");

        // 清理多餘空格
        result = result.split_whitespace().collect::<Vec<_>>().join(" ");

        // 限制長度（使用字元計數而非字節數，正確處理 UTF-8）
        if result.chars().count() > 50 {
            result = result.chars().take(47).collect::<String>() + "...";
        }

        result
    }
}

/// AST 構建器
struct Builder {
    graph: Graph,
}

impl Builder {
    fn new(title: String, direction: String) -> Self {
        Builder {
            graph: Graph::new(title, direction),
        }
    }

    fn build_function(&mut self, func: &syn::ItemFn) -> String {
        let mut last = self.graph.start_id.clone();

        if let Some(body) = &func.block.stmts.first() {
            last = self.build_block(&func.block, &last);
        }

        self.graph.link(&last, &self.graph.end_id.clone(), None, None);
        self.graph.to_mermaid()
    }

    fn build_block(&mut self, block: &Block, entry: &str) -> String {
        let mut last = entry.to_string();

        for stmt in &block.stmts {
            last = self.build_stmt(stmt, &last);
        }

        last
    }

    fn build_stmt(&mut self, stmt: &Stmt, entry: &str) -> String {
        match stmt {
            Stmt::Expr(expr, _) | Stmt::Expr(expr, Some(_)) => self.build_expr(expr, entry),
            Stmt::Local(local) => {
                let label = if let Some(init) = &local.init {
                    format!(
                        "let {} = {}",
                        quote::quote!(#local.pat).to_string(),
                        Self::expr_to_string(&init.expr)
                    )
                } else {
                    format!("let {}", quote::quote!(#local.pat).to_string())
                };

                let node_id = self.graph.add(NodeKind::Op, label);
                self.graph.link(entry, &node_id, None, None);
                node_id
            }
            Stmt::Item(item) => {
                let label = format!("item: {}", Self::item_to_string(item));
                let node_id = self.graph.add(NodeKind::Op, label);
                self.graph.link(entry, &node_id, None, None);
                node_id
            }
            _ => {
                let label = "statement".to_string();
                let node_id = self.graph.add(NodeKind::Op, label);
                self.graph.link(entry, &node_id, None, None);
                node_id
            }
        }
    }

    fn build_expr(&mut self, expr: &Expr, entry: &str) -> String {
        match expr {
            Expr::If(expr_if) => self.build_if(expr_if, entry),
            Expr::While(expr_while) => self.build_while(expr_while, entry),
            Expr::ForLoop(expr_for) => self.build_for(expr_for, entry),
            Expr::Loop(expr_loop) => self.build_loop(expr_loop, entry),
            Expr::Match(expr_match) => self.build_match(expr_match, entry),
            Expr::Block(expr_block) => self.build_expr_block(expr_block, entry),
            Expr::Return(expr_return) => {
                let label = if let Some(val) = &expr_return.expr {
                    format!("return {}", Self::expr_to_string(val))
                } else {
                    "return".to_string()
                };
                let node_id = self.graph.add(NodeKind::Op, label);
                self.graph.link(entry, &node_id, None, None);
                node_id
            }
            _ => {
                let label = Self::expr_to_string(expr);
                let node_id = self.graph.add(NodeKind::Op, label);
                self.graph.link(entry, &node_id, None, None);
                node_id
            }
        }
    }

    fn build_if(&mut self, expr_if: &ExprIf, entry: &str) -> String {
        let cond_text = Self::expr_to_string(&expr_if.cond);
        let cond_node = self.graph.add(NodeKind::Cond, format!("if {}", cond_text));
        self.graph.link(entry, &cond_node, None, None);

        // Then 分支
        let then_last = self.build_block(&expr_if.then_branch, &cond_node);

        if let Some((_, else_branch)) = &expr_if.else_branch {
            // Else 分支
            let else_last = self.build_expr(else_branch, &cond_node);
            let merge = self.graph.add(NodeKind::Op, String::new());
            self.graph.link(&then_last, &merge, None, None);
            self.graph.link(&else_last, &merge, None, None);
            merge
        } else {
            let merge = self.graph.add(NodeKind::Op, String::new());
            self.graph.link(&then_last, &merge, None, None);
            self.graph.link(&cond_node, &merge, Some("No"), None);
            merge
        }
    }

    fn build_while(&mut self, expr_while: &ExprWhile, entry: &str) -> String {
        let cond_text = Self::expr_to_string(&expr_while.cond);
        let while_node = self.graph.add(NodeKind::Cond, format!("while {}", cond_text));
        self.graph.link(entry, &while_node, None, None);

        let body_last = self.build_block(&expr_while.body, &while_node);
        self.graph.link(&body_last, &while_node, Some("loop"), None);

        let exit = self.graph.add(NodeKind::Op, String::new());
        self.graph.link(&while_node, &exit, Some("exit"), None);
        exit
    }

    fn build_for(&mut self, expr_for: &ExprForLoop, entry: &str) -> String {
        let pat = quote::quote!(#expr_for.pat).to_string();
        let iter = Self::expr_to_string(&expr_for.expr);
        let for_node = self.graph.add(NodeKind::Cond, format!("for {} in {}", pat, iter));
        self.graph.link(entry, &for_node, None, None);

        let body_last = self.build_block(&expr_for.body, &for_node);
        self.graph.link(&body_last, &for_node, Some("loop"), None);

        let exit = self.graph.add(NodeKind::Op, String::new());
        self.graph.link(&for_node, &exit, Some("exit"), None);
        exit
    }

    fn build_loop(&mut self, expr_loop: &ExprLoop, entry: &str) -> String {
        let loop_node = self.graph.add(NodeKind::Cond, "loop".to_string());
        self.graph.link(entry, &loop_node, None, None);

        let body_last = self.build_block(&expr_loop.body, &loop_node);
        self.graph.link(&body_last, &loop_node, Some("loop"), None);

        let exit = self.graph.add(NodeKind::Op, String::new());
        self.graph.link(&loop_node, &exit, Some("exit"), None);
        exit
    }

    fn build_match(&mut self, expr_match: &ExprMatch, entry: &str) -> String {
        let expr_text = Self::expr_to_string(&expr_match.expr);
        let match_node = self.graph.add(NodeKind::Cond, format!("match {}", expr_text));
        self.graph.link(entry, &match_node, None, None);

        let merge = self.graph.add(NodeKind::Op, String::new());

        for arm in &expr_match.arms {
            let pat = quote::quote!(#arm.pat).to_string();
            let arm_node = self.graph.add(NodeKind::Op, format!("case {}", pat));
            self.graph.link(&match_node, &arm_node, None, None);

            let arm_last = self.build_expr(&arm.body, &arm_node);
            self.graph.link(&arm_last, &merge, None, None);
        }

        merge
    }

    fn build_expr_block(&mut self, expr_block: &ExprBlock, entry: &str) -> String {
        self.build_block(&expr_block.block, entry)
    }

    fn expr_to_string(expr: &Expr) -> String {
        let text = quote::quote!(#expr).to_string();
        if text.len() > 30 {
            format!("{}...", &text[..27])
        } else {
            text
        }
    }

    fn item_to_string(item: &Item) -> String {
        match item {
            Item::Fn(func) => format!("fn {}", func.sig.ident),
            Item::Struct(s) => format!("struct {}", s.ident),
            Item::Enum(e) => format!("enum {}", e.ident),
            Item::Mod(m) => format!("mod {}", m.ident),
            _ => "item".to_string(),
        }
    }
}

/// 掃描 Rust 文件
fn scan_rust_files(root: &Path, ignore: &[&str], max_files: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();

    fn walk(dir: &Path, files: &mut Vec<PathBuf>, ignore: &[&str], max_files: usize) {
        if files.len() >= max_files {
            return;
        }

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                // 檢查忽略模式
                if ignore.iter().any(|ig| path.to_string_lossy().contains(ig)) {
                    continue;
                }

                if path.is_dir() {
                    walk(&path, files, ignore, max_files);
                } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    files.push(path);
                    if files.len() >= max_files {
                        break;
                    }
                }
            }
        }
    }

    walk(root, &mut files, ignore, max_files);
    files
}

/// 為單個文件構建流程圖
fn build_for_file(file_path: &Path, direction: &str) -> Vec<(String, String)> {
    let mut charts = Vec::new();

    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return charts,
    };

    let syntax = match syn::parse_file(&content) {
        Ok(s) => s,
        Err(_) => return charts,
    };

    for item in syntax.items {
        if let Item::Fn(func) = item {
            let func_name = func.sig.ident.to_string();
            let params: Vec<String> = func
                .sig
                .inputs
                .iter()
                .map(|arg| quote::quote!(#arg).to_string())
                .collect();

            let title = format!("Function: {}({})", func_name, params.join(", "));

            let mut builder = Builder::new(title, direction.to_string());
            let mermaid = builder.build_function(&func);

            charts.push((func_name, mermaid));
        }
    }

    charts
}

/// 分析並生成 Mermaid 圖表
fn analyze_and_generate(
    root_path: &Path,
    output_dir: &Path,
    ignore_patterns: &[&str],
    max_files: usize,
    direction: &str,
) {
    println!("{}", "=".repeat(80));
    println!("Rust 程式碼流程圖生成工具");
    println!("{}", "=".repeat(80));
    println!("輸入路徑: {}", root_path.display());
    println!("輸出路徑: {}", output_dir.display());
    println!("最大檔案數: {}", max_files);
    println!("流程圖方向: {}", direction);
    println!("{}", "=".repeat(80));
    println!();

    let files = scan_rust_files(root_path, ignore_patterns, max_files);
    println!("找到 {} 個 Rust 檔案", files.len());
    println!("開始生成 Mermaid 流程圖...");

    // 創建輸出目錄
    fs::create_dir_all(output_dir).ok();

    let mut total_charts = 0;

    for (i, file_path) in files.iter().enumerate() {
        if (i + 1) % 10 == 0 {
            println!("進度: {}/{}", i + 1, files.len());
        }

        let charts = build_for_file(file_path, direction);

        if charts.is_empty() {
            continue;
        }

        // 生成輸出文件名
        let rel_path = file_path.strip_prefix(root_path).unwrap_or(file_path);
        let file_prefix = rel_path
            .to_string_lossy()
            .replace(['/', '\\'], "_")
            .replace(".rs", "");

        for (func_name, mermaid) in charts {
            let safe_name = func_name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
            let output_file = output_dir.join(format!("{}_{}.mmd", file_prefix, safe_name));

            fs::write(output_file, mermaid).ok();
            total_charts += 1;
        }
    }

    println!();
    println!("完成！共生成 {} 個流程圖檔案", total_charts);
    println!("輸出目錄: {}", output_dir.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // 簡單參數解析
    let mut input_path = PathBuf::from("./services/scan/engines/rust_engine");
    let mut output_dir = PathBuf::from("./docs/diagrams/rust");
    let mut direction = "TB".to_string();
    let mut max_files = 10000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                input_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" | "-o" => {
                output_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--direction" | "-d" => {
                direction = args[i + 1].clone();
                i += 2;
            }
            "--max-files" | "-m" => {
                max_files = args[i + 1].parse().unwrap_or(10000);
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    r#"
使用方法:
  cargo run --bin rs2mermaid -- [選項]

選項:
  -i, --input <path>      輸入目錄或檔案路徑 (預設: ./services/scan/engines/rust_engine)
  -o, --output <path>     輸出目錄 (預設: ./docs/diagrams/rust)
  -d, --direction <dir>   流程圖方向 TB/LR/RL/BT (預設: TB)
  -m, --max-files <num>   最大處理檔案數 (預設: 10000)
  -h, --help              顯示幫助訊息

範例:
  cargo run --bin rs2mermaid -- -i ./src -o ./output -d LR
                "#
                );
                return;
            }
            _ => i += 1,
        }
    }

    let ignore_patterns = &["target", ".git", "node_modules", "__pycache__"];

    analyze_and_generate(&input_path, &output_dir, ignore_patterns, max_files, &direction);
}
