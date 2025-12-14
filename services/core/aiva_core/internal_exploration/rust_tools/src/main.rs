// rs2mermaid.rs - All-in-One Rust AST Analysis Tool
// Version: 2.0 (Feature Parity with Python/Go)
//
// 功能整合：
// 1. AST 解析與流程圖生成 (基礎功能)
// 2. 跨檔案數據流串接 Stitching (對標 aiva_flow_analyzer.py)
// 3. 功能分類與統計 (對標 aiva_flow_classifier.py)
// 4. CLI 指令手冊生成 (對標 aiva_cli_implementation.py)
// 5. 系統瓶頸分析 (對標 aiva_flow_analyzer.py 統計)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::env;
use syn::{visit::Visit, *};

mod paths_config;
use paths_config::PathsConfig;

// ==========================================
// Part 1: 基礎數據結構 (Graph & Nodes)
// ==========================================

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
        let cleaned: String = id
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect();
        
        if cleaned.is_empty() { return "node".to_string(); }
        
        // 確保 ID 不以數字開頭
        if let Some(first) = cleaned.chars().next() {
            if first.is_numeric() {
                return format!("n_{}", cleaned);
            }
        }
        cleaned
    }
}

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
        let mut graph = Graph {
            title,
            direction,
            nodes: Vec::new(),
            counter: 0,
            start_id: String::new(),
            end_id: String::new(),
        };
        let start = graph.add(NodeKind::Start, "Start".to_string());
        graph.start_id = start;
        let end = graph.add(NodeKind::End, "End".to_string());
        graph.end_id = end;
        graph
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
        for node in &self.nodes {
            lines.push(format!("    {}", self.format_node(node)));
        }
        lines.push(String::new());
        for node in &self.nodes {
            for (i, next_id) in node.nexts.iter().enumerate() {
                lines.push(format!("    {}", self.format_edge(node, next_id, i)));
            }
        }
        lines.join("\n")
    }

    fn format_node(&self, node: &Node) -> String {
        let text = node.label.replace("\"", "'").replace("\n", " ");
        let text = if text.len() > 50 { format!("{}...", &text[..47]) } else { text };
        
        match node.kind {
            NodeKind::Cond => format!("{}{{\"{}\"}}", node.id, text),
            NodeKind::Start | NodeKind::End => format!("{}([\"{}\"])", node.id, text),
            _ => format!("{}[\"{}\"]", node.id, text),
        }
    }

    fn format_edge(&self, from: &Node, to_id: &str, index: usize) -> String {
        if from.kind == NodeKind::Cond {
            if index == 0 { return format!("{} -->|Yes| {}", from.id, to_id); }
            return format!("{} -->|No| {}", from.id, to_id);
        }
        if let Some(edge) = from.edge_info.get(to_id) {
            if !edge.label.is_empty() {
                return format!("{} -->|{}| {}", from.id, edge.label, to_id);
            }
        }
        format!("{} --> {}", from.id, to_id)
    }
}

// ==========================================
// Part 2: 數據流串接結構 (Stitcher)
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalCall {
    function_name: String,
    module_alias: String, // 模組或結構體名稱
    caller_func: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScriptNode {
    file_path: String,
    module_name: String,
    defined_funcs: Vec<String>,
    external_calls: Vec<ExternalCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct Connection {
    from_script: String,
    from_func: String,
    to_script: String,
    to_func: String,
    call_expr: String,
}

struct Stitcher {
    script_nodes: HashMap<String, ScriptNode>, // FilePath -> Node
    module_map: HashMap<String, Vec<String>>,  // ModuleName -> [FilePaths]
    real_connections: Vec<Connection>,
}

impl Stitcher {
    fn new() -> Self {
        Stitcher {
            script_nodes: HashMap::new(),
            module_map: HashMap::new(),
            real_connections: Vec::new(),
        }
    }

    fn add_script(&mut self, file_path: String, module_name: String, funcs: Vec<String>, calls: Vec<ExternalCall>) {
        let node = ScriptNode {
            file_path: file_path.clone(),
            module_name: module_name.clone(),
            defined_funcs: funcs,
            external_calls: calls,
        };
        
        self.script_nodes.insert(file_path.clone(), node);
        self.module_map.entry(module_name).or_default().push(file_path);
    }

    // 核心串接邏輯：解析跨檔案/跨模組調用
    fn resolve_connections(&mut self) {
        println!("🔗 正在執行跨檔案數據流串接 (Data Flow Stitching)...");
        
        // 複製數據以避免借用衝突
        let nodes: Vec<(String, Vec<ExternalCall>)> = self.script_nodes
            .iter()
            .map(|(path, node)| (path.clone(), node.external_calls.clone()))
            .collect();

        for (source_path, calls) in nodes {
            for call in calls {
                self.find_provider(&source_path, &call);
            }
        }
        println!("✅ 串接完成：發現 {} 條跨檔案連接", self.real_connections.len());
    }

    fn find_provider(&mut self, source_path: &str, call: &ExternalCall) {
        // 1. 精確模組匹配
        if !call.module_alias.is_empty() {
            if let Some(candidates) = self.module_map.get(&call.module_alias).cloned() {
                self.check_candidates(source_path, call, &candidates);
                return;
            }
        }

        // 2. 結構體方法匹配 (Struct::Method)
        // 有時 module_alias 可能是變數名，我們嘗試全域搜尋是否有 Struct::Method 符合
        let all_files: Vec<String> = self.script_nodes.keys().cloned().collect();
        self.check_candidates(source_path, call, &all_files);
    }

    fn check_candidates(&mut self, source_path: &str, call: &ExternalCall, candidates: &[String]) {
        for target_file in candidates {
            if target_file == source_path { continue; } // 忽略自調用

            if let Some(target_node) = self.script_nodes.get(target_file) {
                for def in &target_node.defined_funcs {
                    // 匹配邏輯：
                    // 1. 完全匹配: "my_func" == "my_func"
                    // 2. 結構體匹配: "MyStruct::my_method" vs 調用 "my_method" (如果 alias 符合)
                    // 3. 完整路徑: "mod::func"
                    
                    let is_match = if !call.module_alias.is_empty() {
                        // 檢查 Struct::Method 定義
                        def == &format!("{}::{}", call.module_alias, call.function_name)
                    } else {
                        def == &call.function_name
                    };

                    if is_match {
                        let conn = Connection {
                            from_script: source_path.to_string(),
                            from_func: call.caller_func.clone(),
                            to_script: target_file.clone(),
                            to_func: def.clone(),
                            call_expr: if call.module_alias.is_empty() { 
                                call.function_name.clone() 
                            } else { 
                                format!("{}::{}", call.module_alias, call.function_name) 
                            },
                        };

                        if !self.real_connections.contains(&conn) {
                            self.real_connections.push(conn);
                        }
                        return; // 找到即止
                    }
                }
            }
        }
    }

    // 生成系統架構圖 (System Flow)
    fn generate_system_flow(&self) -> String {
        let mut lines = vec![
            "flowchart TB".to_string(),
            "    %% 系統級數據流圖 (自動生成)".to_string(),
        ];
        
        let mut nodes = HashSet::new();

        for conn in &self.real_connections {
            let from_name = Path::new(&conn.from_script).file_stem().unwrap().to_str().unwrap();
            let to_name = Path::new(&conn.to_script).file_stem().unwrap().to_str().unwrap();
            
            let from_id = Node::sanitize_id(from_name);
            let to_id = Node::sanitize_id(to_name);

            if nodes.insert(from_id.clone()) {
                lines.push(format!("    {}[[\"{}\"]]", from_id, from_name));
            }
            if nodes.insert(to_id.clone()) {
                lines.push(format!("    {}[[\"{}\"]]", to_id, to_name));
            }

            lines.push(format!("    {} -->|{}| {}", from_id, conn.call_expr, to_id));
        }

        lines.join("\n")
    }

    // 分支統計分析
    fn analyze_branches(&self) -> HashMap<String, HashMap<String, usize>> {
        let mut fan_in = HashMap::new();
        let mut fan_out = HashMap::new();

        for conn in &self.real_connections {
            let src = Path::new(&conn.from_script).file_name().unwrap().to_str().unwrap().to_string();
            let dst = Path::new(&conn.to_script).file_name().unwrap().to_str().unwrap().to_string();
            
            *fan_out.entry(src).or_insert(0) += 1;
            *fan_in.entry(dst).or_insert(0) += 1;
        }

        let mut result = HashMap::new();
        
        let mut high_fan_out = HashMap::new();
        for (k, v) in fan_out { if v > 2 { high_fan_out.insert(k, v); } }
        result.insert("fan_out_nodes".to_string(), high_fan_out);

        let mut high_fan_in = HashMap::new();
        for (k, v) in fan_in { if v > 2 { high_fan_in.insert(k, v); } }
        result.insert("fan_in_nodes".to_string(), high_fan_in);

        result
    }
}

// ==========================================
// Part 3: AST 解析器 (Builder)
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlowMetadata {
    source_file: String,
    function_name: String,
    module: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    call_targets: Vec<String>,
    category: String,
    description: String,
}

struct Builder<'a> {
    graph: Graph,
    current: String,
    metadata: FlowMetadata,
    external_calls: Vec<ExternalCall>,
    current_func_name: String,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Builder<'a> {
    fn new(title: String, direction: String) -> Self {
        let graph = Graph::new(title, direction);
        let start_id = graph.start_id.clone();
        
        Builder {
            graph,
            current: start_id,
            metadata: FlowMetadata {
                source_file: String::new(),
                function_name: String::new(),
                module: String::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                call_targets: Vec::new(),
                category: String::new(),
                description: String::new(),
            },
            external_calls: Vec::new(),
            current_func_name: String::new(),
            _marker: std::marker::PhantomData,
        }
    }

    fn process_fn(&mut self, func: &'a ItemFn) {
        let name = func.sig.ident.to_string();
        self.current_func_name = name.clone();
        self.metadata.function_name = name;

        for input in &func.sig.inputs {
            if let FnArg::Typed(pat) = input {
                if let Pat::Ident(ident) = &*pat.pat {
                    self.metadata.inputs.push(ident.ident.to_string());
                }
            }
        }

        self.visit_block(&func.block);
        
        let end = self.graph.end_id.clone();
        self.graph.link(&self.current, &end, None, None);
    }

    fn process_method(&mut self, method: &'a ImplItemFn, struct_name: &str) {
        let name = format!("{}::{}", struct_name, method.sig.ident);
        self.current_func_name = name.clone();
        self.metadata.function_name = name;

        self.visit_block(&method.block);
        
        let end = self.graph.end_id.clone();
        self.graph.link(&self.current, &end, None, None);
    }
}

impl<'a> Visit<'a> for Builder<'a> {
    fn visit_expr_if(&mut self, i: &'a ExprIf) {
        let cond_id = self.graph.add(NodeKind::Cond, "If Condition".to_string());
        self.graph.link(&self.current, &cond_id, None, None);
        
        // Then
        self.current = cond_id.clone();
        let then_node = self.graph.add(NodeKind::Op, "Then".to_string());
        self.graph.link(&cond_id, &then_node, Some("Yes"), None);
        self.current = then_node;
        self.visit_block(&i.then_branch);
        let then_end = self.current.clone();

        // Else
        let mut else_end = None;
        if let Some((_, else_branch)) = &i.else_branch {
            let else_node = self.graph.add(NodeKind::Op, "Else".to_string());
            self.graph.link(&cond_id, &else_node, Some("No"), None);
            self.current = else_node;
            self.visit_expr(else_branch);
            else_end = Some(self.current.clone());
        }

        // Merge
        let merge = self.graph.add(NodeKind::Op, String::new());
        self.graph.link(&then_end, &merge, None, None);
        if let Some(end) = else_end {
            self.graph.link(&end, &merge, None, None);
        } else {
            self.graph.link(&cond_id, &merge, Some("No"), None);
        }
        self.current = merge;
    }

    fn visit_expr_call(&mut self, i: &'a ExprCall) {
        let mut func_name = String::new();
        let mut module_alias = String::new();

        if let Expr::Path(path) = &*i.func {
            if let Some(segment) = path.path.segments.last() {
                func_name = segment.ident.to_string();
            }
            if path.path.segments.len() > 1 {
                module_alias = path.path.segments.first().unwrap().ident.to_string();
            }
        }

        if !func_name.is_empty() {
            let label = format!("Call: {}", func_name);
            let node = self.graph.add(NodeKind::Op, label);
            self.graph.link(&self.current, &node, None, None);
            self.current = node;

            self.metadata.call_targets.push(func_name.clone());
            self.external_calls.push(ExternalCall {
                function_name: func_name,
                module_alias,
                caller_func: self.current_func_name.clone(),
            });
        }

        for arg in &i.args { self.visit_expr(arg); }
    }

    fn visit_expr_method_call(&mut self, i: &'a ExprMethodCall) {
        let method_name = i.method.to_string();
        let label = format!("Method: .{}()", method_name);
        
        let node = self.graph.add(NodeKind::Op, label);
        self.graph.link(&self.current, &node, None, None);
        self.current = node;

        // 簡單推斷 Receiver 作為模組別名 (例如 user.save() -> user)
        let mut receiver_alias = String::new();
        if let Expr::Path(path) = &*i.receiver {
            if let Some(seg) = path.path.segments.last() {
                receiver_alias = seg.ident.to_string();
            }
        }

        self.metadata.call_targets.push(method_name.clone());
        self.external_calls.push(ExternalCall {
            function_name: method_name,
            module_alias: receiver_alias,
            caller_func: self.current_func_name.clone(),
        });

        for arg in &i.args { self.visit_expr(arg); }
    }

    fn visit_expr_return(&mut self, i: &'a ExprReturn) {
        let node = self.graph.add(NodeKind::Op, "Return".to_string());
        self.graph.link(&self.current, &node, None, None);
        let end = self.graph.end_id.clone();
        self.graph.link(&node, &end, None, None);
        self.current = node;
        if let Some(expr) = &i.expr { self.visit_expr(expr); }
    }
}

// ==========================================
// Part 4: 主流程與分類器 (Classifier & Main)
// ==========================================

struct Classifier {
    categories: HashMap<String, Vec<FlowMetadata>>,
}

impl Classifier {
    fn new() -> Self {
        Classifier { categories: HashMap::new() }
    }

    fn classify(&mut self, metadata: &mut FlowMetadata) {
        let name = metadata.function_name.to_lowercase();
        let cat = if name.contains("scan") || name.contains("detect") { "reconnaissance" }
        else if name.contains("exploit") || name.contains("attack") { "exploitation" }
        else if name.contains("analyze") || name.contains("parse") { "analysis" }
        else if name.contains("report") || name.contains("generate") { "reporting" }
        else if name.contains("store") || name.contains("save") || name.contains("db") { "persistence" }
        else { "other" };
        
        metadata.category = cat.to_string();
        // Description placeholder - to be filled by LLM analysis
        metadata.description = format!("[PLACEHOLDER] {}", metadata.function_name);
        
        self.categories.entry(cat.to_string()).or_default().push(metadata.clone());
    }

    fn get_result(&self) -> serde_json::Value {
        let mut summary = HashMap::new();
        let mut total = 0;
        for (k, v) in &self.categories {
            summary.insert(k.clone(), v.len());
            total += v.len();
        }
        
        serde_json::json!({
            "total_flows": total,
            "categories": self.categories,
            "summary": summary
        })
    }
}

fn generate_cli(result: &serde_json::Value) -> String {
    let mut sb = String::from("# AIVA Rust Analysis CLI Commands\n\n");
    if let Some(cats) = result.get("categories").and_then(|c| c.as_object()) {
        for (cat, items) in cats {
            sb.push_str(&format!("## Category: {}\n", cat.to_uppercase()));
            if let Some(arr) = items.as_array() {
                for item in arr {
                    let file = item.get("source_file").and_then(|s| s.as_str()).unwrap_or("");
                    let func = item.get("function_name").and_then(|s| s.as_str()).unwrap_or("");
                    sb.push_str(&format!("cargo run --bin rs2mermaid -- --file {} --func {}\n", file, func));
                }
            }
            sb.push_str("\n");
        }
    }
    sb
}

fn scan_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if !path.ends_with("target") && !path.ends_with(".git") {
                    scan_files(&path, files);
                }
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let input_dir = args.iter().find(|a| a.starts_with("--input=")).map(|a| &a[8..]).unwrap_or(".");
    
    // 獲取路徑配置
    let paths_config = PathsConfig::new();
    let default_output = paths_config.get_default_output_dir();
    let output_dir = args.iter()
        .find(|a| a.starts_with("--output="))
        .map(|a| &a[9..])
        .unwrap_or(default_output.to_str().unwrap_or("./analysis_output"));

    println!("🚀 開始 AIVA Rust 全功能分析...");
    println!("📂 輸入: {}", input_dir);
    println!("📂 輸出: {}", output_dir);

    fs::create_dir_all(output_dir)?;

    let mut files = Vec::new();
    scan_files(Path::new(input_dir), &mut files);
    println!("📊 發現 {} 個 Rust 檔案", files.len());

    let mut stitcher = Stitcher::new();
    let mut classifier = Classifier::new();
    let mut all_meta = Vec::new();

    // 1. Scanning Phase
    for file_path in &files {
        let content = fs::read_to_string(file_path);
        if content.is_err() { continue; }
        let content = content.unwrap();
        
        let syntax = syn::parse_file(&content);
        if syntax.is_err() { 
            println!("⚠️  無法解析: {:?}", file_path);
            continue; 
        }
        let syntax = syntax.unwrap();

        let file_name = file_path.to_string_lossy().to_string();
        let module_name = file_path.file_stem().unwrap().to_string_lossy().to_string();

        let mut defined_funcs = Vec::new();
        let mut all_calls = Vec::new();

        // 處理一般函數
        for item in &syntax.items {
            if let Item::Fn(func) = item {
                let mut builder = Builder::new(func.sig.ident.to_string(), "TB".to_string());
                builder.process_fn(func);
                
                let out_name = format!("{}_{}.mmd", module_name, func.sig.ident);
                fs::write(Path::new(output_dir).join(out_name), builder.graph.to_mermaid())?;

                let mut meta = builder.metadata.clone();
                meta.source_file = file_name.clone();
                meta.module = module_name.clone();
                
                classifier.classify(&mut meta);
                all_meta.push(meta);
                
                defined_funcs.push(builder.current_func_name.clone());
                all_calls.extend(builder.external_calls);
            }
            
            // 處理 impl 區塊 (結構體方法)
            if let Item::Impl(impl_block) = item {
                // 獲取 Struct 名稱
                let struct_name = if let Type::Path(type_path) = &*impl_block.self_ty {
                    type_path.path.segments.last().unwrap().ident.to_string()
                } else {
                    "Unknown".to_string()
                };

                for impl_item in &impl_block.items {
                    if let ImplItem::Fn(method) = impl_item {
                        let mut builder = Builder::new(method.sig.ident.to_string(), "TB".to_string());
                        builder.process_method(method, &struct_name);
                        
                        let out_name = format!("{}_{}_{}.mmd", module_name, struct_name, method.sig.ident);
                        fs::write(Path::new(output_dir).join(out_name), builder.graph.to_mermaid())?;

                        let mut meta = builder.metadata.clone();
                        meta.source_file = file_name.clone();
                        meta.module = module_name.clone();
                        
                        classifier.classify(&mut meta);
                        all_meta.push(meta);

                        defined_funcs.push(builder.current_func_name.clone());
                        all_calls.extend(builder.external_calls);
                    }
                }
            }
        }

        stitcher.add_script(file_name, module_name, defined_funcs, all_calls);
    }

    // 2. Stitching Phase
    stitcher.resolve_connections();

    // 3. Analysis Phase
    let branch_stats = stitcher.analyze_branches();

    // 4. Output Results
    // A. 系統全圖
    let system_flow = stitcher.generate_system_flow();
    fs::write(Path::new(output_dir).join("system_flow.mmd"), system_flow)?;

    // B. CLI 手冊
    let class_result = classifier.get_result();
    let cli_docs = generate_cli(&class_result);
    fs::write(Path::new(output_dir).join("cli_commands.sh"), cli_docs)?;

    // C. 完整 JSON 報告
    let final_report = serde_json::json!({
        "summary": {
            "total_files": files.len(),
            "total_funcs": all_meta.len(),
            "real_connections": stitcher.real_connections.len(),
        },
        "classification": class_result,
        "branch_analysis": branch_stats,
        "flow_chains": stitcher.real_connections,
        "functions": all_meta
    });

    fs::write(
        Path::new(output_dir).join("analysis_results.json"), 
        serde_json::to_string_pretty(&final_report)?
    )?;

    println!("\n✅ 分析工作全部完成！");
    println!("   📄 系統架構圖: {}/system_flow.mmd", output_dir);
    println!("   📄 完整數據報告: {}/analysis_results.json", output_dir);
    println!("   📄 CLI 指令手冊: {}/cli_commands.sh", output_dir);

    Ok(())
}
