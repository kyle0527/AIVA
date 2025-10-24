# AIVA Features - Rust 開發指南 🦀

> **定位**: 核心效能引擎、記憶體安全、系統層安全  
> **規模**: 1,804 個 Rust 組件 (67%)  
> **職責**: SAST 引擎、DAG 系統、漏洞檢測、密碼學模組、檔案分析

---

## 🎯 **Rust 在 AIVA 中的角色**

### **🚀 核心引擎定位**
Rust 是 AIVA Features 模組的「**核心效能引擎**」，負責最關鍵的安全分析任務：

```
🦀 Rust 核心安全架構
├── 🔍 SAST 靜態分析引擎 (578組件)
│   ├── AST 解析與分析 (150組件)
│   ├── 語義分析引擎 (200組件)
│   ├── 模式匹配系統 (120組件)
│   └── 漏洞檢測邏輯 (108組件)
├── 📊 DAG 依賴分析系統 (425組件)
│   ├── 圖形結構處理 (180組件)
│   ├── 循環依賴檢測 (120組件)
│   ├── 關鍵路徑分析 (85組件)
│   └── 依賴樹最佳化 (40組件)
├── 🛡️ 漏洞檢測引擎 (357組件)
│   ├── CVE 數據處理 (120組件)
│   ├── 風險評估演算法 (100組件)
│   ├── 修復建議生成 (87組件)
│   └── 安全評分計算 (50組件)
├── 🔐 密碼學與加密 (286組件)
│   ├── 雜湊演算法實現 (90組件)
│   ├── 對稱/非對稱加密 (80組件)
│   ├── 數位簽章驗證 (70組件)
│   └── 密鑰管理系統 (46組件)
└── 📁 檔案處理與分析 (158組件)
    ├── 二進位檔案解析 (60組件)
    ├── 壓縮檔案處理 (45組件)
    ├── 檔案完整性檢查 (30組件)
    └── 元資料萃取 (23組件)
```

### **⚡ Rust 組件統計**
- **SAST 引擎**: 578 個組件 (32% - 核心分析能力)
- **DAG 系統**: 425 個組件 (23.5% - 依賴關係處理)
- **漏洞檢測**: 357 個組件 (19.8% - 安全風險識別)  
- **密碼學模組**: 286 個組件 (15.9% - 加密與驗證)
- **檔案處理**: 158 個組件 (8.8% - 檔案格式支援)

---

## 🏗️ **Rust 架構模式**

### **🔍 SAST 靜態分析引擎**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Error};

/// SAST 分析引擎核心結構
#[derive(Debug)]
pub struct SASTEngine {
    /// 語言解析器註冊表
    parsers: HashMap<String, Arc<dyn LanguageParser + Send + Sync>>,
    /// 規則引擎
    rule_engine: Arc<RuleEngine>,
    /// 結果快取
    result_cache: Arc<RwLock<HashMap<String, AnalysisResult>>>,
    /// 效能指標收集器
    metrics: Arc<SASTMetrics>,
}

/// 程式碼分析結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// 檔案路徑
    pub file_path: String,
    /// 程式語言
    pub language: String,
    /// 發現的問題
    pub findings: Vec<SecurityFinding>,
    /// 分析統計
    pub statistics: AnalysisStatistics,
    /// 分析時間戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 安全問題發現
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFinding {
    /// 問題ID
    pub id: String,
    /// 嚴重程度
    pub severity: Severity,
    /// 問題類型
    pub category: VulnerabilityCategory,
    /// 問題描述
    pub description: String,
    /// 檔案位置
    pub location: CodeLocation,
    /// 修復建議
    pub remediation: Option<String>,
    /// 信心度 (0.0-1.0)
    pub confidence: f64,
}

/// 程式碼位置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
    pub snippet: String,
}

impl SASTEngine {
    /// 建立新的 SAST 引擎
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();
        
        // 註冊多語言解析器
        parsers.insert("rust".to_string(), Arc::new(RustParser::new()));
        parsers.insert("python".to_string(), Arc::new(PythonParser::new()));
        parsers.insert("javascript".to_string(), Arc::new(JavaScriptParser::new()));
        parsers.insert("go".to_string(), Arc::new(GoParser::new()));
        
        Ok(Self {
            parsers,
            rule_engine: Arc::new(RuleEngine::new()?),
            result_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(SASTMetrics::new()),
        })
    }
    
    /// 分析單一檔案
    pub async fn analyze_file(&self, file_path: &str) -> Result<AnalysisResult> {
        let start_time = std::time::Instant::now();
        
        // 1. 檢查快取
        let cache_key = self.generate_cache_key(file_path).await?;
        {
            let cache = self.result_cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                self.metrics.cache_hit();
                return Ok(cached_result.clone());
            }
        }
        
        // 2. 檢測程式語言
        let language = self.detect_language(file_path)?;
        
        // 3. 獲取對應的解析器
        let parser = self.parsers.get(&language)
            .ok_or_else(|| Error::msg(format!("Unsupported language: {}", language)))?;
        
        // 4. 讀取檔案內容
        let content = tokio::fs::read_to_string(file_path).await?;
        
        // 5. 解析 AST
        let ast = parser.parse(&content)?;
        
        // 6. 執行安全分析
        let findings = self.rule_engine.analyze(&ast, &language).await?;
        
        // 7. 計算統計資訊
        let statistics = self.calculate_statistics(&content, &findings);
        
        // 8. 建立分析結果
        let result = AnalysisResult {
            file_path: file_path.to_string(),
            language,
            findings,
            statistics,
            timestamp: chrono::Utc::now(),
        };
        
        // 9. 更新快取
        {
            let mut cache = self.result_cache.write().await;
            cache.insert(cache_key, result.clone());
        }
        
        // 10. 記錄效能指標
        let duration = start_time.elapsed();
        self.metrics.record_analysis_time(duration);
        
        Ok(result)
    }
    
    /// 分析整個專案
    pub async fn analyze_project(&self, project_path: &str) -> Result<Vec<AnalysisResult>> {
        use tokio_stream::StreamExt;
        
        // 1. 發現所有支援的檔案
        let file_paths = self.discover_source_files(project_path).await?;
        
        // 2. 並行分析檔案 (限制並發數以避免資源耗盡)
        let semaphore = Arc::new(tokio::sync::Semaphore::new(num_cpus::get()));
        let mut tasks = Vec::new();
        
        for file_path in file_paths {
            let engine = Arc::new(self);
            let permit = semaphore.clone();
            
            let task = tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                engine.analyze_file(&file_path).await
            });
            
            tasks.push(task);
        }
        
        // 3. 收集所有結果
        let mut results = Vec::new();
        for task in tasks {
            match task.await? {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Analysis failed: {}", e);
                    // 繼續處理其他檔案，不因單一檔案失敗而停止
                }
            }
        }
        
        Ok(results)
    }
    
    /// 檢測檔案的程式語言
    fn detect_language(&self, file_path: &str) -> Result<String> {
        use std::path::Path;
        
        let path = Path::new(file_path);
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        match extension {
            "rs" => Ok("rust".to_string()),
            "py" => Ok("python".to_string()),
            "js" | "ts" | "jsx" | "tsx" => Ok("javascript".to_string()),
            "go" => Ok("go".to_string()),
            "c" | "cpp" | "cc" | "cxx" => Ok("cpp".to_string()),
            "java" => Ok("java".to_string()),
            _ => Err(Error::msg(format!("Unknown file extension: {}", extension)))
        }
    }
    
    /// 發現專案中的所有原始碼檔案
    async fn discover_source_files(&self, project_path: &str) -> Result<Vec<String>> {
        use tokio_stream::wrappers::ReadDirStream;
        
        let mut file_paths = Vec::new();
        let mut dir_stream = ReadDirStream::new(tokio::fs::read_dir(project_path).await?);
        
        while let Some(entry) = dir_stream.next().await {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(path_str) = path.to_str() {
                    if self.is_source_file(path_str) {
                        file_paths.push(path_str.to_string());
                    }
                }
            } else if path.is_dir() {
                // 遞迴處理子目錄
                let sub_files = self.discover_source_files(path.to_str().unwrap()).await?;
                file_paths.extend(sub_files);
            }
        }
        
        Ok(file_paths)
    }
    
    /// 檢查是否為支援的原始碼檔案
    fn is_source_file(&self, file_path: &str) -> bool {
        let supported_extensions = &[
            ".rs", ".py", ".js", ".ts", ".jsx", ".tsx", 
            ".go", ".c", ".cpp", ".cc", ".cxx", ".java"
        ];
        
        supported_extensions.iter().any(|ext| file_path.ends_with(ext))
    }
    
    /// 計算分析統計資訊
    fn calculate_statistics(&self, content: &str, findings: &[SecurityFinding]) -> AnalysisStatistics {
        let lines_of_code = content.lines().count() as u32;
        let total_findings = findings.len() as u32;
        
        let severity_counts = findings.iter().fold(
            HashMap::new(),
            |mut acc, finding| {
                *acc.entry(finding.severity.clone()).or_insert(0) += 1;
                acc
            }
        );
        
        AnalysisStatistics {
            lines_of_code,
            total_findings,
            severity_counts,
        }
    }
}

/// 分析統計資訊
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    pub lines_of_code: u32,
    pub total_findings: u32,
    pub severity_counts: HashMap<Severity, u32>,
}

/// 嚴重程度枚舉
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// 漏洞類別
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityCategory {
    SQLInjection,
    CrossSiteScripting,
    BufferOverflow,
    UseAfterFree,
    RaceCondition,
    InsecureCrypto,
    HardcodedSecrets,
    PathTraversal,
    CommandInjection,
    Other(String),
}
```

### **📊 DAG 依賴分析系統**

```rust
use petgraph::{Graph, Directed};
use petgraph::graph::{NodeIndex, EdgeIndex};
use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

/// DAG 依賴分析引擎
pub struct DependencyAnalyzer {
    /// 依賴圖
    graph: Graph<DependencyNode, DependencyEdge, Directed>,
    /// 節點索引映射
    node_indices: HashMap<String, NodeIndex>,
    /// 分析結果快取
    analysis_cache: HashMap<String, DependencyAnalysis>,
}

/// 依賴節點
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyNode {
    /// 模組名稱
    pub name: String,
    /// 模組版本
    pub version: String,
    /// 模組類型
    pub node_type: NodeType,
    /// 關聯的安全資訊
    pub security_info: SecurityInfo,
}

/// 依賴邊
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// 依賴類型
    pub edge_type: EdgeType,
    /// 版本約束
    pub version_constraint: String,
    /// 是否為可選依賴
    pub optional: bool,
}

/// 節點類型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Library,
    Application,
    SystemDependency,
    TransitiveDependency,
}

/// 邊類型
#[derive(Debug, Clone)]
pub enum EdgeType {
    DirectDependency,
    DevDependency,
    PeerDependency,
    RuntimeDependency,
}

/// 安全資訊
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityInfo {
    /// 已知漏洞
    pub vulnerabilities: Vec<Vulnerability>,
    /// 安全評分 (0-100)
    pub security_score: f32,
    /// 最後掃描時間
    pub last_scan: chrono::DateTime<chrono::Utc>,
}

/// 漏洞資訊
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// CVE ID
    pub cve_id: String,
    /// CVSS 評分
    pub cvss_score: f32,
    /// 嚴重程度
    pub severity: String,
    /// 描述
    pub description: String,
    /// 修復版本
    pub fixed_version: Option<String>,
}

/// 依賴分析結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    /// 總節點數
    pub total_nodes: usize,
    /// 總邊數
    pub total_edges: usize,
    /// 循環依賴
    pub circular_dependencies: Vec<CircularDependency>,
    /// 關鍵路徑
    pub critical_paths: Vec<CriticalPath>,
    /// 安全風險統計
    pub security_summary: SecuritySummary,
    /// 建議
    pub recommendations: Vec<Recommendation>,
}

impl DependencyAnalyzer {
    /// 建立新的依賴分析器
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            analysis_cache: HashMap::new(),
        }
    }
    
    /// 新增依賴節點
    pub fn add_dependency(&mut self, node: DependencyNode) -> NodeIndex {
        let node_id = format!("{}:{}", node.name, node.version);
        
        if let Some(&existing_index) = self.node_indices.get(&node_id) {
            return existing_index;
        }
        
        let index = self.graph.add_node(node);
        self.node_indices.insert(node_id, index);
        index
    }
    
    /// 新增依賴關係
    pub fn add_dependency_edge(
        &mut self, 
        from: NodeIndex, 
        to: NodeIndex, 
        edge: DependencyEdge
    ) -> EdgeIndex {
        self.graph.add_edge(from, to, edge)
    }
    
    /// 執行完整的依賴分析
    pub async fn analyze(&mut self) -> Result<DependencyAnalysis> {
        // 1. 檢測循環依賴
        let circular_deps = self.detect_circular_dependencies().await?;
        
        // 2. 找出關鍵路徑
        let critical_paths = self.find_critical_paths().await?;
        
        // 3. 分析安全風險
        let security_summary = self.analyze_security_risks().await?;
        
        // 4. 生成建議
        let recommendations = self.generate_recommendations(&circular_deps, &security_summary).await?;
        
        let analysis = DependencyAnalysis {
            total_nodes: self.graph.node_count(),
            total_edges: self.graph.edge_count(),
            circular_dependencies: circular_deps,
            critical_paths,
            security_summary,
            recommendations,
        };
        
        Ok(analysis)
    }
    
    /// 檢測循環依賴
    async fn detect_circular_dependencies(&self) -> Result<Vec<CircularDependency>> {
        use petgraph::algo::kosaraju_scc;
        
        let sccs = kosaraju_scc(&self.graph);
        let mut circular_deps = Vec::new();
        
        for scc in sccs {
            if scc.len() > 1 {
                // 找到強連通分量，表示有循環依賴
                let nodes: Vec<String> = scc.iter()
                    .map(|&node_index| {
                        let node = &self.graph[node_index];
                        format!("{}:{}", node.name, node.version)
                    })
                    .collect();
                
                circular_deps.push(CircularDependency {
                    nodes,
                    severity: self.calculate_circular_dependency_severity(&scc),
                });
            }
        }
        
        Ok(circular_deps)
    }
    
    /// 尋找關鍵路徑
    async fn find_critical_paths(&self) -> Result<Vec<CriticalPath>> {
        use petgraph::algo::dijkstra;
        
        let mut critical_paths = Vec::new();
        
        // 找出所有入口節點（沒有入邊的節點）
        let entry_nodes: Vec<NodeIndex> = self.graph.node_indices()
            .filter(|&node_index| {
                self.graph.edges_directed(node_index, petgraph::Incoming).count() == 0
            })
            .collect();
        
        // 找出所有出口節點（沒有出邊的節點）
        let exit_nodes: Vec<NodeIndex> = self.graph.node_indices()
            .filter(|&node_index| {
                self.graph.edges_directed(node_index, petgraph::Outgoing).count() == 0
            })
            .collect();
        
        // 計算從每個入口到每個出口的路徑
        for &entry in &entry_nodes {
            let distances = dijkstra(&self.graph, entry, None, |_| 1);
            
            for &exit in &exit_nodes {
                if let Some(&distance) = distances.get(&exit) {
                    // 重建路徑
                    let path = self.reconstruct_path(entry, exit, &distances);
                    
                    critical_paths.push(CriticalPath {
                        nodes: path,
                        length: distance,
                        risk_score: self.calculate_path_risk_score(&path).await,
                    });
                }
            }
        }
        
        // 按風險分數排序
        critical_paths.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap());
        
        Ok(critical_paths)
    }
    
    /// 分析安全風險
    async fn analyze_security_risks(&self) -> Result<SecuritySummary> {
        let mut high_risk_nodes = 0;
        let mut medium_risk_nodes = 0;
        let mut low_risk_nodes = 0;
        let mut total_vulnerabilities = 0;
        
        for node_index in self.graph.node_indices() {
            let node = &self.graph[node_index];
            let vuln_count = node.security_info.vulnerabilities.len();
            total_vulnerabilities += vuln_count;
            
            match node.security_info.security_score {
                score if score < 40.0 => high_risk_nodes += 1,
                score if score < 70.0 => medium_risk_nodes += 1,
                _ => low_risk_nodes += 1,
            }
        }
        
        Ok(SecuritySummary {
            high_risk_nodes,
            medium_risk_nodes,
            low_risk_nodes,
            total_vulnerabilities,
            average_security_score: self.calculate_average_security_score(),
        })
    }
    
    /// 計算路徑風險分數
    async fn calculate_path_risk_score(&self, path_nodes: &[String]) -> f32 {
        let mut total_risk = 0.0;
        let mut node_count = 0;
        
        for node_id in path_nodes {
            if let Some(node_index) = self.find_node_by_id(node_id) {
                let node = &self.graph[*node_index];
                
                // 基於安全分數和漏洞數量計算風險
                let vuln_risk = node.security_info.vulnerabilities.len() as f32 * 10.0;
                let score_risk = (100.0 - node.security_info.security_score) / 100.0 * 50.0;
                
                total_risk += vuln_risk + score_risk;
                node_count += 1;
            }
        }
        
        if node_count > 0 {
            total_risk / node_count as f32
        } else {
            0.0
        }
    }
    
    /// 生成建議
    async fn generate_recommendations(
        &self,
        circular_deps: &[CircularDependency],
        security_summary: &SecuritySummary,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        // 循環依賴建議
        if !circular_deps.is_empty() {
            recommendations.push(Recommendation {
                category: "Circular Dependencies".to_string(),
                priority: "High".to_string(),
                description: format!(
                    "發現 {} 個循環依賴，建議重構架構以消除循環引用",
                    circular_deps.len()
                ),
                action: "重構模組結構，引入介面層或抽象層來打破循環依賴".to_string(),
            });
        }
        
        // 安全風險建議
        if security_summary.high_risk_nodes > 0 {
            recommendations.push(Recommendation {
                category: "Security Risks".to_string(),
                priority: "Critical".to_string(),
                description: format!(
                    "發現 {} 個高風險依賴項，包含 {} 個已知漏洞",
                    security_summary.high_risk_nodes,
                    security_summary.total_vulnerabilities
                ),
                action: "立即更新高風險依賴項到安全版本，或尋找替代方案".to_string(),
            });
        }
        
        Ok(recommendations)
    }
}

/// 循環依賴
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularDependency {
    pub nodes: Vec<String>,
    pub severity: String,
}

/// 關鍵路徑
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub nodes: Vec<String>,
    pub length: i32,
    pub risk_score: f32,
}

/// 安全摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySummary {
    pub high_risk_nodes: usize,
    pub medium_risk_nodes: usize,
    pub low_risk_nodes: usize,
    pub total_vulnerabilities: usize,
    pub average_security_score: f32,
}

/// 建議
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: String,
    pub priority: String,
    pub description: String,
    pub action: String,
}
```

---

## 🛠️ **Rust 開發環境設定**

### **📦 Cargo.toml 配置**
```toml
[package]
name = "aiva-features-core"
version = "0.1.0"
edition = "2021"
authors = ["AIVA Security Team <security@aiva.com>"]
description = "AIVA Features 核心 Rust 組件"
license = "MIT"

[dependencies]
# 異步運行時
tokio = { version = "1.32", features = ["full"] }
tokio-stream = "0.1"

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# 錯誤處理
anyhow = "1.0"
thiserror = "1.0"

# 日誌
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# HTTP 客戶端
reqwest = { version = "0.11", features = ["json"] }

# 圖結構
petgraph = "0.6"

# 時間處理
chrono = { version = "0.4", features = ["serde"] }

# 雜湊和加密
sha2 = "0.10"
blake3 = "1.5"
ring = "0.17"

# 正則表達式
regex = "1.9"

# 平行處理
rayon = "1.8"
num_cpus = "1.16"

# 記憶體映射檔案
memmap2 = "0.9"

# AST 解析
tree-sitter = "0.22"
tree-sitter-rust = "0.21"
tree-sitter-python = "0.21"

[dev-dependencies]
# 測試
tokio-test = "0.4"
criterion = "0.5"
proptest = "1.2"

# 模擬
mockall = "0.11"

[profile.release]
# 最佳化配置
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.dev]
# 開發配置
opt-level = 0
debug = true
debug-assertions = true
overflow-checks = true

# 基準測試配置
[[bench]]
name = "sast_benchmark"
harness = false

[[bench]]
name = "dag_benchmark"  
harness = false

# 二進位檔案
[[bin]]
name = "sast-analyzer"
path = "src/bin/sast_analyzer.rs"

[[bin]]
name = "dag-analyzer"
path = "src/bin/dag_analyzer.rs"
```

### **🚀 快速開始**
```bash
# 1. Rust 環境設定
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup update

# 2. 工具鏈安裝
rustup component add clippy rustfmt
cargo install cargo-audit cargo-watch cargo-expand

# 3. 專案建置
cd services/features/
cargo build --release

# 4. 執行測試
cargo test --all-features

# 5. 效能基準測試
cargo bench

# 6. 程式碼品質檢查
cargo clippy -- -D warnings
cargo fmt --check
cargo audit
```

---

## 🧪 **測試策略**

### **🔍 單元測試範例**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_sast_engine_analyze_file() {
        // 準備測試資料
        let engine = SASTEngine::new().expect("Failed to create SAST engine");
        let test_code = r#"
            fn vulnerable_function(user_input: &str) -> String {
                // 潛在的 SQL 注入漏洞
                format!("SELECT * FROM users WHERE name = '{}'", user_input)
            }
        "#;
        
        // 建立臨時測試檔案
        let temp_file = create_temp_file("test.rs", test_code).await;
        
        // 執行分析
        let result = engine.analyze_file(&temp_file).await;
        
        // 驗證結果
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.language, "rust");
        assert!(!analysis.findings.is_empty());
        
        // 檢查是否檢測到 SQL 注入風險
        let sql_injection_found = analysis.findings.iter()
            .any(|finding| matches!(finding.category, VulnerabilityCategory::SQLInjection));
        assert!(sql_injection_found, "Should detect SQL injection vulnerability");
        
        // 清理
        cleanup_temp_file(&temp_file).await;
    }
    
    #[tokio::test]
    async fn test_dependency_analyzer_circular_detection() {
        let mut analyzer = DependencyAnalyzer::new();
        
        // 建立循環依賴
        let node_a = DependencyNode {
            name: "module_a".to_string(),
            version: "1.0.0".to_string(),
            node_type: NodeType::Library,
            security_info: SecurityInfo::default(),
        };
        
        let node_b = DependencyNode {
            name: "module_b".to_string(),
            version: "1.0.0".to_string(),
            node_type: NodeType::Library,
            security_info: SecurityInfo::default(),
        };
        
        let index_a = analyzer.add_dependency(node_a);
        let index_b = analyzer.add_dependency(node_b);
        
        // A -> B -> A (循環)
        analyzer.add_dependency_edge(index_a, index_b, DependencyEdge::default());
        analyzer.add_dependency_edge(index_b, index_a, DependencyEdge::default());
        
        // 執行分析
        let analysis = analyzer.analyze().await.expect("Analysis failed");
        
        // 驗證循環依賴檢測
        assert!(!analysis.circular_dependencies.is_empty());
        assert_eq!(analysis.circular_dependencies.len(), 1);
        assert_eq!(analysis.circular_dependencies[0].nodes.len(), 2);
    }
    
    // 屬性測試 (Property Testing)
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_security_score_bounds(
            score in 0.0f32..100.0f32
        ) {
            let security_info = SecurityInfo {
                vulnerabilities: vec![],
                security_score: score,
                last_scan: chrono::Utc::now(),
            };
            
            // 安全分數應該在有效範圍內
            assert!(security_info.security_score >= 0.0);
            assert!(security_info.security_score <= 100.0);
        }
        
        #[test]
        fn test_vulnerability_severity_consistency(
            cvss_score in 0.0f32..10.0f32
        ) {
            let severity = match cvss_score {
                s if s >= 9.0 => "Critical",
                s if s >= 7.0 => "High", 
                s if s >= 4.0 => "Medium",
                _ => "Low",
            };
            
            let vulnerability = Vulnerability {
                cve_id: "CVE-2023-12345".to_string(),
                cvss_score,
                severity: severity.to_string(),
                description: "Test vulnerability".to_string(),
                fixed_version: None,
            };
            
            // CVSS 分數與嚴重程度應該一致
            match vulnerability.cvss_score {
                s if s >= 9.0 => assert_eq!(vulnerability.severity, "Critical"),
                s if s >= 7.0 => assert_eq!(vulnerability.severity, "High"),
                s if s >= 4.0 => assert_eq!(vulnerability.severity, "Medium"),
                _ => assert_eq!(vulnerability.severity, "Low"),
            }
        }
    }
    
    // 基準測試
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn benchmark_sast_analysis(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let engine = rt.block_on(async { SASTEngine::new().unwrap() });
        
        c.bench_function("sast_analyze_rust_file", |b| {
            b.iter(|| {
                rt.block_on(async {
                    engine.analyze_file(black_box("test_data/sample.rs")).await
                })
            })
        });
    }
    
    fn benchmark_dag_analysis(c: &mut Criterion) {
        let mut analyzer = DependencyAnalyzer::new();
        
        // 建立測試圖
        for i in 0..1000 {
            let node = DependencyNode {
                name: format!("module_{}", i),
                version: "1.0.0".to_string(),
                node_type: NodeType::Library,
                security_info: SecurityInfo::default(),
            };
            analyzer.add_dependency(node);
        }
        
        c.bench_function("dag_analyze_1000_nodes", |b| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    analyzer.analyze().await
                })
            })
        });
    }
    
    criterion_group!(benches, benchmark_sast_analysis, benchmark_dag_analysis);
    criterion_main!(benches);
}

// 測試輔助函數
async fn create_temp_file(name: &str, content: &str) -> String {
    use std::io::Write;
    
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(name);
    
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
    
    file_path.to_string_lossy().to_string()
}

async fn cleanup_temp_file(file_path: &str) {
    let _ = std::fs::remove_file(file_path);
}
```

---

## 📈 **效能優化指南**

### **⚡ 記憶體最佳化**
```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use memmap2::MmapOptions;

/// 高效能檔案處理器
pub struct OptimizedFileProcessor {
    /// 記憶體映射快取
    mmap_cache: Arc<dashmap::DashMap<String, Arc<memmap2::Mmap>>>,
    /// 處理統計
    stats: ProcessingStats,
}

#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub files_processed: AtomicUsize,
    pub bytes_processed: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub cache_misses: AtomicUsize,
}

impl OptimizedFileProcessor {
    /// 高效能檔案讀取 (使用記憶體映射)
    pub async fn read_large_file(&self, file_path: &str) -> Result<&[u8]> {
        // 檢查快取
        if let Some(mmap) = self.mmap_cache.get(file_path) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(&mmap);
        }
        
        // 記憶體映射檔案
        let file = std::fs::File::open(file_path)?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)?
        };
        
        let mmap_arc = Arc::new(mmap);
        self.mmap_cache.insert(file_path.to_string(), mmap_arc.clone());
        
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.stats.files_processed.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_processed.fetch_add(mmap_arc.len(), Ordering::Relaxed);
        
        Ok(&mmap_arc)
    }
    
    /// 並行處理多個檔案
    pub async fn process_files_parallel<F, R>(
        &self,
        file_paths: Vec<String>,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(&[u8]) -> R + Send + Sync,
        R: Send,
    {
        use rayon::prelude::*;
        
        // 使用 rayon 進行 CPU 密集型並行處理
        let results: Result<Vec<_>> = file_paths
            .into_par_iter()
            .map(|file_path| -> Result<R> {
                let content = futures::executor::block_on(
                    self.read_large_file(&file_path)
                )?;
                Ok(processor(content))
            })
            .collect();
        
        results
    }
}

/// 零複製字串解析器
pub struct ZeroCopyParser<'a> {
    content: &'a [u8],
    position: usize,
}

impl<'a> ZeroCopyParser<'a> {
    pub fn new(content: &'a [u8]) -> Self {
        Self { content, position: 0 }
    }
    
    /// 解析下一個 token (零複製)
    pub fn next_token(&mut self) -> Option<&'a [u8]> {
        // 跳過空白字符
        while self.position < self.content.len() && 
              self.content[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
        
        if self.position >= self.content.len() {
            return None;
        }
        
        let start = self.position;
        
        // 找到 token 結束位置
        while self.position < self.content.len() && 
              !self.content[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
        
        Some(&self.content[start..self.position])
    }
    
    /// 解析行 (零複製)
    pub fn next_line(&mut self) -> Option<&'a [u8]> {
        if self.position >= self.content.len() {
            return None;
        }
        
        let start = self.position;
        
        // 找到換行符
        while self.position < self.content.len() {
            if self.content[self.position] == b'\n' {
                let line = &self.content[start..self.position];
                self.position += 1; // 跳過換行符
                return Some(line);
            }
            self.position += 1;
        }
        
        // 檔案末尾沒有換行符的情況
        if start < self.content.len() {
            Some(&self.content[start..])
        } else {
            None
        }
    }
}

/// SIMD 加速的雜湊計算
#[cfg(target_arch = "x86_64")]
mod simd_hash {
    use std::arch::x86_64::*;
    
    /// 使用 SIMD 指令加速雜湊計算
    pub unsafe fn fast_hash(data: &[u8]) -> u64 {
        let mut hash = 0u64;
        let chunks = data.chunks_exact(32);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // 載入 32 bytes 到 SIMD 暫存器
            let vector = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // 執行向量化運算
            let hash_vector = _mm256_set1_epi64x(hash as i64);
            let result = _mm256_xor_si256(vector, hash_vector);
            
            // 提取結果
            let mut result_bytes = [0u8; 32];
            _mm256_storeu_si256(result_bytes.as_mut_ptr() as *mut __m256i, result);
            
            // 合併結果
            for &byte in &result_bytes {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
            }
        }
        
        // 處理剩餘字節
        for &byte in remainder {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        
        hash
    }
}
```

---

## 🔧 **部署與維運**

### **🐳 多階段 Docker 建置**
```dockerfile
# Dockerfile.rust
# 階段 1: 建置環境
FROM rust:1.75-alpine AS builder

# 安裝必要工具
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    git

WORKDIR /app

# 複製 Cargo 檔案
COPY Cargo.toml Cargo.lock ./

# 建立空的 src 目錄並建置依賴項（快取最佳化）
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# 複製實際原始碼
COPY src ./src

# 重新建置應用程式
RUN touch src/main.rs && \
    cargo build --release

# 階段 2: 執行環境
FROM alpine:latest

RUN apk --no-cache add \
    ca-certificates \
    tzdata

WORKDIR /app

# 複製二進位檔案
COPY --from=builder /app/target/release/sast-analyzer ./
COPY --from=builder /app/target/release/dag-analyzer ./

# 建立非 root 使用者
RUN addgroup -g 1001 -S aiva && \
    adduser -S -D -H -u 1001 -h /app -s /sbin/nologin -G aiva -g aiva aiva

USER aiva

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ./sast-analyzer --health-check || exit 1

EXPOSE 8080

CMD ["./sast-analyzer"]
```

### **🚀 Kubernetes 部署**
```yaml
# k8s-rust-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiva-rust-core
  labels:
    app: aiva-rust-core
    component: security-engine
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: aiva-rust-core
  template:
    metadata:
      labels:
        app: aiva-rust-core
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
      containers:
      - name: sast-analyzer
        image: aiva/rust-core:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: RUST_BACKTRACE
          value: "1"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aiva-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: aiva-rust-config
---
apiVersion: v1
kind: Service
metadata:
  name: aiva-rust-service
  labels:
    app: aiva-rust-core
spec:
  ports:
  - port: 80
    targetPort: 8080
    name: http
  - port: 9090
    targetPort: 9090
    name: metrics
  selector:
    app: aiva-rust-core
```

---

**📝 版本**: v2.0 - Rust Development Guide  
**🔄 最後更新**: 2024-10-24  
**🦀 Rust 版本**: 1.75+  
**👥 維護團隊**: AIVA Rust Core Team

*這是 AIVA Features 模組 Rust 組件的完整開發指南，專注於高效能、記憶體安全和系統層安全分析功能的實現。*