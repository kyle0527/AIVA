// paths_config.rs - Rust 路徑配置 (v2.1)
// 與 Python paths_config.py 同步
// 數據存放位置: services/integration/data/internal_exploration/
//
// ⚠️ 重要：AI Core 只有 Python，Rust 分析工具僅用於分析外部模組
// 內部模組 (aiva_core) 使用 Python 分析工具
// 外部模組 (features/scan) 使用多語言分析工具 (包含此 Rust 工具)

use std::path::PathBuf;
use std::env;

pub struct PathsConfig {
    pub project_root: PathBuf,
    pub services_root: PathBuf,
    pub integration_root: PathBuf,
    pub integration_data_root: PathBuf,
    
    // Internal Exploration 路徑 (統一數據存放位置)
    pub internal_exploration_data: PathBuf,
    pub analysis_results_dir: PathBuf,
    pub analysis_history_dir: PathBuf,
    pub self_healing_dir: PathBuf,
    
    // 分類數據路徑
    pub internal_classification_file: PathBuf,
    pub external_classification_file: PathBuf,
    pub combined_classification_file: PathBuf,
    
    // 其他整合模組路徑
    pub attack_paths_dir: PathBuf,
    pub experiences_dir: PathBuf,
    pub training_data_dir: PathBuf,
    
    // 環境變量控制
    pub use_integrated_paths: bool,
}

impl PathsConfig {
    pub fn new() -> Self {
        // 嘗試從環境變量獲取專案根目錄
        let project_root = env::var("AIVA_PROJECT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                // 獲取當前工作目錄並向上尋找
                let current_dir = env::current_dir().unwrap_or_default();
                Self::find_project_root(&current_dir)
            });
        
        let services_root = project_root.join("services");
        let integration_root = services_root.join("integration");
        let integration_data_root = integration_root.join("data");
        
        let internal_exploration_data = integration_data_root.join("internal_exploration");
        
        // 檢查環境變量
        let use_integrated_paths = env::var("AIVA_USE_INTEGRATED_PATHS")
            .map(|v| v != "false")
            .unwrap_or(true);
        
        PathsConfig {
            project_root: project_root.clone(),
            services_root,
            integration_root,
            integration_data_root: integration_data_root.clone(),
            internal_exploration_data: internal_exploration_data.clone(),
            analysis_results_dir: internal_exploration_data.join("analysis_results"),
            analysis_history_dir: internal_exploration_data.join("analysis_history"),
            self_healing_dir: internal_exploration_data.join("self_healing"),
            
            // 分類數據文件路徑
            internal_classification_file: internal_exploration_data.join("internal_classification.json"),
            external_classification_file: internal_exploration_data.join("external_classification.json"),
            combined_classification_file: internal_exploration_data.join("classification_data.json"),
            
            attack_paths_dir: integration_data_root.join("attack_paths"),
            experiences_dir: integration_data_root.join("experiences"),
            training_data_dir: integration_data_root.join("training"),
            use_integrated_paths,
        }
    }
    
    /// 尋找專案根目錄 (包含 services/ 目錄的位置)
    fn find_project_root(start: &Path) -> PathBuf {
        let mut current = start.to_path_buf();
        
        // 向上搜尋最多 10 層
        for _ in 0..10 {
            if current.join("services").exists() && current.join("services").join("core").exists() {
                return current;
            }
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                break;
            }
        }
        
        // 如果找不到，返回當前目錄
        start.to_path_buf()
    }
    
    /// 確保所有目錄存在
    pub fn ensure_directories(&self) -> std::io::Result<()> {
        let dirs = vec![
            &self.internal_exploration_data,
            &self.analysis_results_dir,
            &self.analysis_history_dir,
            &self.self_healing_dir,
            &self.attack_paths_dir,
            &self.experiences_dir,
            &self.training_data_dir,
        ];
        
        for dir in dirs {
            std::fs::create_dir_all(dir)?;
        }
        
        // 確保語言特定的輸出目錄存在
        for lang in &["rust", "go", "typescript", "python"] {
            std::fs::create_dir_all(self.analysis_results_dir.join(lang))?;
        }
        
        Ok(())
    }
    
    /// 獲取特定工具的輸出目錄
    pub fn get_analysis_output_dir(&self, tool_name: &str) -> PathBuf {
        if tool_name == "self_healing" {
            return self.self_healing_dir.clone();
        }
        
        let tool_dir = self.analysis_results_dir.join(tool_name);
        let _ = std::fs::create_dir_all(&tool_dir);
        tool_dir
    }
    
    /// 獲取默認輸出目錄（向後兼容）
    pub fn get_default_output_dir(&self) -> PathBuf {
        if self.use_integrated_paths {
            let _ = self.ensure_directories();
            self.get_analysis_output_dir("rust")
        } else {
            // 向後兼容：使用舊路徑
            PathBuf::from("./analysis_output")
        }
    }
    
    /// 打印路徑摘要 (調試用)
    pub fn print_summary(&self) {
        println!("🔧 AIVA Rust 路徑配置");
        println!("   PROJECT_ROOT: {:?}", self.project_root);
        println!("   DATA_ROOT: {:?}", self.internal_exploration_data);
        println!("   RUST_OUTPUT: {:?}", self.get_analysis_output_dir("rust"));
        println!("   CLASSIFICATION: {:?}", self.external_classification_file);
    }
}

use std::path::Path;

impl Default for PathsConfig {
    fn default() -> Self {
        Self::new()
    }
}
