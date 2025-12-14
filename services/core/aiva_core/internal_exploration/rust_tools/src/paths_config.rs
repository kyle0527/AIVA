// paths_config.rs - Rust 路徑配置
// 自動從 Python paths.py 生成
use std::path::PathBuf;
use std::env;

pub struct PathsConfig {
    pub project_root: PathBuf,
    pub services_root: PathBuf,
    pub integration_root: PathBuf,
    pub integration_data_root: PathBuf,
    
    // Internal Exploration 路徑
    pub internal_exploration_data: PathBuf,
    pub analysis_results_dir: PathBuf,
    pub analysis_history_dir: PathBuf,
    pub self_healing_dir: PathBuf,
    
    // 其他整合模組路徑
    pub attack_paths_dir: PathBuf,
    pub experiences_dir: PathBuf,
    pub training_data_dir: PathBuf,
    
    // 環境變量控制
    pub use_integrated_paths: bool,
}

impl PathsConfig {
    pub fn new() -> Self {
        // 獲取當前工作目錄
        // 當前文件: services/core/aiva_core/internal_exploration/rust_tools/src/paths_config.rs
        // 向上 6 層到達專案根目錄
        let current_dir = env::current_dir().unwrap();
        let project_root = current_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .unwrap_or(&current_dir)
            .to_path_buf();
        
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
            attack_paths_dir: integration_data_root.join("attack_paths"),
            experiences_dir: integration_data_root.join("experiences"),
            training_data_dir: integration_data_root.join("training"),
            use_integrated_paths,
        }
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
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self::new()
    }
}
