// 靜態分析器

use crate::models::SastIssue;
// use crate::parsers::CodeParser; // Reserved for future AST-based analysis
use crate::rules::RuleEngine;
use anyhow::Result;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

pub struct StaticAnalyzer {
    rule_engine: RuleEngine,
}

impl StaticAnalyzer {
    pub fn new() -> Self {
        Self {
            rule_engine: RuleEngine::new(),
        }
    }
    
    pub async fn analyze_file(&mut self, file_path: &str) -> Result<Vec<SastIssue>> {
        let source_code = fs::read_to_string(file_path)?;
        
        // 從檔案副檔名判斷語言
        let language = self.detect_language(file_path);
        
        // 使用規則引擎檢查
        let mut issues = self.rule_engine.check_code(&source_code, &language);
        
        // 填充檔案路徑
        for issue in &mut issues {
            issue.file_path = file_path.to_string();
        }
        
        Ok(issues)
    }
    
    pub async fn analyze_directory(&mut self, dir_path: &str) -> Result<Vec<SastIssue>> {
        let mut all_issues = Vec::new();
        
        for entry in WalkDir::new(dir_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            if !path.is_file() {
                continue;
            }
            
            // 過濾程式碼文件
            if !self.is_code_file(path) {
                continue;
            }
            
            match self.analyze_file(path.to_str().unwrap()).await {
                Ok(issues) => all_issues.extend(issues),
                Err(e) => {
                    tracing::warn!("Failed to analyze {}: {}", path.display(), e);
                }
            }
        }
        
        Ok(all_issues)
    }
    
    fn detect_language(&self, file_path: &str) -> String {
        match Path::new(file_path).extension().and_then(|s| s.to_str()) {
            Some("py") => "python".to_string(),
            Some("js") | Some("jsx") => "javascript".to_string(),
            Some("go") => "go".to_string(),
            Some("java") => "java".to_string(),
            _ => "unknown".to_string(),
        }
    }
    
    fn is_code_file(&self, path: &Path) -> bool {
        match path.extension().and_then(|s| s.to_str()) {
            Some("py") | Some("js") | Some("jsx") | Some("go") | Some("java") => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_detect_language() {
        let analyzer = StaticAnalyzer::new();
        assert_eq!(analyzer.detect_language("test.py"), "python");
        assert_eq!(analyzer.detect_language("test.js"), "javascript");
        assert_eq!(analyzer.detect_language("test.go"), "go");
    }
}
