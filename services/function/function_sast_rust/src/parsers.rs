// 語法解析器模組

use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Tree};

pub struct CodeParser {
    parser: Parser,
}

impl CodeParser {
    pub fn new(language: &str) -> Result<Self> {
        let mut parser = Parser::new();
        
        let tree_sitter_lang = match language.to_lowercase().as_str() {
            "python" => tree_sitter_python::language(),
            "javascript" | "js" => tree_sitter_javascript::language(),
            "go" => tree_sitter_go::language(),
            "java" => tree_sitter_java::language(),
            _ => anyhow::bail!("Unsupported language: {}", language),
        };
        
        parser
            .set_language(tree_sitter_lang)
            .context("Failed to set parser language")?;
        
        Ok(Self { parser })
    }
    
    pub fn parse(&mut self, source_code: &str) -> Result<Tree> {
        self.parser
            .parse(source_code, None)
            .context("Failed to parse source code")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_python() {
        let mut parser = CodeParser::new("python").unwrap();
        let code = "def hello():\n    print('world')";
        let tree = parser.parse(code).unwrap();
        assert!(tree.root_node().has_error() == false);
    }
}
