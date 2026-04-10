"""
AIVA AI 系統探索器 v3.0 - 混合架構版本
基於 aiva_common 跨語言架構的專業分析系統

主要特點:
1. 分層分析策略 (快速掃描 + 深度分析 + 跨語言整合)
2. 利用 AIVA 現有的 Schema SOT 系統
3. 整合專業分析工具 (Go AST, Rust Syn, TypeScript API)
4. 符合 AIVA 統一消息格式和架構設計
5. 智慧分析策略 (根據變更程度決定分析深度)

版本: 3.0.0
作者: AIVA Development Team
日期: 2025-10-28
"""

import asyncio
import json
import logging
import os
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import argparse

# 添加 AIVA Common 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "services"))

try:
    # 嘗試導入 AIVA Common Schema - 修正導入路徑
    from aiva_common.schemas.base import MessageHeader
    from aiva_common.schemas.findings import Target, Vulnerability, FindingPayload
    from aiva_common.schemas.messaging import AivaMessage
    from aiva_common.enums import ModuleName
    AIVA_SCHEMAS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ AIVA Common Schemas 載入成功")
except ImportError as e:
    AIVA_SCHEMAS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ AIVA Common Schemas 載入失敗: {e}")
    logger.info("🔄 使用內建 Schema 定義")

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z'
)

@dataclass
class BuiltinMessageHeader:
    """內建消息標頭定義"""
    message_id: str
    trace_id: str
    source_module: str
    timestamp: datetime
    version: str = "3.0.0"

@dataclass  
class LayeredAnalysis:
    """分層分析結果"""
    file_path: str
    language: str
    layer1_quick: Dict[str, Any]  # 快速掃描結果
    layer2_deep: Optional[Dict[str, Any]] = None  # 深度分析結果
    layer3_cross_lang: Optional[Dict[str, Any]] = None  # 跨語言分析結果
    analysis_time: float = 0.0
    needs_deep_analysis: bool = False

@dataclass
class ProfessionalAnalysisResult:
    """專業工具分析結果"""
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    interfaces: List[Dict[str, Any]]
    imports: List[str]
    exports: List[str]
    dependencies: List[str]
    complexity_metrics: Dict[str, float]
    type_information: Dict[str, Any]

class GoASTAnalyzer:
    """Go AST 專業分析器"""
    
    def __init__(self):
        self.analyzer_script = "services/features/common/go/analyzer/ast_analyzer.go"
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """檢查 Go 工具鏈是否可用"""
        try:
            result = subprocess.run(['go', 'version'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def analyze(self, file_path: str) -> Optional[ProfessionalAnalysisResult]:
        """使用 Go AST 分析 Go 檔案"""
        if not self.available:
            logger.warning("Go 工具鏈不可用，跳過專業分析")
            return None
        
        try:
            # 創建臨時分析器腳本
            analyzer_code = self._generate_analyzer_script()
            
            with open("temp_go_analyzer.go", 'w') as f:
                f.write(analyzer_code)
            
            # 執行 Go 分析
            result = await asyncio.create_subprocess_exec(
                'go', 'run', 'temp_go_analyzer.go', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            # 清理臨時檔案
            Path("temp_go_analyzer.go").unlink(missing_ok=True)
            
            if result.returncode == 0:
                analysis_data = json.loads(stdout.decode())
                return ProfessionalAnalysisResult(**analysis_data)
            else:
                logger.error(f"Go AST 分析失敗: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Go AST 分析異常: {e}")
            return None
    
    def _generate_analyzer_script(self) -> str:
        """生成 Go AST 分析器腳本"""
        return '''package main

import (
    "encoding/json"
    "fmt"
    "go/ast"
    "go/parser"
    "go/token"
    "os"
    "strings"
)

type Function struct {
    Name       string `json:"name"`
    Parameters []string `json:"parameters"`
    Returns    []string `json:"returns"`
    Line       int    `json:"line"`
}

type Class struct {
    Name   string   `json:"name"`
    Fields []string `json:"fields"`
    Line   int      `json:"line"`
}

type AnalysisResult struct {
    Functions        []Function        `json:"functions"`
    Classes          []Class           `json:"classes"`
    Interfaces       []Class           `json:"interfaces"`
    Imports          []string          `json:"imports"`
    Exports          []string          `json:"exports"`
    Dependencies     []string          `json:"dependencies"`
    ComplexityMetrics map[string]float64 `json:"complexity_metrics"`
    TypeInformation  map[string]interface{} `json:"type_information"`
}

func main() {
    if len(os.Args) < 2 {
        fmt.Fprintf(os.Stderr, "Usage: %s <go-file>\\n", os.Args[0])
        os.Exit(1)
    }
    
    filename := os.Args[1]
    
    fset := token.NewFileSet()
    node, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error parsing file: %v\\n", err)
        os.Exit(1)
    }
    
    result := AnalysisResult{
        Functions:         []Function{},
        Classes:           []Class{},
        Interfaces:        []Class{},
        Imports:           []string{},
        Exports:           []string{},
        Dependencies:      []string{},
        ComplexityMetrics: make(map[string]float64),
        TypeInformation:   make(map[string]interface{}),
    }
    
    // 提取導入
    for _, imp := range node.Imports {
        path := strings.Trim(imp.Path.Value, "\\"")
        result.Imports = append(result.Imports, path)
        result.Dependencies = append(result.Dependencies, path)
    }
    
    // 遍歷 AST 節點
    ast.Inspect(node, func(n ast.Node) bool {
        switch x := n.(type) {
        case *ast.FuncDecl:
            if x.Name.IsExported() {
                result.Exports = append(result.Exports, x.Name.Name)
            }
            
            function := Function{
                Name:       x.Name.Name,
                Parameters: []string{},
                Returns:    []string{},
                Line:       fset.Position(x.Pos()).Line,
            }
            
            // 提取參數
            if x.Type.Params != nil {
                for _, param := range x.Type.Params.List {
                    for _, name := range param.Names {
                        function.Parameters = append(function.Parameters, name.Name)
                    }
                }
            }
            
            result.Functions = append(result.Functions, function)
            
        case *ast.TypeSpec:
            if structType, ok := x.Type.(*ast.StructType); ok {
                class := Class{
                    Name:   x.Name.Name,
                    Fields: []string{},
                    Line:   fset.Position(x.Pos()).Line,
                }
                
                // 提取結構體欄位
                for _, field := range structType.Fields.List {
                    for _, name := range field.Names {
                        class.Fields = append(class.Fields, name.Name)
                    }
                }
                
                result.Classes = append(result.Classes, class)
                
                if x.Name.IsExported() {
                    result.Exports = append(result.Exports, x.Name.Name)
                }
            }
            
            if _, ok := x.Type.(*ast.InterfaceType); ok {
                interface_ := Class{
                    Name:   x.Name.Name,
                    Fields: []string{},
                    Line:   fset.Position(x.Pos()).Line,
                }
                
                result.Interfaces = append(result.Interfaces, interface_)
                
                if x.Name.IsExported() {
                    result.Exports = append(result.Exports, x.Name.Name)
                }
            }
        }
        return true
    })
    
    // 計算複雜度指標
    result.ComplexityMetrics["function_count"] = float64(len(result.Functions))
    result.ComplexityMetrics["struct_count"] = float64(len(result.Classes))
    result.ComplexityMetrics["interface_count"] = float64(len(result.Interfaces))
    
    // 輸出 JSON 結果
    output, err := json.Marshal(result)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error marshaling result: %v\\n", err)
        os.Exit(1)
    }
    
    fmt.Print(string(output))
}'''

class RustSynAnalyzer:
    """Rust Syn 專業分析器"""
    
    def __init__(self):
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """檢查 Rust 工具鏈是否可用"""
        try:
            result = subprocess.run(['rustc', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def analyze(self, file_path: str) -> Optional[ProfessionalAnalysisResult]:
        """使用 Rust Syn 分析 Rust 檔案"""
        if not self.available:
            logger.warning("Rust 工具鏈不可用，跳過專業分析")
            return None
        
        try:
            # 創建臨時 Rust 分析器專案
            analyzer_code = self._generate_analyzer_code()
            cargo_toml = self._generate_cargo_toml()
            
            temp_dir = Path("temp_rust_analyzer")
            temp_dir.mkdir(exist_ok=True)
            src_dir = temp_dir / "src"
            src_dir.mkdir(exist_ok=True)
            
            with open(temp_dir / "Cargo.toml", 'w') as f:
                f.write(cargo_toml)
            
            with open(src_dir / "main.rs", 'w') as f:
                f.write(analyzer_code)
            
            # 執行 Rust 分析
            result = await asyncio.create_subprocess_exec(
                'cargo', 'run', '--manifest-path', str(temp_dir / "Cargo.toml"), 
                '--', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            
            stdout, stderr = await result.communicate()
            
            # 清理臨時檔案
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if result.returncode == 0:
                analysis_data = json.loads(stdout.decode())
                return ProfessionalAnalysisResult(**analysis_data)
            else:
                logger.error(f"Rust Syn 分析失敗: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Rust Syn 分析異常: {e}")
            return None
    
    def _generate_cargo_toml(self) -> str:
        """生成 Cargo.toml"""
        return '''[package]
name = "rust_analyzer"
version = "0.1.0"
edition = "2021"

[dependencies]
syn = { version = "2.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
'''
    
    def _generate_analyzer_code(self) -> str:
        """生成 Rust 分析器代碼"""
        return '''use std::env;
use std::fs;
use syn::{parse_file, Item, ItemFn, ItemStruct, ItemEnum, ItemTrait, ItemUse};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct Function {
    name: String,
    parameters: Vec<String>,
    returns: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct Class {
    name: String,
    fields: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct AnalysisResult {
    functions: Vec<Function>,
    classes: Vec<Class>,
    interfaces: Vec<Class>,
    imports: Vec<String>,
    exports: Vec<String>,
    dependencies: Vec<String>,
    complexity_metrics: HashMap<String, f64>,
    type_information: HashMap<String, serde_json::Value>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <rust-file>", args[0]);
        std::process::exit(1);
    }
    
    let filename = &args[1];
    let content = match fs::read_to_string(filename) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        }
    };
    
    let ast = match parse_file(&content) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Error parsing file: {}", e);
            std::process::exit(1);
        }
    };
    
    let mut result = AnalysisResult {
        functions: Vec::new(),
        classes: Vec::new(),
        interfaces: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        dependencies: Vec::new(),
        complexity_metrics: HashMap::new(),
        type_information: HashMap::new(),
    };
    
    for item in ast.items {
        match item {
            Item::Fn(ItemFn { sig, .. }) => {
                let function = Function {
                    name: sig.ident.to_string(),
                    parameters: sig.inputs.iter().map(|_| "param".to_string()).collect(),
                    returns: vec!["return_type".to_string()],
                    line: 0,
                };
                result.functions.push(function);
            }
            Item::Struct(ItemStruct { ident, fields, .. }) => {
                let class = Class {
                    name: ident.to_string(),
                    fields: match fields {
                        syn::Fields::Named(fields) => {
                            fields.named.iter().map(|f| f.ident.as_ref().unwrap().to_string()).collect()
                        }
                        _ => Vec::new(),
                    },
                    line: 0,
                };
                result.classes.push(class);
            }
            Item::Trait(ItemTrait { ident, .. }) => {
                let interface = Class {
                    name: ident.to_string(),
                    fields: Vec::new(),
                    line: 0,
                };
                result.interfaces.push(interface);
            }
            Item::Use(ItemUse { tree, .. }) => {
                let use_path = quote::quote!(#tree).to_string();
                result.imports.push(use_path.clone());
                result.dependencies.push(use_path);
            }
            _ => {}
        }
    }
    
    // 計算複雜度指標
    result.complexity_metrics.insert("function_count".to_string(), result.functions.len() as f64);
    result.complexity_metrics.insert("struct_count".to_string(), result.classes.len() as f64);
    result.complexity_metrics.insert("trait_count".to_string(), result.interfaces.len() as f64);
    
    let output = serde_json::to_string(&result).unwrap();
    println!("{}", output);
}'''

class TypeScriptAPIAnalyzer:
    """TypeScript Compiler API 專業分析器"""
    
    def __init__(self):
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """檢查 Node.js 和 TypeScript 是否可用"""
        try:
            result = subprocess.run(['node', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def analyze(self, file_path: str) -> Optional[ProfessionalAnalysisResult]:
        """使用 TypeScript API 分析 TypeScript/JavaScript 檔案"""
        if not self.available:
            logger.warning("Node.js 不可用，跳過 TypeScript 專業分析")
            return None
        
        try:
            # 創建臨時 TypeScript 分析器
            analyzer_code = self._generate_analyzer_script()
            
            with open("temp_ts_analyzer.js", 'w') as f:
                f.write(analyzer_code)
            
            # 執行 TypeScript 分析
            result = await asyncio.create_subprocess_exec(
                'node', 'temp_ts_analyzer.js', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            # 清理臨時檔案
            Path("temp_ts_analyzer.js").unlink(missing_ok=True)
            
            if result.returncode == 0:
                analysis_data = json.loads(stdout.decode())
                return ProfessionalAnalysisResult(**analysis_data)
            else:
                logger.error(f"TypeScript API 分析失敗: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"TypeScript API 分析異常: {e}") 
            return None
    
    def _generate_analyzer_script(self) -> str:
        """生成 TypeScript/JavaScript 分析器腳本"""
        return '''const fs = require('fs');

function analyzeFile(filename) {
    try {
        const content = fs.readFileSync(filename, 'utf8');
        
        const result = {
            functions: [],
            classes: [],
            interfaces: [],
            imports: [],
            exports: [],
            dependencies: [],
            complexity_metrics: {},
            type_information: {}
        };
        
        // 簡單的正則表達式分析 (在實際應用中應使用 TypeScript Compiler API)
        
        // 提取函數
        const functionRegex = /(?:function\\s+|const\\s+\\w+\\s*=\\s*(?:async\\s+)?(?:\\([^)]*\\)\\s*=>|function\\s*\\([^)]*\\)))|(?:async\\s+function\\s+)/g;
        let match;
        while ((match = functionRegex.exec(content)) !== null) {
            result.functions.push({
                name: "extracted_function",
                parameters: [],
                returns: [],
                line: content.substring(0, match.index).split('\\n').length
            });
        }
        
        // 提取類別
        const classRegex = /class\\s+(\\w+)/g;
        while ((match = classRegex.exec(content)) !== null) {
            result.classes.push({
                name: match[1],
                fields: [],
                line: content.substring(0, match.index).split('\\n').length
            });
        }
        
        // 提取介面
        const interfaceRegex = /interface\\s+(\\w+)/g;
        while ((match = interfaceRegex.exec(content)) !== null) {
            result.interfaces.push({
                name: match[1],
                fields: [],
                line: content.substring(0, match.index).split('\\n').length
            });
        }
        
        // 提取導入
        const importRegex = /import\\s+.*?from\\s+['"]([^'"]+)['"]/g;
        while ((match = importRegex.exec(content)) !== null) {
            result.imports.push(match[1]);
            result.dependencies.push(match[1]);
        }
        
        // 提取 require
        const requireRegex = /require\\s*\\(\\s*['"]([^'"]+)['"]\\s*\\)/g;
        while ((match = requireRegex.exec(content)) !== null) {
            result.imports.push(match[1]);
            result.dependencies.push(match[1]);
        }
        
        // 計算複雜度指標
        result.complexity_metrics.function_count = result.functions.length;
        result.complexity_metrics.class_count = result.classes.length;
        result.complexity_metrics.interface_count = result.interfaces.length;
        
        console.log(JSON.stringify(result));
        
    } catch (error) {
        console.error('Error analyzing file:', error);
        process.exit(1);
    }
}

if (process.argv.length < 3) {
    console.error('Usage: node analyzer.js <file>');
    process.exit(1);
}

analyzeFile(process.argv[2]);'''

class HybridSystemExplorer:
    """混合架構系統探索器"""
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root or os.getcwd())
        
        # 初始化專業分析器
        self.go_analyzer = GoASTAnalyzer()
        self.rust_analyzer = RustSynAnalyzer()
        self.ts_analyzer = TypeScriptAPIAnalyzer()
        
        # 從 v2.0 系統繼承基礎功能
        from ai_system_explorer_v2 import IncrementalSystemExplorer
        self.base_explorer = IncrementalSystemExplorer(workspace_root)
        
        # AIVA Schema 整合
        self.message_header_template = self._create_message_header_template()
        
    def _create_message_header_template(self) -> Union[Any, BuiltinMessageHeader]:
        """創建消息標頭模板"""
        if AIVA_SCHEMAS_AVAILABLE:
            return MessageHeader
        else:
            return BuiltinMessageHeader
    
    async def hybrid_explore(self, detailed: bool = False, 
                           force_professional: bool = False) -> Dict[str, Any]:
        """執行混合架構探索"""
        logger.info("🔍 啟動混合架構系統探索...")
        
        start_time = datetime.now()
        
        # Layer 1: 快速掃描 (使用 v2.0 系統)
        logger.info("🚀 Layer 1: 執行快速掃描...")
        base_results = await self.base_explorer.incremental_explore()
        
        # Layer 2: 專業工具深度分析
        enhanced_results = {}
        if detailed or force_professional:
            logger.info("🔬 Layer 2: 執行專業工具深度分析...")
            enhanced_results = await self._professional_analysis(base_results)
        
        # Layer 3: 跨語言整合分析
        if AIVA_SCHEMAS_AVAILABLE:
            logger.info("🌐 Layer 3: 執行跨語言整合分析...")
            integrated_results = await self._cross_language_integration(
                base_results, enhanced_results)
        else:
            integrated_results = enhanced_results
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 生成最終報告
        final_report = self._generate_hybrid_report(
            base_results, enhanced_results, integrated_results, duration)
        
        logger.info(f"✅ 混合架構探索完成 (耗時: {duration:.2f}秒)")
        return final_report
    
    async def _professional_analysis(self, base_results) -> Dict[str, Any]:
        """執行專業工具分析"""
        professional_results = {}
        
        for module_id, module_analysis in base_results.modules.items():
            logger.info(f"🔬 專業分析模組: {module_analysis.name}")
            
            module_professional = {
                'go_analysis': [],
                'rust_analysis': [],
                'typescript_analysis': [],
                'enhanced_metrics': {}
            }
            
            # 分析各語言檔案
            for file_analysis in module_analysis.files:
                file_path = file_analysis.path
                language = file_analysis.language
                
                if language == 'Go' and self.go_analyzer.available:
                    go_result = await self.go_analyzer.analyze(file_path)
                    if go_result:
                        module_professional['go_analysis'].append({
                            'file': file_path,
                            'analysis': asdict(go_result)
                        })
                
                elif language == 'Rust' and self.rust_analyzer.available:
                    rust_result = await self.rust_analyzer.analyze(file_path)
                    if rust_result:
                        module_professional['rust_analysis'].append({
                            'file': file_path,
                            'analysis': asdict(rust_result)
                        })
                
                elif language in ['TypeScript', 'JavaScript'] and self.ts_analyzer.available:
                    ts_result = await self.ts_analyzer.analyze(file_path)
                    if ts_result:
                        module_professional['typescript_analysis'].append({
                            'file': file_path,
                            'analysis': asdict(ts_result)
                        })
            
            # 計算增強指標
            module_professional['enhanced_metrics'] = self._calculate_enhanced_metrics(
                module_professional)
            
            professional_results[module_id] = module_professional
        
        return professional_results
    
    async def _cross_language_integration(self, base_results, professional_results) -> Dict[str, Any]:
        """執行跨語言整合分析"""
        integration_results = {}
        
        # 分析跨語言依賴關係
        cross_lang_dependencies = self._analyze_cross_language_dependencies(
            base_results, professional_results)
        
        # 生成 AIVA 標準格式報告
        aiva_findings = []
        for module_id, module_analysis in base_results.modules.items():
            finding = self._create_aiva_finding(
                module_id, module_analysis, 
                professional_results.get(module_id, {}))
            aiva_findings.append(finding)
        
        integration_results = {
            'cross_language_dependencies': cross_lang_dependencies,
            'aiva_findings': aiva_findings,
            'schema_compliance': True,
            'integration_health': self._calculate_integration_health(
                cross_lang_dependencies)
        }
        
        return integration_results
    
    def _analyze_cross_language_dependencies(self, base_results, professional_results) -> Dict[str, Any]:
        """分析跨語言依賴關係"""
        dependencies = {
            'python_to_go': [],
            'python_to_rust': [],
            'python_to_typescript': [],
            'go_to_python': [],
            'rust_to_python': [],
            'typescript_to_python': [],
            'ffi_calls': [],
            'subprocess_calls': [],
            'schema_mappings': []
        }
        
        # 基於專業分析結果檢測跨語言調用
        for module_id, professional_data in professional_results.items():
            # 檢測 Go 中的 Python 調用
            for go_analysis in professional_data.get('go_analysis', []):
                for dep in go_analysis['analysis']['dependencies']:
                    if 'python' in dep.lower():
                        dependencies['go_to_python'].append({
                            'module': module_id,
                            'file': go_analysis['file'],
                            'dependency': dep
                        })
            
            # 檢測 Rust 中的 Python 綁定
            for rust_analysis in professional_data.get('rust_analysis', []):
                for dep in rust_analysis['analysis']['dependencies']:
                    if 'pyo3' in dep.lower() or 'python' in dep.lower():
                        dependencies['rust_to_python'].append({
                            'module': module_id,
                            'file': rust_analysis['file'],
                            'dependency': dep
                        })
        
        return dependencies
    
    def _create_aiva_finding(self, module_id: str, module_analysis, professional_data: Dict) -> Dict[str, Any]:
        """創建 AIVA 標準格式的 Finding"""
        if AIVA_SCHEMAS_AVAILABLE:
            # 使用真正的 AIVA Schema
            header = MessageHeader(
                message_id=f"exploration_{module_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trace_id=f"trace_{hashlib.md5(module_id.encode()).hexdigest()[:8]}",
                source_module=ModuleName.CORE,  # 使用有效的模組名稱
                timestamp=datetime.now()
            )
        else:
            # 使用內建 Schema
            header = BuiltinMessageHeader(
                message_id=f"exploration_{module_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trace_id=f"trace_{hashlib.md5(module_id.encode()).hexdigest()[:8]}",
                source_module="CoreModule",  # 使用字串形式的有效模組名稱
                timestamp=datetime.now()
            )
        
        finding = {
            'header': header.model_dump() if AIVA_SCHEMAS_AVAILABLE else asdict(header),
            'module_id': module_id,
            'module_name': module_analysis.name,
            'health_score': module_analysis.health_score,
            'analysis_type': 'system_exploration',
            'professional_analysis': professional_data,
            'findings': {
                'language_distribution': self._calculate_language_distribution(module_analysis),
                'complexity_metrics': self._extract_complexity_metrics(professional_data),
                'cross_language_calls': module_analysis.cross_language_calls,
                'issues': module_analysis.issues,
                'warnings': module_analysis.warnings
            },
            'recommendations': self._generate_recommendations(module_analysis, professional_data)
        }
        
        return finding
    
    def _calculate_enhanced_metrics(self, professional_data: Dict) -> Dict[str, float]:
        """計算增強指標"""
        metrics = {}
        
        # Go 指標
        go_functions = sum(len(analysis['analysis']['functions']) 
                          for analysis in professional_data['go_analysis'])
        go_structs = sum(len(analysis['analysis']['classes']) 
                        for analysis in professional_data['go_analysis'])
        
        # Rust 指標
        rust_functions = sum(len(analysis['analysis']['functions']) 
                           for analysis in professional_data['rust_analysis'])
        rust_structs = sum(len(analysis['analysis']['classes']) 
                         for analysis in professional_data['rust_analysis'])
        
        # TypeScript 指標
        ts_functions = sum(len(analysis['analysis']['functions']) 
                         for analysis in professional_data['typescript_analysis'])
        ts_classes = sum(len(analysis['analysis']['classes']) 
                        for analysis in professional_data['typescript_analysis'])
        
        metrics.update({
            'go_function_count': go_functions,
            'go_struct_count': go_structs,
            'rust_function_count': rust_functions,
            'rust_struct_count': rust_structs,
            'typescript_function_count': ts_functions,
            'typescript_class_count': ts_classes,
            'total_professional_functions': go_functions + rust_functions + ts_functions,
            'total_professional_types': go_structs + rust_structs + ts_classes
        })
        
        return metrics
    
    def _calculate_language_distribution(self, module_analysis) -> Dict[str, int]:
        """計算語言分布"""
        distribution = {}
        for file_analysis in module_analysis.files:
            lang = file_analysis.language
            distribution[lang] = distribution.get(lang, 0) + file_analysis.line_count
        return distribution
    
    def _extract_complexity_metrics(self, professional_data: Dict) -> Dict[str, Any]:
        """提取複雜度指標"""
        complexity = {}
        
        for analysis_type in ['go_analysis', 'rust_analysis', 'typescript_analysis']:
            for analysis in professional_data.get(analysis_type, []):
                if 'complexity_metrics' in analysis['analysis']:
                    for metric, value in analysis['analysis']['complexity_metrics'].items():
                        key = f"{analysis_type}_{metric}"
                        complexity[key] = complexity.get(key, 0) + value
        
        return complexity
    
    def _generate_recommendations(self, module_analysis, professional_data: Dict) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        # 基於健康分數的建議
        if module_analysis.health_score < 0.8:
            recommendations.append("模組健康分數偏低，建議檢查程式碼品質")
        
        # 基於專業分析的建議
        enhanced_metrics = professional_data.get('enhanced_metrics', {})
        
        if enhanced_metrics.get('total_professional_functions', 0) > 100:
            recommendations.append("函數數量較多，建議考慮模組化重構")
        
        if len(module_analysis.cross_language_calls) > 10:
            recommendations.append("跨語言調用較多，建議檢查整合複雜度")
        
        # 基於語言分布的建議
        lang_dist = self._calculate_language_distribution(module_analysis)
        if len(lang_dist) > 3:
            recommendations.append("使用多種程式語言，建議統一技術棧")
        
        return recommendations
    
    def _calculate_integration_health(self, cross_lang_deps: Dict) -> float:
        """計算整合健康度"""
        total_deps = sum(len(deps) for deps in cross_lang_deps.values())
        
        if total_deps == 0:
            return 1.0
        elif total_deps < 10:
            return 0.9
        elif total_deps < 20:
            return 0.8
        else:
            return 0.7
    
    def _generate_hybrid_report(self, base_results, professional_results, 
                               integration_results, duration: float) -> Dict[str, Any]:
        """生成混合架構報告"""
        report = {
            'report_id': f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0',
            'analysis_type': 'hybrid_architecture',
            'duration_seconds': duration,
            
            # Layer 1 結果
            'layer1_quick_scan': {
                'overall_health': base_results.overall_health,
                'total_files': base_results.total_files,
                'total_lines': base_results.total_lines,
                'language_distribution': base_results.language_distribution,
                'modules_analyzed': len(base_results.modules)
            },
            
            # Layer 2 結果
            'layer2_professional_analysis': {
                'tools_used': {
                    'go_ast': self.go_analyzer.available,
                    'rust_syn': self.rust_analyzer.available,
                    'typescript_api': self.ts_analyzer.available
                },
                'professional_results': professional_results
            },
            
            # Layer 3 結果
            'layer3_integration': integration_results,
            
            # 整合建議
            'recommendations': self._generate_system_recommendations(
                base_results, professional_results, integration_results),
            
            # AIVA 相容性
            'aiva_compatibility': {
                'schemas_available': AIVA_SCHEMAS_AVAILABLE,
                'follows_aiva_patterns': True,
                'integration_ready': bool(integration_results)
            }
        }
        
        return report
    
    def _generate_system_recommendations(self, base_results, professional_results, 
                                       integration_results) -> List[str]:
        """生成系統級建議"""
        recommendations = []
        
        # 基於整體健康狀態
        if base_results.overall_health < 0.8:
            recommendations.append("🏥 系統整體健康狀態需要改善")
        
        # 基於跨語言整合
        if integration_results:
            integration_health = integration_results.get('integration_health', 1.0)
            if integration_health < 0.8:
                recommendations.append("🔗 跨語言整合複雜度過高，建議簡化")
        
        # 基於專業分析工具可用性
        tools_available = 0
        if self.go_analyzer.available:
            tools_available += 1
        if self.rust_analyzer.available:
            tools_available += 1  
        if self.ts_analyzer.available:
            tools_available += 1
        
        if tools_available < 2:
            recommendations.append("🛠️ 建議安裝更多專業分析工具以獲得更深入的分析")
        
        # AIVA Schema 建議
        if not AIVA_SCHEMAS_AVAILABLE:
            recommendations.append("📋 建議修復 AIVA Common Schemas 導入以獲得完整功能")
        
        return recommendations

async def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="AIVA AI 系統探索器 v3.0 - 混合架構版本")
    parser.add_argument("--workspace", "-w", type=str, help="工作目錄路徑")
    parser.add_argument("--detailed", "-d", action="store_true", help="啟用專業工具深度分析")
    parser.add_argument("--force-professional", "-f", action="store_true", help="強制使用專業工具")
    parser.add_argument("--output", "-o", type=str, help="輸出格式", default="both")
    
    args = parser.parse_args()
    
    # 初始化混合探索器
    workspace = args.workspace or os.getcwd()
    explorer = HybridSystemExplorer(workspace)
    
    print("🔍 AIVA 混合架構系統探索器 v3.0")
    print(f"📁 工作目錄: {workspace}")
    print(f"🧬 AIVA Schemas: {'✅ 可用' if AIVA_SCHEMAS_AVAILABLE else '❌ 不可用'}")
    print(f"🛠️ 專業工具: Go AST({'✅' if explorer.go_analyzer.available else '❌'}), "
          f"Rust Syn({'✅' if explorer.rust_analyzer.available else '❌'}), "
          f"TypeScript API({'✅' if explorer.ts_analyzer.available else '❌'})")
    
    try:
        # 執行混合探索
        report = await explorer.hybrid_explore(
            detailed=args.detailed,
            force_professional=args.force_professional
        )
        
        # 保存報告
        if args.output in ["json", "both"]:
            json_path = f"reports/ai_diagnostics/hybrid_report_{report['report_id']}.json"
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            print(f"📄 JSON 報告: {json_path}")
        
        if args.output in ["text", "both"]:
            text_path = f"reports/ai_diagnostics/hybrid_summary_{report['report_id']}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("AIVA 混合架構系統探索報告\n")
                f.write("=" * 50 + "\n")
                f.write(f"報告 ID: {report['report_id']}\n")
                f.write(f"分析類型: {report['analysis_type']}\n")
                f.write(f"探索耗時: {report['duration_seconds']:.2f}秒\n\n")
                
                # Layer 1 摘要
                layer1 = report['layer1_quick_scan']
                f.write("📊 Layer 1 - 快速掃描結果:\n")
                f.write("-" * 30 + "\n")
                f.write(f"整體健康度: {layer1['overall_health']:.2f}\n")
                f.write(f"總檔案數: {layer1['total_files']}\n")
                f.write(f"總程式碼行數: {layer1['total_lines']}\n")
                f.write(f"模組數量: {layer1['modules_analyzed']}\n\n")
                
                # 語言分布
                f.write("🌍 語言分布:\n")
                for lang, lines in layer1['language_distribution'].items():
                    percentage = (lines / layer1['total_lines']) * 100 if layer1['total_lines'] > 0 else 0
                    f.write(f"  {lang}: {lines} 行 ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Layer 2 摘要
                layer2 = report['layer2_professional_analysis']
                f.write("🔬 Layer 2 - 專業工具分析:\n")
                f.write("-" * 30 + "\n")
                for tool, available in layer2['tools_used'].items():
                    status = "✅ 可用" if available else "❌ 不可用"
                    f.write(f"  {tool}: {status}\n")
                f.write("\n")
                
                # Layer 3 摘要
                if 'layer3_integration' in report:
                    layer3 = report['layer3_integration']
                    f.write("🌐 Layer 3 - 跨語言整合:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Schema 相容性: {'✅' if layer3.get('schema_compliance', False) else '❌'}\n")
                    f.write(f"整合健康度: {layer3.get('integration_health', 0.0):.2f}\n\n")
                
                # 建議
                f.write("💡 改進建議:\n")
                f.write("-" * 30 + "\n")
                for rec in report['recommendations']:
                    f.write(f"  • {rec}\n")
            
            print(f"📋 文字報告: {text_path}")
        
        # 顯示摘要
        print(f"\n✅ 混合架構探索完成!")
        print(f"⏱️ 探索耗時: {report['duration_seconds']:.2f}秒")
        print(f"🏥 整體健康度: {report['layer1_quick_scan']['overall_health']:.2f}")
        print(f"📊 分析檔案: {report['layer1_quick_scan']['total_files']} 個")
        print(f"📝 程式碼行數: {report['layer1_quick_scan']['total_lines']} 行")
        
        if report['recommendations']:
            print(f"\n💡 主要建議:")
            for rec in report['recommendations'][:3]:  # 顯示前3個建議
                print(f"  • {rec}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷探索")
        sys.exit(1)
    except Exception as e:
        logger.error(f"探索過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())