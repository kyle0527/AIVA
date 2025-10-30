const fs = require('fs');

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
        const functionRegex = /(?:function\s+|const\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|function\s*\([^)]*\)))|(?:async\s+function\s+)/g;
        let match;
        while ((match = functionRegex.exec(content)) !== null) {
            result.functions.push({
                name: "extracted_function",
                parameters: [],
                returns: [],
                line: content.substring(0, match.index).split('\n').length
            });
        }
        
        // 提取類別
        const classRegex = /class\s+(\w+)/g;
        while ((match = classRegex.exec(content)) !== null) {
            result.classes.push({
                name: match[1],
                fields: [],
                line: content.substring(0, match.index).split('\n').length
            });
        }
        
        // 提取介面
        const interfaceRegex = /interface\s+(\w+)/g;
        while ((match = interfaceRegex.exec(content)) !== null) {
            result.interfaces.push({
                name: match[1],
                fields: [],
                line: content.substring(0, match.index).split('\n').length
            });
        }
        
        // 提取導入
        const importRegex = /import\s+.*?from\s+['"]([^'"]+)['"]/g;
        while ((match = importRegex.exec(content)) !== null) {
            result.imports.push(match[1]);
            result.dependencies.push(match[1]);
        }
        
        // 提取 require
        const requireRegex = /require\s*\(\s*['"]([^'"]+)['"]\s*\)/g;
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

analyzeFile(process.argv[2]);