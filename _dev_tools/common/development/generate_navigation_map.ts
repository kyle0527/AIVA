/**
 * 生成函數調用導航圖
 * 用於快速定位和導航到詳細流程圖
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as ts from 'typescript';

interface FunctionInfo {
  name: string;
  fileName: string;
  calls: string[];  // 調用的其他函數
  hasCallback: boolean;  // 是否包含 callback
  mmdFile: string;  // 對應的 mmd 檔案
}

/**
 * 分析單一檔案的函數調用關係
 */
function analyzeFunctionCalls(filePath: string): FunctionInfo[] {
  const functions: FunctionInfo[] = [];
  const source = fs.readFileSync(filePath, 'utf-8');
  const sourceFile = ts.createSourceFile(
    filePath,
    source,
    ts.ScriptTarget.Latest,
    true
  );

  function visit(node: ts.Node, parentFunc?: FunctionInfo) {
    // 記錄函數定義
    if (
      ts.isFunctionDeclaration(node) ||
      ts.isMethodDeclaration(node) ||
      ts.isArrowFunction(node)
    ) {
      let funcName = 'anonymous';
      
      if (ts.isFunctionDeclaration(node) && node.name) {
        funcName = node.name.text;
      } else if (ts.isMethodDeclaration(node) && ts.isIdentifier(node.name)) {
        funcName = node.name.text;
      }

      const info: FunctionInfo = {
        name: funcName,
        fileName: path.basename(filePath, path.extname(filePath)),
        calls: [],
        hasCallback: false,
        mmdFile: `${path.basename(filePath, path.extname(filePath))}_${funcName}.mmd`
      };

      functions.push(info);

      // 遞迴分析函數體
      if (node.body) {
        ts.forEachChild(node.body, (child) => visit(child, info));
      }
    }

    // 記錄函數調用
    if (ts.isCallExpression(node) && parentFunc) {
      const callExpr = node.expression.getText(sourceFile);
      
      // 提取函數名（去除物件前綴）
      const match = callExpr.match(/(?:await\s+)?(?:\w+\.)*(\w+)$/);
      if (match) {
        parentFunc.calls.push(match[1]);
      }

      // 檢查是否有 callback 參數
      for (const arg of node.arguments) {
        if (ts.isArrowFunction(arg) || ts.isFunctionExpression(arg)) {
          parentFunc.hasCallback = true;
        }
      }
    }

    ts.forEachChild(node, (child) => visit(child, parentFunc));
  }

  visit(sourceFile);
  return functions;
}

/**
 * 生成 Mermaid 導航圖
 */
function generateNavigationMap(functions: FunctionInfo[]): string {
  let mermaid = 'flowchart TB\n';
  
  // 建立節點
  functions.forEach((func, idx) => {
    const nodeId = `f${idx}`;
    const label = func.name === 'anonymous' 
      ? `${func.name}<br/>(callback)` 
      : func.name;
    
    // 使用不同形狀標示不同類型
    if (func.name === 'main') {
      mermaid += `    ${nodeId}[["⭐ ${label}"]] \n`;
    } else if (func.hasCallback) {
      mermaid += `    ${nodeId}["📦 ${label}<br/>(含callback)"] \n`;
    } else if (func.name === 'anonymous') {
      mermaid += `    ${nodeId}>"🔄 ${label}"] \n`;
    } else {
      mermaid += `    ${nodeId}["${label}"] \n`;
    }
  });

  mermaid += '\n';

  // 建立調用關係
  functions.forEach((func, idx) => {
    const nodeId = `f${idx}`;
    
    func.calls.forEach(calledFunc => {
      const targetIdx = functions.findIndex(f => f.name === calledFunc);
      if (targetIdx >= 0) {
        mermaid += `    ${nodeId} --> f${targetIdx}\n`;
      }
    });

    // 如果有 callback，連接到 anonymous
    if (func.hasCallback) {
      const anonIdx = functions.findIndex(f => f.name === 'anonymous');
      if (anonIdx >= 0 && anonIdx !== idx) {
        mermaid += `    ${nodeId} -.callback.-> f${anonIdx}\n`;
      }
    }
  });

  mermaid += '\n';

  // 添加點擊連結（註解說明）
  mermaid += '    %% 對應的詳細流程圖檔案：\n';
  functions.forEach((func, idx) => {
    mermaid += `    %% f${idx}: ${func.mmdFile}\n`;
  });

  return mermaid;
}

/**
 * 生成 Markdown 索引
 */
function generateMarkdownIndex(functions: FunctionInfo[]): string {
  let md = '# 函數流程圖導航索引\n\n';
  md += '> 此檔案由工具自動生成，用於快速定位和導航到詳細流程圖\n\n';
  md += '## 📊 函數調用關係圖\n\n';
  md += '```mermaid\n';
  md += generateNavigationMap(functions);
  md += '```\n\n';
  
  md += '## 📋 函數詳細清單\n\n';
  md += '| 函數名稱 | 類型 | 調用的函數 | 詳細流程圖 |\n';
  md += '|---------|------|-----------|----------|\n';
  
  functions.forEach(func => {
    const type = func.name === 'main' ? '⭐ 入口' :
                 func.hasCallback ? '📦 含callback' :
                 func.name === 'anonymous' ? '🔄 callback' : '📝 一般';
    const calls = func.calls.length > 0 ? func.calls.join(', ') : '-';
    md += `| \`${func.name}\` | ${type} | ${calls} | [${func.mmdFile}](${func.mmdFile}) |\n`;
  });

  md += '\n## 🔍 除錯指南\n\n';
  md += '### 快速定位問題\n\n';
  md += '1. **查看調用關係圖**：找到出問題的函數在哪個位置\n';
  md += '2. **點擊對應的 .mmd 檔案**：查看該函數的詳細流程\n';
  md += '3. **追蹤調用鏈**：從 main → initialize/consumeTasks → callback\n\n';
  md += '### 常見除錯場景\n\n';
  md += '- **初始化失敗**：查看 `initialize.mmd`\n';
  md += '- **任務處理錯誤**：查看 `anonymous.mmd`（callback 邏輯）\n';
  md += '- **關閉流程問題**：查看 `shutdown.mmd`\n\n';

  return md;
}

/**
 * 主程式
 */
function main() {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.log('用法: ts-node generate_navigation_map.ts <input-file> <output-dir>');
    console.log('範例: ts-node generate_navigation_map.ts src/index.ts output/');
    process.exit(1);
  }

  const inputFile = args[0];
  const outputDir = args[1];

  if (!fs.existsSync(inputFile)) {
    console.error(`❌ 檔案不存在: ${inputFile}`);
    process.exit(1);
  }

  console.log('================================================================================');
  console.log('函數調用關係分析工具');
  console.log('================================================================================');
  console.log(`輸入檔案: ${inputFile}`);
  console.log(`輸出目錄: ${outputDir}`);
  console.log('================================================================================\n');

  // 分析函數
  const functions = analyzeFunctionCalls(inputFile);
  console.log(`✅ 找到 ${functions.length} 個函數`);

  // 確保輸出目錄存在
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // 生成導航圖
  const navMap = generateNavigationMap(functions);
  const navMapPath = path.join(outputDir, 'NAVIGATION.mmd');
  fs.writeFileSync(navMapPath, navMap);
  console.log(`✅ 生成導航圖: ${navMapPath}`);

  // 生成 Markdown 索引
  const mdIndex = generateMarkdownIndex(functions);
  const mdIndexPath = path.join(outputDir, 'INDEX.md');
  fs.writeFileSync(mdIndexPath, mdIndex);
  console.log(`✅ 生成索引文件: ${mdIndexPath}`);

  console.log('\n================================================================================');
  console.log('✅ 完成！');
  console.log('================================================================================');
}

main();
