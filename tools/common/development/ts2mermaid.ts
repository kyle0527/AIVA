/**
 * ts2mermaid.ts
 * -------------
 * TypeScript AST 解析與 Mermaid 流程圖產生工具
 */

import * as ts from 'typescript';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * 流程圖節點
 */
class Node {
  id: string;
  label: string;
  kind: string;
  nexts: Node[] = [];
  edgeInfo: Map<string, { label: string; style: string }> = new Map();

  constructor(id: string, label: string, kind: string = 'op') {
    this.id = this.sanitizeId(id);
    this.label = label;
    this.kind = kind;
  }

  private sanitizeId(idStr: string): string {
    // Mermaid node IDs 應該是字母數字加底線/破折號
    let cleaned = idStr.replace(/[^a-zA-Z0-9_-]/g, '_');
    // 確保以字母開頭
    if (cleaned && /^[0-9_-]/.test(cleaned[0])) {
      cleaned = `n_${cleaned}`;
    }
    // 移除連續底線
    cleaned = cleaned.replace(/_+/g, '_');
    return cleaned || 'node';
  }
}

/**
 * 流程圖
 */
class Graph {
  title: string;
  direction: string;
  nodes: Node[] = [];
  counter: number = 0;
  start: Node;
  end: Node;

  constructor(title: string, direction: string = 'TD') {
    this.title = title;
    this.direction = this.validateDirection(direction);
    this.start = this.add('start', '開始');
    this.end = this.add('end', '結束');
  }

  private validateDirection(dir: string): string {
    const valid = ['TD', 'TB', 'BT', 'RL', 'LR'];
    dir = dir.toUpperCase();
    if (dir === 'TD') dir = 'TB'; // TD deprecated
    return valid.includes(dir) ? dir : 'TB';
  }

  add(kind: string, label: string): Node {
    this.counter++;
    const node = new Node(`n${this.counter}`, label, kind);
    this.nodes.push(node);
    return node;
  }

  link(from: Node, to: Node, label: string = '', style: string = '-->'): void {
    if (!from.nexts.includes(to)) {
      from.nexts.push(to);
    }
    if (label || style) {
      from.edgeInfo.set(to.id, { label, style });
    }
  }

  toMermaid(): string {
    const lines: string[] = [`flowchart ${this.direction}`];

    // 節點定義
    for (const node of this.nodes) {
      lines.push(`    ${this.formatNode(node)}`);
    }

    lines.push('');

    // 連接
    for (const node of this.nodes) {
      for (let i = 0; i < node.nexts.length; i++) {
        const next = node.nexts[i];
        if (node.kind === 'cond') {
          if (i === 0) {
            lines.push(`    ${node.id} -->|Yes| ${next.id}`);
          } else if (i === 1) {
            lines.push(`    ${node.id} -->|No| ${next.id}`);
          } else {
            lines.push(`    ${node.id} --> ${next.id}`);
          }
        } else {
          const edge = node.edgeInfo.get(next.id);
          if (edge?.label) {
            lines.push(`    ${node.id} -->|${edge.label}| ${next.id}`);
          } else {
            lines.push(`    ${node.id} --> ${next.id}`);
          }
        }
      }
    }

    return lines.join('\n');
  }

  private formatNode(node: Node): string {
    const text = this.sanitizeText(node.label);

    switch (node.kind) {
      case 'cond':
        return `${node.id}{"${text}"}`;
      case 'start':
      case 'end':
        return `${node.id}(["${text}"])`;
      case 'subgraph':
        return `${node.id}[/"${text}"/]`;
      default:
        return `${node.id}["${text}"]`;
    }
  }

  private sanitizeText(text: string): string {
    // 處理空字串或空白字元
    if (!text || text.trim() === '') {
      return '(empty)';  // Mermaid 不允許完全空的節點標籤
    }

    // Mermaid 最小化轉義對照表 - 只轉義會破壞 Mermaid 語法的字元
    // 根據 Mermaid 官方文件，使用雙引號包裹文字即可安全顯示大部分特殊字元
    // 只需轉義：雙引號本身、反斜線
    const escapeMap: { [key: string]: string } = {
      '"': '#quot;',   // 雙引號必須轉義（因為用雙引號包裹文字）
      '\\': '#92;',    // 反斜線轉義
      '\n': '<br/>',   // 換行轉為 HTML 換行
      '\t': '    '     // Tab 轉為空格
    };

    // 執行轉義
    text = text.replace(/["\\\n\t]/g, (match) => escapeMap[match]);

    // 限制長度
    if (text.length > 60) {
      text = text.slice(0, 57) + '...';
    }

    return text;
  }
}

/**
 * AST 構建器
 */
class Builder {
  private graph: Graph;
  private sourceFile: ts.SourceFile;

  constructor(title: string, sourceFile: ts.SourceFile, direction: string = 'TD') {
    this.graph = new Graph(title, direction);
    this.sourceFile = sourceFile;
  }

  buildFunction(func: ts.FunctionDeclaration | ts.MethodDeclaration | ts.ArrowFunction): Graph {
    let last = this.graph.start;

    if (func.body) {
      if (ts.isBlock(func.body)) {
        last = this.buildBlock(func.body.statements, last);
      } else {
        // Arrow function with expression body
        const label = this.getExpressionLabel(func.body);
        const node = this.graph.add('op', `return ${label}`);
        this.graph.link(last, node);
        last = node;
      }
    }

    this.graph.link(last, this.graph.end);
    return this.graph;
  }

  private buildBlock(statements: ts.NodeArray<ts.Statement>, entry: Node): Node {
    let last = entry;
    for (const stmt of statements) {
      last = this.buildStatement(stmt, last);
    }
    return last;
  }

  private buildStatement(stmt: ts.Statement, entry: Node): Node {
    if (ts.isIfStatement(stmt)) {
      return this.buildIfStatement(stmt, entry);
    } else if (ts.isWhileStatement(stmt)) {
      return this.buildWhileStatement(stmt, entry);
    } else if (ts.isForStatement(stmt) || ts.isForOfStatement(stmt) || ts.isForInStatement(stmt)) {
      return this.buildForStatement(stmt, entry);
    } else if (ts.isSwitchStatement(stmt)) {
      return this.buildSwitchStatement(stmt, entry);
    } else if (ts.isTryStatement(stmt)) {
      return this.buildTryStatement(stmt, entry);
    } else if (ts.isBlock(stmt)) {
      return this.buildBlock(stmt.statements, entry);
    } else {
      const label = this.getStatementLabel(stmt);
      const node = this.graph.add('op', label);
      this.graph.link(entry, node);
      return node;
    }
  }

  private buildIfStatement(stmt: ts.IfStatement, entry: Node): Node {
    const condText = this.getExpressionLabel(stmt.expression);
    const condNode = this.graph.add('cond', `if ${condText}`);
    this.graph.link(entry, condNode);

    // Then 分支
    const thenLast = this.buildStatement(stmt.thenStatement, condNode);

    if (stmt.elseStatement) {
      // Else 分支
      const elseLast = this.buildStatement(stmt.elseStatement, condNode);
      const merge = this.graph.add('op', '');
      this.graph.link(thenLast, merge);
      this.graph.link(elseLast, merge);
      return merge;
    } else {
      const merge = this.graph.add('op', '');
      this.graph.link(thenLast, merge);
      this.graph.link(condNode, merge, 'No');
      return merge;
    }
  }

  private buildWhileStatement(stmt: ts.WhileStatement, entry: Node): Node {
    const condText = this.getExpressionLabel(stmt.expression);
    const whileNode = this.graph.add('cond', `while ${condText}`);
    this.graph.link(entry, whileNode);

    const bodyLast = this.buildStatement(stmt.statement, whileNode);
    this.graph.link(bodyLast, whileNode, 'loop');

    const exit = this.graph.add('op', '');
    this.graph.link(whileNode, exit, 'exit');
    return exit;
  }

  private buildForStatement(
    stmt: ts.ForStatement | ts.ForOfStatement | ts.ForInStatement,
    entry: Node
  ): Node {
    let label = 'for';

    if (ts.isForStatement(stmt)) {
      const cond = stmt.condition ? this.getExpressionLabel(stmt.condition) : 'true';
      label = `for (${cond})`;
    } else if (ts.isForOfStatement(stmt)) {
      const varName = stmt.initializer.getText(this.sourceFile);
      const iter = this.getExpressionLabel(stmt.expression);
      label = `for (${varName} of ${iter})`;
    } else if (ts.isForInStatement(stmt)) {
      const varName = stmt.initializer.getText(this.sourceFile);
      const iter = this.getExpressionLabel(stmt.expression);
      label = `for (${varName} in ${iter})`;
    }

    const forNode = this.graph.add('cond', label);
    this.graph.link(entry, forNode);

    const bodyLast = this.buildStatement(stmt.statement, forNode);
    this.graph.link(bodyLast, forNode, 'loop');

    const exit = this.graph.add('op', '');
    this.graph.link(forNode, exit, 'exit');
    return exit;
  }

  private buildSwitchStatement(stmt: ts.SwitchStatement, entry: Node): Node {
    const expr = this.getExpressionLabel(stmt.expression);
    const switchNode = this.graph.add('cond', `switch ${expr}`);
    this.graph.link(entry, switchNode);

    const merge = this.graph.add('op', '');

    for (const clause of stmt.caseBlock.clauses) {
      let caseLabel = 'default';
      if (ts.isCaseClause(clause) && clause.expression) {
        caseLabel = `case ${this.getExpressionLabel(clause.expression)}`;
      }

      const caseNode = this.graph.add('op', caseLabel);
      this.graph.link(switchNode, caseNode);

      let caseLast = caseNode;
      for (const caseStmt of clause.statements) {
        caseLast = this.buildStatement(caseStmt, caseLast);
      }

      this.graph.link(caseLast, merge);
    }

    return merge;
  }

  private buildTryStatement(stmt: ts.TryStatement, entry: Node): Node {
    const tryNode = this.graph.add('op', 'try');
    this.graph.link(entry, tryNode);

    const tryLast = this.buildBlock(stmt.tryBlock.statements, tryNode);
    const merge = this.graph.add('op', '');
    this.graph.link(tryLast, merge);

    if (stmt.catchClause) {
      const catchParam = stmt.catchClause.variableDeclaration
        ? stmt.catchClause.variableDeclaration.name.getText(this.sourceFile)
        : 'error';
      const catchNode = this.graph.add('op', `catch (${catchParam})`);
      this.graph.link(tryNode, catchNode);

      const catchLast = this.buildBlock(stmt.catchClause.block.statements, catchNode);
      this.graph.link(catchLast, merge);
    }

    if (stmt.finallyBlock) {
      const finallyNode = this.graph.add('op', 'finally');
      this.graph.link(merge, finallyNode);
      return this.buildBlock(stmt.finallyBlock.statements, finallyNode);
    }

    return merge;
  }

  private getStatementLabel(stmt: ts.Statement): string {
    if (ts.isReturnStatement(stmt)) {
      if (stmt.expression) {
        const value = this.getExpressionLabel(stmt.expression);
        return value.length > 30 ? `return ${value.slice(0, 27)}...` : `return ${value}`;
      }
      return 'return';
    } else if (ts.isVariableStatement(stmt)) {
      const decl = stmt.declarationList.declarations[0];
      const name = decl.name.getText(this.sourceFile);
      const init = decl.initializer ? this.getExpressionLabel(decl.initializer) : '';
      const keyword = stmt.declarationList.flags & ts.NodeFlags.Const ? 'const' : 
                      stmt.declarationList.flags & ts.NodeFlags.Let ? 'let' : 'var';
      return init ? `${keyword} ${name} = ${init}` : `${keyword} ${name}`;
    } else if (ts.isExpressionStatement(stmt)) {
      return this.getExpressionLabel(stmt.expression);
    } else {
      const text = stmt.getText(this.sourceFile);
      return text.length > 50 ? text.slice(0, 47) + '...' : text;
    }
  }

  private getExpressionLabel(expr: ts.Expression): string {
    const text = expr.getText(this.sourceFile);
    return text.length > 30 ? text.slice(0, 27) + '...' : text;
  }
}

/**
 * 掃描 TypeScript 文件
 */
function scanTsFiles(
  rootPath: string,
  ignorePatterns: string[],
  maxFiles: number
): string[] {
  const files: string[] = [];

  function walk(dir: string) {
    if (files.length >= maxFiles) return;

    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      // 檢查忽略模式
      if (ignorePatterns.some(pattern => fullPath.includes(pattern))) {
        continue;
      }

      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && /\.(ts|tsx)$/.test(entry.name)) {
        files.push(fullPath);
        if (files.length >= maxFiles) break;
      }
    }
  }

  walk(rootPath);
  return files;
}

/**
 * 為單個文件構建流程圖
 */
function buildForFile(
  filePath: string,
  direction: string = 'TD'
): Array<{ name: string; mermaid: string }> {
  const charts: Array<{ name: string; mermaid: string }> = [];

  try {
    const source = fs.readFileSync(filePath, 'utf-8');
    const sourceFile = ts.createSourceFile(
      filePath,
      source,
      ts.ScriptTarget.Latest,
      true
    );

    // 訪問所有函數
    function visit(node: ts.Node) {
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

        // 獲取參數
        const params = node.parameters.map(p => p.name.getText(sourceFile)).join(', ');
        const title = `Function: ${funcName}(${params})`;

        const builder = new Builder(title, sourceFile, direction);
        const graph = builder.buildFunction(node);

        charts.push({
          name: funcName,
          mermaid: graph.toMermaid(),
        });
      }

      ts.forEachChild(node, visit);
    }

    visit(sourceFile);
  } catch (error) {
    console.error(`❌ 解析失敗: ${filePath}`, error);
  }

  return charts;
}

/**
 * 分析並生成 Mermaid 圖表
 */
function analyzeAndGenerate(
  rootPath: string,
  outputDir: string,
  ignorePatterns: string[] = [
    'node_modules',
    '.git',
    'dist',
    'build',
    '__pycache__',
    'target',
  ],
  maxFiles: number = 10000,
  direction: string = 'TD'
): void {
  console.log('='.repeat(80));
  console.log('TypeScript 程式碼流程圖生成工具');
  console.log('='.repeat(80));
  console.log(`輸入路徑: ${rootPath}`);
  console.log(`輸出路徑: ${outputDir}`);
  console.log(`最大檔案數: ${maxFiles}`);
  console.log(`流程圖方向: ${direction}`);
  console.log('='.repeat(80));
  console.log();

  const files = scanTsFiles(rootPath, ignorePatterns, maxFiles);
  console.log(`找到 ${files.length} 個 TypeScript 檔案`);
  console.log('開始生成 Mermaid 流程圖...');

  // 創建輸出目錄
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  let totalCharts = 0;

  for (let i = 0; i < files.length; i++) {
    if ((i + 1) % 10 === 0) {
      console.log(`進度: ${i + 1}/${files.length}`);
    }

    const filePath = files[i];
    const charts = buildForFile(filePath, direction);

    if (charts.length === 0) continue;

    // 生成輸出文件名
    const relPath = path.relative(rootPath, filePath);
    const filePrefix = relPath.replace(/[/\\]/g, '_').replace(/\.(ts|tsx)$/, '');

    for (const chart of charts) {
      const safeName = chart.name.replace(/[^a-zA-Z0-9_-]/g, '_');
      const outputFile = path.join(outputDir, `${filePrefix}_${safeName}.mmd`);

      fs.writeFileSync(outputFile, chart.mermaid, 'utf-8');
      totalCharts++;
    }
  }

  console.log();
  console.log(`完成！共生成 ${totalCharts} 個流程圖檔案`);
  console.log(`輸出目錄: ${outputDir}`);
}

/**
 * 主函數
 */
function main() {
  const args = process.argv.slice(2);

  // 簡單參數解析
  let inputPath = './services/scan/engines/typescript_engine';
  let outputDir = './docs/diagrams/typescript';
  let direction = 'TB';
  let maxFiles = 10000;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' || args[i] === '-i') {
      inputPath = args[++i];
    } else if (args[i] === '--output' || args[i] === '-o') {
      outputDir = args[++i];
    } else if (args[i] === '--direction' || args[i] === '-d') {
      direction = args[++i];
    } else if (args[i] === '--max-files' || args[i] === '-m') {
      maxFiles = parseInt(args[++i], 10);
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log(`
使用方法:
  ts-node ts2mermaid.ts [選項]

選項:
  -i, --input <path>      輸入目錄或檔案路徑 (預設: ./services/scan/engines/typescript_engine)
  -o, --output <path>     輸出目錄 (預設: ./docs/diagrams/typescript)
  -d, --direction <dir>   流程圖方向 TB/LR/RL/BT (預設: TB)
  -m, --max-files <num>   最大處理檔案數 (預設: 10000)
  -h, --help              顯示幫助訊息

範例:
  ts-node ts2mermaid.ts -i ./src -o ./output -d LR
      `);
      process.exit(0);
    }
  }

  analyzeAndGenerate(inputPath, outputDir, undefined, maxFiles, direction);
}

// 執行（ES modules 方式 - 使用 import.meta.main 檢測主模組）
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

if (import.meta.main) {
  main();
}
