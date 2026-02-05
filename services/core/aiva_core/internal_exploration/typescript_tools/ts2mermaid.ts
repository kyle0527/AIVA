/**
 * ts2mermaid.ts - All-in-One TypeScript AST Analysis & Stitching Tool
 * Version: 2.0 (Feature Parity with Python/Go/Rust Suite)
 *
 * 功能整合：
 * 1. [Analyzer] AST 解析與單檔 Mermaid 生成
 * 2. [Stitcher] 跨檔案數據流串接 (Cross-file Data Flow)
 * 3. [Analyzer] 系統架構圖生成 (System Flow)
 * 4. [Classifier] 功能自動分類
 * 5. [CLI Impl] 自動生成 CLI 執行手冊
 * 6. [Analysis] 分支熱點與瓶頸分析
 */

import * as ts from 'typescript';
import * as fs from 'node:fs';
import * as path from 'node:path';

// ==========================================
// Part 1: 基礎圖形結構 (Mermaid Graph)
// ==========================================

class GraphNode {
  id: string;
  label: string;
  kind: string;
  nexts: string[] = [];
  edgeInfo: Map<string, { label: string; style: string }> = new Map();

  constructor(id: string, label: string, kind: string = 'op') {
    this.id = this.sanitizeId(id);
    this.label = label;
    this.kind = kind;
  }

  private sanitizeId(idStr: string): string {
    let cleaned = idStr.replaceAll(/[^a-zA-Z0-9_-]/g, '_');
    if (cleaned && /^\d/.test(cleaned[0])) {
      cleaned = `n_${cleaned}`;
    }
    // Remove consecutive underscores
    cleaned = cleaned.replaceAll(/_+/g, '_');
    return cleaned || 'node';
  }
}

class Graph {
  title: string;
  direction: string;
  nodes: GraphNode[] = [];
  counter: number = 0;
  startId: string;
  endId: string;

  constructor(title: string, direction: string = 'TD') {
    this.title = title;
    this.direction = direction;
    this.startId = this.add('start', 'Start');
    this.endId = this.add('end', 'End');
  }

  add(kind: string, label: string): string {
    this.counter++;
    const id = `n${this.counter}`;
    const node = new GraphNode(id, label, kind);
    this.nodes.push(node);
    return id;
  }

  link(fromId: string, toId: string, label: string = '', style: string = '-->'): void {
    const node = this.nodes.find(n => n.id === fromId);
    if (node) {
      if (!node.nexts.includes(toId)) {
        node.nexts.push(toId);
      }
      if (label || style) {
        node.edgeInfo.set(toId, { label, style });
      }
    }
  }

  toMermaid(): string {
    const lines = [`flowchart ${this.direction}`];
    
    for (const node of this.nodes) {
      lines.push(`    ${this.formatNode(node)}`);
    }
    lines.push('');
    
    for (const node of this.nodes) {
      for (let i = 0; i < node.nexts.length; i++) {
        const nextId = node.nexts[i];
        lines.push(`    ${this.formatEdge(node, nextId, i)}`);
      }
    }
    return lines.join('\n');
  }

  private formatNode(node: GraphNode): string {
    let text = node.label.replaceAll(/"/g, "'").replaceAll(/\n/g, ' ');
    if (text.length > 50) text = text.substring(0, 47) + '...';
    
    switch (node.kind) {
      case 'cond': return `${node.id}{"${text}"}`;
      case 'start':
      case 'end': return `${node.id}(["${text}"])`;
      default: return `${node.id}["${text}"]`;
    }
  }

  private formatEdge(from: GraphNode, toId: string, index: number): string {
    if (from.kind === 'cond') {
      if (index === 0) return `${from.id} -->|Yes| ${toId}`;
      return `${from.id} -->|No| ${toId}`;
    }
    const edge = from.edgeInfo.get(toId);
    if (edge?.label) {
      return `${from.id} -->|${edge.label}| ${toId}`;
    }
    return `${from.id} --> ${toId}`;
  }
}

// ==========================================
// Part 2: 數據流串接結構 (Stitcher)
// ==========================================

interface ExternalCall {
  functionName: string;
  moduleAlias: string; // e.g., 'fs' in fs.readFile
  callerFunc: string;
}

interface ScriptNode {
  filePath: string;
  moduleName: string;
  imports: Map<string, string>; // alias -> library/path
  definedFuncs: string[];
  externalCalls: ExternalCall[];
}

interface Connection {
  fromScript: string;
  fromFunc: string;
  toScript: string;
  toFunc: string;
  callExpr: string;
}

class Stitcher {
  scriptNodes: Map<string, ScriptNode> = new Map();
  // Map module name (e.g. 'UserService') to list of file paths
  moduleMap: Map<string, string[]> = new Map();
  realConnections: Connection[] = [];

  addScript(filePath: string, moduleName: string, imports: Map<string, string>, funcs: string[], calls: ExternalCall[]) {
    this.scriptNodes.set(filePath, {
      filePath,
      moduleName,
      imports,
      definedFuncs: funcs,
      externalCalls: calls
    });

    if (!this.moduleMap.has(moduleName)) {
      this.moduleMap.set(moduleName, []);
    }
    this.moduleMap.get(moduleName)!.push(filePath);
  }

  resolveConnections() {
    console.log("🔗 正在執行跨檔案數據流串接 (Data Flow Stitching)...");
    
    for (const [sourcePath, node] of this.scriptNodes) {
      for (const call of node.externalCalls) {
        // 1. 嘗試解析 Import Alias
        let targetFileHint = "";
        if (call.moduleAlias && node.imports.has(call.moduleAlias)) {
           // import { foo } from './bar' -> moduleAlias='foo', resolves to './bar'
           // import * as bar from './bar' -> moduleAlias='bar', resolves to './bar'
           targetFileHint = node.imports.get(call.moduleAlias) || "";
        }

        this.findProvider(sourcePath, call, targetFileHint);
      }
    }
    console.log(`✅ 串接完成：發現 ${this.realConnections.length} 條跨檔案連接`);
  }

  private findProvider(sourcePath: string, call: ExternalCall, fileHint: string) {
    // 策略 1: 如果有明確的 import 路徑
    if (fileHint) {
        // 嘗試解析相對路徑
        let targetAbsPath = "";
        if (fileHint.startsWith('.')) {
            targetAbsPath = path.resolve(path.dirname(sourcePath), fileHint);
            // 嘗試補全副檔名
            const extensions = ['.ts', '.tsx', '.js', '.jsx', '/index.ts', ''];
            for (const ext of extensions) {
                const tryPath = targetAbsPath + ext;
                if (this.scriptNodes.has(tryPath)) {
                    this.checkCandidates(sourcePath, call, [tryPath]);
                    return;
                }
            }
        }
    }

    // 策略 2: 根據 moduleAlias 模糊匹配 (例如 user.save() -> 找 user 相關檔案)
    if (call.moduleAlias) {
        // 在所有檔案中搜尋包含 alias 的檔案
        const candidates: string[] = [];
        for (const [p, node] of this.scriptNodes) {
            if (node.moduleName.toLowerCase().includes(call.moduleAlias.toLowerCase())) {
                candidates.push(p);
            }
        }
        if (candidates.length > 0) {
            this.checkCandidates(sourcePath, call, candidates);
            return;
        }
    }

    // 策略 3: 全局搜尋函數定義 (最寬鬆)
    const allFiles = Array.from(this.scriptNodes.keys());
    this.checkCandidates(sourcePath, call, allFiles);
  }

  private checkCandidates(sourcePath: string, call: ExternalCall, candidates: string[]) {
    for (const targetFile of candidates) {
        if (targetFile === sourcePath) continue;

        const targetNode = this.scriptNodes.get(targetFile);
        if (!targetNode) continue;

        for (const def of targetNode.definedFuncs) {
            // 匹配邏輯:
            // 1. 完全匹配: myFunc == myFunc
            // 2. Class Method: UserService.login vs login (call.functionName)
            const isMatch = def === call.functionName || 
                            def.endsWith(`.${call.functionName}`) ||
                            def === call.moduleAlias; // 有時 alias 本身就是導入的函數

            if (isMatch) {
                const conn: Connection = {
                    fromScript: sourcePath,
                    fromFunc: call.callerFunc,
                    toScript: targetFile,
                    toFunc: def,
                    callExpr: call.moduleAlias ? `${call.moduleAlias}.${call.functionName}` : call.functionName
                };

                // 簡單去重
                const exists = this.realConnections.some(c => 
                    c.fromScript === conn.fromScript && 
                    c.toScript === conn.toScript && 
                    c.callExpr === conn.callExpr
                );

                if (!exists) {
                    this.realConnections.push(conn);
                }
                return; // 找到即止
            }
        }
    }
  }

  generateSystemFlow(): string {
    const lines = ["flowchart TB", "    %% 系統級數據流圖 (自動生成)"];
    const nodes = new Set<string>();

    for (const conn of this.realConnections) {
        const fromName = path.basename(conn.fromScript, path.extname(conn.fromScript));
        const toName = path.basename(conn.toScript, path.extname(conn.toScript));
        
        const fromId = this.sanitizeId(fromName);
        const toId = this.sanitizeId(toName);

        if (!nodes.has(fromId)) {
            lines.push(`    ${fromId}[["${fromName}"]]`);
            nodes.add(fromId);
        }
        if (!nodes.has(toId)) {
            lines.push(`    ${toId}[["${toName}"]]`);
            nodes.add(toId);
        }

        lines.push(`    ${fromId} -->|${conn.callExpr}| ${toId}`);
    }

    return lines.join('\n');
  }

  analyzeBranches(): any {
    const fanIn: Record<string, number> = {};
    const fanOut: Record<string, number> = {};

    for (const conn of this.realConnections) {
        const src = path.basename(conn.fromScript);
        const dst = path.basename(conn.toScript);

        fanOut[src] = (fanOut[src] || 0) + 1;
        fanIn[dst] = (fanIn[dst] || 0) + 1;
    }

    const highFanOut: Record<string, number> = {};
    const highFanIn: Record<string, number> = {};

    Object.entries(fanOut).forEach(([k, v]) => { if (v > 2) highFanOut[k] = v; });
    Object.entries(fanIn).forEach(([k, v]) => { if (v > 2) highFanIn[k] = v; });

    return {
        fan_out_nodes: highFanOut,
        fan_in_nodes: highFanIn
    };
  }

  private sanitizeId(id: string): string {
    return id.replaceAll(/[^a-zA-Z0-9]/g, '_');
  }
}

// ==========================================
// Part 3: AST 解析器 (Builder)
// ==========================================

interface FlowMetadata {
  sourceFile: string;
  functionName: string;
  module: string;
  inputs: string[];
  outputs: string[];
  callTargets: string[];
  category: string;
  description: string;
}

class Builder {
  graph: Graph;
  currentId: string;
  metadata: FlowMetadata;
  externalCalls: ExternalCall[] = [];
  sourceFile: ts.SourceFile;
  imports: Map<string, string>;

  constructor(title: string, sourceFile: ts.SourceFile, imports: Map<string, string>) {
    this.graph = new Graph(title, 'TD');
    this.currentId = this.graph.startId;
    this.sourceFile = sourceFile;
    this.imports = imports;
    
    this.metadata = {
      sourceFile: sourceFile.fileName,
      functionName: title,
      module: path.basename(sourceFile.fileName),
      inputs: [],
      outputs: [],
      callTargets: [],
      category: '',
      description: ''
    };
  }

  build(node: ts.Node) {
    if (ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node) || ts.isArrowFunction(node) || ts.isFunctionExpression(node)) {
        this.extractParams(node);
        if (node.body && ts.isBlock(node.body)) {
            this.visitBlock(node.body);
        }
        this.graph.link(this.currentId, this.graph.endId);
    }
  }

  private extractParams(node: ts.FunctionLikeDeclaration) {
      for (const param of node.parameters) {
          if (ts.isIdentifier(param.name)) {
              this.metadata.inputs.push(param.name.text);
          }
      }
  }

  private visitBlock(block: ts.Block) {
      for (const stmt of block.statements) {
          this.visitNode(stmt);
      }
  }

  private visitNode(node: ts.Node) {
      if (ts.isIfStatement(node)) {
          this.handleIf(node);
      } else if (ts.isReturnStatement(node)) {
          this.handleReturn(node);
      } else if (ts.isForStatement(node) || ts.isForOfStatement(node) || ts.isForInStatement(node) || ts.isWhileStatement(node)) {
          this.handleLoop(node);
      } else if (ts.isVariableStatement(node)) {
          node.declarationList.declarations.forEach(d => {
              if (d.initializer) this.visitNode(d.initializer);
          });
      } else if (ts.isExpressionStatement(node)) {
          this.visitNode(node.expression);
      } else if (ts.isCallExpression(node)) {
          this.handleCall(node);
      } else if (ts.isAwaitExpression(node)) {
          this.visitNode(node.expression);
      } else if (ts.isBinaryExpression(node)) {
           this.visitNode(node.left);
           this.visitNode(node.right);
      }
  }

  private handleIf(node: ts.IfStatement) {
      const condId = this.graph.add('cond', 'If Condition');
      this.graph.link(this.currentId, condId);

      // Then
      this.currentId = condId;
      const thenNode = this.graph.add('op', 'Then');
      this.graph.link(condId, thenNode, 'Yes');
      this.currentId = thenNode;
      this.visitNode(node.thenStatement);
      const thenEnd = this.currentId;

      // Else
      let elseEnd = null;
      if (node.elseStatement) {
          this.currentId = condId;
          const elseNode = this.graph.add('op', 'Else');
          this.graph.link(condId, elseNode, 'No');
          this.currentId = elseNode;
          this.visitNode(node.elseStatement);
          elseEnd = this.currentId;
      }

      // Merge
      const merge = this.graph.add('op', '');
      this.graph.link(thenEnd, merge);
      if (elseEnd) {
          this.graph.link(elseEnd, merge);
      } else {
          this.graph.link(condId, merge, 'No');
      }
      this.currentId = merge;
  }

  private handleLoop(node: ts.Statement) {
      const loopId = this.graph.add('cond', 'Loop');
      this.graph.link(this.currentId, loopId);
      
      const bodyId = this.graph.add('op', 'Body');
      this.graph.link(loopId, bodyId, 'Loop');
      this.currentId = bodyId;

      // TS AST specific handling for body
      if ('statement' in node) {
          this.visitNode((node as any).statement);
      }

      this.graph.link(this.currentId, loopId, 'Next');
      
      const exitId = this.graph.add('op', '');
      this.graph.link(loopId, exitId, 'Done');
      this.currentId = exitId;
  }

  private handleReturn(node: ts.ReturnStatement) {
      const retId = this.graph.add('op', 'Return');
      this.graph.link(this.currentId, retId);
      this.graph.link(retId, this.graph.endId);
      this.currentId = retId;
      
      if (node.expression) {
          // Record output types if possible, here just placeholders
          this.metadata.outputs.push("result"); 
      }
  }

  private handleCall(node: ts.CallExpression) {
      let funcName = "";
      let moduleAlias = "";

      if (ts.isIdentifier(node.expression)) {
          // func()
          funcName = node.expression.text;
      } else if (ts.isPropertyAccessExpression(node.expression)) {
          // obj.method()
          funcName = node.expression.name.text;
          if (ts.isIdentifier(node.expression.expression)) {
              moduleAlias = node.expression.expression.text;
          }
      }

      if (funcName) {
          const label = moduleAlias ? `${moduleAlias}.${funcName}` : funcName;
          const nodeId = this.graph.add('op', `Call: ${label}`);
          this.graph.link(this.currentId, nodeId);
          this.currentId = nodeId;

          this.metadata.callTargets.push(label);
          this.externalCalls.push({
              functionName: funcName,
              moduleAlias: moduleAlias,
              callerFunc: this.metadata.functionName
          });
      }

      // Visit args
      node.arguments.forEach(arg => this.visitNode(arg));
  }
}

// ==========================================
// Part 4: 分類器 (Classifier)
// ==========================================

class Classifier {
  categories: Map<string, FlowMetadata[]> = new Map();

  classify(metadata: FlowMetadata) {
      const name = metadata.functionName.toLowerCase();
      let cat = "other";

      if (name.includes("scan") || name.includes("detect")) cat = "reconnaissance";
      else if (name.includes("exploit") || name.includes("attack")) cat = "exploitation";
      else if (name.includes("analyze") || name.includes("parse")) cat = "analysis";
      else if (name.includes("report") || name.includes("generate")) cat = "reporting";
      else if (name.includes("store") || name.includes("save") || name.includes("db")) cat = "persistence";

      metadata.category = cat;
      // Description placeholder - to be filled by LLM analysis
      metadata.description = `[PLACEHOLDER] ${metadata.functionName} in ${path.basename(metadata.sourceFile)}`;

      if (!this.categories.has(cat)) this.categories.set(cat, []);
      this.categories.get(cat)!.push(metadata);
  }

  getResult() {
      const summary: Record<string, number> = {};
      let total = 0;
      const catsObj: Record<string, FlowMetadata[]> = {};

      for (const [k, v] of this.categories) {
          summary[k] = v.length;
          catsObj[k] = v;
          total += v.length;
      }

      return {
          total_flows: total,
          categories: catsObj,
          summary: summary
      };
  }
}

function generateCLI(result: any): string {
  let sb = "# AIVA TypeScript Analysis CLI Commands\n# Auto-generated by ts2mermaid\n\n";
  
  const categories = result.categories;
  for (const cat of Object.keys(categories).sort()) {
      sb += `## Category: ${cat.toUpperCase()}\n`;
      for (const item of categories[cat]) {
          sb += `# ${item.description}\n`;
          sb += `npx ts-node ts2mermaid.ts --file "${item.sourceFile}" --func "${item.functionName}"\n\n`;
      }
  }
  return sb;
}

// ==========================================
// Part 5: 主程序 (Orchestrator)
// ==========================================

function getAllFiles(dir: string, files: string[] = []) {
  if (fs.existsSync(dir)) {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
              if (entry.name !== 'node_modules' && entry.name !== '.git') {
                  getAllFiles(fullPath, files);
              }
          } else if (fullPath.endsWith('.ts') || fullPath.endsWith('.tsx')) {
              files.push(fullPath);
          }
      }
  }
  return files;
}

function parseImports(sourceFile: ts.SourceFile): Map<string, string> {
    const imports = new Map<string, string>();
    
    // Visit AST to find imports
    function visit(node: ts.Node) {
        if (ts.isImportDeclaration(node)) {
            const moduleSpecifier = (node.moduleSpecifier as ts.StringLiteral).text;
            
            if (node.importClause) {
                // import x from 'y'
                if (node.importClause.name) {
                    imports.set(node.importClause.name.text, moduleSpecifier);
                }
                // import { x, y } from 'z'
                if (node.importClause.namedBindings && ts.isNamedImports(node.importClause.namedBindings)) {
                    node.importClause.namedBindings.elements.forEach(el => {
                        imports.set(el.name.text, moduleSpecifier);
                    });
                }
                // import * as ns from 'z'
                if (node.importClause.namedBindings && ts.isNamespaceImport(node.importClause.namedBindings)) {
                    imports.set(node.importClause.namedBindings.name.text, moduleSpecifier);
                }
            }
        }
        ts.forEachChild(node, visit);
    }
    
    visit(sourceFile);
    return imports;
}

function processFile(filePath: string, outputDir: string, stitcher: Stitcher): FlowMetadata[] {
    const code = fs.readFileSync(filePath, 'utf-8');
    const sourceFile = ts.createSourceFile(filePath, code, ts.ScriptTarget.Latest, true);
    const imports = parseImports(sourceFile);
    const metaList: FlowMetadata[] = [];
    const definedFuncs: string[] = [];
    const allCalls: ExternalCall[] = [];

    function visit(node: ts.Node) {
        let funcName = "";
        let isFunc = false;

        // Function Declaration
        if (ts.isFunctionDeclaration(node) && node.name) {
            funcName = node.name.text;
            isFunc = true;
        } 
        // Class Method
        else if (ts.isMethodDeclaration(node) && ts.isIdentifier(node.name)) {
            // Try to find class name parent
            let parent = node.parent;
            let className = "Class";
            if (ts.isClassDeclaration(parent) && parent.name) {
                className = parent.name.text;
            }
            funcName = `${className}.${node.name.text}`;
            isFunc = true;
        }
        // Variable Arrow Function: const foo = () => {}
        else if (ts.isVariableDeclaration(node) && node.initializer && 
                (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer)) && 
                ts.isIdentifier(node.name)) {
            funcName = node.name.text;
            isFunc = true;
        }

        if (isFunc && funcName) {
            definedFuncs.push(funcName);
            const builder = new Builder(funcName, sourceFile, imports);
            
            // Build graph for the function body
            if (ts.isVariableDeclaration(node) && node.initializer) {
                 builder.build(node.initializer); // pass the arrow function
            } else {
                 builder.build(node);
            }

            // Save .mmd
            const safeName = funcName.replaceAll(/[^a-zA-Z0-9]/g, '_');
            const outPath = path.join(outputDir, `${path.basename(filePath)}_${safeName}.mmd`);
            fs.writeFileSync(outPath, builder.graph.toMermaid());

            // Collect data
            const meta = builder.metadata;
            metaList.push(meta);
            allCalls.push(...builder.externalCalls);
        }

        ts.forEachChild(node, visit);
    }

    visit(sourceFile);

    // Add to stitcher
    stitcher.addScript(filePath, path.basename(filePath), imports, definedFuncs, allCalls);

    return metaList;
}

function main() {
    const args = process.argv.slice(2);
    const getArg = (name: string) => {
        const arg = args.find(a => a.startsWith(name));
        return arg ? arg.split('=')[1] : null;
    }

    const inputDir = getArg('--input') || '.';
    
    // 使用統一路徑配置
    const defaultOutputDir = getDefaultOutputDir();
    let outputDir: string = getArg('--output') || defaultOutputDir;

    console.log("========================================");
    console.log("🚀 AIVA TypeScript 全功能分析器 (ts2mermaid v2.0)");
    console.log("========================================");
    console.log(`📂 輸入目錄: ${inputDir}`);
    console.log(`📂 輸出目錄: ${outputDir}`);

    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const files = getAllFiles(inputDir);
    console.log(`📊 發現 ${files.length} 個 TypeScript 檔案`);

    const stitcher = new Stitcher();
    const classifier = new Classifier();
    const allMeta: FlowMetadata[] = [];

    // 1. Scanning
    for (const file of files) {
        try {
            const metas = processFile(file, outputDir, stitcher);
            for (const m of metas) {
                classifier.classify(m);
                allMeta.push(m);
            }
        } catch (e) {
            console.error(`❌ Error processing ${file}:`, e);
        }
    }

    // 2. Stitching
    stitcher.resolveConnections();

    // 3. Analysis
    const branchStats = stitcher.analyzeBranches();

    /**
     * 將 realConnections 轉換成 Python FlowExecutor 需要的 flows 格式
     * 這是為了讓 Python 執行器能夠讀取並執行 TypeScript 分析結果
     */
    function convertConnectionsToFlows(connections: Connection[]): any[] {
        return connections.map((conn, idx) => ({
            id: idx + 1,
            path: [conn.fromFunc, conn.toFunc],
            full_path: [conn.fromScript, conn.toScript],
            func_names: [conn.fromFunc, conn.toFunc],
            length: 2,
            start: conn.fromFunc,
            end: conn.toFunc,
            classifications: [
                {
                    script: conn.fromFunc,
                    module: "typescript_module",
                    component_type: "程式組件",
                    description: `${conn.fromFunc} - TypeScript 功能`
                },
                {
                    script: conn.toFunc,
                    module: "typescript_module",
                    component_type: "程式組件",
                    description: `${conn.toFunc} - TypeScript 功能`
                }
            ],
            language: "typescript",
            cli_command: `ts-node ${conn.toScript}`,
            structured_tags: ["language:typescript", "type:程式"]
        }));
    }

    // 4. Output
    // System Flow
    fs.writeFileSync(path.join(outputDir, 'system_flow.mmd'), stitcher.generateSystemFlow());
    
    // CLI Docs
    const classResult = classifier.getResult();
    fs.writeFileSync(path.join(outputDir, 'cli_commands.sh'), generateCLI(classResult));

    // Full JSON Report - 統一格式，相容 Python FlowExecutor
    const flows = convertConnectionsToFlows(stitcher.realConnections);
    
    const report = {
        metadata: {
            tool: "ts2mermaid",
            version: "2.0",
            language: "typescript",
            generated_at: new Date().toISOString(),
            total_flows: flows.length,
            total_files: files.length,
            schema_version: "3.3",
            ai_compatible: true
        },
        // ✅ 新增：FlowExecutor 需要的統一格式
        flows: flows,
        
        // ✅ 保留：TypeScript 工具原有的所有字段
        summary: {
            total_files: files.length,
            total_funcs: allMeta.length,
            real_connections: stitcher.realConnections.length,
        },
        classification: classResult,
        branch_analysis: branchStats,
        flow_chains: stitcher.realConnections,
        functions: allMeta
    };

    fs.writeFileSync(path.join(outputDir, 'analysis_results.json'), JSON.stringify(report, null, 2));

    console.log("\n✅ 分析工作全部完成！");
    console.log(`   📄 系統架構圖: ${outputDir}/system_flow.mmd`);
    console.log(`   📄 完整數據報告: ${outputDir}/analysis_results.json`);
    console.log(`   📄 CLI 指令手冊: ${outputDir}/cli_commands.sh`);
}

main();
