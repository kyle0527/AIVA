// go2mermaid.go - All-in-One Go AST Analysis Tool
//
// 功能整合：
// 1. AST 解析與流程圖生成 (對標 aiva_flow_analyzer.py)
// 2. 跨檔案數據流串接 Stitching (對標 aiva_flow_analyzer.py)
// 3. 功能分類與統計 (對標 aiva_flow_classifier.py)
// 4. CLI 指令手冊生成 (對標 aiva_cli_implementation.py)
// 5. 系統瓶頸分析 (對標 aiva_flow_analyzer.py 統計報告)

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// ==========================================
// Part 1: 基礎數據結構 (Graph & Metadata)
// ==========================================

// Node 流程圖節點
type Node struct {
	ID       string
	Label    string
	Kind     string // op, cond, start, end
	Nexts    []*Node
	EdgeInfo map[string]EdgeStyle
}

// EdgeStyle 邊的樣式
type EdgeStyle struct {
	Label string
	Style string
}

// Graph 流程圖結構
type Graph struct {
	Title     string
	Direction string
	Nodes     []*Node
	Counter   int
	Start     *Node
	End       *Node
}

// FlowMetadata 單一函數的元數據 (用於分類與檢索)
type FlowMetadata struct {
	SourceFile   string   `json:"source_file"`
	FunctionName string   `json:"function_name"`
	Module       string   `json:"module"`
	Inputs       []string `json:"inputs"`
	Outputs      []string `json:"outputs"`
	CallTargets  []string `json:"call_targets"` // 記錄此函數內部調用了誰
	Category     string   `json:"category"`
	Description  string   `json:"description"` // 自動生成的描述
}

// ==========================================
// Part 2: 數據流串接系統 (Stitcher)
// ==========================================

// ExternalCall 記錄外部調用
type ExternalCall struct {
	FunctionName string `json:"function_name"` // e.g. "Println" or "MyFunc"
	PackageAlias string `json:"package_alias"` // e.g. "fmt" or "utils"
	CallerFunc   string `json:"caller_func"`   // 誰調用的
}

// ScriptNode 代表一個 Go 檔案的分析單元
type ScriptNode struct {
	FilePath      string            `json:"file_path"`
	PackageName   string            `json:"package_name"`
	Imports       map[string]string `json:"imports"`        // alias -> full_path
	DefinedFuncs  []string          `json:"defined_funcs"`  // 此檔案定義的函數
	ExternalCalls []ExternalCall    `json:"external_calls"` // 此檔案發出的調用
}

// Connection 代表真實的跨檔案連接
type Connection struct {
	FromScript   string `json:"from_script"`
	FromFunc     string `json:"from_func"`
	ToScript     string `json:"to_script"`
	ToFunc       string `json:"to_func"`
	CallExpr     string `json:"call_expr"` // e.g. "utils.Helper"
}

// Stitcher 負責將孤立的檔案串接成系統
type Stitcher struct {
	ScriptNodes     map[string]*ScriptNode // FilePath -> Node
	PackageMap      map[string][]string    // PackageName -> [FilePaths]
	RealConnections []Connection           `json:"real_connections"`
}

func NewStitcher() *Stitcher {
	return &Stitcher{
		ScriptNodes:     make(map[string]*ScriptNode),
		PackageMap:      make(map[string][]string),
		RealConnections: []Connection{},
	}
}

// AddScript 加入腳本節點
func (s *Stitcher) AddScript(filePath, pkgName string, imports map[string]string, funcs []string, calls []ExternalCall) {
	node := &ScriptNode{
		FilePath:      filePath,
		PackageName:   pkgName,
		Imports:       imports,
		DefinedFuncs:  funcs,
		ExternalCalls: calls,
	}
	s.ScriptNodes[filePath] = node
	s.PackageMap[pkgName] = append(s.PackageMap[pkgName], filePath)
}

// ResolveConnections 核心串接邏輯：解析跨檔案調用
func (s *Stitcher) ResolveConnections() {
	fmt.Println("🔗 正在執行跨檔案數據流串接 (Data Flow Stitching)...")

	for sourcePath, node := range s.ScriptNodes {
		for _, call := range node.ExternalCalls {
			// 1. 同 Package 內部調用 (無 Alias)
			if call.PackageAlias == "" {
				s.findProviderInPackage(sourcePath, call.CallerFunc, node.PackageName, call.FunctionName, "")
				continue
			}

			// 2. 跨 Package 調用
			// 在真實 Go 專案需解析 go.mod，這裡模擬 Python 版採用「模糊匹配」策略
			// 嘗試用 alias 直接匹配 package name
			targetPkg := call.PackageAlias
			s.findProviderInPackage(sourcePath, call.CallerFunc, targetPkg, call.FunctionName, fmt.Sprintf("%s.%s", targetPkg, call.FunctionName))
		}
	}
	fmt.Printf("✅ 串接完成：發現 %d 條跨檔案連接\n", len(s.RealConnections))
}

func (s *Stitcher) findProviderInPackage(sourcePath, callerFunc, targetPkg, targetFunc, callExpr string) {
	candidateFiles, exists := s.PackageMap[targetPkg]
	if !exists {
		return // 可能是標準庫或第三方庫，暫時忽略
	}

	for _, targetFile := range candidateFiles {
		// 忽略自我調用
		if targetFile == sourcePath {
			continue
		}

		targetNode := s.ScriptNodes[targetFile]
		for _, def := range targetNode.DefinedFuncs {
			// 支援結構體方法匹配: "Struct.Method" vs "Method"
			// 簡單處理：如果定義包含目標函數名
			if def == targetFunc || strings.HasSuffix(def, "."+targetFunc) {
				conn := Connection{
					FromScript: sourcePath,
					FromFunc:   callerFunc,
					ToScript:   targetFile,
					ToFunc:     def,
					CallExpr:   callExpr,
				}
				if callExpr == "" {
					conn.CallExpr = targetFunc
				}

				// 去重
				isDup := false
				for _, existing := range s.RealConnections {
					if existing == conn {
						isDup = true
						break
					}
				}
				if !isDup {
					s.RealConnections = append(s.RealConnections, conn)
				}
				return // 找到一個定義即可 (簡化處理重載)
			}
		}
	}
}

// AnalyzeBranches 分支統計分析 (對標 Python 版的 analyze_branch_patterns)
func (s *Stitcher) AnalyzeBranches() map[string]interface{} {
	fanIn := make(map[string]int)
	fanOut := make(map[string]int)

	for _, conn := range s.RealConnections {
		src := filepath.Base(conn.FromScript)
		dst := filepath.Base(conn.ToScript)
		fanOut[src]++
		fanIn[dst]++
	}

	// 識別瓶頸
	highFanOut := make(map[string]int)
	highFanIn := make(map[string]int)

	for k, v := range fanOut {
		if v > 2 {
			highFanOut[k] = v
		}
	}
	for k, v := range fanIn {
		if v > 2 {
			highFanIn[k] = v
		}
	}

	return map[string]interface{}{
		"fan_out_nodes":     highFanOut,
		"fan_in_nodes":      highFanIn,
		"total_connections": len(s.RealConnections),
	}
}

// GenerateSystemFlow 生成系統級架構圖
func (s *Stitcher) GenerateSystemFlow() string {
	var sb strings.Builder
	sb.WriteString("flowchart TB\n")
	sb.WriteString("    %% 系統級數據流圖 (自動生成)\n")

	nodes := make(map[string]string) // id -> label

	for _, conn := range s.RealConnections {
		fromLabel := filepath.Base(conn.FromScript)
		toLabel := filepath.Base(conn.ToScript)
		
		fromID := sanitizeID(fromLabel)
		toID := sanitizeID(toLabel)

		if _, ok := nodes[fromID]; !ok {
			sb.WriteString(fmt.Sprintf("    %s[\"%s\"]\n", fromID, fromLabel))
			nodes[fromID] = fromLabel
		}
		if _, ok := nodes[toID]; !ok {
			sb.WriteString(fmt.Sprintf("    %s[\"%s\"]\n", toID, toLabel))
			nodes[toID] = toLabel
		}

		sb.WriteString(fmt.Sprintf("    %s -->|%s| %s\n", fromID, conn.CallExpr, toID))
	}

	return sb.String()
}

// ==========================================
// Part 3: 分類與 CLI 生成 (Classifier & CLI)
// ==========================================

// ClassificationResult 分類結果容器
type ClassificationResult struct {
	TotalFlows int                       `json:"total_flows"`
	Categories map[string][]FlowMetadata `json:"categories"`
	Summary    map[string]int            `json:"summary"`
}

// Classifier 分類器
type Classifier struct {
	categories map[string][]FlowMetadata
}

func NewClassifier() *Classifier {
	return &Classifier{
		categories: make(map[string][]FlowMetadata),
	}
}

func (c *Classifier) Classify(metadata *FlowMetadata) {
	funcName := strings.ToLower(metadata.FunctionName)
	file := strings.ToLower(metadata.SourceFile)

	// 簡單的關鍵字分類規則 (對標 Python 版)
	if strings.Contains(funcName, "scan") || strings.Contains(funcName, "detect") {
		metadata.Category = "reconnaissance"
	} else if strings.Contains(funcName, "exploit") || strings.Contains(funcName, "attack") {
		metadata.Category = "exploitation"
	} else if strings.Contains(funcName, "analyze") || strings.Contains(funcName, "parse") {
		metadata.Category = "analysis"
	} else if strings.Contains(funcName, "report") || strings.Contains(funcName, "generate") {
		metadata.Category = "reporting"
	} else if strings.Contains(funcName, "store") || strings.Contains(funcName, "save") || strings.Contains(file, "db") {
		metadata.Category = "persistence"
	} else {
		metadata.Category = "other"
	}

	// Description placeholder - to be filled by LLM analysis
	metadata.Description = fmt.Sprintf("[PLACEHOLDER] %s 功能", metadata.FunctionName)
	c.categories[metadata.Category] = append(c.categories[metadata.Category], *metadata)
}

func (c *Classifier) GetResult() *ClassificationResult {
	summary := make(map[string]int)
	total := 0
	for k, v := range c.categories {
		summary[k] = len(v)
		total += len(v)
	}
	return &ClassificationResult{
		TotalFlows: total,
		Categories: c.categories,
		Summary:    summary,
	}
}

// GenerateCLI 生成 CLI 指令腳本
func GenerateCLI(result *ClassificationResult) string {
	var sb strings.Builder
	sb.WriteString("# AIVA Go Analysis CLI Commands\n")
	sb.WriteString("# Auto-generated by go2mermaid\n\n")
	
	// 按分類排序
	keys := make([]string, 0, len(result.Categories))
	for k := range result.Categories {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, cat := range keys {
		sb.WriteString(fmt.Sprintf("## Category: %s\n", strings.ToUpper(cat)))
		for _, flow := range result.Categories[cat] {
			// 模擬指令：在真實場景中這裡可能是調用該函數的指令
			sb.WriteString(fmt.Sprintf("# %s\n", flow.Description))
			sb.WriteString(fmt.Sprintf("go run %s --func=%s\n\n", flow.SourceFile, flow.FunctionName))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// ==========================================
// Part 4: AST Builder (單檔解析)
// ==========================================

// Builder AST 解析器
type Builder struct {
	graph         *Graph
	current       *Node
	metadata      *FlowMetadata
	externalCalls []ExternalCall
}

func NewBuilder(title, direction string) *Builder {
	return &Builder{
		graph: NewGraph(title, direction),
		metadata: &FlowMetadata{
			Inputs: []string{}, Outputs: []string{}, CallTargets: []string{},
		},
		externalCalls: []ExternalCall{},
	}
}

// BuildFunction 構建單一函數的流程圖
func (b *Builder) BuildFunction(fn *ast.FuncDecl) *Graph {
	b.current = b.graph.Start
	
	// 1. 函數名稱 (含 Receiver)
	b.metadata.FunctionName = getFunctionName(fn)

	// 2. 參數解析
	if fn.Type.Params != nil {
		for _, field := range fn.Type.Params.List {
			for _, name := range field.Names {
				b.metadata.Inputs = append(b.metadata.Inputs, name.Name)
			}
		}
	}

	// 3. 返回值解析
	if fn.Type.Results != nil {
		for i := range fn.Type.Results.List {
			b.metadata.Outputs = append(b.metadata.Outputs, fmt.Sprintf("result_%d", i))
		}
	}

	// 4. 函數體解析
	if fn.Body != nil {
		for _, stmt := range fn.Body.List {
			b.buildStmt(stmt)
		}
	}

	b.graph.Link(b.current, b.graph.End, "", "")
	return b.graph
}

func (b *Builder) buildStmt(stmt ast.Stmt) {
	switch s := stmt.(type) {
	case *ast.IfStmt:
		cond := b.graph.Add("cond", "If Condition")
		b.graph.Link(b.current, cond, "", "")
		
		// Then Block
		b.current = cond
		thenNode := b.graph.Add("op", "Then")
		b.graph.Link(cond, thenNode, "Yes", "")
		b.current = thenNode
		b.inspectBlock(s.Body.List)
		thenEnd := b.current

		// Else Block
		var elseEnd *Node
		if s.Else != nil {
			elseNode := b.graph.Add("op", "Else")
			b.graph.Link(cond, elseNode, "No", "")
			b.current = elseNode
			b.inspectStmt(s.Else)
			elseEnd = b.current
		}

		// Merge
		merge := b.graph.Add("op", "")
		b.graph.Link(thenEnd, merge, "", "")
		if elseEnd != nil {
			b.graph.Link(elseEnd, merge, "", "")
		} else {
			b.graph.Link(cond, merge, "No", "")
		}
		b.current = merge

	case *ast.ReturnStmt:
		ret := b.graph.Add("op", "Return")
		b.graph.Link(b.current, ret, "", "")
		b.graph.Link(ret, b.graph.End, "", "")
		b.current = ret

	case *ast.ForStmt, *ast.RangeStmt:
		loop := b.graph.Add("cond", "Loop")
		b.graph.Link(b.current, loop, "", "")
		
		// Body
		body := b.graph.Add("op", "Body")
		b.graph.Link(loop, body, "Loop", "")
		b.current = body
		
		// 遞歸檢查 Loop 內的調用
		if forStmt, ok := s.(*ast.ForStmt); ok {
			b.inspectBlock(forStmt.Body.List)
		}
		if rangeStmt, ok := s.(*ast.RangeStmt); ok {
			b.inspectBlock(rangeStmt.Body.List)
		}
		
		b.graph.Link(b.current, loop, "Next", "")
		
		// Exit
		exit := b.graph.Add("op", "")
		b.graph.Link(loop, exit, "Done", "")
		b.current = exit

	case *ast.ExprStmt:
		b.inspectExpr(s.X)
		// 簡化圖形：不是每個 Expr 都畫框，只畫重要的
		// node := b.graph.Add("op", fmt.Sprintf("%T", s.X))
		// b.graph.Link(b.current, node, "", "")
		// b.current = node

	case *ast.AssignStmt:
		for _, expr := range s.Rhs {
			b.inspectExpr(expr)
		}
		node := b.graph.Add("op", "Assignment")
		b.graph.Link(b.current, node, "", "")
		b.current = node

	default:
		// 預設行為：繼續遞歸尋找調用
		b.inspectStmt(s)
	}
}

// 遞歸檢查工具
func (b *Builder) inspectBlock(stmts []ast.Stmt) {
	for _, s := range stmts {
		b.buildStmt(s) // 使用 buildStmt 以保持圖形連貫
	}
}

func (b *Builder) inspectStmt(stmt ast.Stmt) {
	// 僅提取調用資訊，不畫圖
	ast.Inspect(stmt, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			b.inspectExpr(call)
		}
		return true
	})
}

func (b *Builder) inspectExpr(expr ast.Expr) {
	if call, ok := expr.(*ast.CallExpr); ok {
		currentFunc := b.metadata.FunctionName
		
		// 1. 直接調用 func()
		if ident, ok := call.Fun.(*ast.Ident); ok {
			target := ident.Name
			b.metadata.CallTargets = append(b.metadata.CallTargets, target)
			b.externalCalls = append(b.externalCalls, ExternalCall{
				FunctionName: target,
				PackageAlias: "",
				CallerFunc:   currentFunc,
			})
		}
		// 2. 套件調用 pkg.Func() 或 struct.Method()
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			if xIdent, ok := sel.X.(*ast.Ident); ok {
				pkgOrStruct := xIdent.Name
				method := sel.Sel.Name
				fullCall := fmt.Sprintf("%s.%s", pkgOrStruct, method)
				
				b.metadata.CallTargets = append(b.metadata.CallTargets, fullCall)
				b.externalCalls = append(b.externalCalls, ExternalCall{
					FunctionName: method,
					PackageAlias: pkgOrStruct, // 這裡可能是 package alias 也可能是變數名
					CallerFunc:   currentFunc,
				})
			}
		}
	}
}

// getFunctionName 處理函數名稱 (含 Receiver)
func getFunctionName(fn *ast.FuncDecl) string {
	name := fn.Name.Name
	if fn.Recv != nil {
		for _, field := range fn.Recv.List {
			typePart := ""
			if starExpr, ok := field.Type.(*ast.StarExpr); ok {
				if ident, ok := starExpr.X.(*ast.Ident); ok {
					typePart = ident.Name
				}
			} else if ident, ok := field.Type.(*ast.Ident); ok {
				typePart = ident.Name
			}
			if typePart != "" {
				return typePart + "." + name
			}
		}
	}
	return name
}

// ==========================================
// Part 5: 圖形基礎功能 (Graph Helpers)
// ==========================================

func NewGraph(title, direction string) *Graph {
	g := &Graph{
		Title:     title,
		Direction: direction,
		Nodes:     []*Node{},
		Counter:   0,
	}
	g.Start = g.Add("start", "Start")
	g.End = g.Add("end", "End")
	return g
}

func (g *Graph) Add(kind, label string) *Node {
	g.Counter++
	node := &Node{
		ID:       fmt.Sprintf("n%d", g.Counter),
		Label:    label,
		Kind:     kind,
		Nexts:    []*Node{},
		EdgeInfo: make(map[string]EdgeStyle),
	}
	g.Nodes = append(g.Nodes, node)
	return node
}

func (g *Graph) Link(from, to *Node, label, style string) {
	if from == nil || to == nil {
		return
	}
	from.Nexts = append(from.Nexts, to)
	if label != "" || style != "" {
		from.EdgeInfo[to.ID] = EdgeStyle{Label: label, Style: style}
	}
}

func (g *Graph) ToMermaid() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("flowchart %s\n", g.Direction))
	for _, n := range g.Nodes {
		sb.WriteString(fmt.Sprintf("    %s\n", formatNode(n)))
	}
	sb.WriteString("\n")
	for _, n := range g.Nodes {
		for i, next := range n.Nexts {
			sb.WriteString(fmt.Sprintf("    %s\n", formatEdge(n, next, i)))
		}
	}
	return sb.String()
}

func formatNode(n *Node) string {
	text := sanitizeText(n.Label)
	switch n.Kind {
	case "cond":
		return fmt.Sprintf("%s{\"%s\"}", n.ID, text)
	case "start", "end":
		return fmt.Sprintf("%s([\"%s\"])", n.ID, text)
	default:
		return fmt.Sprintf("%s[\"%s\"]", n.ID, text)
	}
}

func formatEdge(from, to *Node, index int) string {
	edge := from.EdgeInfo[to.ID]
	if from.Kind == "cond" {
		if index == 0 {
			return fmt.Sprintf("%s -->|Yes| %s", from.ID, to.ID)
		}
		return fmt.Sprintf("%s -->|No| %s", from.ID, to.ID)
	}
	if edge.Label != "" {
		return fmt.Sprintf("%s -->|%s| %s", from.ID, edge.Label, to.ID)
	}
	return fmt.Sprintf("%s --> %s", from.ID, to.ID)
}

func sanitizeText(text string) string {
	text = strings.ReplaceAll(text, "\"", "'")
	if len(text) > 40 {
		return text[:37] + "..."
	}
	return text
}

func sanitizeID(id string) string {
	reg := regexp.MustCompile("[^a-zA-Z0-9_]")
	return reg.ReplaceAllString(id, "_")
}

// ==========================================
// Part 6: 主程序 (Orchestrator)
// ==========================================

func processFile(filePath string, outputDir string, stitcher *Stitcher) ([]*FlowMetadata, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	pkgName := node.Name.Name
	imports := make(map[string]string)

	// 解析 Imports
	for _, imp := range node.Imports {
		path := strings.Trim(imp.Path.Value, "\"")
		var alias string
		if imp.Name != nil {
			alias = imp.Name.Name
		} else {
			parts := strings.Split(path, "/")
			alias = parts[len(parts)-1]
		}
		imports[alias] = path
	}

	var metaList []*FlowMetadata
	var definedFuncs []string
	var allCalls []ExternalCall

	for _, decl := range node.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcName := getFunctionName(fn)
			definedFuncs = append(definedFuncs, funcName)

			// 構建單檔流程圖
			builder := NewBuilder(funcName, "TB")
			graph := builder.BuildFunction(fn)

			// 收集調用信息
			allCalls = append(allCalls, builder.externalCalls...)

			// 保存 .mmd
			outFile := filepath.Join(outputDir, fmt.Sprintf("%s_%s.mmd", filepath.Base(filePath), sanitizeID(funcName)))
			os.WriteFile(outFile, []byte(graph.ToMermaid()), 0644)

			meta := builder.metadata
			meta.SourceFile = filePath
			meta.Module = pkgName
			
			metaList = append(metaList, meta)
		}
	}

	// 加入 Stitcher 進行後續串接
	stitcher.AddScript(filePath, pkgName, imports, definedFuncs, allCalls)

	return metaList, nil
}

func main() {
	inputDir := flag.String("input", ".", "輸入目錄")
	outputDir := flag.String("output", "./go_analysis", "輸出目錄")
	
	flag.Parse()

	fmt.Println("🚀 開始 AIVA Go 全功能分析...")
	fmt.Printf("📂 掃描目錄: %s\n", *inputDir)

	os.MkdirAll(*outputDir, 0755)
	
	stitcher := NewStitcher()
	classifier := NewClassifier()
	
	var allMeta []FlowMetadata

	// 1. 掃描與解析階段 (Scanning & Parsing)
	err := filepath.Walk(*inputDir, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && strings.HasSuffix(path, ".go") {
			// 忽略 test 檔案
			if strings.HasSuffix(path, "_test.go") {
				return nil
			}
			
			metas, err := processFile(path, *outputDir, stitcher)
			if err == nil {
				for _, m := range metas {
					classifier.Classify(m) // 進行分類
					allMeta = append(allMeta, *m)
				}
			}
		}
		return nil
	})

	if err != nil {
		fmt.Printf("❌ 掃描失敗: %v\n", err)
		return
	}

	fmt.Printf("📊 已處理 %d 個檔案，提取 %d 個函數\n", len(stitcher.ScriptNodes), len(allMeta))

	// 2. 數據流串接階段 (Stitching)
	stitcher.ResolveConnections()

	// 3. 統計分析階段 (Analysis)
	branchStats := stitcher.AnalyzeBranches()

	// 4. 輸出結果 (Output)
	
	// A. 系統全圖
	systemFlow := stitcher.GenerateSystemFlow()
	os.WriteFile(filepath.Join(*outputDir, "system_flow.mmd"), []byte(systemFlow), 0644)

	// B. CLI 指令手冊
	classResult := classifier.GetResult()
	cliDocs := GenerateCLI(classResult)
	os.WriteFile(filepath.Join(*outputDir, "cli_commands.sh"), []byte(cliDocs), 0644)

	// convertConnectionsToFlows 將 RealConnections 轉換成 Python FlowExecutor 需要的 flows 格式
	// 這是為了讓 Python 執行器能夠讀取並執行 Go 分析結果
	convertConnectionsToFlows := func(connections []Connection) []map[string]interface{} {
		flows := make([]map[string]interface{}, len(connections))
		
		for i, conn := range connections {
			flows[i] = map[string]interface{}{
				"id":         i + 1,
				"path":       []string{conn.FromFunc, conn.ToFunc},
				"full_path":  []string{conn.FromScript, conn.ToScript},
				"func_names": []string{conn.FromFunc, conn.ToFunc},
				"length":     2,
				"start":      conn.FromFunc,
				"end":        conn.ToFunc,
				"classifications": []map[string]interface{}{
					{
						"script":         conn.FromFunc,
						"module":         "go_module",
						"component_type": "程式組件",
						"description":    fmt.Sprintf("%s - Go 功能", conn.FromFunc),
					},
					{
						"script":         conn.ToFunc,
						"module":         "go_module",
						"component_type": "程式組件",
						"description":    fmt.Sprintf("%s - Go 功能", conn.ToFunc),
					},
				},
				"language":        "go",
				"cli_command":     fmt.Sprintf("go run %s", conn.ToScript),
				"structured_tags": []string{"language:go", "type:程式"},
			}
		}
		
		return flows
	}

	// C. 完整 JSON 報告 - 統一格式，相容 Python FlowExecutor
	flows := convertConnectionsToFlows(stitcher.RealConnections)
	
	report := map[string]interface{}{
		"metadata": map[string]interface{}{
			"tool":           "go2mermaid",
			"version":        "2.0",
			"language":       "go",
			"generated_at":   time.Now().Format(time.RFC3339),
			"total_flows":    len(flows),
			"total_files":    len(stitcher.ScriptNodes),
			"schema_version": "3.3",
			"ai_compatible":  true,
		},
		// ✅ 新增：FlowExecutor 需要的統一格式
		"flows": flows,
		
		// ✅ 保留：Go 工具原有的所有字段
		"summary": map[string]interface{}{
			"total_files":      len(stitcher.ScriptNodes),
			"total_funcs":      len(allMeta),
			"real_connections": len(stitcher.RealConnections),
			"categories":       classResult.Summary,
		},
		"branch_analysis": branchStats,
		"flow_chains":     stitcher.RealConnections,
		"functions":       allMeta,
	}
	
	jsonData, _ := json.MarshalIndent(report, "", "  ")
	os.WriteFile(filepath.Join(*outputDir, "analysis_results.json"), []byte(jsonData), 0644)

	fmt.Println("\n✅ 分析工作全部完成！")
	fmt.Printf("   📄 系統架構圖: %s/system_flow.mmd\n", *outputDir)
	fmt.Printf("   📄 完整數據報告: %s/analysis_results.json\n", *outputDir)
	fmt.Printf("   📄 CLI 指令手冊: %s/cli_commands.sh\n", *outputDir)
}
