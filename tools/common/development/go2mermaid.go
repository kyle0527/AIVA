// go2mermaid.go - Go AST 解析與 Mermaid 流程圖產生工具
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Node 代表流程圖中的一個節點
type Node struct {
	ID       string
	Label    string
	Kind     string // op, cond, start, end
	Nexts    []*Node
	EdgeInfo map[string]EdgeStyle
}

// EdgeStyle 定義邊的樣式和標籤
type EdgeStyle struct {
	Label string
	Style string // -->, -.->等
}

// Graph 代表整個流程圖
type Graph struct {
	Title     string
	Direction string
	Nodes     []*Node
	Counter   int
	Start     *Node
	End       *Node
}

// NewGraph 創建新的流程圖
func NewGraph(title, direction string) *Graph {
	g := &Graph{
		Title:     title,
		Direction: validateDirection(direction),
		Nodes:     []*Node{},
		Counter:   0,
	}
	g.Start = g.Add("start", "開始")
	g.End = g.Add("end", "結束")
	return g
}

func validateDirection(dir string) string {
	dir = strings.ToUpper(dir)
	validDirs := map[string]bool{"TB": true, "TD": true, "BT": true, "LR": true, "RL": true}
	if validDirs[dir] {
		if dir == "TD" {
			return "TB" // TD deprecated
		}
		return dir
	}
	return "TB"
}

// Add 添加節點到圖
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

// Link 連接兩個節點
func (g *Graph) Link(from, to *Node, label, style string) {
	if from == nil || to == nil {
		return
	}
	
	// 避免重複連接
	for _, next := range from.Nexts {
		if next == to {
			return
		}
	}
	
	from.Nexts = append(from.Nexts, to)
	if label != "" || style != "" {
		from.EdgeInfo[to.ID] = EdgeStyle{Label: label, Style: style}
	}
}

// ToMermaid 生成 Mermaid 語法
func (g *Graph) ToMermaid() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("flowchart %s\n", g.Direction))
	
	// 輸出節點定義
	for _, n := range g.Nodes {
		sb.WriteString(fmt.Sprintf("    %s\n", formatNode(n)))
	}
	
	sb.WriteString("\n")
	
	// 輸出連接
	for _, n := range g.Nodes {
		for i, next := range n.Nexts {
			edge := formatEdge(n, next, i)
			sb.WriteString(fmt.Sprintf("    %s\n", edge))
		}
	}
	
	return sb.String()
}

func formatNode(n *Node) string {
	text := sanitizeText(n.Label)
	
	// 如果標籤為空，使用不間斷空格
	if text == "" {
		text = "&#160;"
	}
	
	// 根據 Mermaid 官方標準，包含特殊字符的文本需要用雙引號包裹
	switch n.Kind {
	case "cond":
		// 菱形節點
		return fmt.Sprintf("%s{\"%s\"}", n.ID, text)
	case "start", "end":
		// 圓角矩形 (體育場形狀)
		return fmt.Sprintf("%s([\"%s\"])", n.ID, text)
	case "subgraph":
		// 平行四邊形
		return fmt.Sprintf("%s[/\"%s\"/]", n.ID, text)
	default:
		// 矩形
		return fmt.Sprintf("%s[\"%s\"]", n.ID, text)
	}
}

func formatEdge(from, to *Node, index int) string {
	edge := from.EdgeInfo[to.ID]
	
	if from.Kind == "cond" {
		if index == 0 {
			return fmt.Sprintf("%s -->|Yes| %s", from.ID, to.ID)
		} else if index == 1 {
			return fmt.Sprintf("%s -->|No| %s", from.ID, to.ID)
		}
	}
	
	if edge.Label != "" {
		return fmt.Sprintf("%s -->|%s| %s", from.ID, edge.Label, to.ID)
	}
	
	style := edge.Style
	if style == "" {
		style = "-->"
	}
	return fmt.Sprintf("%s %s %s", from.ID, style, to.ID)
}

func sanitizeText(text string) string {
	// Mermaid 官方標準: 在雙引號包裹的文本中，只需要轉義少數字符
	// 關鍵: & 必須最先轉換!
	
	// 1. 先轉換 & (必須第一個，否則會破壞後續的 HTML 實體)
	text = strings.ReplaceAll(text, "&", "&amp;")
	
	// 2. 轉換雙引號和尖括號
	text = strings.ReplaceAll(text, `"`, "&quot;")
	text = strings.ReplaceAll(text, "<", "&lt;")
	text = strings.ReplaceAll(text, ">", "&gt;")
	
	// 3. 處理換行和控制字符 (換行使用 <br/> 標籤)
	text = strings.ReplaceAll(text, "\n", "<br/>")
	text = strings.ReplaceAll(text, "\r", "")
	text = strings.ReplaceAll(text, "\t", " ")
	
	// 4. 清理多餘空格
	text = strings.TrimSpace(text)
	re := regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")
	
	// 5. 限制長度
	if len(text) > 50 {
		text = text[:47] + "..."
	}
	
	return text
}

// Builder 構建流程圖
type Builder struct {
	fset  *token.FileSet
	graph *Graph
}

// NewBuilder 創建新的構建器
func NewBuilder(fset *token.FileSet, title, direction string) *Builder {
	return &Builder{
		fset:  fset,
		graph: NewGraph(title, direction),
	}
}

// BuildFunction 為函數構建流程圖
func (b *Builder) BuildFunction(fn *ast.FuncDecl) *Graph {
	last := b.graph.Start
	
	if fn.Body != nil {
		last = b.buildBlockStmt(fn.Body, last)
	}
	
	b.graph.Link(last, b.graph.End, "", "")
	return b.graph
}

func (b *Builder) buildBlockStmt(block *ast.BlockStmt, entry *Node) *Node {
	if block == nil {
		return entry
	}
	
	last := entry
	for _, stmt := range block.List {
		last = b.buildStmt(stmt, last)
	}
	return last
}

func (b *Builder) buildStmt(stmt ast.Stmt, entry *Node) *Node {
	if stmt == nil {
		return entry
	}
	
	switch s := stmt.(type) {
	case *ast.IfStmt:
		return b.buildIfStmt(s, entry)
	case *ast.ForStmt:
		return b.buildForStmt(s, entry)
	case *ast.RangeStmt:
		return b.buildRangeStmt(s, entry)
	case *ast.SwitchStmt:
		return b.buildSwitchStmt(s, entry)
	case *ast.SelectStmt:
		return b.buildSelectStmt(s, entry)
	case *ast.ReturnStmt:
		return b.buildReturnStmt(s, entry)
	case *ast.DeferStmt:
		return b.buildDeferStmt(s, entry)
	case *ast.GoStmt:
		return b.buildGoStmt(s, entry)
	case *ast.BlockStmt:
		return b.buildBlockStmt(s, entry)
	default:
		label := b.getStmtLabel(stmt)
		node := b.graph.Add("op", label)
		b.graph.Link(entry, node, "", "")
		return node
	}
}

func (b *Builder) buildIfStmt(stmt *ast.IfStmt, entry *Node) *Node {
	condText := b.exprToString(stmt.Cond)
	if len(condText) > 30 {
		condText = condText[:27] + "..."
	}
	
	condNode := b.graph.Add("cond", fmt.Sprintf("if %s", condText))
	b.graph.Link(entry, condNode, "", "")
	
	// Then 分支
	thenLast := b.buildBlockStmt(stmt.Body, condNode)
	
	if stmt.Else != nil {
		// Else 分支
		elseLast := b.buildStmt(stmt.Else, condNode)
		
		// 合併點
		merge := b.graph.Add("op", "")
		b.graph.Link(thenLast, merge, "", "")
		b.graph.Link(elseLast, merge, "", "")
		return merge
	}
	
	// 沒有 else，創建合併點
	merge := b.graph.Add("op", "")
	b.graph.Link(thenLast, merge, "", "")
	b.graph.Link(condNode, merge, "No", "")
	return merge
}

func (b *Builder) buildForStmt(stmt *ast.ForStmt, entry *Node) *Node {
	var condText string
	if stmt.Cond != nil {
		condText = b.exprToString(stmt.Cond)
	} else {
		condText = "true"
	}
	
	if len(condText) > 25 {
		condText = condText[:22] + "..."
	}
	
	forNode := b.graph.Add("cond", fmt.Sprintf("for %s", condText))
	b.graph.Link(entry, forNode, "", "")
	
	if stmt.Body != nil {
		bodyLast := b.buildBlockStmt(stmt.Body, forNode)
		b.graph.Link(bodyLast, forNode, "loop", "")
	}
	
	exit := b.graph.Add("op", "")
	b.graph.Link(forNode, exit, "exit", "")
	return exit
}

func (b *Builder) buildRangeStmt(stmt *ast.RangeStmt, entry *Node) *Node {
	key := "_"
	if stmt.Key != nil {
		key = b.exprToString(stmt.Key)
	}
	
	value := ""
	if stmt.Value != nil {
		value = ", " + b.exprToString(stmt.Value)
	}
	
	iter := b.exprToString(stmt.X)
	if len(iter) > 20 {
		iter = iter[:17] + "..."
	}
	
	rangeNode := b.graph.Add("cond", fmt.Sprintf("for %s%s := range %s", key, value, iter))
	b.graph.Link(entry, rangeNode, "", "")
	
	if stmt.Body != nil {
		bodyLast := b.buildBlockStmt(stmt.Body, rangeNode)
		b.graph.Link(bodyLast, rangeNode, "loop", "")
	}
	
	exit := b.graph.Add("op", "")
	b.graph.Link(rangeNode, exit, "exit", "")
	return exit
}

func (b *Builder) buildSwitchStmt(stmt *ast.SwitchStmt, entry *Node) *Node {
	tagText := "switch"
	if stmt.Tag != nil {
		tagText = fmt.Sprintf("switch %s", b.exprToString(stmt.Tag))
	}
	
	switchNode := b.graph.Add("cond", tagText)
	b.graph.Link(entry, switchNode, "", "")
	
	merge := b.graph.Add("op", "")
	
	for _, clause := range stmt.Body.List {
		if cc, ok := clause.(*ast.CaseClause); ok {
			var caseLabel string
			if cc.List == nil {
				caseLabel = "default"
			} else {
				cases := []string{}
				for _, expr := range cc.List {
					cases = append(cases, b.exprToString(expr))
				}
				caseLabel = fmt.Sprintf("case %s", strings.Join(cases, ", "))
			}
			
			caseNode := b.graph.Add("op", caseLabel)
			b.graph.Link(switchNode, caseNode, "", "")
			
			caseLast := caseNode
			for _, caseStmt := range cc.Body {
				caseLast = b.buildStmt(caseStmt, caseLast)
			}
			
			b.graph.Link(caseLast, merge, "", "")
		}
	}
	
	return merge
}

func (b *Builder) buildSelectStmt(stmt *ast.SelectStmt, entry *Node) *Node {
	selectNode := b.graph.Add("cond", "select")
	b.graph.Link(entry, selectNode, "", "")
	
	merge := b.graph.Add("op", "")
	
	for _, clause := range stmt.Body.List {
		if cc, ok := clause.(*ast.CommClause); ok {
			var caseLabel string
			if cc.Comm == nil {
				caseLabel = "default"
			} else {
				caseLabel = fmt.Sprintf("case %s", b.getStmtLabel(cc.Comm))
			}
			
			caseNode := b.graph.Add("op", caseLabel)
			b.graph.Link(selectNode, caseNode, "", "")
			
			caseLast := caseNode
			for _, caseStmt := range cc.Body {
				caseLast = b.buildStmt(caseStmt, caseLast)
			}
			
			b.graph.Link(caseLast, merge, "", "")
		}
	}
	
	return merge
}

func (b *Builder) buildReturnStmt(stmt *ast.ReturnStmt, entry *Node) *Node {
	label := "return"
	if len(stmt.Results) > 0 {
		results := []string{}
		for _, result := range stmt.Results {
			results = append(results, b.exprToString(result))
		}
		returnVal := strings.Join(results, ", ")
		if len(returnVal) > 30 {
			returnVal = returnVal[:27] + "..."
		}
		label = fmt.Sprintf("return %s", returnVal)
	}
	
	node := b.graph.Add("op", label)
	b.graph.Link(entry, node, "", "")
	return node
}

func (b *Builder) buildDeferStmt(stmt *ast.DeferStmt, entry *Node) *Node {
	callStr := b.exprToString(stmt.Call.Fun)
	node := b.graph.Add("op", fmt.Sprintf("defer %s", callStr))
	b.graph.Link(entry, node, "", "")
	return node
}

func (b *Builder) buildGoStmt(stmt *ast.GoStmt, entry *Node) *Node {
	callStr := b.exprToString(stmt.Call.Fun)
	node := b.graph.Add("op", fmt.Sprintf("go %s", callStr))
	b.graph.Link(entry, node, "", "")
	return node
}

func (b *Builder) getStmtLabel(stmt ast.Stmt) string {
	pos := b.fset.Position(stmt.Pos())
	line := pos.Line
	
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		lhs := []string{}
		for _, l := range s.Lhs {
			lhs = append(lhs, b.exprToString(l))
		}
		
		rhs := []string{}
		for _, r := range s.Rhs {
			rhs = append(rhs, b.exprToString(r))
		}
		
		lhsStr := strings.Join(lhs, ", ")
		rhsStr := strings.Join(rhs, ", ")
		
		if len(rhsStr) > 20 {
			rhsStr = rhsStr[:17] + "..."
		}
		
		op := "="
		if s.Tok.String() != "=" {
			op = s.Tok.String()
		}
		
		return fmt.Sprintf("[L%d] %s %s %s", line, lhsStr, op, rhsStr)
		
	case *ast.ExprStmt:
		exprStr := b.exprToString(s.X)
		if len(exprStr) > 40 {
			exprStr = exprStr[:37] + "..."
		}
		return fmt.Sprintf("[L%d] %s", line, exprStr)
		
	case *ast.SendStmt:
		return fmt.Sprintf("[L%d] %s <- %s", line, b.exprToString(s.Chan), b.exprToString(s.Value))
		
	case *ast.IncDecStmt:
		return fmt.Sprintf("[L%d] %s%s", line, b.exprToString(s.X), s.Tok.String())
		
	default:
		return fmt.Sprintf("[L%d] %T", line, stmt)
	}
}

func (b *Builder) exprToString(expr ast.Expr) string {
	if expr == nil {
		return ""
	}
	
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
		
	case *ast.BasicLit:
		return e.Value
		
	case *ast.BinaryExpr:
		left := b.exprToString(e.X)
		right := b.exprToString(e.Y)
		return fmt.Sprintf("%s %s %s", left, e.Op.String(), right)
		
	case *ast.UnaryExpr:
		return fmt.Sprintf("%s%s", e.Op.String(), b.exprToString(e.X))
		
	case *ast.CallExpr:
		funName := b.exprToString(e.Fun)
		args := []string{}
		for _, arg := range e.Args {
			argStr := b.exprToString(arg)
			if len(argStr) > 15 {
				argStr = argStr[:12] + "..."
			}
			args = append(args, argStr)
		}
		
		argsStr := strings.Join(args, ", ")
		if len(argsStr) > 30 {
			argsStr = "..."
		}
		
		return fmt.Sprintf("%s(%s)", funName, argsStr)
		
	case *ast.SelectorExpr:
		return fmt.Sprintf("%s.%s", b.exprToString(e.X), e.Sel.Name)
		
	case *ast.IndexExpr:
		return fmt.Sprintf("%s[%s]", b.exprToString(e.X), b.exprToString(e.Index))
		
	case *ast.StarExpr:
		return fmt.Sprintf("*%s", b.exprToString(e.X))
		
	case *ast.CompositeLit:
		typeName := ""
		if e.Type != nil {
			typeName = b.exprToString(e.Type)
		}
		return fmt.Sprintf("%s{...}", typeName)
		
	default:
		return "..."
	}
}

// scanGoFiles 掃描目錄中的 Go 文件
func scanGoFiles(root string, ignore []string, maxFiles int) ([]string, error) {
	var files []string
	
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// 檢查是否在忽略列表中
		for _, ig := range ignore {
			if strings.Contains(path, ig) {
				if info.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}
		
		// 只處理 .go 文件
		if !info.IsDir() && strings.HasSuffix(path, ".go") {
			files = append(files, path)
			if len(files) >= maxFiles {
				return filepath.SkipDir
			}
		}
		
		return nil
	})
	
	return files, err
}

// analyzeFile 分析單個 Go 文件
func analyzeFile(filePath, direction string) ([]Chart, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	
	var charts []Chart
	
	// 為每個函數生成流程圖
	for _, decl := range node.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcName := fn.Name.Name
			
			// 構建參數列表
			params := []string{}
			if fn.Type.Params != nil {
				for _, field := range fn.Type.Params.List {
					for _, name := range field.Names {
						params = append(params, name.Name)
					}
				}
			}
			
			title := fmt.Sprintf("Function: %s(%s)", funcName, strings.Join(params, ", "))
			if len(title) > 60 {
				title = fmt.Sprintf("Function: %s(...)", funcName)
			}
			
			builder := NewBuilder(fset, title, direction)
			graph := builder.BuildFunction(fn)
			
			charts = append(charts, Chart{
				Name:    funcName,
				Mermaid: graph.ToMermaid(),
			})
		}
	}
	
	return charts, nil
}

// Chart 代表一個流程圖
type Chart struct {
	Name    string
	Mermaid string
}

func main() {
	// 命令行參數
	inputPath := flag.String("i", "", "輸入文件或目錄路徑")
	outputDir := flag.String("o", "./diagrams", "輸出目錄")
	direction := flag.String("d", "TB", "流程圖方向 (TB/LR/RL/BT)")
	maxFiles := flag.Int("m", 1000, "最大處理文件數")
	
	flag.Parse()
	
	if *inputPath == "" {
		fmt.Println("錯誤: 必須指定輸入路徑")
		fmt.Println("\n使用方法:")
		fmt.Println("  go run go2mermaid.go -i <file_or_dir> [-o <output_dir>] [-d <direction>]")
		fmt.Println("\n範例:")
		fmt.Println("  go run go2mermaid.go -i ../internal/ssrf/detector/ssrf.go")
		fmt.Println("  go run go2mermaid.go -i ../internal -o ./output -d LR")
		os.Exit(1)
	}
	
	info, err := os.Stat(*inputPath)
	if err != nil {
		fmt.Printf("錯誤: 無法訪問路徑 %s: %v\n", *inputPath, err)
		os.Exit(1)
	}
	
	// 創建輸出目錄
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Printf("錯誤: 無法創建輸出目錄: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Go 程式碼流程圖生成工具")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("輸入路徑: %s\n", *inputPath)
	fmt.Printf("輸出目錄: %s\n", *outputDir)
	fmt.Printf("流程圖方向: %s\n", *direction)
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
	
	var files []string
	
	if info.IsDir() {
		// 掃描目錄
		ignore := []string{"vendor", ".git", "testdata", "node_modules"}
		files, err = scanGoFiles(*inputPath, ignore, *maxFiles)
		if err != nil {
			fmt.Printf("錯誤: 掃描目錄失敗: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("找到 %d 個 Go 文件\n\n", len(files))
	} else {
		// 單個文件
		files = []string{*inputPath}
	}
	
	totalCharts := 0
	
	for i, filePath := range files {
		if (i+1)%10 == 0 {
			fmt.Printf("進度: %d/%d\n", i+1, len(files))
		}
		
		charts, err := analyzeFile(filePath, *direction)
		if err != nil {
			fmt.Printf("警告: 分析文件 %s 失敗: %v\n", filePath, err)
			continue
		}
		
		if len(charts) == 0 {
			continue
		}
		
		// 生成輸出文件名
		relPath, _ := filepath.Rel(*inputPath, filePath)
		if relPath == "." || relPath == filePath {
			relPath = filepath.Base(filePath)
		}
		
		// 清理文件名
		re := regexp.MustCompile(`[^a-zA-Z0-9_-]+`)
		cleanPath := re.ReplaceAllString(relPath, "_")
		cleanPath = strings.TrimSuffix(cleanPath, "_go")
		
		// 為每個函數生成單獨的文件
		for _, chart := range charts {
			safeName := re.ReplaceAllString(chart.Name, "_")
			outputFile := filepath.Join(*outputDir, fmt.Sprintf("%s_%s.mmd", cleanPath, safeName))
			
			if err := os.WriteFile(outputFile, []byte(chart.Mermaid), 0644); err != nil {
				fmt.Printf("警告: 寫入文件 %s 失敗: %v\n", outputFile, err)
				continue
			}
			
			totalCharts++
		}
	}
	
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("完成！共生成 %d 個流程圖文件\n", totalCharts)
	fmt.Printf("輸出目錄: %s\n", *outputDir)
	fmt.Println(strings.Repeat("=", 80))
}
