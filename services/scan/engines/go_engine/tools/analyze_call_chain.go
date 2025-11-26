// analyze_call_chain.go - 分析 Go 代碼的調用鏈,找出 HTTP 請求在哪裡被阻斷
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

type CallChain struct {
	Function   string
	Line       int
	Code       string
	CallsTo    []string
	HasReturn  bool
	ReturnType string
	Conditions []string
}

type Analysis struct {
	Chains map[string]*CallChain
	HTTPSendPoint string
	HTTPRecvPoint string
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run analyze_call_chain.go <go_file>")
		fmt.Println("Example: go run analyze_call_chain.go ../internal/ssrf/detector/ssrf.go")
		os.Exit(1)
	}

	filename := os.Args[1]
	
	// 解析文件
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		fmt.Printf("❌ 解析失敗: %v\n", err)
		os.Exit(1)
	}

	analysis := &Analysis{
		Chains: make(map[string]*CallChain),
	}

	// 遍歷 AST
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncDecl:
			analyzeFuncDecl(fset, x, analysis)
		}
		return true
	})

	// 輸出分析結果
	printAnalysis(analysis)
}

func analyzeFuncDecl(fset *token.FileSet, fn *ast.FuncDecl, analysis *Analysis) {
	funcName := fn.Name.Name
	
	chain := &CallChain{
		Function: funcName,
		Line:     fset.Position(fn.Pos()).Line,
		CallsTo:  []string{},
		Conditions: []string{},
	}

	// 檢查是否包含 HTTP 請求相關代碼
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.AssignStmt:
			// 尋找 client.Do(req) 調用
			for _, expr := range stmt.Rhs {
				if call, ok := expr.(*ast.CallExpr); ok {
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if sel.Sel.Name == "Do" {
							if x, ok := sel.X.(*ast.Ident); ok && x.Name == "client" {
								line := fset.Position(call.Pos()).Line
								analysis.HTTPSendPoint = fmt.Sprintf("%s:%d", funcName, line)
								chain.Code = fmt.Sprintf("Line %d: resp, err := d.client.Do(req)", line)
							}
						}
					}
					
					// 記錄所有函數調用
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						callTarget := sel.Sel.Name
						chain.CallsTo = append(chain.CallsTo, callTarget)
					} else if ident, ok := call.Fun.(*ast.Ident); ok {
						chain.CallsTo = append(chain.CallsTo, ident.Name)
					}
				}
			}
			
			// 尋找 io.ReadAll 調用
			for _, expr := range stmt.Rhs {
				if call, ok := expr.(*ast.CallExpr); ok {
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if sel.Sel.Name == "ReadAll" {
							if x, ok := sel.X.(*ast.Ident); ok && x.Name == "io" {
								line := fset.Position(call.Pos()).Line
								analysis.HTTPRecvPoint = fmt.Sprintf("%s:%d", funcName, line)
							}
						}
					}
				}
			}

		case *ast.IfStmt:
			// 分析 if 條件
			if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
				cond := analyzeCondition(fset, binExpr)
				chain.Conditions = append(chain.Conditions, cond)
				
				// 檢查是否有 early return
				if hasReturn(stmt.Body) {
					chain.HasReturn = true
				}
			}

		case *ast.ReturnStmt:
			chain.HasReturn = true
			if len(stmt.Results) > 0 {
				if ident, ok := stmt.Results[0].(*ast.Ident); ok {
					chain.ReturnType = ident.Name
				}
			}
		}
		return true
	})

	analysis.Chains[funcName] = chain
}

func analyzeCondition(fset *token.FileSet, expr *ast.BinaryExpr) string {
	left := exprToString(expr.X)
	op := expr.Op.String()
	right := exprToString(expr.Y)
	line := fset.Position(expr.Pos()).Line
	return fmt.Sprintf("Line %d: if %s %s %s", line, left, op, right)
}

func exprToString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.BasicLit:
		return e.Value
	case *ast.BinaryExpr:
		return fmt.Sprintf("(%s %s %s)", exprToString(e.X), e.Op.String(), exprToString(e.Y))
	case *ast.SelectorExpr:
		return fmt.Sprintf("%s.%s", exprToString(e.X), e.Sel.Name)
	default:
		return "..."
	}
}

func hasReturn(block *ast.BlockStmt) bool {
	hasRet := false
	ast.Inspect(block, func(n ast.Node) bool {
		if _, ok := n.(*ast.ReturnStmt); ok {
			hasRet = true
			return false
		}
		return true
	})
	return hasRet
}

func printAnalysis(analysis *Analysis) {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("🔍 Go 代碼調用鏈分析")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()

	fmt.Println("📍 關鍵點定位:")
	fmt.Printf("  ✅ HTTP 發送點: %s\n", analysis.HTTPSendPoint)
	fmt.Printf("  ✅ HTTP 接收點: %s\n", analysis.HTTPRecvPoint)
	fmt.Println()

	// 找出包含 HTTP 操作的函數
	fmt.Println("🔗 調用鏈分析:")
	fmt.Println()

	for funcName, chain := range analysis.Chains {
		if strings.Contains(funcName, "test") || strings.Contains(funcName, "SSRF") || 
		   strings.Contains(funcName, "Scan") || len(chain.Conditions) > 0 {
			fmt.Printf("📦 函數: %s (Line %d)\n", funcName, chain.Line)
			
			if chain.Code != "" {
				fmt.Printf("   💡 關鍵代碼: %s\n", chain.Code)
			}
			
			if len(chain.Conditions) > 0 {
				fmt.Println("   ⚠️  條件判斷:")
				for _, cond := range chain.Conditions {
					fmt.Printf("      • %s\n", cond)
				}
			}
			
			if chain.HasReturn {
				fmt.Printf("   ↩️  有 Return 語句")
				if chain.ReturnType != "" {
					fmt.Printf(" (返回: %s)", chain.ReturnType)
				}
				fmt.Println()
			}
			
			if len(chain.CallsTo) > 0 {
				fmt.Printf("   📞 調用: %s\n", strings.Join(chain.CallsTo, ", "))
			}
			
			fmt.Println()
		}
	}

	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("💡 診斷建議:")
	fmt.Println(strings.Repeat("=", 80))
	
	// 從 HTTP 發送點追蹤
	if analysis.HTTPSendPoint != "" {
		parts := strings.Split(analysis.HTTPSendPoint, ":")
		funcName := parts[0]
		
		if chain, ok := analysis.Chains[funcName]; ok {
			fmt.Printf("\n在函數 %s 中:\n", funcName)
			
			if len(chain.Conditions) > 0 {
				fmt.Println("\n⚠️  發現條件判斷可能阻斷執行流:")
				for i, cond := range chain.Conditions {
					fmt.Printf("  %d. %s\n", i+1, cond)
					
					// 檢查是否為錯誤檢查
					if strings.Contains(cond, "err") && strings.Contains(cond, "!=") {
						fmt.Println("     ⚠️  這是錯誤檢查!如果 err != nil,可能直接返回")
					}
				}
			}
			
			if chain.HasReturn && chain.ReturnType == "nil" {
				fmt.Println("\n❌ 警告:函數會返回 nil,可能導致請求結果被忽略!")
			}
		}
	}
	
	fmt.Println()
}
