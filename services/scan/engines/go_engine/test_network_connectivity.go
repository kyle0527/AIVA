package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run test_network_connectivity.go <url>")
		fmt.Println("Example: go run test_network_connectivity.go http://localhost:8080/WebGoat")
		os.Exit(1)
	}

	testURL := os.Args[1]
	
	fmt.Printf("=== 測試網絡連通性 ===\n")
	fmt.Printf("目標 URL: %s\n", testURL)
	fmt.Printf("時間: %s\n\n", time.Now().Format("2006-01-02 15:04:05"))

	// 創建 HTTP Client (與 SSRF 掃描器相同配置)
	client := &http.Client{
		Timeout: 10 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	// 創建請求 (與 SSRF 掃描器相同方式)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", testURL, nil)
	if err != nil {
		fmt.Printf("❌ 創建請求失敗: %v\n", err)
		os.Exit(1)
	}

	// 設置相同的 User-Agent
	req.Header.Set("User-Agent", "AIVA-SSRF-Scanner/1.0")

	fmt.Printf("🚀 發送 HTTP 請求...\n")
	fmt.Printf("   Method: %s\n", req.Method)
	fmt.Printf("   Host: %s\n", req.Host)
	fmt.Printf("   User-Agent: %s\n", req.Header.Get("User-Agent"))
	fmt.Printf("   Timeout: %v\n", client.Timeout)
	fmt.Printf("\n")

	// 執行請求
	startTime := time.Now()
	resp, err := client.Do(req)
	duration := time.Since(startTime)

	fmt.Printf("⏱️  請求耗時: %v\n", duration)

	if err != nil {
		fmt.Printf("❌ HTTP 請求失敗:\n")
		fmt.Printf("   錯誤類型: %T\n", err)
		fmt.Printf("   錯誤訊息: %v\n", err)
		
		// 詳細錯誤分析
		switch {
		case ctx.Err() == context.DeadlineExceeded:
			fmt.Printf("   原因: 請求超時 (>%v)\n", client.Timeout)
		case ctx.Err() == context.Canceled:
			fmt.Printf("   原因: 請求被取消\n")
		default:
			fmt.Printf("   原因: 網絡連接問題 (DNS/連接被拒絕/防火牆等)\n")
		}
		
		fmt.Printf("\n🔍 建議檢查:\n")
		fmt.Printf("   1. 靶場是否正在運行\n")
		fmt.Printf("   2. 端口是否開放\n")
		fmt.Printf("   3. 防火牆是否阻擋\n")
		fmt.Printf("   4. URL 是否正確\n")
		
		os.Exit(1)
	}

	// 成功取得響應
	defer resp.Body.Close()
	
	fmt.Printf("✅ HTTP 請求成功!\n")
	fmt.Printf("   狀態碼: %d %s\n", resp.StatusCode, resp.Status)
	fmt.Printf("   Content-Type: %s\n", resp.Header.Get("Content-Type"))
	fmt.Printf("   Content-Length: %s\n", resp.Header.Get("Content-Length"))
	fmt.Printf("   Server: %s\n", resp.Header.Get("Server"))
	
	// 讀取前 500 個字元的響應內容
	body := make([]byte, 500)
	n, _ := resp.Body.Read(body)
	if n > 0 {
		fmt.Printf("\n📄 響應內容預覽 (前 %d 字元):\n", n)
		fmt.Printf("----------------------------------------\n")
		fmt.Printf("%s", string(body[:n]))
		if n == 500 {
			fmt.Printf("\n... (截斷)")
		}
		fmt.Printf("\n----------------------------------------\n")
	}

	fmt.Printf("\n🎉 網絡連通性測試通過!\n")
	fmt.Printf("   掃描器應該能夠連接到此目標\n")
}