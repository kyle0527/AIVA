// 最小 HTTP 測試 - 驗證 client.Do() 是否能真正發送請求
package main

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

func main() {
	fmt.Println("=== 最小 HTTP 測試 ===")
	fmt.Println("目標: http://localhost:8080/WebGoat")
	fmt.Println()

	// 創建與掃描器相同的 HTTP Client
	client := &http.Client{
		Timeout: 10 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	// 測試 1: 基本請求
	fmt.Println("測試 1: 基本 GET 請求到 /WebGoat")
	testURL := "http://localhost:8080/WebGoat"
	
	ctx := context.Background()
	req1, err := http.NewRequestWithContext(ctx, "GET", testURL, nil)
	if err != nil {
		fmt.Printf("❌ 創建請求失敗: %v\n", err)
		return
	}
	req1.Header.Set("User-Agent", "AIVA-SSRF-Scanner/1.0")

	start1 := time.Now()
	resp1, err1 := client.Do(req1)
	duration1 := time.Since(start1)

	if err1 != nil {
		fmt.Printf("❌ 請求失敗 (耗時 %v): %v\n", duration1, err1)
		fmt.Printf("   錯誤類型: %T\n", err1)
	} else {
		fmt.Printf("✅ 請求成功 (耗時 %v)\n", duration1)
		fmt.Printf("   狀態碼: %d\n", resp1.StatusCode)
		fmt.Printf("   Content-Type: %s\n", resp1.Header.Get("Content-Type"))
		resp1.Body.Close()
	}

	fmt.Println()

	// 測試 2: 帶 SSRF Payload 的請求
	fmt.Println("測試 2: 帶 SSRF Payload 的請求")
	testURL2 := "http://localhost:8080/WebGoat?url=http://169.254.169.254/latest/meta-data/"
	
	req2, err := http.NewRequestWithContext(ctx, "GET", testURL2, nil)
	if err != nil {
		fmt.Printf("❌ 創建請求失敗: %v\n", err)
		return
	}
	req2.Header.Set("User-Agent", "AIVA-SSRF-Scanner/1.0")

	start2 := time.Now()
	resp2, err2 := client.Do(req2)
	duration2 := time.Since(start2)

	if err2 != nil {
		fmt.Printf("❌ 請求失敗 (耗時 %v): %v\n", duration2, err2)
		fmt.Printf("   錯誤類型: %T\n", err2)
	} else {
		fmt.Printf("✅ 請求成功 (耗時 %v)\n", duration2)
		fmt.Printf("   狀態碼: %d\n", resp2.StatusCode)
		fmt.Printf("   Content-Type: %s\n", resp2.Header.Get("Content-Type"))
		resp2.Body.Close()
	}

	fmt.Println()

	// 測試 3: 連續 5 次請求（模擬掃描器行為）
	fmt.Println("測試 3: 連續 5 次請求")
	successCount := 0
	failCount := 0
	var totalDuration time.Duration

	for i := 1; i <= 5; i++ {
		testURL3 := fmt.Sprintf("http://localhost:8080/WebGoat?test=%d", i)
		req3, _ := http.NewRequestWithContext(ctx, "GET", testURL3, nil)
		req3.Header.Set("User-Agent", "AIVA-SSRF-Scanner/1.0")

		start3 := time.Now()
		resp3, err3 := client.Do(req3)
		duration3 := time.Since(start3)
		totalDuration += duration3

		if err3 != nil {
			failCount++
			fmt.Printf("  [%d] ❌ 失敗 (%v): %v\n", i, duration3, err3)
		} else {
			successCount++
			fmt.Printf("  [%d] ✅ 成功 (%v): HTTP %d\n", i, duration3, resp3.StatusCode)
			resp3.Body.Close()
		}
	}

	fmt.Println()
	fmt.Printf("統計: 成功 %d, 失敗 %d, 平均耗時 %v\n", 
		successCount, failCount, totalDuration/5)
	
	if successCount == 0 {
		fmt.Println()
		fmt.Println("⚠️ 所有請求都失敗了！")
		fmt.Println("可能原因:")
		fmt.Println("  1. WebGoat 沒有運行在 localhost:8080")
		fmt.Println("  2. 防火牆阻擋了連接")
		fmt.Println("  3. WebGoat 配置問題")
		fmt.Println()
		fmt.Println("請執行以下命令驗證靶場:")
		fmt.Println("  Test-NetConnection -ComputerName localhost -Port 8080")
	}
}
