package main

import (
	"log"
	"os"

	"github.com/kyle0527/aiva/services/scan/go_engine/internal/sca/scanner"
)

func main() {
	// 初始化 Logger (使用標準庫)
	log.SetOutput(os.Stdout)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// 創建 SCA 掃描器
	scaScanner := scanner.NewSCAScanner()

	log.Println("[INFO] SCA Scanner initialized in standalone mode")

	// v2.0 架構: 使用 CLI 模式，由 Python Coordinator 調用
	// 接收 JSON 參數，輸出 JSON 結果，無需 RabbitMQ
	// 目前只是驗證編譯
	_ = scaScanner

	log.Println("[INFO] SCA Scanner ready")
	select {} // 保持運行
}

