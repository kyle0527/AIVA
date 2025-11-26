package main

import (
	"fmt"
	"os"

	"github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/sca/scanner"
	"go.uber.org/zap"
)

func main() {
	// 初始化 Logger
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	// 創建 SCA 掃描器
	scaScanner := scanner.NewSCAScanner(logger)

	logger.Info("SCA Scanner initialized",
		zap.String("version", "1.0.0"),
		zap.String("mode", "standalone"),
	)

	// v2.0 架構: 使用 CLI 模式，由 Python Coordinator 調用
	// 接收 JSON 參數，輸出 JSON 結果，無需 RabbitMQ
	// 目前只是驗證編譯
	_ = scaScanner

	logger.Info("SCA Scanner ready")
	select {} // 保持運行
}
