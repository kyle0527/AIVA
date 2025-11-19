package main

import (
	"fmt"
	"os"

	"github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/cspm/scanner"
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

	// 創建 CSPM 掃描器
	cspmScanner := scanner.NewCSPMScanner(logger)

	logger.Info("CSPM Scanner initialized",
		zap.String("version", "1.0.0"),
		zap.String("mode", "standalone"),
	)

	// TODO: 實現 RabbitMQ Worker 或命令行參數處理
	// 目前只是驗證編譯
	_ = cspmScanner

	logger.Info("CSPM Scanner ready")
	select {} // 保持運行
}
