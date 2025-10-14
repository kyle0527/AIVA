package config

import (
	"fmt"
	"os"
)

// Config 統一的配置結構
type Config struct {
	RabbitMQURL  string
	ServiceName  string
	LogLevel     string
	TaskQueue    string
	ResultQueue  string
	Environment  string
}

// LoadConfig 從環境變數載入配置
func LoadConfig(serviceName string) (*Config, error) {
	config := &Config{
		RabbitMQURL:  getEnv("RABBITMQ_URL", "amqp://aiva:dev_password@localhost:5672/"),
		ServiceName:  serviceName,
		LogLevel:     getEnv("LOG_LEVEL", "info"),
		TaskQueue:    getEnv("TASK_QUEUE", fmt.Sprintf("task.function.%s", serviceName)),
		ResultQueue:  getEnv("RESULT_QUEUE", "results.function.finding"),
		Environment:  getEnv("ENVIRONMENT", "development"),
	}
	
	// 驗證必要配置
	if config.RabbitMQURL == "" {
		return nil, fmt.Errorf("RABBITMQ_URL 不能為空")
	}
	
	return config, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
