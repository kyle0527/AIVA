package mq

import (
	"testing"
)

func TestMaskPassword(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "標準 AMQP URL",
			input:    "amqp://aiva:dev_password@localhost:5672/",
			expected: "amqp://aiva:%2A%2A%2A@localhost:5672/", // URL 編碼後的 ***
		},
		{
			name:     "無密碼",
			input:    "amqp://localhost:5672/",
			expected: "amqp://localhost:5672/",
		},
		{
			name:     "空字串",
			input:    "",
			expected: "",
		},
		{
			name:     "複雜密碼",
			input:    "amqp://user:P@ssw0rd!@remote.host:5672/vhost",
			expected: "amqp://user:%2A%2A%2A@remote.host:5672/vhost", // URL 編碼後的 ***
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := maskPassword(tt.input)
			if result != tt.expected {
				t.Errorf("maskPassword(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestNewMQClient(t *testing.T) {
	// 這是一個範例測試,實際需要 mock RabbitMQ 連接
	t.Skip("需要 RabbitMQ 伺服器才能執行此測試")

	// client, err := NewMQClient("amqp://guest:guest@localhost:5672/", nil)
	// if err != nil {
	// 	t.Fatalf("NewMQClient() error = %v", err)
	// }
	// defer client.Close()
}
