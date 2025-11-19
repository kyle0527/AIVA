package oob

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// =====================================================================
// OOB (Out-of-Band) Monitor - SSRF 無回顯檢測
// =====================================================================
// 來源: C:\Users\User\Downloads\新增資料夾 (6)\collaborator.go
// 用途: 監聽外部回連，驗證無回顯 SSRF 漏洞
// =====================================================================

// Interaction 代表一次外部回連記錄
type Interaction struct {
	Token     string    `json:"token"`
	SourceIP  string    `json:"source_ip"`
	Timestamp time.Time `json:"timestamp"`
	Type      string    `json:"type"` // HTTP or DNS
}

// Monitor 負責監聽外部回連 (Collaborator Server)
// 在實際部署中，這通常是一個獨立的服務，但為了簡化，我們可以將其嵌入在 Worker 中
type Monitor struct {
	hits      map[string][]Interaction // 存儲收到的回連 (Token -> Interactions)
	mu        sync.RWMutex
	ServerURL string // 本機監聽的外部可訪問 URL (例如 http://1.2.3.4:8080)
}

// NewMonitor 創建新的 OOB 監測器
// publicIP: 公網可訪問的 IP 地址
// port: 監聽端口 (建議 8080)
func NewMonitor(publicIP string, port int) *Monitor {
	return &Monitor{
		hits:      make(map[string][]Interaction),
		ServerURL: fmt.Sprintf("http://%s:%d", publicIP, port),
	}
}

// GeneratePayload 生成一個帶有唯一 Token 的測試 Payload
// 返回: (token, payload_url)
// 例如: ("abc-123-xyz", "http://1.2.3.4:8080/oob/abc-123-xyz")
func (m *Monitor) GeneratePayload() (string, string) {
	token := uuid.New().String()
	return token, fmt.Sprintf("%s/oob/%s", m.ServerURL, token)
}

// StartHTTP 啟動 HTTP 監聽器 (非阻塞)
// 監聽 /oob/{token} 路徑的訪問請求
func (m *Monitor) StartHTTP(port int) {
	http.HandleFunc("/oob/", func(w http.ResponseWriter, r *http.Request) {
		token := r.URL.Path[len("/oob/"):]
		
		m.recordInteraction(token, r.RemoteAddr, "HTTP")
		
		// 返回一個看起來正常的響應，避免被目標懷疑
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	go func() {
		logrus.Infof("OOB Monitor listening on port %d", port)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
			logrus.Errorf("Failed to start OOB monitor: %v", err)
		}
	}()
}

// Verify 檢查指定的 Token 是否收到過回連
// 用於驗證 SSRF Payload 是否成功觸發外部請求
func (m *Monitor) Verify(token string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, exists := m.hits[token]
	return exists
}

// GetInteractions 獲取指定 Token 的所有回連記錄
func (m *Monitor) GetInteractions(token string) []Interaction {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.hits[token]
}

// recordInteraction 記錄一次回連事件（內部方法）
func (m *Monitor) recordInteraction(token, ip, interactionType string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	logrus.Warnf("[OOB] Detected interaction for token %s from %s", token, ip)
	
	m.hits[token] = append(m.hits[token], Interaction{
		Token:     token,
		SourceIP:  ip,
		Timestamp: time.Now(),
		Type:      interactionType,
	})
}
