package models

// FunctionTaskPayload AuthN 任務載荷
type FunctionTaskPayload struct {
	TaskID       string      `json:"task_id"`
	FunctionType string      `json:"function_type"`
	Target       TaskTarget  `json:"target"`
	Options      TaskOptions `json:"options,omitempty"`
}

// TaskTarget 測試目標
type TaskTarget struct {
	URL      string            `json:"url"`
	Username string            `json:"username,omitempty"`
	Headers  map[string]string `json:"headers,omitempty"`
}

// TaskOptions 測試選項
type TaskOptions struct {
	TestType       string   `json:"test_type,omitempty"`        // brute_force, weak_config, token
	Usernames      []string `json:"usernames,omitempty"`        // 暴力破解用戶名列表
	Passwords      []string `json:"passwords,omitempty"`        // 暴力破解密碼列表
	MaxAttempts    int      `json:"max_attempts,omitempty"`     // 最大嘗試次數
	RateLimitDelay int      `json:"rate_limit_delay,omitempty"` // 限速延遲(ms)
	JWTToken       string   `json:"jwt_token,omitempty"`        // JWT token
	SessionToken   string   `json:"session_token,omitempty"`    // Session token
}

// FindingPayload AuthN 發現載荷
type FindingPayload struct {
	FindingID      string          `json:"finding_id"`
	TaskID         string          `json:"task_id"`
	ScanID         string          `json:"scan_id"`
	Status         string          `json:"status"`
	Vulnerability  Vulnerability   `json:"vulnerability"`
	Target         FindingTarget   `json:"target"`
	Evidence       FindingEvidence `json:"evidence"`
	Impact         FindingImpact   `json:"impact"`
	Recommendation string          `json:"recommendation"`
}

// Vulnerability 漏洞資訊
type Vulnerability struct {
	Name       string `json:"name"`
	CWE        string `json:"cwe"`
	Severity   string `json:"severity"`
	Confidence string `json:"confidence"`
}

// FindingTarget 發現目標
type FindingTarget struct {
	URL      string            `json:"url"`
	Endpoint string            `json:"endpoint"`
	Method   string            `json:"method"`
	Headers  map[string]string `json:"headers,omitempty"`
}

// FindingEvidence 證據
type FindingEvidence struct {
	Request        string            `json:"request"`
	Response       string            `json:"response"`
	Payload        string            `json:"payload"`
	ProofOfConcept string            `json:"proof_of_concept"`
	Details        map[string]string `json:"details,omitempty"`
}

// FindingImpact 影響
type FindingImpact struct {
	Description    string `json:"description"`
	BusinessImpact string `json:"business_impact"`
	Exploitability string `json:"exploitability"`
}

// BruteForceResult 暴力破解結果
type BruteForceResult struct {
	Vulnerable    bool
	Username      string
	Password      string
	AttemptsCount int
	ResponseTime  int64
	RateLimited   bool
	AccountLocked bool
}

// WeakConfigResult 弱配置結果
type WeakConfigResult struct {
	Vulnerable       bool
	ConfigType       string // password_policy, session_timeout, etc.
	Details          string
	ActualValue      string
	RecommendedValue string
}

// TokenTestResult Token 測試結果
type TokenTestResult struct {
	Vulnerable     bool
	TokenType      string // jwt, session, api_key
	Issue          string
	Details        string
	DecodedPayload map[string]interface{}
}
