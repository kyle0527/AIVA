package internal

import (
	"fmt"
)

type AuthnTask struct {
    Username string            `json:"username"`
    Extra    map[string]string `json:"extra,omitempty"`
}

type AuthnFinding struct {
    Name        string `json:"name"`
    Description string `json:"description"`
    Severity    string `json:"severity"`
    Evidence    string `json:"evidence,omitempty"`
}

type AuthnEngine struct {
    config AuthnConfig
}

func NewAuthnEngine(cfg AuthnConfig) *AuthnEngine {
    return &AuthnEngine{config: cfg}
}

func (e *AuthnEngine) RunTests(task AuthnTask) ([]AuthnFinding, error) {
    findings := []AuthnFinding{}

    if e.config.WeakPasswordTest {
        if f := e.testWeakPassword(task); f != nil {
            findings = append(findings, *f)
        }
    }
    if e.config.Bypass2FATest {
        // ❌ 修復: 移除虛假模擬，強制進行真實測試
        return nil, fmt.Errorf("❌ 2FA 繞過測試需要真實的網路協定測試\n" +
            "請實現遅接後的真實 OTP 繞過測試邏輯")
    }
    if e.config.SessionHijackTest {
        // ❌ 修復: 移除虛假模擬，強制進行真實測試
        return nil, fmt.Errorf("❌ 工作階段劫持測試需要真實的 Cookie 操作\n" +
            "請實現 HTTP 標頭和 Session 管理的真實測試")
    }
    return findings, nil
}

func (e *AuthnEngine) testWeakPassword(task AuthnTask) *AuthnFinding {
    // ❌ 修復: 移除虛假成功邏輯，強制進行真實登入測試
    if len(task.Username) == 0 {
        return &AuthnFinding{
            Name: "Configuration Error",
            Description: "Username is required for weak password testing",
            Severity: "Error",
        }
    }
    
    // 需要實現真實的 HTTP 請求測試，而非模擬結果
    // TODO: 實現真實的身份驗證測試邏輯
    return &AuthnFinding{
        Name: "Real Testing Required",
        Description: "Weak password testing requires actual HTTP authentication attempts",
        Severity: "Info",
        Evidence: "Implementation needed",
    }
}
