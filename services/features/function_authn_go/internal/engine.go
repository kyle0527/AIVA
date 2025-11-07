package internal

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

func (e *AuthnEngine) RunTests(task AuthnTask) []AuthnFinding {
    findings := []AuthnFinding{}

    if e.config.WeakPasswordTest {
        if f := e.testWeakPassword(task); f != nil {
            findings = append(findings, *f)
        }
    }
    if e.config.Bypass2FATest {
        // 示意：真實情境應對具體目標執行協議層測試
        findings = append(findings, AuthnFinding{
            Name: "2FA Bypass (Simulated)",
            Description: "Second factor authentication could be bypassed in simulation",
            Severity: "High",
        })
    }
    if e.config.SessionHijackTest {
        findings = append(findings, AuthnFinding{
            Name: "Session Hijack (Simulated)",
            Description: "Session fixation/hijack scenario simulated",
            Severity: "Medium",
        })
    }
    return findings
}

func (e *AuthnEngine) testWeakPassword(task AuthnTask) *AuthnFinding {
    for _, pwd := range e.config.CommonPasswords {
        // 模擬：假設使用弱密碼可登入成功
        if len(task.Username) > 0 && (pwd == "admin" || pwd == "password") {
            return &AuthnFinding{
                Name: "Weak Password Login",
                Description: "User can log in with a weak/default password",
                Severity: "Medium",
                Evidence: pwd,
            }
        }
    }
    return nil
}
