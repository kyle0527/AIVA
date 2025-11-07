package internal

type AuthnConfig struct {
    WeakPasswordTest  bool
    CommonPasswords   []string
    Bypass2FATest     bool
    SessionHijackTest bool
    MaxLoginAttempts  int
}

func DefaultConfig() AuthnConfig {
    return AuthnConfig{
        WeakPasswordTest:  true,
        CommonPasswords:   []string{"admin", "password", "123456", "qwerty"},
        Bypass2FATest:     true,
        SessionHijackTest: true,
        MaxLoginAttempts:  5,
    }
}
