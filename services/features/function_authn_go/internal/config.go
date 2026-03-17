package internal

// AuthnConfig holds all configuration for authentication testing.
type AuthnConfig struct {
	WeakPasswordTest  bool
	CommonPasswords   []string
	Bypass2FATest     bool
	SessionHijackTest bool
	MaxLoginAttempts  int
	TargetURL         string // Login endpoint URL
	UsernameField     string // HTML form field name for username
	PasswordField     string // HTML form field name for password
	TimeoutSeconds    int    // HTTP request timeout
	RateLimitMs       int    // Delay between attempts in ms (rate limiting)
}

func DefaultConfig() AuthnConfig {
	return AuthnConfig{
		WeakPasswordTest: true,
		CommonPasswords: []string{
			"admin", "password", "123456", "qwerty", "admin123",
			"root", "toor", "test", "test123", "guest",
			"letmein", "welcome", "monkey", "dragon", "master",
			"login", "abc123", "passw0rd", "1234567890", "password1",
		},
		Bypass2FATest:     true,
		SessionHijackTest: true,
		MaxLoginAttempts:  5,
		UsernameField:     "username",
		PasswordField:     "password",
		TimeoutSeconds:    10,
		RateLimitMs:       500,
	}
}
