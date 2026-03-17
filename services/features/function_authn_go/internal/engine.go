package internal

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"strings"
	"time"
)

// AuthnTask represents a single authentication test task.
type AuthnTask struct {
	Username string            `json:"username"`
	Extra    map[string]string `json:"extra,omitempty"`
}

// AuthnFinding represents a single authentication vulnerability finding.
type AuthnFinding struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Severity    string `json:"severity"`
	Evidence    string `json:"evidence,omitempty"`
}

// AuthnEngine orchestrates all authentication tests.
type AuthnEngine struct {
	config AuthnConfig
	client *http.Client
}

// NewAuthnEngine creates a new engine with the given config.
func NewAuthnEngine(cfg AuthnConfig) *AuthnEngine {
	jar, _ := cookiejar.New(nil)
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: false},
	}
	client := &http.Client{
		Timeout: time.Duration(cfg.TimeoutSeconds) * time.Second,
		Jar:     jar,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}
	return &AuthnEngine{config: cfg, client: client}
}

// RunTests executes all enabled authentication tests and returns findings.
func (e *AuthnEngine) RunTests(task AuthnTask) ([]AuthnFinding, error) {
	findings := []AuthnFinding{}

	targetURL := e.config.TargetURL
	if u, ok := task.Extra["target_url"]; ok && u != "" {
		targetURL = u
	}
	if targetURL == "" {
		return nil, fmt.Errorf("target_url is required (set in config or task.Extra)")
	}

	if e.config.WeakPasswordTest {
		weakFindings := e.testWeakPassword(task, targetURL)
		findings = append(findings, weakFindings...)
	}

	if e.config.SessionHijackTest {
		sessionFindings := e.testSessionSecurity(targetURL)
		findings = append(findings, sessionFindings...)
	}

	if e.config.Bypass2FATest {
		twoFAFindings := e.test2FABypass(task, targetURL)
		findings = append(findings, twoFAFindings...)
	}

	return findings, nil
}

// testWeakPassword performs real HTTP login attempts with common passwords.
func (e *AuthnEngine) testWeakPassword(task AuthnTask, targetURL string) []AuthnFinding {
	var findings []AuthnFinding

	if task.Username == "" {
		findings = append(findings, AuthnFinding{
			Name:        "Configuration Error",
			Description: "Username is required for weak password testing",
			Severity:    "Error",
		})
		return findings
	}

	usernameField := e.config.UsernameField
	passwordField := e.config.PasswordField
	if f, ok := task.Extra["username_field"]; ok && f != "" {
		usernameField = f
	}
	if f, ok := task.Extra["password_field"]; ok && f != "" {
		passwordField = f
	}

	// Get baseline response for failed login (use a random password)
	baselineBody, baselineStatus := e.doLogin(targetURL, usernameField, passwordField, task.Username, "___baseline_invalid_pwd_12345___")

	attempts := 0
	for _, password := range e.config.CommonPasswords {
		if attempts >= e.config.MaxLoginAttempts {
			findings = append(findings, AuthnFinding{
				Name:        "Rate Limit Reached",
				Description: fmt.Sprintf("Stopped after %d attempts to avoid account lockout", e.config.MaxLoginAttempts),
				Severity:    "Info",
				Evidence:    fmt.Sprintf("Tested %d passwords", attempts),
			})
			break
		}

		body, status := e.doLogin(targetURL, usernameField, passwordField, task.Username, password)
		attempts++

		// Detect successful login: status differs, or response body significantly changes
		if e.isLoginSuccess(baselineBody, baselineStatus, body, status) {
			findings = append(findings, AuthnFinding{
				Name:        "Weak Password Detected",
				Description: fmt.Sprintf("Account '%s' uses weak password '%s'", task.Username, password),
				Severity:    "Critical",
				Evidence:    fmt.Sprintf("HTTP %d response indicated successful authentication", status),
			})
			break // Stop after first successful credential
		}

		// Check for account lockout indicators
		if e.isAccountLocked(body) {
			findings = append(findings, AuthnFinding{
				Name:        "Account Lockout Detected",
				Description: fmt.Sprintf("Account '%s' was locked after %d attempts", task.Username, attempts),
				Severity:    "Info",
				Evidence:    "Lockout indicator detected in response",
			})
			break
		}

		// Rate limiting delay
		if e.config.RateLimitMs > 0 {
			time.Sleep(time.Duration(e.config.RateLimitMs) * time.Millisecond)
		}
	}

	if len(findings) == 0 {
		findings = append(findings, AuthnFinding{
			Name:        "No Weak Passwords Found",
			Description: fmt.Sprintf("Tested %d common passwords against user '%s'", attempts, task.Username),
			Severity:    "Info",
			Evidence:    "All tested passwords were rejected",
		})
	}

	return findings
}

// testSessionSecurity analyzes session cookie security attributes.
func (e *AuthnEngine) testSessionSecurity(targetURL string) []AuthnFinding {
	var findings []AuthnFinding

	resp, err := e.client.Get(targetURL)
	if err != nil {
		findings = append(findings, AuthnFinding{
			Name:        "Session Test Connection Error",
			Description: fmt.Sprintf("Could not connect to target: %v", err),
			Severity:    "Info",
		})
		return findings
	}
	defer resp.Body.Close()

	cookies := resp.Cookies()
	if len(cookies) == 0 {
		findings = append(findings, AuthnFinding{
			Name:        "No Session Cookies",
			Description: "No cookies were set by the target, session testing skipped",
			Severity:    "Info",
		})
		return findings
	}

	isHTTPS := strings.HasPrefix(targetURL, "https")

	for _, cookie := range cookies {
		cookieName := strings.ToLower(cookie.Name)
		isSessionCookie := strings.Contains(cookieName, "session") ||
			strings.Contains(cookieName, "sid") ||
			strings.Contains(cookieName, "token") ||
			strings.Contains(cookieName, "auth") ||
			strings.Contains(cookieName, "jwt") ||
			strings.Contains(cookieName, "phpsessid") ||
			strings.Contains(cookieName, "jsessionid") ||
			strings.Contains(cookieName, "asp.net_sessionid")

		if !isSessionCookie {
			continue
		}

		// Check HttpOnly flag
		if !cookie.HttpOnly {
			findings = append(findings, AuthnFinding{
				Name:        "Missing HttpOnly Flag",
				Description: fmt.Sprintf("Session cookie '%s' lacks HttpOnly flag, vulnerable to XSS cookie theft", cookie.Name),
				Severity:    "Medium",
				Evidence:    fmt.Sprintf("Set-Cookie: %s (HttpOnly=false)", cookie.Name),
			})
		}

		// Check Secure flag (only relevant for HTTPS)
		if isHTTPS && !cookie.Secure {
			findings = append(findings, AuthnFinding{
				Name:        "Missing Secure Flag",
				Description: fmt.Sprintf("Session cookie '%s' lacks Secure flag on HTTPS site, may be transmitted over HTTP", cookie.Name),
				Severity:    "Medium",
				Evidence:    fmt.Sprintf("Set-Cookie: %s (Secure=false)", cookie.Name),
			})
		}

		// Check SameSite attribute
		if cookie.SameSite == http.SameSiteDefaultMode || cookie.SameSite == 0 {
			findings = append(findings, AuthnFinding{
				Name:        "Missing SameSite Attribute",
				Description: fmt.Sprintf("Session cookie '%s' lacks SameSite attribute, potentially vulnerable to CSRF", cookie.Name),
				Severity:    "Low",
				Evidence:    fmt.Sprintf("Set-Cookie: %s (SameSite not set)", cookie.Name),
			})
		}

		// Check session ID entropy (length check)
		if len(cookie.Value) < 16 {
			findings = append(findings, AuthnFinding{
				Name:        "Short Session ID",
				Description: fmt.Sprintf("Session cookie '%s' has a short value (%d chars), may be predictable", cookie.Name, len(cookie.Value)),
				Severity:    "High",
				Evidence:    fmt.Sprintf("Session ID length: %d characters", len(cookie.Value)),
			})
		}
	}

	return findings
}

// test2FABypass tests common 2FA bypass techniques.
func (e *AuthnEngine) test2FABypass(task AuthnTask, targetURL string) []AuthnFinding {
	var findings []AuthnFinding

	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		return findings
	}

	// Test 1: Direct access to post-auth pages (skip 2FA step)
	postAuthPaths := []string{"/dashboard", "/home", "/account", "/profile", "/admin", "/api/user"}
	baseURL := fmt.Sprintf("%s://%s", parsedURL.Scheme, parsedURL.Host)

	for _, path := range postAuthPaths {
		testURL := baseURL + path
		resp, err := e.client.Get(testURL)
		if err != nil {
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		// If we get 200 with content that looks like an authenticated page
		if resp.StatusCode == 200 && !e.looksLikeLoginPage(string(body)) {
			findings = append(findings, AuthnFinding{
				Name:        "Potential 2FA Bypass via Direct Access",
				Description: fmt.Sprintf("Post-authentication page %s accessible without completing 2FA flow", path),
				Severity:    "High",
				Evidence:    fmt.Sprintf("HTTP %d on %s with authenticated content", resp.StatusCode, testURL),
			})
		}
	}

	// Test 2: 2FA code manipulation - test null/empty/zero codes
	twoFAPaths := []string{"/verify", "/2fa", "/otp", "/mfa", "/verify-otp", "/two-factor"}
	testCodes := []string{"", "000000", "null", "0", "999999"}

	for _, path := range twoFAPaths {
		testURL := baseURL + path
		for _, code := range testCodes {
			formData := url.Values{}
			formData.Set("code", code)
			formData.Set("otp", code)
			formData.Set("token", code)

			resp, err := e.client.PostForm(testURL, formData)
			if err != nil {
				continue
			}
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			// Check if any null/zero code was accepted
			if resp.StatusCode == 200 || resp.StatusCode == 302 {
				bodyStr := string(body)
				if !strings.Contains(strings.ToLower(bodyStr), "invalid") &&
					!strings.Contains(strings.ToLower(bodyStr), "error") &&
					!strings.Contains(strings.ToLower(bodyStr), "incorrect") &&
					!e.looksLikeLoginPage(bodyStr) {
					findings = append(findings, AuthnFinding{
						Name:        "Potential 2FA Code Bypass",
						Description: fmt.Sprintf("2FA endpoint %s accepted code '%s'", path, code),
						Severity:    "Critical",
						Evidence:    fmt.Sprintf("HTTP %d on POST %s with code='%s'", resp.StatusCode, testURL, code),
					})
					break // One finding per path is enough
				}
			}
		}
	}

	// Test 3: Check if 2FA can be skipped by manipulating response
	// Test JSON API endpoints that might expose 2FA status
	apiPaths := []string{"/api/auth/status", "/api/user/me", "/api/session"}
	for _, path := range apiPaths {
		testURL := baseURL + path
		resp, err := e.client.Get(testURL)
		if err != nil {
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 200 {
			var jsonResp map[string]interface{}
			if json.Unmarshal(body, &jsonResp) == nil {
				// Check for 2FA related fields
				for key := range jsonResp {
					keyLower := strings.ToLower(key)
					if strings.Contains(keyLower, "2fa") || strings.Contains(keyLower, "mfa") ||
						strings.Contains(keyLower, "two_factor") || strings.Contains(keyLower, "otp") {
						findings = append(findings, AuthnFinding{
							Name:        "2FA Status Exposed via API",
							Description: fmt.Sprintf("API endpoint %s exposes 2FA configuration (field: %s)", path, key),
							Severity:    "Medium",
							Evidence:    fmt.Sprintf("JSON field '%s' found in %s response", key, testURL),
						})
					}
				}
			}
		}
	}

	if len(findings) == 0 {
		findings = append(findings, AuthnFinding{
			Name:        "2FA Bypass Test Completed",
			Description: "No obvious 2FA bypass vulnerabilities detected via common techniques",
			Severity:    "Info",
			Evidence:    fmt.Sprintf("Tested %d post-auth paths and %d 2FA endpoints", len(postAuthPaths), len(twoFAPaths)),
		})
	}

	return findings
}

// doLogin performs a single login attempt and returns response body and status code.
func (e *AuthnEngine) doLogin(targetURL, usernameField, passwordField, username, password string) (string, int) {
	formData := url.Values{}
	formData.Set(usernameField, username)
	formData.Set(passwordField, password)

	resp, err := e.client.PostForm(targetURL, formData)
	if err != nil {
		return "", 0
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", resp.StatusCode
	}

	return string(body), resp.StatusCode
}

// isLoginSuccess compares login response against baseline to detect successful authentication.
func (e *AuthnEngine) isLoginSuccess(baselineBody string, baselineStatus int, body string, status int) bool {
	// Different status code (e.g., 302 redirect on success vs 200 on failure)
	if status != baselineStatus && (status == 302 || status == 301 || status == 303) {
		return true
	}

	// Response body significantly different from baseline (different page = likely success)
	if baselineBody != "" && body != "" {
		baseLen := len(baselineBody)
		bodyLen := len(body)
		if baseLen > 0 {
			diff := float64(bodyLen-baseLen) / float64(baseLen)
			if diff > 0.3 || diff < -0.3 {
				// Additionally check for success indicators
				bodyLower := strings.ToLower(body)
				successIndicators := []string{"dashboard", "welcome", "logout", "sign out", "my account", "profile"}
				for _, indicator := range successIndicators {
					if strings.Contains(bodyLower, indicator) {
						return true
					}
				}
			}
		}
	}

	return false
}

// isAccountLocked checks if response indicates account lockout.
func (e *AuthnEngine) isAccountLocked(body string) bool {
	if body == "" {
		return false
	}
	bodyLower := strings.ToLower(body)
	lockoutIndicators := []string{
		"account locked", "too many attempts", "temporarily blocked",
		"account disabled", "try again later", "rate limit",
		"maximum attempts", "account suspended",
	}
	for _, indicator := range lockoutIndicators {
		if strings.Contains(bodyLower, indicator) {
			return true
		}
	}
	return false
}

// looksLikeLoginPage checks if the HTML body looks like a login/auth page.
func (e *AuthnEngine) looksLikeLoginPage(body string) bool {
	bodyLower := strings.ToLower(body)
	loginIndicators := []string{
		"<input", "password", "login", "sign in", "log in", "authenticate",
	}
	matchCount := 0
	for _, indicator := range loginIndicators {
		if strings.Contains(bodyLower, indicator) {
			matchCount++
		}
	}
	return matchCount >= 2
}
