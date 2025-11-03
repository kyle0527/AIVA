/**
 * AIVA Common TypeScript Schemas - 自動生成
 * ==========================================
 * 
 * AIVA跨語言Schema統一定義 - 以手動維護版本為準
 * 
 * ⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
 * 📅 最後更新: 2025-10-30T00:00:00.000000
 * 🔄 Schema 版本: 1.1.0
 * 
 * 遵循單一事實原則，與 Python aiva_common.schemas 保持完全一致
 */

// ==================== 枚舉類型定義 ====================

export enum Severity {
  CRITICAL = "critical",
  HIGH = "high",
  MEDIUM = "medium",
  LOW = "low",
  INFORMATIONAL = "info"
}

export enum Confidence {
  CERTAIN = "certain",
  FIRM = "firm",
  POSSIBLE = "possible"
}

export enum VulnerabilityType {
  XSS = "XSS",
  SQLI = "SQL Injection",
  SSRF = "SSRF",
  IDOR = "IDOR",
  BOLA = "BOLA",
  INFO_LEAK = "Information Leak",
  WEAK_AUTH = "Weak Authentication",
  RCE = "Remote Code Execution",
  AUTHENTICATION_BYPASS = "Authentication Bypass"
}

// ==================== 基礎介面定義 ====================

/**
 * 訊息標頭 - 用於所有訊息的統一標頭格式
 */
export interface MessageHeader {
  message_id: string;
  trace_id: string;
  correlation_id?: string | null;
  /** 來源模組名稱 */
  source_module: string;
  timestamp?: string;
  version?: string;
}

/**
 * 目標資訊 - 漏洞所在位置
 */
export interface Target {
  url: any;
  parameter?: string | null;
  method?: string | null;
  headers?: Record<string, any>;
  params?: Record<string, any>;
  body?: string | null;
}

/**
 * 漏洞基本資訊 - 用於 Finding 中的漏洞描述。符合標準：CWE、CVE、CVSS v3.1/v4.0、OWASP
 */
export interface Vulnerability {
  name: any;
  /** CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/ */
  cwe?: string | null;
  /** CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/ */
  cve?: string | null;
  severity: any;
  confidence: any;
  description?: string | null;
  /** CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/ */
  cvss_score?: any;
  /** CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H */
  cvss_vector?: string | null;
  /** OWASP Top 10 分類，例如: A03:2021-Injection */
  owasp_category?: string | null;
}

/**
 * 資產基本資訊
 */
export interface Asset {
  asset_id: string;
  type: string;
  value: string;
  parameters?: string[];
  has_form?: boolean;
}

/**
 * 認證資訊
 */
export interface Authentication {
  method?: string;
  credentials?: Record<string, any>;
}

/**
 * 執行錯誤統一格式
 */
export interface ExecutionError {
  error_id: string;
  error_type: string;
  message: string;
  payload?: string | null;
  vector?: string | null;
  timestamp?: string;
  attempts?: number;
}

/**
 * 技術指紋
 */
export interface Fingerprints {
  web_server?: Record<string, any>;
  framework?: Record<string, any>;
  language?: Record<string, any>;
  waf_detected?: boolean;
  waf_vendor?: string | null;
}

/**
 * 速率限制
 */
export interface RateLimit {
  requests_per_second?: number;
  burst?: number;
}

/**
 * 風險因子
 */
export interface RiskFactor {
  /** 風險因子名稱 */
  factor_name: string;
  /** 權重 */
  weight: number;
  /** 因子值 */
  value: number;
  /** 因子描述 */
  description?: string | null;
}

/**
 * 掃描範圍
 */
export interface ScanScope {
  exclusions?: string[];
  include_subdomains?: boolean;
  allowed_hosts?: string[];
}

/**
 * 掃描摘要
 */
export interface Summary {
  urls_found?: number;
  forms_found?: number;
  apis_found?: number;
  scan_duration_seconds?: number;
}

/**
 * 任務依賴
 */
export interface TaskDependency {
  /** 依賴類型 */
  dependency_type: string;
  /** 依賴任務ID */
  dependent_task_id: string;
  /** 依賴條件 */
  condition?: string | null;
  /** 是否必需 */
  required?: boolean;
}

/**
 * AI 驅動漏洞驗證請求
 */
export interface AIVerificationRequest {
  verification_id: string;
  finding_id: string;
  scan_id: string;
  vulnerability_type: any;
  target: any;
  evidence: any;
  verification_mode?: string;
  context?: Record<string, any>;
}

/**
 * AI 驅動漏洞驗證結果
 */
export interface AIVerificationResult {
  verification_id: string;
  finding_id: string;
  verification_status: string;
  confidence_score: number;
  verification_method: string;
  test_steps?: string[];
  observations?: string[];
  recommendations?: string[];
  timestamp?: string;
}

/**
 * 程式碼層面根因分析結果
 */
export interface CodeLevelRootCause {
  analysis_id: string;
  vulnerable_component: string;
  affected_findings: string[];
  code_location?: string | null;
  vulnerability_pattern?: string | null;
  fix_recommendation?: string | null;
}

/**
 * 目標資訊 - 漏洞所在位置
 */
export interface FindingTarget {
  url: any;
  parameter?: string | null;
  method?: string | null;
  headers?: Record<string, any>;
  params?: Record<string, any>;
  body?: string | null;
}

/**
 * JavaScript 分析結果
 */
export interface JavaScriptAnalysisResult {
  analysis_id: string;
  url: string;
  source_size_bytes: number;
  dangerous_functions?: string[];
  external_resources?: string[];
  data_leaks?: Record<string, any>;
  findings?: string[];
  apis_called?: string[];
  ajax_endpoints?: string[];
  suspicious_patterns?: string[];
  risk_score?: number;
  security_score?: number;
  timestamp?: string;
}

/**
 * SAST-DAST 資料流關聯結果
 */
export interface SASTDASTCorrelation {
  correlation_id: string;
  sast_finding_id: string;
  dast_finding_id: string;
  data_flow_path: string[];
  verification_status: string;
  confidence_score: number;
  explanation?: string | null;
}

/**
 * 敏感資訊匹配結果
 */
export interface SensitiveMatch {
  match_id: string;
  pattern_name: string;
  matched_text: string;
  context: string;
  confidence: number;
  line_number?: any;
  file_path?: string | null;
  url?: string | null;
  severity?: any;
}

/**
 * 漏洞關聯分析結果
 */
export interface VulnerabilityCorrelation {
  correlation_id: string;
  correlation_type: string;
  related_findings: string[];
  confidence_score: number;
  root_cause?: string | null;
  common_components?: string[];
  explanation?: string | null;
  timestamp?: string;
}

// ==================== 訊息通訊類型 ====================

/**
 * AIVA統一訊息格式 - 所有跨服務通訊的標準信封
 */
export interface AivaMessage {
  /** 訊息標頭 */
  header: MessageHeader;
  /** 訊息主題 */
  topic: string;
  /** Schema版本 */
  schema_version: string;
  /** 訊息載荷 */
  payload: Record<string, any>;
}

/**
 * 統一請求格式 - 模組間請求通訊
 */
export interface AIVARequest {
  /** 請求識別碼 */
  request_id: string;
  /** 來源模組 */
  source_module: string;
  /** 目標模組 */
  target_module: string;
  /** 請求類型 */
  request_type: string;
  /** 請求載荷 */
  payload: Record<string, any>;
  /** 追蹤識別碼 */
  trace_id?: string | null;
  /** 逾時秒數 */
  timeout_seconds: number;
  /** 中繼資料 */
  metadata?: Record<string, any>;
  /** 時間戳 */
  timestamp: string;
}

/**
 * 統一響應格式 - 模組間響應通訊
 */
export interface AIVAResponse {
  /** 對應的請求識別碼 */
  request_id: string;
  /** 響應類型 */
  response_type: string;
  /** 執行是否成功 */
  success: boolean;
  /** 響應載荷 */
  payload?: Record<string, any> | null;
  /** 錯誤代碼 */
  error_code?: string | null;
  /** 錯誤訊息 */
  error_message?: string | null;
  /** 中繼資料 */
  metadata?: Record<string, any>;
  /** 時間戳 */
  timestamp: string;
}

// ==================== 漏洞發現類型 ====================

/**
 * 漏洞發現載荷 - 掃描結果的標準格式
 */
export interface FindingPayload {
  /** 發現識別碼 */
  finding_id: string;
  /** 任務識別碼 */
  task_id: string;
  /** 掃描識別碼 */
  scan_id: string;
  /** 發現狀態 */
  status: string;
  /** 漏洞資訊 */
  vulnerability: Vulnerability;
  /** 目標資訊 */
  target: Target;
  /** 使用的策略 */
  strategy?: string | null;
  /** 證據資料 */
  evidence?: FindingEvidence | null;
  /** 影響評估 */
  impact?: FindingImpact | null;
  /** 修復建議 */
  recommendation?: FindingRecommendation | null;
  /** 中繼資料 */
  metadata?: Record<string, any>;
  /** 建立時間 */
  created_at: string;
  /** 更新時間 */
  updated_at: string;
}

/**
 * 漏洞證據
 */
export interface FindingEvidence {
  /** 攻擊載荷 */
  payload?: string | null;
  /** 響應時間差異 */
  response_time_delta?: number | null;
  /** 資料庫版本 */
  db_version?: string | null;
  /** HTTP請求 */
  request?: string | null;
  /** HTTP響應 */
  response?: string | null;
  /** 證明資料 */
  proof?: string | null;
}

/**
 * 漏洞影響評估
 */
export interface FindingImpact {
  /** 影響描述 */
  description?: string | null;
  /** 業務影響 */
  business_impact?: string | null;
  /** 技術影響 */
  technical_impact?: string | null;
  /** 受影響用戶數 */
  affected_users?: number | null;
  /** 估計成本 */
  estimated_cost?: number | null;
}

/**
 * 漏洞修復建議
 */
export interface FindingRecommendation {
  /** 修復方法 */
  fix?: string | null;
  /** 修復優先級 */
  priority?: string | null;
  /** 修復步驟 */
  remediation_steps?: string[];
  /** 參考資料 */
  references?: string[];
}

/**
 * Token 測試結果
 */
export interface TokenTestResult {
  /** 是否存在漏洞 */
  vulnerable: boolean;
  /** Token 類型 (jwt, session, api, etc.) */
  token_type: string;
  /** 發現的問題 */
  issue: string;
  /** 詳細描述 */
  details: string;
  /** 解碼後的載荷內容 */
  decoded_payload?: Record<string, any> | null;
  /** 漏洞嚴重程度 */
  severity?: string;
  /** 測試類型 */
  test_type: string;
}

// ==================== 任務管理類型 ====================

/**
 * 功能任務載荷 - 掃描任務的標準格式
 */
export interface FunctionTaskPayload {
  /** 任務識別碼 */
  task_id: string;
  /** 掃描識別碼 */
  scan_id: string;
  /** 任務優先級 */
  priority: number;
  /** 掃描目標 */
  target: FunctionTaskTarget;
  /** 任務上下文 */
  context: FunctionTaskContext;
  /** 掃描策略 */
  strategy: string;
  /** 自訂載荷 */
  custom_payloads?: string[];
  /** 測試配置 */
  test_config: FunctionTaskTestConfig;
}

/**
 * 功能任務目標
 */
export interface FunctionTaskTarget {
}

/**
 * 功能任務上下文
 */
export interface FunctionTaskContext {
  /** 資料庫類型提示 */
  db_type_hint?: string | null;
  /** 是否檢測到WAF */
  waf_detected: boolean;
  /** 相關發現 */
  related_findings?: string[];
}

/**
 * 功能任務測試配置
 */
export interface FunctionTaskTestConfig {
  /** 標準載荷列表 */
  payloads: string[];
  /** 自訂載荷列表 */
  custom_payloads?: string[];
  /** 是否進行Blind XSS測試 */
  blind_xss: boolean;
  /** 是否進行DOM測試 */
  dom_testing: boolean;
  /** 請求逾時(秒) */
  timeout?: number | null;
}

/**
 * 掃描任務載荷 - 用於SCA/SAST等需要項目URL的掃描任務
 */
export interface ScanTaskPayload {
  /** 任務識別碼 */
  task_id: string;
  /** 掃描識別碼 */
  scan_id: string;
  /** 任務優先級 */
  priority: number;
  /** 掃描目標 (包含URL) */
  target: Target;
  /** 掃描類型 */
  scan_type: string;
  /** 代碼倉庫資訊 (分支、commit等) */
  repository_info?: Record<string, any> | null;
  /** 掃描逾時(秒) */
  timeout?: number | null;
}
