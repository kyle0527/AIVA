/**
 * AIVA Common TypeScript Schemas
 * 
 * 這個文件包含所有與 Python aiva_common.schemas 對應的 TypeScript 類型定義
 * 遵循單一事實原則，與 Python 版本保持完全一致
 * 
 * 命名規範: 統一使用 snake_case 以匹配 Python 版本
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
  AUTHENTICATION_BYPASS = "Authentication Bypass",
  // Business Logic Vulnerabilities
  PRICE_MANIPULATION = "Price Manipulation",
  WORKFLOW_BYPASS = "Workflow Bypass",
  RACE_CONDITION = "Race Condition",
  FORCED_BROWSING = "Forced Browsing",
  STATE_MANIPULATION = "State Manipulation"
}

export enum VulnerabilityStatus {
  NEW = "new",
  OPEN = "open",
  IN_PROGRESS = "in_progress",
  FIXED = "fixed",
  VERIFIED = "verified",
  RISK_ACCEPTED = "risk_accepted",
  FALSE_POSITIVE = "false_positive",
  WONT_FIX = "wont_fix",
  DUPLICATE = "duplicate"
}

export enum TaskStatus {
  PENDING = "pending",
  QUEUED = "queued",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled"
}

export enum ScanStatus {
  INITIALIZING = "initializing",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled"
}

// ==================== 基礎介面定義 ====================

/**
 * 經驗樣本 - 對應 Python ExperienceSample
 */
export interface ExperienceSample {
  sample_id: string;
  state: string;
  action: string;
  reward: number;
  next_state: string;
  timestamp: string;
  done?: boolean;
  metadata?: Record<string, any>;
}

/**
 * 能力資訊 - 對應 Python CapabilityInfo
 */
export interface CapabilityInfo {
  capability_id: string;
  name: string;
  description: string;
  category: string;
  supported_languages: string[];
  version?: string;
  status?: string;
  metadata?: Record<string, any>;
}

/**
 * 能力評分卡 - 對應 Python CapabilityScorecard
 */
export interface CapabilityScorecard {
  capability_id: string;
  name: string;
  overall_score: number;
  performance_score: number;
  reliability_score: number;
  security_score: number;
  usability_score: number;
  evidence_count: number;
  last_evaluated: string;
  evaluator_version?: string;
  metadata?: Record<string, any>;
}

/**
 * 漏洞評分卡 - 對應 Python VulnerabilityScorecard
 */
export interface VulnerabilityScorecard {
  vulnerability_id: string;
  name: string;
  severity: Severity;
  confidence: Confidence;
  cvss_score?: number;
  impact_score: number;
  exploitability_score: number;
  risk_level: string;
  business_impact?: string;
  remediation_effort?: string;
  recommended_actions: string[];
  created_at: string;
  updated_at?: string;
  evaluator_version?: string;
}

/**
 * 漏洞基本資訊 - 對應 Python Vulnerability
 */
export interface Vulnerability {
  name: VulnerabilityType;
  cwe?: string | null;           // 格式: CWE-XXX
  cve?: string | null;           // 格式: CVE-YYYY-NNNNN
  severity: Severity;
  confidence: Confidence;
  description?: string | null;
  cvss_score?: number | null;    // 0.0-10.0
  cvss_vector?: string | null;   // CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
  owasp_category?: string | null; // A03:2021-Injection
}

/**
 * 目標資訊 - 對應 Python Target
 */
export interface Target {
  url: any;                      // Accept arbitrary URL-like values
  parameter?: string | null;
  method?: string | null;
  headers?: Record<string, string>;
  params?: Record<string, any>;
  body?: string | null;
}

/**
 * 漏洞證據 - 對應 Python FindingEvidence
 */
export interface FindingEvidence {
  payload?: string | null;
  response_time_delta?: number | null;
  db_version?: string | null;
  request?: string | null;
  response?: string | null;
  proof?: string | null;
}

/**
 * 漏洞影響描述 - 對應 Python FindingImpact
 */
export interface FindingImpact {
  description?: string | null;
  business_impact?: string | null;
  technical_impact?: string | null;
  affected_users?: number | null;
  estimated_cost?: number | null;
}

/**
 * 漏洞修復建議 - 對應 Python FindingRecommendation
 */
export interface FindingRecommendation {
  fix?: string | null;
  priority?: string | null;
  remediation_steps?: string[];
  references?: string[];
}

/**
 * 漏洞發現 Payload - 對應 Python FindingPayload
 * 統一的漏洞報告格式
 */
export interface FindingPayload {
  finding_id: string;            // 必須以 "finding_" 開頭
  task_id: string;               // 必須以 "task_" 開頭  
  scan_id: string;               // 必須以 "scan_" 開頭
  status: string;                // "confirmed" | "potential" | "false_positive" | "needs_review"
  vulnerability: Vulnerability;
  target: Target;
  strategy?: string | null;
  evidence?: FindingEvidence | null;
  impact?: FindingImpact | null;
  recommendation?: FindingRecommendation | null;
  metadata?: Record<string, any>;
  created_at: string;            // ISO 8601 格式
  updated_at: string;            // ISO 8601 格式
}

// ==================== 擴展類型 ====================

/**
 * 敏感資訊匹配結果 - 對應 Python SensitiveMatch
 */
export interface SensitiveMatch {
  match_id: string;
  pattern_name: string;          // "password", "api_key", "credit_card", "private_key"
  matched_text: string;
  context: string;               // 前後文 (遮蔽敏感部分)
  confidence: number;            // 0.0 - 1.0
  line_number?: number | null;
  file_path?: string | null;
  url?: string | null;
  severity: Severity;
}

/**
 * JavaScript 分析結果 - 對應 Python JavaScriptAnalysisResult
 */
export interface JavaScriptAnalysisResult {
  analysis_id: string;
  url: string;
  source_size_bytes: number;
  
  // 詳細分析結果
  dangerous_functions?: string[];     // eval, Function, setTimeout等
  external_resources?: string[];      // 外部 URL
  data_leaks?: Record<string, string>[]; // 數據洩漏信息
  
  // 通用欄位 (保持兼容)
  findings?: string[];                // ["uses_eval", "dom_manipulation"]
  apis_called?: string[];             // 發現的 API 端點
  ajax_endpoints?: string[];          // AJAX 呼叫端點
  suspicious_patterns?: string[];
  
  // 評分欄位
  risk_score?: number;                // 0.0 - 10.0
  security_score?: number;            // 0-100 分
  
  timestamp: string;                  // ISO 8601 格式
}

/**
 * 漏洞關聯分析結果 - 對應 Python VulnerabilityCorrelation
 */
export interface VulnerabilityCorrelation {
  correlation_id: string;
  correlation_type: string;           // "code_level", "data_flow", "attack_chain"
  related_findings: string[];         // finding_ids
  confidence_score: number;           // 0.0 - 1.0
  root_cause?: string | null;
  common_components?: string[];
  explanation?: string | null;
  timestamp: string;                  // ISO 8601 格式
}

/**
 * AI 驅動漏洞驗證請求 - 對應 Python AIVerificationRequest
 */
export interface AIVerificationRequest {
  verification_id: string;
  finding_id: string;
  scan_id: string;
  vulnerability_type: VulnerabilityType;
  target: Target;
  evidence: FindingEvidence;
  verification_mode?: string;         // "non_destructive" | "safe" | "full"
  context?: Record<string, any>;
}

/**
 * AI 驅動漏洞驗證結果 - 對應 Python AIVerificationResult
 */
export interface AIVerificationResult {
  verification_id: string;
  finding_id: string;
  verification_status: string;        // "confirmed" | "false_positive" | "needs_review"
  confidence_score: number;           // 0.0 - 1.0
  verification_method: string;
  test_steps?: string[];
  observations?: string[];
  recommendations?: string[];
  timestamp: string;                  // ISO 8601 格式
}

// ==================== 工具函數 ====================

/**
 * 驗證 FindingPayload 的 ID 格式
 */
export function validateFindingId(finding_id: string): boolean {
  return finding_id.startsWith("finding_");
}

export function validateTaskId(task_id: string): boolean {
  return task_id.startsWith("task_");
}

export function validateScanId(scan_id: string): boolean {
  return scan_id.startsWith("scan_");
}

/**
 * 驗證 FindingPayload 狀態
 */
export function validateFindingStatus(status: string): boolean {
  const allowedStatuses = new Set(["confirmed", "potential", "false_positive", "needs_review"]);
  return allowedStatuses.has(status);
}

/**
 * 創建標準的 FindingPayload
 */
export function createFindingPayload(
  finding_id: string,
  task_id: string,
  scan_id: string,
  vulnerability: Vulnerability,
  target: Target,
  options: {
    status?: string;
    strategy?: string;
    evidence?: FindingEvidence;
    impact?: FindingImpact;
    recommendation?: FindingRecommendation;
    metadata?: Record<string, any>;
  } = {}
): FindingPayload {
  // 驗證 ID 格式
  if (!validateFindingId(finding_id)) {
    throw new Error("finding_id must start with 'finding_'");
  }
  if (!validateTaskId(task_id)) {
    throw new Error("task_id must start with 'task_'");
  }
  if (!validateScanId(scan_id)) {
    throw new Error("scan_id must start with 'scan_'");
  }

  const status = options.status || "confirmed";
  if (!validateFindingStatus(status)) {
    throw new Error(`Invalid status: ${status}. Must be one of: confirmed, potential, false_positive, needs_review`);
  }

  const now = new Date().toISOString();

  return {
    finding_id,
    task_id,
    scan_id,
    status,
    vulnerability,
    target,
    strategy: options.strategy || null,
    evidence: options.evidence || null,
    impact: options.impact || null,
    recommendation: options.recommendation || null,
    metadata: options.metadata || {},
    created_at: now,
    updated_at: now
  };
}

/**
 * 生成標準的 finding ID
 */
export function generateFindingId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substr(2, 9);
  return `finding_${timestamp}_${random}`;
}

/**
 * 生成標準的 task ID
 */
export function generateTaskId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substr(2, 9);
  return `task_${timestamp}_${random}`;
}

/**
 * 生成標準的 scan ID
 */
export function generateScanId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substr(2, 9);
  return `scan_${timestamp}_${random}`;
}

// ==================== 類型守衛 ====================

export function isFindingPayload(obj: any): obj is FindingPayload {
  return (
    obj &&
    typeof obj === 'object' &&
    typeof obj.finding_id === 'string' &&
    typeof obj.task_id === 'string' &&
    typeof obj.scan_id === 'string' &&
    typeof obj.status === 'string' &&
    obj.vulnerability &&
    obj.target &&
    validateFindingId(obj.finding_id) &&
    validateTaskId(obj.task_id) &&
    validateScanId(obj.scan_id) &&
    validateFindingStatus(obj.status)
  );
}

export function isVulnerability(obj: any): obj is Vulnerability {
  return (
    obj &&
    typeof obj === 'object' &&
    Object.values(VulnerabilityType).includes(obj.name) &&
    Object.values(Severity).includes(obj.severity) &&
    Object.values(Confidence).includes(obj.confidence)
  );
}

export function isTarget(obj: any): obj is Target {
  return (
    obj &&
    typeof obj === 'object' &&
    obj.url !== undefined
  );
}