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
  name: VulnerabilityType;
  /** CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/ */
  cwe?: string | null;
  /** CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/ */
  cve?: string | null;
  severity: Severity;
  confidence: Confidence;
  description?: string | null;
  /** CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/ */
  cvss_score?: number | null;
  /** CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H */
  cvss_vector?: string | null;
  /** OWASP Top 10 分類，例如: A03:2021-Injection */
  owasp_category?: string | null;
}

// ==================== 漏洞發現類型 ====================

/**
 * 漏洞證據 - 用於儲存漏洞驗證的具體證據
 */
export interface FindingEvidence {
  /** 攻擊載荷 */
  payload?: string | null;
  /** 響應時間差異（毫秒） */
  response_time_delta?: number | null;
  /** 資料庫版本資訊 */
  db_version?: string | null;
  /** 完整的 HTTP 請求 */
  request?: string | null;
  /** 完整的 HTTP 響應 */
  response?: string | null;
  /** 漏洞存在的具體證明 */
  proof?: string | null;
}

/**
 * 漏洞影響描述 - 描述漏洞對系統和業務的影響
 */
export interface FindingImpact {
  /** 影響描述 */
  description?: string | null;
  /** 業務影響評估 */
  business_impact?: string | null;
  /** 技術影響評估 */
  technical_impact?: string | null;
  /** 受影響用戶數量 */
  affected_users?: number | null;
  /** 預估損失成本 */
  estimated_cost?: number | null;
}

/**
 * 漏洞修復建議 - 提供具體的修復指導
 */
export interface FindingRecommendation {
  /** 修復方案 */
  fix?: string | null;
  /** 修復優先級 */
  priority?: string | null;
  /** 修復步驟 */
  remediation_steps?: string[];
  /** 參考資料 */
  references?: string[];
}

/**
 * 漏洞發現 Payload - 統一的漏洞報告格式
 */
export interface FindingPayload {
  /** 發現唯一識別碼 - 必須以 "finding_" 開頭 */
  finding_id: string;
  /** 任務唯一識別碼 - 必須以 "task_" 開頭 */
  task_id: string;
  /** 掃描唯一識別碼 - 必須以 "scan_" 開頭 */
  scan_id: string;
  /** 發現狀態: "confirmed" | "potential" | "false_positive" | "needs_review" */
  status: string;
  /** 漏洞資訊 */
  vulnerability: Vulnerability;
  /** 目標資訊 */
  target: Target;
  /** 測試策略 */
  strategy?: string | null;
  /** 漏洞證據 */
  evidence?: FindingEvidence | null;
  /** 影響評估 */
  impact?: FindingImpact | null;
  /** 修復建議 */
  recommendation?: FindingRecommendation | null;
  /** 額外元數據 */
  metadata?: Record<string, any>;
  /** 創建時間 - ISO 8601 格式 */
  created_at: string;
  /** 更新時間 - ISO 8601 格式 */
  updated_at: string;
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