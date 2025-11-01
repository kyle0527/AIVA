/**
 * AIVA Web UI 數據合約定義
 * 
 * 基於 aiva_common.schemas 標準化Web前端的數據結構
 * 確保前後端數據一致性
 */

// ==================== 基礎API響應格式 ====================

interface APIResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  timestamp?: string;
  trace_id?: string;
}

// ==================== 認證相關合約 ====================

interface LoginRequest {
  username: string;
  password: string;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: UserInfo;
  issued_at: string;
}

interface UserInfo {
  user_id: string;
  username: string;
  role: string;
  permissions: string[];
  last_login?: string;
}

// ==================== 掃描相關合約 ====================

interface ScanRequest {
  target_url: string;
  scan_type: ScanType;
  scan_scope: ScanScope;
  authentication?: Authentication;
  rate_limit?: RateLimit;
}

interface ScanScope {
  include_subdomains: boolean;
  max_depth: number;
  exclude_paths: string[];
  include_paths: string[];
}

interface Authentication {
  type: 'none' | 'basic' | 'bearer' | 'cookie';
  credentials?: Record<string, string>;
  headers?: Record<string, string>;
}

interface RateLimit {
  requests_per_second: number;
  concurrent_requests: number;
  delay_between_requests: number;
}

// ==================== 漏洞發現合約 ====================

interface Finding {
  finding_id: string;
  vulnerability: Vulnerability;
  target: FindingTarget;
  evidence: FindingEvidence;
  impact: FindingImpact;
  recommendation: FindingRecommendation;
  created_at: string;
  updated_at: string;
}

interface Vulnerability {
  name: string;
  description: string;
  severity: Severity;
  confidence: Confidence;
  vulnerability_type: VulnerabilityType;
  cve_id?: string;
  cwe_id?: string;
  cvss_metrics?: CVSSv3Metrics;
}

interface FindingTarget {
  url: string;
  method: string;
  parameters: Record<string, any>;
  headers: Record<string, string>;
  body?: string;
}

interface FindingEvidence {
  request: string;
  response: string;
  proof_of_concept: string;
  screenshots?: string[];
  additional_data: Record<string, any>;
}

interface FindingImpact {
  description: string;
  affected_users: string;
  business_impact: string;
  technical_impact: string;
  exploitability: string;
}

interface FindingRecommendation {
  short_term: string[];
  long_term: string[];
  references: string[];
}

// ==================== 資產相關合約 ====================

interface Asset {
  asset_id: string;
  url: string;
  asset_type: AssetType;
  status: AssetStatus;
  fingerprints: TechnicalFingerprint;
  last_scanned: string;
  risk_score: number;
}

interface TechnicalFingerprint {
  technologies: string[];
  frameworks: string[];
  server_info: Record<string, string>;
  security_headers: Record<string, string>;
}

// ==================== 掃描狀態和結果 ====================

interface ScanStatus {
  scan_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_phase: string;
  started_at: string;
  estimated_completion?: string;
  findings_count: number;
  assets_discovered: number;
}

interface ScanResult {
  scan_id: string;
  target_info: Asset;
  summary: ScanSummary;
  findings: Finding[];
  assets: Asset[];
  completed_at: string;
  scan_duration: number;
}

interface ScanSummary {
  total_findings: number;
  severity_breakdown: Record<Severity, number>;
  vulnerability_types: Record<VulnerabilityType, number>;
  risk_score: number;
  compliance_status: string;
}

// ==================== 儀表板相關合約 ====================

interface DashboardStats {
  total_scans: number;
  active_scans: number;
  total_findings: number;
  critical_findings: number;
  assets_monitored: number;
  last_scan_time: string;
  system_health: SystemHealth;
}

interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical';
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_connections: number;
  last_heartbeat: string;
}

// ==================== 報告相關合約 ====================

interface ReportRequest {
  scan_id: string;
  report_type: ReportType;
  format: ReportFormat;
  include_raw_data: boolean;
  filters?: ReportFilters;
}

interface ReportFilters {
  severity_levels: Severity[];
  vulnerability_types: VulnerabilityType[];
  date_range?: {
    start: string;
    end: string;
  };
}

// ==================== 枚舉類型 ====================

type Severity = 'critical' | 'high' | 'medium' | 'low' | 'info';
type Confidence = 'certain' | 'firm' | 'possible';
type VulnerabilityType = 'XSS' | 'SQL Injection' | 'SSRF' | 'IDOR' | 'BOLA' | 'Information Leak' | 'Weak Authentication' | 'RCE' | 'Authentication Bypass';
type ScanType = 'full' | 'quick' | 'targeted' | 'api' | 'mobile';
type AssetType = 'endpoint' | 'form' | 'api' | 'websocket' | 'file';
type AssetStatus = 'active' | 'inactive' | 'error';
type ReportType = 'executive' | 'technical' | 'compliance' | 'developer';
type ReportFormat = 'pdf' | 'html' | 'json' | 'xml' | 'csv';

// ==================== CVSS 相關 ====================

interface CVSSv3Metrics {
  base_score: number;
  base_severity: Severity;
  vector_string: string;
  attack_vector: string;
  attack_complexity: string;
  privileges_required: string;
  user_interaction: string;
  scope: string;
  confidentiality_impact: string;
  integrity_impact: string;
  availability_impact: string;
}

// ==================== 通知相關合約 ====================

interface NotificationPayload {
  notification_id: string;
  type: NotificationType;
  title: string;
  message: string;
  severity: Severity;
  timestamp: string;
  read: boolean;
  actions?: NotificationAction[];
}

interface NotificationAction {
  label: string;
  action: string;
  url?: string;
}

type NotificationType = 'scan_completed' | 'critical_finding' | 'system_alert' | 'update_available';

// ==================== UI狀態管理 ====================

interface UIState {
  currentUser?: UserInfo;
  activeScans: ScanStatus[];
  notifications: NotificationPayload[];
  systemHealth: SystemHealth;
  selectedScan?: string;
  isLoading: boolean;
  lastRefresh: string;
}

interface ViewConfig {
  theme: 'light' | 'dark';
  language: 'zh-tw' | 'en';
  autoRefresh: boolean;
  refreshInterval: number;
  showAdvancedOptions: boolean;
  defaultScanType: ScanType;
}

// ==================== 匯出所有合約 ====================

export {
  // API響應
  APIResponse,
  
  // 認證
  LoginRequest,
  TokenResponse,
  UserInfo,
  
  // 掃描
  ScanRequest,
  ScanScope,
  Authentication,
  RateLimit,
  ScanStatus,
  ScanResult,
  ScanSummary,
  
  // 漏洞
  Finding,
  Vulnerability,
  FindingTarget,
  FindingEvidence,
  FindingImpact,
  FindingRecommendation,
  CVSSv3Metrics,
  
  // 資產
  Asset,
  TechnicalFingerprint,
  
  // 儀表板
  DashboardStats,
  SystemHealth,
  
  // 報告
  ReportRequest,
  ReportFilters,
  
  // 通知
  NotificationPayload,
  NotificationAction,
  
  // UI狀態
  UIState,
  ViewConfig,
  
  // 類型
  Severity,
  Confidence,
  VulnerabilityType,
  ScanType,
  AssetType,
  AssetStatus,
  ReportType,
  ReportFormat,
  NotificationType
};