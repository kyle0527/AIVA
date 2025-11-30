// AUTO-GENERATED from JSON Schema by AIVA Official Tools
// Generated at: 2025-10-18T16:13:22.880488
// Do not edit manually - changes will be overwritten

export interface AIExperienceCreatedEvent {
  experience_id: string;
  training_id?: string | null;
  trace_id: string;
  vulnerability_type: string;
  quality_score: number;
  success: boolean;
  plan_summary?: {};
  result_summary?: {};
  metadata?: {};
  timestamp?: string;
}

export interface AILanguageModel {
  /** 模型名稱 */
  model_name: string;
  /** 支援的語言 */
  supported_languages: ProgrammingLanguage[];
  /** 模型類型 */
  model_type: string;
  /** 模型版本 */
  version: string;
  /** 能力列表 */
  capabilities: string[];
  /** 訓練資料大小 */
  training_data_size?: number | null;
  /** 精確度指標 */
  accuracy_metrics?: {};
  /** API 端點 */
  api_endpoint?: string | null;
  /** 是否需要認證 */
  authentication_required?: boolean;
}

export interface AIModelDeployCommand {
  model_id: string;
  model_version: string;
  checkpoint_path: string;
  deployment_target?: string;
  deployment_config?: {};
  require_validation?: boolean;
  min_performance_threshold?: {};
  metadata?: {};
}

export interface AIModelUpdatedEvent {
  model_id: string;
  model_version: string;
  training_id?: string | null;
  update_type: string;
  performance_metrics?: {};
  model_path?: string | null;
  checkpoint_path?: string | null;
  is_deployed?: boolean;
  metadata?: {};
  timestamp?: string;
}

export interface AITraceCompletedEvent {
  trace_id: string;
  session_id?: string | null;
  training_id?: string | null;
  total_steps: number;
  successful_steps: number;
  failed_steps: number;
  duration_seconds: number;
  final_success: boolean;
  plan_type: string;
  metadata?: {};
  timestamp?: string;
}

export interface AITrainingCompletedPayload {
  training_id: string;
  status: string;
  total_episodes: number;
  successful_episodes: number;
  failed_episodes: number;
  total_duration_seconds: number;
  total_samples: number;
  high_quality_samples: number;
  medium_quality_samples: number;
  low_quality_samples: number;
  final_avg_reward?: number | null;
  final_avg_quality?: number | null;
  best_episode_reward?: number | null;
  model_checkpoint_path?: string | null;
  model_metrics?: {};
  error_message?: string | null;
  metadata?: {};
  completed_at?: string;
}

export interface AITrainingProgressPayload {
  training_id: string;
  episode_number: number;
  total_episodes: number;
  successful_episodes?: number;
  failed_episodes?: number;
  total_samples?: number;
  high_quality_samples?: number;
  avg_reward?: number | null;
  avg_quality?: number | null;
  best_reward?: number | null;
  model_metrics?: {};
  status?: string;
  metadata?: {};
  timestamp?: string;
}

export interface AITrainingStartPayload {
  training_id: string;
  training_type: string;
  scenario_id?: string | null;
  target_vulnerability?: string | null;
  config: ModelTrainingConfig;
  metadata?: {};
}

export interface AIVACommand {
  command_id: string;
  command_type: string;
  source_module: string;
  target_module: string;
  payload: {};
  priority?: number;
  trace_id?: string | null;
  metadata?: {};
  timestamp: string;
}

export interface AIVAEvent {
  event_id: string;
  event_type: string;
  source_module: string;
  payload: {};
  trace_id?: string | null;
  metadata?: {};
  timestamp: string;
}

export interface AIVARequest {
  request_id: string;
  source_module: string;
  target_module: string;
  request_type: string;
  payload: {};
  trace_id?: string | null;
  timeout_seconds?: number;
  metadata?: {};
  timestamp: string;
}

export interface AIVAResponse {
  request_id: string;
  response_type: string;
  success: boolean;
  payload?: {} | null;
  error_code?: string | null;
  error_message?: string | null;
  metadata?: {};
  timestamp: string;
}

export interface AIVerificationRequest {
  verification_id: string;
  finding_id: string;
  scan_id: string;
  vulnerability_type: VulnerabilityType;
  target: Target;
  evidence: FindingEvidence;
  verification_mode?: string;
  context?: {};
}

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

export interface APISchemaPayload {
  schema_id: string;
  scan_id: string;
  schema_type: string;
  schema_content: {} | string;
  base_url: string;
  authentication?: Authentication;
}

export interface APISecurityTestPayload {
  task_id: string;
  scan_id: string;
  api_type: string;
  api_schema?: APISchemaPayload | null;
  test_cases?: APITestCase[];
  authentication?: Authentication;
}

export interface APITestCase {
  test_id: string;
  test_type: string;
  endpoint: string;
  method: string;
  test_vectors?: {}[];
  expected_behavior?: string | null;
}

export interface AdaptiveBehaviorInfo {
  initial_batch_size?: number;
  final_batch_size?: number;
  rate_adjustments?: number;
  protection_detections?: number;
  bypass_attempts?: number;
  success_rate?: number;
  details?: {};
}

export interface AivaMessage {
  header: MessageHeader;
  topic: Topic;
  schema_version?: string;
  payload: {};
}

export interface Asset {
  asset_id: string;
  type: string;
  value: string;
  parameters?: string[] | null;
  has_form?: boolean;
}

export interface AssetInventoryItem {
  /** 資產唯一標識 */
  asset_id: string;
  /** 資產類型 */
  asset_type: string;
  /** 資產名稱 */
  name: string;
  /** IP地址 */
  ip_address?: string | null;
  /** 主機名 */
  hostname?: string | null;
  /** 域名 */
  domain?: string | null;
  /** 開放端口 */
  ports?: number[];
  /** 技術指紋 */
  fingerprints?: TechnicalFingerprint[];
  /** 業務重要性 */
  business_criticality: string;
  /** 負責人 */
  owner?: string | null;
  /** 環境類型 */
  environment: string;
  /** 最後掃描時間 */
  last_scanned?: string | null;
  /** 漏洞數量 */
  vulnerability_count: number;
  /** 風險評分 */
  risk_score: number;
  discovered_at?: string;
  updated_at?: string;
  /** 元數據 */
  metadata?: {};
}

export interface AssetLifecyclePayload {
  asset_id: string;
  asset_type: AssetType;
  value: string;
  environment: Environment;
  business_criticality: BusinessCriticality;
  data_sensitivity?: DataSensitivity | null;
  asset_exposure?: AssetExposure | null;
  owner?: string | null;
  team?: string | null;
  compliance_tags?: ComplianceFramework[];
  metadata?: {};
  created_at?: string;
}

export interface AttackPathEdge {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  edge_type: AttackPathEdgeType;
  risk_score?: number;
  properties?: {};
}

export interface AttackPathNode {
  node_id: string;
  node_type: AttackPathNodeType;
  name: string;
  properties?: {};
}

export interface AttackPathPayload {
  path_id: string;
  scan_id: string;
  source_node: AttackPathNode;
  target_node: AttackPathNode;
  nodes: AttackPathNode[];
  edges: AttackPathEdge[];
  total_risk_score: number;
  path_length: number;
  description?: string | null;
  timestamp?: string;
}

export interface AttackPathRecommendation {
  path_id: string;
  risk_level: RiskLevel;
  priority_score: number;
  executive_summary: string;
  technical_explanation: string;
  business_impact: string;
  remediation_steps: string[];
  quick_wins?: string[];
  affected_assets?: string[];
  estimated_effort: string;
  estimated_risk_reduction: number;
  timestamp?: string;
}

export interface AttackPlan {
  plan_id: string;
  scan_id: string;
  attack_type: VulnerabilityType;
  steps: AttackStep[];
  dependencies?: {};
  context?: {};
  target_info?: {};
  created_at?: string;
  created_by?: string;
  mitre_techniques?: string[];
  mitre_tactics?: string[];
  capec_id?: string | null;
  metadata?: {};
}

export interface AttackStep {
  step_id: string;
  action: string;
  tool_type: string;
  target?: {};
  parameters?: {};
  expected_result?: string | null;
  timeout_seconds?: number;
  retry_count?: number;
  mitre_technique_id?: string | null;
  mitre_tactic?: string | null;
  metadata?: {};
}

export interface AuthZAnalysisPayload {
  task_id: string;
  scan_id: string;
  analysis_type: string;
  target?: string | null;
}

export interface AuthZCheckPayload {
  task_id: string;
  scan_id: string;
  user_id: string;
  resource: string;
  permission: string;
  context?: {};
}

export interface AuthZResultPayload {
  task_id: string;
  scan_id: string;
  decision: string;
  analysis?: {};
  recommendations?: string[];
  timestamp?: string;
}

export interface Authentication {
  method?: string;
  credentials?: {} | null;
}

export interface BizLogicResultPayload {
  task_id: string;
  scan_id: string;
  test_type: string;
  status: string;
  findings?: {}[];
  statistics?: {};
  timestamp?: string;
}

export interface BizLogicTestPayload {
  task_id: string;
  scan_id: string;
  test_type: string;
  target_urls: {};
  test_config?: {};
  product_id?: string | null;
  workflow_steps?: {}[];
}

export interface CVEReference {
  /** CVE ID (格式: CVE-YYYY-NNNNN) */
  cve_id: string;
  description?: string | null;
  cvss_score?: number | null;
  cvss_vector?: string | null;
  references?: string[];
  published_date?: string | null;
  last_modified_date?: string | null;
}

export interface CVSSv3Metrics {
  /** 攻擊向量 */
  attack_vector: "N" | "A" | "L" | "P";
  /** 攻擊複雜度 */
  attack_complexity: "L" | "H";
  /** 所需權限 */
  privileges_required: "N" | "L" | "H";
  /** 用戶交互 */
  user_interaction: "N" | "R";
  /** 範圍 */
  scope: "U" | "C";
  /** 機密性影響 */
  confidentiality: "N" | "L" | "H";
  /** 完整性影響 */
  integrity: "N" | "L" | "H";
  /** 可用性影響 */
  availability: "N" | "L" | "H";
  /** 漏洞利用代碼成熟度 */
  exploit_code_maturity?: "X" | "H" | "F" | "P" | "U";
  /** 修復級別 */
  remediation_level?: "X" | "U" | "W" | "T" | "O";
  /** 報告置信度 */
  report_confidence?: "X" | "C" | "R" | "U";
  /** 機密性要求 */
  confidentiality_requirement?: "X" | "L" | "M" | "H";
  /** 完整性要求 */
  integrity_requirement?: "X" | "L" | "M" | "H";
  /** 可用性要求 */
  availability_requirement?: "X" | "L" | "M" | "H";
  /** 基本分數 */
  base_score?: number | null;
  /** 時間分數 */
  temporal_score?: number | null;
  /** 環境分數 */
  environmental_score?: number | null;
  /** CVSS 向量字符串 */
  vector_string?: string | null;
}

export interface CWEReference {
  /** CWE ID (格式: CWE-XXX) */
  cwe_id: string;
  name?: string | null;
  description?: string | null;
  weakness_category?: string | null;
  likelihood_of_exploit?: string | null;
}

export interface CodeLevelRootCause {
  analysis_id: string;
  vulnerable_component: string;
  affected_findings: string[];
  code_location?: string | null;
  vulnerability_pattern?: string | null;
  fix_recommendation?: string | null;
}

export interface CodeQualityReport {
  /** 程式語言 */
  language: ProgrammingLanguage;
  /** 檔案路徑 */
  file_path: string;
  /** 品質指標 */
  metrics: {};
  /** 發現的問題 */
  issues?: string[];
  /** 改進建議 */
  suggestions?: string[];
  /** 整體評分 */
  overall_score: number;
  /** 分析時間 */
  timestamp: string;
}

export interface ConfigUpdatePayload {
  update_id: string;
  config_items?: {};
}

export interface CrossLanguageAnalysis {
  /** 分析ID */
  analysis_id: string;
  /** 專案名稱 */
  project_name: string;
  /** 分析的語言 */
  languages_analyzed: ProgrammingLanguage[];
  /** 跨語言問題 */
  cross_language_issues?: string[];
  /** 語言整合點 */
  integration_points?: string[];
  /** 安全邊界 */
  security_boundaries?: string[];
  /** 資料流風險 */
  data_flow_risks?: string[];
  /** 建議 */
  recommendations?: string[];
  /** 風險評分 */
  risk_score: number;
}

export interface DiscoveredAsset {
  asset_id: string;
  asset_type: AssetType;
  value: string;
  discovery_method: string;
  confidence: Confidence;
  metadata?: {};
  discovered_at?: string;
}

export interface EASMAsset {
  /** 資產ID */
  asset_id: string;
  /** 資產類型 */
  asset_type: string;
  /** 資產值 */
  value: string;
  /** 發現方法 */
  discovery_method: string;
  /** 發現來源 */
  discovery_source: string;
  /** 首次發現時間 */
  first_discovered: string;
  /** 最後發現時間 */
  last_seen: string;
  /** 資產狀態 */
  status: string;
  /** 置信度 */
  confidence: number;
  /** 檢測到的技術 */
  technologies?: string[];
  /** 運行的服務 */
  services?: {}[];
  /** SSL證書信息 */
  certificates?: {}[];
  /** 風險評分 */
  risk_score: number;
  /** 漏洞數量 */
  vulnerability_count: number;
  /** 暴露級別 */
  exposure_level: string;
  /** 業務單位 */
  business_unit?: string | null;
  /** 負責人 */
  owner?: string | null;
  /** 重要性 */
  criticality: string;
  /** 合規狀態 */
  compliance_status?: {};
  /** 政策違規 */
  policy_violations?: string[];
  /** 元數據 */
  metadata?: {};
}

export interface EASMDiscoveryPayload {
  discovery_id: string;
  scan_id: string;
  discovery_type: string;
  targets: string[];
  scope?: ScanScope;
  max_depth?: number;
  passive_only?: boolean;
}

export interface EASMDiscoveryResult {
  discovery_id: string;
  scan_id: string;
  status: string;
  discovered_assets?: {}[];
  statistics?: {};
  timestamp?: string;
}

export interface EarlyStoppingInfo {
  reason: StoppingReason;
  timestamp?: string;
  total_tests: number;
  completed_tests: number;
  remaining_tests: number;
  details?: {};
}

export interface EnhancedAttackPath {
  /** 路徑ID */
  path_id: string;
  /** 目標資產 */
  target_asset: string;
  /** 路徑節點 */
  nodes: EnhancedAttackPathNode[];
  /** 邊關係 */
  edges: {}[];
  /** 路徑可行性 */
  path_feasibility: number;
  /** 估計時間(分鐘) */
  estimated_time: number;
  /** 所需技能等級 */
  skill_level_required: string;
  /** 成功概率 */
  success_probability: number;
  /** 被檢測概率 */
  detection_probability: number;
  /** 總體風險 */
  overall_risk: number;
  /** 阻斷控制 */
  blocking_controls?: string[];
  /** 檢測控制 */
  detection_controls?: string[];
  /** 元數據 */
  metadata?: {};
}

export interface EnhancedAttackPathNode {
  /** 節點ID */
  node_id: string;
  /** 節點類型 */
  node_type: string;
  /** 節點名稱 */
  name: string;
  /** 節點描述 */
  description?: string | null;
  /** 可利用性 */
  exploitability: number;
  /** 影響度 */
  impact: number;
  /** 難度 */
  difficulty: number;
  /** MITRE技術ID */
  mitre_technique?: string | null;
  /** MITRE戰術 */
  mitre_tactic?: string | null;
  /** 前置條件 */
  prerequisites?: string[];
  /** 後果 */
  consequences?: string[];
  /** 元數據 */
  metadata?: {};
}

export interface EnhancedFindingPayload {
  finding_id: string;
  task_id: string;
  scan_id: string;
  status: string;
  vulnerability: EnhancedVulnerability;
  target: Target;
  strategy?: string | null;
  evidence?: FindingEvidence | null;
  impact?: FindingImpact | null;
  recommendation?: FindingRecommendation | null;
  sarif_result?: SARIFResult | null;
  metadata?: {};
  created_at?: string;
  updated_at?: string;
}

export interface EnhancedFunctionTaskTarget {
  /** 目標URL */
  url: string;
  /** HTTP方法 */
  method?: string;
  /** HTTP標頭 */
  headers?: {};
  /** Cookie */
  cookies?: {};
  /** 參數 */
  parameters?: {};
  /** 請求體 */
  body?: string | null;
  /** 是否需要認證 */
  auth_required?: boolean;
}

export interface EnhancedFunctionTelemetry {
  payloads_sent?: number;
  detections?: number;
  attempts?: number;
  errors?: string[];
  duration_seconds?: number;
  timestamp?: string;
  error_records?: ErrorRecord[];
  oast_callbacks?: OastCallbackDetail[];
  early_stopping?: EarlyStoppingInfo | null;
  adaptive_behavior?: AdaptiveBehaviorInfo | null;
}

export interface EnhancedIOCRecord {
  /** IOC唯一標識符 */
  ioc_id: string;
  /** IOC類型 */
  ioc_type: string;
  /** IOC值 */
  value: string;
  /** 威脅類型 */
  threat_type?: string | null;
  /** 惡意軟體家族 */
  malware_family?: string | null;
  /** 攻擊活動 */
  campaign?: string | null;
  /** 嚴重程度 */
  severity: Severity;
  /** 可信度 0-100 */
  confidence: number;
  /** 聲譽分數 */
  reputation_score: number;
  /** 首次發現時間 */
  first_seen?: string | null;
  /** 最後發現時間 */
  last_seen?: string | null;
  /** 過期時間 */
  expires_at?: string | null;
  /** 標籤 */
  tags?: string[];
  /** MITRE ATT&CK技術 */
  mitre_techniques?: string[];
  /** 元數據 */
  metadata?: {};
}

export interface EnhancedModuleStatus {
  /** 模組名稱 */
  module_name: ModuleName;
  /** 模組版本 */
  version: string;
  /** 運行狀態 */
  status: string;
  /** 健康評分 */
  health_score: number;
  /** CPU使用率 */
  cpu_usage: number;
  /** 內存使用(MB) */
  memory_usage: number;
  /** 活躍連接數 */
  active_connections: number;
  /** 處理任務數 */
  tasks_processed: number;
  /** 待處理任務數 */
  tasks_pending: number;
  /** 錯誤次數 */
  error_count: number;
  /** 啟動時間 */
  started_at: string;
  /** 最後心跳 */
  last_heartbeat: string;
  /** 運行時間(秒) */
  uptime_seconds: number;
  /** 元數據 */
  metadata?: {};
}

export interface EnhancedRiskAssessment {
  /** 評估ID */
  assessment_id: string;
  /** 目標ID */
  target_id: string;
  /** 總體風險評分 */
  overall_risk_score: number;
  /** 可能性評分 */
  likelihood_score: number;
  /** 影響評分 */
  impact_score: number;
  /** 風險級別 */
  risk_level: Severity;
  /** 風險分類 */
  risk_category: string;
  /** 風險因子列表 */
  risk_factors: RiskFactor[];
  /** CVSS評分 */
  cvss_metrics?: CVSSv3Metrics | null;
  /** 業務影響描述 */
  business_impact?: string | null;
  /** 受影響資產 */
  affected_assets?: string[];
  /** 緩解策略 */
  mitigation_strategies?: string[];
  /** 殘餘風險 */
  residual_risk: number;
  assessed_at?: string;
  /** 有效期限 */
  valid_until?: string | null;
  /** 元數據 */
  metadata?: {};
}

export interface EnhancedScanRequest {
  /** 掃描ID */
  scan_id: string;
  /** 目標URL列表 */
  targets: string[];
  /** 掃描範圍 */
  scope: EnhancedScanScope;
  /** 掃描策略 */
  strategy: string;
  /** 優先級 1-10 */
  priority?: number;
  /** 最大執行時間(秒) */
  max_duration?: number;
  /** 額外元數據 */
  metadata?: {};
}

export interface EnhancedScanScope {
  /** 包含的主機 */
  included_hosts?: string[];
  /** 排除的主機 */
  excluded_hosts?: string[];
  /** 包含的路徑 */
  included_paths?: string[];
  /** 排除的路徑 */
  excluded_paths?: string[];
  /** 最大掃描深度 */
  max_depth?: number;
}

export interface EnhancedTaskExecution {
  /** 任務ID */
  task_id: string;
  /** 任務類型 */
  task_type: string;
  /** 執行模組 */
  module_name: ModuleName;
  /** 優先級 */
  priority: number;
  /** 超時時間(秒) */
  timeout?: number;
  /** 重試次數 */
  retry_count?: number;
  /** 任務依賴 */
  dependencies?: TaskDependency[];
  /** 執行狀態 */
  status: TestStatus;
  /** 執行進度 */
  progress: number;
  /** 結果數據 */
  result_data?: {};
  /** 錯誤消息 */
  error_message?: string | null;
  /** CPU使用率 */
  cpu_usage?: number | null;
  /** 內存使用(MB) */
  memory_usage?: number | null;
  created_at?: string;
  /** 開始時間 */
  started_at?: string | null;
  /** 完成時間 */
  completed_at?: string | null;
  /** 元數據 */
  metadata?: {};
}

export interface EnhancedVulnerability {
  /** 漏洞唯一標識 */
  vulnerability_id: string;
  /** 漏洞標題 */
  title: string;
  /** 漏洞描述 */
  description: string;
  /** 漏洞類型 */
  vulnerability_type: string;
  /** 嚴重性 */
  severity: "low" | "medium" | "high" | "critical";
  /** 漏洞URL */
  url: string;
  /** 參數名 */
  parameter?: string | null;
  /** 參數位置 */
  location: string;
  /** CVSS v3.1 指標 */
  cvss_metrics?: CVSSv3Metrics | null;
  /** AI 置信度 */
  ai_confidence: number;
  /** AI 風險評估 */
  ai_risk_assessment?: {};
  /** 可利用性分數 */
  exploitability_score: number;
  /** 攻擊向量 */
  attack_vector: string;
  /** 攻擊複雜度 */
  attack_complexity: string;
  /** 利用前提 */
  prerequisites?: string[];
  /** 業務影響 */
  business_impact?: {};
  /** 技術影響 */
  technical_impact?: {};
  /** 修復難度 */
  remediation_effort: string;
  /** 修復優先級 */
  remediation_priority: number;
  /** 修復建議 */
  fix_recommendations?: string[];
  /** 是否有概念驗證 */
  poc_available?: boolean;
  /** 是否已驗證 */
  verified?: boolean;
  /** 誤報概率 */
  false_positive_probability: number;
  discovered_at?: string;
  /** 最後驗證時間 */
  last_verified_at?: string | null;
  /** 標籤 */
  tags?: string[];
  /** 參考資料 */
  references?: string[];
  /** 額外元數據 */
  metadata?: {};
}

export interface EnhancedVulnerabilityCorrelation {
  /** 關聯分析ID */
  correlation_id: string;
  /** 主要漏洞ID */
  primary_vulnerability: string;
  /** 相關漏洞列表 */
  related_vulnerabilities: string[];
  /** 關聯強度 */
  correlation_strength: number;
  /** 關聯類型 */
  correlation_type: string;
  /** 組合風險評分 */
  combined_risk_score: number;
  /** 利用複雜度 */
  exploitation_complexity: number;
  /** 攻擊場景 */
  attack_scenarios?: string[];
  /** 建議利用順序 */
  recommended_order?: string[];
  /** 協調緩解措施 */
  coordinated_mitigation?: string[];
  /** 優先級排序 */
  priority_ranking?: string[];
  analyzed_at?: string;
  /** 元數據 */
  metadata?: {};
}

export interface ErrorRecord {
  category: ErrorCategory;
  message: string;
  timestamp?: string;
  details?: {};
}

export interface ExecutionError {
  error_id: string;
  error_type: string;
  message: string;
  payload?: string | null;
  vector?: string | null;
  timestamp?: string;
  attempts?: number;
}

export interface ExperienceSample {
  /** 樣本唯一標識 */
  sample_id: string;
  /** 會話ID */
  session_id: string;
  /** 計劃ID */
  plan_id: string;
  /** 執行前狀態 */
  state_before: {};
  /** 採取的行動 */
  action_taken: {};
  /** 執行後狀態 */
  state_after: {};
  /** 獎勵值 */
  reward: number;
  /** 獎勵分解 (completion, success, sequence, goal) */
  reward_breakdown?: {};
  /** 環境上下文 */
  context?: {};
  /** 目標信息 */
  target_info?: {};
  timestamp?: string;
  /** 執行時長 */
  duration_ms?: number | null;
  /** 樣本質量分數 */
  quality_score?: number | null;
  /** 是否為正樣本 */
  is_positive: boolean;
  /** 樣本置信度 */
  confidence?: number;
  /** 學習標籤 */
  learning_tags?: string[];
  /** 難度級別 */
  difficulty_level?: number;
}

export interface ExploitPayload {
  /** 載荷ID */
  payload_id: string;
  /** 載荷類型 */
  payload_type: string;
  /** 載荷內容 */
  payload_content: string;
  /** 編碼方式 */
  encoding?: string;
  /** 是否混淆 */
  obfuscation?: boolean;
  /** 繞過技術 */
  bypass_technique?: string | null;
  /** 目標技術 */
  target_technology?: string[];
  /** 所需上下文 */
  required_context?: {};
  /** 效果評分 */
  effectiveness_score: number;
  /** 逃避檢測能力 */
  detection_evasion: number;
  /** 成功率 */
  success_rate: number;
  /** 使用次數 */
  usage_count: number;
  /** 元數據 */
  metadata?: {};
}

export interface ExploitResult {
  /** 結果ID */
  result_id: string;
  /** 利用ID */
  exploit_id: string;
  /** 目標ID */
  target_id: string;
  /** 利用是否成功 */
  success: boolean;
  /** 嚴重程度 */
  severity: Severity;
  /** 影響級別 */
  impact_level: string;
  /** 利用技術 */
  exploit_technique: string;
  /** 使用的載荷 */
  payload_used: string;
  /** 執行時間(秒) */
  execution_time: number;
  /** 獲得的訪問權限 */
  access_gained?: {};
  /** 提取的數據 */
  data_extracted?: string[];
  /** 系統影響 */
  system_impact?: string | null;
  /** 是否繞過檢測 */
  detection_bypassed: boolean;
  /** 留下的痕跡 */
  artifacts_left?: string[];
  /** 修復是否已驗證 */
  remediation_verified?: boolean;
  /** 是否需要重測 */
  retest_required?: boolean;
  executed_at?: string;
  /** 元數據 */
  metadata?: {};
}

export interface FeedbackEventPayload {
  task_id: string;
  scan_id: string;
  event_type: string;
  details?: {};
  form_url?: string | null;
}

export interface FindingEvidence {
  payload?: string | null;
  response_time_delta?: number | null;
  db_version?: string | null;
  request?: string | null;
  response?: string | null;
  proof?: string | null;
}

export interface FindingImpact {
  description?: string | null;
  business_impact?: string | null;
  technical_impact?: string | null;
  affected_users?: number | null;
  estimated_cost?: number | null;
}

export interface FindingPayload {
  finding_id: string;
  task_id: string;
  scan_id: string;
  status: string;
  vulnerability: Vulnerability;
  target: Target;
  strategy?: string | null;
  evidence?: FindingEvidence | null;
  impact?: FindingImpact | null;
  recommendation?: FindingRecommendation | null;
  metadata?: {};
  created_at?: string;
  updated_at?: string;
}

export interface FindingRecommendation {
  fix?: string | null;
  priority?: string | null;
  remediation_steps?: string[];
  references?: string[];
}

export interface FindingTarget {
  url: any;
  parameter?: string | null;
  method?: string | null;
  headers?: {};
  params?: {};
  body?: string | null;
}

export interface Fingerprints {
  web_server?: {} | null;
  framework?: {} | null;
  language?: {} | null;
  waf_detected?: boolean;
  waf_vendor?: string | null;
}

export interface FunctionExecutionResult {
  findings: {}[];
  telemetry: {};
  errors?: {}[];
  duration_seconds?: number;
  timestamp?: string;
}

export interface FunctionTaskContext {
  db_type_hint?: string | null;
  waf_detected?: boolean;
  related_findings?: string[] | null;
}

export interface FunctionTaskPayload {
  task_id: string;
  scan_id: string;
  priority?: number;
  target: FunctionTaskTarget;
  context?: FunctionTaskContext;
  strategy?: string;
  custom_payloads?: string[] | null;
  test_config?: FunctionTaskTestConfig;
}

export interface FunctionTaskTarget {
  url: any;
  parameter?: string | null;
  method?: string;
  parameter_location?: string;
  headers?: {};
  cookies?: {};
  form_data?: {};
  json_data?: {} | null;
  body?: string | null;
}

export interface FunctionTaskTestConfig {
  payloads?: string[];
  custom_payloads?: string[];
  blind_xss?: boolean;
  dom_testing?: boolean;
  timeout?: number | null;
}

export interface FunctionTelemetry {
  payloads_sent?: number;
  detections?: number;
  attempts?: number;
  errors?: string[];
  duration_seconds?: number;
  timestamp?: string;
}

export interface HeartbeatPayload {
  module: ModuleName;
  worker_id: string;
  capacity: number;
}

export interface JavaScriptAnalysisResult {
  analysis_id: string;
  url: string;
  source_size_bytes: number;
  dangerous_functions?: string[];
  external_resources?: string[];
  data_leaks?: {}[];
  findings?: string[];
  apis_called?: string[];
  ajax_endpoints?: string[];
  suspicious_patterns?: string[];
  risk_score?: number;
  security_score?: number;
  timestamp?: string;
}

export interface LanguageDetectionResult {
  /** 主要程式語言 */
  primary_language: ProgrammingLanguage;
  /** 檢測信心度 */
  confidence: number;
  /** 次要程式語言 */
  secondary_languages?: ProgrammingLanguage[];
  /** 檢測到的框架 */
  frameworks?: LanguageFramework[];
  /** 檔案副檔名 */
  file_extensions?: string[];
  /** 程式碼行數 */
  lines_of_code: number;
}

export interface LanguageInteroperability {
  /** 來源語言 */
  source_language: ProgrammingLanguage;
  /** 目標語言 */
  target_language: ProgrammingLanguage;
  /** 互操作方法 */
  interop_method: string;
  /** 安全考量 */
  security_considerations?: string[];
  /** 效能影響 */
  performance_impact?: string | null;
  /** 相容性問題 */
  compatibility_issues?: string[];
  /** 建議 */
  recommendations?: string[];
}

export interface LanguageSpecificPayload {
  /** 目標程式語言 */
  language: ProgrammingLanguage;
  /** 載荷類型 */
  payload_type: string;
  /** 載荷內容 */
  payload_content: string;
  /** 編碼方式 */
  encoding?: string;
  /** 預期行為 */
  expected_behavior?: string | null;
  /** 繞過技術 */
  bypass_techniques?: string[];
  /** 目標函數 */
  target_functions?: string[];
  /** 成功指標 */
  success_indicators?: string[];
}

export interface LanguageSpecificScanConfig {
  /** 目標程式語言 */
  language: ProgrammingLanguage;
  /** 要檢查的安全模式 */
  scan_patterns: SecurityPattern[];
  /** 程式碼品質指標 */
  quality_metrics?: CodeQualityMetric[];
  /** 排除路徑 */
  exclude_paths?: string[];
  /** 包含檔案模式 */
  include_patterns?: string[];
  /** 自訂規則 */
  custom_rules?: string[];
  /** 最大檔案大小（bytes） */
  max_file_size?: number;
  /** 掃描超時時間（秒） */
  timeout_seconds?: number;
}

export interface LanguageSpecificVulnerability {
  /** 程式語言 */
  language: ProgrammingLanguage;
  /** 漏洞類型 */
  vulnerability_type: VulnerabilityByLanguage;
  /** 嚴重程度 */
  severity: Severity;
  /** 漏洞描述 */
  description: string;
  /** 問題程式碼片段 */
  code_snippet?: string | null;
  /** 行號 */
  line_number?: number | null;
  /** 檔案路徑 */
  file_path?: string | null;
  /** 函數名稱 */
  function_name?: string | null;
  /** 修復建議 */
  remediation?: string | null;
  /** CWE ID */
  cwe_id?: string | null;
  /** OWASP 分類 */
  owasp_category?: string | null;
}

export interface MessageHeader {
  message_id: string;
  trace_id: string;
  correlation_id?: string | null;
  source_module: ModuleName;
  timestamp?: string;
  version?: string;
}

export interface ModelTrainingConfig {
  config_id: string;
  model_type: string;
  training_mode: string;
  batch_size?: number;
  learning_rate?: number;
  epochs?: number;
  validation_split?: number;
  early_stopping?: boolean;
  patience?: number;
  reward_function?: string;
  discount_factor?: number;
  exploration_rate?: number;
  hyperparameters?: {};
  metadata?: {};
}

export interface ModelTrainingResult {
  training_id: string;
  config: {};
  model_version: string;
  training_samples: number;
  validation_samples: number;
  training_loss: number;
  validation_loss: number;
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1_score?: number | null;
  average_reward?: number | null;
  training_duration_seconds?: number;
  started_at?: string;
  completed_at?: string;
  metrics?: {};
  model_path?: string | null;
  metadata?: {};
}

export interface ModuleStatus {
  module: ModuleName;
  status: string;
  worker_id: string;
  worker_count?: number;
  queue_size?: number;
  tasks_completed?: number;
  tasks_failed?: number;
  last_heartbeat?: string;
  metrics?: {};
  uptime_seconds?: number;
}

export interface MultiLanguageCodebase {
  /** 專案名稱 */
  project_name: string;
  /** 語言分布（語言：程式碼行數） */
  languages: {};
  /** 主要程式語言 */
  primary_language: ProgrammingLanguage;
  /** 使用的框架 */
  frameworks?: LanguageFramework[];
  /** 總檔案數 */
  total_files: number;
  /** 總程式碼行數 */
  total_lines: number;
  /** 各語言漏洞分布 */
  vulnerability_distribution?: {};
  /** 依賴套件（語言：套件列表） */
  dependencies?: {};
}

export interface NotificationPayload {
  notification_id: string;
  notification_type: string;
  priority: string;
  title: string;
  message: string;
  details?: {};
  recipients?: string[];
  attachments?: {}[];
  timestamp?: string;
}

export interface OastCallbackDetail {
  callback_type: string;
  token: string;
  source_ip: string;
  timestamp: string;
  protocol?: string | null;
  raw_data?: {};
}

export interface OastEvent {
  event_id: string;
  probe_token: string;
  event_type: string;
  source_ip: string;
  timestamp?: string;
  protocol?: string | null;
  raw_request?: string | null;
  raw_data?: {};
}

export interface OastProbe {
  probe_id: string;
  token: string;
  callback_url: string;
  task_id: string;
  scan_id: string;
  created_at?: string;
  expires_at?: string | null;
  status?: string;
}

export interface PlanExecutionMetrics {
  plan_id: string;
  session_id: string;
  expected_steps: number;
  executed_steps: number;
  completed_steps: number;
  failed_steps: number;
  skipped_steps: number;
  extra_actions: number;
  completion_rate: number;
  success_rate: number;
  sequence_accuracy: number;
  goal_achieved: boolean;
  reward_score: number;
  total_execution_time: number;
  timestamp?: string;
}

export interface PlanExecutionResult {
  result_id: string;
  plan_id: string;
  session_id: string;
  plan: AttackPlan;
  trace: TraceRecord[];
  metrics: PlanExecutionMetrics;
  findings?: {}[];
  anomalies?: string[];
  recommendations?: string[];
  status: string;
  completed_at?: string;
  metadata?: {};
}

export interface PostExResultPayload {
  task_id: string;
  scan_id: string;
  test_type: PostExTestType;
  findings?: {}[];
  risk_level: ThreatLevel;
  safe_mode: boolean;
  authorization_verified?: boolean;
  timestamp?: string;
}

export interface PostExTestPayload {
  task_id: string;
  scan_id: string;
  test_type: PostExTestType;
  target: string;
  safe_mode?: boolean;
  authorization_token?: string | null;
  context?: {};
}

export interface RAGKnowledgeUpdatePayload {
  knowledge_type: string;
  content: string;
  source_id?: string | null;
  category?: string | null;
  tags?: string[];
  related_cve?: string | null;
  related_cwe?: string | null;
  mitre_techniques?: string[];
  confidence?: number;
  metadata?: {};
}

export interface RAGQueryPayload {
  query_id: string;
  query_text: string;
  top_k?: number;
  min_similarity?: number;
  knowledge_types?: string[] | null;
  categories?: string[] | null;
  metadata?: {};
}

export interface RAGResponsePayload {
  query_id: string;
  results?: {}[];
  total_results: number;
  avg_similarity?: number | null;
  enhanced_context?: string | null;
  metadata?: {};
  timestamp?: string;
}

export interface RateLimit {
  requests_per_second?: number;
  burst?: number;
}

export interface RemediationGeneratePayload {
  task_id: string;
  scan_id: string;
  finding_id: string;
  vulnerability_type: VulnerabilityType;
  remediation_type: RemediationType;
  context?: {};
  auto_apply?: boolean;
}

export interface RemediationResultPayload {
  task_id: string;
  scan_id: string;
  finding_id: string;
  remediation_type: RemediationType;
  status: string;
  patch_content?: string | null;
  instructions?: string[];
  verification_steps?: string[];
  risk_assessment?: {};
  timestamp?: string;
}

export interface RiskAssessmentContext {
  environment: Environment;
  business_criticality: BusinessCriticality;
  data_sensitivity?: DataSensitivity | null;
  asset_exposure?: AssetExposure | null;
  compliance_tags?: ComplianceFramework[];
  asset_value?: number | null;
  user_base?: number | null;
  sla_hours?: number | null;
}

export interface RiskAssessmentResult {
  finding_id: string;
  technical_risk_score: number;
  business_risk_score: number;
  risk_level: RiskLevel;
  priority_score: number;
  context_multiplier: number;
  business_impact?: {};
  recommendations?: string[];
  estimated_effort?: string | null;
  timestamp?: string;
}

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

export interface RiskTrendAnalysis {
  period_start: string;
  period_end: string;
  total_vulnerabilities: number;
  risk_distribution: {};
  average_risk_score: number;
  trend: string;
  improvement_percentage?: number | null;
  top_risks?: {}[];
}

export interface SARIFLocation {
  /** 資源URI */
  uri: string;
  /** 開始行號 */
  start_line?: number | null;
  /** 開始列號 */
  start_column?: number | null;
  /** 結束行號 */
  end_line?: number | null;
  /** 結束列號 */
  end_column?: number | null;
}

export interface SARIFReport {
  /** SARIF版本 */
  version?: string;
  /** JSON Schema URL */
  $schema?: string;
  /** 運行列表 */
  runs: SARIFRun[];
  /** 屬性 */
  properties?: {};
}

export interface SARIFResult {
  /** 規則ID */
  rule_id: string;
  /** 消息 */
  message: string;
  /** 級別 */
  level: "error" | "warning" | "info" | "note";
  /** 位置列表 */
  locations: SARIFLocation[];
  /** 部分指紋 */
  partial_fingerprints?: {};
  /** 屬性 */
  properties?: {};
}

export interface SARIFRule {
  /** 規則ID */
  id: string;
  /** 規則名稱 */
  name: string;
  /** 簡短描述 */
  short_description: string;
  /** 完整描述 */
  full_description?: string | null;
  /** 幫助URI */
  help_uri?: string | null;
  /** 默認級別 */
  default_level?: "error" | "warning" | "info" | "note";
  /** 屬性 */
  properties?: {};
}

export interface SARIFRun {
  /** 工具信息 */
  tool: SARIFTool;
  /** 結果列表 */
  results: SARIFResult[];
  /** 調用信息 */
  invocations?: {}[];
  /** 工件信息 */
  artifacts?: {}[];
  /** 屬性 */
  properties?: {};
}

export interface SARIFTool {
  /** 工具名稱 */
  name: string;
  /** 版本 */
  version: string;
  /** 信息URI */
  information_uri?: string | null;
  /** 規則列表 */
  rules?: SARIFRule[];
}

export interface SASTDASTCorrelation {
  correlation_id: string;
  sast_finding_id: string;
  dast_finding_id: string;
  data_flow_path: string[];
  verification_status: string;
  confidence_score: number;
  explanation?: string | null;
}

export interface SIEMEvent {
  /** 事件ID */
  event_id: string;
  /** 事件類型 */
  event_type: string;
  /** 來源系統 */
  source_system: string;
  /** 事件時間戳 */
  timestamp: string;
  received_at?: string;
  /** 嚴重程度 */
  severity: Severity;
  /** 事件分類 */
  category: string;
  /** 事件子分類 */
  subcategory?: string | null;
  /** 來源IP */
  source_ip?: string | null;
  /** 來源端口 */
  source_port?: number | null;
  /** 目標IP */
  destination_ip?: string | null;
  /** 目標端口 */
  destination_port?: number | null;
  /** 用戶名 */
  username?: string | null;
  /** 資產ID */
  asset_id?: string | null;
  /** 主機名 */
  hostname?: string | null;
  /** 事件描述 */
  description: string;
  /** 原始日誌 */
  raw_log?: string | null;
  /** 觸發的關聯規則 */
  correlation_rules?: string[];
  /** 相關事件ID */
  related_events?: string[];
  /** 處理狀態 */
  status?: string;
  /** 分配給 */
  assigned_to?: string | null;
  /** 元數據 */
  metadata?: {};
}

export interface SIEMEventPayload {
  event_id: string;
  event_type: string;
  severity: string;
  source: string;
  destination?: string | null;
  message: string;
  details?: {};
  timestamp?: string;
}

export interface ScanCompletedPayload {
  scan_id: string;
  status: string;
  summary: Summary;
  assets?: Asset[];
  fingerprints?: Fingerprints | null;
  error_info?: string | null;
}

export interface ScanScope {
  exclusions?: string[];
  include_subdomains?: boolean;
  allowed_hosts?: string[];
}

export interface ScanStartPayload {
  scan_id: string;
  targets: string[];
  scope?: ScanScope;
  authentication?: Authentication;
  strategy?: string;
  rate_limit?: RateLimit;
  custom_headers?: {};
  x_forwarded_for?: string | null;
}

export interface ScenarioTestResult {
  test_id: string;
  scenario_id: string;
  model_version: string;
  generated_plan: {};
  execution_result: {};
  score: number;
  comparison: {};
  passed: boolean;
  tested_at?: string;
  metadata?: {};
}

export interface SensitiveMatch {
  match_id: string;
  pattern_name: string;
  matched_text: string;
  context: string;
  confidence: number;
  line_number?: number | null;
  file_path?: string | null;
  url?: string | null;
  severity?: Severity;
}

export interface SessionState {
  session_id: string;
  plan_id: string;
  scan_id: string;
  status: string;
  current_step_index?: number;
  completed_steps?: string[];
  pending_steps?: string[];
  context?: {};
  variables?: {};
  started_at?: string;
  updated_at?: string;
  timeout_at?: string | null;
  metadata?: {};
}

export interface StandardScenario {
  scenario_id: string;
  name: string;
  description: string;
  vulnerability_type: VulnerabilityType;
  difficulty_level: string;
  target_config: {};
  expected_plan: {};
  success_criteria: {};
  tags?: string[];
  created_at?: string;
  metadata?: {};
}

export interface Summary {
  urls_found?: number;
  forms_found?: number;
  apis_found?: number;
  scan_duration_seconds?: number;
}

export interface SystemOrchestration {
  /** 編排ID */
  orchestration_id: string;
  /** 編排名稱 */
  orchestration_name: string;
  /** 模組狀態列表 */
  module_statuses: EnhancedModuleStatus[];
  /** 掃描配置 */
  scan_configuration?: {};
  /** 資源分配 */
  resource_allocation?: {};
  /** 整體狀態 */
  overall_status: string;
  /** 活躍掃描數 */
  active_scans: number;
  /** 排隊任務數 */
  queued_tasks: number;
  /** 系統CPU */
  system_cpu: number;
  /** 系統內存(MB) */
  system_memory: number;
  /** 網絡吞吐量(Mbps) */
  network_throughput: number;
  created_at?: string;
  updated_at?: string;
  /** 元數據 */
  metadata?: {};
}

export interface Target {
  url: any;
  parameter?: string | null;
  method?: string | null;
  headers?: {};
  params?: {};
  body?: string | null;
}

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

export interface TaskQueue {
  /** 隊列ID */
  queue_id: string;
  /** 隊列名稱 */
  queue_name: string;
  /** 最大併發任務數 */
  max_concurrent_tasks?: number;
  /** 任務超時(秒) */
  task_timeout?: number;
  /** 等待任務 */
  pending_tasks?: string[];
  /** 運行任務 */
  running_tasks?: string[];
  /** 完成任務 */
  completed_tasks?: string[];
  /** 總處理數 */
  total_processed: number;
  /** 成功率 */
  success_rate: number;
  /** 平均執行時間 */
  average_execution_time: number;
  created_at?: string;
  last_activity?: string;
  /** 元數據 */
  metadata?: {};
}

export interface TaskUpdatePayload {
  task_id: string;
  scan_id: string;
  status: string;
  worker_id: string;
  details?: {} | null;
}

export interface TechnicalFingerprint {
  /** 技術名稱 */
  technology: string;
  /** 版本信息 */
  version?: string | null;
  /** 置信度 */
  confidence: number;
  /** 檢測方法 */
  detection_method: string;
  /** 檢測證據 */
  evidence?: string[];
  /** 技術分類 */
  category: string;
  /** 子分類 */
  subcategory?: string | null;
  /** 已知漏洞 */
  known_vulnerabilities?: string[];
  /** 是否已停止支持 */
  eol_status?: boolean | null;
  /** 額外信息 */
  metadata?: {};
}

export interface TestExecution {
  /** 執行ID */
  execution_id: string;
  /** 測試案例ID */
  test_case_id: string;
  /** 目標URL */
  target_url: string;
  /** 超時時間(秒) */
  timeout?: number;
  /** 重試次數 */
  retry_attempts?: number;
  /** 執行狀態 */
  status: TestStatus;
  start_time?: string;
  /** 結束時間 */
  end_time?: string | null;
  /** 執行時間(秒) */
  duration?: number | null;
  /** 是否成功 */
  success: boolean;
  /** 是否發現漏洞 */
  vulnerability_found: boolean;
  /** 結果置信度 */
  confidence_level: Confidence;
  /** 請求數據 */
  request_data?: {};
  /** 響應數據 */
  response_data?: {};
  /** 證據列表 */
  evidence?: string[];
  /** 錯誤消息 */
  error_message?: string | null;
  /** CPU使用率 */
  cpu_usage?: number | null;
  /** 內存使用(MB) */
  memory_usage?: number | null;
  /** 網絡流量(bytes) */
  network_traffic?: number | null;
  /** 元數據 */
  metadata?: {};
}

export interface TestStrategy {
  /** 策略ID */
  strategy_id: string;
  /** 策略名稱 */
  strategy_name: string;
  /** 目標類型 */
  target_type: string;
  /** 測試分類 */
  test_categories: string[];
  /** 測試順序 */
  test_sequence: string[];
  /** 是否並行執行 */
  parallel_execution?: boolean;
  /** 觸發條件 */
  trigger_conditions?: string[];
  /** 停止條件 */
  stop_conditions?: string[];
  /** 優先級權重 */
  priority_weights?: {};
  /** 資源限制 */
  resource_limits?: {};
  /** 是否啟用學習 */
  learning_enabled?: boolean;
  /** 適應閾值 */
  adaptation_threshold: number;
  /** 效果評分 */
  effectiveness_score: number;
  /** 使用次數 */
  usage_count: number;
  /** 成功率 */
  success_rate: number;
  created_at?: string;
  /** 元數據 */
  metadata?: {};
}

export interface ThreatIntelLookupPayload {
  task_id: string;
  scan_id: string;
  indicator: string;
  indicator_type: IOCType;
  sources?: IntelSource[] | null;
  enrich?: boolean;
}

export interface ThreatIntelResultPayload {
  task_id: string;
  scan_id: string;
  indicator: string;
  indicator_type: IOCType;
  threat_level: ThreatLevel;
  sources?: {};
  mitre_techniques?: string[];
  enrichment_data?: {};
  timestamp?: string;
}

export interface TraceRecord {
  trace_id: string;
  plan_id: string;
  step_id: string;
  session_id: string;
  tool_name: string;
  input_data?: {};
  output_data?: {};
  status: string;
  error_message?: string | null;
  execution_time_seconds?: number;
  timestamp?: string;
  environment_response?: {};
  metadata?: {};
}

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

export interface VulnerabilityDiscovery {
  /** 發現ID */
  discovery_id: string;
  /** 漏洞ID */
  vulnerability_id: string;
  /** 相關資產ID */
  asset_id: string;
  /** 漏洞標題 */
  title: string;
  /** 漏洞描述 */
  description: string;
  /** 嚴重程度 */
  severity: Severity;
  /** 置信度 */
  confidence: Confidence;
  /** 漏洞類型 */
  vulnerability_type: string;
  /** 受影響組件 */
  affected_component?: string | null;
  /** 攻擊向量 */
  attack_vector?: string | null;
  /** 檢測方法 */
  detection_method: string;
  /** 掃描器名稱 */
  scanner_name: string;
  /** 掃描規則ID */
  scan_rule_id?: string | null;
  /** 漏洞證據 */
  evidence?: string[];
  /** 概念驗證 */
  proof_of_concept?: string | null;
  /** 誤報可能性 */
  false_positive_likelihood: number;
  /** 影響評估 */
  impact_assessment?: string | null;
  /** 可利用性 */
  exploitability?: string | null;
  /** 修復建議 */
  remediation_advice?: string | null;
  /** 修復優先級 */
  remediation_priority?: string | null;
  /** CVE標識符 */
  cve_ids?: string[];
  /** CWE標識符 */
  cwe_ids?: string[];
  /** CVSS評分 */
  cvss_score?: number | null;
  discovered_at?: string;
  /** 元數據 */
  metadata?: {};
}

export interface VulnerabilityLifecyclePayload {
  vulnerability_id: string;
  finding_id: string;
  asset_id: string;
  vulnerability_type: VulnerabilityType;
  severity: Severity;
  confidence: Confidence;
  status: VulnerabilityStatus;
  exploitability?: Exploitability | null;
  assigned_to?: string | null;
  due_date?: string | null;
  first_detected?: string;
  last_seen?: string;
  resolution_date?: string | null;
  metadata?: {};
}

export interface VulnerabilityUpdatePayload {
  vulnerability_id: string;
  status: VulnerabilityStatus;
  assigned_to?: string | null;
  comment?: string | null;
  metadata?: {};
  updated_by?: string | null;
  timestamp?: string;
}

export interface WebhookPayload {
  /** Webhook ID */
  webhook_id: string;
  /** 事件類型 */
  event_type: string;
  /** 來源系統 */
  source: string;
  timestamp?: string;
  /** 事件數據 */
  data?: {};
  /** 交付URL */
  delivery_url?: string | null;
  /** 重試次數 */
  retry_count?: number;
  /** 最大重試次數 */
  max_retries?: number;
  /** 狀態 */
  status?: string;
  /** 交付時間 */
  delivered_at?: string | null;
  /** 錯誤消息 */
  error_message?: string | null;
  /** 元數據 */
  metadata?: {};
}
