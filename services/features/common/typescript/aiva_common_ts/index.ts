/**
 * AIVA Common TypeScript - 統一 TypeScript AI 組件庫和 Schema 定義
 * 
 * 提供與 Python aiva_common 對應的 TypeScript 實現
 * 支持 AIVA 系統中 TypeScript 模組的 AI 功能需求和數據結構標準化
 * 
 * 架構位置: services/features/common/typescript/aiva_common_ts/
 * 用途: TypeScript 模組的 AI 能力評估、經驗管理和統一數據結構
 */

// ============================================================================
// AI 組件導出
// ============================================================================

// Capability Evaluator
export { 
  default as AIVACapabilityEvaluator,
  createCapabilityEvaluator,
  type CapabilityEvidence,
  type EvaluationMetric,
  type BenchmarkTest,
  type CapabilityAssessment,
  EvidenceType,
  MetricType,
  BenchmarkType,
  EvaluationDimension
} from './capability-evaluator';

// Experience Manager
export {
  default as AIVAExperienceManager,
  createExperienceManager,
  type ExperienceSample,
  type LearningSession,
  type ExperienceFilter,
  type ExperienceStatistics,
  SessionType,
  SessionStatus,
  SortBy,
  StorageBackend
} from './experience-manager';

// ============================================================================
// 統一 Schema 定義導出 (對應 Python aiva_common.schemas)
// ============================================================================

// 枚舉類型
export {
  Severity,
  Confidence,
  VulnerabilityType,
  VulnerabilityStatus,
  TaskStatus,
  ScanStatus,
  
  // 工具函數
  validateFindingId,
  validateTaskId,
  validateScanId,
  validateFindingStatus,
  createFindingPayload,
  generateFindingId,
  generateTaskId,
  generateScanId,
  
  // 類型守衛
  isFindingPayload,
  isVulnerability,
  isTarget
} from './schemas';

// ============================================================================
// 性能優化配置導出 (對應 Python performance_config.py)
// ============================================================================

export {
  CacheStrategy,
  ProcessingMode,
  PerformanceOptimizer,
  OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG,
  OPTIMIZED_EXPERIENCE_MANAGER_CONFIG,
  PERFORMANCE_BENCHMARKS,
  createOptimizedConfigs,
  performanceMonitor,
  batchProcessor,
  cached
} from './performance-config';

// Schema 類型定義
export type {
  FindingPayload,
  Vulnerability,
  Target,
  FindingEvidence,
  FindingImpact,
  FindingRecommendation,
  SensitiveMatch,
  JavaScriptAnalysisResult,
  VulnerabilityCorrelation,
  AIVerificationRequest,
  AIVerificationResult
} from './schemas';

// 性能配置類型定義
export type {
  PerformanceConfig,
  CapabilityEvaluatorConfig,
  ExperienceManagerConfig,
  PerformanceMetric
} from './performance-config';

// ============================================================================
// 版本和元數據
// ============================================================================

export const VERSION = '1.0.0';
export const DESCRIPTION = 'AIVA Common TypeScript AI Components & Schemas';
export const COMPATIBLE_WITH_PYTHON_VERSION = '1.0.0';

// ============================================================================
// 工具函數
// ============================================================================

/**
 * 檢查與 Python aiva_common 的兼容性
 */
export function checkPythonCompatibility(pythonVersion: string): boolean {
  return pythonVersion === COMPATIBLE_WITH_PYTHON_VERSION;
}

/**
 * 獲取 TypeScript AI 組件信息
 */
export function getComponentInfo(): Record<string, any> {
  return {
    name: 'aiva_common_ts',
    version: VERSION,
    description: DESCRIPTION,
    components: [
      'AIVACapabilityEvaluator',
      'AIVAExperienceManager',
      'UnifiedSchemas'
    ],
    compatibleWithPython: COMPATIBLE_WITH_PYTHON_VERSION,
    features: [
      'AI 能力評估',
      '經驗樣本管理',
      '強化學習支持',
      '連續監控',
      '質量評估',
      '統計分析',
      '統一數據結構',
      '跨語言 API 兼容'
    ]
  };
}

/**
 * 創建默認的 AI 組件配置
 */
export function createDefaultAIConfig(): {
  capabilityEvaluator: any;
  experienceManager: any;
} {
  return {
    capabilityEvaluator: {
      evaluatorId: `ts_evaluator_${Date.now()}`,
      continuousMonitoring: true,
      benchmarkTimeout: 30000
    },
    experienceManager: {
      managerId: `ts_exp_mgr_${Date.now()}`,
      storageBackend: 'memory',
      deduplicationEnabled: true,
      autoCleanup: true
    }
  };
}

console.log(`[AIVA Common TypeScript] Loaded version ${VERSION} with unified schemas`);